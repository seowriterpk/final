import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions # For specific API errors
import asyncio
import aiohttp
import io
import re
import time
import logging
import nest_asyncio
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Tuple, Optional
import traceback # For logging full tracebacks

# Apply nest_asyncio for environments like Streamlit that might have running loops
try:
    nest_asyncio.apply()
except RuntimeError: # Already applied or not needed
    pass

# --- Configuration ---
APP_TITLE = "⚡ Advanced WhatsApp Group Enricher ⚡"
MAX_LINKS_PER_UPLOAD = 50 # Keep the limit for stability
CSV_HEADERS = ["Group Name", "Join Link", "Description", "Category", "Country", "Keywords", "Profile Image URL"]
GEMINI_MODEL = 'gemini-1.5-flash' # Or 'gemini-pro'
WHATSAPP_FETCH_TIMEOUT = 10 # Seconds
GEMINI_REQUEST_TIMEOUT = 60 # Seconds for AI generation call
GEMINI_MAX_RETRIES = 2
GEMINI_RETRY_DELAY = 2 # Base seconds for backoff

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def parse_links(uploaded_file) -> List[str]:
    """Reads links from uploaded file (CSV or TXT). Improved validation."""
    links = []
    if uploaded_file is None:
        return []

    try:
        content = uploaded_file.getvalue()
        content_str = content.decode('utf-8', errors='ignore') # Use ignore for resilience

        if uploaded_file.name.lower().endswith('.csv'):
            # Attempt to read intelligently (handle header/no header)
            try:
                # Try reading first column, no header
                df = pd.read_csv(io.StringIO(content_str), header=None, usecols=[0], skipinitialspace=True)
                potential_links = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
                # Basic check if first item looks like a link
                if not potential_links or not potential_links[0].startswith('http'):
                    raise ValueError("First item doesn't look like a link, trying with header.")
                links = potential_links
            except (pd.errors.ParserError, IndexError, ValueError):
                logger.info("CSV parsing failed without header or first item wasn't a link, trying with header.")
                # Fallback: Try reading with header, use first column
                df = pd.read_csv(io.StringIO(content_str), skipinitialspace=True)
                if df.empty or len(df.columns) == 0:
                     st.warning("Uploaded CSV seems empty or has no columns.")
                     return []
                # Try to find a column that looks like it contains links
                link_col = None
                for col in df.columns:
                     # Check if a good portion of the column starts with http
                     if df[col].astype(str).str.startswith('http').sum() > len(df) * 0.5:
                          link_col = col
                          break
                if link_col:
                     links = df[link_col].dropna().astype(str).str.strip().tolist()
                     logger.info(f"Using column '{link_col}' from CSV.")
                else:
                     st.warning("Could not automatically detect a column with links in the CSV. Trying first column.")
                     links = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()

        elif uploaded_file.name.lower().endswith('.txt'):
            links = [line.strip() for line in content_str.splitlines() if line.strip().startswith('http')] # Basic filter
        else:
            st.error("Unsupported file format. Please upload a CSV or TXT file.")
            return []

        # Validate WhatsApp link format rigorously
        whatsapp_pattern = re.compile(r'^https://chat\.whatsapp\.com/([a-zA-Z0-9]{22})$')
        valid_links = []
        invalid_lines = []
        for idx, link in enumerate(links):
            if isinstance(link, str) and whatsapp_pattern.match(link):
                 valid_links.append(link)
            else:
                 # Don't log every invalid line if file is huge, maybe sample?
                 if idx < 10 : # Log first few invalid lines for debugging
                     logger.warning(f"Invalid line format found: '{link}'")
                 invalid_lines.append(link)

        if invalid_lines:
             st.warning(f"Filtered out {len(invalid_lines)} lines that were not valid WhatsApp group invite links.")

        return valid_links

    except Exception as e:
        st.error(f"Error reading or parsing file: {e}")
        logger.error(f"Error parsing file {uploaded_file.name}: {e}", exc_info=True)
        return []


async def validate_gemini_api_key(api_key: str) -> Tuple[bool, Optional[str]]:
    """Validates Gemini API key by making a lightweight call."""
    if not api_key:
        return False, "API Key is missing."
    try:
        # Use the new client initialization
        client = genai.Client(api_key=api_key)
        # A simple, fast call like listing models
        _ = client.list_models()
        logger.info("Gemini API Key appears valid.")
        # You could add a check here to ensure the specific model 'gemini-1.5-flash' is available if needed
        return True, None
    except google_exceptions.PermissionDenied:
         logger.error("Gemini API Key validation failed: Permission Denied (Invalid Key).")
         return False, "API Key is invalid or has insufficient permissions."
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API Key validation failed: {e}")
        return False, f"API communication error: {e}"
    except Exception as e: # Catch potential unexpected errors during client init
        logger.error(f"Unexpected error during API key validation: {e}", exc_info=True)
        return False, f"Unexpected validation error: {type(e).__name__}"


async def fetch_group_info_async(session: aiohttp.ClientSession, link: str) -> Dict[str, Any]:
    """
    Fetches WhatsApp group info (Name, Image URL) asynchronously.
    Returns a dictionary with 'name', 'pic_url', 'error', 'is_active'.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36', # Slightly updated UA
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Connection': 'keep-alive',
        'DNT': '1', # Do Not Track header
        'Upgrade-Insecure-Requests': '1',
    }
    result = {"name": None, "pic_url": None, "error": None, "is_active": False}
    placeholder_pic_url = "https://via.placeholder.com/150/CCCCCC/808080?text=No+Image"

    try:
        async with session.get(link, headers=headers, timeout=WHATSAPP_FETCH_TIMEOUT, allow_redirects=True, ssl=False) as response: # Added ssl=False for potential local issues, use with caution
            logger.info(f"Fetching {link} - Status: {response.status}, Final URL: {response.url}")

            if response.status == 404:
                result["error"] = "Fetch Error: Group not found (404) or link expired."
                return result # is_active remains False

            # Check if redirected away from chat.whatsapp.com (can indicate issues)
            if "chat.whatsapp.com" not in str(response.url):
                 result["error"] = f"Fetch Error: Redirected away from WhatsApp to {response.url.host}"
                 # Could still try parsing if content looks right, but likely indicates an issue
                 # For now, consider it inactive.
                 return result

            response.raise_for_status() # Raise for other bad statuses (4xx, 5xx)

            html_content = await response.text()

            # Quick checks for known "invalid" messages before full parsing
            lc_html = html_content.lower()
            if "invite link was revoked" in lc_html or "couldn't find this group" in lc_html or "link reset" in lc_html:
                 result["error"] = "Fetch Error: Link revoked, reset, or group unavailable."
                 return result # is_active remains False

            soup = BeautifulSoup(html_content, 'html.parser')

            # --- Extract Name ---
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                result["name"] = og_title['content'].strip()
            else:
                # Fallback: Try finding h2 (less reliable)
                h2_tag = soup.find('h2')
                if h2_tag:
                     result["name"] = h2_tag.get_text(strip=True)

            # --- Extract Image ---
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                result["pic_url"] = og_image['content'].strip()

            # --- Validation & Status ---
            if result["name"]:
                result["is_active"] = True # Consider active if we got a name
                result["error"] = None # Clear any previous non-critical error if successful
                if not result["pic_url"]:
                     result["pic_url"] = placeholder_pic_url # Use placeholder if image specifically failed
                     logger.warning(f"Name found but image not found for {link}. Using placeholder.")
            else:
                # If no name found after trying, assume failure
                result["error"] = "Fetch Error: Could not extract group name from page."
                result["is_active"] = False
                # Don't bother setting pic_url if name extraction failed

            if result["is_active"]:
                 logger.info(f"Successfully fetched info for {link} - Name: '{result['name']}'")
            else:
                 logger.warning(f"Failed to fetch active group info for {link}. Error: {result['error']}")

            return result

    except aiohttp.ClientResponseError as e:
        result["error"] = f"Fetch Error: HTTP {e.status} {e.message}"
        logger.warning(f"{result['error']} for link {link}")
    except asyncio.TimeoutError:
        result["error"] = f"Fetch Error: Request timed out after {WHATSAPP_FETCH_TIMEOUT}s"
        logger.warning(f"{result['error']} for link {link}")
    except aiohttp.ClientConnectionError as e:
        result["error"] = f"Fetch Error: Connection error ({type(e).__name__})"
        logger.warning(f"{result['error']} for link {link}: {e}")
    except aiohttp.ClientError as e: # Catch other aiohttp client errors
        result["error"] = f"Fetch Error: Network/Client error ({type(e).__name__})"
        logger.warning(f"{result['error']} for link {link}: {e}")
    except Exception as e:
        result["error"] = f"Fetch Error: Unexpected error ({type(e).__name__})"
        logger.error(f"Unexpected error fetching {link}: {e}", exc_info=True)

    result["is_active"] = False # Ensure inactive on any exception
    return result


async def generate_ai_content_async(api_key: str, group_name: str, category: str, country: str) -> Dict[str, Any]:
    """Generates description and keywords using the Gemini API asynchronously with retries."""
    result = {"description": None, "keywords": None, "error": None}
    if not group_name: # Should not happen if called correctly, but safety check
        result["error"] = "AI Error: Missing group name"
        return result

    try:
        # Initialize client within the task - safer for potential auth context issues
        # If performance is critical and key doesn't change, could potentially pass client instance
        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        result["error"] = f"AI Error: Client Init Failed ({type(e).__name__})"
        return result

    prompt = f"""
    You are an expert SEO content writer for an online group directory.
    Analyze the following WhatsApp Group details:
    Group Name: "{group_name}"
    Category: "{category}"
    Country: "{country}"

    Instructions:
    1. Write a unique, engaging, SEO-friendly description (MAXIMUM 50 words). Focus on the likely topic and target audience based *only* on the provided Name, Category, and Country. Use natural language.
    2. Generate exactly 3 relevant, distinct keywords (comma-separated). Keywords should reflect the Name, Category, and Country, suitable for search queries.

    Output Format (Strictly follow this):
    Description: [Generated description here]
    Keywords: [Keyword1, Keyword2, Keyword3]
    """

    # Configure generation settings if needed (optional)
    generation_config = genai.types.GenerationConfig(
        # max_output_tokens=150, # Estimate based on 50 words + keywords
        temperature=0.7, # Balance creativity and predictability
        # top_p=0.9,
        # top_k=40
    )

    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            # Run the synchronous SDK call in a thread pool
            response = await asyncio.to_thread(
                client.generate_content,
                contents=[prompt],
                generation_config=generation_config,
                request_options={'timeout': GEMINI_REQUEST_TIMEOUT} # Add request timeout
            )
            response.resolve() # Ensure completion if streaming internally

            if not response.candidates or not response.candidates[0].content.parts:
                 if response.prompt_feedback.block_reason:
                      block_reason = response.prompt_feedback.block_reason.name
                      safety_ratings = {r.category.name: r.probability.name for r in response.prompt_feedback.safety_ratings}
                      logger.warning(f"Gemini content blocked for '{group_name}'. Reason: {block_reason}. Ratings: {safety_ratings}")
                      result["error"] = f"AI Error: Content Blocked ({block_reason})"
                 else:
                      logger.warning(f"Gemini response empty/malformed for '{group_name}'. Response: {response}")
                      result["error"] = "AI Error: Empty or malformed response"
                 return result # Don't retry on empty/blocked/malformed

            content = response.text.strip()
            logger.info(f"Gemini Raw Response for '{group_name}' (Attempt {attempt+1}): {content[:150]}...")

            # --- Parse the response ---
            desc_match = re.search(r"Description:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
            keywords_match = re.search(r"Keywords:\s*(.*)", content, re.IGNORECASE | re.DOTALL)

            parsed_desc = desc_match.group(1).strip() if desc_match else None
            parsed_keywords = keywords_match.group(1).strip() if keywords_match else None

            if parsed_desc and parsed_keywords:
                # Basic cleanup and validation
                # Word count check (approximate)
                word_count = len(parsed_desc.split())
                if word_count > 60: # Allow a bit more margin
                    logger.warning(f"Gemini description for '{group_name}' exceeded word limit ({word_count} words), truncating.")
                    # Simple truncation (could be smarter)
                    parsed_desc = ' '.join(parsed_desc.split()[:55]) + "..."

                keywords_list = [k.strip() for k in parsed_keywords.split(',') if k.strip()]
                final_keywords = ", ".join(keywords_list[:3]) # Ensure max 3 valid keywords

                result["description"] = parsed_desc
                result["keywords"] = final_keywords
                result["error"] = None # Success
                logger.info(f"Successfully generated AI content for '{group_name}'.")
                return result # Successful generation
            else:
                logger.warning(f"Could not parse Description/Keywords from Gemini response for '{group_name}'. Raw: {content}")
                result["error"] = "AI Error: Parsing failed"
                # Consider retrying if parsing fails? For now, return error.
                return result

        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Gemini Rate Limit/Quota Exceeded for '{group_name}' (Attempt {attempt+1}): {e}")
            if attempt < GEMINI_MAX_RETRIES:
                wait_time = GEMINI_RETRY_DELAY * (2 ** attempt) # Exponential backoff
                logger.info(f"Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            else:
                result["error"] = "AI Error: Rate Limit/Quota Exceeded after retries"
                return result
        except google_exceptions.FailedPrecondition as e:
             logger.error(f"Gemini Failed Precondition for '{group_name}' (Attempt {attempt+1}): {e}")
             result["error"] = f"AI Error: Failed Precondition ({e})" # Often related to billing/API enablement
             return result # Unlikely to succeed on retry
        except google_exceptions.InvalidArgument as e:
             logger.error(f"Gemini Invalid Argument for '{group_name}' (Attempt {attempt+1}): {e}")
             result["error"] = f"AI Error: Invalid Argument ({e})" # Prompt issue? Model issue?
             return result # Unlikely to succeed on retry
        except google_exceptions.PermissionDenied as e:
             logger.error(f"Gemini Permission Denied during generation for '{group_name}': {e}")
             result["error"] = "AI Error: Invalid API Key or Permissions"
             return result # No point retrying
        except google_exceptions.Aborted as e:
             # Can sometimes be transient, worth retrying
             logger.warning(f"Gemini request aborted for '{group_name}' (Attempt {attempt+1}): {e}")
             if attempt < GEMINI_MAX_RETRIES:
                 wait_time = GEMINI_RETRY_DELAY * (2 ** attempt)
                 logger.info(f"Retrying in {wait_time:.1f}s...")
                 await asyncio.sleep(wait_time)
             else:
                 result["error"] = "AI Error: Request Aborted after retries"
                 return result
        except google_exceptions.GoogleAPIError as e: # Catch-all for other Google API errors
            logger.error(f"Gemini API Error for '{group_name}' (Attempt {attempt+1}): {e}", exc_info=True)
            if attempt < GEMINI_MAX_RETRIES:
                wait_time = GEMINI_RETRY_DELAY * (2 ** attempt)
                logger.info(f"Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            else:
                result["error"] = f"AI Error: API Error ({type(e).__name__}) after retries"
                return result
        except Exception as e: # Catch unexpected errors during generation/threading
            logger.error(f"Unexpected error during AI generation for '{group_name}': {e}", exc_info=True)
            result["error"] = f"AI Error: Unexpected ({type(e).__name__})"
            return result # Don't retry unexpected errors

    # Should only be reached if all retries failed
    logger.error(f"AI content generation failed for '{group_name}' after all retries.")
    return result


async def process_link_data_async(session: aiohttp.ClientSession, link: str, api_key: str, category: str, country: str) -> Dict[str, Any]:
    """Fetches info, validates, generates AI content for a single link."""
    final_data = {
        "Group Name": f"Processing Error ({link})", # Default in case of unexpected failure
        "Join Link": link,
        "Description": "N/A",
        "Category": category,
        "Country": country,
        "Keywords": "N/A",
        "Profile Image URL": "N/A",
        "internal_fetch_error": None, # For logging/debugging
        "internal_ai_error": None,
        "status": "failed" # Overall status
    }

    try:
        # 1. Fetch group info
        fetch_result = await fetch_group_info_async(session, link)
        final_data["internal_fetch_error"] = fetch_result["error"]

        # 2. Validate Fetch Result & Skip if Inactive/Failed
        if not fetch_result["is_active"] or not fetch_result["name"]:
            error_msg = fetch_result["error"] or "Fetch Failed (Unknown)"
            final_data["Group Name"] = f"Skipped ({error_msg})"
            final_data["Profile Image URL"] = fetch_result["pic_url"] or "N/A" # Keep pic if found, even if name failed
            final_data["status"] = "skipped"
            logger.info(f"Skipping AI step for {link} due to fetch status: {error_msg}")
            return final_data # Return early

        # If fetch succeeded:
        final_data["Group Name"] = fetch_result["name"]
        final_data["Profile Image URL"] = fetch_result["pic_url"]

        # 3. Generate AI Content
        ai_result = await generate_ai_content_async(api_key, fetch_result["name"], category, country)
        final_data["internal_ai_error"] = ai_result["error"]

        if ai_result["error"]:
            final_data["Description"] = f"{ai_result['error']}"
            final_data["Keywords"] = "AI Error"
            final_data["status"] = "ai_failed"
        else:
            final_data["Description"] = ai_result["description"]
            final_data["Keywords"] = ai_result["keywords"]
            final_data["status"] = "success"

    except Exception as e:
        # Catch unexpected errors during the orchestration of fetch/AI
        logger.error(f"Critical error processing link {link}: {e}\n{traceback.format_exc()}")
        final_data["Group Name"] = "System Error During Processing"
        final_data["Description"] = f"Unexpected Error: {type(e).__name__}"
        final_data["Keywords"] = "System Error"
        final_data["internal_fetch_error"] = final_data["internal_fetch_error"] or f"System Error: {type(e).__name__}"
        final_data["status"] = "failed"

    return final_data


async def process_all_links_async(links: List[str], api_key: str, category: str, country: str, progress_bar, status_text) -> List[Dict[str, Any]]:
    """Processes all links concurrently using asyncio.gather."""
    total_links = len(links)
    processed_count = 0
    results = []
    tasks = []

    # Use a single session for connection pooling
    connector = aiohttp.TCPConnector(limit=20, ssl=False) # Increase concurrent connections slightly, adjust ssl as needed
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create all tasks
        for link in links:
            tasks.append(asyncio.create_task(process_link_data_async(session, link, api_key, category, country)))

        # Process tasks as they complete using asyncio.as_completed for progress updates
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                # This catches errors *within* the asyncio task handling itself, less likely now
                logger.error(f"Critical error awaiting task result: {e}", exc_info=True)
                # Add a placeholder error result if a task fails catastrophically
                results.append({
                    "Group Name": "System Task Error", "Join Link": "Unknown", "Description": str(e),
                    "Category": category, "Country": country, "Keywords": "Error", "Profile Image URL": "Error",
                    "internal_fetch_error": str(e), "internal_ai_error": None, "status": "failed"
                 })

            processed_count += 1
            progress = processed_count / total_links
            if progress_bar and status_text: # Check if UI elements exist
                progress_bar.progress(progress)
                status_text.text(f"Processing: {processed_count}/{total_links} links completed...")

    # Sort results by the original order of links if needed (optional)
    # This requires mapping original links to results, adds complexity
    # results_dict = {res["Join Link"]: res for res in results}
    # ordered_results = [results_dict.get(link, {}) for link in links] # Basic ordering attempt

    return results # Return in completion order for simplicity


# --- Streamlit App UI ---

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.title(f"🚀 {APP_TITLE} 🚀")

st.markdown(f"""
This tool automates fetching WhatsApp group details and generating SEO-optimized content using Google Gemini.

**Workflow:**
1.  Enter your **Google Gemini API Key** (stored temporarily in session state).
2.  Provide **Category** and **Country** for the batch.
3.  Upload a **CSV or TXT file** with WhatsApp group join links (one per line or first column). Max **{MAX_LINKS_PER_UPLOAD}** links.
4.  Click **"🔥 Start Processing"**.
5.  Monitor progress. Results and logs will appear below.
6.  Download the final CSV.

**Disclaimer:** WhatsApp actively prevents scraping. Fetching group names/images is **highly unreliable** and may frequently fail or return outdated information. Use at your own discretion. Gemini results depend on the quality of the fetched group name.
""")

# --- Input Section (Sidebar) ---
with st.sidebar:
    st.header("🛠️ Configuration")

    # Use session state to keep inputs populated
    if 'api_key' not in st.session_state: st.session_state.api_key = ""
    if 'category' not in st.session_state: st.session_state.category = ""
    if 'country' not in st.session_state: st.session_state.country = ""

    st.session_state.api_key = st.text_input("🔑 Google Gemini API Key", type="password", value=st.session_state.api_key, help="Get key from Google AI Studio. Required.")
    st.session_state.category = st.text_input("🏷️ Group Category", placeholder="e.g., Technology, Education", value=st.session_state.category, help="Required.")
    st.session_state.country = st.text_input("🌍 Group Country", placeholder="e.g., India, USA, Global", value=st.session_state.country, help="Required.")

    st.header("📤 Upload Links File")
    uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"], key="file_uploader")

    process_button = st.button("🔥 Start Processing", type="primary", use_container_width=True)

# --- Processing & Results Area (Main Page) ---
results_placeholder = st.container() # Placeholder for results table and download
logs_placeholder = st.container() # Placeholder for logs/skipped links

# Initialize session state for results and logs if they don't exist
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'log_messages' not in st.session_state: st.session_state.log_messages = []
if 'run_key' not in st.session_state: st.session_state.run_key = 0 # To differentiate runs

if process_button:
    st.session_state.run_key += 1 # Increment run key to clear old results/logs visually
    st.session_state.results_df = None
    st.session_state.log_messages = []
    run_id = st.session_state.run_key # Capture run ID for this execution

    # Clear previous outputs
    results_placeholder.empty()
    logs_placeholder.empty()

    # --- Input Validation ---
    valid_inputs = True
    if not st.session_state.api_key:
        st.sidebar.error("❌ Gemini API Key is required.")
        valid_inputs = False
    if not st.session_state.category:
        st.sidebar.error("❌ Category is required.")
        valid_inputs = False
    if not st.session_state.country:
        st.sidebar.error("❌ Country is required.")
        valid_inputs = False
    if uploaded_file is None:
        st.sidebar.error("❌ Please upload a file.")
        valid_inputs = False

    if valid_inputs:
        with results_placeholder:
            st.info("✅ Inputs Validated. Starting process...")

            # 1. Validate API Key
            with st.spinner("🔑 Validating Gemini API Key..."):
                key_is_valid, key_error = asyncio.run(validate_gemini_api_key(st.session_state.api_key))

            if not key_is_valid:
                st.error(f"API Key Validation Failed: {key_error}")
            else:
                st.success("🔑 API Key Validated.")

                # 2. Parse Links
                with st.spinner("📄 Parsing uploaded file..."):
                    links_to_process = parse_links(uploaded_file)

                if not links_to_process:
                    st.warning("⚠️ No valid WhatsApp links found in the file or file is empty/unreadable.")
                else:
                    link_count = len(links_to_process)
                    if link_count > MAX_LINKS_PER_UPLOAD:
                        st.warning(f"⚠️ Found {link_count} links. Processing only the first {MAX_LINKS_PER_UPLOAD} for stability.")
                        links_to_process = links_to_process[:MAX_LINKS_PER_UPLOAD]
                    else:
                         st.info(f"Found {len(links_to_process)} valid links to process.")

                    # 3. Execute Processing
                    start_time = time.time()
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    status_text.text("🚀 Initializing processing...")

                    try:
                        # Run the main async processing function
                        all_results_raw = asyncio.run(process_all_links_async(
                            links_to_process,
                            st.session_state.api_key,
                            st.session_state.category,
                            st.session_state.country,
                            progress_bar,
                            status_text
                        ))
                        st.session_state.processing_done = True # Mark as done

                    except Exception as e:
                        st.error(f"An unexpected error occurred during the main processing loop: {e}")
                        logger.error("Critical error during asyncio.run(process_all_links_async)", exc_info=True)
                        st.session_state.processing_done = False
                        all_results_raw = [] # Ensure empty results on critical failure


                    end_time = time.time()
                    processing_time = end_time - start_time

                    if st.session_state.processing_done and all_results_raw:
                        progress_bar.progress(1.0)
                        status_text.success(f"🏁 Processing Complete in {processing_time:.2f} seconds!")
                        logger.info(f"Processed {len(links_to_process)} links in {processing_time:.2f} seconds.")

                        # --- Prepare DataFrame and Stats ---
                        df_results = pd.DataFrame(all_results_raw)

                        # Separate logs/skipped from successful/ai_failed
                        st.session_state.log_messages = df_results[df_results['status'].isin(['skipped', 'failed', 'ai_failed'])][['Join Link', 'Group Name', 'internal_fetch_error', 'internal_ai_error']].to_dict('records')

                        # Prepare final DataFrame for display/download
                        df_display = df_results[df_results['status'].isin(['success', 'ai_failed'])].copy() # Keep AI failures for download, but filter system errors
                        if not df_display.empty:
                            # Ensure all required columns exist, even if empty
                            for col in CSV_HEADERS:
                                if col not in df_display.columns:
                                    df_display[col] = "N/A" # Or appropriate default
                            st.session_state.results_df = df_display[CSV_HEADERS] # Select and order
                        else:
                             st.session_state.results_df = pd.DataFrame(columns=CSV_HEADERS) # Empty DF with correct headers


                        # --- Display Results ---
                        st.subheader("📊 Processed Data Preview (Successful/AI Failed Links)")
                        if not st.session_state.results_df.empty:
                            st.dataframe(st.session_state.results_df, use_container_width=True)

                            # --- Download Button ---
                            csv_buffer = io.StringIO()
                            st.session_state.results_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                            csv_data = csv_buffer.getvalue()

                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_data,
                                file_name=f"whatsapp_enriched_{st.session_state.category}_{st.session_state.country}_{int(time.time())}.csv",
                                mime="text/csv",
                                key=f"download_button_{run_id}" # Unique key per run
                            )
                        else:
                             st.info("No links were successfully processed or had AI generated (check logs below).")


                    elif st.session_state.processing_done:
                         status_text.warning("🏁 Processing finished, but no results were generated (all links might have failed fetch).")
                         st.session_state.results_df = pd.DataFrame(columns=CSV_HEADERS) # Show empty table structure
                         st.session_state.log_messages = pd.DataFrame(all_results_raw)[['Join Link', 'Group Name', 'internal_fetch_error', 'internal_ai_error']].to_dict('records') if all_results_raw else []

# --- Display Logs/Skipped Links (always attempts to display if state has messages) ---
if st.session_state.log_messages:
    with logs_placeholder:
        st.subheader(" Memos & Logs ")
        total_processed = len(st.session_state.results_df) if st.session_state.results_df is not None else 0
        total_logged = len(st.session_state.log_messages)
        st.info(f"Successfully generated data for {total_processed} links. Encountered issues or skipped {total_logged} links.")

        with st.expander(f"🔍 View Details for {total_logged} Skipped/Failed Links", expanded=False):
            log_df = pd.DataFrame(st.session_state.log_messages)
            log_df.rename(columns={
                'Join Link': 'Link',
                'Group Name': 'Status/Initial Name',
                'internal_fetch_error': 'Fetch Error',
                'internal_ai_error': 'AI Error'
            }, inplace=True)
            st.dataframe(log_df[['Link', 'Status/Initial Name', 'Fetch Error', 'AI Error']], use_container_width=True)
            st.caption("Status starting with 'Skipped' indicates the link was likely invalid, expired, or couldn't be read. 'AI Error' indicates issues during description/keyword generation.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("App Version: 3.0 (Async, New Gemini SDK, Enhanced Error Handling)")
