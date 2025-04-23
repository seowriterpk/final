import streamlit as st
import pandas as pd
import google.generativeai as genai
import asyncio
import aiohttp
import io
import re
import time
import logging
import nest_asyncio
from bs4 import BeautifulSoup # Added for potentially better parsing

# Apply nest_asyncio to allow running asyncio event loops within Streamlit
nest_asyncio.apply()

# --- Configuration ---
APP_TITLE = "WhatsApp Group Info Extractor & Enricher v2"
MAX_LINKS_PER_UPLOAD = 50
CSV_HEADERS = ["Group Name", "Join Link", "Description", "Category", "Country", "Keywords", "Profile Image URL"]
GEMINI_MODEL = 'gemini-1.5-flash' # Or use 'gemini-pro'
PROCESSING_DELAY_SECONDS = 3 # Delay between processing each link

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def parse_links(uploaded_file):
    """Reads links from uploaded file (CSV or TXT). Assumes links are in the first column for CSV."""
    links = []
    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue()
            content_str = content.decode('utf-8', errors='ignore')

            if uploaded_file.name.lower().endswith('.csv'):
                try:
                    # Try reading assuming first column is links, no header
                    df = pd.read_csv(io.StringIO(content_str), header=None, usecols=[0], squeeze=True)
                    if pd.api.types.is_string_dtype(df): # Check if it read as a Series of strings
                         links = df.astype(str).str.strip().tolist()
                    else: # Maybe it has a header or different structure
                         raise ValueError("CSV parsing failed, trying with header.")
                except (ValueError, IndexError, pd.errors.ParserError):
                     # Fallback: Try reading with header, taking first column
                     logger.info("Initial CSV parse failed, retrying with header assumption.")
                     df = pd.read_csv(io.StringIO(content_str))
                     if df.empty or len(df.columns) == 0:
                          st.error("CSV file seems empty or unreadable.")
                          return []
                     links = df.iloc[:, 0].astype(str).str.strip().tolist()

            elif uploaded_file.name.lower().endswith('.txt'):
                links = [line.strip() for line in content_str.splitlines() if line.strip()]
            else:
                st.error("Unsupported file format. Please upload a CSV or TXT file.")
                return []

            # Validate WhatsApp link format
            whatsapp_pattern = re.compile(r'^https://chat\.whatsapp\.com/([a-zA-Z0-9]{22})$')
            valid_links = []
            invalid_links = []
            for link in links:
                 if isinstance(link, str) and whatsapp_pattern.match(link):
                      valid_links.append(link)
                 else:
                      invalid_links.append(link)

            if invalid_links:
                 st.warning(f"Filtered out {len(invalid_links)} lines that were not valid WhatsApp group invite links.")
                 # Optionally log invalid links if needed: logger.warning(f"Invalid links found: {invalid_links}")

            return valid_links

        except Exception as e:
            st.error(f"Error reading or parsing file: {e}")
            logger.error(f"Error parsing file {uploaded_file.name}: {e}", exc_info=True)
            return []
    return []

async def validate_gemini_api_key(api_key):
    """Quickly checks if the Gemini API key is likely valid by listing models."""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        # Check if a common model exists in the list
        if any(m.name == f'models/{GEMINI_MODEL}' for m in models):
            logger.info("Gemini API Key appears valid.")
            return True, None
        else:
             logger.warning(f"Gemini API Key configured, but model '{GEMINI_MODEL}' not found in list.")
             # Could still be valid but model name is wrong, proceed with caution
             return True, f"Warning: Model '{GEMINI_MODEL}' might not be available."

    except Exception as e:
        logger.error(f"Gemini API Key validation failed: {e}")
        error_message = f"Gemini API Key is invalid or configuration failed: {type(e).__name__}"
        if "API key not valid" in str(e):
            error_message = "Google Gemini API Key is not valid. Please check the key."
        return False, error_message


async def fetch_group_info_async(session, link):
    """
    Attempts to fetch group info (Name and Image URL) from the invite link page.
    Uses BeautifulSoup for slightly more robust parsing than pure regex.
    Returns {"name": str|None, "pic_url": str|None, "error": str|None, "is_active": bool}
    'is_active' is True if the page loaded and appears to be a valid group invite page.
    """
    placeholder_name = "Group Fetch Failed"
    placeholder_pic_url = "https://via.placeholder.com/150/CCCCCC/808080?text=No+Image"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9', # Request English page if possible
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Connection': 'keep-alive',
    }
    timeout = aiohttp.ClientTimeout(total=15) # Increased timeout slightly

    try:
        async with session.get(link, headers=headers, timeout=timeout, allow_redirects=True) as response:
            logger.info(f"Fetching {link} - Status: {response.status}")
            if response.status == 404:
                logger.warning(f"Link {link} returned 404 (Not Found/Expired). Skipping.")
                return {"name": None, "pic_url": None, "error": "Group not found (404) or link expired.", "is_active": False}

            response.raise_for_status() # Raise exception for other bad status codes (e.g., 403, 5xx)
            html_content = await response.text()

            # Check for common signs of inactive/revoked links in the content
            if "invite link was revoked" in html_content.lower() or "couldn't find this group" in html_content.lower():
                 logger.warning(f"Link {link} appears revoked or invalid based on page content. Skipping.")
                 return {"name": None, "pic_url": None, "error": "Link revoked or group not found.", "is_active": False}

            # Use BeautifulSoup to parse
            soup = BeautifulSoup(html_content, 'html.parser')

            group_name = None
            profile_pic_url = None

            # Strategy 1: Open Graph meta tags (usually most reliable)
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                group_name = og_title['content'].strip()

            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                profile_pic_url = og_image['content'].strip()

            # Strategy 2: Fallback using specific elements (less reliable, might need adjustment)
            if not group_name:
                # Example: WhatsApp web sometimes uses an h2 tag
                h2_tag = soup.find('h2') # Very generic, refine if possible
                if h2_tag:
                    group_name = h2_tag.get_text(strip=True)

            # If still no name or image, it might be a loading page or structured differently
            if not group_name:
                 logger.warning(f"Could not extract group name for {link}. Page structure might have changed.")
                 group_name = placeholder_name # Use placeholder if extraction failed
                 # Return as inactive if name couldn't be found, as it's critical
                 # return {"name": None, "pic_url": profile_pic_url, "error": "Could not parse group name.", "is_active": False}

            if not profile_pic_url:
                 logger.warning(f"Could not extract profile picture for {link}. Using placeholder.")
                 profile_pic_url = placeholder_pic_url # Use placeholder

            logger.info(f"Successfully fetched info for {link} - Name: '{group_name}', Pic Found: {profile_pic_url != placeholder_pic_url}")
            return {"name": group_name, "pic_url": profile_pic_url, "error": None, "is_active": True}

    except aiohttp.ClientResponseError as e:
        logger.warning(f"HTTP Error fetching {link}: {e.status} {e.message}")
        error_msg = f"HTTP Error {e.status}."
        return {"name": None, "pic_url": None, "error": error_msg, "is_active": False}
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching {link}")
        return {"name": None, "pic_url": None, "error": "Request timed out.", "is_active": False}
    except aiohttp.ClientError as e:
        logger.error(f"ClientError fetching {link}: {e}", exc_info=False)
        return {"name": None, "pic_url": None, "error": f"Network/Client Error: {type(e).__name__}.", "is_active": False}
    except Exception as e:
        logger.error(f"Unexpected error fetching {link}: {e}", exc_info=True)
        return {"name": None, "pic_url": None, "error": f"Unexpected fetch error: {type(e).__name__}.", "is_active": False}


async def generate_ai_content_async(api_key, group_name, category, country):
    """Generates description and keywords using Gemini API asynchronously."""
    # Re-configure Gemini within the async task if needed, or rely on global config
    # For safety in async, re-configuring might be slightly better if context switches matter.
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        logger.error(f"Failed to configure Gemini for group '{group_name}': {e}")
        return {"description": "Error: Gemini Config", "keywords": "Error", "error": f"Gemini Configuration Error: {e}"}

    # Construct the prompt
    prompt = f"""
    You are an expert SEO content writer specializing in online group directories.
    Given the following WhatsApp Group information:
    Group Name: "{group_name}"
    Category: "{category}"
    Country: "{country}"

    Generate the following, focusing on natural language, engagement, and SEO optimization:
    1. A unique, compelling, and SEO-friendly description (strictly maximum 50 words). Describe the group's likely purpose and who might benefit from joining, based *only* on its name, category, and country. Avoid generic phrases.
    2. Exactly 3 relevant and distinct keywords (comma-separated) that people would use to search for this group. Base keywords on the name, category, and country.

    Format the output EXACTLY like this, with no extra text, introduction, or explanation:
    Description: [Your generated description here]
    Keywords: [Keyword1, Keyword2, Keyword3]
    """

    retries = 3 # Increased retries slightly
    for attempt in range(retries):
        try:
            response = await asyncio.to_thread(model.generate_content, prompt) # Run sync SDK call in thread pool
            response.resolve()

            if not response.parts:
                 raise ValueError("Gemini response is empty or malformed.")

            content = response.text.strip()
            logger.info(f"Gemini Raw Response for '{group_name}' (Attempt {attempt+1}): {content[:100]}...")

            # --- Parse the response ---
            desc_match = re.search(r"Description:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
            keywords_match = re.search(r"Keywords:\s*(.*)", content, re.IGNORECASE | re.DOTALL)

            description = desc_match.group(1).strip() if desc_match else "Error: Could not parse description."
            keywords = keywords_match.group(1).strip() if keywords_match else "Error: Could not parse keywords."

            # Cleanup / Validation
            # Simple word count check (approximate)
            if len(description.split()) > 55: # Allow slightly over 50 during generation
                 # Try to truncate gracefully, might need better logic
                 description = ' '.join(description.split()[:50]) + "..."
                 logger.warning(f"Gemini description for '{group_name}' exceeded word limit, truncated.")

            keywords_list = [k.strip() for k in keywords.split(',') if k.strip()]
            keywords = ", ".join(keywords_list[:3]) # Ensure max 3 valid keywords

            if "Error:" in description or "Error:" in keywords:
                 logger.warning(f"Parsing failed for Gemini response for '{group_name}'. Raw: {content}")
                 # Consider retrying if parsing fails, though it might be a consistent model issue
                 # For now, return the error

            return {"description": description, "keywords": keywords, "error": None}

        except Exception as e:
            logger.error(f"Error generating content for '{group_name}' (Attempt {attempt + 1}/{retries}): {e}", exc_info=False)
            error_msg = f"Gemini API Error: {type(e).__name__}"

            if "API key not valid" in str(e):
                 error_msg = "Invalid API Key" # Keep it concise for the output row
                 # No point retrying if key is bad
                 return {"description": "Error", "keywords": "Error", "error": error_msg}
            elif "rate limit" in str(e).lower():
                 error_msg = "Rate limit exceeded"
                 if attempt < retries - 1:
                     wait_time = (2 ** attempt) + 1 # Exponential backoff with slight jitter
                     logger.info(f"Rate limit hit for '{group_name}'. Retrying in {wait_time}s...")
                     await asyncio.sleep(wait_time)
                     continue
                 else:
                     logger.error(f"Gemini rate limit exceeded after retries for '{group_name}'.")
                     return {"description": "Error", "keywords": "Error", "error": error_msg}
            elif isinstance(e, (google.generativeai.types.StopCandidateException, google.api_core.exceptions.InvalidArgument)):
                 # Handle safety settings or bad input
                 error_msg = f"Content generation blocked or invalid input: {e}"
                 logger.error(f"{error_msg} for group '{group_name}'")
                 # Probably won't succeed on retry
                 return {"description": "Error: Content Blocked", "keywords": "Error", "error": error_msg}
            else:
                 # General Gemini/API error
                  if attempt < retries - 1:
                     logger.info(f"Gemini error for '{group_name}'. Retrying in 3s...")
                     await asyncio.sleep(3)
                     continue
                  else:
                     logger.error(f"Gemini failed after retries for '{group_name}'. Last error: {e}")
                     return {"description": "Error", "keywords": "Error", "error": error_msg}

    # Should only be reached if all retries fail for retryable errors
    return {"description": "Error: Retries Failed", "keywords": "Error", "error": "Gemini failed after multiple retries."}

async def process_link_data_async(session, link, api_key, category, country):
    """
    Processes a single link: Fetch -> Validate -> Generate AI Content.
    Skips AI generation if fetch fails or group is inactive.
    """
    # 1. Fetch group info
    fetch_result = await fetch_group_info_async(session, link)
    group_name = fetch_result["name"]
    pic_url = fetch_result["pic_url"]
    fetch_error = fetch_result["error"]
    is_active = fetch_result["is_active"]

    # 2. Validate & Skip if needed
    if not is_active or group_name is None:
        logger.info(f"Skipping AI generation for inactive/failed link: {link} (Error: {fetch_error})")
        # Still return a row, but mark clearly that it was skipped/failed
        return {
            "Group Name": f"Skipped/Invalid ({fetch_error or 'Fetch Failed'})",
            "Join Link": link,
            "Description": "N/A",
            "Category": category,
            "Country": country,
            "Keywords": "N/A",
            "Profile Image URL": pic_url or "N/A", # Use fetched pic URL even if name failed, if available
            "fetch_error": fetch_error or "Group Inactive or Name Fetch Failed",
            "ai_error": None
        }

    # 3. Generate AI Content (only if fetch was successful and group seems active)
    ai_result = await generate_ai_content_async(api_key, group_name, category, country)
    description = ai_result["description"]
    keywords = ai_result["keywords"]
    ai_error = ai_result["error"]

    # 4. Combine results
    return {
        "Group Name": group_name,
        "Join Link": link,
        "Description": description if not ai_error else f"AI Error: {ai_error}",
        "Category": category,
        "Country": country,
        "Keywords": keywords if not ai_error else "AI Error",
        "Profile Image URL": pic_url, # Should have a valid one from fetch
        "fetch_error": fetch_error, # Can be None even if successful
        "ai_error": ai_error
    }

async def process_all_links_async(links, api_key, category, country, progress_bar, status_text):
    """Processes all links concurrently with delays."""
    results = []
    total_links = len(links)
    processed_count = 0

    # Use aiohttp ClientSession for connection pooling
    connector = aiohttp.TCPConnector(limit=10) # Limit concurrent connections if needed
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        # Create all tasks first
        for link in links:
             task = asyncio.create_task(process_link_data_async(session, link, api_key, category, country))
             tasks.append(task)

        # Process tasks as they complete, adding delay *after* each completion
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                # This catches errors *within* the asyncio task handling itself
                logger.error(f"Critical error processing a link task: {e}", exc_info=True)
                results.append({
                    "Group Name": "System Error", "Join Link": "Unknown", "Description": str(e),
                    "Category": category, "Country": country, "Keywords": "Error", "Profile Image URL": "Error",
                    "fetch_error": str(e), "ai_error": None
                 })

            processed_count += 1
            progress = processed_count / total_links
            progress_bar.progress(progress)
            status_text.text(f"Processed {processed_count}/{total_links} links...")

            # Add delay AFTER processing a link, before starting the next iteration's await
            if processed_count < total_links:
                 await asyncio.sleep(PROCESSING_DELAY_SECONDS)

    return results

# --- Streamlit App UI ---

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"🚀 {APP_TITLE}")

st.markdown(f"""
Upload a list of WhatsApp group invite links (CSV/TXT). The app will attempt to fetch the group name & picture, then use Google Gemini to generate descriptions & keywords.

**Workflow:**
1.  Enter your **Google Gemini API Key**.
2.  Enter the **Category** and **Country** for this batch of links.
3.  Upload your file (Max **{MAX_LINKS_PER_UPLOAD}** links).
4.  Click **Process Links**.
5.  Wait for processing (includes a **{PROCESSING_DELAY_SECONDS}-second delay** between links).
6.  Download the results CSV.

**Disclaimer:** Fetching WhatsApp group info is **unreliable** due to website changes and anti-scraping measures. Expect some failures or placeholder data.
""")

# --- Inputs ---
st.sidebar.header("🔑 API & Batch Info")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password", help="Get key from Google AI Studio")
category = st.sidebar.text_input("Group Category", placeholder="e.g., Technology, Education")
country = st.sidebar.text_input("Group Country", placeholder="e.g., India, USA, Global")

st.sidebar.header("📤 Upload Links")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"])

process_button = st.sidebar.button("✨ Process Links ✨", key="process_button", type="primary")

# --- Session State for Results ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

# --- Processing Logic ---
if process_button:
    st.session_state.results_df = None # Clear previous results
    st.session_state.processing_done = False
    has_error = False

    if not api_key:
        st.sidebar.error("❌ Gemini API Key is required.")
        has_error = True
    if not category:
        st.sidebar.error("❌ Category is required.")
        has_error = True
    if not country:
        st.sidebar.error("❌ Country is required.")
        has_error = True
    if uploaded_file is None:
        st.sidebar.error("❌ Please upload a file.")
        has_error = True

    if not has_error:
        # 1. Validate API Key FIRST
        with st.spinner("Validating Gemini API Key..."):
            key_is_valid, key_error = asyncio.run(validate_gemini_api_key(api_key))

        if not key_is_valid:
            st.error(f"API Key Validation Failed: {key_error}")
        else:
            if key_error: # Handle warnings like model not found
                 st.warning(key_error)
            st.success("API Key seems valid. Proceeding...")

            # 2. Parse Links
            with st.spinner("Parsing uploaded file..."):
                links = parse_links(uploaded_file)

            if not links:
                st.warning("No valid WhatsApp links found in the file or file couldn't be parsed.")
            else:
                link_count = len(links)
                if link_count > MAX_LINKS_PER_UPLOAD:
                    st.warning(f"⚠️ Found {link_count} links. Processing only the first {MAX_LINKS_PER_UPLOAD}.")
                    links = links[:MAX_LINKS_PER_UPLOAD]

                st.info(f"Found {len(links)} valid links. Starting processing with a {PROCESSING_DELAY_SECONDS}s delay between links...")

                start_time = time.time()
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Processing 0/{} links...".format(len(links)))

                # 3. Run Async Processing
                results = []
                try:
                    # Ensure an event loop exists and run the main async function
                    results = asyncio.run(process_all_links_async(links, api_key, category, country, progress_bar, status_text))
                    st.session_state.processing_done = True # Mark as done
                except Exception as e:
                    st.error(f"An unexpected error occurred during processing: {e}")
                    logger.error("Error during asyncio.run(process_all_links_async)", exc_info=True)
                    st.session_state.processing_done = False # Mark as not successfully done

                end_time = time.time()
                processing_time = end_time - start_time

                if st.session_state.processing_done:
                     progress_bar.progress(1.0)
                     status_text.success(f"🏁 Processing Complete in {processing_time:.2f} seconds!")
                     logger.info(f"Processed {len(links)} links in {processing_time:.2f} seconds.")

                     if results:
                          df = pd.DataFrame(results)
                          # Select and reorder columns for final output/display
                          st.session_state.results_df = df[CSV_HEADERS]
                     else:
                          st.warning("Processing finished, but no results were generated.")
                          st.session_state.results_df = None

# --- Display Results & Download ---
if st.session_state.get('processing_done') and st.session_state.get('results_df') is not None:
    st.subheader("📊 Processed Data")
    df_display = st.session_state.results_df
    st.dataframe(df_display)

    # Calculate basic stats from the original results DataFrame before column selection
    # Need to re-run processing or store the full 'results' list in session state for this
    # For simplicity, we'll just count visible errors in the final DF
    fetch_failed_count = df_display['Group Name'].str.contains("Skipped/Invalid", na=False).sum()
    ai_error_count = df_display['Description'].str.contains("Error:", na=False).sum() + \
                     df_display['Keywords'].str.contains("Error", na=False).sum()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Links Skipped / Fetch Failed", value=f"{fetch_failed_count} / {len(df_display)}")
    with col2:
        st.metric("AI Generation Issues", value=f"{ai_error_count} / {len(df_display)}")

    if fetch_failed_count > 0:
        st.warning("Some links were skipped due to being invalid, expired, or fetch errors (see 'Skipped/Invalid' in 'Group Name').")
    if ai_error_count > 0:
        st.warning("Some AI descriptions/keywords could not be generated (see 'Error:' in 'Description'/'Keywords'). Check API key, quota, or Gemini status.")

    # Prepare CSV for download
    csv_buffer = io.StringIO()
    df_display.to_csv(csv_buffer, index=False, encoding='utf-8-sig') # utf-8-sig for better Excel compatibility
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name=f"whatsapp_group_data_{category}_{country}_{int(time.time())}.csv",
        mime="text/csv",
        key="download_button"
    )
elif st.session_state.get('processing_done'):
     st.info("Processing finished, but no data was generated or an error occurred.")


st.sidebar.markdown("---")
st.sidebar.info("App V2 - Added API key validation & processing delay.")
