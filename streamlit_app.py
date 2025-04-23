
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

# Apply nest_asyncio to allow running asyncio event loops within Streamlit
nest_asyncio.apply()

# --- Configuration ---
APP_TITLE = "WhatsApp Group Info Extractor & Enricher"
MAX_LINKS_PER_UPLOAD = 50
CSV_HEADERS = ["Group Name", "Join Link", "Description", "Category", "Country", "Keywords", "Profile Image URL"]
GEMINI_MODEL = 'gemini-1.5-flash' # Or use 'gemini-pro' if needed

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
            # Try decoding as UTF-8, ignore errors for broader compatibility
            content_str = content.decode('utf-8', errors='ignore')

            if uploaded_file.name.lower().endswith('.csv'):
                # Read CSV, assuming links are in the first column, skip header if present
                df = pd.read_csv(io.StringIO(content_str), header=None)
                # Check if first row looks like a header (optional, basic check)
                if isinstance(df.iloc[0, 0], str) and ("http" not in df.iloc[0, 0].lower()):
                     df = pd.read_csv(io.StringIO(content_str)) # Re-read with header
                links = df.iloc[:, 0].astype(str).str.strip().tolist()

            elif uploaded_file.name.lower().endswith('.txt'):
                links = [line.strip() for line in content_str.splitlines() if line.strip()]
            else:
                st.error("Unsupported file format. Please upload a CSV or TXT file.")
                return []

            # Basic validation: Filter for potential WhatsApp links
            whatsapp_pattern = re.compile(r'https?://chat\.whatsapp\.com/([a-zA-Z0-9]{22})')
            valid_links = [link for link in links if whatsapp_pattern.match(link)]
            invalid_count = len(links) - len(valid_links)
            if invalid_count > 0:
                 st.warning(f"Filtered out {invalid_count} lines that didn't look like WhatsApp group links.")
            return valid_links

        except Exception as e:
            st.error(f"Error reading file: {e}")
            logger.error(f"Error parsing file {uploaded_file.name}: {e}", exc_info=True)
            return []
    return []

async def fetch_group_info_async(session, link):
    """
    --- CRITICAL WARNING ---
    This function attempts to fetch group info. WhatsApp heavily protects against this.
    It's HIGHLY UNRELIABLE and likely to FAIL or return generic HTML.
    It does NOT use any official API. Success is NOT guaranteed.
    We simulate success in case of failure for demo purposes.
    DO NOT rely on this for production without a more robust (and likely ToS-violating) method.
    --- CRITICAL WARNING ---
    """
    # Placeholder data in case of failure
    placeholder_name = f"Group (Check Link Manually)"
    placeholder_pic_url = "https://via.placeholder.com/150/CCCCCC/808080?text=No+Image"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    # Basic timeout for the request
    timeout = aiohttp.ClientTimeout(total=10) # 10 seconds total timeout

    try:
        # VERY BASIC ATTEMPT - WhatsApp Web's invite page structure changes and uses JS heavily
        async with session.get(link, headers=headers, timeout=timeout, allow_redirects=True) as response:
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            html_content = await response.text()

            # --- Extremely Fragile Parsing Logic ---
            # This will break easily if WhatsApp changes their page structure.
            group_name = placeholder_name
            profile_pic_url = placeholder_pic_url

            # Try finding name using common patterns (highly unreliable)
            name_match = re.search(r'<meta property="og:title" content="(.*?)"', html_content)
            if name_match:
                group_name = name_match.group(1).strip()
            else:
                # Fallback attempt (even less reliable)
                name_match_alt = re.search(r'<h2 class=".*?">(.*?)</h2>', html_content, re.IGNORECASE)
                if name_match_alt:
                     group_name = name_match_alt.group(1).strip()


            # Try finding image using common patterns (highly unreliable)
            pic_match = re.search(r'<meta property="og:image" content="(.*?)"', html_content)
            if pic_match:
                profile_pic_url = pic_match.group(1).strip()

            logger.info(f"Attempted fetch for {link}. Found Name: '{group_name != placeholder_name}', Found Pic: '{profile_pic_url != placeholder_pic_url}'")
            # Simulate success if parsing failed, for demo purposes ONLY
            # In a real scenario, you'd likely return None or raise an error here
            # if group_name == placeholder_name:
            #    logger.warning(f"Using placeholder name for {link}")
            # if profile_pic_url == placeholder_pic_url:
            #     logger.warning(f"Using placeholder image for {link}")

            return {"name": group_name, "pic_url": profile_pic_url, "error": None}

    except aiohttp.ClientResponseError as e:
        logger.warning(f"HTTP Error fetching {link}: {e.status} {e.message}")
        error_msg = f"HTTP Error {e.status}. Link might be expired, private, or invalid."
        if e.status == 404:
            error_msg = "Group not found (404) or link expired."
        return {"name": placeholder_name, "pic_url": placeholder_pic_url, "error": error_msg}
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching {link}")
        return {"name": placeholder_name, "pic_url": placeholder_pic_url, "error": "Request timed out."}
    except aiohttp.ClientError as e:
        logger.error(f"ClientError fetching {link}: {e}", exc_info=False) # Avoid traceback spam
        return {"name": placeholder_name, "pic_url": placeholder_pic_url, "error": f"Network/Client Error: {type(e).__name__}."}
    except Exception as e:
        logger.error(f"Unexpected error fetching {link}: {e}", exc_info=True)
        return {"name": placeholder_name, "pic_url": placeholder_pic_url, "error": f"Unexpected fetch error: {type(e).__name__}."}


async def generate_ai_content_async(api_key, group_name, category, country):
    """Generates description and keywords using Gemini API asynchronously."""
    if not group_name or group_name.startswith("Group (Check Link Manually)"):
         return {"description": "N/A (Could not fetch group name)", "keywords": "N/A", "error": "Missing group name"}

    # Configure Gemini
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")
        return {"description": "Error", "keywords": "Error", "error": f"Gemini Configuration Error: {e}"}

    # Construct the prompt
    prompt = f"""
    Given the following WhatsApp Group information:
    Group Name: "{group_name}"
    Category: "{category}"
    Country: "{country}"

    Generate the following, focusing on SEO optimization for a group directory website:
    1. A unique, engaging, and SEO-friendly description (maximum 50 words). Describe what the group is likely about based on its name, category, and country.
    2. Exactly 3 relevant keywords (comma-separated) based on the group name, category, and country.

    Format the output EXACTLY like this, with no extra text before or after:
    Description: [Your generated description here]
    Keywords: [Keyword1, Keyword2, Keyword3]
    """

    retries = 2
    for attempt in range(retries):
        try:
            # Use generate_content_async for async call if available and needed,
            # but standard generate_content is often sufficient within an async function
            # if it doesn't block the *entire* event loop excessively.
            # For simplicity here, using the sync version within the async task.
            # For true non-blocking AI calls, explore the async version of the library if provided.
            response = model.generate_content(prompt)
            response.resolve() # Ensure response is complete, handles potential streaming internally

            if not response.parts:
                 raise ValueError("Gemini response is empty or malformed.")

            content = response.text.strip()
            logger.info(f"Gemini Raw Response for '{group_name}': {content[:100]}...") # Log snippet

            # --- Parse the response ---
            desc_match = re.search(r"Description:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
            keywords_match = re.search(r"Keywords:\s*(.*)", content, re.IGNORECASE | re.DOTALL)

            description = desc_match.group(1).strip() if desc_match else "Error: Could not parse description."
            keywords = keywords_match.group(1).strip() if keywords_match else "Error: Could not parse keywords."

            # Simple cleanup / validation
            if len(description) > 70 * 6: # A bit more than 50 words average chars
                 description = description[:350] + "..." # Rough truncate
            keywords = ", ".join([k.strip() for k in keywords.split(',')[:3]]) # Ensure max 3 keywords

            return {"description": description, "keywords": keywords, "error": None}

        except Exception as e:
            logger.error(f"Error generating content for '{group_name}' (Attempt {attempt + 1}/{retries}): {e}", exc_info=False)
            if "API key not valid" in str(e):
                 st.error("Google Gemini API Key is invalid. Please check and re-enter.")
                 return {"description": "Error", "keywords": "Error", "error": "Invalid API Key"}
            if "rate limit" in str(e).lower():
                 error_msg = "Rate limit exceeded. Please wait and try again."
                 if attempt < retries - 1:
                     await asyncio.sleep(2 ** attempt) # Exponential backoff
                     continue
                 else:
                     return {"description": "Error", "keywords": "Error", "error": error_msg}
            # Handle other specific Gemini errors if needed
            error_msg = f"Gemini API Error: {type(e).__name__}"
            if attempt < retries - 1:
                 await asyncio.sleep(1) # Simple delay before retry
                 continue
            else:
                 return {"description": "Error", "keywords": "Error", "error": error_msg}

    return {"description": "Error", "keywords": "Error", "error": "Gemini failed after retries."} # Should not be reached if retry logic is correct

async def process_link_data(session, link, api_key, category, country):
    """Fetches info and then generates AI content for a single link."""
    fetch_result = await fetch_group_info_async(session, link)
    group_name = fetch_result["name"]
    pic_url = fetch_result["pic_url"]
    fetch_error = fetch_result["error"]

    # Log fetch errors clearly
    if fetch_error:
        logger.warning(f"Fetch failed for {link}: {fetch_error}")
        # Decide if you want to proceed with AI generation even if fetch failed partially
        # For now, we proceed if we got *any* name, even the placeholder
        # If fetch error is critical (e.g., 404), maybe skip AI?

    ai_result = await generate_ai_content_async(api_key, group_name, category, country)
    description = ai_result["description"]
    keywords = ai_result["keywords"]
    ai_error = ai_result["error"]

    # Combine results
    return {
        "Group Name": group_name if not fetch_error or group_name != f"Group (Check Link Manually)" else f"Fetch Failed ({fetch_error})",
        "Join Link": link,
        "Description": description if not ai_error else f"AI Error: {ai_error}",
        "Category": category,
        "Country": country,
        "Keywords": keywords if not ai_error else "AI Error",
        "Profile Image URL": pic_url if not fetch_error else "Fetch Error",
        "fetch_error": fetch_error, # Keep track of errors internally if needed
        "ai_error": ai_error
    }


async def process_all_links(links, api_key, category, country, progress_bar, status_text):
    """Processes all links concurrently."""
    results = []
    total_links = len(links)
    processed_count = 0

    # Use aiohttp ClientSession for connection pooling
    async with aiohttp.ClientSession() as session:
        tasks = [process_link_data(session, link, api_key, category, country) for link in links]

        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                # This catches errors *within* the asyncio task handling itself, less common
                logger.error(f"Critical error processing a link task: {e}", exc_info=True)
                # Add a placeholder error result
                results.append({
                    "Group Name": "Processing Error", "Join Link": "Unknown", "Description": str(e),
                    "Category": category, "Country": country, "Keywords": "Error", "Profile Image URL": "Error",
                    "fetch_error": str(e), "ai_error": None
                 })

            processed_count += 1
            progress = processed_count / total_links
            progress_bar.progress(progress)
            status_text.text(f"Processed {processed_count}/{total_links} links...")

    return results

# --- Streamlit App UI ---

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"🚀 {APP_TITLE}")

st.markdown("""
Upload a list of WhatsApp group invite links (CSV or TXT, one link per line or in the first column).
The app will attempt to fetch the group name and profile picture, then use Google Gemini
to generate an SEO-friendly description and keywords based on the name, category, and country you provide.

**Important Notes:**
*   **Fetching group info is unreliable** and may fail due to WhatsApp's restrictions. Results might show placeholders.
*   Provide your **Google Gemini API Key**. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).
*   Maximum **{} links** per upload.
*   Processing speed depends on network latency and Gemini API response times. The target of 1.5 seconds for 50 links is **highly unrealistic**.
""".format(MAX_LINKS_PER_UPLOAD))

st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter your Google Gemini API Key", type="password")
category = st.sidebar.text_input("Enter Category for these groups", placeholder="e.g., Technology, Gaming, News")
country = st.sidebar.text_input("Enter Country for these groups", placeholder="e.g., India, USA, Global")

st.sidebar.header("📤 Upload Links")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or TXT file", type=["csv", "txt"])

process_button = st.sidebar.button("✨ Process Links ✨", disabled=(not uploaded_file or not api_key or not category or not country))

if process_button:
    if not api_key:
        st.error("❌ Please enter your Google Gemini API Key.")
    elif not category:
        st.error("❌ Please enter a Category.")
    elif not country:
        st.error("❌ Please enter a Country.")
    elif uploaded_file is None:
        st.error("❌ Please upload a file.")
    else:
        st.info("Parsing uploaded file...")
        links = parse_links(uploaded_file)

        if not links:
            st.warning("No valid WhatsApp links found in the file.")
        else:
            link_count = len(links)
            if link_count > MAX_LINKS_PER_UPLOAD:
                st.warning(f"⚠️ Found {link_count} links. Processing the first {MAX_LINKS_PER_UPLOAD}.")
                links = links[:MAX_LINKS_PER_UPLOAD]

            st.info(f"Found {len(links)} valid links. Starting processing...")

            start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Processing 0/{} links...".format(len(links)))

            # Run the async processing function
            # Need to get or create an event loop when running within Streamlit/threading
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                     loop = asyncio.new_event_loop()
                     asyncio.set_event_loop(loop)

                results = loop.run_until_complete(process_all_links(links, api_key, category, country, progress_bar, status_text))

            except RuntimeError as e:
                 if "cannot run loop while another loop is running" in str(e):
                      # If nest_asyncio didn't work or is unavailable, run directly
                      logger.warning("Asyncio loop conflict, running directly.")
                      results = asyncio.run(process_all_links(links, api_key, category, country, progress_bar, status_text))
                 else:
                     raise e # Re-raise other runtime errors

            end_time = time.time()
            processing_time = end_time - start_time

            progress_bar.progress(1.0) # Ensure bar is full
            status_text.text(f"🏁 Processing Complete in {processing_time:.2f} seconds!")
            logger.info(f"Processed {len(links)} links in {processing_time:.2f} seconds.")

            # --- Display Results & Download ---
            if results:
                df = pd.DataFrame(results)

                # Reorder columns to the desired format and select only them
                df_display = df[CSV_HEADERS]

                st.subheader("📊 Processed Data Preview")
                st.dataframe(df_display)

                # Count errors
                fetch_errors = df['fetch_error'].notna().sum()
                ai_errors = df['ai_error'].notna().sum()
                st.metric("Fetch Issues (Name/Pic)", value=f"{fetch_errors} / {len(df)}")
                st.metric("AI Generation Issues", value=f"{ai_errors} / {len(df)}")
                if fetch_errors > 0:
                    st.warning("Some group names/images couldn't be fetched reliably (see placeholders/errors). This is expected behavior.")


                # Prepare CSV for download
                csv_buffer = io.StringIO()
                df_display.to_csv(csv_buffer, index=False, encoding='utf-8')
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"whatsapp_group_data_{category}_{country}.csv",
                    mime="text/csv",
                )
            else:
                st.error("No results were generated. Check logs or inputs.")

st.sidebar.markdown("---")
st.sidebar.info("App developed based on user requirements. Fetching reliability is limited.")
