---
title: WhatsApp SEO Generator (Proxy Ver)
emoji: 🌍
colorFrom: blue # Valid color
colorTo: indigo # Valid color
sdk: streamlit
sdk_version: "1.33.0" # Match requirements.txt
app_file: app.py
pinned: false
# python_version: "3.10" # Specify if needed
---

# WhatsApp Group SEO Content Generator (v2.1 - Proxy Enabled)

This Streamlit application attempts to automate enriching WhatsApp group links with scraped metadata and AI-generated SEO content. **This version is designed to use external proxies for scraping reliability.**

## Core Features

-   **Proxy Integration:** Uses HTTP/SOCKS5 proxies configured via Hugging Face Secrets to improve scraping success against WhatsApp's anti-bot measures.
-   **CSV Input:** Processes up to 50 unique WhatsApp group links (`Join Links` column).
-   **Resilient Scraping:** Attempts data extraction using configurable selectors; handles network/proxy/HTTP errors with retries and specific logging.
-   **AI Content (Gemini):** Generates SEO descriptions & keywords using a configurable Gemini model (default: `gemini-1.5-flash-latest`). Handles API errors robustly.
-   **Graceful Fallbacks:** Provides fallback data/content when scraping or API calls fail.
-   **Detailed Error Handling:** Logs specific error types (Network, Proxy, Scrape, API, Config) to UI and downloadable report.
-   **Secure Configuration:** Relies on **Hugging Face Secrets** for API Key and Proxy Credentials.

## MANDATORY Setup: Proxies & Secrets

**This application WILL likely FAIL to scrape reliably without high-quality proxies.** Free proxies are insufficient.

1.  **Obtain Paid Proxies:** Get rotating residential or mobile proxies from a reputable provider (e.g., Bright Data, Oxylabs, Smartproxy). You will need Host, Port, Username, and Password.
2.  **Set Hugging Face Secrets:** In your Space -> Settings -> Secrets, add:
    *   `GEMINI_API_KEY`: Your Google Gemini API Key.
    *   `PROXY_HOST`: Your proxy provider's gateway host.
    *   `PROXY_PORT`: The proxy port.
    *   `PROXY_USER`: Your proxy username.
    *   `PROXY_PASS`: Your proxy password.
    *   `PROXY_TYPE`: `http`, `socks5`, or `socks5h` (use `socks5h` if available for DNS via proxy).

## How to Use

1.  **Setup Secrets:** Configure **all** required secrets as described above.
2.  **Prepare CSV:** Create `Join Links` CSV (max 50 unique URLs).
3.  **Run:** Access deployed Space or run locally (`pip install -r requirements.txt`, `streamlit run app.py`).
4.  **Configure (Sidebar):** Enter Country/Category, Upload CSV. Check Proxy/API status indicators.
5.  **Process:** Click the main **Process Links** button.
6.  **Monitor & Download:** Track progress. Results & download appear below. Check error log.

## CSV Output Columns

(Same as previous version: Group Name, Join Link, Description, Category, Country, Keywords, Profile Image) - Fields will indicate errors where appropriate.

## Deployment

(Standard HF Spaces deployment: Create Space, Streamlit SDK, Upload `app.py`, `requirements.txt`, `README.md`) - Ensure pinned `requirements.txt` versions.

## 🚨 CRITICAL DISCLAIMERS (READ CAREFULLY) 🚨

*   **PAID PROXIES ARE ESSENTIAL:** Scraping WhatsApp reliably from cloud platforms **requires good rotating residential/mobile proxies.** Configure them via Secrets. Without them, expect frequent scraping failures (Network/DNS/Block errors).
*   **SCRAPING IS FRAGILE & REQUIRES MAINTENANCE:** WhatsApp **constantly changes its website**. The CSS selectors (`SCRAPE_..._SELECTORS` in `app.py`) **WILL BREAK**. This app requires ongoing developer effort to update selectors by inspecting live WhatsApp pages.
*   **ENVIRONMENTAL LIMITATIONS:** Cloud network restrictions or proxy provider issues can still cause failures.
*   **TERMS OF SERVICE:** Automation likely violates WhatsApp's ToS. Use ethically and at your own risk.
*   **COSTS:** You are responsible for all proxy service costs and Gemini API usage costs.
*   **NO GUARANTEES / LIABILITY:** This is a tool attempting a difficult task. Success is not guaranteed. No liability is assumed.