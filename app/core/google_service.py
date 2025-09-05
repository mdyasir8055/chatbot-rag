# app/core/google_service.py

import os
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# Allow either GOOGLE_CSE_ID or GOOGLE_CX in .env
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "") or os.getenv("GOOGLE_CX", "")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ProductQAAssistant/1.0; +https://example.local)"
}

# =========================
# Google Custom Search
# =========================
def google_search(query: str, num: int = 5) -> List[Tuple[str, str]]:
    """Search Google Custom Search API and return (title, link)."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "num": num
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("items", [])
    return [(item["title"], item["link"]) for item in items]

# =========================
# Web Scraping
# =========================
def fetch_page_content(url: str) -> str:
    """Fetch a webpage and extract visible text."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        return text[:4000]  # truncate for token safety
    except Exception as e:
        return f"(fetch-error) {e}"
