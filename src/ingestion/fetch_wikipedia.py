"""
Wikipedia page fetcher and cleaner.
Downloads a Wikipedia page by URL, parses the HTML with BeautifulSoup,
extracts the main article text (paragraphs only), and removes citation
markers like [1], [2]. Returns a clean (title, text) pair ready for chunking.
"""

from typing import Tuple

import requests
from bs4 import BeautifulSoup

from src.utils.text_cleaning import basic_clean


def fetch_html(url: str, timeout_seconds: int = 20) -> str:
    response = requests.get(url, timeout=timeout_seconds, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return response.text


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1")
    if title_tag and title_tag.get_text(strip=True):
        return title_tag.get_text(strip=True)
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)
    return "Unknown Title"


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    if content is None:
        return ""

    # Prefer paragraph text
    paragraphs = content.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    return basic_clean(text)


def is_valid_page(text: str, min_words: int = 200) -> bool:
    return len(text.split()) >= min_words


def fetch_and_clean(url: str) -> Tuple[str, str]:
    html = fetch_html(url)
    title = extract_title(html)
    text = extract_main_text(html)
    return title, text
