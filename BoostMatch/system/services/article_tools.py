# services/article_tools.py

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

# -----------------------------------
# Shared request headers
# -----------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/144.0.0.0 Safari/537.36"
    )
}

# ===================================
# CORE PUBLIC FUNCTIONS
# ===================================

def get_article_content(url: str) -> str:
    """
    NLP / ML optimized extractor.
    Returns CLEAN article text only.
    This is what SBERT & Boosting models should use.
    """
    data = _extract_article_data(url)
    return data.get("content", "")


def get_article_data(url: str) -> dict:
    """
    UI / explainability extractor.
    Returns structured article information.
    """
    return _extract_article_data(url)


# ===================================
# INTERNAL IMPLEMENTATION
# ===================================

def _extract_article_data(url: str) -> dict:
    if not url:
        return _empty_article(url)

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return _empty_article(url)

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noisy elements
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()

    title = _extract_title(soup)
    description = _extract_description(soup)
    publish_date = _extract_publish_date(soup)
    source = _extract_source(url, soup)
    content = _extract_main_text(soup)

    return {
        "title": title,
        "description": description,
        "content": content,
        "source": source,
        "publish_date": publish_date,
        "url": url
    }


# ===================================
# EXTRACTION HELPERS
# ===================================

def _extract_title(soup):
    for selector in [
        ("meta", {"property": "og:title"}),
        ("meta", {"name": "title"}),
    ]:
        tag = soup.find(*selector)
        if tag and tag.get("content"):
            return tag["content"].strip()

    if soup.title:
        return soup.title.get_text(strip=True)

    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else None


def _extract_description(soup):
    tag = soup.find("meta", property="og:description")
    if tag and tag.get("content"):
        return tag["content"].strip()

    tag = soup.find("meta", attrs={"name": "description"})
    return tag["content"].strip() if tag else None


def _extract_publish_date(soup):
    tag = soup.find("meta", property="article:published_time")
    return tag["content"] if tag else None


def _extract_source(url, soup):
    tag = soup.find("meta", property="og:site_name")
    if tag and tag.get("content"):
        return tag["content"]

    return urlparse(url).netloc


def _extract_main_text(soup) -> str:
    """
    Clean article text optimized for NLP similarity
    """
    containers = [
        soup.find("article"),
        soup.find("main"),
        soup.find("div", class_=re.compile(r"(article|content|post|story)", re.I)),
        soup.find("div", id=re.compile(r"(article|content|post|story)", re.I)),
    ]

    paragraphs = []

    for container in containers:
        if container:
            paragraphs = [
                p.get_text(strip=True)
                for p in container.find_all("p")
                if len(p.get_text(strip=True)) > 40
            ]
            if paragraphs:
                break

    # Fallback: all paragraphs
    if not paragraphs:
        paragraphs = [
            p.get_text(strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 40
        ]

    text = " ".join(paragraphs)
    return " ".join(text.split())


def _empty_article(url):
    return {
        "title": None,
        "description": None,
        "content": "",
        "source": urlparse(url).netloc if url else None,
        "publish_date": None,
        "url": url
    }