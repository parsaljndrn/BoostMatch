import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/144.0.0.0 Safari/537.36"
    )
}

MIN_ARTICLE_LENGTH = 300

SOCIAL_DOMAINS = [
    "facebook.com",
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "fb.watch"
]


def extract_article_for_nlp(url: str) -> str:
    """
    STRICT article extractor.
    Raises ValueError if link is missing or not an article.
    """

    if not url:
        raise ValueError(
            "This Facebook post does not contain an article link. "
            "Please paste a Facebook post with an article attached."
        )

    if _is_social_or_video_link(url):
        raise ValueError(
            "The link attached is not an article link. "
            "Please paste a Facebook post link with an article link attached to it."
        )

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        raise ValueError(
            "Failed to fetch the attached article. "
            "The website may be unreachable or blocked."
        )

    soup = BeautifulSoup(response.text, "html.parser")
    _clean_dom(soup)

    article_text = _extract_main_text(soup)

    if len(article_text) < MIN_ARTICLE_LENGTH:
        raise ValueError(
            "The link attached is not an article link. "
            "Please paste a Facebook post link with an article link attached to it."
        )

    return article_text


# ================= HELPERS =================

def _is_social_or_video_link(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(d in domain for d in SOCIAL_DOMAINS)


def _clean_dom(soup):
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()


def _extract_main_text(soup) -> str:
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

    if not paragraphs:
        paragraphs = [
            p.get_text(strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 40
        ]

    return " ".join(paragraphs)