import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
from deep_translator import GoogleTranslator
from langdetect import detect

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

# =====================================================
# TRANSLATION HELPER
# =====================================================

def _translate_to_english(text: str) -> str:
    """
    Detects language and translates to English ONLY if needed.
    Handles large text safely using chunking.
    """
    if not text:
        return text

    try:
        lang = detect(text)

        # Skip translation if already English
        if lang == "en":
            return text

        translator = GoogleTranslator(source='auto', target='en')

        # Handle long text (Google limit ~5000 chars)
        if len(text) > 4500:
            chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
            translated_chunks = [translator.translate(chunk) for chunk in chunks]
            return " ".join(translated_chunks)

        return translator.translate(text)

    except Exception as e:
        print(f"[Warning] Translation failed: {e}")
        return text
    

# =====================================================
# HEADLINE EXTRACTOR (NEW FUNCTION)
# =====================================================

def extract_article_headline(url: str) -> str | None:
    """
    Extracts the headline of an article.
    
    Returns the headline if extraction succeeds.
    Returns None if the post does not contain a valid article.
    """

    if not url:
        # No article link
        return None

    if _is_social_or_video_link(url):
        # Not an article link
        return None

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        # Could not fetch URL
        print(f"[Warning] Failed to fetch article: {url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    _clean_dom(soup)

    headline = _extract_headline(soup)

    if not headline:
        # Could not extract headline
        print(f"[Warning] Unable to extract headline from: {url}")
        return None
    
    # Optional: translate headline
    headline = _translate_to_english(headline)

    return headline

# =====================================================
# ORIGINAL FULL ARTICLE EXTRACTOR (UNCHANGED)
# =====================================================

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

    # ✅ NEW STEP: TRANSLATION (AFTER EXTRACTION, BEFORE MODEL)
    article_text = _translate_to_english(article_text)

    return article_text


# ================= HELPERS =================

def _is_social_or_video_link(url: str):
    domain = urlparse(url).netloc.lower()
    return any(d in domain for d in SOCIAL_DOMAINS)


def _clean_dom(soup):
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()


# =====================================================
# HEADLINE EXTRACTION LOGIC
# =====================================================

def _extract_headline(soup):

    # 1️⃣ OpenGraph title (most reliable)
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title.get("content").strip()

    # ⭐ IMPROVEMENT (Twitter headline support)
    twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
    if twitter_title and twitter_title.get("content"):
        return twitter_title.get("content").strip()

    # 3️⃣ Standard HTML <title>
    title_tag = soup.find("title")
    if title_tag and title_tag.text:
        return title_tag.text.strip()

    # 4️⃣ First H1
    h1_tag = soup.find("h1")
    if h1_tag and h1_tag.text:
        return h1_tag.text.strip()

    return None


# =====================================================
# ARTICLE TEXT EXTRACTION
# =====================================================

def _extract_main_text(soup):

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