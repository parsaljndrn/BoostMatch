import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


AD_DOMAINS = [
    "shopee", "lazada", "tokopedia", "amazon", "ebay", "aliexpress",
    "shein", "temu", "zalora", "grabfood", "foodpanda",
    "doubleclick", "googleadservices", "googlesyndication",
    "adservice", "pagead", "adclick", "adnxs", "taboola",
    "outbrain", "revcontent", "mgid", "zergnet", "sharethrough",
    "ads.yahoo", "adsystem", "adroll", "criteo", "adform",
    "pubmatic", "rubiconproject", "openx", "appnexus",
    "scorecardresearch", "quantserve", "chartbeat",
]

SHOP_DOMAINS = [
    "shopee", "lazada", "tokopedia", "amazon", "ebay", "aliexpress",
    "shein", "temu", "zalora", "etsy", "walmart", "target",
    "bestbuy", "rakuten", "jd.com", "flipkart",
]

SOCIAL_DOMAINS = [
    "facebook.com", "fb.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "youtube.com", "youtu.be", "reddit.com",
    "linkedin.com", "pinterest.com", "snapchat.com",
]

VIDEO_DOMAINS = [
    "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
    "twitch.tv", "streamable.com",
]

AD_CLASS_ID_PATTERN = re.compile(
    r"ad|ads|advert|advertisement|sponsor|promo|banner|popup|"
    r"widget|sidebar|related|recommended|taboola|outbrain",
    re.IGNORECASE,
)

AD_TEXT_PATTERN = re.compile(
    r"(sponsored|advertisement|shop\s+now|buy\s+now|add\s+to\s+cart"
    r"|free\s+shipping|promo\s+code|discount\s+code|click\s+here\s+to\s+buy"
    r"|limited\s+offer|flash\s+sale|exclusive\s+deal)",
    re.IGNORECASE,
)


def classify_link(url: str) -> str:
    """
    Classify a URL before fetching it.

    Returns one of:
      'article'  - likely a news/blog article
      'shop'     - e-commerce / product page
      'social'   - social media platform
      'video'    - video platform
      'unknown'  - could not determine, will still attempt extraction
    """
    if not url:
        return "unknown"

    domain = urlparse(url).netloc.lower()
    domain = domain.replace("www.", "")

    if any(d in domain for d in SHOP_DOMAINS):
        return "shop"
    if any(d in domain for d in VIDEO_DOMAINS):
        return "video"
    if any(d in domain for d in SOCIAL_DOMAINS):
        return "social"

    return "article"


def _clean_soup(soup):
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "form"]):
        tag.decompose()

    for a_tag in soup.find_all("a", href=True):
        if any(domain in a_tag.get("href", "").lower() for domain in AD_DOMAINS):
            a_tag.decompose()

    for tag in soup.find_all(class_=AD_CLASS_ID_PATTERN):
        tag.decompose()

    for tag in soup.find_all(id=AD_CLASS_ID_PATTERN):
        tag.decompose()

    return soup


def _extract_paragraphs(soup) -> str:
    clean = []
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text and not AD_TEXT_PATTERN.search(text):
            clean.append(text)
    return " ".join(clean)


def _extract_fallback(soup) -> str:
    seen = set()
    chunks = []
    candidates = soup.find_all(["div", "span", "li", "td", "blockquote", "section", "article", "h1", "h2", "h3", "h4"])
    for tag in candidates:
        text = tag.get_text(separator=" ", strip=True)
        if len(text) > 40 and text not in seen and not AD_TEXT_PATTERN.search(text):
            seen.add(text)
            chunks.append(text)
    return " ".join(chunks)


def get_article_content(url: str) -> tuple[str, str]:
    """
    Fetch and clean content from a URL.

    Returns:
        (text_content, link_type)
        link_type is one of: 'article', 'shop', 'video', 'social', 'unknown'
    """
    if not url or not url.strip():
        return "", "unknown"

    link_type = classify_link(url)

    if link_type == "shop":
        print(f"[article_tools] Skipping shop link: {url}")
        return "", "shop"

    if link_type == "social":
        print(f"[article_tools] Skipping social link: {url}")
        return "", "social"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[article_tools] Failed to fetch: {e}")
        return "", link_type

    soup = BeautifulSoup(response.text, "html.parser")
    soup = _clean_soup(soup)

    text_content = _extract_paragraphs(soup)

    if len(text_content.strip()) < 100:
        print("[article_tools] Paragraph extraction too short, trying fallback...")
        text_content = _extract_fallback(soup)

    text_content = " ".join(text_content.split())

    if len(text_content.strip()) < 50:
        link_type = "unknown"

    return text_content, link_type
