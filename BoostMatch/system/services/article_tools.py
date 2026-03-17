import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

MIN_ARTICLE_LENGTH = 50

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


def _domain_of(url: str) -> str:
    return urlparse(url).netloc.lower().replace("www.", "")


def classify_link(url: str) -> str:
    if not url:
        return "unknown"
    domain = _domain_of(url)
    if any(d in domain for d in SHOP_DOMAINS):
        return "shop"
    if any(d in domain for d in VIDEO_DOMAINS):
        return "video"
    if any(d in domain for d in SOCIAL_DOMAINS):
        return "social"
    return "article"


def _check_redirect(url: str) -> tuple[str, bool]:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10,
                             headers={"User-Agent": "Mozilla/5.0"})
        final_url = resp.url
        if final_url and final_url.rstrip("/") != url.rstrip("/"):
            final_domain = _domain_of(final_url)
            is_ad = any(d in final_domain for d in AD_DOMAINS + SHOP_DOMAINS)
            print(f"[article_tools] Redirect: {url} -> {final_url} | ad={is_ad}")
            return final_url, is_ad
    except Exception as e:
        print(f"[article_tools] Redirect check failed: {e}")
    return url, False


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


def _extract_headline(soup):
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title.get("content").strip()

    twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
    if twitter_title and twitter_title.get("content"):
        return twitter_title.get("content").strip()

    title_tag = soup.find("title")
    if title_tag and title_tag.text:
        return title_tag.text.strip()

    h1_tag = soup.find("h1")
    if h1_tag and h1_tag.text:
        return h1_tag.text.strip()

    return None


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


def extract_article_headline(url: str) -> str:
    if not url:
        return None

    if _is_social_or_video_link(url):
        return None

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        print(f"[article_tools] Failed to fetch headline from: {url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    _clean_soup(soup)

    headline = _extract_headline(soup)

    if not headline:
        print(f"[article_tools] Unable to extract headline from: {url}")
        return None

    return headline


def _is_social_or_video_link(url: str):
    domain = urlparse(url).netloc.lower()
    return any(d in domain for d in SOCIAL_DOMAINS + VIDEO_DOMAINS)


def get_article_content(url: str) -> tuple[str, str]:
    if not url or not url.strip():
        return "", "unknown"

    final_url, is_ad_redirect = _check_redirect(url)

    if is_ad_redirect:
        print("LINK REDIRECTS TO AN AD")
        return "", "ad_redirect"

    link_type = classify_link(final_url)

    if link_type == "shop":
        print("LINK REDIRECTS TO AN AD")
        return "", "ad_redirect"

    if link_type in ("social", "video"):
        print(f"CANNOT ANALYZE: LINK ATTACHMENT IS NOT AN ARTICLE ({link_type})")
        return "", "not_article"

    try:
        response = requests.get(final_url, headers=HEADERS, timeout=12)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[article_tools] Failed to fetch: {e}")
        return "", "unknown"

    soup = BeautifulSoup(response.text, "html.parser")
    soup = _clean_soup(soup)

    text_content = _extract_paragraphs(soup)

    if len(text_content.strip()) < 100:
        print("[article_tools] Paragraph extraction too short, trying fallback...")
        text_content = _extract_fallback(soup)

    text_content = " ".join(text_content.split())

    if len(text_content.strip()) < MIN_ARTICLE_LENGTH:
        print("CANNOT ANALYZE: LINK ATTACHMENT IS NOT AN ARTICLE")
        return "", "not_article"

    return text_content, "article"
