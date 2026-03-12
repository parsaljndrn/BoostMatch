import requests
import re
import os
from urllib.parse import urlparse, parse_qs
from langdetect import detect
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

load_dotenv()

GRAPH_VERSION = "v24.0"
PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")

FACEBOOK_DOMAINS = {
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
    "fb.watch",
}

NON_POST_PATHS = [
    "profile.php",
    "/photo/",
    "/photos/",
    "/watch/",
    "/groups/",
    "/pages/",
    "/events/",
]

URL_PATTERN = r"https?://[^\s]+"

BLOCKED_DOMAINS = [
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "twitter.com",
    "tiktok.com",
]


# =========================================================
# REDIRECT RESOLVER
# =========================================================

def resolve_redirect(url: str) -> str:

    try:

        response = requests.head(
            url,
            allow_redirects=True,
            timeout=5
        )

        return response.url

    except requests.RequestException:

        return url


# =========================================================
# FACEBOOK URL NORMALIZER
# =========================================================

def normalize_facebook_url(url: str) -> str:

    parsed = urlparse(url)
    path = parsed.path.lower()

    if "/share/" in path:
        return resolve_redirect(url)

    if parsed.netloc == "fb.watch":
        return resolve_redirect(url)

    return url


# =========================================================
# POST ID EXTRACTION
# =========================================================

def extract_post_id(fb_url: str) -> str:

    if not fb_url:
        raise ValueError(
            "The link pasted is not a Facebook link. Please paste a Facebook link."
        )

    fb_url = normalize_facebook_url(fb_url)

    parsed = urlparse(fb_url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    if not any(d in domain for d in FACEBOOK_DOMAINS):
        raise ValueError(
            "The link pasted is not a Facebook link. Please paste a Facebook link."
        )

    if any(p in path for p in NON_POST_PATHS):
        raise ValueError(
            "The link pasted is not a Facebook post link."
        )

    match = re.search(r"/posts/(\d+)", path)

    if match:
        return match.group(1)

    if "permalink.php" in path:

        query = parse_qs(parsed.query)

        story_fbid = query.get("story_fbid", [None])[0]
        page_id = query.get("id", [None])[0]

        if story_fbid and page_id:
            return f"{page_id}_{story_fbid}"

    match = re.search(r"(\d{8,})", fb_url)

    if match:
        return match.group(1)

    raise ValueError(
        "The link pasted is not a Facebook post link."
    )


# =========================================================
# TEXT UTILITIES
# =========================================================

def extract_urls_from_text(text: str):

    return re.findall(URL_PATTERN, text)


def extract_first_url(text: str):

    urls = extract_urls_from_text(text)

    if urls:
        return urls[0]

    return None


def remove_urls_from_text(text: str):

    text = re.sub(URL_PATTERN, "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# LANGUAGE DETECTION + TRANSLATION
# =========================================================

def normalize_language(text: str):

    if not text:
        return text, "unknown"

    text_for_detection = re.sub(URL_PATTERN, "", text).strip()

    if not text_for_detection:
        text_for_detection = text

    try:

        language = detect(text_for_detection)

    except:

        return text, "unknown"

    if language == "en":
        return text, "en"

    try:

        translated = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(text)

        return translated, language

    except:

        return text, language


# =========================================================
# ATTACHMENT URL EXTRACTION (NEW)
# =========================================================

def extract_attachment_urls(data: dict):

    urls = []

    attachments = data.get("attachments", {}).get("data", [])

    for item in attachments:

        target = item.get("target", {})
        url = target.get("url")

        if url:
            urls.append(url)

        # handle carousel / multi-attachment
        subattachments = item.get("subattachments", {}).get("data", [])

        for sub in subattachments:

            target = sub.get("target", {})
            url = target.get("url")

            if url:
                urls.append(url)

    return urls


# =========================================================
# ARTICLE DOMAIN VALIDATOR
# =========================================================

def is_valid_article_link(url: str) -> bool:

    domain = urlparse(url).netloc.lower()

    for blocked in BLOCKED_DOMAINS:
        if blocked in domain:
            return False

    return True


# =========================================================
# FETCH FACEBOOK POST
# =========================================================

def fetch_facebook_post(post_id: str) -> dict:

    if not PAGE_ACCESS_TOKEN:
        raise ValueError("FB_PAGE_ACCESS_TOKEN is not configured.")

    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{post_id}"

    params = {
        "fields": "message,link,attachments{target,url,subattachments}",
        "access_token": PAGE_ACCESS_TOKEN,
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.Timeout:
        raise ValueError(
            "Facebook API timeout. Please try again."
        )

    except requests.exceptions.RequestException:
        raise ValueError(
            "Failed to connect to Facebook Graph API."
        )

    if "error" in data:
        raise ValueError(data["error"]["message"])

    caption = (data.get("message") or "").strip()

    attached_link = data.get("link")

    attachment_links = extract_attachment_urls(data)

    # -----------------------------------------------------
    # FILTER: NO CAPTION
    # -----------------------------------------------------

    if not caption:
        raise ValueError(
            "The Facebook post link have no caption text available."
        )

    url_only_pattern = r"^https?://\S+$"

    if re.fullmatch(url_only_pattern, caption):
        raise ValueError(
            "The Facebook post link have no caption text available."
        )

    # -----------------------------------------------------
    # ARTICLE LINK DETECTION
    # -----------------------------------------------------

    article_link = attached_link

    if not article_link and attachment_links:
        article_link = attachment_links[0]

    if not article_link:
        article_link = extract_first_url(caption)

    if not article_link:
        raise ValueError(
            "The link attached is not article link, please paste a Facebook post link with article link attached to it."
        )

    article_link = resolve_redirect(article_link)

    if not is_valid_article_link(article_link):
        raise ValueError(
            "The link attached is not a valid article link."
        )

    caption_en, original_language = normalize_language(caption)

    clean_caption = remove_urls_from_text(caption_en)

    return {
        "caption": clean_caption,
        "original_language": original_language,
        "article_link": article_link,
    }