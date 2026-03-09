import requests
import re
import os
from urllib.parse import urlparse, parse_qs

GRAPH_VERSION = "v24.0"
PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")

# ---------------------------------------------------------
# FACEBOOK CONFIG
# ---------------------------------------------------------

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
    """
    Resolve shortened or redirect URLs (bit.ly, fb share links).
    """

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
    """
    Normalize Facebook URLs like share links.
    """

    parsed = urlparse(url)
    path = parsed.path.lower()

    # share links must be resolved
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

    # Validate Facebook domain
    if not any(d in domain for d in FACEBOOK_DOMAINS):
        raise ValueError(
            "The link pasted is not a Facebook link. Please paste a Facebook link."
        )

    # Reject known non-post paths
    if any(p in path for p in NON_POST_PATHS):
        raise ValueError(
            "The link pasted is not a Facebook post link."
        )

    # -----------------------------------------------------
    # /username/posts/{post_id}
    # -----------------------------------------------------

    match = re.search(r"/posts/(\d+)", path)

    if match:
        return match.group(1)

    # -----------------------------------------------------
    # permalink.php?id=PAGE&story_fbid=POST
    # -----------------------------------------------------

    if "permalink.php" in path:

        query = parse_qs(parsed.query)

        story_fbid = query.get("story_fbid", [None])[0]
        page_id = query.get("id", [None])[0]

        if story_fbid and page_id:
            return f"{page_id}_{story_fbid}"

    # -----------------------------------------------------
    # fallback numeric ID (videos, reels, etc.)
    # -----------------------------------------------------

    match = re.search(r"(\d{8,})", fb_url)

    if match:
        return match.group(1)

    raise ValueError(
        "The link pasted is not a Facebook post link."
    )


# =========================================================
# URL EXTRACTION UTILITIES
# =========================================================

def extract_urls_from_text(text: str) -> list:
    return re.findall(URL_PATTERN, text)


def extract_first_url(text: str) -> str | None:

    urls = extract_urls_from_text(text)

    if urls:
        return urls[0]

    return None


def remove_urls_from_text(text: str) -> str:

    cleaned = re.sub(URL_PATTERN, "", text)

    return cleaned.strip()


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
        "fields": "message,link",
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

    # -----------------------------------------------------
    # FILTER: NO CAPTION
    # -----------------------------------------------------

    if not caption:
        raise ValueError(
            "The Facebook post link have no caption text available."
        )

    # -----------------------------------------------------
    # FILTER: CAPTION ONLY URL
    # -----------------------------------------------------

    url_only_pattern = r"^https?://\S+$"

    if re.fullmatch(url_only_pattern, caption):
        raise ValueError(
            "The Facebook post link have no caption text available."
        )

    # -----------------------------------------------------
    # ARTICLE LINK DETECTION
    # -----------------------------------------------------

    article_link = attached_link

    if not article_link:
        article_link = extract_first_url(caption)

    if not article_link:
        raise ValueError(
            "The link attached is not article link, please paste a Facebook post link with article link attached to it."
        )

    # -----------------------------------------------------
    # EXPAND SHORT LINKS
    # -----------------------------------------------------

    article_link = resolve_redirect(article_link)

    # -----------------------------------------------------
    # VALIDATE ARTICLE DOMAIN
    # -----------------------------------------------------

    if not is_valid_article_link(article_link):
        raise ValueError(
            "The link attached is not a valid article link."
        )

    # -----------------------------------------------------
    # CLEAN CAPTION
    # -----------------------------------------------------

    clean_caption = remove_urls_from_text(caption)

    return {
        "caption": clean_caption,
        "article_link": article_link,
    }