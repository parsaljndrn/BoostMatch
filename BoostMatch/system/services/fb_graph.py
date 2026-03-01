import requests
import re
import os
from urllib.parse import urlparse, parse_qs

GRAPH_VERSION = "v24.0"
PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")

FACEBOOK_DOMAINS = {
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
    "fb.watch",
}

# Facebook links that are NOT posts
NON_POST_PATHS = [
    "profile.php",
    "/photo/",
    "/photos/",
    "/watch/",
    "/groups/",
    "/pages/",
    "/events/",
]


# =========================================================
# POST ID EXTRACTION
# =========================================================

def extract_post_id(fb_url: str) -> str:
    """
    Accepts ONLY Facebook POST links.
    Raises clear errors for invalid input.
    """

    if not fb_url:
        raise ValueError(
            "The link pasted is not a Facebook link. Please paste a Facebook link."
        )

    parsed = urlparse(fb_url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    # 1️⃣ Validate Facebook domain
    if not any(d in domain for d in FACEBOOK_DOMAINS):
        raise ValueError(
            "The link pasted is not a Facebook link. Please paste a Facebook link."
        )

    # 2️⃣ Reject known non-post Facebook links
    if any(p in path for p in NON_POST_PATHS):
        raise ValueError(
            "The link pasted is not a Facebook post link."
        )

    # 3️⃣ /posts/{id}
    match = re.search(r"/posts/(\d+)", path)
    if match:
        return match.group(1)

    # 4️⃣ permalink.php?id=PAGE_ID&story_fbid=POST_ID
    if "permalink.php" in path:
        query = parse_qs(parsed.query)
        story_fbid = query.get("story_fbid", [None])[0]
        page_id = query.get("id", [None])[0]
        if story_fbid and page_id:
            return f"{page_id}_{story_fbid}"

    # 5️⃣ reels / videos / fallback numeric ID
    match = re.search(r"(\d{8,})", fb_url)
    if match:
        return match.group(1)

    # 6️⃣ Facebook but not a post
    raise ValueError(
        "The link pasted is not a Facebook post link."
    )


# =========================================================
# GRAPH API FETCH
# =========================================================

def fetch_facebook_post(post_id: str) -> dict:
    """
    Fetch caption and attached link from Facebook Graph API.
    Enforces caption presence.
    """

    if not PAGE_ACCESS_TOKEN:
        raise ValueError("FB_PAGE_ACCESS_TOKEN is not configured.")

    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{post_id}"
    params = {
        "fields": "message,link",
        "access_token": PAGE_ACCESS_TOKEN,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

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

    # Caption is missing OR only a URL
    url_only_pattern = r"^https?://\S+$"

    if not caption or re.fullmatch(url_only_pattern, caption):
        raise ValueError(
        "The Facebook post link have no caption text available."
    )   

    return {
        "caption": caption,
        "article_link": data.get("link"),
    }