import requests
import re
import os
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()
GRAPH_VERSION = "v24.0"
PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")  # Make sure this is set in your env


def extract_post_id(fb_url: str) -> str | None:
    """Extract Facebook post ID from common URL formats"""
    if not fb_url:
        return None

    # /posts/123456789
    match = re.search(r"/posts/(\d+)", fb_url)
    if match:
        return match.group(1)

    # permalink.php?id=xxx&story_fbid=yyy
    parsed = urlparse(fb_url)
    if "permalink.php" in parsed.path:
        query = parse_qs(parsed.query)
        story_fbid = query.get("story_fbid", [None])[0]
        page_id = query.get("id", [None])[0]
        if story_fbid and page_id:
            return f"{page_id}_{story_fbid}"

    # fallback: any numeric ID in URL
    match = re.search(r"(\d{8,})", fb_url)
    if match:
        return match.group(1)

    return None


def fetch_facebook_post(post_id: str) -> dict:
    """Fetch Facebook post caption and link using Graph API"""
    if not post_id:
        raise ValueError("Invalid post ID")
    if not PAGE_ACCESS_TOKEN:
        raise ValueError("FB_PAGE_ACCESS_TOKEN not set in environment")

    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{post_id}"
    params = {"fields": "message,link", "access_token": PAGE_ACCESS_TOKEN}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        raise Exception("Facebook API timeout")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Facebook API request failed: {e}")

    if "error" in data:
        raise Exception(data["error"]["message"])

    return {
        "caption": data.get("message", ""),
        "article_link": data.get("link")
    }