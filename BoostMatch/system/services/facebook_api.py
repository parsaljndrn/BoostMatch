import requests
import re
from urllib.parse import urlparse, parse_qs

PAGE_ACCESS_TOKEN = "PAGE ACCESS TOKEN"


def extract_post_id(fb_url):
    """Extract Facebook post ID from multiple URL formats"""

    match = re.search(r"/posts/(\d+)", fb_url)
    if match:
        return match.group(1)

    parsed = urlparse(fb_url)
    if "permalink.php" in parsed.path:
        query = parse_qs(parsed.query)
        story_fbid = query.get("story_fbid", [None])[0]
        page_id = query.get("id", [None])[0]
        if story_fbid and page_id:
            return f"{page_id}_{story_fbid}"

    return None


def fetch_facebook_post(post_id):
    """Fetch Facebook post data only"""

    url = f"https://graph.facebook.com/v19.0/{post_id}"
    params = {
        "fields": "message,link",
        "access_token": PAGE_ACCESS_TOKEN
    }

    response = requests.get(url, params=params).json()

    if "error" in response:
        raise Exception(response["error"]["message"])

    return response
