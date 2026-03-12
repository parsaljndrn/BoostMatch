import os
import re
import requests
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()

GRAPH_VERSION = "v24.0"


def get_access_token() -> str:
    token = os.getenv("FB_PAGE_ACCESS_TOKEN", "").strip()
    if not token:
        raise ValueError("FB_PAGE_ACCESS_TOKEN not set in .env file")
    return token


def extract_post_id(fb_url: str) -> tuple[str | None, str]:
    if not fb_url:
        return None, "post"

    fb_url = fb_url.strip()

    # /reel/123456 or /reels/123456
    m = re.search(r"/reels?/(\d+)", fb_url)
    if m:
        return m.group(1), "video"

    # /videos/123456
    m = re.search(r"/videos/(\d+)", fb_url)
    if m:
        return m.group(1), "video"

    # /watch/?v=123456
    m = re.search(r"[?&]v=(\d+)", fb_url)
    if m:
        return m.group(1), "video"

    # /posts/123456
    m = re.search(r"/posts/(\d+)", fb_url)
    if m:
        return m.group(1), "post"

    # /photo/123456 or /photo/?fbid=123456
    m = re.search(r"/photo(?:s)?/(\d+)", fb_url)
    if m:
        return m.group(1), "post"

    # /story.php?story_fbid=xxx&id=yyy or permalink.php?story_fbid=xxx&id=yyy
    parsed = urlparse(fb_url)
    if any(x in parsed.path for x in ["permalink.php", "story.php"]):
        qs = parse_qs(parsed.query)
        story_fbid = qs.get("story_fbid", [None])[0]
        page_id = qs.get("id", [None])[0]
        if story_fbid and page_id:
            return f"{page_id}_{story_fbid}", "post"
        if story_fbid:
            return story_fbid, "post"

    # ?fbid=xxx
    m = re.search(r"fbid=(\d+)", fb_url)
    if m:
        return m.group(1), "post"

    # /share/p/AbCdEf  or  /share/r/AbCdEf  (new FB short share URLs)
    # These don't have numeric IDs — need to resolve the redirect first
    m = re.search(r"/share/(?:p|r|v)/([A-Za-z0-9_-]+)", fb_url)
    if m:
        return _resolve_share_url(fb_url)

    # numeric_id_pfbid format: 61587599125854_pfbid0...
    m = re.search(r"(\d{10,})_pfbid", fb_url)
    if m:
        return m.group(1), "post"

    # ?id=123456
    m = re.search(r"[?&]id=(\d+)", fb_url)
    if m:
        return m.group(1), "post"

    # fallback: any 8+ digit number in the URL
    m = re.search(r"(\d{8,})", fb_url)
    if m:
        return m.group(1), "post"

    return None, "post"


def _resolve_share_url(fb_url: str) -> tuple[str | None, str]:
    try:
        resp = requests.head(fb_url, allow_redirects=True, timeout=10)
        final_url = resp.url
        print(f"[fb_graph] Resolved share URL to: {final_url}")
        if final_url and final_url != fb_url:
            return extract_post_id(final_url)
    except Exception as e:
        print(f"[fb_graph] Could not resolve share URL: {e}")
    return None, "post"


def _fetch_as_post(post_id: str, access_token: str):
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{post_id}"
    params = {
        "fields": "message,story,attachments{url,unshimmed_url,media,target,type}",
        "access_token": access_token,
    }
    resp = requests.get(url, params=params, timeout=15)
    print(f"[fb_graph] post status: {resp.status_code}")
    print(f"[fb_graph] post response: {resp.text[:800]}")
    return resp


def _fetch_as_video(video_id: str, access_token: str):
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{video_id}"
    params = {
        "fields": "description,title,message,embeddable,source,permalink_url",
        "access_token": access_token,
    }
    resp = requests.get(url, params=params, timeout=15)
    print(f"[fb_graph] video status: {resp.status_code}")
    print(f"[fb_graph] video response: {resp.text[:800]}")
    return resp


def fetch_facebook_post(post_id: str, id_type: str = "post") -> dict:
    if not post_id:
        raise ValueError("Invalid or missing Facebook post ID")

    access_token = get_access_token()

    try:
        if id_type == "video":
            resp = _fetch_as_video(post_id, access_token)
        else:
            resp = _fetch_as_post(post_id, access_token)

        if resp.status_code == 400 and id_type == "post":
            print("[fb_graph] Post fetch failed, retrying as video...")
            resp = _fetch_as_video(post_id, access_token)
            id_type = "video"

        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.HTTPError as e:
        try:
            err_data = resp.json()
            err_msg = err_data.get("error", {}).get("message", str(e))
            err_code = err_data.get("error", {}).get("code", "")
            raise Exception(f"Facebook API error ({err_code}): {err_msg}")
        except (ValueError, UnboundLocalError):
            raise Exception(f"Facebook API HTTP error: {e}")
    except requests.exceptions.Timeout:
        raise Exception("Facebook API timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Facebook API request failed: {e}")

    if "error" in data:
        err = data["error"]
        raise Exception(f"Facebook API error ({err.get('code')}): {err.get('message')}")

    caption = (
        data.get("message")
        or data.get("description")
        or data.get("title")
        or data.get("story")
        or ""
    ).strip()

    article_link = ""
    video_url = ""

    if id_type == "video":
        video_url = data.get("source", "")
        article_link = data.get("permalink_url", "")
    else:
        attachments = data.get("attachments", {}).get("data", [])
        if attachments:
            attach = attachments[0]
            attach_type = attach.get("type", "")

            article_link = (
                attach.get("unshimmed_url")
                or attach.get("url")
                or (attach.get("target") or {}).get("url", "")
            )

            media = attach.get("media") or {}
            video_url = media.get("source", "")

            if attach_type in ("video_inline", "video_autoplay") and not video_url:
                article_link = ""

    return {
        "caption": caption or "",
        "article_link": article_link or "",
        "video_url": video_url or "",
    }
