import requests
import re
import os
from urllib.parse import urlparse, parse_qs
from langdetect import detect
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import emoji

load_dotenv()

GRAPH_VERSION = "v24.0"
PAGE_ACCESS_TOKEN = os.environ.get("FB_PAGE_ACCESS_TOKEN")

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
    "facebook.com", # Added to prevent circular loops
    "fb.me"
]
def clean_fb_caption(text: str) -> str:
    """
    Remove unwanted Facebook tracking tokens from the caption.
    Example token: icmlkETFLOWMyRHhlYmR2ZWJVQlEzc3J0...
    """
    if not text:
        return ""
    # Remove tokens that are unlikely natural language: sequences of 20+ letters/numbers/underscores
    text = re.sub(r"\s+[A-Za-z0-9_]{20,}", "", text)
    return text.strip()

# =========================================================
# REDIRECT RESOLVER
# =========================================================

def resolve_redirect(url: str) -> str:
    try:
        # Some sites block HEAD requests; using GET with stream=True is safer
        response = requests.get(
            url,
            allow_redirects=True,
            timeout=5,
            stream=True
        )
        return response.url
    except requests.RequestException:
        return url

# =========================================================
# FACEBOOK URL NORMALIZER
# =========================================================

def normalize_facebook_url(url: str) -> str:
    if not url:
        return url

    url = url.strip()

    # ✅ Fix missing https://
    if not url.startswith("http"):
        url = "https://" + url

    parsed = urlparse(url)

    # ✅ Handle sharer.php (IMPORTANT)
    if "sharer.php" in parsed.path:
        qs = parse_qs(parsed.query)
        return qs.get("u", [url])[0]

    # ✅ Resolve fb.watch and /share/
    if (
        "fb.watch" in parsed.netloc or
        "/share/" in parsed.path or
        "pfbid" in url   # 🔥 ADD THIS
    ):
        return resolve_redirect(url)

    return url

# =========================================================
# POST ID EXTRACTION
# =========================================================
def extract_post_id(fb_url: str):
    """
    Extracts a Facebook post/reel/video ID and returns a tuple: (id, type)
    type: 'post', 'reel', or 'video'
    """
    if not fb_url:
        raise ValueError("Please paste a Facebook link.")
    
    fb_url = normalize_facebook_url(fb_url)
    parsed = urlparse(fb_url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    
    # ✅ Step 1: Domain check
    if not any(d in domain for d in FACEBOOK_DOMAINS):
        raise ValueError("The link pasted is not a Facebook link.")

    # ✅ Step 2: Block NON-POST paths
    for invalid in NON_POST_PATHS:
        if invalid in path:
            raise ValueError(
                "Invalid Facebook URL. Please paste a Facebook POST link, not a profile or page."
            )

    # ✅ Step 3: Ensure it's a POST-type URL
    if not any(x in path for x in [
        "/posts/",
        "/videos/",
        "/reel/",
        "permalink.php",
        "/v/",
        "/share/" 
    ]):
        raise ValueError(
            "Invalid Facebook URL. Please paste a Facebook POST link."
        )

    # ==============================
    # ✅ ID EXTRACTION STARTS HERE
    # ==============================

    # Normal post
    match = re.search(r"/posts/(\d+)", path)
    if match:
        return match.group(1), "post"

    # Reels
    reel_match = re.search(r"/reel/([a-zA-Z0-9]+)", path)
    if reel_match:
        return reel_match.group(1), "reel"

    # Shared videos
    video_match = re.search(r"/v/([a-zA-Z0-9]+)", path)
    if video_match:
        return video_match.group(1), "video"
    
    share_match = re.search(r"/share/p/([a-zA-Z0-9]+)", path)
    if share_match:
        return share_match.group(1), "post"
    
    # Permalink.php
    if "permalink.php" in path:
        query = parse_qs(parsed.query)
        story_fbid = query.get("story_fbid", [None])[0]
        page_id = query.get("id", [None])[0]
        if story_fbid and page_id:
            return f"{page_id}_{story_fbid}", "post"

    # ⚠️ Safe fallback (now protected by validation above)
    match = re.search(r"(\d{8,})", fb_url)
    if match:
        return match.group(1), "post"

    raise ValueError("Could not extract a Post ID from this Facebook link.")

# =========================================================
# TEXT UTILITIES
# =========================================================

def extract_urls_from_text(text: str):
    return re.findall(URL_PATTERN, text)

def clean_caption_text(text: str) -> str:
    if not text:
        return ""

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove trailing long alphanumeric sequences (fbclid)
    text = re.sub(r'(\s|^)[A-Za-z0-9]{20,}(\s[A-Za-z0-9]{20,})*$', '', text)

    # Normalize whitespace (IMPORTANT)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def is_meaningful_text(text: str) -> bool:
    import re
    words = re.findall(r'\b[A-Za-z]{3,}\b', text)
    return len(words) >= 2
# =========================================================
# LANGUAGE DETECTION + TRANSLATION
# =========================================================

def normalize_language(text: str):
    if not text:
        return text, "unknown"
    
    # Strip URLs for cleaner detection
    text_for_detection = re.sub(URL_PATTERN, "", text).strip()
    if not text_for_detection:
        return text, "unknown"
        
    try:
        language = detect(text_for_detection)
    except:
        return text, "unknown"
        
    if language == "en":
        return text, "en"
        
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated, language
    except:
        return text, language

# =========================================================
# ATTACHMENT URL EXTRACTION
# =========================================================

def extract_attachment_urls(data: dict):
    """
    Specifically looks for external shared links (articles) 
    and avoids internal FB photo/video links.
    """
    urls = []
    attachments = data.get("attachments", {}).get("data", [])
    
    for item in attachments:
        # 'share' type usually indicates an external article
        url = item.get("url")
        if url and "facebook.com" not in urlparse(url).netloc:
            urls.append(url)
            
        # Check subattachments (like in a carousel)
        subattachments = item.get("subattachments", {}).get("data", [])
        for sub in subattachments:
            sub_url = sub.get("url")
            if sub_url and "facebook.com" not in urlparse(sub_url).netloc:
                urls.append(sub_url)
    return urls

# =========================================================
# ARTICLE DOMAIN VALIDATOR
# =========================================================

def is_valid_article_link(url: str) -> bool:
    if not url: return False
    domain = urlparse(url).netloc.lower()
    # Check if domain is blocked
    for blocked in BLOCKED_DOMAINS:
        if blocked in domain:
            return False
    return True

def extract_video_url(data):
    attachments = data.get("attachments", {}).get("data", [])

    for item in attachments:
        if item.get("media_type") == "video":
            media = item.get("media") or {}
            # ✅ PRIORITY: actual video file
            if media.get("source"):
                return media.get("source")
            return item.get("url")

        subattachments = item.get("subattachments", {}).get("data", [])
        for sub in subattachments:
            if sub.get("media_type") == "video":
                media = sub.get("media") or {}
                if media.get("source"):
                    return media.get("source")
                return sub.get("url")

    return None

# =========================================================
# FETCH FACEBOOK POST
# =========================================================

def fetch_facebook_post(fb_url: str) -> dict:
    if not PAGE_ACCESS_TOKEN:
        raise ValueError("FB_PAGE_ACCESS_TOKEN is not configured.")

    # 1️⃣ Extract ID and type
    post_id, id_type = extract_post_id(fb_url)

    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{post_id}"
    if id_type == "post":
        fields = "message,link,attachments{media_type,url,media,subattachments{media_type,url,media}}"
    else:  # reel or video
        fields = "description,source,thumbnails"

    params = {"fields": fields, "access_token": PAGE_ACCESS_TOKEN}

    # 2️⃣ Fetch data
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError:
        raise ValueError(
            "Unable to fetch Facebook post. The link may be invalid, private, or not a post."
        )
    except Exception:
        raise ValueError(
            "An error occurred while connecting to Facebook. Please try again."
        )

    # 3️⃣ Extract caption and video_url safely
    if id_type == "post":
        raw_caption = (data.get("message") or "").strip()
        video_url = extract_video_url(data)
    else:  # video / reel
        raw_caption = (data.get("description") or "").strip()
        video_url = data.get("source")

    # 4️⃣ Identify best article link
    found_article_link = None
    attached_link = data.get("link")
    if attached_link and is_valid_article_link(attached_link):
        found_article_link = attached_link

    if not found_article_link:
        potential_links = extract_attachment_urls(data)
        if potential_links:
            found_article_link = potential_links[0]

    if not found_article_link:
        urls_in_text = extract_urls_from_text(raw_caption)
        for u in urls_in_text:
            if is_valid_article_link(u):
                found_article_link = u
                break

    # 5️⃣ Decide comparison type
    caption_en, original_language = normalize_language(raw_caption)
    clean_caption = clean_caption_text(caption_en)
    
    final_article_link = resolve_redirect(found_article_link) if found_article_link else None
    comparison_type = None
    if final_article_link and caption_en.strip():
        comparison_type = "caption_article"
    elif video_url and caption_en.strip():
        comparison_type = "caption_video"
    elif video_url and not caption_en.strip() and final_article_link:
        comparison_type = "video_article"
    elif video_url and not caption_en.strip() and not final_article_link:
        comparison_type = "video_only"

    # 6️⃣ Normalize caption
    caption_en, original_language = normalize_language(raw_caption)
    clean_caption = clean_caption_text(caption_en)

    return {
        "caption": clean_caption,
        "original_language": original_language,
        "article_link": final_article_link,
        "video_url": video_url,
        "comparison_type": comparison_type
    }

    return {
        "caption": clean_caption,
        "original_language": original_language,
        "article_link": final_article_link,
        "video_url": video_url,
        "comparison_type": comparison_type
    }