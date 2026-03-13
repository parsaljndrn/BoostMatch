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
    "facebook.com", # Added to prevent circular loops
    "fb.me"
]

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
        raise ValueError("Please paste a Facebook link.")
    
    fb_url = normalize_facebook_url(fb_url)
    parsed = urlparse(fb_url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    
    if not any(d in domain for d in FACEBOOK_DOMAINS):
        raise ValueError("The link pasted is not a Facebook link.")
    
    # Check if it's a direct photo/video link which we might want to reject
    # but allow if it's part of a post structure
    
    match = re.search(r"/posts/(\d+)", path)
    if match:
        return match.group(1)
        
    if "permalink.php" in path:
        query = parse_qs(parsed.query)
        story_fbid = query.get("story_fbid", [None])[0]
        page_id = query.get("id", [None])[0]
        if story_fbid and page_id:
            return f"{page_id}_{story_fbid}"
            
    # Fallback for numeric IDs in URL
    match = re.search(r"(\d{8,})", fb_url)
    if match:
        return match.group(1)
        
    raise ValueError("Could not extract a Post ID from this Facebook link.")

# =========================================================
# TEXT UTILITIES
# =========================================================

def extract_urls_from_text(text: str):
    return re.findall(URL_PATTERN, text)

def clean_caption_text(text: str) -> str:
    if not text:
        return text

    # Remove emojis
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove URLs (We extract them before cleaning in the fetch function)
    text = re.sub(r'https?://\S+', '', text)

    # Remove trailing long alphanumeric sequences (fbclid)
    text = re.sub(r'(\s|^)[A-Za-z0-9]{20,}(\s[A-Za-z0-9]{20,})*$', '', text)

    # Collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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

# =========================================================
# FETCH FACEBOOK POST
# =========================================================

def fetch_facebook_post(post_id: str) -> dict:
    if not PAGE_ACCESS_TOKEN:
        raise ValueError("FB_PAGE_ACCESS_TOKEN is not configured.")
        
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{post_id}"
    params = {
        # 'link' is the attached link, 'message' is the caption
        "fields": "message,link,attachments{url,type,subattachments{url,type}}",
        "access_token": PAGE_ACCESS_TOKEN,
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise ValueError(f"Error connecting to Facebook API: {str(e)}")

    raw_caption = (data.get("message") or "").strip()
    
    # 1. Identify the best Article Link
    # Logic: Priority 1: Official attached link -> Priority 2: Links inside attachments -> Priority 3: Pasted links in text
    
    found_article_link = None
    
    # Check for direct attachment link
    attached_link = data.get("link")
    if attached_link and is_valid_article_link(attached_link):
        found_article_link = attached_link
    
    # If not found, check attachments (useful for carousels)
    if not found_article_link:
        potential_links = extract_attachment_urls(data)
        if potential_links:
            found_article_link = potential_links[0]
            
    # If still not found, search the caption for a pasted URL
    if not found_article_link:
        urls_in_text = extract_urls_from_text(raw_caption)
        for u in urls_in_text:
            if is_valid_article_link(u):
                found_article_link = u
                break

    # Final verification and cleanup
    if not found_article_link:
        raise ValueError("No valid external article link found in this post.")
        
    # Resolve any shortlinks (bit.ly, tinyurl, or fb redirectors)
    final_article_link = resolve_redirect(found_article_link)

    # 2. Process Caption
    if not raw_caption:
        # If there's no text, we just have a link. 
        # You can decide if you want to allow "empty" captions.
        pass 

    caption_en, original_language = normalize_language(raw_caption)
    clean_caption = clean_caption_text(caption_en)

    return {
        "caption": clean_caption,
        "original_language": original_language,
        "article_link": final_article_link,
    }