# facebook_fetcher.py
import requests
import re
from urllib.parse import urlparse, parse_qs
from newspaper import Article


PAGE_ACCESS_TOKEN = "EAAgaEvldRRoBQgkkqMKFJVAO6P355ew2Vq6EXw39l7fRE4BKeuzw8edorGHpORjsaQwbzFAmYzXrrZAbIJnloov9grboK7Hsb9F3AxA9domLYXAjNS2DzAU0G42yy44m7Q5pqZA8kmfBXhpxGhuG1ZApsCtFcVaVJAcawddnDVYm4z9pDF2TzyFBzk5TofwV0r55yAm6KHVMYZBNOueUY78KXl6E7BOwc11lqsN2a2aC0spYkAXCigMcOd35ob1wl0el4iehWsAmfrm45SmcJ1mjnmOZBQwG5Xv8ZD"


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
    """Fetch caption and embedded article link"""

    url = f"https://graph.facebook.com/v19.0/{post_id}"
    params = {
        "fields": "message,link",
        "access_token": PAGE_ACCESS_TOKEN
    }

    response = requests.get(url, params=params).json()

    if "error" in response:
        raise Exception(response["error"]["message"])

    return response.get("message", ""), response.get("link", "")


def fetch_article_text(url):
    """Download and extract article content"""

    article = Article(url)
    article.download()
    article.parse()
    return article.text
