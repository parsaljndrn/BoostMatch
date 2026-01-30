from flask import Flask, render_template, request
import requests
import re
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)

PAGE_ACCESS_TOKEN = "EAAgaEvldRRoBQgkkqMKFJVAO6P355ew2Vq6EXw39l7fRE4BKeuzw8edorGHpORjsaQwbzFAmYzXrrZAbIJnloov9grboK7Hsb9F3AxA9domLYXAjNS2DzAU0G42yy44m7Q5pqZA8kmfBXhpxGhuG1ZApsCtFcVaVJAcawddnDVYm4z9pDF2TzyFBzk5TofwV0r55yAm6KHVMYZBNOueUY78KXl6E7BOwc11lqsN2a2aC0spYkAXCigMcOd35ob1wl0el4iehWsAmfrm45SmcJ1mjnmOZBQwG5Xv8ZD"

# 🔹 Extract post ID from multiple Facebook URL formats
def extract_post_id(fb_url):
    # Case 1: /posts/POST_ID
    match = re.search(r"/posts/(\d+)", fb_url)
    if match:
        return match.group(1)

    # Case 2: permalink.php?story_fbid=XXX&id=YYY
    parsed = urlparse(fb_url)
    if "permalink.php" in parsed.path:
        query = parse_qs(parsed.query)
        story_fbid = query.get("story_fbid", [None])[0]
        page_id = query.get("id", [None])[0]
        if story_fbid and page_id:
            return f"{page_id}_{story_fbid}"

    return None

# 🔹 Fetch caption from Facebook Graph API
def get_post_caption(post_id):
    url = f"https://graph.facebook.com/v19.0/{post_id}"
    params = {
        "fields": "message,permalink_url",
        "access_token": PAGE_ACCESS_TOKEN
    }
    return requests.get(url, params=params).json()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    caption = None
    post_id = None
    status = None
    error = None

    if request.method == "POST":
        fb_link = request.form["fb_link"]
        post_id = extract_post_id(fb_link)

        # 🔍 DEBUG (terminal only)
        print("EXTRACTED POST ID:", post_id)

        if not post_id:
            error = "Invalid Facebook post link format."
        else:
            data = get_post_caption(post_id)

            # ❌ Handle Facebook API errors clearly
            if "error" in data:
                error = data["error"]["message"]
            else:
                caption = data.get("message", "")

             

    return render_template(
        "index.html",
        result=result,
        caption=caption,
        post_id=post_id,
        status=status,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
