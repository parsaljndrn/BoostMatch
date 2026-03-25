from flask import Flask, render_template, request, redirect, url_for, session
from services.fb_graph import extract_post_id, fetch_facebook_post, clean_fb_caption
from services.article_tools import extract_article_headline
from services.matcher import check_misleading
from services.analysis_service import classify_post, is_facebook_url, normalize_text
import validators
import os
import re

app = Flask(__name__)
# The secret key is essential to make session.pop() work correctly
app.secret_key = os.urandom(24) 

def is_facebook_post_url(url: str) -> bool:
    if not url:
        return False

    return any(x in url for x in [
        "/posts/",
        "/videos/",
        "fb.watch",
        "permalink.php",
        "/reel/",
        "/share/" 
    ])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        fb_url = request.form.get("fb_url", "").strip()
        caption = request.form.get("caption", "").strip()
        link = request.form.get("link", "").strip() or None
        # Manual input validation: empty or link-only
        if not caption and not fb_url:
            return render_template("index.html", result=None, error="Cannot analyze input: caption is empty. Please enter a caption.")

        url_pattern = r"^(https?://)?(www\.)?[-\w]+(\.\w{2,})+(/[\w\-./?%&=]*)?$"
        if caption and re.fullmatch(url_pattern, caption):
            return render_template("index.html", result=None, error="Cannot analyze input: caption appears to be only a link. Please enter actual text.")

        # Validation: If URL is invalid, redirect to home (scrolls to top)
        print("POST received")
        print("fb_url:", fb_url)
        print("caption:", caption)
        print("link:", link)

        try:
            # -------------------------------
            # CASE 1: Facebook input selected
            # -------------------------------
            if fb_url:
                # Validate FB URL
                if not is_facebook_url(fb_url):
                    raise ValueError("Invalid URL. Please paste a valid Facebook post link.")
                if not validators.url(fb_url):
                    raise ValueError("Invalid URL format.")

                # Fetch and classify FB post
                post_data = fetch_facebook_post(fb_url)

                # --- NORMALIZE FACEBOOK CAPTION AND ARTICLE ---
                raw_caption = post_data.get("caption") or ""
                cleaned_caption = clean_fb_caption(raw_caption)
                post_caption = normalize_text(cleaned_caption)
                post_article = normalize_text(post_data.get("article_link") or "")

                result = classify_post(
                    caption=post_caption,
                    article_link=post_article or None,
                    video_url=post_data.get("video_url")
                )

                session['result'] = {
                    "prediction": result['prediction'],
                    "similarity": round(result['cosine_similarity'] * 100, 2) if result['cosine_similarity'] else None,
                    "caption": result['caption_used'],
                    "article_link": post_data.get("article_link"),
                    "article_text": result['article_used']
                }
                return redirect(url_for("index"))

            # -------------------------------
            # CASE 2: Manual input selected
            # -------------------------------
            elif caption:
                # Manual input validation
                url_pattern = r"^(https?://)?(www\.)?[-\w]+(\.\w{2,})+(/[\w\-./?%&=]*)?$"
                if not caption:
                    raise ValueError("Cannot analyze input: caption is empty. Please enter a caption.")
                if re.fullmatch(url_pattern, caption):
                    raise ValueError("Cannot analyze input: caption appears to be only a link. Please enter actual text.")

                # --- TRANSLATE & NORMALIZE MANUAL CAPTION LIKE FB PATH ---
                from deep_translator import GoogleTranslator

                # 1. Translate to English if not already
                translated_caption = GoogleTranslator(source='auto', target='en').translate(caption)

                # 2. Normalize text exactly like FB captions
                normalized_caption = normalize_text(translated_caption)

                # 3. Normalize link too (optional)
                normalized_link = normalize_text(link) if link else None

                # 4. Classify
                result = classify_post(
                    caption=normalized_caption,
                    article_link=normalized_link or None,
                    video_url=None
                )

                session['result'] = {
                    "prediction": result['prediction'],
                    "similarity": round(result['cosine_similarity'] * 100, 2) if result['cosine_similarity'] else None,
                    "caption": result['caption_used'],
                    "article_link": normalized_link or None,
                    "article_text": result['article_used']
                }
                return redirect(url_for("index"))

            # -------------------------------
            # CASE 3: Neither FB nor manual input provided
            # -------------------------------
            else:
                raise ValueError("Please provide either a Facebook URL or a caption for manual input.")

        # ✅ ONE centralized error handler
        except Exception as e:
            return render_template("index.html", result=None, error=str(e))

    result = session.pop('result', None)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)