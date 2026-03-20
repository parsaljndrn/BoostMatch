from flask import Flask, render_template, request, redirect, url_for, session
from services.fb_graph import extract_post_id, fetch_facebook_post
from services.article_tools import extract_article_headline
from services.matcher import check_misleading
from services.analysis_service import classify_post, is_facebook_url
import validators
import os

app = Flask(__name__)
# The secret key is essential to make session.pop() work correctly
app.secret_key = os.urandom(24) 

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        fb_url = request.form.get("fb_url", "").strip()
        caption = request.form.get("caption", "").strip()
        link = request.form.get("link", "").strip() or None

        # Validation: If URL is invalid, redirect to home (scrolls to top)
        print("POST received")
        print("fb_url:", fb_url)
        print("caption:", caption)
        print("link:", link)

        try:
            # ✅ Only validate if fb_url is provided
            if fb_url and not is_facebook_url(fb_url):
                raise ValueError("Invalid URL. Please paste a valid Facebook post link.")

            # CASE 1: Facebook URL provided
            if fb_url:
                if not validators.url(fb_url):
                    raise ValueError("Invalid URL format.")

                post_data = fetch_facebook_post(fb_url)

                result = classify_post(
                    caption=post_data.get("caption"),
                    article_link=post_data.get("article_link"),
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

            # CASE 2: Manual input
            elif caption:
                # Use analysis_service to handle missing article or video automatically
                result = classify_post(
                    caption=caption,
                    article_link=link or None,
                    video_url=None
                )

                session['result'] = {
                    "prediction": result['prediction'],
                    "similarity": round(result['cosine_similarity'] * 100, 2) if result['cosine_similarity'] else None,
                    "caption": result['caption_used'],
                    "article_link": link or None,
                    "article_text": result['article_used']
                }

                return redirect(url_for("index"))

            else:
                return redirect(url_for("index"))

        # ✅ ONE centralized error handler
        except Exception as e:
            return render_template("index.html", result=None, error=str(e))

    # GET logic: 
    # session.pop() removes the item after reading it. 
    # If the user hits 'Refresh', session.get('result') will be None.
    result = session.pop('result', None)
    # If no result exists (like after a refresh), the HTML won't render the 
    # result-card, and the browser will naturally stay at the top.
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)