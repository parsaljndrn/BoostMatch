from flask import Flask, render_template, request
from services.fb_graph import extract_post_id, fetch_facebook_post
from services.article_tools import extract_article_for_nlp
from services.matcher import check_misleading
import validators

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        fb_url = request.form.get("fb_url", "").strip()

        # Basic URL sanity check only
        if not fb_url or not validators.url(fb_url):
            error = "Please enter a valid Facebook post URL."
            return render_template("index.html", result=result, error=error)

        try:
            post_id = extract_post_id(fb_url)
            if not post_id:
                raise ValueError("Unable to extract Facebook post ID.")

            post_data = fetch_facebook_post(post_id)
            caption = post_data.get("caption", "")
            article_link = post_data.get("article_link")

            # STRICT ARTICLE REQUIREMENT
            article_text = extract_article_for_nlp(article_link)

            similarity, prediction = check_misleading(caption, article_text)

            result = {
                "prediction": prediction,
                "similarity": round(similarity * 100, 2),
                "caption": caption,
                "article_link": article_link,
                "article_text": article_text[:4000]
            }

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)