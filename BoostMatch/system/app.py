from flask import Flask, render_template, request

from services.fb_graph import extract_post_id, fetch_facebook_post
from services.article_tools import get_article_data
from services.matcher import check_misleading

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        fb_url = request.form.get("fb_url")

        try:
            post_id = extract_post_id(fb_url)
            if not post_id:
                raise ValueError("Invalid Facebook post URL")

            post_data = fetch_facebook_post(post_id)
            caption = post_data.get("caption", "")
            article_link = post_data.get("article_link")

            if not article_link:
                raise ValueError("No article link found in post")

            article_data = get_article_data(article_link)
            article_text = article_data["content"]

            similarity, prediction = check_misleading(caption, article_text)

            # Prepare result for template
            result = {
                "prediction": prediction,
                "similarity": round(similarity * 100, 2),
                "caption": caption,
                "article_link": article_link,
                "article_text": article_text[:4000]  # scrollable box
            }

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)