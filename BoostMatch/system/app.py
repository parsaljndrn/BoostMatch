from flask import Flask, render_template, request, redirect, url_for, session
from services.fb_graph import extract_post_id, fetch_facebook_post
from services.article_tools import extract_article_headline
from services.matcher import check_misleading
import validators
import os

app = Flask(__name__)
# The secret key is essential to make session.pop() work correctly
app.secret_key = os.urandom(24) 

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        fb_url = request.form.get("fb_url", "").strip()

        # Validation: If URL is invalid, redirect to home (scrolls to top)
        if not fb_url or not validators.url(fb_url):
            return redirect(url_for("index"))

        try:
            # 1. Extraction and API Calls
            post_id = extract_post_id(fb_url)
            post_data = fetch_facebook_post(post_id)

            caption = post_data.get("caption")
            article_link = post_data.get("article_link")
            headline = extract_article_headline(article_link)

            # 2. Matching Logic
            similarity, prediction = check_misleading(caption, headline)

            # 3. Store result in session
            session['result'] = {
                "prediction": prediction,
                "similarity": round(similarity * 100, 2),
                "caption": caption,
                "article_link": article_link,
                "headline": headline
            }
            
            # 4. PRG Pattern: Redirect to GET to prevent "Form Resubmission" 
            # and to allow the refresh-to-clear logic.
            return redirect(url_for("index"))

        except Exception as e:
            # In case of error, we render directly so the user stays at the 
            # input form to see the error message.
            return render_template("index.html", error=str(e))

    # GET logic: 
    # session.pop() removes the item after reading it. 
    # If the user hits 'Refresh', session.get('result') will be None.
    result = session.pop('result', None)
    
    # If no result exists (like after a refresh), the HTML won't render the 
    # result-card, and the browser will naturally stay at the top.
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)