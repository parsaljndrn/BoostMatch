from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    caption = request.form.get("caption")
    link = request.form.get("link")

    # TEMPORARY PLACEHOLDER
    # ML prediction will go here later
    result = "Prediction not yet implemented"

    return render_template(
        "result.html",
        caption=caption,
        link=link,
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)
