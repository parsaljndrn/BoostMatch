import re
from flask import Flask, render_template, request
import joblib

# Load model and vectorizer
model = joblib.load("xgb_modeltrained.pkl")
vectorizer = joblib.load("tfidftrained.pkl")

app = Flask(__name__)



@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probs = None
    sentence1, sentence2 = "", ""

    if request.method == "POST":
        sentence1 = request.form.get("sentence1", "")
        sentence2 = request.form.get("sentence2", "")
        # Merge sentences (important! just like training)
        merged_text = (sentence1 + " " + sentence2)
        X_input = vectorizer.transform([merged_text])
      
       # Predictions 
        pred = model.predict(X_input)[0]

       # Predict probabilities
        probs = model.predict_proba(X_input)[0]

        # Map result (adjust if needed: 0=fake, 1=real)
        label_map = {0: "Fake", 1: "Real"}
        result = label_map[int(pred)]
        probs = {
            "Fake": round(probs[0] * 100, 2),
            "Real": round(probs[1] * 100, 2)
        }
    return render_template("index.html", result=result, probs=probs, sentence1=sentence1, sentence2=sentence2)


if __name__ == "__main__":
    app.run(debug=True)
