import joblib
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .STYLO_EXTRACTING import extract_all_features

# clean relative paths (project-root based)
MODEL_PATH = Path("models/boostmatch/boostmatch.pkl")
SBERT_PATH = Path("models/sbert/fine_tuned_sbert")

boost_model = joblib.load(MODEL_PATH)
sbert_model = SentenceTransformer(str(SBERT_PATH))

feature_names = boost_model.get_booster().feature_names


def check_misleading(caption: str, article_text: str):

    caption = caption or ""
    article_text = article_text or ""

    if not caption.strip() or not article_text.strip():
        return 0.0, "INSUFFICIENT TEXT"

    emb1 = sbert_model.encode(caption, convert_to_numpy=True)
    emb2 = sbert_model.encode(article_text, convert_to_numpy=True)

    cos_sim = float(
        cosine_similarity(
            emb1.reshape(1, -1),
            emb2.reshape(1, -1)
        )[0][0]
    )

    try:
        stylometry = extract_all_features(caption)
    except:
        stylometry = {}

    features = {
        **stylometry,
        "cosine_similarity": cos_sim
    }

    df = pd.DataFrame([features])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_names].astype(float)

    pred = boost_model.predict(df)[0]
    prediction = "NOT MISLEADING" if pred == 1 else "NOT"

    return cos_sim, prediction