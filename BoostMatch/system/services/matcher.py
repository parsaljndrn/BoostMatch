import joblib
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from STYLO_EXTRACTING import extract_all_features

# ==============================
# BUILD PATH TO CSV
# ==============================
BASE_DIR = Path(__file__).resolve().parent
model = BASE_DIR.parent / "models" / "boostmatch" / "boostmatch.pkl"
sberto = BASE_DIR.parent / "models" / "sbert" / "fine_tuned_sbert"

# Load the trained model
model = joblib.load(model)
sbert_model = SentenceTransformer(str(sberto))

# Get feature names from model
feature_names = model.get_booster().feature_names


def check_misleading(caption: str, article_text: str):
    emb1 = sbert_model.encode(caption)
    emb2 = sbert_model.encode(article_text)

    cos_sim = cosine_similarity([emb1], [emb2])[0][0]

    stylometry = extract_all_features(caption)

    features = {
        **stylometry,
        "cosine_similarity": cos_sim
    }

    df = pd.DataFrame([features])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_names]

    pred = model.predict(df)[0]
    prediction = "MISLEADING" if pred == 1 else "NOT MISLEADING"

    return cos_sim, prediction
