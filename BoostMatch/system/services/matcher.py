import joblib
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .STYLO_EXTRACTING import extract_all_features

MODEL_PATH = Path("models/boostmatch/boostmatch.pkl")
SBERT_PATH = Path("models/sbert/fine_tuned_sbert")

boost_model = joblib.load(MODEL_PATH)
sbert_model = SentenceTransformer(str(SBERT_PATH))

feature_names: list[str] = boost_model.get_booster().feature_names


def check_misleading(caption: str, article_text: str) -> tuple[float, str]:
    caption = (caption or "").strip()
    article_text = (article_text or "").strip()

    if not caption or not article_text:
        return 0.0, "INSUFFICIENT TEXT"

    emb_caption = sbert_model.encode(caption, convert_to_numpy=True)
    emb_article = sbert_model.encode(article_text, convert_to_numpy=True)
    cos_sim = float(
        cosine_similarity(
            emb_caption.reshape(1, -1),
            emb_article.reshape(1, -1),
        )[0][0]
    )

    try:
        stylometry = extract_all_features(caption)
    except Exception as e:
        print(f"[matcher] Stylometry extraction failed: {e}")
        stylometry = {}

    features = {**stylometry, "cosine_similarity": cos_sim}
    df = pd.DataFrame([features])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names].astype(float)

    pred = boost_model.predict(df)[0]
    label = "MISLEADING" if int(pred) == 1 else "NOT MISLEADING"

    return cos_sim, label
