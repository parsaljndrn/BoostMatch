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

    cos_sim_cutoff = 0.5

    caption = caption or ""
    article_text = article_text or ""

    # If caption is empty, can't proceed
    if not caption.strip():
        raise ValueError("No caption provided. Please paste an input with a caption.")

    # Compute cosine similarity if article text exists
    if article_text.strip():
        emb1 = sbert_model.encode(caption, convert_to_numpy=True)
        emb2 = sbert_model.encode(article_text, convert_to_numpy=True)
        cos_sim = float(
            cosine_similarity(
                emb1.reshape(1, -1),
                emb2.reshape(1, -1)
            )[0][0]
        )
    else:
        cos_sim = None  # No article, so cosine similarity is unavailable
    
    if cos_sim is None:
        raise ValueError(
                "Please paste an input with an article link."
        )

    # Extract stylometric features 
    try:
        stylometry = extract_all_features(caption, prefix="caption_")
    except Exception as e:
        print("Stylometry error:", e)
        stylometry = {}

    # Combine features
    features = {**stylometry}
    if cos_sim is not None:
        features["cosine_similarity"] = cos_sim
    else:
        features["cosine_similarity"] = 0.0  # default if no article

    # Make sure all features exist
    df = pd.DataFrame([features])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names].astype(float)

    # Predict class and probability
    pred = boost_model.predict(df)[0]
    proba = boost_model.predict_proba(df)[0]  # [prob_class0, prob_class1]

    print("Cosine Similarity:", cos_sim)
    print("Predicted probabilities:", proba)
    print("Prediction from model (raw):", pred)
    print("Input features:", df)

    # Apply cosine similarity threshold override
    if cos_sim is not None and cos_sim < cos_sim_cutoff:
        pred = 0  # force MISLEADING
        print(f"Prediction overridden due to low cosine similarity < {cos_sim_cutoff}")

    prediction = "NOT MISLEADING" if pred == 1 else "MISLEADING"


    return cos_sim, prediction