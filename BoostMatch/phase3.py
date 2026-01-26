import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from stylo_extracting import extract_all_features

# Load dataset (PART 2)
df = pd.read_csv("cleaneddata_part21.csv")

# Load fine-tuned SBERT
sbert = SentenceTransformer("fine_tuned_sbert2")

print("✅ Dataset & models loaded")

def get_sbert_similarity(title, full_text):
    emb1 = sbert.encode(title)
    emb2 = sbert.encode(full_text)
    return cosine_similarity([emb1], [emb2])[0][0]


rows = []

for _, row in df.iterrows():
    try:
        # SBERT
        sim = get_sbert_similarity(row["title"], row["full_text"])

        # Stylometry
        stylo = extract_all_features(row["full_text"])

        # Combine
        features = {
            **stylo,
            "cosine_similarity": sim,
            "label": row["label"]
        }

        rows.append(features)

    except Exception as e:
        print("⚠️ Skipped row due to error:", e)


features_df = pd.DataFrame(rows)

features_df.to_csv("final_features.csv", index=False)

print("✅ Feature extraction complete!")
print("Rows:", len(features_df))
print("Columns:", len(features_df.columns))

