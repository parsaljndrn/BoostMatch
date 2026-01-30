from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm

# Load fine-tuned SBERT
model = SentenceTransformer("fine_tuned_sbert2")

# Load dataset
df = pd.read_csv("cleaneddata_part21.csv")

# Drop missing values
df = df.dropna(subset=["title", "text", "label"]).reset_index(drop=True)

# Convert to lists
titles = df["title"].tolist()
texts = df["text"].tolist()

# Encode in batches
title_embs = model.encode(titles, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
text_embs  = model.encode(texts, batch_size=32, convert_to_tensor=True, show_progress_bar=True)

# Compute cosine similarities (title ↔ text per row)
cosine_sims = util.cos_sim(title_embs, text_embs).diagonal().cpu().numpy()

# Attach to DataFrame
df["cosine_similarity"] = cosine_sims

# Save once
df.to_csv("extracteddatasbert2.csv", index=False)

print("✅ Extraction complete")
print("Total rows:", len(df))
print("Similarity range:", df["cosine_similarity"].min(), "→", df["cosine_similarity"].max())
