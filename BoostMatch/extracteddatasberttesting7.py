from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Load datasets
df = pd.read_csv("extracteddatasbert.csv")

# Load models
baseline_model = SentenceTransformer("all-MiniLM-L6-v2")
finetuned_model = SentenceTransformer("fine_tuned_sbert2")

def compute_similarity(model, df, sample_size=300):
    df_sample = df.sample(sample_size, random_state=42)
    scores_real, scores_fake = [], []

    for _, row in df_sample.iterrows():
        emb1 = model.encode(row["title"])
        emb2 = model.encode(row["text"])
        score = util.cos_sim(emb1, emb2).item()

        if row["label"] == 1:
            scores_real.append(score)
        else:
            scores_fake.append(score)

    return scores_real, scores_fake

# Compute similarities
base_real, base_fake = compute_similarity(baseline_model, df)
fine_real, fine_fake = compute_similarity(finetuned_model, df)

def cohens_d(a, b):
    pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std

# Metrics
print("\n🧪 BASELINE SBERT")
print("Mean REAL:", np.mean(base_real))
print("Mean FAKE:", np.mean(base_fake))
print("Cohen's d:", cohens_d(base_real, base_fake))

print("\n🧪 FINETUNED SBERT")
print("Mean REAL:", np.mean(fine_real))
print("Mean FAKE:", np.mean(fine_fake))
print("Cohen's d:", cohens_d(fine_real, fine_fake))

# AUC comparison
y_true = [1]*len(base_real) + [0]*len(base_fake)

auc_base = roc_auc_score(y_true, base_real + base_fake)
auc_fine = roc_auc_score(y_true, fine_real + fine_fake)

print("\nAUC Baseline:", auc_base)
print("AUC Finetuned:", auc_fine)
