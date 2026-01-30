import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# CONFIG
# ===============================
DATA_PATH = "extracteddatasbert.csv"
MODEL_PATH = "fine_tuned_sbert2"
SAMPLE_SIZE = 200
RANDOM_STATE = 42

# ===============================
# LOAD DATA
# ===============================
print("📂 Loading extracted SBERT dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Shape: {df.shape}")
print("Columns:", list(df.columns))
print("\nLabel distribution:")
print(df["label"].value_counts())

# ===============================
# TEST 1 — NUMERICAL SANITY CHECK
# ===============================
print("\n🧪 TEST 1 — Numerical Sanity Check")
print(df["cosine_similarity"].describe())

missing = df["cosine_similarity"].isna().sum()
print("Missing values:", missing)

if missing == 0:
    print("✅ PASS: No missing values")
else:
    print("❌ FAIL: Missing values detected")

# ===============================
# TEST 2 — SEMANTIC SANITY CHECK
# ===============================
print("\n🧪 TEST 2 — Semantic Sanity Check (Human Intuition)")
sample_rows = df.sample(5, random_state=RANDOM_STATE)

for i, row in sample_rows.iterrows():
    print("\nCaption:", row["title"])
    print("Article snippet:", row["text"][:200], "...")
    print("Cosine similarity:", row["cosine_similarity"])

# ===============================
# TEST 3 — REAL vs FAKE DISTRIBUTION
# ===============================
print("\n🧪 TEST 3 — Label-Based Similarity Check")

real = df[df["label"] == 1]["cosine_similarity"]
fake = df[df["label"] == 0]["cosine_similarity"]

print("Average REAL similarity:", real.mean())
print("Average FAKE similarity:", fake.mean())

if real.mean() > fake.mean():
    print("✅ PASS: REAL similarity > FAKE similarity")
else:
    print("❌ WARNING: Overlap or reversed trend")

# ===============================
# TEST 4 — DISTRIBUTION PLOT
# ===============================
print("\n🧪 TEST 4 — Similarity Distribution Plot")

plt.figure(figsize=(8,5))
plt.hist(real, bins=30, alpha=0.6, label="REAL")
plt.hist(fake, bins=30, alpha=0.6, label="FAKE")
plt.legend()
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("SBERT Similarity Distribution")
plt.show()

# ===============================
# TEST 5 — COHEN'S D (EFFECT SIZE)
# ===============================
print("\n🧪 TEST 5 — Effect Size (Cohen's d)")

mean_real = real.mean()
mean_fake = fake.mean()

std_real = real.std()
std_fake = fake.std()

pooled_std = np.sqrt((std_real**2 + std_fake**2) / 2)
cohens_d = (mean_real - mean_fake) / pooled_std

print("Cohen's d:", cohens_d)

if cohens_d > 0.8:
    print("✅ STRONG separation")
else:
    print("⚠️ Weak or moderate separation")

# ===============================
# TEST 6 — CORRELATION WITH LABEL
# ===============================
print("\n🧪 TEST 6 — Correlation with Label")
print(df[["label", "cosine_similarity"]].corr())

# ===============================
# TEST 7 — BASELINE COMPARISON
# ===============================
print("\n🧪 TEST 7 — Baseline SBERT Comparison")

baseline_model = SentenceTransformer("all-MiniLM-L6-v2")
fine_tuned_model = SentenceTransformer(MODEL_PATH)

baseline_scores = []
finetuned_scores = []

sample_df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

for _, row in sample_df.iterrows():
    e1_base = baseline_model.encode(row["title"])
    e2_base = baseline_model.encode(row["text"])
    baseline_scores.append(cosine_similarity([e1_base], [e2_base])[0][0])

    finetuned_scores.append(row["cosine_similarity"])

print("Baseline mean similarity:", np.mean(baseline_scores))
print("Fine-tuned mean similarity:", np.mean(finetuned_scores))

if np.mean(finetuned_scores) > np.mean(baseline_scores):
    print("✅ PASS: Fine-tuned SBERT outperforms baseline")
else:
    print("⚠️ WARNING: No improvement over baseline")

# ===============================
# FINAL VERDICT
# ===============================
print("\n✅ SBERT EXTRACTION TESTING COMPLETE")
