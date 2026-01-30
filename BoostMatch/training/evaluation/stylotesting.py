import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== LOAD EXTRACTED FEATURES =====================
FILE_PATH = "extracteddatastylo.csv"

df = pd.read_csv(FILE_PATH)

print("✅ Stylometry dataset loaded")
print("Shape:", df.shape)
print("Columns:", len(df.columns))
print("Label distribution:\n", df["Label"].value_counts(), "\n")


# ===================== TEST 1: MISSING VALUES =====================
print("🧪 TEST 1 — Missing Values Check")
missing = df.isnull().sum().sum()
print("Total missing values:", missing)

if missing == 0:
    print("✅ PASS: No missing values\n")
else:
    print("❌ FAIL: Missing values detected\n")


# ===================== TEST 2: FEATURE RANGE CHECK =====================
print("🧪 TEST 2 — Feature Range Sanity Check")

numeric_cols = df.select_dtypes(include=np.number).columns
stats = df[numeric_cols].describe()

print(stats.loc[["min", "max"]])

print("\n✅ Check manually:")
print("- Ratios should be between 0 and 1")
print("- Sentiment between -1 and +1")
print("- Counts should be >= 0\n")


# ===================== TEST 3: SINGLE ROW MANUAL INSPECTION =====================
print("🧪 TEST 3 — Single Row Inspection")

row = df.iloc[0]

print("Caption:")
print(row["Caption"])
print("\nStylometric values:")
print("Exclamation count:", row["caption_regex_exclamation_count"])
print("Uppercase ratio:", row["caption_regex_uppercase_ratio"])
print("VADER compound:", row["caption_vader_compound"])
print("Noun ratio:", row["caption_spacy_pos_noun_ratio"])
print("Sentence count:", row["caption_spacy_sentence_count"])
print("Lexical diversity:", row["caption_regex_lexical_diversity"])
print("\n✅ Does this match human intuition?\n")


# ===================== TEST 4: FAKE vs REAL FEATURE DISTRIBUTION =====================
print("🧪 TEST 4 — Fake vs Real Distribution")

real = df[df["Label"] == 1]["caption_vader_compound"]
fake = df[df["Label"] == 0]["caption_vader_compound"]

print("Average REAL sentiment:", real.mean())
print("Average FAKE sentiment:", fake.mean())

plt.hist(real, bins=20, alpha=0.6, label="REAL")
plt.hist(fake, bins=20, alpha=0.6, label="FAKE")
plt.legend()
plt.title("VADER Sentiment Distribution (Stylometry Test)")
plt.xlabel("Compound Sentiment")
plt.ylabel("Frequency")
plt.show()


# ===================== TEST 5: CORRELATION WITH LABEL =====================
print("🧪 TEST 5 — Feature Correlation with Label")

correlations = df[numeric_cols].corr()["Label"].sort_values(ascending=False)

print("\nTop features correlated with REAL:")
print(correlations.head(10))

print("\nTop features correlated with FAKE:")
print(correlations.tail(10))

print("\n✅ Stylometry testing complete!")
