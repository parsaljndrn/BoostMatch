import pandas as pd
from pathlib import Path

# ==============================
# BUILD PATH TO CSV
# ==============================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH1 = BASE_DIR.parent / "datasets" / "extracteddatasbert.csv"
DATASET_PATH2 = BASE_DIR.parent / "datasets" / "extracteddatastylo.csv"
# ==============================
# LOAD CSV
# ==============================
# 1️⃣ Load datasets
df1 = pd.read_csv(DATASET_PATH1)
df2 = pd.read_csv(DATASET_PATH1)
print(f"Loaded CSV with shape: {df1.shape}")
print(f"Loaded CSV with shape: {df2.shape}")



# Combine them side-by-side (add columns)
combined = pd.concat([df1, df2], axis=1)

#Drop a specific column (example: 'text')
combined = combined.drop(columns=["Caption","Content","Label"])

combined = combined.rename(columns={"title": "Caption"})
combined = combined.rename(columns={"text": "Content"})
combined = combined.rename(columns={"label": "Label"})

combined.to_csv("final_features.csv", index=False)

print("✅ Combined dataset (side by side) saved as 'combined_side_by_side.csv'")
