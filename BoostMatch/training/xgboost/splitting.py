import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ==============================
# BUILD PATH TO CSV
# ==============================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "datasets" / "final_features.csv"
# ==============================
# LOAD CSV
# ==============================
# 1️⃣ Load datasets
data1 = pd.read_csv(DATASET_PATH)
print(f"Loaded CSV with shape: {data1.shape}")

# ==============================
# CONFIG
# ==============================
INPUT_FILE = data1
TRAIN_OUT = "X_train.csv"
TEST_OUT = "X_test.csv"
LABEL_COL = "Label"

# Known text columns that must NEVER be included
TEXT_COLUMNS = [
    "title",
    "text",
    "Caption",
    "Content",
    "full_text"
]

# ==============================
# LOAD DATA
# ==============================
print("📂 Loading merged dataset...")
df = pd.read_csv(INPUT_FILE)

print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# ==============================
# TRAIN / TEST SPLIT
# ==============================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[LABEL_COL]
)

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# ==============================
# SEPARATE LABEL
# ==============================
y_train = train_df[LABEL_COL]
y_test = test_df[LABEL_COL]

# ==============================
# DROP TEXT + LABEL
# ==============================
cols_to_drop = [LABEL_COL] + [c for c in TEXT_COLUMNS if c in df.columns]

X_train = train_df.drop(columns=cols_to_drop)
X_test = test_df.drop(columns=cols_to_drop)

# ==============================
# SAFETY CHECK (NUMERIC ONLY)
# ==============================
non_numeric_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

if non_numeric_cols:
    print("❌ ERROR: Non-numeric columns found:")
    print(non_numeric_cols)
    raise ValueError("Remove non-numeric columns before training!")

print("✅ All features are numeric")

# ==============================
# SAVE OUTPUT FILES (NO SCALING)
# ==============================
X_train_df = X_train.copy()
X_train_df[LABEL_COL] = y_train.values

X_test_df = X_test.copy()
X_test_df[LABEL_COL] = y_test.values

X_train_df.to_csv(TRAIN_OUT, index=False)
X_test_df.to_csv(TEST_OUT, index=False)

print("✅ Train/test split complete (no scaling applied).")
print(f"Saved: {TRAIN_OUT}")
print(f"Saved: {TEST_OUT}")
