import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================
# CONFIG
# ==============================
INPUT_FILE = "final_features.csv"
TRAIN_OUT = "X_train_scaled.csv"
TEST_OUT = "X_test_scaled.csv"
LABEL_COL = "Label"

# Known text columns that must NEVER be scaled
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
# SAFETY CHECK (CRITICAL)
# ==============================
non_numeric_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

if non_numeric_cols:
    print("❌ ERROR: Non-numeric columns found:")
    print(non_numeric_cols)
    raise ValueError("Remove non-numeric columns before scaling!")

print("✅ All features are numeric")

# ==============================
# SCALING
# ==============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# SAVE OUTPUT FILES
# ==============================
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_train_scaled_df[LABEL_COL] = y_train.values

X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_test_scaled_df[LABEL_COL] = y_test.values

X_train_scaled_df.to_csv(TRAIN_OUT, index=False)
X_test_scaled_df.to_csv(TEST_OUT, index=False)

print("✅ Scaling complete!")
print(f"Saved: {TRAIN_OUT}")
print(f"Saved: {TEST_OUT}")
