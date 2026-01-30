import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==============================
# CONFIG
# ==============================
TRAIN_FILE = "X_train.csv"
TEST_FILE = "X_test.csv"
LABEL_COL = "Label"
MODEL_OUT = "boostmatch.pkl"

# ==============================
# LOAD DATA
# ==============================
print("📂 Loading scaled datasets...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# ==============================
# SPLIT FEATURES / LABEL
# ==============================
X_train = train_df.drop(columns=[LABEL_COL])
y_train = train_df[LABEL_COL]

X_test = test_df.drop(columns=[LABEL_COL])
y_test = test_df[LABEL_COL]

print(f"Train shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")

# ==============================
# CREATE XGBOOST MODEL
# ==============================
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# ==============================
# TRAIN MODEL
# ==============================
print("\n🚀 Training XGBoost model...")
model.fit(X_train, y_train)

# ==============================
# PREDICTIONS
# ==============================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ==============================
# EVALUATION METRICS
# ==============================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n📊 Evaluation Results")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\n📄 Classification Report")
print(classification_report(y_test, y_pred))

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["FAKE (0)", "REAL (1)"],
    yticklabels=["FAKE (0)", "REAL (1)"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix — XGBoost")
plt.tight_layout()
plt.show()

# ==============================
# SAVE MODEL (PKL)
# ==============================
joblib.dump(model, MODEL_OUT)
print(f"\n💾 Model saved as {MODEL_OUT}")
print("✅ XGBoost training complete!")
