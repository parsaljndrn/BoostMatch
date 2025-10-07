import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
# ---------- Load Dataset 1 ----------
data1 = pd.read_csv("newsraw.csv", low_memory=False)
data1["title"] = data1["title"].fillna("")
data1["text"] = data1["text"].fillna("")
data1["content"] = data1["title"] + " " + data1["text"]

data1["label"] = data1["label"].astype(str).str.lower()
data1["label"] = data1["label"].map({"fake": 0, "real": 1})
data1 = data1.dropna(subset=["label"])
data1["label"] = data1["label"].astype(int)

# ---------- Load Dataset 2 ----------
data2 = pd.read_csv("finalen.csv", low_memory=False)
data2["title"] = data2["title"].fillna("")
data2["text"] = data2["text"].fillna("")
data2["merged"] = data2["title"] + " " + data2["text"]

data2["label"] = data2["label"].astype(str).str.lower()
data2["label"] = data2["label"].map({"0": 0, "1": 1})
data2 = data2.dropna(subset=["label"])
data2["label"] = data2["label"].astype(int)

# 4. Convert TITLE (string → numeric) using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)  # pick top 10000 words as features

# Split each dataset separately
X1 = vectorizer.fit_transform(data1["content"])
y1 = data1["label"]

X2 = vectorizer.fit_transform(data2["merged"])
y2 = data2["label"]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=42, stratify=y1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state=42, stratify=y2)

# 6. Train XGBoost
# ---------- Train Model 1 ----------
model1 = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    base_score=0.5,
    use_label_encoder=False
)
model1.fit(X1_train, y1_train)

# ---------- Train Model 2 ----------
model2 = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    base_score=0.5,
    use_label_encoder=False
)
model2.fit(X2_train, y2_train)

# 7. Predict
# ---------- Ensemble Prediction (majority vote) ----------
def ensemble_predict(X):
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    preds = np.array([pred1, pred2])
    final_pred = np.round(np.mean(preds, axis=0))  # majority voting
    return final_pred

# ---------- Test on Dataset 1 ----------
# 8.1 Accuracy
y_pred_ensemble1 = ensemble_predict(X1_test)
print("Ensemble Accuracy on Dataset 1:", accuracy_score(y1_test, y_pred_ensemble1))

# 8.2 Classification report
print("\nClassification Report:")
print(classification_report(y1_test, y_pred_ensemble1))

# 8.3 Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y1_test, y_pred_ensemble1))

# ---------- Test on Dataset 2 ----------
# 8.1 Accuracy
y_pred_ensemble2 = ensemble_predict(X2_test)
print("Ensemble Accuracy on Dataset 2:", accuracy_score(y2_test, y_pred_ensemble2))

# 8.2 Classification report
print("\nClassification Report:")
print(classification_report(y2_test, y_pred_ensemble2))

# 8.3 Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y2_test, y_pred_ensemble2))


y_true_combined = np.concatenate([y1_test, y2_test])
y_pred_combined = np.concatenate([y_pred_ensemble1, y_pred_ensemble2])

acc_combined = accuracy_score(y_true_combined, y_pred_combined)
print("Combined Accuracy:", acc_combined)


# 9 Save both model and vectorizer
#joblib.dump(model, r"D:\COLLEGE\CCS 3102\project_folder\xgb_modeltrained.pkl")
#joblib.dump(tfidf, r"D:\COLLEGE\CCS 3102\project_folder\tfidftrained..pkl")

#print("Model and vectorizer saved successfully!")