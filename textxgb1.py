import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
data = pd.read_csv("newsraw.csv", low_memory=False)

# 2. Fill missing values in title with empty string
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

# 2.1 Merge title + content into one column
data["content"] = data["title"] + " " + data["text"]

# 3. Convert labels (real/fake → 1/0)
data["label"] = data["label"].astype(str).str.lower()
data["label"] = data["label"].map({"fake": 0, "real": 1})
data = data.dropna(subset=["label"])
data["label"] = data["label"].astype(int)


# 4. Convert TITLE (string → numeric) using TF-IDF
tfidf = TfidfVectorizer(max_features=10000)  # pick top 10000 words as features
X = tfidf.fit_transform(data["content"])     
y = data["label"]


# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# 6. Train XGBoost
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    base_score=0.5,
    use_label_encoder=False
)
model.fit(X_train, y_train)


# 7. Predict
y_pred = model.predict(X_test)

# 8.1 Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# 8.2 Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8.3 Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 9 Save both model and vectorizer
#joblib.dump(model, r"D:\COLLEGE\CCS 3102\project_folder\xgb_modelhyper.pkl")
#joblib.dump(tfidf, r"D:\COLLEGE\CCS 3102\project_folder\tfidfhyper..pkl")

#print("Model and vectorizer saved successfully!")