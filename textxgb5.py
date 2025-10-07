import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load multiple datasets
df1 = pd.read_csv("newsraw.csv", low_memory=False)
df2 = pd.read_csv("finalen.csv", low_memory=False)

# Combine them
df = pd.concat([df1, df2], ignore_index=True)

# Clean and preprocess as usual
df.dropna(inplace=True)

# Convert labels (example: "real"=1, "fake"=0)
df["label"] = df["label"].map({"real" or "1": 1, "fake" or "0": 0})

# Merge title and content
df["content"] = df["title"] + " " + df["text"]

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["content"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = xgb.XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
