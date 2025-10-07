import pandas as pd

# 1. Load dataset
data = pd.read_csv("newstest3.csv", low_memory=False)

# Keep only rows where label is 'real' or 'fake'
data = data[data['label'].isin(['REAL', 'FAKE'])].copy()

# Drop rows where 'label' column is NaN
data = data.dropna(subset=['label'])

# Optionally convert to 1 and 0
data['label'] = data['label'].map({'REAL': 1, 'FAKE': 0})

print(data)
