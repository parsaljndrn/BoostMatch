import pandas as pd

df1 = pd.read_csv("extracteddatasbert.csv")
df2 = pd.read_csv("extracteddatastylo.csv")

# Combine them side-by-side (add columns)
combined = pd.concat([df1, df2], axis=1)

#Drop a specific column (example: 'text')
combined = combined.drop(columns=["Caption","Content","Label"])

combined = combined.rename(columns={"title": "Caption"})
combined = combined.rename(columns={"text": "Content"})
combined = combined.rename(columns={"label": "Label"})

combined.to_csv("final_features.csv", index=False)

print("✅ Combined dataset (side by side) saved as 'combined_side_by_side.csv'")
