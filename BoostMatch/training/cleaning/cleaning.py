import pandas as pd
import re

# 1️⃣ Load datasets
data1 = pd.read_csv("cleaneddata_part2.csv")


# 2️⃣ Standardize label values (convert all to 0 and 1)
# Assume: real = 1, fake = 0
def normalize_label(label):
    if isinstance(label, str):
        label = label.lower().strip()
        if label == "real":
            return 1
        elif label == "fake":
            return 0
    return label  # keep numeric values as is

for df in [data1]:
    df["label"] = df["label"].apply(normalize_label)

# 3️⃣ Merge all datasets into one

# 4️⃣ Cleanse text data (basic cleaning)
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)       # remove links

        text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
        return text
    return ""

data1["title"] = data1["title"].apply(clean_text)
data1["text"] = data1["text"].apply(clean_text)


combined = data1[['title', 'text', 'label']]
# 5️⃣ Drop missing or invalid rows
combined = combined.dropna(subset=["title", "text", "label"])
combined = combined[combined['title'].str.strip() != '']             # remove empty titles
combined = combined[combined['text'].str.strip() != '']
combined = combined[combined['label'].astype(str).str.strip() != ''] # remove empty labels   
combined = combined[combined["label"].isin([0, 1])]


# 6️⃣ Merge title and text columns
combined["merged"] = combined["title"] + " " + combined["text"]
combined.drop_duplicates(subset=["merged"], inplace=True)
combined = combined.dropna(subset=["label", "title", "text"])
combined["label"] = combined["label"].astype(int)


# 7️⃣ Save combined dataset
combined.to_csv("cleaneddata_part21.csv", index=False)
print("✅ Combined and cleaned dataset saved as 'cleaneddataset2.csv'")

print("✅ Remaining rows after dropping NaN labels:", combined.shape)
print("✅ Any NaN labels left?", combined["label"].isna().sum())
print("✅ Label value counts:\n", combined["label"].value_counts())
print(f"\n✅ total rows: {len(combined)} final rows remaining")