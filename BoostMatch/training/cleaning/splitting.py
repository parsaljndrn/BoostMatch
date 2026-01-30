import pandas as pd

df = pd.read_csv("dataset1.csv")

# Remove accidental index columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Remove rows with missing essential values
df = df.dropna(subset=["title", "text", "label"])

# Normalize whitespace (title)
df["title"] = df["title"].astype(str).str.strip()
df["title"] = df["title"].str.replace(r"\s+", " ", regex=True)

# Normalize whitespace (text)
df["text"] = df["text"].astype(str).str.strip()
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

# Create merged column (for modeling)
df["full_text"] = df["title"] + ". " + df["text"]

# Ensure label is integer
df["label"] = df["label"].astype(int)


# Shuffle the dataset (optional but recommended)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Compute half the length
half = len(df) // 2

# Split into two halves
data1 = df.iloc[:half]
data2 = df.iloc[half:]

# Save them as new CSV files
data1.to_csv("cleaneddata_part1.csv", index=False)
data2.to_csv("cleaneddata_part2.csv", index=False)

print("✅ Split complete!")
print(f"Part 1 rows: {len(data1)}")
print(f"Part 2 rows: {len(data2)}")


