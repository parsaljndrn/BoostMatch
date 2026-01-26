import pandas as pd
import random
from sentence_transformers import InputExample , SentenceTransformer , losses
from torch.utils.data import DataLoader

df = pd.read_csv("cleaneddata_part1.csv")

print(df.head())
print(df["label"].value_counts())

model = SentenceTransformer("all-MiniLM-L6-v2")

#Build Positive Pairs
train_examples = []

for _, row in df.iterrows():
    train_examples.append(
        InputExample(
            texts=[row["title"], row["full_text"]],
            label=1.0
        )
    )

# Build Negative Pairs
texts = df["full_text"].tolist()

for _, row in df.iterrows():
    negative_text = random.choice(texts)
    if negative_text != row["full_text"]:
        train_examples.append(
            InputExample(
                texts=[row["title"], negative_text],
                label=0.0
            )
        )

#data loader
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=16
)

#loss function
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=2,
    warmup_steps=100,
    show_progress_bar=True
)

# Save the model
model.save("fine_tuned_sbert")
print("✅ SBERT fine-tuning complete!")


## Testing the model
emb1 = model.encode("Government approves new law")
emb2 = model.encode("The senate passed the bill today")
emb3 = model.encode("Football match ends in draw")

from sklearn.metrics.pairwise import cosine_similarity

print(cosine_similarity([emb1], [emb2])[0][0])  # should be HIGH
print(cosine_similarity([emb1], [emb3])[0][0])  # should be LOW
