import pandas as pd
import random
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

df = pd.read_csv("cleaneddata_part1.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

train_examples = []

texts = df["full_text"].tolist()

for _, row in df.iterrows():
    # POSITIVE PAIR (caption ↔ its article)
    sim_score = 0.9 if row["label"] == 1 else 0.2

    train_examples.append(
        InputExample(
            texts=[row["title"], row["full_text"]],
            label=sim_score
        )
    )

    # NEGATIVE PAIR (caption ↔ random article)
    negative_text = random.choice(texts)
    if negative_text != row["full_text"]:
        train_examples.append(
            InputExample(
                texts=[row["title"], negative_text],
                label=0.0
            )
        )

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=16
)

train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    show_progress_bar=True
)

model.save("fine_tuned_sbert1")
print("✅ SBERT fine-tuning complete!")
