import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("fine_tuned_sbert2")

# TEST 1 — Direct Caption-Article Test (BASIC SANITY CHECK)
# This test checks if the model assigns higher similarity
# to a real caption-article pair than to a fake one.
#This test ensures the model learned something basic.
real_caption = "Government approves new education reform bill"
real_article = """
The Department of Education confirmed today that a new education reform bill
was approved after deliberations in congress...
"""

fake_caption = "Government secretly bans all private schools"
fake_article = """
The Department of Education confirmed today that a new education reform bill
was approved after deliberations in congress...
"""

def sim(a, b):
    e1 = model.encode(a)
    e2 = model.encode(b)
    return cosine_similarity([e1], [e2])[0][0]

print("REAL ↔ REAL:", sim(real_caption, real_article))
print("FAKE ↔ REAL:", sim(fake_caption, real_article))


#TEST 2 — Dataset-Level Validation (STATISTICAL EVIDENCE)
#This test provides statistical evidence of model performance.
# It checks that real caption-article pairs have higher similarity
# than fake caption-article pairs on average.
#This test proves SBERT learned the right semantic structure.
df = pd.read_csv("cleaneddata_part1.csv")

scores_real = []
scores_fake = []

df_sample = df.sample(200, random_state=42)

for _, row in df_sample.iterrows():  # sample for speed
    score = sim(row["title"], row["full_text"])
    if row["label"] == 1:
        scores_real.append(score)
    else:
        scores_fake.append(score)

print("Average REAL similarity:", sum(scores_real) / len(scores_real))
print("Average FAKE similarity:", sum(scores_fake) / len(scores_fake))

#TEST 3 — Distribution Plot (Visual Confirmation)
#This test visualizes the similarity distributions
# for real vs. fake caption-article pairs.
#This is gold for your Results chapter.
plt.hist(scores_real, bins=20, alpha=0.6, label="REAL")
plt.hist(scores_fake, bins=20, alpha=0.6, label="FAKE")
plt.legend()
plt.title("SBERT Similarity Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.show()



# TEST 4 — Failure Case Test (EDGE CASE ANALYSIS)
# This test checks the model's behavior on an unrelated caption-article pair.
# This helps identify potential weaknesses.
# This test is crucial for understanding model limitations.

caption = "President signs new tax reform law"
unrelated_article = "The basketball finals ended with a dramatic overtime win."

print(sim(caption, unrelated_article))





# Calculate Cohen’s d for effect size
# This quantifies the difference between real and fake similarity scores.
# A higher Cohen’s d indicates better model discrimination.

cohens_d_values = []

for seed in range(10):  # 10 runs
    df_sample = df.sample(200, random_state=seed)

    scores_real = []
    scores_fake = []

    for _, row in df_sample.iterrows():
        score = sim(row["title"], row["full_text"])
        if row["label"] == 1:
            scores_real.append(score)
        else:
            scores_fake.append(score)

    mean_real = np.mean(scores_real)
    mean_fake = np.mean(scores_fake)
    std_real = np.std(scores_real)
    std_fake = np.std(scores_fake)

    pooled_std = np.sqrt((std_real**2 + std_fake**2) / 2)
    cohens_d = (mean_real - mean_fake) / pooled_std
    cohens_d_values.append(cohens_d)

print("Cohen’s d (mean ± std):",
      np.mean(cohens_d_values), "±", np.std(cohens_d_values))



print("✅ SBERT testing complete!")
