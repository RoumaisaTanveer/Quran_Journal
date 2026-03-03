# save as retag.py and run: python retag.py
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

df = pd.read_csv("quran_emotion_tagged.csv")
ayahs = df['ayah_en'].fillna('').astype(str).tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')

# One descriptive sentence per emotion — written in Quranic language
# so they embed close to actual ayah translations
EMOTION_QUERIES = {
    "anxious":     "Allah is sufficient for us, do not worry, trust in Him, ease after difficulty, He calms the heart",
    "stressed":    "Allah does not burden a soul beyond capacity, relief from hardship, patience in trials",
    "sad":         "do not despair of Allah's mercy, He is with the patient, comfort after grief and sorrow",
    "angry":       "restrain anger, forgive others, Allah loves those who control themselves, patience",
    "lonely":      "Allah is closer than your jugular vein, He is always with you, never alone",
    "heartbroken": "Allah heals the broken heart, He knows your pain, seek refuge in Him after betrayal",
    "hopeful":     "do not lose hope in Allah's mercy, His promise is true, relief comes after hardship",
    "grateful":    "be thankful to Allah, He increases those who are grateful, praise Him for blessings",
    "tired":       "after hardship comes ease, seek strength in prayer, Allah gives rest to the weary",
    "confused":    "Allah guides to the straight path, seek His light in darkness, He gives wisdom and clarity",
    "reflective":  "reflect on the signs of Allah in creation, remember Him, contemplate His wisdom",
    "peaceful":    "hearts find rest in remembrance of Allah, tranquility and contentment through faith",
    "happy":       "praise Allah for joy and happiness, gratitude for His blessings and mercy",
    "content":     "trust Allah's plan, acceptance of His decree, He is enough, satisfaction in His will",
}

print("Encoding ayahs...")
ayah_embeddings = model.encode(ayahs, convert_to_tensor=True, show_progress_bar=True)

print("Encoding emotion queries...")
emotion_names = list(EMOTION_QUERIES.keys())
query_embeddings = model.encode(list(EMOTION_QUERIES.values()), convert_to_tensor=True)

print("Computing similarities...")
# For each ayah, find similarity to all 14 emotion queries
sims = util.cos_sim(ayah_embeddings, query_embeddings)  # shape: (6237, 14)

# Assign top emotion (highest similarity score)
top_emotion_indices = torch.argmax(sims, dim=1).tolist()
top_scores = torch.max(sims, dim=1).values.tolist()

df['emotion_v2'] = [emotion_names[i] for i in top_emotion_indices]
df['emotion_score'] = [round(s, 4) for s in top_scores]

print("\nNew emotion distribution:")
print(df['emotion_v2'].value_counts().to_string())
print("\nAverage similarity score:", round(df['emotion_score'].mean(), 4))
print("Low confidence ayahs (score < 0.25):", (df['emotion_score'] < 0.25).sum())

df.to_csv("quran_emotion_tagged.csv", index=False)
print("\nSaved! Column 'emotion_v2' added to CSV.")
