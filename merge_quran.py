import pandas as pd

# Load both files
emotion_df = pd.read_csv("quran_emotion_tagged.csv")
translation_df = pd.read_csv("quran_with_surah_info.csv")

# Make sure the columns we need are there
translation_df = translation_df.rename(columns={
    "SurahID": "surah_no",
    "AyahID": "ayah_no_surah",
    "English": "ayah_en"
})

# Merge using Surah + Ayah number
merged = pd.merge(
    emotion_df.drop(columns=["ayah_en"]),  # remove old English
    translation_df[["surah_no", "ayah_no_surah", "ayah_en"]],
    on=["surah_no", "ayah_no_surah"],
    how="left"
)

# Save updated file
merged.to_csv("updated_file.csv", index=False)
print("✅ Replaced English based on Surah + Ayah match. Saved as updated_file.csv.")
