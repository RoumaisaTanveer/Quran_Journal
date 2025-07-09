from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from schema import EmotionUpdate,AyahMatch,MatchResponse,EntryRequest,HistoryItem
from utils import  detect_emotion,generate_comfort_message

# ─── FASTAPI ────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── DATA + MODEL ──────────────────────────────────────────────────────
df = pd.read_csv("quran_emotion_tagged.csv")
ayahs = df['ayah_en'].astype(str).tolist()
model = SentenceTransformer('all-MiniLM-L6-v2')
ayah_embeddings = model.encode(ayahs, convert_to_tensor=True)

history_db = []  


    







@app.post("/match-ayahs", response_model=MatchResponse)
def match_ayahs(request: EntryRequest):
    entry_embedding = model.encode(request.entry, convert_to_tensor=True)
    similarities = util.cos_sim(entry_embedding, ayah_embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:request.top_n]

    matches = []
    for idx in top_indices:
        row = df.iloc[int(idx)]
        matches.append(AyahMatch(
            surah=row['surah_name_roman'],
            ayah_no=int(row['ayah_no_surah']),
            ayah=row['ayah_en'].strip(),
            ayah_ar=row['ayah_ar'].strip()
        ))

    emotion = detect_emotion(request.entry)
    comfort = generate_comfort_message(request.entry)

    entry_id = len(history_db)
    history_db.append({
        "entry": request.entry,
        "matches": matches,
        "comfort": comfort,
        "emotion_before": emotion,
        "emotion_after": None,
    })

    return MatchResponse(
        matches=matches,
        comfort=comfort,
        emotion_before=emotion,
        entry_id=entry_id
    )

@app.post("/update-emotion")
def update_emotion(data: EmotionUpdate):
    if 0 <= data.entry_id < len(history_db):
        history_db[data.entry_id]["emotion_after"] = data.emotion_after
        return {"message": "Emotion updated"}
    return {"error": "Invalid entry ID"}

@app.get("/history", response_model=List[HistoryItem])
def get_history():
    return history_db


@app.delete("/delete-entry/{entry_id}")
def delete_entry(entry_id: int):
    if 0 <= entry_id < len(history_db):
        history_db.pop(entry_id)
        return {"message": "Deleted"}
    return {"error": "Invalid entry ID"}
