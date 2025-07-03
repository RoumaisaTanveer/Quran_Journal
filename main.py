from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# 🔸 Create the FastAPI app first
app = FastAPI()

# 🔸 Add CORS middleware immediately after defining app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can limit to ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔸 Load data and model
df = pd.read_csv("quran_emotion_tagged.csv")
ayahs = df['ayah_en'].astype(str).tolist()
model = SentenceTransformer('all-MiniLM-L6-v2')
ayah_embeddings = model.encode(ayahs, convert_to_tensor=True)

# 🔸 Pydantic models
class EntryRequest(BaseModel):
    entry: str
    top_n: int = 3

class AyahMatch(BaseModel):
    surah: str
    ayah_no: int
    ayah: str

class MatchResponse(BaseModel):
    matches: List[AyahMatch]

# 🔸 Route
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
            ayah=row['ayah_en'].strip()
        ))

    return MatchResponse(matches=matches)
