# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util

# # 🔸 Create the FastAPI app first
# app = FastAPI()

# # 🔸 Add CORS middleware immediately after defining app
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # You can limit to ["http://127.0.0.1:5500"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 🔸 Load data and model
# df = pd.read_csv("quran_emotion_tagged.csv")
# ayahs = df['ayah_en'].astype(str).tolist()
# model = SentenceTransformer('all-MiniLM-L6-v2')
# ayah_embeddings = model.encode(ayahs, convert_to_tensor=True)

# # 🔸 Pydantic models
# class EntryRequest(BaseModel):
#     entry: str
#     top_n: int = 3

# class AyahMatch(BaseModel):
#     surah: str
#     ayah_no: int
#     ayah: str

# class MatchResponse(BaseModel):
#     matches: List[AyahMatch]

# # 🔸 Route
# @app.post("/match-ayahs", response_model=MatchResponse)
# def match_ayahs(request: EntryRequest):
#     entry_embedding = model.encode(request.entry, convert_to_tensor=True)
#     similarities = util.cos_sim(entry_embedding, ayah_embeddings)[0]
#     top_indices = similarities.argsort(descending=True)[:request.top_n]

#     matches = []
#     for idx in top_indices:
#         row = df.iloc[int(idx)]
#         matches.append(AyahMatch(
#             surah=row['surah_name_roman'],
#             ayah_no=int(row['ayah_no_surah']),
#             ayah=row['ayah_en'].strip()
#         ))

#     return MatchResponse(matches=matches)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests

# ─── CONFIG ─────────────────────────────────────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GPT_MODEL = "mistralai/mistral-7b-instruct"  # Free, works even at -0.01 credit

# ─── FASTAPI + CORS ─────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── DATA AND EMBEDDINGS ───────────────────────────────────────────────
df = pd.read_csv("quran_emotion_tagged.csv")
ayahs = df['ayah_en'].astype(str).tolist()
model = SentenceTransformer('all-MiniLM-L6-v2')
ayah_embeddings = model.encode(ayahs, convert_to_tensor=True)

# ─── REQUEST/RESPONSE MODELS ────────────────────────────────────────────
class EntryRequest(BaseModel):
    entry: str
    top_n: int = 3

class AyahMatch(BaseModel):
    surah: str
    ayah_no: int
    ayah: str

class MatchResponse(BaseModel):
    matches: List[AyahMatch]
    comfort: str

# ─── OPENROUTER COMFORT MESSAGE ─────────────────────────────────────────
def generate_comfort_message(entry: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GPT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You're a soft, kind Islamic companion who replies to emotional journal entries in simple, warm words. "
                    "Start with this exact header: 'Use short sentences. Avoid poetry or deep metaphors. "
                    "No quotes, no 'I'm sorry to hear'. Just calming, real advice, like a good friend. Keep it natural and human."
                )
            },
            {
                "role": "user",
                "content": f"My journal entry: {entry}\n\nWrite a 2–3 line comforting message."
            }
        ]
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=10)
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("❌ OpenRouter request failed:", e)
        return "You're doing better than you think. One step at a time is still progress. 🤍"


# ─── ROUTE ──────────────────────────────────────────────────────────────
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

    comfort = generate_comfort_message(request.entry)
    return MatchResponse(matches=matches, comfort=comfort)
