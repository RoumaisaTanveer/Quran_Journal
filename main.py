# # from fastapi import FastAPI
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# # from typing import List
# # import pandas as pd
# # from sentence_transformers import SentenceTransformer, util

# # # 🔸 Create the FastAPI app first
# # app = FastAPI()

# # # 🔸 Add CORS middleware immediately after defining app
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # You can limit to ["http://127.0.0.1:5500"]
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # 🔸 Load data and model
# # df = pd.read_csv("quran_emotion_tagged.csv")
# # ayahs = df['ayah_en'].astype(str).tolist()
# # model = SentenceTransformer('all-MiniLM-L6-v2')
# # ayah_embeddings = model.encode(ayahs, convert_to_tensor=True)

# # # 🔸 Pydantic models
# # class EntryRequest(BaseModel):
# #     entry: str
# #     top_n: int = 3

# # class AyahMatch(BaseModel):
# #     surah: str
# #     ayah_no: int
# #     ayah: str

# # class MatchResponse(BaseModel):
# #     matches: List[AyahMatch]

# # # 🔸 Route
# # @app.post("/match-ayahs", response_model=MatchResponse)
# # def match_ayahs(request: EntryRequest):
# #     entry_embedding = model.encode(request.entry, convert_to_tensor=True)
# #     similarities = util.cos_sim(entry_embedding, ayah_embeddings)[0]
# #     top_indices = similarities.argsort(descending=True)[:request.top_n]

# #     matches = []
# #     for idx in top_indices:
# #         row = df.iloc[int(idx)]
# #         matches.append(AyahMatch(
# #             surah=row['surah_name_roman'],
# #             ayah_no=int(row['ayah_no_surah']),
# #             ayah=row['ayah_en'].strip()
# #         ))

# #     return MatchResponse(matches=matches)


# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util
# import requests

# # ─── CONFIG ─────────────────────────────────────────────────────────────
# OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# GPT_MODEL = "mistralai/mistral-7b-instruct"  # Free, works even at -0.01 credit

# # ─── FASTAPI + CORS ─────────────────────────────────────────────────────
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ─── DATA AND EMBEDDINGS ───────────────────────────────────────────────
# df = pd.read_csv("quran_emotion_tagged.csv")
# ayahs = df['ayah_en'].astype(str).tolist()
# model = SentenceTransformer('all-MiniLM-L6-v2')
# ayah_embeddings = model.encode(ayahs, convert_to_tensor=True)

# # ─── REQUEST/RESPONSE MODELS ────────────────────────────────────────────
# class EntryRequest(BaseModel):
#     entry: str
#     top_n: int = 3

# class AyahMatch(BaseModel):
#     surah: str
#     ayah_no: int
#     ayah: str

# class MatchResponse(BaseModel):
#     matches: List[AyahMatch]
#     comfort: str

# # ─── OPENROUTER COMFORT MESSAGE ─────────────────────────────────────────
# def generate_comfort_message(entry: str) -> str:
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": GPT_MODEL,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": (
#                     "You're a soft, kind Islamic companion who replies to emotional journal entries in simple, warm words. "
#                     "Start with this exact header: 'Use short sentences. Avoid poetry or deep metaphors. "
#                     "No quotes, no 'I'm sorry to hear'. Just calming, real advice, like a good friend. Keep it natural and human."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": f"My journal entry: {entry}\n\nWrite a 2–3 line comforting message."
#             }
#         ]
#     }

#     try:
#         response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=10)
#         result = response.json()
#         return result["choices"][0]["message"]["content"].strip()
#     except Exception as e:
#         print("❌ OpenRouter request failed:", e)
#         return "You're doing better than you think. One step at a time is still progress. 🤍"


# # ─── ROUTE ──────────────────────────────────────────────────────────────
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

#     comfort = generate_comfort_message(request.entry)
#     return MatchResponse(matches=matches, comfort=comfort)




from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests
import os

# ─── CONFIG ─────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
GPT_MODEL = os.getenv("GPT_MODEL", "mistralai/mistral-7b-instruct")

# ─── FASTAPI ────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── DATA + MODEL ──────────────────────────────────────────────────────
df = pd.read_csv("quran_emotion_tagged.csv")
ayahs = df['ayah_en'].astype(str).tolist()
model = SentenceTransformer('all-MiniLM-L6-v2')
ayah_embeddings = model.encode(ayahs, convert_to_tensor=True)

history_db = []  # 🔸 In-memory storage

# ─── MODELS ─────────────────────────────────────────────────────────────
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
    emotion_before: str
    entry_id: int

class EmotionUpdate(BaseModel):
    entry_id: int
    emotion_after: str

class HistoryItem(BaseModel):
    entry: str
    emotion_before: str
    emotion_after: Optional[str]
    comfort: str
    matches: List[AyahMatch]
    

# ─── UTILITIES ──────────────────────────────────────────────────────────
def call_openrouter(system: str, user: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    }

    try:
        res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        # Log full response if missing keys
        if "choices" not in data or "message" not in data["choices"][0]:
            print("⚠️ Unexpected OpenRouter response:", data)
            return ""

        return data["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        print("❌ OpenRouter error:", e)
        return ""

def detect_emotion(entry: str) -> str:
    system = (
        "You are an emotion classifier. Your job is to detect the dominant emotion in a journal entry. "
        "Respond with exactly ONE WORD (lowercase), and choose only from this list: "
        "sad, anxious, hopeful, grateful, angry, stressed, tired, peaceful, confused, happy, lonely, heartbroken, content, reflective. "
        "Do NOT explain. Do NOT use any other words."
    )
    user = f"Emotion in journal entry:\n{entry}"

    raw = call_openrouter(system, user)
    print("🧪 Raw emotion response:", repr(raw))  # ✅ Log the raw response

    if not raw:
        return "unsure"

    first_word = raw.strip().lower().split()[0]
    valid_emotions = {
        "sad", "anxious", "hopeful", "grateful", "angry", "stressed", "tired",
        "peaceful", "confused", "happy", "lonely", "heartbroken", "content", "reflective"
    }

    if first_word in valid_emotions:
        print("✅ Detected emotion:", first_word)
        return first_word
    else:
        print("⚠️ Emotion not in valid list:", first_word)
        return "unsure"





def generate_comfort_message(entry: str) -> str:
    system = ("You're a soft, kind Islamic companion who replies to emotional journal entries in simple, warm words. "
              "Use short sentences. Avoid poetry or deep metaphors. No quotes, no 'I'm sorry to hear'. Just calming, real advice, like a good friend.")
    user = f"My journal entry: {entry}\n\nWrite a 2–3 line comforting message."
    return call_openrouter(system, user) or "You're doing better than you think. One step at a time is still progress. 🤍"

# ─── ROUTES ──────────────────────────────────────────────────────────────
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
