from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import requests
import os
from collections import Counter
from datetime import datetime

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
GPT_MODEL = os.getenv("GPT_MODEL", "mistralai/mistral-7b-instruct")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

df = pd.read_csv("quran_emotion_tagged.csv")
if 'emotion_v2' in df.columns:
    print("Using emotion_v2")
    df['_emotion_tag'] = df['emotion_v2']
else:
    df['_emotion_tag'] = df['emotion'].fillna('')

if 'emotion_score' in df.columns:
    df['_eligible'] = df['emotion_score'] >= 0.25
    print(f"Excluding {(df['emotion_score'] < 0.25).sum()} low-confidence ayahs")
else:
    df['_eligible'] = True

eligible_indices = df[df['_eligible']].index.tolist()
print(f"Matching pool: {len(eligible_indices)} ayahs")

ayahs = df['ayah_en'].astype(str).tolist()
model = SentenceTransformer('all-MiniLM-L6-v2')
ayah_embeddings = model.encode(ayahs, convert_to_tensor=True)

history_db: List[Dict] = []
bookmarks_db: List[Dict] = []

VALID_EMOTIONS = {"sad","anxious","hopeful","grateful","angry","stressed","tired","peaceful","confused","happy","lonely","heartbroken","content","reflective"}

class EntryRequest(BaseModel):
    entry: str
    top_n: int = 3

class AyahMatch(BaseModel):
    surah: str
    ayah_no: int
    ayah: str
    ayah_ar: str
    ayah_index: int

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
    timestamp: str

class BookmarkRequest(BaseModel):
    ayah_index: int
    note: Optional[str] = ""

class BookmarkItem(BaseModel):
    ayah_index: int
    surah: str
    ayah_no: int
    ayah: str
    ayah_ar: str
    note: str
    saved_at: str

class WeeklyPattern(BaseModel):
    total_entries: int
    emotion_frequency: Dict[str, int]
    dominant_emotion: str
    shift_to_positive: int
    most_shown_surah: str

def call_openrouter(system: str, user: str, max_tokens: int = 200) -> str:
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GPT_MODEL, "max_tokens": max_tokens, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    try:
        res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=15)
        res.raise_for_status()
        data = res.json()
        if "choices" not in data or not data["choices"]:
            return ""
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("OpenRouter error:", e)
        return ""

def rewrite_as_spiritual_query(entry: str, emotion: str) -> str:
    themes = {
        "anxious":     "Allah is sufficient put trust in Him relieve worry calm the heart ease distress",
        "stressed":    "Allah does not burden a soul beyond capacity relief from hardship ease after difficulty",
        "sad":         "Allah is with the patient do not despair of Allah mercy relief after grief sorrow",
        "angry":       "restrain anger forgive people Allah loves those who control themselves patience",
        "lonely":      "Allah is closer than jugular vein He hears you He is with you always near",
        "heartbroken": "Allah heals hearts broken trust betrayal seek refuge in Allah mend the soul",
        "hopeful":     "do not lose hope in Allah mercy He answers dua victory comes after patience",
        "grateful":    "be grateful to Allah thank Him for blessings He increases those who are thankful",
        "tired":       "after hardship comes ease Allah gives rest to the weary seek strength in prayer",
        "confused":    "Allah guides to straight path seek His guidance light in darkness clarity wisdom",
        "reflective":  "reflect on creation signs of Allah wisdom in the heavens earth remember Allah",
        "peaceful":    "hearts find rest in remembrance of Allah tranquility contentment trust in Him",
        "happy":       "praise Allah for joy gratitude for blessings He is the source of all good",
        "content":     "Allah is enough trust His plan acceptance of decree satisfaction in His will",
    }
    query = themes.get(emotion, "seeking guidance comfort and mercy from Allah in hardship")
    print(f"Spiritual query for '{emotion}': {query}")
    return query

EMOTION_KEYWORDS: dict = {
    "grateful": [("grateful",3),("gratitude",3),("thankful",3),("thank",2),("blessed",3),("blessing",3),("alhamdulillah",3),("appreciate",2),("appreciation",2),("fortunate",2),("realize how much",2),("take for granted",2),("so much",1),("abundance",1)],
    "reflective": [("reflective",3),("reflect",2),("thinking about",2),("pondering",2),("contemplat",2),("life",1),("meaning",2),("purpose",2),("makes me think",2),("realize",1),("looking back",2),("quiet",1),("still",1),("calm",1),("at peace",2)],
    "anxious": [("anxious",3),("anxiety",3),("worried",3),("worry",3),("nervous",2),("panic",3),("scared",2),("fear",2),("racing",2),("can't sleep",3),("overthinking",3),("won't stop",2),("mind",1),("exam",2),("test",1),("fail",2),("overwhelming",2),("what if",2)],
    "sad": [("sad",3),("sadness",3),("cry",2),("crying",2),("tears",2),("depressed",3),("depression",3),("empty",2),("hopeless",2),("miserable",3),("down",1),("unhappy",2),("grief",3),("grieve",2),("miss",1),("lost someone",3)],
    "lonely": [("lonely",3),("loneliness",3),("alone",2),("isolated",3),("no one",2),("nobody",2),("invisible",3),("disconnected",3),("no friends",2),("left out",2),("abandoned",2)],
    "heartbroken": [("heartbroken",3),("heartbreak",3),("broken heart",3),("betrayed",3),("betrayal",3),("hurt me",2),("used me",2),("trusted",2),("lied",2),("cheated",2),("can't forgive",2),("burning",1),("can't let go",2)],
    "angry": [("angry",3),("anger",3),("furious",3),("rage",3),("frustrated",2),("frustration",2),("unfair",2),("not fair",2),("injustice",3),("mad",2),("annoyed",1),("hate",2)],
    "stressed": [("stressed",3),("stress",3),("overwhelmed",3),("pressure",2),("burden",2),("too much",2),("can't handle",2),("deadline",2),("exhausted from",2),("burning out",3),("burnout",3)],
    "tired": [("tired",3),("exhausted",3),("fatigue",2),("fatigued",2),("drained",3),("worn out",2),("can't continue",2),("no energy",2),("sleepy",1),("can't go on",2)],
    "hopeful": [("hopeful",3),("hope",2),("optimistic",3),("believe",2),("will get better",3),("going to be okay",3),("light",1),("trust allah",2),("inshallah",2),("dua",2),("praying",1)],
    "peaceful": [("peaceful",3),("peace",2),("calm",2),("serene",3),("tranquil",3),("at ease",2),("relaxed",2),("settled",1),("quiet inside",2),("still heart",2)],
    "happy": [("happy",3),("happiness",3),("joy",2),("joyful",2),("ecstatic",3),("elated",3),("wonderful",2),("great day",2),("excited",2),("love life",2)],
    "content": [("content",3),("contented",3),("satisfied",2),("enough",1),("acceptance",2),("accept",1),("okay with",2),("at peace with",2)],
    "confused": [("confused",3),("confusion",3),("lost",2),("don't know",2),("unclear",2),("don't understand",2),("which way",2),("what to do",2),("no direction",2),("guidance",1)],
}

def keyword_detect_emotion(entry: str) -> str:
    text = entry.lower()
    scores: dict[str, float] = {}
    for emotion, kw_list in EMOTION_KEYWORDS.items():
        score = sum(w for kw, w in kw_list if kw in text)
        if score > 0:
            scores[emotion] = score
    if not scores:
        return "unsure"
    best = max(scores, key=lambda e: scores[e])
    if scores[best] < 2:
        return "unsure"
    print(f"Keyword scores: {sorted(scores.items(), key=lambda x: -x[1])[:4]}")
    print(f"Keyword emotion: {best}")
    return best

def llm_detect_emotion(entry: str) -> str:
    system = "You are an emotion classifier. Output ONLY one word from: sad anxious hopeful grateful angry stressed tired peaceful confused happy lonely heartbroken content reflective\nONE WORD only. Example: grateful"
    raw = call_openrouter(system, f"Entry: {entry}", max_tokens=10)
    if not raw:
        return "reflective"
    cleaned = raw.lower().strip().strip("*_`\"'.,!?:")
    for word in cleaned.split():
        w = word.strip("*_`\"'.,!?:")
        if w in VALID_EMOTIONS:
            return w
    for emotion in VALID_EMOTIONS:
        if emotion in cleaned:
            return emotion
    return "reflective"

def detect_emotion(entry: str) -> str:
    result = keyword_detect_emotion(entry)
    if result != "unsure":
        return result
    return llm_detect_emotion(entry)

# ── Exclusions ──────────────────────────────────────────────────────────
GLOBAL_EXCLUSIONS = [
    "invoke blessings on him",
    "bestow blessings on the prophet",
    "turn your face to the sacred",
    "turn your faces towards it",
    "sacred mosque",
    "wrongdoers among them",
    "minds are diseased",
    "[o muhammad]",
    "o muhammad",
    "give good tidings [o",
    "imminent victory",
    "help from god and imminent",
    "in the name of allah",
    "stand up praying for nearly",
    "two-thirds of the night",
    "when the prayer is ended, disperse", 
    "disperse in the land",
]

EMOTION_EXCLUSIONS = {
    "grateful":    ["imminent victory","help from god and","give good tidings","invoke blessings","bestow blessings on the prophet"],
    "anxious":     ["gathered before","day of judgement","day of resurrection","reckoning","hellfire","punishment","hypocrites","disbelievers","fight","kill","slew","misfortune befall"],
    "stressed":    ["day of judgement","hellfire","fight","kill","hypocrites","disbelievers","punishment","gathered before"],
    "sad":         ["hellfire","fight them","kill","disbelievers","punishment"],
    "lonely":      ["hellfire","fight","kill","disbelievers","punishment","gathered before","day of judgement"],
    "heartbroken": ["hellfire","fight","kill","disbelievers","punishment"],
    "angry":       ["hellfire","fight them","kill","gathered before","day of judgement"],
    "tired": [
        "fighting for the cause",
        "others who may be fighting",
        "when the prayer is ended, disperse",
        "disperse in the land",
        "praying at dawn for god",
        "seeking god's bounty",
        "stand up praying for nearly",
        "two-thirds of the night",
        "hellfire", "fight", "kill", "disbelievers", "punishment",
    ],
}

def is_eligible_ayah(idx: int, emotion: str) -> bool:
    text = str(df.iloc[idx]['ayah_en']).lower()
    if any(phrase in text for phrase in GLOBAL_EXCLUSIONS):
        return False
    if any(phrase in text for phrase in EMOTION_EXCLUSIONS.get(emotion, [])):
        return False
    return True

def get_candidate_indices(emotion: str) -> List[int]:
    if '_emotion_tag' in df.columns:
        mask = (df['_emotion_tag'] == emotion) & (df['_eligible'] == True)
        indices = df[mask].index.tolist()
        print(f"emotion_v2 == '{emotion}' → {len(indices)} raw candidates")
        indices = [i for i in indices if is_eligible_ayah(i, emotion)]
        print(f"After exclusions: {len(indices)} candidates")
        if len(indices) >= 10:
            return indices
    print("Using full eligible pool fallback")
    return [i for i in eligible_indices if is_eligible_ayah(i, emotion)]

def mmr_select(query_embedding, candidate_embeddings, candidate_indices, top_n=3, lambda_param=0.85, already_shown=[]):
    import torch
    filtered = [(ci, ce) for ci, ce in zip(candidate_indices, candidate_embeddings) if ci not in already_shown]
    if not filtered:
        filtered = list(zip(candidate_indices, candidate_embeddings))
    if not filtered:
        return []
    c_indices, c_embeds = zip(*filtered)
    c_embeds_tensor = torch.stack(list(c_embeds))
    query_sims = util.cos_sim(query_embedding, c_embeds_tensor)[0].tolist()
    selected, remaining = [], list(range(len(c_indices)))
    while len(selected) < top_n and remaining:
        if not selected:
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            sel_embeds = torch.stack([c_embeds_tensor[s] for s in selected])
            best, best_score = None, -float('inf')
            for i in remaining:
                redundancy = max(util.cos_sim(c_embeds_tensor[i].unsqueeze(0), sel_embeds)[0].tolist())
                score = lambda_param * query_sims[i] - (1 - lambda_param) * redundancy
                if score > best_score:
                    best_score, best = score, i
        selected.append(best)
        remaining.remove(best)
    return [c_indices[i] for i in selected]

COMFORT_FALLBACKS = {
    "grateful":    "MashaAllah — that quiet feeling of noticing your blessings is itself a gift from Allah. Alhamdulillah for the eyes to see, the heart to feel. Keep returning to gratitude; it opens more doors than you know. 🤍",
    "reflective":  "These quiet moments of reflection are rare and precious. Allah loves the heart that pauses and thinks. Whatever you're working through, trust that clarity will come — He guides those who seek. 🌙",
    "anxious":     "Allah knows the weight of what you're carrying right now. Take a breath — حَسْبُنَا اللَّهُ وَنِعْمَ الوَكِيل. He is enough, and He does not burden a soul beyond what it can bear. You will get through this. 🤍",
    "stressed":    "When everything feels too much, remember: Allah is closer to you than your own heartbeat. You don't have to solve it all today. Just take the next small step and leave the rest to Him. 🌿",
    "sad":         "It's okay to feel sad. Even the Prophets wept. Allah does not waste your tears — every one of them is seen and counted. Cry, rest, and know that after hardship always comes ease. 💛",
    "lonely":      "Feeling alone is one of the hardest feelings. But know that Allah is with you even when no one else is — He hears you before you even speak. You are never truly alone. 🤍",
    "heartbroken": "A broken heart is still a heart that loved, and love is never wasted in the eyes of Allah. Let yourself heal slowly. He is the only one who can truly mend what is broken. 🌙",
    "angry":       "Your anger makes sense. Seek refuge in Allah — أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ. Allah is Just; He sees everything. You don't have to carry the burden of fixing it alone. 🌿",
    "tired":       "Rest is not weakness — even our Prophet ﷺ rested. Your body and soul have rights over you. Be gentle with yourself today. Allah sees your effort even when you feel you have nothing left. 🤍",
    "confused":    "Not knowing which way to go is the perfect time to turn to Allah. Pray Istikhara, make dua, and trust that He will make the right path clear. Confusion is just the moment before clarity. 🌙",
    "hopeful":     "That hope you feel? Hold onto it tightly — it is a gift from Allah. He who gave you the hope will give you the thing you hope for, in His perfect time. Keep making dua. 💛",
    "peaceful":    "Alhamdulillah for this peace. It is a sign that your heart is close to Allah. Protect this feeling through dhikr and gratitude — it is one of the most precious states to be in. 🌿",
    "happy":       "Alhamdulillah for this happiness! Express it, share it, and thank Allah for it. Joy is a mercy from Him — savour it fully and let it fill your heart with gratitude. 💛",
    "content":     "Contentment — رِضا — is one of the highest stations of the heart. You have found something rare. Thank Allah for it and let it be a foundation, not a ceiling. 🌙",
}

def generate_comfort_message(entry: str, emotion: str) -> str:
    system = "You're a soft, kind Islamic companion. Use simple, warm, short sentences. No poetry, no 'I'm sorry'. Just calm Islamic advice like a good friend."
    user = f"Feeling: {emotion}\nEntry: {entry}\n\nWrite 2-3 comforting lines."
    result = call_openrouter(system, user, max_tokens=150)
    return result if result else COMFORT_FALLBACKS.get(emotion, "Allah sees every step you take. One day at a time. 🤍")

def get_shown_indices() -> List[int]:
    shown = []
    for item in history_db:
        for m in item.get("matches", []):
            idx = m.get("ayah_index", -1) if isinstance(m, dict) else getattr(m, "ayah_index", -1)
            shown.append(idx)
    return [i for i in shown if i >= 0]

@app.post("/match-ayahs", response_model=MatchResponse)
def match_ayahs(request: EntryRequest):
    import torch

    emotion = detect_emotion(request.entry)
    spiritual_query = rewrite_as_spiritual_query(request.entry, emotion)
    query_embedding = model.encode(spiritual_query, convert_to_tensor=True)
    candidate_indices = get_candidate_indices(emotion)
    candidate_embeddings = [ayah_embeddings[i] for i in candidate_indices]

    raw_sims = util.cos_sim(query_embedding, torch.stack(candidate_embeddings))[0].tolist()
    SIMILARITY_THRESHOLD = 0.25
    threshold_filtered = [(ci, ce) for ci, ce, sim in zip(candidate_indices, candidate_embeddings, raw_sims) if sim >= SIMILARITY_THRESHOLD]
    print(f"Similarity filter: {len(candidate_indices)} → {len(threshold_filtered)}")

    if len(threshold_filtered) < request.top_n * 3:
        top_k = sorted(zip(candidate_indices, candidate_embeddings, raw_sims), key=lambda x: -x[2])
        threshold_filtered = [(ci, ce) for ci, ce, _ in top_k[:max(request.top_n * 5, 15)]]

    filtered_indices, filtered_embeds = zip(*threshold_filtered)
    filtered_indices, filtered_embeds = list(filtered_indices), list(filtered_embeds)

    selected_indices = mmr_select(query_embedding, filtered_embeds, filtered_indices, top_n=request.top_n, lambda_param=0.85, already_shown=get_shown_indices())

    matches = []
    for idx in selected_indices:
        row = df.iloc[int(idx)]
        matches.append(AyahMatch(surah=row['surah_name_roman'], ayah_no=int(row['ayah_no_surah']), ayah=row['ayah_en'].strip(), ayah_ar=row['ayah_ar'].strip(), ayah_index=int(idx)))

    comfort = generate_comfort_message(request.entry, emotion)
    entry_id = len(history_db)
    history_db.append({"entry": request.entry, "matches": [m.dict() for m in matches], "comfort": comfort, "emotion_before": emotion, "emotion_after": None, "timestamp": datetime.utcnow().isoformat()})

    return MatchResponse(matches=matches, comfort=comfort, emotion_before=emotion, entry_id=entry_id)

@app.post("/update-emotion")
def update_emotion(data: EmotionUpdate):
    if 0 <= data.entry_id < len(history_db):
        history_db[data.entry_id]["emotion_after"] = data.emotion_after
        return {"message": "Emotion updated"}
    raise HTTPException(status_code=404, detail="Invalid entry ID")

@app.get("/history", response_model=List[HistoryItem])
def get_history():
    return [HistoryItem(entry=item["entry"], emotion_before=item["emotion_before"], emotion_after=item.get("emotion_after"), comfort=item["comfort"], matches=[AyahMatch(**m) for m in item["matches"]], timestamp=item.get("timestamp", "")) for item in history_db]

@app.delete("/delete-entry/{entry_id}")
def delete_entry(entry_id: int):
    if 0 <= entry_id < len(history_db):
        history_db.pop(entry_id)
        return {"message": "Deleted"}
    raise HTTPException(status_code=404, detail="Invalid entry ID")

@app.post("/bookmark", response_model=BookmarkItem)
def add_bookmark(req: BookmarkRequest):
    if req.ayah_index < 0 or req.ayah_index >= len(df):
        raise HTTPException(status_code=400, detail="Invalid ayah index")
    for bm in bookmarks_db:
        if bm["ayah_index"] == req.ayah_index:
            return BookmarkItem(**bm)
    row = df.iloc[req.ayah_index]
    item = {"ayah_index": req.ayah_index, "surah": row['surah_name_roman'], "ayah_no": int(row['ayah_no_surah']), "ayah": row['ayah_en'].strip(), "ayah_ar": row['ayah_ar'].strip(), "note": req.note or "", "saved_at": datetime.utcnow().isoformat()}
    bookmarks_db.append(item)
    return BookmarkItem(**item)

@app.get("/bookmarks", response_model=List[BookmarkItem])
def get_bookmarks():
    return [BookmarkItem(**b) for b in bookmarks_db]

@app.delete("/bookmark/{ayah_index}")
def remove_bookmark(ayah_index: int):
    global bookmarks_db
    orig = len(bookmarks_db)
    bookmarks_db = [b for b in bookmarks_db if b["ayah_index"] != ayah_index]
    if len(bookmarks_db) == orig:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    return {"message": "Bookmark removed"}

POSITIVE_EMOTIONS = {"hopeful", "grateful", "peaceful", "happy", "content"}

@app.get("/pattern", response_model=WeeklyPattern)
def get_pattern():
    if not history_db:
        raise HTTPException(status_code=404, detail="No history yet")
    emotions_before = [item["emotion_before"] for item in history_db if item["emotion_before"] != "unsure"]
    emotion_freq = dict(Counter(emotions_before))
    dominant = max(emotion_freq, key=emotion_freq.get) if emotion_freq else "unsure"
    shift_count = sum(1 for item in history_db if item.get("emotion_after") in POSITIVE_EMOTIONS and item.get("emotion_before") not in POSITIVE_EMOTIONS)
    surah_counts: Counter = Counter()
    for item in history_db:
        for m in item.get("matches", []):
            surah = m.get("surah") if isinstance(m, dict) else m.surah
            if surah:
                surah_counts[surah] += 1
    return WeeklyPattern(total_entries=len(history_db), emotion_frequency=emotion_freq, dominant_emotion=dominant, shift_to_positive=shift_count, most_shown_surah=surah_counts.most_common(1)[0][0] if surah_counts else "N/A")

@app.post("/reflect-again/{entry_id}", response_model=MatchResponse)
def reflect_again(entry_id: int, top_n: int = 3):
    if entry_id < 0 or entry_id >= len(history_db):
        raise HTTPException(status_code=404, detail="Invalid entry ID")
    return match_ayahs(EntryRequest(entry=history_db[entry_id]["entry"], top_n=top_n))