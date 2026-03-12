from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import requests, os, json
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
GPT_MODEL          = os.getenv("GPT_MODEL", "mistralai/mistral-7b-instruct")

DATA_FILE      = Path("data.json")   # persistent storage
CSV_FILE       = "quran_emotion_tagged.csv"
SCORE_CUTOFF   = 0.25                # min emotion_score to include ayah
SIM_THRESHOLD  = 0.20               # min combined similarity score
TOP_CANDIDATES = 15                  # fallback pool size

# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ── Load CSV ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)

if 'emotion_v2' in df.columns:
    df['_emotion_tag'] = df['emotion_v2']
    print(f"✅ Using emotion_v2 column")
else:
    df['_emotion_tag'] = df['emotion'].fillna('reflective')

if 'emotion_score' in df.columns:
    df['_eligible'] = df['emotion_score'] >= SCORE_CUTOFF
    excluded = (df['emotion_score'] < SCORE_CUTOFF).sum()
    print(f"✅ Excluded {excluded} low-confidence ayahs — pool: {df['_eligible'].sum()}")
else:
    df['_eligible'] = True

eligible_indices = df[df['_eligible']].index.tolist()

# ── Encode all ayahs once at startup ─────────────────────────────────────
print("⏳ Encoding ayahs...")
ayahs = df['ayah_en'].astype(str).tolist()
model = SentenceTransformer('all-MiniLM-L6-v2')
ayah_embeddings = model.encode(ayahs, convert_to_tensor=True, show_progress_bar=False)
print(f"✅ {len(ayahs)} ayahs encoded")

# ── Persistent storage (JSON file) ───────────────────────────────────────
def _load_data() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"history": [], "bookmarks": []}

def _save_data(data: dict):
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

_db = _load_data()
history_db: List[Dict]   = _db.get("history", [])
bookmarks_db: List[Dict] = _db.get("bookmarks", [])

def persist():
    _save_data({"history": history_db, "bookmarks": bookmarks_db})

# ── Pydantic models ───────────────────────────────────────────────────────
VALID_EMOTIONS = {
    "sad","anxious","hopeful","grateful","angry","stressed",
    "tired","peaceful","confused","happy","lonely","heartbroken",
    "content","reflective"
}

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

# ── OpenRouter helper ─────────────────────────────────────────────────────
def call_openrouter(system: str, user: str, max_tokens: int = 200) -> str:
    if not OPENROUTER_API_KEY:
        return ""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GPT_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ]
    }
    try:
        res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=15)
        res.raise_for_status()
        data = res.json()
        if "choices" not in data or not data["choices"]:
            return ""
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"OpenRouter error: {e}")
        return ""

# ── Emotion detection ─────────────────────────────────────────────────────
EMOTION_KEYWORDS: Dict = {
    "grateful":    [("grateful",3),("gratitude",3),("thankful",3),("thank",2),("blessed",3),
                    ("blessing",3),("alhamdulillah",3),("appreciate",2),("fortunate",2),
                    ("realize how much",2),("abundance",1)],
    "reflective":  [("reflective",3),("reflect",2),("thinking about",2),("pondering",2),
                    ("contemplat",2),("meaning",2),("purpose",2),("makes me think",2),
                    ("realize",1),("looking back",2),("quiet",1)],
    "anxious":     [("anxious",3),("anxiety",3),("worried",3),("worry",3),("nervous",2),
                    ("panic",3),("scared",2),("fear",2),("racing thoughts",3),("can't sleep",3),
                    ("overthinking",3),("what if",2),("exam",2),("test",2),("fail",2),
                    ("overwhelming",2),("dread",2)],
    "sad":         [("sad",3),("sadness",3),("cry",2),("crying",2),("tears",2),
                    ("depressed",3),("depression",3),("empty",2),("hopeless",2),
                    ("miserable",3),("grief",3),("miss",1),("lost someone",3),("heartache",2)],
    "lonely":      [("lonely",3),("loneliness",3),("alone",2),("isolated",3),("no one",2),
                    ("nobody",2),("invisible",3),("disconnected",3),("no friends",2),
                    ("left out",2),("abandoned",2)],
    "heartbroken": [("heartbroken",3),("heartbreak",3),("broken heart",3),("betrayed",3),
                    ("betrayal",3),("hurt me",2),("used me",2),("lied",2),("cheated",2),
                    ("can't forgive",2),("can't let go",2)],
    "angry":       [("angry",3),("anger",3),("furious",3),("rage",3),("frustrated",2),
                    ("frustration",2),("unfair",2),("not fair",2),("injustice",3),
                    ("mad",2),("annoyed",2),("hate",2)],
    "stressed":    [("stressed",3),("stress",3),("overwhelmed",3),("pressure",2),
                    ("burden",2),("too much",2),("can't handle",2),("deadline",2),
                    ("exam",2),("exams",2),("not prepared",3),("behind",2),
                    ("burning out",3),("burnout",3),("swamped",2)],
    "tired":       [("tired",3),("exhausted",3),("fatigue",2),("fatigued",2),
                    ("drained",3),("worn out",2),("can't continue",2),("no energy",2),
                    ("sleepy",1),("can't go on",2),("running on empty",3)],
    "hopeful":     [("hopeful",3),("hope",2),("optimistic",3),("believe",2),
                    ("will get better",3),("going to be okay",3),("trust allah",2),
                    ("inshallah",2),("dua",2),("praying",1)],
    "peaceful":    [("peaceful",3),("peace",2),("calm",2),("serene",3),("tranquil",3),
                    ("at ease",2),("relaxed",2),("settled",1),("still heart",2)],
    "happy":       [("happy",3),("happiness",3),("joy",2),("joyful",2),("ecstatic",3),
                    ("wonderful",2),("great day",2),("excited",2),("love life",2)],
    "content":     [("content",3),("contented",3),("satisfied",2),("acceptance",2),
                    ("accept",1),("okay with",2),("at peace with",2)],
    "confused":    [("confused",3),("confusion",3),("lost",2),("don't know",2),
                    ("unclear",2),("don't understand",2),("which way",2),
                    ("what to do",2),("no direction",2),("guidance",1)],
}

def keyword_detect_emotion(entry: str) -> str:
    text = entry.lower()
    scores: Dict[str, float] = {}
    for emotion, kw_list in EMOTION_KEYWORDS.items():
        score = sum(w for kw, w in kw_list if kw in text)
        if score > 0:
            scores[emotion] = score
    if not scores:
        return "unsure"
    best = max(scores, key=lambda e: scores[e])
    if scores[best] < 2:
        return "unsure"
    top4 = sorted(scores.items(), key=lambda x: -x[1])[:4]
    print(f"  Keyword scores: {top4} → {best}")
    return best

def llm_detect_emotion(entry: str) -> str:
    system = (
        "You are an emotion classifier for a Quranic journal app. "
        "Read the journal entry and output ONLY one word from this list:\n"
        "sad anxious hopeful grateful angry stressed tired peaceful confused "
        "happy lonely heartbroken content reflective\n"
        "ONE WORD only. No punctuation. No explanation."
    )
    raw = call_openrouter(system, f"Journal entry: {entry}", max_tokens=10)
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
    print("  Keyword unsure — trying LLM...")
    return llm_detect_emotion(entry)

# ── Spiritual query rewriting ─────────────────────────────────────────────
# Natural first-person language → lands near comfort/mercy verses
# NOT theological keywords → avoids accountability/battle verses
SPIRITUAL_QUERIES = {
    "anxious":     "I am overwhelmed with worry and fear please Allah calm my heart give me peace trust and relief from anxiety",
    "stressed":    "I am exhausted and overwhelmed carrying too many responsibilities I need Allah's help to find relief ease and rest",
    "sad":         "my heart is heavy with grief and sorrow I need Allah's comfort mercy and hope to carry on and feel better",
    "angry":       "I feel anger and injustice help me control myself find patience forgiveness and inner calm through Allah",
    "lonely":      "I feel completely alone and unseen please remind me that Allah is always with me close and near never abandoning me",
    "heartbroken": "my heart is broken and I feel betrayed I need Allah to heal me give me strength and restore my peace",
    "hopeful":     "I believe things will get better I trust in Allah's mercy and His perfect timing and beautiful plan for me",
    "grateful":    "I am filled with gratitude and want to thank Allah for all His blessings mercy and gifts in my life",
    "tired":       "I am deeply exhausted and drained I need rest and Allah's gentle mercy ease comfort and renewed strength",
    "confused":    "I feel lost and don't know which direction to take I need Allah's guidance clarity and light on my path",
    "reflective":  "I am quietly thinking about life meaning and purpose seeking wisdom closeness to Allah and deeper understanding",
    "peaceful":    "my heart is calm and at rest I feel contentment and gratitude in remembering Allah and His blessings",
    "happy":       "I feel joyful and happy alhamdulillah I want to thank Allah for this beautiful blessing and share this joy",
    "content":     "I feel satisfied and at peace with what Allah has given me accepting His will with gratitude and trust",
}

def rewrite_as_spiritual_query(entry: str, emotion: str) -> str:
    base = SPIRITUAL_QUERIES.get(emotion, "I am seeking comfort guidance and mercy from Allah in my struggles")
    # Blend with key words from actual entry for better context matching
    words = [w for w in entry.lower().split() if len(w) > 4][:8]
    context = " ".join(words)
    query = f"{base} {context}".strip()
    print(f"  Spiritual query ({emotion}): {query[:80]}...")
    return query

# ── Exclusion lists ───────────────────────────────────────────────────────
# These are checked against the English translation (lowercase)

GLOBAL_EXCLUSIONS = [
    # Prophet-specific address
    "invoke blessings on him", "bestow blessings on the prophet",
    "[o muhammad]", "o muhammad", "give good tidings [o",
    # Qibla/ritual geography
    "turn your face to the sacred", "turn your faces towards it", "sacred mosque",
    # Disease of heart (polemical)
    "minds are diseased", "hearts are diseased",
    # Military victory proclamations
    "imminent victory", "help from god and imminent",
    # Ritual rulings
    "in the name of allah", "stand up praying for nearly",
    "two-thirds of the night", "when the prayer is ended, disperse",
    "disperse in the land",
    # Burden-bearing (misread as comfort but is about Judgement Day)
    "bear another's burden", "no soul shall bear the burden of another",
    # Death timing
    "appointed time has come", "god will not grant a reprieve to a soul",
]

# Wrong for ANY personal emotional journal entry
PERSONAL_JOURNAL_BLOCKLIST = [
    # Judgement Day burden verses
    "no bearer of a burden", "bear the burden of another",
    "burden-bearer shall bear", "over-laden soul",
    "everyone must bear the consequence",
    # Death / reprieve verses
    "appointed time has come", "god will not grant a reprieve",
    # Threatening — replacement / defeat
    "he will bring in your place another people", "cannot defeat his purpose",
    "nor have you any friend or helper besides god",
    "you have none besides god to protect", "you have none besides god",
    "none besides god to protect or help",
    # Charity commands / financial rulings
    "niggardly", "whoever stints", "spend for god's cause",
    "let the man of means spend", "spend in accordance with his means",
    "resources are restricted",
    # Hollow rhetorical fragments
    "that is not difficult for god", "that is easy for god",
    "nothing is difficult for god",
    # Polemical challenges
    "shall i seek a lord other than",
    # War / violence
    "strike their necks", "cast terror", "smite",
    "prisoners of war", "taken captive",
    # Ritual/legal instructions
    "disperse in the land", "when the prayer is ended",
    "stand up praying for nearly", "two-thirds of the night",
    "praying at dawn for god",
    # Power declaration without personal comfort
    "to god belongs the kingdom of the heavens and of the earth. he gives life and death",
]

# Per-emotion additional exclusions
EMOTION_EXCLUSIONS: Dict[str, List[str]] = {
    "grateful": [
        "imminent victory", "help from god and", "give good tidings",
        "invoke blessings", "bestow blessings on the prophet",
    ],
    "anxious": [
        "gathered before", "day of judgement", "day of resurrection",
        "reckoning", "hellfire", "punishment", "hypocrites",
        "disbelievers", "fight", "kill", "slew", "misfortune befall",
        "wrath", "torment",
    ],
    "stressed": [
        "day of judgement", "hellfire", "fight", "kill",
        "hypocrites", "disbelievers", "punishment", "gathered before",
        "burden-bearer shall bear", "over-laden soul",
        "no bearer of a burden", "bear the burden of another",
        "niggardly", "stints", "you stand in need",
        "cannot defeat his purpose", "nor have you any friend or helper",
        "he will bring in your place another people",
        "everyone must bear the consequence",
        "shall i seek a lord other than", "spend for god's cause",
        "let the man of means spend", "spend in accordance with his means",
        "resources are restricted", "that is not difficult for god",
        "you have none besides god",
    ],
    "sad": [
        "hellfire", "fight them", "kill", "disbelievers",
        "punishment", "wrath", "torment",
    ],
    "lonely": [
        "hellfire", "fight", "kill", "disbelievers", "punishment",
        "gathered before", "day of judgement", "wrath",
    ],
    "heartbroken": [
        "hellfire", "fight", "kill", "disbelievers", "punishment", "wrath",
    ],
    "angry": [
        "hellfire", "fight them", "kill", "gathered before",
        "day of judgement", "smite", "strike",
    ],
    "tired": [
        "fighting for the cause", "others who may be fighting",
        "praying at dawn for god", "seeking god's bounty",
        "hellfire", "fight", "kill", "disbelievers", "punishment",
    ],
    "hopeful": [
        "hellfire", "punishment", "wrath", "hypocrites", "disbelievers",
    ],
    "reflective": [
        "hellfire", "fight", "kill", "punishment",
    ],
    "confused": [
        "hellfire", "fight", "kill", "punishment", "hypocrites",
    ],
}

def is_eligible_ayah(idx: int, emotion: str) -> bool:
    text = str(df.iloc[idx]['ayah_en']).lower()
    if any(phrase in text for phrase in GLOBAL_EXCLUSIONS):
        return False
    if any(phrase in text for phrase in PERSONAL_JOURNAL_BLOCKLIST):
        return False
    if any(phrase in text for phrase in EMOTION_EXCLUSIONS.get(emotion, [])):
        return False
    return True

def get_candidate_indices(emotion: str) -> List[int]:
    # Primary: use emotion_v2 tag
    if '_emotion_tag' in df.columns:
        mask = (df['_emotion_tag'] == emotion) & (df['_eligible'] == True)
        raw = df[mask].index.tolist()
        filtered = [i for i in raw if is_eligible_ayah(i, emotion)]
        print(f"  emotion_v2=='{emotion}': {len(raw)} raw → {len(filtered)} after exclusions")
        if len(filtered) >= 10:
            return filtered

    # Fallback: full eligible pool
    print(f"  ⚠️ Small pool — using full eligible fallback")
    return [i for i in eligible_indices if is_eligible_ayah(i, emotion)]

# ── MMR (Maximal Marginal Relevance) ─────────────────────────────────────
def mmr_select(
    query_embedding,
    candidate_embeddings,
    candidate_indices: List[int],
    top_n: int = 3,
    lambda_param: float = 0.85,
    already_shown: List[int] = []
) -> List[int]:
    import torch

    filtered = [
        (ci, ce) for ci, ce in zip(candidate_indices, candidate_embeddings)
        if ci not in already_shown
    ]
    if not filtered:
        filtered = list(zip(candidate_indices, candidate_embeddings))
    if not filtered:
        return []

    c_indices, c_embeds = zip(*filtered)
    c_tensor = torch.stack(list(c_embeds))
    query_sims = util.cos_sim(query_embedding, c_tensor)[0].tolist()

    selected, remaining = [], list(range(len(c_indices)))

    while len(selected) < top_n and remaining:
        if not selected:
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            sel_embeds = torch.stack([c_tensor[s] for s in selected])
            best, best_score = None, -float('inf')
            for i in remaining:
                redundancy = max(
                    util.cos_sim(c_tensor[i].unsqueeze(0), sel_embeds)[0].tolist()
                )
                score = lambda_param * query_sims[i] - (1 - lambda_param) * redundancy
                if score > best_score:
                    best_score, best = score, i
        selected.append(best)
        remaining.remove(best)

    return [c_indices[i] for i in selected]

# ── Comfort messages (fallback when LLM unavailable) ─────────────────────
COMFORT_FALLBACKS = {
    "grateful":    "MashaAllah — that quiet feeling of noticing your blessings is itself a gift from Allah. Keep returning to gratitude; it opens more doors than you know. 🤍",
    "reflective":  "These quiet moments of reflection are rare and precious. Allah loves the heart that pauses and thinks. Whatever you're working through, clarity will come. 🌙",
    "anxious":     "Allah knows the weight of what you're carrying right now. Take a breath — حَسْبُنَا اللَّهُ وَنِعْمَ الوَكِيل. He is enough, and He does not burden a soul beyond what it can bear. 🤍",
    "stressed":    "When everything feels too much, remember: Allah is closer to you than your own heartbeat. You don't have to solve it all today. Just take the next small step. 🌿",
    "sad":         "It's okay to feel sad. Even the Prophets wept. Allah does not waste your tears — every one of them is seen and counted. After hardship always comes ease. 💛",
    "lonely":      "Allah is with you even when no one else is — He hears you before you even speak. You are never truly alone. 🤍",
    "heartbroken": "A broken heart is still a heart that loved, and love is never wasted in Allah's eyes. He is the only one who can truly mend what is broken. 🌙",
    "angry":       "Your anger makes sense. أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ — seek refuge in Allah. He is Just; He sees everything. 🌿",
    "tired":       "Rest is not weakness — even our Prophet ﷺ rested. Be gentle with yourself today. Allah sees your effort even when you feel you have nothing left. 🤍",
    "confused":    "Not knowing which way to go is the perfect time to turn to Allah. Pray Istikhara, make dua, trust that He will make the right path clear. 🌙",
    "hopeful":     "That hope you feel? Hold onto it — it is a gift from Allah. He who gave you the hope will give you the thing you hope for, in His perfect time. 💛",
    "peaceful":    "Alhamdulillah for this peace. It is a sign that your heart is close to Allah. Protect this feeling through dhikr and gratitude. 🌿",
    "happy":       "Alhamdulillah for this happiness! Joy is a mercy from Him — savour it fully and let it fill your heart with gratitude. 💛",
    "content":     "Contentment — رِضا — is one of the highest stations of the heart. Thank Allah for it and let it be a foundation, not a ceiling. 🌙",
}

def generate_comfort_message(entry: str, emotion: str) -> str:
    system = (
        "You are a warm, gentle Islamic companion for a journaling app. "
        "Write 2-3 short, comforting lines like a kind friend would. "
        "Use simple language. Can include one short Arabic dua or phrase. "
        "Do NOT write poetry. Do NOT say 'I'm sorry'. Do NOT be preachy."
    )
    user = f"Emotion: {emotion}\nEntry: {entry}\n\nWrite a brief, warm comfort message."
    result = call_openrouter(system, user, max_tokens=150)
    return result if result else COMFORT_FALLBACKS.get(emotion, "Allah sees every step you take. One day at a time. 🤍")

# ── History helpers ───────────────────────────────────────────────────────
def get_shown_indices() -> List[int]:
    shown = []
    for item in history_db[-20:]:  # only last 20 entries to avoid over-exclusion
        for m in item.get("matches", []):
            idx = m.get("ayah_index", -1) if isinstance(m, dict) else getattr(m, "ayah_index", -1)
            shown.append(idx)
    return [i for i in shown if i >= 0]

# ── Main matching endpoint ────────────────────────────────────────────────
@app.post("/match-ayahs", response_model=MatchResponse)
def match_ayahs(request: EntryRequest):
    import torch

    print(f"\n{'='*50}")
    print(f"Entry: {request.entry[:80]}...")

    # 1. Detect emotion
    emotion = detect_emotion(request.entry)
    print(f"  Emotion: {emotion}")

    # 2. Build spiritual query (avoids pulling wrong verse types)
    spiritual_query = rewrite_as_spiritual_query(request.entry, emotion)

    # 3. Encode query + raw entry
    query_embedding = model.encode(spiritual_query, convert_to_tensor=True)
    entry_embedding = model.encode(request.entry, convert_to_tensor=True)

    # 4. Get emotion-filtered candidates
    candidate_indices = get_candidate_indices(emotion)
    if not candidate_indices:
        raise HTTPException(status_code=500, detail="No candidates found")

    candidate_embeddings = [ayah_embeddings[i] for i in candidate_indices]
    c_tensor = torch.stack(candidate_embeddings)

    # 5. Two-stage scoring:
    #    60% spiritual query similarity (correct theme)
    #    40% raw entry similarity (contextual relevance)
    #    This prevents "burden" in entry from pulling Judgement Day verses
    query_sims = util.cos_sim(query_embedding, c_tensor)[0].tolist()
    entry_sims = util.cos_sim(entry_embedding, c_tensor)[0].tolist()
    combined   = [0.6 * q + 0.4 * e for q, e in zip(query_sims, entry_sims)]

    # 6. Threshold filter
    threshold_pairs = [
        (ci, ce, s) for ci, ce, s
        in zip(candidate_indices, candidate_embeddings, combined)
        if s >= SIM_THRESHOLD
    ]
    print(f"  Threshold filter: {len(candidate_indices)} → {len(threshold_pairs)}")

    # Fallback: take top-N if threshold too strict
    if len(threshold_pairs) < request.top_n * 3:
        top_k = sorted(
            zip(candidate_indices, candidate_embeddings, combined),
            key=lambda x: -x[2]
        )
        threshold_pairs = list(top_k[:max(request.top_n * 5, TOP_CANDIDATES)])
        print(f"  Threshold relaxed → top {len(threshold_pairs)}")

    filtered_indices = [x[0] for x in threshold_pairs]
    filtered_embeds  = [x[1] for x in threshold_pairs]

    # 7. MMR selection (diversity + relevance)
    selected_indices = mmr_select(
        query_embedding,
        filtered_embeds,
        filtered_indices,
        top_n=request.top_n,
        lambda_param=0.85,
        already_shown=get_shown_indices()
    )

    # Debug: print selected ayahs
    for idx in selected_indices:
        row = df.iloc[int(idx)]
        score = combined[candidate_indices.index(idx)] if idx in candidate_indices else 0
        print(f"  ✦ [{score:.3f}] {row['surah_name_roman']} {row['ayah_no_surah']}: {str(row['ayah_en'])[:60]}...")

    # 8. Build response
    matches = []
    for idx in selected_indices:
        row = df.iloc[int(idx)]
        matches.append(AyahMatch(
            surah=row['surah_name_roman'],
            ayah_no=int(row['ayah_no_surah']),
            ayah=row['ayah_en'].strip(),
            ayah_ar=row['ayah_ar'].strip(),
            ayah_index=int(idx)
        ))

    comfort  = generate_comfort_message(request.entry, emotion)
    entry_id = len(history_db)

    history_db.append({
        "entry": request.entry,
        "matches": [m.dict() for m in matches],
        "comfort": comfort,
        "emotion_before": emotion,
        "emotion_after": None,
        "timestamp": datetime.utcnow().isoformat()
    })
    persist()  # save to disk

    return MatchResponse(
        matches=matches,
        comfort=comfort,
        emotion_before=emotion,
        entry_id=entry_id
    )

# ── Other endpoints ───────────────────────────────────────────────────────
@app.post("/update-emotion")
def update_emotion(data: EmotionUpdate):
    if 0 <= data.entry_id < len(history_db):
        history_db[data.entry_id]["emotion_after"] = data.emotion_after
        persist()
        return {"message": "Emotion updated"}
    raise HTTPException(status_code=404, detail="Invalid entry ID")

@app.get("/history", response_model=List[HistoryItem])
def get_history():
    return [
        HistoryItem(
            entry=item["entry"],
            emotion_before=item["emotion_before"],
            emotion_after=item.get("emotion_after"),
            comfort=item["comfort"],
            matches=[AyahMatch(**m) for m in item["matches"]],
            timestamp=item.get("timestamp", "")
        )
        for item in history_db
    ]

@app.delete("/delete-entry/{entry_id}")
def delete_entry(entry_id: int):
    if 0 <= entry_id < len(history_db):
        history_db.pop(entry_id)
        persist()
        return {"message": "Deleted"}
    raise HTTPException(status_code=404, detail="Invalid entry ID")

@app.post("/bookmark", response_model=BookmarkItem)
def add_bookmark(req: BookmarkRequest):
    if req.ayah_index < 0 or req.ayah_index >= len(df):
        raise HTTPException(status_code=400, detail="Invalid ayah index")
    for bm in bookmarks_db:
        if bm["ayah_index"] == req.ayah_index:
            return BookmarkItem(**bm)
    row  = df.iloc[req.ayah_index]
    item = {
        "ayah_index": req.ayah_index,
        "surah":      row['surah_name_roman'],
        "ayah_no":    int(row['ayah_no_surah']),
        "ayah":       row['ayah_en'].strip(),
        "ayah_ar":    row['ayah_ar'].strip(),
        "note":       req.note or "",
        "saved_at":   datetime.utcnow().isoformat()
    }
    bookmarks_db.append(item)
    persist()
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
    persist()
    return {"message": "Bookmark removed"}

POSITIVE_EMOTIONS = {"hopeful", "grateful", "peaceful", "happy", "content"}

@app.get("/pattern", response_model=WeeklyPattern)
def get_pattern():
    if not history_db:
        raise HTTPException(status_code=404, detail="No history yet")
    emotions_before = [item["emotion_before"] for item in history_db if item["emotion_before"] != "unsure"]
    emotion_freq    = dict(Counter(emotions_before))
    dominant        = max(emotion_freq, key=emotion_freq.get) if emotion_freq else "unsure"
    shift_count     = sum(
        1 for item in history_db
        if item.get("emotion_after") in POSITIVE_EMOTIONS
        and item.get("emotion_before") not in POSITIVE_EMOTIONS
    )
    surah_counts: Counter = Counter()
    for item in history_db:
        for m in item.get("matches", []):
            surah = m.get("surah") if isinstance(m, dict) else m.surah
            if surah:
                surah_counts[surah] += 1
    return WeeklyPattern(
        total_entries=len(history_db),
        emotion_frequency=emotion_freq,
        dominant_emotion=dominant,
        shift_to_positive=shift_count,
        most_shown_surah=surah_counts.most_common(1)[0][0] if surah_counts else "N/A"
    )

@app.post("/reflect-again/{entry_id}", response_model=MatchResponse)
def reflect_again(entry_id: int, top_n: int = 3):
    if entry_id < 0 or entry_id >= len(history_db):
        raise HTTPException(status_code=404, detail="Invalid entry ID")
    return match_ayahs(EntryRequest(entry=history_db[entry_id]["entry"], top_n=top_n))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ayahs_loaded": len(ayahs),
        "history_count": len(history_db),
        "bookmarks_count": len(bookmarks_db),
    }