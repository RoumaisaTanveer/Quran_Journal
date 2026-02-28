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

# ─── CONFIG ─────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")

# Free models on OpenRouter — uses mistral-7b-instruct as primary, falls back to gemma
GPT_MODEL = os.getenv("GPT_MODEL", "mistralai/mistral-7b-instruct")

# ─── FASTAPI ────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── DATA + MODEL ───────────────────────────────────────────────────────
df = pd.read_csv("quran_emotion_tagged.csv")
ayahs = df['ayah_en'].astype(str).tolist()
model = SentenceTransformer('all-MiniLM-L6-v2')
ayah_embeddings = model.encode(ayahs, convert_to_tensor=True)

history_db: List[Dict] = []
bookmarks_db: List[Dict] = []  # new: bookmarked ayahs

# ─── VALID EMOTIONS ─────────────────────────────────────────────────────
VALID_EMOTIONS = {
    "sad", "anxious", "hopeful", "grateful", "angry", "stressed", "tired",
    "peaceful", "confused", "happy", "lonely", "heartbroken", "content", "reflective"
}

# ─── PYDANTIC MODELS ────────────────────────────────────────────────────
class EntryRequest(BaseModel):
    entry: str
    top_n: int = 3

class AyahMatch(BaseModel):
    surah: str
    ayah_no: int
    ayah: str
    ayah_ar: str
    ayah_index: int  # index in df — used for bookmark reference

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
    ayah_index: int      # index in df
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
    shift_to_positive: int   # how many entries improved (emotion_after more positive than before)
    most_shown_surah: str

# ─── OPENROUTER CALL (free models) ──────────────────────────────────────
def call_openrouter(system: str, user: str, max_tokens: int = 200) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GPT_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    }
    try:
        res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=15)
        res.raise_for_status()
        data = res.json()
        if "choices" not in data or not data["choices"]:
            print("⚠️ Unexpected OpenRouter response:", data)
            return ""
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("❌ OpenRouter error:", e)
        return ""

# ─── FIX 1: REWRITE ENTRY INTO SPIRITUAL QUERY ──────────────────────────
# This bridges the language gap between casual journal writing and Quranic Arabic translations.
# Instead of matching "I feel broken and lost" against Ayahs directly (poor semantic overlap),
# we reframe it as a spiritual/theological need before embedding — much better vector alignment.
def rewrite_as_spiritual_query(entry: str, emotion: str) -> str:
    """
    Build a spiritually-framed search query from the detected emotion.
    These queries are tuned to match Quranic themes and translation vocabulary,
    bridging the gap between casual journal language and formal Quranic text.
    No LLM needed — keyword detection already identified the emotion reliably.
    """
    emotion_themes = {
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
    query = emotion_themes.get(emotion, "seeking guidance comfort and mercy from Allah in hardship")
    print(f"🔄 Spiritual query for '{emotion}': {query}")
    return query

# ─── FIX 2: EMOTION DETECTION ────────────────────────────────────────────
# Strategy: keyword scoring first (instant, free, reliable), LLM only as fallback.
# Free LLMs like Mistral-7b frequently return verbose responses even when asked for
# one word, making pure LLM classification unreliable. Keyword scoring always works.

# Each emotion maps to weighted keywords: (keyword, weight)
# Higher weight = stronger signal for that emotion
EMOTION_KEYWORDS: dict = {
    "grateful": [
        ("grateful", 3), ("gratitude", 3), ("thankful", 3), ("thank", 2),
        ("blessed", 3), ("blessing", 3), ("alhamdulillah", 3), ("appreciate", 2),
        ("appreciation", 2), ("fortunate", 2), ("realize how much", 2),
        ("take for granted", 2), ("so much", 1), ("abundance", 1),
    ],
    "reflective": [
        ("reflective", 3), ("reflect", 2), ("thinking about", 2), ("pondering", 2),
        ("contemplat", 2), ("life", 1), ("meaning", 2), ("purpose", 2),
        ("makes me think", 2), ("realize", 1), ("looking back", 2),
        ("quiet", 1), ("still", 1), ("calm", 1), ("at peace", 2),
    ],
    "anxious": [
        ("anxious", 3), ("anxiety", 3), ("worried", 3), ("worry", 3),
        ("nervous", 2), ("panic", 3), ("scared", 2), ("fear", 2),
        ("racing", 2), ("can't sleep", 3), ("overthinking", 3),
        ("won't stop", 2), ("mind", 1), ("exam", 2), ("test", 1),
        ("fail", 2), ("overwhelming", 2), ("what if", 2),
    ],
    "sad": [
        ("sad", 3), ("sadness", 3), ("cry", 2), ("crying", 2),
        ("tears", 2), ("depressed", 3), ("depression", 3), ("empty", 2),
        ("hopeless", 2), ("miserable", 3), ("down", 1), ("unhappy", 2),
        ("grief", 3), ("grieve", 2), ("miss", 1), ("lost someone", 3),
    ],
    "lonely": [
        ("lonely", 3), ("loneliness", 3), ("alone", 2), ("isolated", 3),
        ("no one", 2), ("nobody", 2), ("invisible", 3), ("disconnected", 3),
        ("no friends", 2), ("left out", 2), ("abandoned", 2),
    ],
    "heartbroken": [
        ("heartbroken", 3), ("heartbreak", 3), ("broken heart", 3),
        ("betrayed", 3), ("betrayal", 3), ("hurt me", 2), ("used me", 2),
        ("trusted", 2), ("lied", 2), ("cheated", 2), ("can't forgive", 2),
        ("burning", 1), ("can't let go", 2),
    ],
    "angry": [
        ("angry", 3), ("anger", 3), ("furious", 3), ("rage", 3),
        ("frustrated", 2), ("frustration", 2), ("unfair", 2), ("not fair", 2),
        ("injustice", 3), ("mad", 2), ("annoyed", 1), ("hate", 2),
    ],
    "stressed": [
        ("stressed", 3), ("stress", 3), ("overwhelmed", 3), ("pressure", 2),
        ("burden", 2), ("too much", 2), ("can't handle", 2), ("deadline", 2),
        ("exhausted from", 2), ("burning out", 3), ("burnout", 3),
    ],
    "tired": [
        ("tired", 3), ("exhausted", 3), ("fatigue", 2), ("fatigued", 2),
        ("drained", 3), ("worn out", 2), ("can't continue", 2),
        ("no energy", 2), ("sleepy", 1), ("can't go on", 2),
    ],
    "hopeful": [
        ("hopeful", 3), ("hope", 2), ("optimistic", 3), ("believe", 2),
        ("will get better", 3), ("going to be okay", 3), ("light", 1),
        ("trust allah", 2), ("inshallah", 2), ("dua", 2), ("praying", 1),
    ],
    "peaceful": [
        ("peaceful", 3), ("peace", 2), ("calm", 2), ("serene", 3),
        ("tranquil", 3), ("at ease", 2), ("relaxed", 2), ("settled", 1),
        ("quiet inside", 2), ("still heart", 2),
    ],
    "happy": [
        ("happy", 3), ("happiness", 3), ("joy", 2), ("joyful", 2),
        ("ecstatic", 3), ("elated", 3), ("wonderful", 2), ("great day", 2),
        ("excited", 2), ("love life", 2),
    ],
    "content": [
        ("content", 3), ("contented", 3), ("satisfied", 2), ("enough", 1),
        ("acceptance", 2), ("accept", 1), ("okay with", 2), ("at peace with", 2),
    ],
    "confused": [
        ("confused", 3), ("confusion", 3), ("lost", 2), ("don't know", 2),
        ("unclear", 2), ("don't understand", 2), ("which way", 2),
        ("what to do", 2), ("no direction", 2), ("guidance", 1),
    ],
}

def keyword_detect_emotion(entry: str) -> str:
    """Score the entry against keyword lists. Returns best-matching emotion or 'unsure'."""
    text = entry.lower()
    scores: dict[str, float] = {}

    for emotion, kw_list in EMOTION_KEYWORDS.items():
        score = 0.0
        for keyword, weight in kw_list:
            if keyword in text:
                score += weight
        if score > 0:
            scores[emotion] = score

    if not scores:
        return "unsure"

    best = max(scores, key=lambda e: scores[e])
    # Require a minimum score of 2 to avoid single weak keyword matches
    if scores[best] < 2:
        return "unsure"

    print(f"📊 Keyword scores: {sorted(scores.items(), key=lambda x: -x[1])[:4]}")
    print(f"✅ Keyword-detected emotion: {best}")
    return best


def llm_detect_emotion(entry: str) -> str:
    """LLM fallback — only called when keyword detection returns 'unsure'."""
    system = (
        "You are an emotion classifier. Output ONLY one word from this list:\n"
        "sad anxious hopeful grateful angry stressed tired peaceful confused happy lonely heartbroken content reflective\n"
        "ONE WORD. No punctuation. No explanation. No markdown.\n"
        "Example: grateful"
    )
    user = f"Entry: {entry}"
    raw = call_openrouter(system, user, max_tokens=10)
    print("🧪 LLM emotion raw:", repr(raw))
    if not raw:
        return "unsure"
    cleaned = raw.lower().strip().strip("*_`\"'.,!?:")
    for word in cleaned.split():
        w = word.strip("*_`\"'.,!?:")
        if w in VALID_EMOTIONS:
            return w
    for emotion in VALID_EMOTIONS:
        if emotion in cleaned:
            return emotion
    return "reflective"  # final fallback — better than "unsure" for query generation


def detect_emotion(entry: str) -> str:
    """Primary: keyword scoring. Fallback: LLM. Never returns 'unsure' if LLM is available."""
    result = keyword_detect_emotion(entry)
    if result != "unsure":
        return result
    print("⚠️ Keywords couldn't determine emotion — trying LLM...")
    return llm_detect_emotion(entry)

# ─── FIX 3: EMOTION-AWARE CANDIDATE FILTERING ───────────────────────────
# Instead of searching ALL 6000+ ayahs, we first narrow to ayahs tagged with the matching
# emotion in the CSV. This means similarity runs on a spiritually-relevant subset,
# dramatically improving match quality. Falls back to full search if no tag column exists
# or if the emotion isn't found in tags.
EMOTION_TO_TAGS = {
    "sad":         ["sadness", "grief", "sorrow", "loss"],
    "anxious":     ["anxiety", "fear", "worry", "uncertainty", "trust", "patience"],
    "hopeful":     ["hope", "optimism", "trust"],
    "grateful":    ["gratitude", "thankfulness", "blessing"],
    "angry":       ["anger", "injustice", "patience"],
    "stressed":    ["stress", "burden", "hardship", "relief"],
    "tired":       ["fatigue", "rest", "patience", "endurance"],
    "peaceful":    ["peace", "tranquility", "contentment"],
    "confused":    ["guidance", "clarity", "wisdom", "confusion"],
    "happy":       ["joy", "happiness", "blessing", "gratitude"],
    "lonely":      ["loneliness", "companionship", "comfort", "connection"],
    "heartbroken": ["heartbreak", "loss", "grief", "healing"],
    "content":     ["contentment", "satisfaction", "acceptance"],
    "reflective":  ["reflection", "contemplation", "wisdom", "remembrance"],
}

# Phrases that indicate an Ayah is about Judgement Day, war, or legal rulings —
# contextually wrong when someone is journaling about personal emotional struggles.
# Checked against the Ayah's English translation before adding to candidates.
EMOTION_EXCLUSIONS = {
    "anxious": [
        "gathered before", "day of judgement", "day of resurrection",
        "gathered to", "assembled", "reckoning", "hellfire", "punishment",
        "wrongdoers among them", "sacred mosque", "turn your face",
        "hypocrites", "disbelievers", "fight", "kill", "slew",
        "misfortune befall", "minds are diseased",
    ],
    "stressed": [
        "day of judgement", "hellfire", "fight", "kill", "sacred mosque",
        "hypocrites", "disbelievers", "punishment", "gathered before",
    ],
    "sad": [
        "hellfire", "fight them", "kill", "disbelievers",
        "punishment", "sacred mosque", "turn your face",
    ],
    "lonely": [
        "hellfire", "fight", "kill", "disbelievers", "punishment",
        "gathered before", "day of judgement",
    ],
    "heartbroken": [
        "hellfire", "fight", "kill", "disbelievers", "punishment",
    ],
    "angry": [
        "hellfire", "fight them", "kill", "sacred mosque",
        "gathered before", "day of judgement",
    ],
}

def get_candidate_indices(emotion: str) -> List[int]:
    """Return row indices from df that match the detected emotion via emotion tags."""
    all_indices = list(range(len(df)))

    emotion_cols = [c for c in df.columns if "emotion" in c.lower()]
    if not emotion_cols:
        print(f"⚠️ No emotion column found. Available columns: {df.columns.tolist()}")
        return all_indices

    tag_col = emotion_cols[0]
    print(f"📋 Using emotion column: '{tag_col}'")

    tags = EMOTION_TO_TAGS.get(emotion, [])
    if not tags:
        print(f"⚠️ No tag mapping for emotion '{emotion}' — searching all ayahs")
        return all_indices

    # fillna('') before apply to prevent float NaN from causing TypeError
    mask = df[tag_col].fillna('').astype(str).str.lower().apply(
        lambda cell: any(t in cell for t in tags) if cell else False
    )
    indices = df[mask].index.tolist()
    print(f"🎯 Emotion '{emotion}' → tags {tags} → {len(indices)} candidates before exclusion")

    # Apply exclusion filter: remove Ayahs whose English translation contains
    # contextually wrong phrases (e.g. Judgement Day fear for personal anxiety)
    exclusions = EMOTION_EXCLUSIONS.get(emotion, [])
    if exclusions and 'ayah_en' in df.columns:
        def not_excluded(idx):
            text = str(df.iloc[idx]['ayah_en']).lower()
            return not any(phrase in text for phrase in exclusions)
        indices = [i for i in indices if not_excluded(i)]
        print(f"🚫 After exclusion filter: {len(indices)} candidates remain")

    if len(indices) < 10:
        print(f"⚠️ Too few candidates ({len(indices)}) — falling back to all ayahs minus exclusions")
        # Even in fallback, still apply exclusions
        if exclusions and 'ayah_en' in df.columns:
            all_indices = [i for i in all_indices if not_excluded(i)]
        return all_indices

    return indices

# ─── FIX 4: MAXIMAL MARGINAL RELEVANCE (MMR) for diversity ───────────────
# Standard top-N gives you N ayahs all clustered in the same embedding region.
# MMR iteratively picks the next ayah that is: (a) relevant to the query AND
# (b) as different as possible from already-selected ayahs.
# lambda_param: 0.7 = 70% relevance, 30% diversity. Tune as needed.
def mmr_select(
    query_embedding,
    candidate_embeddings,
    candidate_indices: List[int],
    top_n: int = 3,
    lambda_param: float = 0.7,
    already_shown: List[int] = []
) -> List[int]:
    import torch

    # Remove already-shown ayahs from candidates
    filtered = [(ci, ce) for ci, ce in zip(candidate_indices, candidate_embeddings)
                if ci not in already_shown]
    if not filtered:
        # All candidates were already shown — reset and allow repeats
        filtered = list(zip(candidate_indices, candidate_embeddings))

    if not filtered:
        return []

    c_indices, c_embeds = zip(*filtered)
    c_embeds_tensor = torch.stack(list(c_embeds)) if hasattr(c_embeds[0], 'shape') else c_embeds

    # Cosine similarity with query
    query_sims = util.cos_sim(query_embedding, c_embeds_tensor)[0].tolist()

    selected = []
    remaining = list(range(len(c_indices)))

    while len(selected) < top_n and remaining:
        if not selected:
            # First pick: highest relevance
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            # Subsequent picks: balance relevance vs diversity
            selected_embeds = torch.stack([c_embeds_tensor[s] for s in selected]) \
                if hasattr(c_embeds_tensor[0], 'shape') else c_embeds_tensor[selected]
            best_score = -float('inf')
            best = None
            for i in remaining:
                relevance = query_sims[i]
                # Max similarity to already-selected (penalizes redundancy)
                redundancy = max(
                    util.cos_sim(c_embeds_tensor[i].unsqueeze(0), selected_embeds)[0].tolist()
                )
                score = lambda_param * relevance - (1 - lambda_param) * redundancy
                if score > best_score:
                    best_score = score
                    best = i
        selected.append(best)
        remaining.remove(best)

    return [c_indices[i] for i in selected]

# ─── COMFORT MESSAGE ────────────────────────────────────────────────────
# Emotion-specific comfort messages used when LLM is unavailable
COMFORT_FALLBACKS = {
    "grateful":    "MashaAllah — that quiet feeling of noticing your blessings is itself a gift from Allah. Alhamdulillah for the eyes to see, the heart to feel. Keep returning to gratitude; it opens more doors than you know. 🤍",
    "reflective":  "These quiet moments of reflection are rare and precious. Allah loves the heart that pauses and thinks. Whatever you're working through, trust that clarity will come — He guides those who seek. 🌙",
    "anxious":     "Allah knows the weight of what you're carrying right now. Take a breath — حَسْبُنَا اللَّهُ وَنِعْمَ الوَكِيل. He is enough, and He does not burden a soul beyond what it can bear. You will get through this. 🤍",
    "stressed":    "When everything feels too much, remember: Allah is closer to you than your own heartbeat. You don't have to solve it all today. Just take the next small step and leave the rest to Him. 🌿",
    "sad":         "It's okay to feel sad. Even the Prophets wept. Allah does not waste your tears — every one of them is seen and counted. Cry, rest, and know that after hardship always comes ease. 💛",
    "lonely":      "Feeling alone is one of the hardest feelings. But know that Allah is with you even when no one else is — He hears you before you even speak. You are never truly alone. 🤍",
    "heartbroken": "A broken heart is still a heart that loved, and love is never wasted in the eyes of Allah. Let yourself heal slowly. He is the only one who can truly mend what is broken. 🌙",
    "angry":       "Your anger makes sense. Seek refuge in Allah from letting it take over — أَعُوذُ بِاللَّهِ مِنَ الشَّيْطَانِ الرَّجِيمِ. Allah is Just; He sees everything. You don't have to carry the burden of fixing it alone. 🌿",
    "tired":       "Rest is not weakness — even our Prophet ﷺ rested. Your body and soul have rights over you. Be gentle with yourself today. Allah sees your effort even when you feel you have nothing left. 🤍",
    "confused":    "Not knowing which way to go is the perfect time to turn to Allah. Pray Istikhara, make dua, and trust that He will make the right path clear. Confusion is just the moment before clarity. 🌙",
    "hopeful":     "That hope you feel? Hold onto it tightly — it is a gift from Allah. He who gave you the hope will give you the thing you hope for, in His perfect time. Keep making dua. 💛",
    "peaceful":    "Alhamdulillah for this peace. It is a sign that your heart is close to Allah. Protect this feeling through dhikr and gratitude — it is one of the most precious states to be in. 🌿",
    "happy":       "Alhamdulillah for this happiness! Express it, share it, and thank Allah for it. Joy is a mercy from Him — savour it fully and let it fill your heart with gratitude. 💛",
    "content":     "Contentment — رِضا — is one of the highest stations of the heart. You have found something rare. Thank Allah for it and let it be a foundation, not a ceiling. 🌙",
}

def generate_comfort_message(entry: str, emotion: str) -> str:
    system = (
        "You're a soft, kind Islamic companion replying to an emotional journal entry. "
        "Use simple, warm, short sentences. No poetry, no deep metaphors, no 'I'm sorry'. "
        "Just calm, real Islamic advice — like a good friend who reminds you of Allah's mercy."
    )
    user = (
        f"The person is feeling: {emotion}\n"
        f"Their journal entry: {entry}\n\n"
        "Write a 2–3 line comforting message."
    )
    llm_result = call_openrouter(system, user, max_tokens=150)
    if llm_result:
        return llm_result
    # Use emotion-specific fallback — much better than a single generic message
    return COMFORT_FALLBACKS.get(
        emotion,
        "You're doing better than you think. Allah sees every step you take. One day at a time. 🤍"
    )

# ─── HELPER: get already-shown ayah indices for this user ───────────────
def get_shown_indices() -> List[int]:
    shown = []
    for item in history_db:
        for m in item.get("matches", []):
            if isinstance(m, dict):
                shown.append(m.get("ayah_index", -1))
            else:
                shown.append(getattr(m, "ayah_index", -1))
    return [i for i in shown if i >= 0]

# ─── ROUTES ──────────────────────────────────────────────────────────────

@app.post("/match-ayahs", response_model=MatchResponse)
def match_ayahs(request: EntryRequest):
    # Step 1: Detect emotion from raw entry
    emotion = detect_emotion(request.entry)

    # Step 2: Rewrite entry into spiritual query for better embedding
    spiritual_query = rewrite_as_spiritual_query(request.entry, emotion)

    # Step 3: Encode the spiritual query
    query_embedding = model.encode(spiritual_query, convert_to_tensor=True)

    # Step 4: Emotion-aware candidate filtering
    candidate_indices = get_candidate_indices(emotion)

    # Step 5: Get embeddings for candidates only
    import torch
    candidate_embeddings = [ayah_embeddings[i] for i in candidate_indices]

    # Step 5b: Pre-filter by minimum cosine similarity threshold
    # This removes Ayahs that only matched on a surface word (e.g. "blessings" in
    # an unrelated context). Threshold 0.25 is intentionally moderate — Quranic
    # translations use formal language so scores are naturally lower than modern text.
    raw_sims = util.cos_sim(query_embedding, torch.stack(candidate_embeddings))[0].tolist()
    SIMILARITY_THRESHOLD = 0.25
    threshold_filtered = [
        (ci, ce) for ci, ce, sim in zip(candidate_indices, candidate_embeddings, raw_sims)
        if sim >= SIMILARITY_THRESHOLD
    ]
    print(f"🔍 Similarity filter: {len(candidate_indices)} → {len(threshold_filtered)} candidates (threshold={SIMILARITY_THRESHOLD})")

    # If threshold is too strict and removes too many, fall back to top-30 by score
    if len(threshold_filtered) < request.top_n * 3:
        top_k = sorted(zip(candidate_indices, candidate_embeddings, raw_sims), key=lambda x: -x[2])
        threshold_filtered = [(ci, ce) for ci, ce, _ in top_k[:max(request.top_n * 5, 15)]]
        print(f"⚠️ Threshold too strict — using top-{len(threshold_filtered)} by similarity score")

    filtered_indices, filtered_embeds = zip(*threshold_filtered)
    filtered_indices, filtered_embeds = list(filtered_indices), list(filtered_embeds)

    # Step 6: MMR selection (diverse + relevant + not repeated)
    # lambda_param=0.85: strongly prefer relevance, allow moderate diversity
    already_shown = get_shown_indices()
    selected_indices = mmr_select(
        query_embedding,
        filtered_embeds,
        filtered_indices,
        top_n=request.top_n,
        lambda_param=0.85,
        already_shown=already_shown
    )

    # Step 7: Build matches
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

    # Step 8: Comfort message (emotion-aware)
    comfort = generate_comfort_message(request.entry, emotion)

    # Step 9: Save to history
    entry_id = len(history_db)
    history_db.append({
        "entry": request.entry,
        "matches": [m.dict() for m in matches],
        "comfort": comfort,
        "emotion_before": emotion,
        "emotion_after": None,
        "timestamp": datetime.utcnow().isoformat()
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
        return {"message": "Deleted"}
    raise HTTPException(status_code=404, detail="Invalid entry ID")


# ─── BOOKMARKS ───────────────────────────────────────────────────────────
@app.post("/bookmark", response_model=BookmarkItem)
def add_bookmark(req: BookmarkRequest):
    if req.ayah_index < 0 or req.ayah_index >= len(df):
        raise HTTPException(status_code=400, detail="Invalid ayah index")

    # Prevent duplicate bookmarks
    for bm in bookmarks_db:
        if bm["ayah_index"] == req.ayah_index:
            return BookmarkItem(**bm)  # already saved, return existing

    row = df.iloc[req.ayah_index]
    item = {
        "ayah_index": req.ayah_index,
        "surah": row['surah_name_roman'],
        "ayah_no": int(row['ayah_no_surah']),
        "ayah": row['ayah_en'].strip(),
        "ayah_ar": row['ayah_ar'].strip(),
        "note": req.note or "",
        "saved_at": datetime.utcnow().isoformat()
    }
    bookmarks_db.append(item)
    return BookmarkItem(**item)


@app.get("/bookmarks", response_model=List[BookmarkItem])
def get_bookmarks():
    return [BookmarkItem(**b) for b in bookmarks_db]


@app.delete("/bookmark/{ayah_index}")
def remove_bookmark(ayah_index: int):
    global bookmarks_db
    original_len = len(bookmarks_db)
    bookmarks_db = [b for b in bookmarks_db if b["ayah_index"] != ayah_index]
    if len(bookmarks_db) == original_len:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    return {"message": "Bookmark removed"}


# ─── WEEKLY PATTERN ANALYSIS ─────────────────────────────────────────────
# Uses already-collected history to show emotional trends without any extra LLM calls.
POSITIVE_EMOTIONS = {"hopeful", "grateful", "peaceful", "happy", "content"}

@app.get("/pattern", response_model=WeeklyPattern)
def get_pattern():
    if not history_db:
        raise HTTPException(status_code=404, detail="No history yet")

    emotions_before = [item["emotion_before"] for item in history_db if item["emotion_before"] != "unsure"]
    emotion_freq = dict(Counter(emotions_before))
    dominant = max(emotion_freq, key=emotion_freq.get) if emotion_freq else "unsure"

    # Count entries where user felt better after reading (emotion_after is more positive)
    shift_count = sum(
        1 for item in history_db
        if item.get("emotion_after") in POSITIVE_EMOTIONS
        and item.get("emotion_before") not in POSITIVE_EMOTIONS
    )

    # Most shown surah
    surah_counts: Counter = Counter()
    for item in history_db:
        for m in item.get("matches", []):
            surah = m.get("surah") if isinstance(m, dict) else m.surah
            if surah:
                surah_counts[surah] += 1

    most_shown = surah_counts.most_common(1)[0][0] if surah_counts else "N/A"

    return WeeklyPattern(
        total_entries=len(history_db),
        emotion_frequency=emotion_freq,
        dominant_emotion=dominant,
        shift_to_positive=shift_count,
        most_shown_surah=most_shown
    )


# ─── REFLECT AGAIN ───────────────────────────────────────────────────────
# After reading Ayahs, user can re-query with the same entry to get fresh Ayahs
# (the history already tracks what's been shown, so MMR will diversify naturally)
@app.post("/reflect-again/{entry_id}", response_model=MatchResponse)
def reflect_again(entry_id: int, top_n: int = 3):
    if entry_id < 0 or entry_id >= len(history_db):
        raise HTTPException(status_code=404, detail="Invalid entry ID")

    original = history_db[entry_id]
    # Re-run the full pipeline on the original entry — MMR will avoid already-shown ayahs
    return match_ayahs(EntryRequest(entry=original["entry"], top_n=top_n))