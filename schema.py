from pydantic import BaseModel
from typing import List, Optional

# ─── MODELS ─────────────────────────────────────────────────────────────
class EntryRequest(BaseModel):
    entry: str
    top_n: int = 3

class AyahMatch(BaseModel):
    surah: str
    ayah_no: int
    ayah: str
    ayah_ar: str 

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