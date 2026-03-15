"""
مع القرآن — Production CSV Retagging Script
============================================
Cleans and retags quran_emotion_tagged.csv for production use.

Three-stage pipeline:
  Stage 1 — Rule-based removal of dangerous/inappropriate ayahs
  Stage 2 — LLM retagging of low-confidence ayahs (score 0.25–0.35)
  Stage 3 — LLM validation of medium-confidence ayahs (score 0.35–0.45)
             that match known problem patterns

Output:
  quran_emotion_tagged_clean.csv  — production-ready CSV
  retag_report.json               — full audit log of every change

Usage:
  pip install pandas requests tqdm
  OPENROUTER_API_KEY=your_key python retag.py

  # Dry run (no LLM calls, only rule-based):
  python retag.py --dry-run

  # Skip stage 3 validation (faster, less thorough):
  python retag.py --skip-validation

  # Resume after interruption:
  python retag.py --resume
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────
INPUT_CSV     = "quran_emotion_tagged.csv"
OUTPUT_CSV    = "quran_emotion_tagged_clean.csv"
REPORT_FILE   = "retag_report.json"
PROGRESS_FILE = "retag_progress.json"  # for resuming

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL          = "meta-llama/llama-3.3-70b-instruct:free"

# Score thresholds
REMOVE_BELOW        = 0.25   # below this: already excluded by main.py
RETAG_BELOW         = 0.35   # 0.25–0.35: low confidence, retag
VALIDATE_BELOW      = 0.45   # 0.35–0.45: medium confidence, validate if flagged
MIN_AYAH_WORDS      = 8      # fragments shorter than this → remove

VALID_EMOTIONS = {
    "sad", "anxious", "hopeful", "grateful", "angry", "stressed",
    "tired", "peaceful", "confused", "happy", "lonely", "heartbroken",
    "content", "reflective"
}

# ── Stage 1: Rule-based removal patterns ─────────────────────────────────
# These patterns indicate an ayah should NEVER appear in a personal journal
# context regardless of emotion tag. Score will be set to 0.0.

REMOVAL_PATTERNS = [
    # ── Death wish ──
    "wish my death had ended all",
    "wish for death, if you are truthful",
    "wish for death if you are truthful",

    # ── Battle / military commands ──
    "fight for the cause of god",
    "fight in the cause of god",
    "go forth, whether lightly or heavily equipped",
    "urge on the believers",
    "strive and struggle, with your goods and your persons",
    "left their homes for the cause of god and then were slain",
    "spent and fought before the victory",
    "do not relent in the pursuit of the enemy",
    "strike their necks",
    "cast terror into the hearts",
    "smite above their necks",
    "prisoners of war",
    "taken captive",
    "fight them until there is no more persecution",

    # ── Ritual law / legal rulings ──
    "do not approach your prayers when you are drunk",
    "stand up praying for nearly two-thirds of the night",
    "two-thirds of the night",
    "when the prayer is ended, disperse in the land",
    "disperse in the land and seek",
    "pray for much of the night",
    "praying at dawn",

    # ── Polemical / disbeliever rebuke ──
    "we put veils over their hearts to prevent them from comprehending",
    "afflict their ears with deafness",
    "turn their backs in aversion",
    "prevent them from testifying against you",
    "you were heedless of this, but now we have removed your veil",
    "your sight today is sharp",
    "minds are diseased",
    "hearts are diseased",

    # ── Violence / graphic ──
    "slaying your sons and sparing only your daughters",
    "subjected you to grievous torment",
    "lest you stone me",
    "stone me to death",

    # ── Judgement Day threat framing ──
    "guard yourselves against the day on which no soul shall in the least avail",
    "no soul shall bear the burden of another",
    "burden-bearer shall bear another's burden",
    "god will not grant a reprieve to a soul when its appointed time has come",
    "no soul shall die except with god's permission and at an appointed time",

    # ── Prophet-specific proclamations ──
    "invoke blessings on him",
    "bestow blessings on the prophet",
    "o believers, you also should invoke blessings on him",
    "help from god and imminent victory",
    "[o muhammad]",

    # ── Punishment / rebuke ──
    "make no excuses; you rejected the faith after you accepted it",
    "we will punish others amongst you, for they are guilty",

    # ── Meaningless fragments ──
    "that is no difficult thing for god",
    "to god belongs the hereafter and this world belong",
    "pray to your lord and sacrifice to him alone",

    # ── Qibla / sacred mosque ritual ──
    "turn your face to the sacred mosque",
    "turn your faces towards it wherever you may be",
]

# ── Stage 1: Wrong-emotion corrections (rule-based retag) ─────────────────
# Ayahs whose correct emotion is CERTAIN from the text — no LLM needed.

FORCED_RETAG = {
    # text_fragment (lowercase) → correct emotion
    "we are closer to him than his jugular vein":         "lonely",
    "closer to him than his jugular vein":                "lonely",
    "we are nearer to him than you, although you cannot": "lonely",
    "god is closer to him than his own jugular vein":     "lonely",
    "your suffering distresses him: he is deeply concerned": "sad",
    "so we heard his prayer and delivered him from sorrow":  "sad",
    "do not lose heart or despair":                       "sad",
    "god wishes to lighten your burdens, for, man has been created weak": "tired",
    "with every hardship there is ease":                  "stressed",
    "verily, with every difficulty comes ease":           "stressed",
    "god does not charge a soul with more than it can bear": "stressed",
    "we never charge a soul with more than it can bear":  "stressed",
    "surely in the remembrance of god hearts can find comfort": "peaceful",
    "hearts find comfort in the remembrance of god":      "peaceful",
    "we do indeed know how your heart is distressed":     "sad",
    "your lord has not forsaken you":                     "sad",
    "did he not find you an orphan and shelter you":      "sad",
    "so be patient, for what god has promised is sure to come": "hopeful",
    "there is good news in this life and in the hereafter": "hopeful",
    "he who will cause me to die and bring me back to life": None,  # remove
    "man is most ungrateful":                             None,     # remove
    "surely, man is most ungrateful":                     None,     # remove
}

# ── Stage 2/3: LLM prompts ────────────────────────────────────────────────

EMOTION_DESCRIPTIONS = {
    "sad":         "deep sadness, grief, crying, emptiness, loss — needs Allah's comfort and mercy",
    "anxious":     "fear, worry, panic, nervousness about the future — needs reassurance and calm",
    "hopeful":     "optimism, trust in Allah, belief things will improve — uplifting and forward-looking",
    "grateful":    "thankfulness, counting blessings, alhamdulillah — celebratory and appreciative",
    "angry":       "frustration, injustice, rage — needs patience, perspective, and self-control",
    "stressed":    "overwhelm, pressure, too many responsibilities — needs ease and relief",
    "tired":       "exhaustion, burnout, physical/emotional depletion — needs rest and gentle encouragement",
    "peaceful":    "calm, tranquility, contentment, settled heart — serene and still",
    "confused":    "lost, no direction, unclear path — needs guidance and clarity",
    "happy":       "joy, excitement, delight, positive energy — celebratory",
    "lonely":      "isolation, feeling unseen, nobody around — needs reminder of Allah's nearness",
    "heartbroken": "betrayal, broken trust, loss of relationship — needs healing and strength",
    "content":     "satisfaction, acceptance, rida — settled and at peace with what Allah gave",
    "reflective":  "contemplation, thinking about life/purpose/meaning — thoughtful and introspective",
}

RETAG_SYSTEM_PROMPT = """You are an Islamic scholar and therapist helping build a Quran journaling app for people going through emotional struggles.

Your job: Given an ayah (Quranic verse), decide the SINGLE best emotion tag for a person reading it during personal journaling.

Rules:
- Choose the emotion this ayah would COMFORT or RESONATE WITH most
- If the ayah is about battle, ritual law, punishment, death threats, or disbeliever rebuke → answer: REMOVE
- If the ayah is a meaningless fragment (under 8 words) → answer: REMOVE
- If it addresses the Prophet specifically (not general believers) → answer: REMOVE
- Output ONLY one word from: sad anxious hopeful grateful angry stressed tired peaceful confused happy lonely heartbroken content reflective REMOVE
- No explanation. No punctuation. Just the single word."""

VALIDATE_SYSTEM_PROMPT = """You are an Islamic scholar reviewing Quran verse emotion tags for a journaling app.

Your job: Decide if the CURRENT emotion tag is correct or suggest a better one.

Rules:
- If the ayah is about battle, punishment, ritual law, death threats, or disbeliever rebuke → answer: REMOVE
- If the current tag feels right → repeat the current tag
- If a different tag fits better → give the better tag
- Output ONLY one word: sad anxious hopeful grateful angry stressed tired peaceful confused happy lonely heartbroken content reflective REMOVE
- No explanation. No punctuation. Just the single word."""


# ── LLM caller ───────────────────────────────────────────────────────────
def call_llm(system: str, user: str, retries: int = 3) -> str:
    if not OPENROUTER_API_KEY:
        return ""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "max_tokens": 10,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ]
    }
    for attempt in range(retries):
        try:
            res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=20)
            res.raise_for_status()
            data = res.json()
            if "choices" not in data or not data["choices"]:
                return ""
            raw = data["choices"][0]["message"]["content"].strip().lower()
            raw = raw.strip("*_`\"'.,!?:\n ")
            for word in raw.split():
                w = word.strip("*_`\"'.,!?:")
                if w in VALID_EMOTIONS or w == "remove":
                    return w
            return ""
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 429 or status == 500:
                wait = 2 ** attempt
                print(f"    HTTP {status} — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    HTTP error {status}: {e}")
                if attempt == retries - 1:
                    return ""
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"    LLM error: {e}")
    return ""


def retag_with_llm(ayah_en: str, ayah_ar: str, current_tag: str, mode: str = "retag") -> str:
    """Call LLM to retag or validate an ayah. Returns new tag or 'remove'."""
    emotion_list = "\n".join([f"- {e}: {d}" for e, d in EMOTION_DESCRIPTIONS.items()])

    if mode == "retag":
        system = RETAG_SYSTEM_PROMPT
        user = f"Ayah (English): {ayah_en}\nAyah (Arabic): {ayah_ar}\n\nEmotion options:\n{emotion_list}\n\nBest emotion tag:"
    else:  # validate
        system = VALIDATE_SYSTEM_PROMPT
        user = f"Ayah (English): {ayah_en}\nAyah (Arabic): {ayah_ar}\nCurrent tag: {current_tag}\n\nEmotion options:\n{emotion_list}\n\nCorrect tag (or REMOVE):"

    return call_llm(system, user)


# ── Progress management ───────────────────────────────────────────────────
def load_progress() -> dict:
    if Path(PROGRESS_FILE).exists():
        try:
            return json.loads(Path(PROGRESS_FILE).read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"processed": {}, "stage": 1}

def save_progress(progress: dict):
    Path(PROGRESS_FILE).write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Main pipeline ─────────────────────────────────────────────────────────
def run(dry_run: bool = False, skip_validation: bool = False, resume: bool = False):
    print("\n" + "="*60)
    print("  مع القرآن — CSV Retagger")
    print("="*60)

    # Load CSV
    df = pd.read_csv(INPUT_CSV)
    print(f"\n✅ Loaded {len(df)} ayahs from {INPUT_CSV}")

    # Load progress if resuming
    progress = load_progress() if resume else {"processed": {}, "stage": 1}
    changes = []  # audit log

    # ── Stage 1: Rule-based removal ──────────────────────────────────────
    print("\n── Stage 1: Rule-based removal & forced retag ──")

    removal_count = 0
    forced_retag_count = 0

    for idx, row in df.iterrows():
        text = str(row['ayah_en']).lower()
        original_score = row['emotion_score']
        original_tag   = row['emotion_v2']

        # Skip if already processed and resuming
        if resume and str(idx) in progress["processed"]:
            continue

        # Rule 1: Too short
        if len(text.split()) < MIN_AYAH_WORDS:
            if original_score >= 0.25:
                df.at[idx, 'emotion_score'] = 0.0
                changes.append({
                    "index": idx,
                    "surah": row['surah_name_roman'],
                    "ayah_no": int(row['ayah_no_surah']),
                    "action": "removed",
                    "reason": "fragment_too_short",
                    "original_tag": original_tag,
                    "new_tag": original_tag,
                    "original_score": float(original_score),
                    "new_score": 0.0,
                    "ayah_en": str(row['ayah_en'])[:120]
                })
                removal_count += 1
            continue

        # Rule 2: Dangerous/inappropriate patterns
        removed = False
        for pattern in REMOVAL_PATTERNS:
            if pattern in text:
                if original_score >= 0.25:
                    df.at[idx, 'emotion_score'] = 0.0
                    changes.append({
                        "index": idx,
                        "surah": row['surah_name_roman'],
                        "ayah_no": int(row['ayah_no_surah']),
                        "action": "removed",
                        "reason": f"pattern: {pattern[:50]}",
                        "original_tag": original_tag,
                        "new_tag": original_tag,
                        "original_score": float(original_score),
                        "new_score": 0.0,
                        "ayah_en": str(row['ayah_en'])[:120]
                    })
                    removal_count += 1
                removed = True
                break
        if removed:
            continue

        # Rule 3: Forced retag
        for fragment, correct_emotion in FORCED_RETAG.items():
            if fragment in text:
                if correct_emotion is None:
                    # Remove
                    df.at[idx, 'emotion_score'] = 0.0
                    changes.append({
                        "index": idx,
                        "surah": row['surah_name_roman'],
                        "ayah_no": int(row['ayah_no_surah']),
                        "action": "removed",
                        "reason": f"forced_remove: {fragment[:50]}",
                        "original_tag": original_tag,
                        "new_tag": original_tag,
                        "original_score": float(original_score),
                        "new_score": 0.0,
                        "ayah_en": str(row['ayah_en'])[:120]
                    })
                    removal_count += 1
                elif correct_emotion != original_tag:
                    # Retag
                    df.at[idx, 'emotion_v2'] = correct_emotion
                    # Boost score slightly since we're confident in this tag
                    new_score = max(float(original_score), 0.45)
                    df.at[idx, 'emotion_score'] = new_score
                    changes.append({
                        "index": idx,
                        "surah": row['surah_name_roman'],
                        "ayah_no": int(row['ayah_no_surah']),
                        "action": "retagged",
                        "reason": f"forced_retag: {fragment[:50]}",
                        "original_tag": original_tag,
                        "new_tag": correct_emotion,
                        "original_score": float(original_score),
                        "new_score": new_score,
                        "ayah_en": str(row['ayah_en'])[:120]
                    })
                    forced_retag_count += 1
                break

    print(f"  Removed:       {removal_count} ayahs")
    print(f"  Force-retagged: {forced_retag_count} ayahs")

    if dry_run:
        print("\n⚠️  Dry run — skipping LLM stages")
    else:
        # ── Stage 2: LLM retagging of low-confidence ayahs ───────────────
        print(f"\n── Stage 2: LLM retagging (score 0.25–0.35) ──")

        low_conf_mask = (
            (df['emotion_score'] >= 0.25) &
            (df['emotion_score'] < RETAG_BELOW)
        )
        low_conf_indices = df[low_conf_mask].index.tolist()
        print(f"  Candidates: {len(low_conf_indices)} ayahs")

        if not OPENROUTER_API_KEY:
            print("  ⚠️  No OPENROUTER_API_KEY — skipping LLM stages")
        else:
            llm_retag_count = 0
            llm_remove_count = 0
            errors = 0

            for idx in tqdm(low_conf_indices, desc="  Retagging"):
                if resume and str(idx) in progress["processed"]:
                    continue

                row = df.iloc[idx]
                original_tag   = row['emotion_v2']
                original_score = row['emotion_score']

                new_tag = retag_with_llm(
                    str(row['ayah_en']),
                    str(row['ayah_ar']),
                    original_tag,
                    mode="retag"
                )

                if not new_tag:
                    errors += 1
                    progress["processed"][str(idx)] = "error"
                    time.sleep(0.5)
                    continue

                if new_tag == "remove":
                    df.at[idx, 'emotion_score'] = 0.0
                    changes.append({
                        "index": idx,
                        "surah": row['surah_name_roman'],
                        "ayah_no": int(row['ayah_no_surah']),
                        "action": "removed",
                        "reason": "llm_stage2",
                        "original_tag": original_tag,
                        "new_tag": original_tag,
                        "original_score": float(original_score),
                        "new_score": 0.0,
                        "ayah_en": str(row['ayah_en'])[:120]
                    })
                    llm_remove_count += 1
                elif new_tag != original_tag:
                    df.at[idx, 'emotion_v2'] = new_tag
                    changes.append({
                        "index": idx,
                        "surah": row['surah_name_roman'],
                        "ayah_no": int(row['ayah_no_surah']),
                        "action": "retagged",
                        "reason": "llm_stage2",
                        "original_tag": original_tag,
                        "new_tag": new_tag,
                        "original_score": float(original_score),
                        "new_score": float(original_score),
                        "ayah_en": str(row['ayah_en'])[:120]
                    })
                    llm_retag_count += 1

                progress["processed"][str(idx)] = new_tag
                save_progress(progress)
                time.sleep(0.3)  # rate limiting

            print(f"  LLM removed:  {llm_remove_count}")
            print(f"  LLM retagged: {llm_retag_count}")
            print(f"  Errors:       {errors}")

        # ── Stage 3: LLM validation of suspicious medium-confidence ──────
        if not skip_validation and OPENROUTER_API_KEY:
            print(f"\n── Stage 3: LLM validation (score 0.35–0.45, flagged patterns) ──")

            # Patterns that suggest possible mislabeling even at medium confidence
            SUSPICIOUS_PATTERNS = [
                "slain", "died", "death", "punish", "torment", "hellfire",
                "disbeliev", "hypocrit", "ungrateful", "fight", "battle",
                "war", "enemy", "destroy", "wrath", "curse", "guilty",
                "burden", "reprieve", "appointed time", "judgement",
                "zakat", "alms", "spend", "charity", "sacrifice",
            ]

            med_conf_mask = (
                (df['emotion_score'] >= RETAG_BELOW) &
                (df['emotion_score'] < VALIDATE_BELOW) &
                (df['emotion_score'] > 0)
            )
            med_conf = df[med_conf_mask]

            # Only validate ones that contain suspicious patterns
            suspicious_indices = []
            for idx, row in med_conf.iterrows():
                text = str(row['ayah_en']).lower()
                if any(p in text for p in SUSPICIOUS_PATTERNS):
                    suspicious_indices.append(idx)

            print(f"  Suspicious candidates: {len(suspicious_indices)} ayahs")

            val_remove_count = 0
            val_retag_count  = 0
            val_errors       = 0

            for idx in tqdm(suspicious_indices, desc="  Validating"):
                if resume and str(idx) in progress["processed"]:
                    continue

                row = df.iloc[idx]
                original_tag   = row['emotion_v2']
                original_score = row['emotion_score']

                new_tag = retag_with_llm(
                    str(row['ayah_en']),
                    str(row['ayah_ar']),
                    original_tag,
                    mode="validate"
                )

                if not new_tag:
                    val_errors += 1
                    progress["processed"][str(idx)] = "error"
                    time.sleep(0.5)
                    continue

                if new_tag == "remove":
                    df.at[idx, 'emotion_score'] = 0.0
                    changes.append({
                        "index": idx,
                        "surah": row['surah_name_roman'],
                        "ayah_no": int(row['ayah_no_surah']),
                        "action": "removed",
                        "reason": "llm_stage3_validation",
                        "original_tag": original_tag,
                        "new_tag": original_tag,
                        "original_score": float(original_score),
                        "new_score": 0.0,
                        "ayah_en": str(row['ayah_en'])[:120]
                    })
                    val_remove_count += 1
                elif new_tag != original_tag:
                    df.at[idx, 'emotion_v2'] = new_tag
                    changes.append({
                        "index": idx,
                        "surah": row['surah_name_roman'],
                        "ayah_no": int(row['ayah_no_surah']),
                        "action": "retagged",
                        "reason": "llm_stage3_validation",
                        "original_tag": original_tag,
                        "new_tag": new_tag,
                        "original_score": float(original_score),
                        "new_score": float(original_score),
                        "ayah_en": str(row['ayah_en'])[:120]
                    })
                    val_retag_count += 1

                progress["processed"][str(idx)] = new_tag
                save_progress(progress)
                time.sleep(0.3)

            print(f"  Validated removed:  {val_remove_count}")
            print(f"  Validated retagged: {val_retag_count}")
            print(f"  Errors:             {val_errors}")

    # ── Save output CSV ──────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Saved clean CSV → {OUTPUT_CSV}")

    # ── Save report ───────────────────────────────────────────────────────
    removed_total  = sum(1 for c in changes if c["action"] == "removed")
    retagged_total = sum(1 for c in changes if c["action"] == "retagged")

    report = {
        "timestamp": datetime.now().isoformat(),
        "input_file": INPUT_CSV,
        "output_file": OUTPUT_CSV,
        "total_ayahs": len(df),
        "total_changes": len(changes),
        "removed": removed_total,
        "retagged": retagged_total,
        "eligible_after": int((df['emotion_score'] >= 0.25).sum()),
        "emotion_distribution_after": df[df['emotion_score'] >= 0.25]['emotion_v2'].value_counts().to_dict(),
        "changes": changes
    }

    Path(REPORT_FILE).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  Total changes:    {len(changes)}")
    print(f"  Removed:          {removed_total}")
    print(f"  Retagged:         {retagged_total}")
    print(f"  Eligible ayahs:   {int((df['emotion_score'] >= 0.25).sum())}")
    print(f"\n  Report → {REPORT_FILE}")
    print(f"  Clean CSV → {OUTPUT_CSV}")
    print("="*60 + "\n")

    # ── Emotion distribution comparison ───────────────────────────────────
    original_df = pd.read_csv(INPUT_CSV)
    orig_dist = original_df[original_df['emotion_score'] >= 0.25]['emotion_v2'].value_counts()
    new_dist  = df[df['emotion_score'] >= 0.25]['emotion_v2'].value_counts()

    print("Emotion pool size — before vs after:")
    print(f"{'Emotion':<15} {'Before':>8} {'After':>8} {'Change':>8}")
    print("-" * 45)
    for emotion in sorted(VALID_EMOTIONS):
        before = orig_dist.get(emotion, 0)
        after  = new_dist.get(emotion, 0)
        diff   = after - before
        sign   = "+" if diff >= 0 else ""
        print(f"  {emotion:<13} {before:>8} {after:>8} {sign}{diff:>7}")

    # Clean up progress file on success
    if Path(PROGRESS_FILE).exists():
        Path(PROGRESS_FILE).unlink()


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retag quran_emotion_tagged.csv for production")
    parser.add_argument("--dry-run",         action="store_true", help="Stage 1 only, no LLM calls")
    parser.add_argument("--skip-validation", action="store_true", help="Skip Stage 3 validation")
    parser.add_argument("--resume",          action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    run(
        dry_run=args.dry_run,
        skip_validation=args.skip_validation,
        resume=args.resume
    )