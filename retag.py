"""
مع القرآن — Anchor-Based CSV Retagger
======================================
Uses all-MiniLM-L6-v2 (same model as main.py) to retag every ayah
by measuring cosine similarity against carefully crafted Islamic
comfort anchors — one per emotion.

No LLM. No API. No rate limits. Runs in ~5 minutes locally.

How it works:
  1. Rule-based removal — dangerous/inappropriate ayahs → score 0.0
  2. Anchor scoring — each eligible ayah is compared against 14 emotion
     anchors using cosine similarity
  3. Retagging — ayah gets the emotion whose anchor it's closest to,
     but only if similarity >= MIN_ANCHOR_SIM (otherwise kept as-is)
  4. Low-confidence removal — ayahs with max anchor sim < REMOVE_SIM
     are removed from the pool (score set to 0.0)

Usage:
  python anchor_retag.py                  # full run
  python anchor_retag.py --dry-run        # stage 1 only, no model
  python anchor_retag.py --preview 20     # show top 20 changes before saving
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────
INPUT_CSV      = "quran_emotion_tagged.csv"
OUTPUT_CSV     = "quran_emotion_tagged_clean.csv"
REPORT_FILE    = "anchor_retag_report.json"
MODEL_NAME     = "all-MiniLM-L6-v2"

MIN_ANCHOR_SIM = 0.25   # below this → ayah removed from pool entirely
RETAG_SIM_GAP  = 0.03   # only retag if new emotion beats old by this margin
MIN_AYAH_WORDS = 8      # fragments shorter than this → remove

VALID_EMOTIONS = {
    "sad", "anxious", "hopeful", "grateful", "angry", "stressed",
    "tired", "peaceful", "confused", "happy", "lonely", "heartbroken",
    "content", "reflective"
}

# ── Emotion anchors ───────────────────────────────────────────────────────
# Each anchor is crafted as the IDEAL comfort language for that emotion.
# Multiple anchors per emotion — averaged for better coverage.

EMOTION_ANCHORS = {
    "sad": [
        "Allah comforts you in your grief and sorrow, your tears are not wasted, He sees every one of them",
        "After every hardship comes ease, do not despair of Allah's mercy, He is with the broken-hearted",
        "Allah knows how your heart aches and is distressed, He will relieve your pain and bring you peace",
        "Do not lose hope, Allah is close to those who are sad and weeping, He will heal your heart",
    ],
    "anxious": [
        "Do not fear, Allah is with you, He is your protector and guardian, trust in Him completely",
        "Allah calms the worried heart, He is sufficient for those who put their trust in Him",
        "Do not be anxious about the future, Allah knows what is best for you and He will take care of you",
        "Seek refuge in Allah from fear and worry, He is the remover of all hardship and anxiety",
    ],
    "stressed": [
        "Allah does not burden a soul with more than it can bear, with every hardship there is ease",
        "Allah will relieve your stress and overwhelm, He is the one who lightens every burden",
        "Turn to Allah when you feel overwhelmed and crushed by responsibilities, He will give you relief",
        "Surely after difficulty comes ease, Allah is the best of helpers for those who are struggling",
    ],
    "tired": [
        "Allah lightens the burdens of the exhausted and weak, man was created weak and Allah is gentle",
        "Rest and find comfort in Allah, He does not burden you beyond your strength, He is merciful",
        "Allah sees your effort and fatigue, He will give you strength and renew your energy",
        "When you are drained and have nothing left, turn to Allah who gives rest to the weary soul",
    ],
    "lonely": [
        "Allah is closer to you than your own jugular vein, He is always with you, you are never alone",
        "Allah hears you before you even speak, He sees you when no one else does, He never abandons you",
        "Your Lord has not forsaken you, He is near to those who feel alone and unseen",
        "Allah is the companion of the lonely, He is always present with those who call on Him",
    ],
    "heartbroken": [
        "Allah heals the broken heart, He mends what is shattered and restores what is lost",
        "When someone betrays you, Allah sees it all, He is the guardian of the wronged and betrayed",
        "Turn to Allah with your broken heart, He is the only one who can truly heal and restore you",
        "Do not let betrayal destroy you, Allah is just and He will not let your pain go unnoticed",
    ],
    "angry": [
        "Seek refuge in Allah from anger and rage, seek patience and forgiveness, Allah is with the patient",
        "Control your anger and return to Allah, deal justly, do not let enmity lead you away from justice",
        "Allah loves those who restrain their anger and forgive others, seek His help in your frustration",
        "When you feel furious and wronged, ask Allah for patience and strength, He is the Most Just",
    ],
    "hopeful": [
        "Allah's promise is true, do not lose hope, the future He has planned for you is beautiful",
        "Trust in Allah's perfect timing, good news awaits those who are patient and believe in Him",
        "Allah never breaks His promise, hold onto hope and trust that He will deliver you",
        "The believer always has hope in Allah's mercy and His plan, things will get better inshallah",
    ],
    "grateful": [
        "Thank Allah for all His blessings, every good thing you have is from Him alone",
        "Alhamdulillah, express gratitude to Allah who rewards the thankful with even more blessings",
        "Allah is bountiful and generous to those who are grateful, count your blessings and praise Him",
        "Gratitude to Allah opens doors of abundance, He rewards those who recognize His gifts",
    ],
    "peaceful": [
        "Hearts find peace and comfort in the remembrance of Allah, dhikr brings tranquility to the soul",
        "The believer whose heart is at rest with Allah has found true peace and contentment",
        "Allah's remembrance calms the heart and brings serenity, seek closeness to Him for peace",
        "True peace comes from being close to Allah, from prayer and gratitude and remembrance",
    ],
    "confused": [
        "Allah guides whoever seeks guidance to the straight path, ask Him for clarity and direction",
        "When you feel lost and confused, turn to Allah in dua, He will illuminate your path",
        "Allah is the guide of those who are lost, He will show you the right direction if you ask Him",
        "Seek guidance from Allah through prayer and trust, He knows what is best for you",
    ],
    "happy": [
        "Alhamdulillah for this joy, all happiness is a gift from Allah, thank Him and cherish it",
        "Allah blesses the believer with happiness and joy, celebrate His blessings with gratitude",
        "Your joy is a mercy from Allah, hold onto it and share it, thank Him for this beautiful gift",
        "The believing heart rejoices in Allah's gifts, every good thing is from His generosity",
    ],
    "content": [
        "Rida and contentment with what Allah has given is one of the highest stations of the heart",
        "The contented believer accepts Allah's will with peace and gratitude, finding richness in little",
        "True contentment comes from accepting what Allah has decreed with a grateful and trusting heart",
        "Allah suffices those who are content with His provision, satisfaction is the greatest wealth",
    ],
    "reflective": [
        "Ponder the signs of Allah in the universe and in yourself, reflection brings wisdom and closeness",
        "The believer reflects on the meaning of life and the purpose Allah has created them for",
        "Take time to think deeply about Allah's creation and His wisdom, this brings peace to the heart",
        "Contemplation and reflection on Allah's words and creation is a path to deeper faith and meaning",
    ],
}

# ── Rule-based removal patterns ───────────────────────────────────────────
# Ayahs matching these patterns are ALWAYS removed regardless of anchor score

REMOVAL_PATTERNS = [
    # Death wish
    "wish my death had ended all",
    "wish for death, if you are truthful",
    "wish for death if you are truthful",
    # Battle / military
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
    "left their homes for the cause of god and then were slain",
    # Ritual law
    "do not approach your prayers when you are drunk",
    "stand up praying for nearly two-thirds of the night",
    "two-thirds of the night",
    "when the prayer is ended, disperse in the land",
    "disperse in the land and seek",
    "pray for much of the night",
    "stand up to pray for much of the night",
    "praying at dawn",
    # Polemical / disbeliever rebuke
    "we put veils over their hearts to prevent them from comprehending",
    "afflict their ears with deafness",
    "turn their backs in aversion",
    "prevent them from testifying against you",
    "you were heedless of this, but now we have removed your veil",
    "your sight today is sharp",
    "minds are diseased",
    "hearts are diseased",
    "blind in this life will be blind in the life to come",
    # Violence / graphic
    "slaying your sons and sparing only your daughters",
    "subjected you to grievous torment",
    "lest you stone me",
    "stone me to death",
    # Judgement Day threat
    "guard yourselves against the day on which no soul shall in the least avail",
    "no soul shall bear the burden of another",
    "burden-bearer shall bear another's burden",
    "god will not grant a reprieve to a soul when its appointed time has come",
    "no soul shall die except with god's permission and at an appointed time",
    "no human being shall be of the least avail to any other human being",
    # Prophet-specific
    "invoke blessings on him",
    "bestow blessings on the prophet",
    "o believers, you also should invoke blessings on him",
    "help from god and imminent victory",
    "[o muhammad]",
    # Punishment / rebuke
    "make no excuses; you rejected the faith after you accepted it",
    "we will punish others amongst you, for they are guilty",
    "capable of retribution",
    # Meaningless fragments / ritual
    "that is no difficult thing for god",
    "pray to your lord and sacrifice to him alone",
    "turn your face to the sacred mosque",
    "turn your faces towards it wherever you may be",
    # Iblis / Satan story
    "what is the matter with you, that you are not among those who have prostrated",
    "he would not leave even one living creature",
    # Zakat / financial rulings
    "for those who collect zakat",
    "freeing slaves",
    "conciliating people's hearts",
    "do not invoke besides god what can neither help nor harm you",
    # Man is ungrateful
    "man is most ungrateful",
    "surely, man is most ungrateful",
    # Tried and tested framing
    "left their homes for the cause",
]

# ── Forced retag (rule-based, 100% certain) ───────────────────────────────
FORCED_RETAG = {
    "we are closer to him than his jugular vein":             "lonely",
    "closer to him than his jugular vein":                    "lonely",
    "we are nearer to him than you, although you cannot":     "lonely",
    "your suffering distresses him: he is deeply concerned":  "sad",
    "so we heard his prayer and delivered him from sorrow":   "sad",
    "do not lose heart or despair":                           "sad",
    "god wishes to lighten your burdens, for, man has been created weak": "tired",
    "with every hardship there is ease":                      "stressed",
    "verily, with every difficulty comes ease":               "stressed",
    "god does not charge a soul with more than it can bear":  "stressed",
    "we never charge a soul with more than it can bear":      "stressed",
    "surely in the remembrance of god hearts can find comfort": "peaceful",
    "hearts find comfort in the remembrance of god":          "peaceful",
    "we do indeed know how your heart is distressed":         "sad",
    "your lord has not forsaken you":                         "lonely",
    "so be patient, for what god has promised is sure to come": "hopeful",
    "there is good news in this life and in the hereafter":   "hopeful",
    "he who will cause me to die and bring me back to life":  None,
    "surely, man is most ungrateful":                         None,
    "man is most ungrateful":                                 None,
}


def stage1_rule_based(df: pd.DataFrame) -> tuple:
    """Apply rule-based removal and forced retag. Returns (df, changes)."""
    changes = []
    removal_count = 0
    forced_count = 0

    for idx, row in df.iterrows():
        text = str(row['ayah_en']).lower()
        original_score = float(row['emotion_score'])
        original_tag   = row['emotion_v2']

        # Too short
        if len(text.split()) < MIN_AYAH_WORDS:
            if original_score >= 0.25:
                df.at[idx, 'emotion_score'] = 0.0
                changes.append({"index": idx, "action": "removed",
                                "reason": "too_short", "original_tag": original_tag,
                                "new_tag": original_tag, "ayah_en": str(row['ayah_en'])[:100]})
                removal_count += 1
            continue

        # Dangerous pattern
        removed = False
        for pattern in REMOVAL_PATTERNS:
            if pattern in text:
                if original_score >= 0.25:
                    df.at[idx, 'emotion_score'] = 0.0
                    changes.append({"index": idx, "action": "removed",
                                   "reason": f"pattern:{pattern[:40]}", "original_tag": original_tag,
                                   "new_tag": original_tag, "ayah_en": str(row['ayah_en'])[:100]})
                    removal_count += 1
                removed = True
                break
        if removed:
            continue

        # Forced retag
        for fragment, correct_emotion in FORCED_RETAG.items():
            if fragment in text:
                if correct_emotion is None:
                    df.at[idx, 'emotion_score'] = 0.0
                    changes.append({"index": idx, "action": "removed",
                                   "reason": "forced_remove", "original_tag": original_tag,
                                   "new_tag": original_tag, "ayah_en": str(row['ayah_en'])[:100]})
                    removal_count += 1
                elif correct_emotion != original_tag:
                    new_score = max(original_score, 0.50)
                    df.at[idx, 'emotion_v2']    = correct_emotion
                    df.at[idx, 'emotion_score'] = new_score
                    changes.append({"index": idx, "action": "retagged",
                                   "reason": "forced_retag", "original_tag": original_tag,
                                   "new_tag": correct_emotion, "ayah_en": str(row['ayah_en'])[:100]})
                    forced_count += 1
                break

    print(f"  Rule removed:   {removal_count}")
    print(f"  Force-retagged: {forced_count}")
    return df, changes


def stage2_anchor_retag(df: pd.DataFrame, changes: list, preview: int = 0) -> tuple:
    """Score every eligible ayah against emotion anchors and retag."""

    print(f"\n── Stage 2: Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  ✅ Model loaded")

    # Encode all anchors — average multiple anchors per emotion
    print(f"  Encoding emotion anchors...")
    anchor_embeddings = {}
    for emotion, anchor_list in EMOTION_ANCHORS.items():
        embeddings = model.encode(anchor_list, convert_to_tensor=True, show_progress_bar=False)
        anchor_embeddings[emotion] = embeddings.mean(dim=0)  # average
    emotion_list = sorted(anchor_embeddings.keys())
    anchor_matrix = torch.stack([anchor_embeddings[e] for e in emotion_list])
    print(f"  ✅ {len(emotion_list)} emotion anchors encoded")

    # Get eligible ayahs only
    eligible_mask = df['emotion_score'] >= 0.25
    eligible_df   = df[eligible_mask]
    print(f"\n  Encoding {len(eligible_df)} eligible ayahs...")

    ayah_texts = eligible_df['ayah_en'].astype(str).tolist()
    ayah_embeddings = model.encode(
        ayah_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=64,
    )

    # Compute similarity matrix: (n_ayahs, n_emotions)
    print(f"  Computing similarities...")
    sim_matrix = util.cos_sim(ayah_embeddings, anchor_matrix).cpu().numpy()

    retag_count  = 0
    remove_count = 0
    preview_list = []

    for i, (idx, row) in enumerate(eligible_df.iterrows()):
        sims          = sim_matrix[i]
        best_idx      = int(np.argmax(sims))
        best_emotion  = emotion_list[best_idx]
        best_sim      = float(sims[best_idx])
        original_tag  = row['emotion_v2']
        original_score = float(row['emotion_score'])

        # Too dissimilar from all anchors → remove
        if best_sim < MIN_ANCHOR_SIM:
            df.at[idx, 'emotion_score'] = 0.0
            changes.append({
                "index": idx, "action": "removed",
                "reason": f"low_anchor_sim:{best_sim:.3f}",
                "original_tag": original_tag, "new_tag": original_tag,
                "best_emotion": best_emotion, "best_sim": round(best_sim, 3),
                "ayah_en": str(row['ayah_en'])[:100]
            })
            remove_count += 1
            continue

        # Retag if best emotion is different AND clearly better
        orig_emotion_idx = emotion_list.index(original_tag) if original_tag in emotion_list else -1
        orig_sim = float(sims[orig_emotion_idx]) if orig_emotion_idx >= 0 else 0.0

        if best_emotion != original_tag and (best_sim - orig_sim) >= RETAG_SIM_GAP:
            df.at[idx, 'emotion_v2']    = best_emotion
            df.at[idx, 'emotion_score'] = round(best_sim, 4)
            change = {
                "index": idx, "action": "retagged",
                "reason": f"anchor_retag",
                "original_tag": original_tag, "new_tag": best_emotion,
                "original_sim": round(orig_sim, 3), "new_sim": round(best_sim, 3),
                "gap": round(best_sim - orig_sim, 3),
                "ayah_en": str(row['ayah_en'])[:100]
            }
            changes.append(change)
            retag_count += 1
            if preview > 0 and len(preview_list) < preview:
                preview_list.append(change)
        else:
            # Keep original tag but update score to anchor similarity
            df.at[idx, 'emotion_score'] = round(best_sim, 4)

    print(f"\n  Anchor removed:  {remove_count}")
    print(f"  Anchor retagged: {retag_count}")

    if preview_list:
        print(f"\n── Preview of first {len(preview_list)} retagging changes:")
        for c in preview_list:
            print(f"  [{c['original_tag']} → {c['new_tag']}] gap={c['gap']:.3f} | {c['ayah_en'][:80]}")

    return df, changes


def run(dry_run: bool = False, preview: int = 0):
    print("\n" + "="*60)
    print("  مع القرآن — Anchor Retagger")
    print("="*60)

    df = pd.read_csv(INPUT_CSV)
    print(f"\n✅ Loaded {len(df)} ayahs")
    print(f"   Eligible (score ≥ 0.25): {int((df['emotion_score'] >= 0.25).sum())}")

    # Stage 1
    print(f"\n── Stage 1: Rule-based removal & forced retag ──")
    df, changes = stage1_rule_based(df)

    if dry_run:
        print("\n⚠️  Dry run — skipping anchor stage")
    else:
        # Stage 2
        print(f"\n── Stage 2: Anchor-based retagging ──")
        df, changes = stage2_anchor_retag(df, changes, preview=preview)

    # Save
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Saved → {OUTPUT_CSV}")

    # Report
    removed_total  = sum(1 for c in changes if c["action"] == "removed")
    retagged_total = sum(1 for c in changes if c["action"] == "retagged")

    report = {
        "timestamp": datetime.now().isoformat(),
        "input": INPUT_CSV, "output": OUTPUT_CSV,
        "total_ayahs": len(df),
        "removed": removed_total,
        "retagged": retagged_total,
        "eligible_after": int((df['emotion_score'] >= 0.25).sum()),
        "emotion_distribution": df[df['emotion_score'] >= 0.25]['emotion_v2'].value_counts().to_dict(),
        "changes": changes
    }
    Path(REPORT_FILE).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Summary
    original_df = pd.read_csv(INPUT_CSV)
    orig_dist = original_df[original_df['emotion_score'] >= 0.25]['emotion_v2'].value_counts()
    new_dist  = df[df['emotion_score'] >= 0.25]['emotion_v2'].value_counts()

    print(f"\n{'='*55}")
    print(f"  Removed:        {removed_total}")
    print(f"  Retagged:       {retagged_total}")
    print(f"  Eligible after: {int((df['emotion_score'] >= 0.25).sum())}")
    print(f"{'='*55}")
    print(f"\n{'Emotion':<15} {'Before':>8} {'After':>8} {'Change':>8}")
    print("-" * 45)
    for emotion in sorted(VALID_EMOTIONS):
        before = orig_dist.get(emotion, 0)
        after  = new_dist.get(emotion, 0)
        diff   = after - before
        sign   = "+" if diff >= 0 else ""
        print(f"  {emotion:<13} {before:>8} {after:>8} {sign}{diff:>7}")
    print(f"\n  Report → {REPORT_FILE}")
    print(f"  Clean CSV → {OUTPUT_CSV}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true", help="Stage 1 only, no model")
    parser.add_argument("--preview",  type=int, default=0, help="Show N sample retagging changes")
    args = parser.parse_args()
    run(dry_run=args.dry_run, preview=args.preview)