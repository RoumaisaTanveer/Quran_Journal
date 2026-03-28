
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = "quran_emotion_tagged_clean.csv"
DEFAULT_OUTPUT = "quran_emotion_retagged_v3.csv"
MODEL_NAME     = "all-MiniLM-L6-v2"

# Minimum cosine similarity to ANY emotion anchor to be included.
# Below this → EXCLUDE (not confident enough for journaling use).
INCLUDE_CUTOFF = 0.28

# If the top emotion wins by less than this margin over second-best → borderline → EXCLUDE.
# Prevents ambiguous ayahs from polluting emotion pools.
MARGIN_CUTOFF  = 0.03

# ── Anchor fixes ──────────────────────────────────────────────────────────────

ANCHORS["sad"] = [
    "My heart is heavy with grief and sorrow, Allah sees my pain and does not abandon me",
    "I have been crying and feel empty inside, Allah is near to those whose hearts are broken",
    "The pain I carry feels unbearable, but Allah promises ease will come after hardship",
    "I miss someone and my heart aches, Allah comforts those who grieve and does not forget them",
    "I feel hopeless and low, but Allah's mercy is greater than any sadness I carry",
    "Even in my darkest moment of sadness, Allah is with me and my tears are not wasted",
]

ANCHORS["reflective"] = [
    "I find myself pausing to think about the deeper meaning and purpose of my life",
    "Looking back on my journey so far and wondering what it all means and where I am headed",
    "I want to understand myself better and think about what truly matters in this life",
    "Sitting quietly and contemplating my choices, my faith, and my relationship with Allah",
    "There is wisdom I am searching for, a deeper understanding of my own soul and existence",
    "I feel drawn inward today, thinking about who I am and who I want to become",
]

ANCHORS["tired"] = [
    "I am completely exhausted and drained, I have nothing left to give today",
    "My body and soul are worn out, I need Allah's mercy and gentleness right now",
    "I feel so depleted I cannot continue, I need rest and Allah's strength to carry on",
    "Everything feels heavy and I am running on empty, Allah knows how tired I truly am",
    "I have been striving so hard for so long, I am weary and need Allah to lighten my burden",
    "My energy is gone and I feel burnt out, Allah is gentle with those who are exhausted",
]

ANCHORS["peaceful"] = [
    "My heart feels calm and still, I am at rest in the remembrance of Allah",
    "I feel a deep inner tranquility today, a quietness that comes from trusting Allah",
    "There is no fear or worry in my heart right now, just stillness and gratitude",
    "I feel settled and at ease, my soul is at peace with what Allah has given me",
    "This moment feels serene and gentle, my heart is soft and content with Allah",
    "A beautiful calmness has come over me, the kind of peace only Allah can give",
]

ANCHORS["heartbroken"] = [
    "Someone I trusted broke my heart and I feel shattered inside from their betrayal",
    "I loved someone and they hurt me deeply, my heart is in pieces and I cannot let go",
    "The pain of betrayal feels unbearable, I trusted them and they destroyed that trust",
    "My heart is broken by someone I cared for, I feel wounded and cannot move on easily",
    "I gave someone everything and they left me broken, Allah alone can heal this wound",
    "The grief of losing someone I loved is overwhelming, only Allah can mend my broken heart",
]

ANCHORS["confused"] = [
    "I do not know which direction to take in life and feel completely lost inside",
    "I am unclear about my path forward and need Allah's guidance and light in my confusion",
    "Everything feels uncertain and I do not know what to do or which choice is right",
    "I feel spiritually and emotionally lost, searching for clarity and direction from Allah",
    "My mind is full of doubts and I cannot see clearly, I need Allah to show me the way",
    "I feel stuck and directionless, not knowing what Allah wants for me or which way to go",
]

# ── New pattern exclusions ─────────────────────────────────────────────────────
EXCLUDE_PATTERNS.append(("kaf ha ya", "muqatta'at opener — not comforting"))
EXCLUDE_PATTERNS.append(("warn men of the day", "warning/punishment framing"))
EXCLUDE_PATTERNS.append(("do not lose heart or appeal for peace when you have gained", "battle strategy"))
EXCLUDE_PATTERNS.append(("we are utterly ruined", "despair framing"))
EXCLUDE_PATTERNS.append(("we are ruined", "despair framing"))
EXCLUDE_PATTERNS.append(("his face darkens", "rebuke/negative framing"))
EXCLUDE_PATTERNS.append(("why do you laugh rather than weep", "rebuke framing"))
EXCLUDE_PATTERNS.append(("follow what is revealed to you", "prophet-specific address"))
EXCLUDE_PATTERNS.append(("aware as to who your enemies are", "enemy framing"))

# ── Run ────────────────────────────────────────────────────────────────────────
df = main(
    input_csv="/content/quran_emotion_tagged_clean.csv",
    output_csv="/content/quran_emotion_retagged_v3.csv",
    include_cutoff=0.22,   # lower slightly to rescue sad/heartbroken pools
    margin_cutoff=0.04,
)

# ── Hard EXCLUDE patterns ─────────────────────────────────────────────────────
# Ayahs matching ANY of these patterns are excluded regardless of score.
# These are semantically dangerous for a journaling / mental health context.
# Extend this list as you discover new patterns from testing.

EXCLUDE_PATTERNS: list[tuple[str, str]] = [
    # Battle / warfare commands
    ("fight", "battle/warfare command"),
    ("kill", "violence command"),
    ("slay", "violence command"),
    ("smite", "violence command"),
    ("strike their necks", "violence command"),
    ("cast terror", "terror/violence"),
    ("go forth", "battle mobilisation"),
    ("strive and struggle with", "battle framing"),

    # Punishment / threat framing
    ("hellfire", "punishment/threat"),
    ("terrible punishment", "punishment/threat"),
    ("painful punishment", "punishment/threat"),
    ("grievous punishment", "punishment/threat"),
    ("we will punish", "punishment/threat"),
    ("torment", "punishment/threat"),
    ("wrath of god", "divine wrath"),
    ("incurred your wrath", "divine wrath"),

    # Judgement Day / death framing
    ("day of judgement", "judgement day"),
    ("day of resurrection", "judgement day"),
    ("no soul shall bear the burden", "judgement day burden"),
    ("no bearer of a burden", "judgement day burden"),
    ("no human being shall be of the least avail", "judgement day isolation"),
    ("appointed time has come", "death/appointed time"),
    ("god will not grant a reprieve", "death framing"),
    ("wish for death", "death command"),

    # Disbeliever rebuke context
    ("those who disbelieve", "disbeliever rebuke"),
    ("those who deny the truth", "disbeliever rebuke"),
    ("bent on denying", "disbeliever rebuke"),
    ("hypocrites", "hypocrite rebuke"),
    ("sealed their hearts", "disbeliever description"),
    ("disease in their hearts", "disbeliever description"),
    ("hearts are diseased", "disbeliever description"),

    # Iblis / Satan narratives
    ("iblis", "iblis/satan narrative"),
    ("why did you not prostrate", "iblis/satan narrative"),
    ("not among those who have prostrated", "iblis/satan narrative"),

    # Adam expulsion narrative  
    ("go down from here as enemies", "expulsion narrative"),
    ("as enemies to one another", "expulsion narrative"),
    ("as enemies to each other", "expulsion narrative"),

    # Ritual/legal rulings (not comforting context)
    ("disperse in the land", "legal/ritual ruling"),
    ("when the prayer is ended", "legal/ritual ruling"),
    ("stand up praying for nearly", "legal/ritual ruling"),
    ("two-thirds of the night", "legal/ritual ruling"),
    ("pay the prescribed alms", "legal/ritual ruling"),
    ("freeing of slaves", "legal/ritual ruling"),
    ("prisoners of war", "legal/ritual ruling"),
    ("taken captive", "legal/ritual ruling"),

    # Muhammad-specific address (not general comfort)
    ("o muhammad", "prophet-specific"),
    ("bestow blessings on the prophet", "prophet-specific"),
    ("invoke blessings on him", "prophet-specific"),
    ("give good tidings [o", "prophet-specific"),

    # Geographic/ritual specifics
    ("sacred mosque", "ritual geography"),
    ("turn your face to the sacred", "ritual geography"),

    # Misc violent/disturbing
    ("slaying your sons", "violence/Pharaoh"),
    ("lest you stone", "stoning"),
    ("stone me to death", "stoning"),
    ("subjected you to grievous", "torture/violence"),
    ("he would not leave even one living creature", "punishment framing"),
    ("cannot defeat his purpose", "power threat"),
    ("bring in your place another people", "replacement threat"),
    ("he will bring in your place", "replacement threat"),
    ("blind in this life will be blind in the life to come", "threat"),
]

# ── Scoring ───────────────────────────────────────────────────────────────────

def check_hard_exclude(text: str) -> str | None:
    """Returns exclude_reason string if ayah matches any hard pattern, else None."""
    t = text.lower()
    for pattern, reason in EXCLUDE_PATTERNS:
        if pattern.lower() in t:
            return reason
    return None


def retag(df: pd.DataFrame, model: SentenceTransformer,
          include_cutoff: float = INCLUDE_CUTOFF,
          margin_cutoff: float = MARGIN_CUTOFF) -> pd.DataFrame:
    print("⏳ Encoding anchor sentences...")
    emotions = list(ANCHORS.keys())

    # One embedding per emotion = mean of all its anchor embeddings
    anchor_embeddings = {}
    for emotion, anchors in ANCHORS.items():
        embs = model.encode(anchors, convert_to_tensor=True, show_progress_bar=False)
        anchor_embeddings[emotion] = embs.mean(dim=0)  # centroid of emotion cluster

    anchor_matrix = torch.stack([anchor_embeddings[e] for e in emotions])
    print(f"  ✅ {len(emotions)} emotion centroids built")

    print(f"⏳ Encoding {len(df)} ayahs...")
    ayah_texts = df["ayah_en"].astype(str).tolist()
    ayah_embs  = model.encode(ayah_texts, convert_to_tensor=True,
                               batch_size=256, show_progress_bar=True)
    print("  ✅ Ayah encoding done")

    print("⏳ Computing similarities...")
    sims = util.cos_sim(ayah_embs, anchor_matrix).cpu().numpy()  # (N, num_emotions)

    labels, scores, reasons = [], [], []
    excluded_count = 0

    for i, text in enumerate(ayah_texts):
        # Step 1: hard exclude check (pattern-based)
        reason = check_hard_exclude(text)
        if reason:
            labels.append("EXCLUDE")
            scores.append(0.0)
            reasons.append(reason)
            excluded_count += 1
            continue

        row_sims = sims[i]
        best_idx   = int(np.argmax(row_sims))
        best_score = float(row_sims[best_idx])

        # Step 2: below confidence threshold → exclude
        if best_score < include_cutoff:
            labels.append("EXCLUDE")
            scores.append(best_score)
            reasons.append(f"low_confidence ({best_score:.3f} < {include_cutoff})")
            excluded_count += 1
            continue

        # Step 3: ambiguous (margin too small) → exclude
        sorted_sims = np.sort(row_sims)[::-1]
        margin = float(sorted_sims[0] - sorted_sims[1])
        if margin < margin_cutoff:
            labels.append("EXCLUDE")
            scores.append(best_score)
            reasons.append(f"ambiguous (margin {margin:.3f} < {margin_cutoff})")
            excluded_count += 1
            continue

        labels.append(emotions[best_idx])
        scores.append(best_score)
        reasons.append("")

    df = df.copy()
    df["emotion_v3"]       = labels
    df["emotion_score_v3"] = [round(s, 4) for s in scores]
    df["exclude_reason"]   = reasons

    print(f"\n✅ Retagging complete")
    print(f"   Total ayahs:  {len(df)}")
    print(f"   Excluded:     {excluded_count} ({excluded_count/len(df)*100:.1f}%)")
    print(f"   Usable pool:  {len(df) - excluded_count}")
    print(f"\n   Emotion distribution (usable only):")
    usable = df[df["emotion_v3"] != "EXCLUDE"]
    for emotion, count in usable["emotion_v3"].value_counts().items():
        print(f"     {emotion:<15} {count:>4}")

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anchor-based Quran emotion retagging")
    parser.add_argument("--input",   default=DEFAULT_INPUT,  help="Input CSV path")
    parser.add_argument("--output",  default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--cutoff",  type=float, default=INCLUDE_CUTOFF,
                        help=f"Min similarity to include (default {INCLUDE_CUTOFF})")
    parser.add_argument("--margin",  type=float, default=MARGIN_CUTOFF,
                        help=f"Min margin between top-2 emotions (default {MARGIN_CUTOFF})")
    parser.add_argument("--inspect", action="store_true",
                        help="After retagging, print 10 sample ayahs per emotion for review")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    print(f"📂 Loading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"   {len(df)} ayahs loaded")

    print(f"\n🤖 Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    df = retag(df, model, include_cutoff=args.cutoff, margin_cutoff=args.margin)

    output_path = Path(args.output)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n💾 Saved to: {output_path}")

    # Sanity check: known-bad ayahs should all be EXCLUDE
    bad_indices = [1833, 1961, 5847, 42, 100]
    print("\n🔍 Sanity check — known-bad ayahs:")
    for i in bad_indices:
        if i < len(df):
            row = df.iloc[i]
            tag    = row["emotion_v3"]
            score  = row["emotion_score_v3"]
            reason = row["exclude_reason"]
            icon   = "✅" if tag == "EXCLUDE" else "❌"
            print(f"  {icon} [{i}] {row['surah_name_roman']} {row['ayah_no_surah']}: "
                  f"{tag} ({score:.3f}) — {reason}")
            print(f"     {str(row['ayah_en'])[:80]}...")

    if args.inspect:
        print("\n\n📋 Sample inspection (10 per emotion):")
        usable = df[df["emotion_v3"] != "EXCLUDE"]
        for emotion in sorted(ANCHORS.keys()):
            pool = usable[usable["emotion_v3"] == emotion].nlargest(10, "emotion_score_v3")
            print(f"\n── {emotion.upper()} (top 10 by score) ──")
            for _, row in pool.iterrows():
                print(f"  [{row.name}] {row['emotion_score_v3']:.3f} "
                      f"{row['surah_name_roman']} {row['ayah_no_surah']}: "
                      f"{str(row['ayah_en'])[:80]}...")


if __name__ == "__main__":
    main()
