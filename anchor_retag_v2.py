
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

# ── Anchor Sentences ──────────────────────────────────────────────────────────
# These are the SPECIFICATION of what each emotion means in this app.
# Each anchor = "an ayah I would want to read if I felt this emotion."
# Multiple anchors per emotion = multiple facets of what comfort looks like.
#
# Tuning guide:
#   - Add anchors if a real ayah keeps getting wrong label
#   - Remove anchors if a pool is over-assigning (too many ayahs, wrong quality)
#   - Anchors should sound like Quranic themes, not user journal entries

ANCHORS: dict[str, list[str]] = {

    "sad": [
        "Allah is close to those whose hearts are broken and heavy with sorrow",
        "He knows your grief and your tears are not wasted — ease will come after hardship",
        "Do not lose hope in the mercy of Allah; He is the Most Merciful",
        "We do indeed know how your heart is distressed — you are not alone in your pain",
        "Your suffering reaches Allah and He is near to those who are patient in grief",
        "After every hardship Allah brings relief; He does not burden a soul beyond its capacity",
    ],

    "anxious": [
        "Do not fear, for Allah is with you; He will calm your heart and ease your worry",
        "Put your trust in Allah — He is sufficient for the one who trusts in Him",
        "Verily with hardship comes ease; Allah does not abandon those who call on Him",
        "Allah calms the hearts of believers with His remembrance — seek refuge in Him",
        "Do not grieve — truly Allah is with those who are patient and trust His plan",
        "Cast your burden on Allah; He knows what is hidden and what is manifest",
    ],

    "stressed": [
        "Indeed with hardship there is ease — Allah promises relief to those who are burdened",
        "Allah does not burden a soul beyond what it can bear — He is gentle with His servants",
        "When you are overwhelmed, turn to Allah alone; He is the reliever of distress",
        "Seek help through patience and prayer — Allah is with those who are steadfast",
        "Your Lord has not forsaken you; relief is coming even when the burden feels unbearable",
        "Allah is the best of planners; trust Him when you cannot see the way forward",
    ],

    "lonely": [
        "We are nearer to him than his jugular vein — Allah is always with you, closer than you know",
        "When you feel unseen and alone, know that Allah sees you and is never far",
        "Allah is the companion of those who have no companion — you are never truly alone",
        "Call upon Me and I will respond to you — Allah hears you even in your loneliest moment",
        "He is with you wherever you are; His presence fills every place of solitude",
        "Even when no one else sees your pain, Allah sees it and He is sufficient for you",
    ],

    "heartbroken": [
        "Allah is with those whose hearts are broken by betrayal and loss — He heals what is shattered",
        "Do not despair — Allah is the mender of broken hearts and the restorer of hope",
        "Your pain from being hurt and betrayed is seen by Allah who is the Most Just",
        "Turn to Allah when your heart is broken; He is the only one who can truly heal you",
        "Allah does not let sincere love go to waste; He sees the hurt in your heart",
        "After betrayal and heartbreak, Allah remains — He is nearer than those who left",
    ],

    "angry": [
        "Those who control their anger and pardon others — Allah loves those who do good",
        "Seek refuge in Allah from anger — He is the patient one and rewards patience",
        "Do not let injustice make you transgress; Allah sees everything and will judge rightly",
        "Return evil with good; Allah is with those who are patient in the face of wrong",
        "Respond to harm with forgiveness — the patient shall inherit great reward",
        "Allah is just and He does not let injustice go unaddressed — trust His judgment",
    ],

    "tired": [
        "Allah does not burden a soul beyond what it can bear — rest is permitted in His mercy",
        "Even the prophets grew weary; be gentle with yourself — Allah sees your exhaustion",
        "Your Lord knows your fatigue; He does not waste the effort of those who strive",
        "Come to Me all who are weary and burdened — Allah's mercy restores the tired soul",
        "Allah is gentle with those who are drained; your weakness does not diminish your worth",
        "He gives rest to those who are exhausted and strength to those who have none left",
    ],

    "hopeful": [
        "Do not despair of the mercy of Allah — He forgives all sins and brings what is good",
        "Indeed with difficulty comes ease — Allah's promise is true and His timing is perfect",
        "Allah answers the call of the one who calls on Him in hope and trust",
        "Your Lord will give you and you will be satisfied — trust in His generous plan",
        "Whoever trusts in Allah, He is sufficient for them — beautiful outcomes await",
        "The believer's hope in Allah is never wasted; He is the best of those relied upon",
    ],

    "grateful": [
        "If you are grateful, I will surely increase you in blessings — Allah rewards thankfulness",
        "Remember Me and I will remember you; give thanks to Me and do not be ungrateful",
        "All blessings come from Allah alone — count His favours and you cannot number them",
        "He who is grateful, his gratitude is for his own soul — thankfulness brings more",
        "Alhamdulillah — all praise belongs to Allah who created and blessed and guided us",
        "This blessing is from My Lord to test whether I am grateful — and gratitude multiplies",
    ],

    "confused": [
        "Guide us to the straight path — Allah alone gives clarity to those who are lost",
        "Allah is the light of the heavens and the earth; He guides whom He wills to His light",
        "Put your trust in Allah and He will clarify your affairs and guide your heart",
        "Ask Allah for guidance — He answers the one who is confused and seeks His direction",
        "When you do not know which way to go, pray Istikhara and trust Allah's wisdom",
        "Your Lord has not forsaken you — He will guide you even through the deepest confusion",
    ],

    "peaceful": [
        "Verily in the remembrance of Allah do hearts find rest and tranquility",
        "Those who believe and whose hearts find rest in the remembrance of Allah",
        "Allah grants peace and contentment to the hearts of those who trust in Him",
        "The righteous shall have no fear nor shall they grieve — they are at peace with Allah",
        "Peace be upon you — you are in the care of the Most Merciful, the Most Peaceful",
        "He it is who sent down serenity into the hearts of the believers to increase their faith",
    ],

    "happy": [
        "This is from the bounty of my Lord — joy and happiness are gifts from Allah",
        "Allah makes the righteous glad and fills their hearts with joy and contentment",
        "The believers shall have glad tidings — their happiness is multiplied by Allah",
        "Say: In the bounty of Allah and His mercy — in that let them rejoice",
        "Alhamdulillah for this joy — every good thing comes from Allah and returns to Him",
        "Those who believe and do righteous deeds — for them is happiness and a beautiful return",
    ],

    "content": [
        "Whoever is content with what Allah has given him will find true peace and sufficiency",
        "Allah is pleased with those who are pleased with His decree — this is the highest station",
        "True richness is the richness of the soul, not of possessions — contentment is a gift",
        "Be satisfied with what Allah has apportioned for you — He gives what is best",
        "The one who trusts Allah's plan and accepts His decree lives with a contented heart",
        "Allah is sufficient for the one who is content with Him — He is the best provider",
    ],

    "reflective": [
        "Do they not reflect? — Allah invites those who think deeply to understand His signs",
        "In the creation of the heavens and the earth there are signs for those who reflect",
        "Those who remember Allah standing, sitting, and lying down and reflect on His creation",
        "Perhaps you will reflect — life's moments are invitations to return to deeper meaning",
        "He gives wisdom to whom He wills — the one who ponders gains understanding",
        "Look and reflect on the signs around you — Allah reveals Himself to those who contemplate",
    ],
}

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
