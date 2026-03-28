"""
مع القرآن — API Test Runner
Run this from anywhere: python test_quran_api.py
Results saved to: quran_test_results.json
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"  # change if needed
results = []

def run(label, method, path, body=None, expect_status=200, note=""):
    url = BASE_URL + path
    try:
        if method == "GET":
            res = requests.get(url, timeout=30)
        elif method == "POST":
            res = requests.post(url, json=body, timeout=30)
        elif method == "DELETE":
            res = requests.delete(url, timeout=30)

        try:
            data = res.json()
        except:
            data = {}

        passed = res.status_code == expect_status
        result = {
            "label": label,
            "note": note,
            "method": method,
            "path": path,
            "status": res.status_code,
            "expected_status": expect_status,
            "passed": passed,
            "response": data
        }

        icon = "✅" if passed else "❌"
        print(f"{icon} [{res.status_code}] {label}")
        if not passed:
            print(f"   Expected {expect_status}, got {res.status_code}")
        elif method == "POST" and path == "/match-ayahs":
            emotion = data.get("emotion_before", "?")
            matches = data.get("matches", [])
            print(f"   Emotion: {emotion}")
            for i, m in enumerate(matches):
                print(f"   {i+1}. {m.get('surah')} {m.get('ayah_no')}: {m.get('ayah','')[:80]}...")
            comfort = data.get("comfort", "")
            print(f"   Comfort: {comfort[:100]}...")

    except Exception as e:
        result = {
            "label": label,
            "note": note,
            "method": method,
            "path": path,
            "passed": False,
            "error": str(e)
        }
        print(f"❌ {label} — ERROR: {e}")

    results.append(result)
    return result.get("response", {})

print("\n" + "="*60)
print("  مع القرآن — API Test Runner")
print("="*60 + "\n")

# ── HEALTH ──────────────────────────────────────────────────────
print("── HEALTH ──")
run("Health check", "GET", "/health", note="Server up + ayahs loaded")
print("\n── MATCH AYAHS ──")

# Core emotions (clear, simple cases)
run(
    "Emotion: sad — clear grief",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I have been crying all day and feel so empty inside like nothing matters anymore",
        "top_n": 3,
    },
)

run(
    "Emotion: stressed — exams",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I'm so stressed about my exams I haven't slept properly in days and I feel like I'm going to fail everything",
        "top_n": 3,
    },
)

run(
    "Emotion: grateful — blessings",
    "POST",
    "/match-ayahs",
    body={
        "entry": "Alhamdulillah I just got news that made me realize how blessed I truly am, so much gratitude in my heart",
        "top_n": 3,
    },
)

run(
    "Emotion: lonely — unseen",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I feel completely alone, nobody checks on me, nobody sees me, I'm invisible to everyone around me",
        "top_n": 3,
    },
)

run(
    "Emotion: angry — injustice",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I'm so angry, it's not fair, I've been treated with such injustice and I can't let it go",
        "top_n": 3,
    },
)

run(
    "Emotion: tired — burnout",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I'm exhausted, completely drained, I have nothing left to give, I can't go on like this anymore",
        "top_n": 3,
    },
)

run(
    "Emotion: heartbroken — betrayal",
    "POST",
    "/match-ayahs",
    body={
        "entry": "My heart is broken, I was betrayed by someone I trusted completely, I can't forgive and can't let go",
        "top_n": 3,
    },
)

run(
    "Emotion: hopeful — future",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I feel hopeful today, I really believe things are going to get better inshallah, making dua and trusting Allah",
        "top_n": 3,
    },
)

run(
    "Emotion: confused — life direction",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I don't know which direction to take in life, I feel so lost and unclear about everything, what should I do",
        "top_n": 3,
    },
)

run(
    "Emotion: peaceful — calm contentment",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I feel calm and at peace today, my heart feels settled and I'm just grateful for everything around me",
        "top_n": 3,
    },
)

# Darker / safety‑critical cases
run(
    "Emotion: despair — passive suicidal",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I don't want to exist anymore, I feel like everyone would be better off without me, life feels pointless",
        "top_n": 3,
    },
)

run(
    "Emotion: self‑blame — guilt and shame",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I hate myself for the mistakes I've made, I feel like Allah will never forgive me and I don't deserve mercy",
        "top_n": 3,
    },
)

run(
    "Emotion: anger at Allah",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I'm angry with Allah, I don't understand why He keeps testing me like this, it feels so unfair",
        "top_n": 3,
    },
)

run(
    "Emotion: trauma flashback",
    "POST",
    "/match-ayahs",
    body={
        "entry": "Memories keep coming back from what happened, I feel scared in my own body and I can't sleep at night",
        "top_n": 3,
    },
)

# Mixed / ambiguous feelings
run(
    "Emotion: mixed — sad and grateful",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I'm sad about what I lost, but also strangely grateful that Allah protected me from what I couldn't see",
        "top_n": 3,
    },
)

run(
    "Emotion: mixed — stressed and hopeful",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I'm stressed and overwhelmed with responsibilities, but deep down I still have hope that Allah will make a way out",
        "top_n": 3,
    },
)

# Edge language / slang / short entries
run(
    "Emotion: short — one‑liner sad",
    "POST",
    "/match-ayahs",
    body={
        "entry": "Today just hurt.",
        "top_n": 3,
    },
)

run(
    "Emotion: slang — anxious",
    "POST",
    "/match-ayahs",
    body={
        "entry": "Lowkey freaking out about everything, my brain won't shut up and my heart is racing",
        "top_n": 3,
    },
)

run(
    "Emotion: vague but heavy",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I don't even know what I'm feeling anymore, it's just this heavy cloud over me all the time",
        "top_n": 3,
    },
)

# LLM / fallback behaviour
run(
    "LLM fallback — super vague",
    "POST",
    "/match-ayahs",
    body={
        "entry": "Today was a day. I just sat there. Things happened. I don't know.",
        "top_n": 3,
    },
    note="No strong emotion words — should fall back to generic comfort / safe verses",
)

# top_n behaviour
run(
    "top_n: 1 (sad)",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I feel sad and lost today",
        "top_n": 1,
    },
)

run(
    "top_n: 5 (stressed)",
    "POST",
    "/match-ayahs",
    body={
        "entry": "I feel anxious and stressed about everything in my life right now",
        "top_n": 5,
    },
)# ── UPDATE EMOTION ───────────────────────────────────────────────
print("\n── UPDATE EMOTION ──")
run("Update emotion — valid", "POST", "/update-emotion",
    body={"entry_id": 0, "emotion_after": "peaceful"})

run("Update emotion — invalid ID", "POST", "/update-emotion",
    body={"entry_id": 9999, "emotion_after": "happy"},
    expect_status=404)

# ── HISTORY ─────────────────────────────────────────────────────
print("\n── HISTORY ──")
run("Get history", "GET", "/history")
run("Delete entry — valid", "DELETE", "/delete-entry/0")
run("Delete entry — invalid", "DELETE", "/delete-entry/9999", expect_status=404)

# ── BOOKMARKS ───────────────────────────────────────────────────
print("\n── BOOKMARKS ──")
run("Add bookmark with note", "POST", "/bookmark",
    body={"ayah_index": 42, "note": "This one really hit me"})

run("Duplicate bookmark", "POST", "/bookmark",
    body={"ayah_index": 42, "note": "another note"},
    note="Should not duplicate")

run("Bookmark — no note", "POST", "/bookmark",
    body={"ayah_index": 100})

run("Bookmark — invalid index", "POST", "/bookmark",
    body={"ayah_index": -1},
    expect_status=400)

run("Get bookmarks", "GET", "/bookmarks")
run("Remove bookmark — valid", "DELETE", "/bookmark/42")
run("Remove bookmark — not found", "DELETE", "/bookmark/9999", expect_status=404)

# ── PATTERN ─────────────────────────────────────────────────────
print("\n── PATTERN ──")
run("Get pattern", "GET", "/pattern")

# ── REFLECT AGAIN ───────────────────────────────────────────────
print("\n── REFLECT AGAIN ──")
run("Reflect again — valid", "POST", "/reflect-again/1?top_n=3")
run("Reflect again — invalid", "POST", "/reflect-again/9999", expect_status=404)

# ── SAVE RESULTS ────────────────────────────────────────────────
out = {
    "timestamp": datetime.now().isoformat(),
    "base_url": BASE_URL,
    "total": len(results),
    "passed": sum(1 for r in results if r.get("passed")),
    "failed": sum(1 for r in results if not r.get("passed")),
    "results": results
}

with open("quran_test_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("\n" + "="*60)
print(f"  Total:  {out['total']}")
print(f"  Passed: {out['passed']} ✅")
print(f"  Failed: {out['failed']} ❌")
print(f"\n  Results saved to: quran_test_results.json")
print("="*60 + "\n")

# ── MATCH AYAHS ─────────────────────────────────────────────────
