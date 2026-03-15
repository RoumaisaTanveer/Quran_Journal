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

# ── MATCH AYAHS ─────────────────────────────────────────────────
print("\n── MATCH AYAHS ──")

r = run("Emotion: sad", "POST", "/match-ayahs",
    body={"entry": "I have been crying all day and feel so empty inside like nothing matters anymore", "top_n": 3})

run("Emotion: stressed", "POST", "/match-ayahs",
    body={"entry": "I'm so stressed about my exams I haven't slept properly in days and I feel like I'm going to fail everything", "top_n": 3})

run("Emotion: grateful", "POST", "/match-ayahs",
    body={"entry": "Alhamdulillah I just got news that made me realize how blessed I truly am so much gratitude in my heart", "top_n": 3})

run("Emotion: lonely", "POST", "/match-ayahs",
    body={"entry": "I feel completely alone nobody checks on me nobody sees me I'm invisible to everyone around me", "top_n": 3})

run("Emotion: angry", "POST", "/match-ayahs",
    body={"entry": "I'm so angry it's not fair I've been treated with such injustice and I can't let it go", "top_n": 3})

run("Emotion: tired", "POST", "/match-ayahs",
    body={"entry": "I'm exhausted completely drained I have nothing left to give I can't go on like this anymore", "top_n": 3})

run("Emotion: heartbroken", "POST", "/match-ayahs",
    body={"entry": "My heart is broken I was betrayed by someone I trusted completely I can't forgive and can't let go", "top_n": 3})

run("Emotion: hopeful", "POST", "/match-ayahs",
    body={"entry": "I feel hopeful today I really believe things are going to get better inshallah making dua and trusting Allah", "top_n": 3})

run("Emotion: confused", "POST", "/match-ayahs",
    body={"entry": "I don't know which direction to take in life I feel so lost and unclear about everything what to do", "top_n": 3})

run("Emotion: peaceful", "POST", "/match-ayahs",
    body={"entry": "I feel calm and at peace today my heart feels settled and I'm just grateful for everything around me", "top_n": 3})

run("LLM fallback — vague", "POST", "/match-ayahs",
    body={"entry": "Today was a day. I just sat there. Things happened. I don't know.", "top_n": 3},
    note="No keywords — should trigger LLM")

run("top_n: 1", "POST", "/match-ayahs",
    body={"entry": "I feel sad and lost today", "top_n": 1},
    note="Should return exactly 1 match")

run("top_n: 5", "POST", "/match-ayahs",
    body={"entry": "I feel anxious and stressed about everything in my life right now", "top_n": 5},
    note="Should return exactly 5 matches")

# ── UPDATE EMOTION ───────────────────────────────────────────────
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
