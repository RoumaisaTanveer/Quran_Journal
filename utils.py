import requests

from settings import OPENROUTER_API_KEY, OPENROUTER_URL, GPT_MODEL

def call_openrouter(system: str, user: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    }

    try:
        res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        # Log full response if missing keys
        if "choices" not in data or "message" not in data["choices"][0]:
            print("⚠️ Unexpected OpenRouter response:", data)
            return ""

        return data["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        print("❌ OpenRouter error:", e)
        return ""

def detect_emotion(entry: str) -> str:
    system = (
        "You are an emotion classifier. Your job is to detect the dominant emotion in a journal entry. "
        "Respond with exactly ONE WORD (lowercase), and choose only from this list: "
        "sad, anxious, hopeful, grateful, angry, stressed, tired, peaceful, confused, happy, lonely, heartbroken, content, reflective. "
        "Do NOT explain. Do NOT use any other words."
    )
    user = f"Emotion in journal entry:\n{entry}"

    raw = call_openrouter(system, user)
    print("🧪 Raw emotion response:", repr(raw))  #  Log the raw response for error detection

    if not raw:
        return "unsure"

    first_word = raw.strip().lower().split()[0]
    valid_emotions = {
        "sad", "anxious", "hopeful", "grateful", "angry", "stressed", "tired",
        "peaceful", "confused", "happy", "lonely", "heartbroken", "content", "reflective"
    }

    if first_word in valid_emotions:
        print("✅ Detected emotion:", first_word)
        return first_word
    else:
        print("⚠️ Emotion not in valid list:", first_word)
        return "unsure"
    

def generate_comfort_message(entry: str) -> str:
    system = ("You're a soft, kind Islamic companion who replies to emotional journal entries in simple, warm words. "
              "Use short sentences. Avoid poetry or deep metaphors. No quotes, no 'I'm sorry to hear'. Just calming, real advice, like a good friend.")
    user = f"My journal entry: {entry}\n\nWrite a 2–3 line comforting message."
    return call_openrouter(system, user) or "You're doing better than you think. One step at a time is still progress. 🤍"