#  Quran-Powered Journaling App 

A simple web app that helps you reflect emotionally and spiritually by journaling. It uses AI to provide relevant Qur’anic ayahs, gentle comforting messages, and tracks your emotional state before and after.

> ⚠️ This is a **first prototype** built for feedback and learning purposes.  
> I’m actively working on developing a more polished and production-ready version as mobile app.


---

## ✨ Features

- 📝 Write journal entries freely
- 🌿 Get **relevant Qur’anic ayahs** in Arabic and English using semantic search
- 🤍 Receive **gentle, calming messages** powered by an LLM
- 🧠 Track your **emotions before and after reflection**
- 📜 View and manage your **journal history**
- 🗑️ Delete entries via a smooth confirmation modal

---

## Demo 



https://github.com/user-attachments/assets/e934c31d-4efe-44f5-a5b3-c902e3a06a95


---

## 🛠️ Tech Stack

| Layer       | Tech Used                                          |
|-------------|----------------------------------------------------|
| **Backend** | FastAPI, SentenceTransformers (`all-MiniLM-L6-v2`) |
| **LLM API** | OpenRouter + Mistral-7B                            |
| **Frontend**| HTML, CSS, Vanilla JavaScript                      |
| **Data**    | `quran_emotion_tagged.csv` (custom ayah dataset)  |

---

## 📦 Setup Instructions

After cloning and installing dependencies, create your own `.env` file with OpenRouter API details.

## 📂 Running the App Locally

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2.Create a .env file in the root directory:
```ini
OPENROUTER_API_KEY=your-api-key-here  
OPENROUTER_URL=https://openrouter.ai/api/v1/chat/completions  
GPT_MODEL=mistralai/mistral-7b-instruc
```
3.Run the backend:

```bash
uvicorn main:app --reload
```
4.Open the frontend:
- Open index.html directly in your browser.

## 📄 License:
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- This project is free to use and modify for non-commercial purposes with proper attribution.

