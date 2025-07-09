import os


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
GPT_MODEL = os.getenv("GPT_MODEL", "mistralai/mistral-7b-instruct")