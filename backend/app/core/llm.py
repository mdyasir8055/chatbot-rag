import requests
from app.config import settings

def call_groq(prompt: str) -> str:
    """Call Groq LLM API for generating responses"""
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}
    payload = {"prompt": prompt, "max_tokens": 200}
    response = requests.post("https://api.groq.com/v1/completions", json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("text", "").strip()
    return "Error: LLM request failed."
