# app/core/groq_service.py
import os, time, requests

BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"
HTTP_TIMEOUT = 20

def _get_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")
    return api_key

def _post_chat(messages, model=DEFAULT_MODEL, temperature=0.2, max_tokens=500) -> str:
    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=HTTP_TIMEOUT)

    if resp.status_code != 200:
        txt = (resp.text or "").strip()
        if resp.status_code in (413, 429):
            # Token-per-minute or rate/size issues
            raise RuntimeError(f"TPM/Rate limit: {resp.status_code} {txt[:240]}")
        if resp.status_code == 401:
            raise RuntimeError("Groq 401 Unauthorized: invalid or missing GROQ_API_KEY.")
        if resp.status_code >= 500:
            raise RuntimeError(f"Groq {resp.status_code} server error. Body: {txt[:240]}")
        raise RuntimeError(f"Groq {resp.status_code} error. Body: {txt[:240]}")

    data = resp.json()
    return (data["choices"][0]["message"]["content"] or "").strip()

def groq_complete(system: str, user: str) -> str:
    """Single-turn completion pinned to llama-3.1-8b-instant with 1 retry on TPM/Rate."""
    messages = [
        {"role": "system", "content": system or ""},
        {"role": "user", "content": user or ""},
    ]
    try:
        return _post_chat(messages, model=DEFAULT_MODEL, temperature=0.3, max_tokens=500)
    except RuntimeError as e:
        msg = str(e)
        if "TPM/Rate limit" in msg:
            time.sleep(3)
            try:
                return _post_chat(messages, model=DEFAULT_MODEL, temperature=0.2, max_tokens=300)
            except Exception as e2:
                return f"Groq error after retry: {e2}"
        return f"Groq error: {e}"
    except Exception as e:
        return f"Groq error: {e}"

def chat_complete(system: str, user: str) -> str:
    # Backwards-compatible alias
    return groq_complete(system, user)
