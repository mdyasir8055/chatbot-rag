# Do_or_Die — AI Sales Agent (Free‑Tier)

PDF/URL/name → topic extraction → PDF‑first answer → Google CSE fallback → Groq synthesis.
Transparent tags: **(found in PDF)** or **(looked up on the web since not in PDF)**.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # fill keys
uvicorn app.main:app --reload --port ${PORT:-8000}
```

Open http://localhost:8000

## Environment
- `GROQ_API_KEY` — Groq key
- `GOOGLE_API_KEY` — Google Programmable Search API key
- `GOOGLE_CX` — CSE ID
- `PORT` — optional, default 8000

## Endpoints
- `POST /api/extract-topics` — (pdf|product_name) → topics
- `POST /api/ask` — (pdf|url|product_name) + question → answer + sources
