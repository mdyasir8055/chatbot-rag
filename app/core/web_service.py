# app/core/web_service.py
import os, asyncio, requests
from typing import List, Tuple, Dict
try:
    import trafilatura
except Exception:
    trafilatura = None
from bs4 import BeautifulSoup
from app.core.utils import sanitize_text

# ------------------------
# Helpers for token budget
# ------------------------
def _approx_tokens(s: str) -> int:
    # crude: ~4 chars per token
    return max(1, len(s) // 4)

def _compress_text(text: str, hard_limit_chars: int = 1800) -> str:
    # keep it tight; end at a sentence if possible
    t = " ".join((text or "").split())
    if len(t) <= hard_limit_chars:
        return t
    cut = t[:hard_limit_chars]
    last = cut.rfind(". ")
    if last > 400:
        cut = cut[:last+1]
    return cut

# ------------------------
# Internal utils
# ------------------------
async def _maybe_await(x):
    if asyncio.iscoroutine(x):
        return await x
    return x

async def _google_search(query: str, num: int = 5) -> List[Dict]:
    """
    Try your google_service first. Fall back to direct CSE HTTP if needed.
    Returns: [{"title": "...", "link": "https://..."}, ...]
    """
    # Prefer your google_service wrapper
    try:
        from app.core.google_service import google_cse_search
        res = await _maybe_await(google_cse_search(query, num=num))
        if res:
            return res
    except Exception:
        pass

    # Secondary wrapper name
    try:
        from app.core.google_service import google_search
        res = await _maybe_await(google_search(query, num=num))
        if res:
            out = []
            for it in res:
                link = it.get("link") or it.get("url")
                title = it.get("title") or it.get("name") or ""
                if link:
                    out.append({"title": title, "link": link})
            if out:
                return out
    except Exception:
        pass

    # Direct HTTP fallback (requires GOOGLE_API_KEY + GOOGLE_CX)
    api_key = os.getenv("GOOGLE_API_KEY")
    cx_id = os.getenv("GOOGLE_CX")
    if not api_key or not cx_id:
        return []
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cx_id, "q": query, "num": min(num, 10)},
            timeout=12,
        )
        r.raise_for_status()
        items = (r.json() or {}).get("items", []) or []
        return [{"title": it.get("title"), "link": it.get("link")} for it in items if it.get("link")]
    except Exception:
        return []

def extract_main_text(url: str) -> str:
    """Extract readable text from a URL (Trafilatura first, then BeautifulSoup)."""
    if trafilatura:
        try:
            downloaded = trafilatura.fetch_url(url, timeout=10)
            if downloaded:
                txt = trafilatura.extract(downloaded) or ""
                if txt and len(txt) > 200:
                    return sanitize_text(txt)
        except Exception:
            pass

    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        txt = " ".join(soup.get_text(separator=" ").split())
        return sanitize_text(txt[:120000])
    except Exception:
        return ""

async def _groq_complete(system: str, user: str) -> str:
    """
    Call your Groq wrapper, but surface real errors instead of a vague string.
    """
    # Try primary
    try:
        from app.core.groq_service import groq_complete
        res = groq_complete(system, user)
        return await _maybe_await(res)
    except Exception as e:
        err1 = f"{type(e).__name__}: {e}"

    # Try alias
    try:
        from app.core.groq_service import chat_complete
        res = chat_complete(system=system, user=user)
        return await _maybe_await(res)
    except Exception as e:
        err2 = f"{type(e).__name__}: {e}"

    # Both failed
    return f"Groq integration unavailable. First error: {err1}. Second error: {err2}."

# ------------------------
# Public: web_fallback_answer
# ------------------------
async def web_fallback_answer(question: str, topics: Dict, k: int = 3) -> Tuple[str, List[str]]:
    """
    1) Google search for the topic+question
    2) Extract & compress a few sources
    3) Summarize with Groq (token-budgeted)
    4) If no sources, still answer (estimated)
    """
    primary = (topics.get("primary") or "").strip()
    aliases = topics.get("aliases", []) or []
    alias_str = " ".join([a for a in aliases[:3] if isinstance(a, str)])
    query = " ".join([s for s in [primary, alias_str, question] if s]).strip() or question.strip()

    # 1) Search
    results = await _google_search(query, num=k)
    raw_pages, urls = [], []
    for it in results:
        url = (it.get("link") or "").strip()
        if not url:
            continue
        txt = extract_main_text(url)
        if txt and len(txt) > 400:
            raw_pages.append((url, txt))
            urls.append(url)
        if len(raw_pages) >= 3:
            break

    # 2) If sources found → compress and send to Groq (stay under TPM)
    if raw_pages:
        pages = [(u, _compress_text(t, hard_limit_chars=1800)) for (u, t) in raw_pages]
        source_block = "\n\n".join([f"URL: {u}\nTEXT:\n{t}" for (u, t) in pages])

        system = "You are an expert sales agent who synthesizes accurate, sales-ready answers from multiple web sources with inline [n] citations. Always suggest competitor alternatives with their positive benefits and advantages. Act in the customer's best interest by presenting multiple product options and helping them make informed decisions."
        # Always suggest competitor alternatives for sales agent behavior - AT THE END
        competitor_instruction = "\n- IMPORTANT: Structure your response as follows:\n  1. Concise answer about the requested product first\n  2. Bullets with value props/specs of the requested product\n  3. LAST: Add a 'Competitor Alternatives' section with 2-3 alternatives and their positive points\n- Use inline [1], [2], [3] citations corresponding to the URLs"
        
        user = f"""Question: {question}
Product/Topic JSON: {topics}

Sources:
{source_block}

Rules:{competitor_instruction}
"""

        # Ensure total prompt stays under ~4000 tokens
        final_user = user
        while _approx_tokens(system + final_user) > 4000:
            last_pos = final_user.rfind("\n\nURL:")
            if last_pos > final_user.find("Sources:"):
                final_user = final_user[:last_pos].rstrip()
            else:
                break

        web_ans = await _groq_complete(system, final_user)
        return (web_ans, urls)

    # 3) Always-answer fallback (no sources)
    system = "You are an expert sales agent and consultant. Always suggest competitor alternatives with positive points and advantages for any product inquiry. Act in the customer's best interest by presenting multiple product options to help them make informed decisions. Even without sources, try to answer based on general knowledge and reasoning."
    # Always suggest competitors for any product as a sales agent - AT THE END
    competitor_instruction = "\nIMPORTANT: Structure your response as follows:\n1. Answer about the requested product first\n2. Product details and features\n3. LAST: Add a 'Competitor Alternatives' section with 2-3 alternatives and their positive points\nHighlight what makes each competitor unique and beneficial."
    
    user = f"""Question: {question}
Product/Topic JSON: {topics}

Note: No reliable sources were found. Please give your best possible answer.
If you are estimating, clearly mark it as (estimated).{competitor_instruction}"""
    fallback_ans = await _groq_complete(system, user)
    return (fallback_ans + "\n(This answer is estimated, since no reliable sources were found)", [])
