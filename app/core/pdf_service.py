import re
from typing import Optional, Dict, List
from io import BytesIO
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

def extract_pdf_text(file_bytes: Optional[bytes], max_pages: int = 6, max_chars: int = 40000) -> str:
    if not file_bytes or not PdfReader:
        return ""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        chunks = []
        for page in reader.pages[:max_pages]:
            txt = page.extract_text() or ""
            txt = re.sub(r"\s+", " ", txt)
            chunks.append(txt.strip())
        text = "\n".join(chunks)
        return text[:max_chars]
    except Exception:
        return ""

_BRAND_GENERIC = re.compile(r"([A-Z][A-Za-z0-9\-]+)\s*\(([^)]+)\)")
_TM = re.compile(r"\b([A-Z][A-Za-z0-9\-]+)(?:®|™)\b")
_TITLE = re.compile(r"([A-Z][A-Za-z0-9]+(?:\s[A-Z0-9][A-Za-z0-9\-]+){0,4})")
_PROPER = re.compile(r"\b[A-Z][A-Za-z0-9\-]{2,}(?:\s[A-Z0-9][A-Za-z0-9\-]{2,}){0,3}\b")
_STOP_PHRASES = {"Limited","Warranty","Manual","Guide","User","Instructions"}

def _score_candidates(text: str, candidates: List[str]) -> str:
    freq = {}
    window = text[:2000]
    toks = _PROPER.findall(window)
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    for c in candidates:
        if c not in freq:
            freq[c] = 1
    ranked = sorted(candidates, key=lambda x: (-freq.get(x,0), -len(x)))
    return ranked[0] if ranked else "Unknown Product"

def extract_topics_heuristic(text: str, user_name_hint: Optional[str] = None) -> Dict[str, Optional[str]]:
    candidates = []
    text_first = text[:3000]
    for m in _BRAND_GENERIC.finditer(text_first):
        brand, generic = m.group(1).strip(), m.group(2).strip()
        if brand not in candidates: candidates.append(brand)
        if generic not in candidates: candidates.append(generic)
    for m in _TM.finditer(text_first):
        t = m.group(1).strip()
        if t not in candidates: candidates.append(t)
    first_lines = "\n".join(text.splitlines()[:20])
    for t in _TITLE.findall(first_lines):
        if len(t.split()) <= 5 and not any(s in t for s in _STOP_PHRASES):
            t = t.strip()
            if t not in candidates: candidates.append(t)
    if user_name_hint:
        hint = user_name_hint.strip()
        if hint and hint not in candidates:
            candidates.append(hint)
    primary = _score_candidates(text, candidates) if (text or candidates) else (user_name_hint or "Unknown Product")
    mg = _BRAND_GENERIC.search(text_first)
    generic = mg.group(2).strip() if mg else None
    seen = set([primary])
    aliases = []
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        aliases.append(c)
        if len(aliases) >= 6:
            break
    return {"primary": primary,"aliases": aliases,"company": None,"generic_name": generic}
