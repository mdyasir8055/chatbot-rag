# app/core/ocr_service.py
from __future__ import annotations
from typing import Tuple, List
import os
import io
import re

from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

# --- Optional Windows/macOS/Linux path configuration via environment ---
# Set these in your .env or environment variables if Poppler/Tesseract aren't on PATH:
#   POPPLER_PATH=C:\\tool\\poppler-25.07.0\\Library\\bin
#   TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
POPPLER_PATH = os.getenv("POPPLER_PATH")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


from pypdf import PdfReader


def _extract_plain_text_len(pdf_bytes: bytes) -> Tuple[int, int]:
    """Return (total_text_len, num_pages) using PyPDF text extraction."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total = 0
        for page in reader.pages:
            txt = page.extract_text() or ""
            total += len(txt.strip())
        return total, len(reader.pages)
    except Exception:
        return 0, 0


def is_scanned(pdf_bytes: bytes, min_chars_per_page: int = 50) -> bool:
    """
    Heuristic: if average extracted chars/page is very low, treat as scanned.
    """
    total, pages = _extract_plain_text_len(pdf_bytes)
    if pages == 0:
        return True  # unreadable via pypdf → likely scanned or malformed
    avg = total / max(1, pages)
    return avg < min_chars_per_page


def _clean_ocr_text(text: str) -> str:
    # basic cleanup: de-hyphenation at line breaks, collapse whitespace
    text = re.sub(r"-\s*\n\s*", "", text)       # dehyphenate line breaks
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def ocr_pdf(pdf_bytes: bytes, dpi: int = 300, lang: str = "eng") -> str:
    """
    Page-wise OCR using Tesseract. Requires:
      - Tesseract installed on the OS
      - Poppler (for pdf2image rasterization)
    """
    # rasterize pages
    images: List[Image.Image] = convert_from_bytes(pdf_bytes, dpi=dpi, poppler_path=POPPLER_PATH or None)
    out_pages = []
    for img in images:
        txt = pytesseract.image_to_string(img, lang=lang) or ""
        out_pages.append(_clean_ocr_text(txt))
    return "\n\n".join(out_pages).strip()
