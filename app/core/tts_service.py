# app/core/tts_service.py
"""Lightweight backend TTS using pyttsx3 (offline). Produces WAV bytes.
Install dependency:
    pip install pyttsx3
On Windows, pyttsx3 uses SAPI5 (works out of the box). On Linux, ensure 'espeak' is available.
"""
from __future__ import annotations
import io
from typing import Optional, List, Dict

try:
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None  # Optional dependency


_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is not installed. Run: pip install pyttsx3")
        _engine = pyttsx3.init()
    return _engine

def list_voices() -> List[Dict[str, str]]:
    """Return available system voices."""
    eng = _get_engine()
    out = []
    for v in eng.getProperty('voices'):
        out.append({
            "id": v.id,
            "name": getattr(v, 'name', ''),
            "languages": getattr(v, 'languages', []).__str__(),
            "gender": getattr(v, 'gender', ''),
            "age": str(getattr(v, 'age', '')),
        })
    return out

def synthesize_wav(text: str, voice_id: Optional[str] = None, rate: Optional[int] = None, volume: Optional[float] = None) -> bytes:
    """Synthesize text to WAV and return raw bytes."""
    if not text or not text.strip():
        raise ValueError("Text is empty")

    eng = _get_engine()

    # Set properties
    if voice_id:
        try:
            eng.setProperty('voice', voice_id)
        except Exception:
            pass
    if rate is not None:
        try:
            eng.setProperty('rate', int(rate))
        except Exception:
            pass
    if volume is not None:
        try:
            eng.setProperty('volume', float(volume))
        except Exception:
            pass

    # pyttsx3 writes to file path; use NamedTemporaryFile pattern
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
        temp_path = tf.name

    try:
        eng.save_to_file(text, temp_path)
        eng.runAndWait()
        with open(temp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
