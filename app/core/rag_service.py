# app/core/rag_service.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np

# Lightweight local embedding model
# Install: sentence-transformers
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks for retrieval.
    """
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    chunks, i = [], 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += step
    return chunks


class RagIndex:
    """
    Very small local vector index. No external services required.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.vecs: np.ndarray | None = None

    def build(self, text: str):
        self.chunks = chunk_text(text, 800, 150)
        if not self.chunks:
            self.vecs = None
            return
        embs = self.model.encode(
            self.chunks,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        self.vecs = embs.astype("float32")

    def retrieve(self, query: str, k: int = 8) -> List[Tuple[str, float]]:
        if self.vecs is None or not self.chunks:
            return []
        q = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0].astype("float32")
        sims = self.vecs @ q  # cosine similarity (vecs normalized)
        idx = np.argsort(sims)[-k:][::-1]
        return [(self.chunks[i], float(sims[i])) for i in idx]


def rag_knows(retrieved: List[Tuple[str, float]]) -> bool:
    """
    Deterministic gates (as specced):
    - >=2 chunks
    - combined evidence >= 250 tokens (approx via words)
    - margin focus: top - median >= 0.05
    """
    if len(retrieved) < 2:
        return False

    combined_len = sum(len(c[0].split()) for c in retrieved[:3])
    if combined_len < 250:
        return False

    scores = [s for _, s in retrieved]
    top, median = scores[0], float(np.median(scores))
    if (top - median) < 0.05:
        return False

    return True


def synthesize_from_chunks(question: str, chunks: List[str]) -> str:
    """
    Synthesize a cautious, doc-grounded answer using the LLM.
    This calls the project's Groq wrapper for consistency.
    """
    # Local import to avoid circulars
    from app.core.groq_service import groq_complete

    # Always suggest competitors as a sales-focused assistant - AT THE END
    competitor_guidance = " Structure your response: 1) Answer from documents first, 2) LAST: Add 'Competitor Alternatives' section with positive points about alternatives to help customers make informed decisions."
    
    system = (
        f"You are a sales-focused assistant who answers using the provided document "
        f"excerpts. If the excerpts are insufficient, reply with 'insufficient evidence'. "
        f"Be concise and factual.{competitor_guidance}"
    )
    joined = "\n\n".join(f"[Doc {i+1}] {c}" for i, c in enumerate(chunks))
    user = f"Question: {question}\n\nUse only this context:\n{joined}"
    return groq_complete(system, user)
