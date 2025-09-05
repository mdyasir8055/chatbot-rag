# app/core/orchestrator.py
from __future__ import annotations
from typing import Optional, Dict, List, Tuple
import asyncio

from app.core.pdf_service import extract_pdf_text, extract_topics_heuristic
from app.core.web_service import web_fallback_answer, extract_main_text
from app.core.rag_service import RagIndex, rag_knows, synthesize_from_chunks
from app.core.ocr_service import is_scanned, ocr_pdf


class Orchestrator:
    """
    Coordinates:
      - Topic init (PDF/URL/Product Name) + optional OCR
      - Dual-engine answering (RAG + CSE) w/ deterministic fallback
      - Comparison Mode (CSE-only)
      - Sales handoff handled in routes
    """
    def __init__(self):
        self.topic: Optional[Dict] = None
        self.history: List[Dict] = []
        self.pdf_text: str = ""
        self.rag_enabled: bool = True
        self.rag_index: RagIndex | None = None

        # Comparison
        self.compare: Dict | None = None

    # ---------- Config ----------
    def set_rag(self, enabled: bool) -> Dict:
        self.rag_enabled = bool(enabled)
        # If disabling, free the index
        if not self.rag_enabled:
            self.rag_index = None
        elif self.rag_enabled and self.pdf_text:
            # lazily (re)build if we have text
            self.rag_index = RagIndex()
            self.rag_index.build(self.pdf_text)
        return {"rag_enabled": self.rag_enabled}

    # ---------- Topic init ----------
    def _detect_topic_from_text(self, text: str, product_name: Optional[str]) -> Dict:
        try:
            return extract_topics_heuristic(text or "", user_name_hint=product_name)
        except Exception:
            primary = (product_name or "").strip() or "Unknown Topic"
            return {"primary": primary, "aliases": [primary]}

    async def init_topic(
        self,
        pdf_bytes: bytes | None = None,
        url: Optional[str] = None,
        product_name: Optional[str] = None,
        rag_enabled: Optional[bool] = None,
        ocr_mode: str = "auto"
    ) -> Dict:
        if rag_enabled is not None:
            self.rag_enabled = bool(rag_enabled)

        text = ""
        meta = {"source": "manual"}

        if pdf_bytes:
            meta["source"] = "pdf"
            # OCR routing
            try:
                needs_ocr = (ocr_mode == "force") or (ocr_mode == "auto" and is_scanned(pdf_bytes))
            except Exception:
                needs_ocr = (ocr_mode == "force")  # fail-closed to non-OCR

            if needs_ocr:
                try:
                    text = ocr_pdf(pdf_bytes)
                    meta["ocr_used"] = True
                except Exception as e:
                    # Fallback to standard extraction with a warning-like marker
                    text = extract_pdf_text(pdf_bytes) or ""
                    meta["ocr_used"] = False
                    meta["ocr_error"] = f"{e.__class__.__name__}: {e}"
            else:
                text = extract_pdf_text(pdf_bytes) or ""
                meta["ocr_used"] = False

            self.pdf_text = text

        elif url:
            meta["source"] = "url"
            text = extract_main_text(url) or ""
            self.pdf_text = ""  # not caching web text as document
        else:
            # Product name only
            self.pdf_text = ""

        self.topic = self._detect_topic_from_text(text, product_name)
        self.topic["meta"] = meta

        # (Re)build RAG if enabled and we have doc text
        if self.rag_enabled and self.pdf_text:
            self.rag_index = RagIndex()
            self.rag_index.build(self.pdf_text)
            index_status = "ready"
        else:
            self.rag_index = None
            index_status = "disabled_or_empty"

        return {
            "topic": self.topic,
            "rag_enabled": self.rag_enabled,
            "index_status": index_status,
        }

    # ---------- Answering ----------
    async def _run_cse(self, question: str) -> Tuple[str, List[str]]:
        try:
            ans, urls = await web_fallback_answer(question=question, topics=self.topic)
            return ans, urls
        except TypeError:
            ans, urls = web_fallback_answer(question=question, topics=self.topic)  # type: ignore
            return ans, urls

    async def answer_dual(self, question: str) -> Dict:
        # Always run CSE
        cse_task = asyncio.create_task(self._run_cse(question))

        # Optional RAG
        rag_payload = {"status": "unknown"}
        if self.rag_enabled and self.rag_index is not None:
            retrieved = self.rag_index.retrieve(question, k=8)
            if rag_knows(retrieved):
                top_chunks = [c for c, _ in retrieved[:5]]
                rag_summary = synthesize_from_chunks(question, top_chunks)
                if "insufficient evidence" not in (rag_summary or "").lower():
                    rag_payload = {
                        "status": "known",
                        "summary": (rag_summary or "").strip(),
                        "citations": [],  # extend if you map chunks→pages
                        "confidence": 0.75
                    }

        cse_answer, cse_urls = await cse_task
        if cse_answer and "(found in PDF)" not in cse_answer and "(looked up on the web" not in cse_answer:
            cse_answer = cse_answer.strip() + "\n(looked up on the web since not in PDF)"

        if rag_payload["status"] == "unknown":
            final = cse_answer
            final_cites = [{"type": "url", "ref": u} for u in (cse_urls or [])]
            out = {
                "rag": rag_payload,
                "cse": {"summary": cse_answer, "sources": cse_urls, "confidence": 0.7},
                "final_answer": final,
                "final_citations": final_cites
            }
        else:
            from app.core.groq_service import groq_complete
            
            # Always act as sales agent suggesting competitors for any product - AT THE END
            sales_instruction = " Structure response: 1) Answer about requested product first, 2) LAST: Add 'Competitor Alternatives' section with positive points about alternatives to help customers decide."
            
            fused = groq_complete(
                f"Merge two short answers, noting agreements or differences in <=2 lines.{sales_instruction}",
                f"Answer A (doc): {rag_payload['summary']}\n\nAnswer B (web): {cse_answer}"
            )
            final = groq_complete(
                f"Compose a single direct answer grounded primarily in Answer A (doc). "
                f"Use Answer B (web) only to fill small gaps. Keep it concise.{sales_instruction}",
                f"Question: {question}\n\nAnswer A (doc): {rag_payload['summary']}\n\nAnswer B (web): {cse_answer}"
            )
            final_cites = [{"type": "pdf", "ref": "document"}] + [{"type": "url", "ref": u} for u in (cse_urls or [])]
            out = {
                "rag": rag_payload,
                "cse": {"summary": cse_answer, "sources": cse_urls, "confidence": 0.7},
                "fused": {"comparator": "aligned", "summary": (fused or "").strip()},
                "final_answer": (final or "").strip(),
                "final_citations": final_cites
            }

        # History
        self.history.append({"role": "user", "text": question})
        self.history.append({"role": "ai", "text": out["final_answer"]})
        return out

    # ---------- Comparison Mode (CSE-only) ----------
    async def compare_init(self, topic_a: Dict, topic_b: Dict):
        self.compare = {"A": topic_a, "B": topic_b}

    async def compare_ask(self, question: str) -> Dict:
        if not self.compare:
            return {"A": {}, "B": {}, "matrix": [], "recommendation": "Initialize comparison first."}

        async def cse_for(t):
            try:
                ans, urls = await web_fallback_answer(question=question, topics=t)
            except TypeError:
                ans, urls = web_fallback_answer(question=question, topics=t)  # type: ignore
            if ans and "(looked up on the web" not in ans:
                ans = ans.strip() + "\n(looked up on the web)"
            return ans, urls

        a_task = asyncio.create_task(cse_for(self.compare["A"]))
        b_task = asyncio.create_task(cse_for(self.compare["B"]))
        (a_ans, a_urls), (b_ans, b_urls) = await asyncio.gather(a_task, b_task)

        from app.core.groq_service import groq_complete
        reco = groq_complete(
            "Recommend A or B concisely (2 lines). If ambiguous, say what extra info is needed.",
            f"Question: {question}\n\nA: {a_ans}\n\nB: {b_ans}"
        )
        matrix = [
            {"dimension": "Answer (web)", "a_value": a_ans, "b_value": b_ans, "a_cites": list(range(len(a_urls))), "b_cites": list(range(len(b_urls)))}
        ]
        return {
            "A": {"summary": a_ans, "sources": a_urls, "confidence": 0.7},
            "B": {"summary": b_ans, "sources": b_urls, "confidence": 0.7},
            "matrix": matrix,
            "comparator": "partial",
            "recommendation": (reco or "").strip()
        }

    # ---------- History ----------
    def get_history(self) -> List[Dict]:
        return list(self.history)

    def clear(self) -> Dict:
        self.topic = None
        self.history.clear()
        self.pdf_text = ""
        self.rag_index = None
        self.compare = None
        return {"ok": True}
