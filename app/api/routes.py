# app/api/routes.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uuid

from app.core.orchestrator import Orchestrator
from app.core.email_service import (
    send_manager_email,
    build_lead_subject,
    build_lead_body,
    MANAGER_EMAIL,
)
from app.core.pdf_service import extract_pdf_text, extract_topics_heuristic
from app.core.web_service import extract_main_text

router = APIRouter()
orch = Orchestrator()


def ok(data) -> JSONResponse:
    return JSONResponse({"ok": True, "data": data})


# ---------- Feature 1: Topic init + Ask ----------
@router.post("/api/init-topic")
async def init_topic(
    product_name: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    rag_enabled: Optional[bool] = Form(True),
    ocr_mode: str = Form("auto"),
    pdf: UploadFile | None = File(None),
):
    try:
        pdf_bytes = await pdf.read() if pdf else None
        result = await orch.init_topic(
            pdf_bytes=pdf_bytes,
            url=url,
            product_name=product_name,
            rag_enabled=rag_enabled,
            ocr_mode=ocr_mode,
        )
        # Back-compat meta for your sidebar
        meta = {"source": result["topic"].get("meta", {}).get("source", "manual")}
        if pdf and pdf.filename:
            meta.update({"filename": pdf.filename})
        return ok({"primary": result["topic"]["primary"], "meta": meta})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/ask")
async def ask(question: str = Form(...)):
    try:
        result = await orch.answer_dual(question)
        # Back-compat flattening for existing frontend: expose final answer & sources
        final_sources = [c.get("ref") for c in result.get("final_citations", [])]
        return ok({
            "answer": result.get("final_answer"),
            "sources": final_sources,
            # full payload if you want to render more (RAG/Web/Fused)
            "raw": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/history")
async def history():
    try:
        return ok(orch.get_history())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/clear")
async def clear():
    try:
        return ok(orch.clear())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- NEW: toggle RAG ----------
@router.post("/api/rag")
async def rag_toggle(enabled: str = Form(...)):
    try:
        state = orch.set_rag(enabled.lower() in ("1", "true", "yes", "on"))
        return ok(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Feature 2: Sales Handoff ----------
@router.post("/api/sales/connect/prepare")
async def sales_prepare(
    user_name: str = Form(...),
    user_email: str = Form(...),
    user_phone: str = Form(...),
    product_ref: str = Form(...),
    summary: str = Form(...),
    best_time: Optional[str] = Form(None),
    quoted_price: Optional[str] = Form(None),
):
    try:
        subject = build_lead_subject(product_ref=product_ref, user_name=user_name)
        body = build_lead_body(
            user_name=user_name,
            user_email=user_email,
            user_phone=user_phone,
            product_ref=product_ref,
            summary=summary,
            best_time=best_time,
            quoted_price=quoted_price,
        )
        request_id = str(uuid.uuid4())
        return ok({"to": MANAGER_EMAIL, "subject": subject, "body": body, "request_id": request_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/sales/connect/send")
async def sales_send(subject: str = Form(...), body: str = Form(...)):
    try:
        sent, info = send_manager_email(subject=subject, body=body)
        return ok({"status": "sent" if sent else "preview", "to": MANAGER_EMAIL, "info": info})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Feature 3: Comparison Mode (CSE-only) ----------
@router.post("/api/compare/init-topic")
async def compare_init(
    a_name: Optional[str] = Form(None),
    a_url: Optional[str] = Form(None),
    a_pdf: UploadFile | None = File(None),
    b_name: Optional[str] = Form(None),
    b_url: Optional[str] = Form(None),
    b_pdf: UploadFile | None = File(None),
):
    try:
        a_bytes = await a_pdf.read() if a_pdf else None
        b_bytes = await b_pdf.read() if b_pdf else None

        def make_topic(name, url, bytes_):
            if bytes_:
                txt = extract_pdf_text(bytes_)
            elif url:
                txt = extract_main_text(url)
            else:
                txt = ""
            return extract_topics_heuristic(txt, user_name_hint=name)

        topic_a = make_topic(a_name, a_url, a_bytes)
        topic_b = make_topic(b_name, b_url, b_bytes)
        await orch.compare_init(topic_a, topic_b)
        return ok({"topicA": topic_a, "topicB": topic_b})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/compare/ask")
async def compare_ask(question: str = Form(...)):
    try:
        result = await orch.compare_ask(question)
        return ok(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
