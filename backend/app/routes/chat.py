from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()
rag = RAGPipeline()

class ChatRequest(BaseModel):
    query: str

@router.post("/")
async def chat(req: ChatRequest):
    try:
        answer = rag.answer(req.query)
        return {"query": req.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))