from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()
rag = RAGPipeline()

class CompareRequest(BaseModel):
    query: str
    doc_a: str  # filename already saved under backend/data
    doc_b: str

@router.post("/")
def compare(req: CompareRequest):
    try:
        result = rag.compare(req.query, req.doc_a, req.doc_b)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))