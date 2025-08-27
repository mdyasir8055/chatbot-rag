from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()
rag = RAGPipeline()

@router.post("/")
async def ingest(file: UploadFile = File(...)):
    try:
        content = await file.read()
        rag.ingest_bytes(content, file.filename)
        return {"status": "ok", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))