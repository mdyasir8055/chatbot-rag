from fastapi import APIRouter, UploadFile, File
from app.services.training_service import ingest_document

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload PDF and add to vector store"""
    content = await file.read()
    filename = file.filename
    ingest_document(content, filename)
    return {"status": "success", "filename": filename}
