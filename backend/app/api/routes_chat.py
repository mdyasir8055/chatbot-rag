from fastapi import APIRouter, Depends
from app.services.rag_service import answer_query
from app.dependencies import get_db

router = APIRouter()

@router.post("/")
def chat(query: str, db=Depends(get_db)):
    """Chat with the RAG bot"""
    response = answer_query(query, db)
    return {"query": query, "response": response}
