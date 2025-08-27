from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()
rag = RAGPipeline()

class LLMSettings(BaseModel):
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None

@router.post("/llm")
def update_llm(settings: LLMSettings):
    """Update Groq API key/model at runtime."""
    out = rag.update_llm(api_key=settings.api_key, model=settings.model, temperature=settings.temperature)
    return {"status": "ok", **out}