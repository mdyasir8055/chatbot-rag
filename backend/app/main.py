from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Use the same routers as the top-level entrypoint to avoid drift
from app.routes.chat import router as chat_router
from app.routes.ingest import router as ingest_router
from app.routes.settings import router as settings_router
from app.routes.compare import router as compare_router

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
app.include_router(settings_router, prefix="/settings", tags=["settings"])
app.include_router(compare_router, prefix="/compare", tags=["compare"])

@app.get("/")
def health_check():
    return {"status": "ok", "service": "rag-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
