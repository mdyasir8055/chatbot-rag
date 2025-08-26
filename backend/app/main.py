from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_chat, routes_admin

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Allowed origins (only your frontend)
origins = [
    "http://localhost:5173",  # Vite/React dev server
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # restrict only to your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(routes_chat.router, prefix="/chat", tags=["Chatbot"])
app.include_router(routes_admin.router, prefix="/admin", tags=["Admin"])

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG Chatbot API is running 🚀"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
