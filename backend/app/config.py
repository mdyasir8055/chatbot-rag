import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "RAG Chatbot"
    # Default to SQLite file inside backend/data to avoid external DB during local dev
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "faiss_index")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

settings = Settings()
