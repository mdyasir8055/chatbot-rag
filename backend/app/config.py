import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "RAG Chatbot"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/chatbot_db")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "faiss_index")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

settings = Settings()
