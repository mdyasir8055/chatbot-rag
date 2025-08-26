from sqlalchemy import Column, Integer, String, Text
from app.core.db import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String(255))
    response = Column(Text)
