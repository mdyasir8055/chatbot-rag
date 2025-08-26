import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from app.config import settings

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_path = settings.VECTOR_DB_PATH
        self.index = None
        self.load_or_create()

    def load_or_create(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(384)

    def add_texts(self, texts):
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings).astype("float32"))
        faiss.write_index(self.index, self.index_path)

    def search(self, query, top_k=3):
        query_emb = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)
        return indices, distances

vector_store = VectorStore()
