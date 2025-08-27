import os
from typing import List
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_store")

class RAGPipeline:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        # Free CPU-only embeddings (no GPU/torch required)
        self.embeddings = FastEmbedEmbeddings()
        # Persistent Chroma store
        self.vstore = Chroma(
            collection_name="rag_docs",
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
        )
        # Groq offers a free tier API key; set GROQ_API_KEY in .env
        self._api_key = os.getenv("GROQ_API_KEY")
        self._model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        self.llm = ChatGroq(
            temperature=0.2,
            groq_api_key=self._api_key,
            model_name=self._model,
        )
        # Conservative splitting; pypdf handles most PDFs, but if pages are image-based, install tesseract OCR.
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def update_llm(self, api_key: str | None = None, model: str | None = None, temperature: float | None = None):
        """Hot-swap LLM settings at runtime."""
        from math import isfinite
        if api_key:
            self._api_key = api_key
        if model:
            self._model = model
        temp = 0.2 if temperature is None else float(temperature)
        temp = temp if isfinite(temp) else 0.2
        self.llm = ChatGroq(
            temperature=temp,
            groq_api_key=self._api_key,
            model_name=self._model,
        )
        return {"model": self._model}

    def _save_bytes(self, content: bytes, filename: str) -> str:
        path = os.path.join(DATA_DIR, filename)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def _load_docs(self, path: str) -> List[Document]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            # Primary: use PyPDFLoader
            try:
                loader = PyPDFLoader(path)
                return loader.load()
            except Exception:
                # Fallback: use PyMuPDF (fitz) to extract text per page
                try:
                    import fitz  # PyMuPDF
                    docs: List[Document] = []
                    with fitz.open(path) as pdf:
                        for i, page in enumerate(pdf):
                            text = page.get_text("text") or ""
                            if text.strip():
                                docs.append(Document(page_content=text, metadata={"source": path, "page": i + 1}))
                    if docs:
                        return docs
                except Exception:
                    pass
                # Last resort: read raw bytes as text
                try:
                    with open(path, "rb") as f:
                        content = f.read().decode("utf-8", errors="ignore")
                    return [Document(page_content=content, metadata={"source": path})]
                except Exception as e:
                    raise e
        else:
            loader = TextLoader(path, autodetect_encoding=True)
            return loader.load()

    def ingest_bytes(self, content: bytes, filename: str):
        path = self._save_bytes(content, filename)
        docs = self._load_docs(path)
        chunks = self.splitter.split_documents(docs)
        self.vstore.add_documents(chunks)
        self.vstore.persist()

    def ingest_folder(self, folder: str = DATA_DIR):
        for name in os.listdir(folder):
            full = os.path.join(folder, name)
            if os.path.isfile(full):
                docs = self._load_docs(full)
                chunks = self.splitter.split_documents(docs)
                self.vstore.add_documents(chunks)
        self.vstore.persist()

    def answer(self, query: str) -> str:
        retriever = self.vstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the user's question.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        resp = self.llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)

    def compare(self, query: str, doc_a: str, doc_b: str) -> dict:
        """Compare two documents' content with respect to a query and produce a concise diff."""
        import os
        a_path = os.path.join(DATA_DIR, doc_a)
        b_path = os.path.join(DATA_DIR, doc_b)
        docs_a = self._load_docs(a_path) if os.path.exists(a_path) else []
        docs_b = self._load_docs(b_path) if os.path.exists(b_path) else []
        text_a = "\n\n".join([d.page_content for d in docs_a])[:8000]
        text_b = "\n\n".join([d.page_content for d in docs_b])[:8000]
        prompt = (
            "You are a comparison assistant. Given a user query and two document excerpts, produce: \n"
            "- Features (bulleted) relevant to the query for each doc\n"
            "- Pros list and Cons list for each doc\n"
            "- A brief summary and who should choose which doc\n"
            "Respond in compact JSON with keys: features_a, pros_a, cons_a, features_b, pros_b, cons_b, summary.\n\n"
            f"Query: {query}\n\nDocument A:\n{text_a}\n\nDocument B:\n{text_b}\n"
        )
        resp = self.llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        # Best-effort parse; keep content if not JSON
        import json
        try:
            data = json.loads(content)
            return data
        except Exception:
            return {"raw": content}