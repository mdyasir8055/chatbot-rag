from app.core.vectorstore import vector_store
import fitz  # PyMuPDF for PDF text

def ingest_document(content: bytes, filename: str):
    with open(filename, "wb") as f:
        f.write(content)

    doc = fitz.open(filename)
    texts = [page.get_text("text") for page in doc]
    vector_store.add_texts(texts)
