from app.core.vectorstore import vector_store
from app.core.llm import call_groq

def answer_query(query: str, db):
    # Step 1: Retrieve relevant docs
    indices, _ = vector_store.search(query)
    context = "Relevant docs: " + str(indices)

    # Step 2: Call LLM
    prompt = f"Answer the query using context:\n{context}\n\nQuery: {query}"
    response = call_groq(prompt)

    if not response.strip():
        return "I don’t know. Data not available."
    return response
