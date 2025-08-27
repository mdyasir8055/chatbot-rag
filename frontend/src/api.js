import axios from "axios";

// Create axios instance
const API = axios.create({
  baseURL: "http://localhost:8000", // ✅ Your FastAPI backend URL
  timeout: 20000, // 20s timeout for heavier uploads/calls
});

// Add a response interceptor for error logging
API.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      console.error("Backend error:", error.response.data);
    } else if (error.request) {
      console.error("No response from backend. Check if FastAPI is running.");
    } else {
      console.error("Error:", error.message);
    }
    return Promise.reject(error);
  }
);

// ---- API helpers ----
export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await API.post("/ingest/", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

export async function sendChat(query) {
  const res = await API.post("/chat/", { query });
  return res.data; // { query, answer } or {query, response}
}

export async function updateLLMSettings({ apiKey, model, temperature }) {
  const res = await API.post("/settings/llm", {
    api_key: apiKey || undefined,
    model: model || undefined,
    temperature: temperature ?? undefined,
  });
  return res.data; // {status: 'ok', model}
}

export async function compareDocuments({ query, docA, docB }) {
  const res = await API.post("/compare/", {
    query,
    doc_a: docA,
    doc_b: docB,
  });
  return res.data; // {status:'ok', result: {...}}
}

export default API;
