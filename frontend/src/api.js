import axios from "axios";

// Create axios instance
const API = axios.create({
  baseURL: "http://127.0.0.1:8000", // ✅ Your FastAPI backend URL
  timeout: 10000, // 10s timeout for requests
  headers: {
    "Content-Type": "application/json",
  },
});

// Add a response interceptor for error logging
API.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with status other than 2xx
      console.error("Backend error:", error.response.data);
    } else if (error.request) {
      // Request made but no response received
      console.error("No response from backend. Check if FastAPI is running.");
    } else {
      // Something else happened
      console.error("Error:", error.message);
    }
    return Promise.reject(error);
  }
);

export default API;
