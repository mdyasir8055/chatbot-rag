# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from app.api.routes import router as api_router
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="AI Sales Agent")

# CORS (keep it simple during dev; tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ API router – assume routes.py uses APIRouter(prefix="/api")
#    If your routes.py has NO prefix, then change this line to:
#    app.include_router(api_router, prefix="/api")
app.include_router(api_router)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ✅ Root HTML -> index.html
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

for r in app.router.routes:
    try:
        print(r.path, r.methods)
    except:
        pass

