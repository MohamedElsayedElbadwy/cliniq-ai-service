"""Central FastAPI entry point for ClinIQ AI capabilities."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mri.router import router as mri_router
from voice.router import router as voice_router

PROJECT_DIR = Path(__file__).resolve().parent
WEB_DIR = PROJECT_DIR / "web"

load_dotenv(PROJECT_DIR / ".env")

app = FastAPI(title="ClinIQ AI Service", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.include_router(voice_router, prefix="/api/ai/voice")
app.include_router(mri_router, prefix="/api/ai")


@app.get("/", include_in_schema=False)
def serve_voice_call() -> FileResponse:
    """Serve the existing Voice Call test interface from the central service."""
    return FileResponse(WEB_DIR / "index.html")
