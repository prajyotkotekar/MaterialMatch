"""
backend/main.py - MaterialMatch FastAPI app.

    uvicorn backend.main:app --reload        # from the project root
    docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import classify, feedback, listings, match
from ml.carbon import DISCLAIMER

app = FastAPI(
    title="MaterialMatch API",
    description=("Waste-to-feedstock matching for Bengaluru. Listings are SYNTHETIC demo data; "
                 "recyclers are sample profiles. " + DISCLAIMER),
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(listings.router)
app.include_router(match.router)
app.include_router(classify.router)
app.include_router(feedback.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}
