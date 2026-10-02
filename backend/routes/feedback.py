"""POST /feedback -> store "was this prediction correct?"; GET /feedback/summary."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend import feedback_store
from backend.routes.classify import read_upload
from backend.security import require_api_key

router = APIRouter(prefix="/feedback", tags=["feedback"])


# Confirmed answers feed the instant feedback memory, so writing is protected by MM_API_KEY when it is set.
@router.post("", status_code=201, dependencies=[Depends(require_api_key)])
async def post_feedback(
    files: list[UploadFile] = File(..., description="the 1-4 photos the prediction was made from"),
    predicted_label: str = Form(..., max_length=40),
    predicted_confidence: float = Form(..., ge=0, le=1),
    is_correct: bool = Form(...),
    actual_label: Literal["textile", "plastic", "construction", "e_waste", "paper", "glass", "metal",
                          "biological", "trash", "other"] | None = Form(None),
    mode: Literal["single", "same_item", "different_items"] = Form("single"),
    predicted_sub_type: str | None = Form(None, max_length=60, description="sub-type the model predicted"),
    actual_sub_type: str | None = Form(None, max_length=60, description="correct sub-type, if the user chose one"),
):
    if not 1 <= len(files) <= 4:
        raise HTTPException(422, "Send 1-4 photos")
    if not is_correct and actual_label is None:
        raise HTTPException(422, "actual_label is required when is_correct is false")
    images = [await read_upload(f) for f in files]
    try:
        return await run_in_threadpool(
            feedback_store.save_feedback, images, predicted_label, predicted_confidence,
            is_correct, actual_label, mode, None, "api", predicted_sub_type, actual_sub_type)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/summary")
def feedback_summary():
    return feedback_store.summary()


@router.get("/memory")
def feedback_memory_stats():
    """What the instant feedback memory currently uses (reset: python -m ml.classifier.feedback_memory --reset)."""
    from ml.classifier import feedback_memory
    return feedback_memory.stats()
