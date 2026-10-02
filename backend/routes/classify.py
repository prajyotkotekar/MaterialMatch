"""POST /classify -> upload a photo, get the waste type. POST /classify/multi -> 2-4 photos."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.schemas import ClassifyResponse, MultiClassifyResponse
from ml.classifier.predict import load_model, predict, predict_group

router = APIRouter(prefix="/classify", tags=["classify"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PHOTOS = 4
_model_slots = asyncio.Semaphore(2)  # CPU inference: more parallel runs only queue up and use more memory


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image larger than 10 MB")
    return data


async def _ensure_model():
    try:
        await run_in_threadpool(load_model)
    except FileNotFoundError as exc:
        raise HTTPException(503, "Classifier not trained yet (ml/classifier/weights/best.pt missing)") from exc


def _bad_image(data: bytes) -> str:
    return f"Could not read image: {'empty file' if not data else 'not a supported image (JPG, PNG, WEBP, HEIC)'}"


@router.post("", response_model=ClassifyResponse)
async def classify(file: UploadFile = File(...)):
    data = await read_upload(file)
    await _ensure_model()
    try:
        async with _model_slots:
            return await run_in_threadpool(predict, data)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, _bad_image(data)) from exc


@router.post("/multi", response_model=MultiClassifyResponse)
async def classify_multi(files: list[UploadFile] = File(...),
                         mode: Literal["same_item", "different_items"] = Form("same_item")):
    """same_item: photos of ONE item -> one combined prediction (probabilities averaged).
    different_items: one prediction per photo, in upload order."""
    if not 1 <= len(files) <= MAX_PHOTOS:
        raise HTTPException(422, f"Send 1-{MAX_PHOTOS} photos")
    images = [await read_upload(f) for f in files]
    await _ensure_model()
    for data in images:
        if not data:
            raise HTTPException(400, _bad_image(data))
    try:
        async with _model_slots:
            results = await run_in_threadpool(predict_group, images, mode == "same_item")
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, "Could not read image: one of the files is not a supported image") from exc
    return {"mode": mode, "results": results}
