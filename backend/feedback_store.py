"""
backend/feedback_store.py - Persist user feedback on photo classifications.

Used by POST /feedback and by the Streamlit app (which runs the same code in-process).
Append-only JSONL log + the photos it refers to, so the model can later be evaluated on
real uploads and retrained. Nothing here retrains the model.

    data/feedback/classifier_feedback.jsonl
    data/feedback/images/<sha256>.<ext>

On hosts with a temporary disk, feedback_sync.py mirrors this folder to a private Hugging Face dataset.
"""

from __future__ import annotations

import hashlib
import os
import io
import json
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_DIR = Path(os.environ.get("MM_FEEDBACK_DIR") or ROOT / "data" / "feedback")  # tests point this elsewhere
LOG_PATH = FEEDBACK_DIR / "classifier_feedback.jsonl"
IMAGE_DIR = FEEDBACK_DIR / "images"
WEIGHTS = ROOT / "ml" / "classifier" / "weights" / "best.pt"

from backend import feedback_sync  # noqa: E402
from ml.taxonomy import UNKNOWN, WASTE_TYPES, photo_hierarchy  # noqa: E402

OTHER = "other"  # a material the model has no class for (e.g. rubber, ceramics)
MODES = ("single", "same_item", "different_items")
IMAGE_FORMATS = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp", "HEIF": "heic"}
MAX_PIXELS = 80_000_000  # same limit as the classifier
# Anyone using a public deployment can add photos, and the synced copy is downloaded and embedded at every start.
MAX_STORED_BYTES = int(os.environ.get("MM_FEEDBACK_MAX_MB") or 1024) * 1024 * 1024

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _weights_id() -> str | None:
    return hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()[:12] if WEIGHTS.exists() else None


def _image_ext(data: bytes) -> str:
    """File extension of a valid photo; anything else (scripts, archives, corrupt files) is refused."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, pixels = im.format, im.width * im.height
            im.verify()
    except Exception as exc:
        raise ValueError("Every file must be a JPG, PNG, WEBP or HEIC photo") from exc
    if fmt not in IMAGE_FORMATS or pixels > MAX_PIXELS:
        raise ValueError("Every file must be a JPG, PNG, WEBP or HEIC photo of at most 80 megapixels")
    return IMAGE_FORMATS[fmt]


def _stored_bytes() -> int:
    return sum(p.stat().st_size for p in IMAGE_DIR.iterdir() if p.is_file()) if IMAGE_DIR.exists() else 0


def _save_image(data: bytes, ext: str) -> tuple[str, str]:
    sha = hashlib.sha256(data).hexdigest()
    path = IMAGE_DIR / f"{sha}.{ext}"
    if not path.exists():
        path.write_bytes(data)
    try:
        return sha, path.relative_to(ROOT).as_posix()
    except ValueError:  # MM_FEEDBACK_DIR outside the project (tests)
        return sha, path.as_posix()


def save_feedback(images: list[bytes], predicted_label: str, predicted_confidence: float,
                  is_correct: bool, actual_label: str | None = None, mode: str = "single",
                  probabilities: list[dict] | None = None, source: str = "api",
                  predicted_sub_type: str | None = None, actual_sub_type: str | None = None) -> dict:
    """Store one answer to "Was this prediction correct?". Returns the stored record.
    Raises ValueError for invalid input, OverflowError when the photo store is full (MAX_STORED_BYTES).

    is_correct refers to the whole prediction (waste type AND sub-type). When it is False,
    actual_label (waste type, or "other") is required; actual_sub_type is optional
    ("unknown"/None = user didn't say) and must belong to actual_label's photo sub-types.
    """
    if not images:
        raise ValueError("At least one image is required")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if predicted_label not in (*WASTE_TYPES, UNKNOWN):
        raise ValueError(f"predicted_label must be one of {(*WASTE_TYPES, UNKNOWN)}")
    known_subs = {s for subs in photo_hierarchy().values() for s in subs}
    if predicted_sub_type not in (None, "", UNKNOWN, *known_subs):
        raise ValueError("predicted_sub_type is not a sub-type the model knows")
    exts = [_image_ext(b) for b in images]
    if is_correct and predicted_label == UNKNOWN:  # "not a supported material" was right
        actual = OTHER
    else:
        actual = predicted_label if is_correct else actual_label
    actual_sub = predicted_sub_type if is_correct else actual_sub_type
    if actual not in (*WASTE_TYPES, OTHER):
        raise ValueError(f"actual_label must be one of {(*WASTE_TYPES, OTHER)}")
    if actual_sub in ("", UNKNOWN):
        actual_sub = None
    allowed_subs = photo_hierarchy().get(actual, [])
    if actual_sub is not None and actual_sub not in allowed_subs:
        raise ValueError(f"actual_sub_type for '{actual}' must be one of {allowed_subs}")

    with _lock:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        new_bytes = sum(len(b) for b, ext in zip(images, exts)
                        if not (IMAGE_DIR / f"{hashlib.sha256(b).hexdigest()}.{ext}").exists())
        if new_bytes and _stored_bytes() + new_bytes > MAX_STORED_BYTES:
            raise OverflowError("Feedback storage is full; the answer was not saved")
        saved = [_save_image(b, ext) for b, ext in zip(images, exts)]
        record = {
            "id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "mode": mode,
            # same photos -> same key; the latest answer for a key is the one that counts
            "item_key": hashlib.sha256("".join(s for s, _ in saved).encode()).hexdigest()[:16],
            "image_sha256": [s for s, _ in saved],
            "image_files": [p for _, p in saved],
            "predicted_label": predicted_label,
            "predicted_sub_type": predicted_sub_type,
            "predicted_confidence": round(float(predicted_confidence), 4),
            "probabilities": probabilities or [],
            "is_correct": bool(is_correct),
            "actual_label": actual,
            "actual_sub_type": actual_sub,
            "type_correct": actual == predicted_label,
            "subtype_correct": None if actual_sub is None or predicted_sub_type is None
                               else (actual == predicted_label and actual_sub == predicted_sub_type),
            "model_weights": _weights_id(),
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    feedback_sync.push(LOG_PATH, [IMAGE_DIR / Path(p).name for p in record["image_files"]])
    return record


def restore() -> int:
    """Bring back the feedback saved remotely before this disk was wiped (no-op unless sync is configured)."""
    return feedback_sync.restore(LOG_PATH)


def load_feedback(latest_only: bool = True) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    rows = []
    with _lock, open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:  # blank or half-written line: skip it rather than break every reader
                continue
    if latest_only:
        rows = list({r["item_key"]: r for r in rows}.values())
    return rows


def summary() -> dict:
    rows = load_feedback()
    n = len(rows)
    return {
        "n_items": n,
        "n_correct": sum(r["is_correct"] for r in rows),
        "accuracy_on_feedback": round(sum(r["is_correct"] for r in rows) / n, 3) if n else None,
        "n_other_material": sum(r["actual_label"] == OTHER for r in rows),
        "corrections": dict(Counter(f"{r['predicted_label']} -> {r['actual_label']}"
                                    for r in rows if not r["is_correct"])),
        "subtype_corrections": dict(Counter(
            f"{r['predicted_label']}/{r.get('predicted_sub_type')} -> {r['actual_label']}/{r.get('actual_sub_type')}"
            for r in rows if not r["is_correct"] and r.get("actual_sub_type"))),
        "note": ("Confirmed answers are used immediately for similar photos (feedback memory) and by "
                 "retrain_with_feedback.py; the model weights change only when that script promotes a new model."),
    }
