"""
feedback_memory.py - Let confirmed user feedback improve predictions immediately (no retraining).

Every photo a user confirmed or corrected (data/feedback/classifier_feedback.jsonl, latest answer
per photo set) is embedded once with the current model. predict.py then compares a new photo with
these embeddings: when a confirmed photo is closer than the calibrated `memory_threshold`
(calibrate_ood.py - the distance at which a nearest neighbour's waste type is right >= 98% of the
time), the prediction is blended toward the user's label, and the photo no longer counts as
"unlike any training image". The closer the match, the stronger the blend (at most 95%).

    python -m ml.classifier.feedback_memory --stats     # what the memory contains
    python -m ml.classifier.feedback_memory --reset     # stop using feedback given until now
                                                        # (non-destructive: records are kept for retraining)

The memory is a shortcut for photos that look like ones already corrected; it does not change
the model. retrain_with_feedback.py folds the feedback into the model itself.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_DIR = Path(os.environ.get("MM_FEEDBACK_DIR") or ROOT / "data" / "feedback")  # same as feedback_store
FEEDBACK_LOG = FEEDBACK_DIR / "classifier_feedback.jsonl"
STATE_FILE = FEEDBACK_DIR / "memory_state.json"
OTHER = "other"
MAX_WEIGHT = 0.95

_lock = threading.Lock()
_state: dict = {"log": FEEDBACK_LOG, "state": STATE_FILE, "enabled": True}
_vectors: dict[tuple[str, str], np.ndarray] = {}   # (weights_key, image_sha) -> unit embedding
_built: dict[str, tuple] = {}                        # weights_key -> (signature, entries)


def configure(log_path: Path | None = None, state_path: Path | None = None,
              enabled: bool | None = None) -> None:
    """Point the memory at another feedback log (tests) or switch it off."""
    with _lock:
        if log_path is not None:
            _state["log"] = Path(log_path)
        if state_path is not None:
            _state["state"] = Path(state_path)
        if enabled is not None:
            _state["enabled"] = enabled
        _built.clear()


def _ignore_before() -> str | None:
    p = _state["state"]
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")).get("ignore_before")
    return None


def records() -> list[dict]:
    """Latest answer per photo set, after the last reset, that names a material (or 'other')."""
    log = _state["log"]
    if not log.exists():
        return []
    cutoff = _ignore_before()
    latest: dict[str, dict] = {}
    with open(log, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                latest[r["item_key"]] = r
    out = []
    for r in latest.values():
        if cutoff and r["created_at"] < cutoff:
            continue
        if not r.get("actual_label") or r["actual_label"] == "unknown":
            continue
        out.append(r)
    return out


def _signature() -> tuple:
    log, st = _state["log"], _state["state"]
    return (log.stat().st_mtime_ns if log.exists() else 0, log.stat().st_size if log.exists() else 0,
            st.stat().st_mtime_ns if st.exists() else 0)


def version() -> str:
    """Changes whenever the memory's contents may have changed (use it as a cache key)."""
    return "-".join(map(str, _signature())) if _state["enabled"] else "off"


def entries(weights_key: str, embed_paths) -> dict | None:
    """{'emb': N x D unit vectors, 'labels': [(waste_type, sub_type|None)], 'items': [...]} or None.

    embed_paths(list[Path]) -> N x D embeddings with the current model (called only for photos
    not embedded yet for these weights).
    """
    if not _state["enabled"]:
        return None
    sig = _signature()
    with _lock:
        hit = _built.get(weights_key)
        if hit and hit[0] == sig:
            return hit[1]
    recs = records()
    rows = []
    for r in recs:
        for sha, rel in zip(r["image_sha256"], r["image_files"]):
            path = (ROOT / rel) if not Path(rel).is_absolute() else Path(rel)
            if not path.exists():
                path = _state["log"].parent / "images" / Path(rel).name
            if path.exists():
                rows.append((sha, path, (r["actual_label"], r.get("actual_sub_type")), r["item_key"]))
    missing = [(sha, p) for sha, p, _, _ in rows if (weights_key, sha) not in _vectors]
    if missing:
        F = np.asarray(embed_paths([p for _, p in missing]), dtype=np.float32)
        F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
        for (sha, _), v in zip(missing, F):
            _vectors[(weights_key, sha)] = v
    result = None
    if rows:
        result = {"emb": np.stack([_vectors[(weights_key, sha)] for sha, _, _, _ in rows]),
                  "labels": [lab for _, _, lab, _ in rows],
                  "items": [key for _, _, _, key in rows]}
    with _lock:
        _built[weights_key] = (sig, result)
    return result


def reset() -> str:
    """Ignore all feedback given until now (records stay on disk for retraining)."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _state["state"].parent.mkdir(parents=True, exist_ok=True)
    _state["state"].write_text(json.dumps({"ignore_before": now}), encoding="utf-8")
    _built.clear()
    return now


def stats() -> dict:
    recs = records()
    from collections import Counter
    return {"enabled": _state["enabled"], "ignore_before": _ignore_before(), "photo_sets": len(recs),
            "photos": sum(len(r["image_files"]) for r in recs),
            "labels": dict(Counter(f"{r['actual_label']}/{r.get('actual_sub_type')}" for r in recs))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--reset", action="store_true", help="stop using feedback given until now (non-destructive)")
    a = ap.parse_args(argv)
    if a.reset:
        print(f"Feedback memory now ignores answers before {reset()} (the records are kept).")
    print(json.dumps(stats(), indent=2))


if __name__ == "__main__":
    main()
