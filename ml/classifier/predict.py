"""
predict.py - MaterialMatch photo classifier: image -> waste_type -> sub_type (+ unknown check).

    from ml.classifier.predict import predict
    r = predict("photo.jpg")      # path, bytes, file-like, PIL.Image or RGB numpy array
    r["waste_type"], r["sub_type"], r["confidence"], r["is_unknown"]

The v2 model is ONE classifier over leaf classes "<waste_type>__<sub_type>" (27 leaves).
The waste-type probability is the exact sum of its leaves' probabilities, and the sub-type
confidence is P(sub_type | waste_type), so type and sub-type can never disagree.

Result (per image):
    waste_type, sub_type          best waste type and its best sub-type ("unknown"/None if rejected)
    label                         == waste_type (kept for older clients)
    confidence                    == type_confidence (kept for older clients)
    type_confidence               P(waste_type)
    subtype_confidence            P(sub_type | waste_type)
    leaf_confidence               P(waste_type, sub_type)
    is_confident                  type_confidence >= min_confidence and not unknown
    subtype_is_confident          subtype_confidence >= DEFAULT_SUBTYPE_MIN_CONFIDENCE
    top_k                         [{label: waste_type, confidence}]  (waste-type level, as before)
    top_k_subtypes                [{label: leaf, waste_type, sub_type, confidence}]
    status                        "detected" | "confirm" | "unknown":
                                    unknown  = extremely unfamiliar photo, OR unfamiliar AND no clear
                                               winner, OR it resembles photos users marked "other"
                                    confirm  = unfamiliar photo OR no clear winner (best guess shown)
                                    detected = neither
    is_unknown, unknown_reason    status == "unknown" (+ why); best_guess is always filled
    confirm_reason                why a best guess needs confirming (status "confirm")
    ood                           raw scores + thresholds used (calibrate_ood.py)
    memory                        set when confirmed user feedback adjusted this result
                                  (feedback_memory.py): similar photos, distance, blend weight,
                                  and the model's own guess before the adjustment
    leaf_probs                    every leaf's probability (needed to combine photos)
    model_version                 "v2_hierarchical" or "v1_flat"

The older 4-class checkpoint (weights/best_v1_4class.pt) still works: its classes are waste
types, sub_type is None and no unknown check is run.

    python ml/classifier/predict.py photo1.jpg photo2.jpg [--same-item] [--json]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps

try:
    from ml.classifier import feedback_memory
except ImportError:  # run as a script from ml/classifier
    import feedback_memory

for _heif_mod in ("pi_heif", "pillow_heif"):
    try:
        __import__(_heif_mod).register_heif_opener()
        break
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "weights" / "best.pt"
SEP = "__"
UNKNOWN = "unknown"
# v1: measured 2026-09-28 on the 4-class test split (94.8% of correct predictions >= 0.9).
# v2: calibrate_ood.py stores its own value in <weights>.ood.npz, which overrides this.
DEFAULT_MIN_CONFIDENCE = 0.9
DEFAULT_SUBTYPE_MIN_CONFIDENCE = 0.5

_MODEL_CACHE: dict[str, Any] = {}
_OOD_CACHE: dict[str, dict | None] = {}
_LOCK = threading.Lock()
_PREDICT_LOCK = threading.RLock()


def load_model(weights: str | Path | None = None):
    """Load (and cache) a YOLO model. Raises FileNotFoundError if weights are missing."""
    path = (Path(weights) if weights else DEFAULT_WEIGHTS).expanduser().resolve()
    key = str(path)
    with _LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {path}. Train first with:\n"
                "  python ml/classifier/train_yolo.py --auto")
        from ultralytics import YOLO

        model = YOLO(str(path))
        if model.task not in ("classify", "detect"):
            raise ValueError(f"Unsupported model task '{model.task}' in {path}")
        _MODEL_CACHE[key] = model
        _OOD_CACHE[key] = _load_ood(path)
        return model


def _load_ood(weights_path: Path) -> dict | None:
    f = weights_path.with_suffix(".ood.npz")
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    bank = z["bank"].astype(np.float32)
    return {"bank": bank, "k": int(z["k"]), "knn_threshold": float(z["knn_threshold"]),
            "knn_threshold_by_type": json.loads(str(z["knn_threshold_by_type"])) if "knn_threshold_by_type" in z else {},
            "knn_far_by_type": json.loads(str(z["knn_far_by_type"])) if "knn_far_by_type" in z else {},
            "memory_threshold": float(z["memory_threshold"]) if "memory_threshold" in z else None,
            "msp_threshold": float(z["msp_threshold"]),
            "min_confidence": float(z["min_confidence"]) if "min_confidence" in z else None}


def ood_calibration(weights: str | Path | None = None) -> dict | None:
    load_model(weights)
    return _OOD_CACHE[str((Path(weights) if weights else DEFAULT_WEIGHTS).expanduser().resolve())]


def split_leaf(name: str) -> tuple[str, str | None]:
    return tuple(name.split(SEP, 1)) if SEP in name else (name, None)


def is_hierarchical(model) -> bool:
    return any(SEP in n for n in model.names.values())


MAX_PIXELS = 80_000_000  # larger than any phone photo; a bigger image is refused before it is decoded


def check_pixels(img: Image.Image) -> None:
    """Refuse decompression bombs: a tiny compressed file that would decode to gigabytes."""
    if img.width * img.height > MAX_PIXELS:
        raise ValueError(f"Image too large ({img.width}x{img.height} pixels)")


def _to_pil(image: Any) -> Image.Image:
    """Accept a path, raw bytes, file-like object, PIL image or RGB numpy array."""
    try:
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, (str, Path)):
            p = Path(image).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            img = Image.open(p)
        elif isinstance(image, (bytes, bytearray, memoryview)):
            if len(image) == 0:
                raise ValueError("Empty image bytes")
            img = Image.open(io.BytesIO(bytes(image)))
        elif hasattr(image, "read"):
            img = Image.open(image)
        elif hasattr(image, "shape") and hasattr(image, "dtype"):
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"Expected HxWx3 RGB array, got shape {arr.shape}")
            if arr.dtype != np.uint8:
                arr = np.clip(arr * (255 if arr.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr[..., :3])
        else:
            raise TypeError(f"Unsupported image type: {type(image).__name__}")
        check_pixels(img)  # header only, nothing decoded yet
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except FileNotFoundError:
        raise
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:  # truncated or corrupt file
        raise ValueError(f"Could not decode image: {exc}") from exc


def knn_distance(features: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine distance to the k nearest training embeddings (higher = less familiar)."""
    f = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    sims = f @ bank.T
    k = min(k, bank.shape[0])
    top = np.partition(sims, -k, axis=1)[:, -k:]
    return 1.0 - top.mean(axis=1)


def _features_and_probs(model, pil_images: list, **kwargs) -> tuple[list, np.ndarray | None]:
    """Run the model once; also capture the classifier head's input (the image embedding)."""
    results, emb = _run_hooked(model, pil_images, **kwargs)
    if emb is None and model.task == "classify":
        # The first call creates the predictor, whose AutoBackend runs its own copy of the model
        # (on a GPU always) that did not exist yet when the hooks were attached: retry once.
        results, emb = _run_hooked(model, pil_images, **kwargs)
    return results, emb


def _run_hooked(model, pil_images: list, **kwargs) -> tuple[list, np.ndarray | None]:
    # ultralytics' AutoBackend may run a deep copy of YOLO.model (after the first call), so hook
    # the classifier head of every module that might actually run; only one of them fires.
    candidates = []
    if model.task == "classify":
        candidates.append(model.model)
        runtime = getattr(getattr(getattr(model, "predictor", None), "model", None), "backend", None)
        if runtime is not None and hasattr(runtime, "model"):
            candidates.append(runtime.model)
    linears = {}
    for m in candidates:
        head = m.model[-1] if hasattr(m, "model") else None
        if head is not None and hasattr(head, "linear"):
            linears[id(head.linear)] = head.linear
    feats: list = []
    hooks = [lin.register_forward_pre_hook(lambda _m, inp: feats.append(inp[0].detach().float().cpu().numpy()))
             for lin in linears.values()]
    try:
        results = model.predict(pil_images, verbose=False, **kwargs)
    finally:
        for h in hooks:
            h.remove()
    emb = np.concatenate(feats, axis=0) if feats else None
    if emb is not None and len(emb) != len(results):
        emb = None
    return results, emb


def _leaf_probs(result, model) -> dict[str, float]:
    names = model.names
    if result.probs is not None:
        p = result.probs.data.float().cpu().numpy()
        return {names[i]: float(p[i]) for i in range(len(p))}
    best: dict[str, float] = {}  # detection checkpoint: highest box score per class
    if result.boxes is not None and len(result.boxes):
        for c, s in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            best[names[int(c)]] = max(best.get(names[int(c)], 0.0), float(s))
    return best


REASON_UNFAMILIAR = "the photo looks different from the training images"
REASON_CLOSE_CALL = "no material is clearly more likely than the others"
REASON_VERY_UNFAMILIAR = "the photo looks very different from anything the model was trained on"
REASON_OTHER = "it resembles photos users marked as another material"
REASON_SMALL_EWASTE = "small, low-resolution photo - the model over-predicts e-waste for these"
# Measured 2026-09-28: every ewaste_small training image is 150x150, and non-e-waste test photos
# shrunk to 100 px are called e-waste 15% of the time (70 px: 24%; full size: 0%). Until a
# retrain with low-resolution copies of all classes, a small photo predicted as e-waste is never
# shown as "detected".
SMALL_SIDE = 160


def small_image_check(result: dict, small: bool) -> dict:
    if (small and result.get("status") == "detected" and result.get("waste_type") == "e_waste"
            and not result.get("memory")):
        result["status"], result["confirm_reason"], result["is_confident"] = "confirm", REASON_SMALL_EWASTE, False
    result["small_image"] = small
    return result


def ood_status(type_conf: float, knn: float | None, knn_thr: float, far_thr: float | None,
               msp_thr: float, familiar: bool = False, other_mass: float = 0.0) -> tuple[str, list[str]]:
    """'unknown' only when there is real doubt about the material itself:
    extremely unfamiliar, or unfamiliar AND a close call, or similar to user-marked 'other' photos.
    One doubt alone -> 'confirm' (the best guess is shown and pre-filled). `familiar` = a confirmed
    feedback photo is within memory_threshold, which overrides the unfamiliarity checks."""
    unfamiliar = knn is not None and knn > knn_thr and not familiar
    very = knn is not None and far_thr is not None and knn > far_thr and not familiar
    close_call = type_conf < msp_thr
    if other_mass > type_conf:
        return "unknown", [REASON_OTHER]
    if very:
        return "unknown", [REASON_VERY_UNFAMILIAR]
    if unfamiliar and close_call:
        return "unknown", [REASON_UNFAMILIAR, REASON_CLOSE_CALL]
    if unfamiliar or close_call:
        return "confirm", [REASON_UNFAMILIAR if unfamiliar else REASON_CLOSE_CALL]
    return "detected", []


# Chosen by evaluate_memory.py on the VALIDATION split (best fixed-minus-broken); test, with val as
# memory: type 94.15% -> 94.44% (19 fixed, 5 broken of 4,944). Selection was not changed after test.
MEMORY_CURVE = "linear"


def memory_weight(d: np.ndarray, threshold: float, curve: str | None = None) -> np.ndarray:
    """How strongly a confirmed photo at cosine distance d pulls the prediction (0 at the threshold)."""
    x = np.clip(np.asarray(d) / threshold, 0, 1)
    curve = curve or MEMORY_CURVE
    if curve == "linear":
        return 1 - x
    if curve == "quadratic":
        return 1 - x ** 2
    if curve == "step":
        return np.ones_like(x)
    raise ValueError(curve)


def apply_memory(leaf_probs: dict[str, float], emb: np.ndarray, mem: dict, threshold: float) -> tuple[dict, dict | None]:
    """Blend the model's leaf probabilities toward the labels of confirmed feedback photos that are
    closer than `threshold`. Returns (new leaf_probs, memory dict for from_leaf_probs or None)."""
    e = emb / max(float(np.linalg.norm(emb)), 1e-8)
    d = 1.0 - mem["emb"] @ e
    near = np.nonzero(d <= threshold)[0]
    if not len(near):
        return leaf_probs, None
    w = memory_weight(d[near], threshold)
    lam = float(min(feedback_memory.MAX_WEIGHT, w.max()))
    target: dict[str, float] = {}
    other = 0.0
    for wi, idx in zip(w, near):
        wt, st = mem["labels"][idx]
        if wt == feedback_memory.OTHER:
            other += wi
            continue
        leaves = [l for l in leaf_probs if split_leaf(l)[0] == wt]
        if st and f"{wt}{SEP}{st}" in leaf_probs:
            target[f"{wt}{SEP}{st}"] = target.get(f"{wt}{SEP}{st}", 0.0) + wi
        elif leaves:  # type confirmed, sub-type not given: keep the model's own sub-type split
            tot = sum(leaf_probs[l] for l in leaves) or 1.0
            for l in leaves:
                target[l] = target.get(l, 0.0) + wi * (leaf_probs[l] / tot if tot else 1 / len(leaves))
    total = sum(target.values()) + other
    total = float(total)
    new = {l: float((1 - lam) * p + lam * target.get(l, 0.0) / total) for l, p in leaf_probs.items()}
    labels = [mem["labels"][i] for i in near]
    info = {"n_similar": int(len(near)), "nearest_distance": round(float(d[near].min()), 4),
            "threshold": round(threshold, 4), "weight": round(lam, 3),
            "labels": sorted({f"{a}/{b}" if b else a for a, b in labels}),
            "note": f"adjusted using {len(near)} similar photo{'s' if len(near) > 1 else ''} confirmed by users"}
    return new, {"familiar": True, "other_mass": float(lam * other / total), "info": info}


def from_leaf_probs(leaf_probs: dict[str, float], top_k: int = 3,
                    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                    knn: float | None = None, calib: dict | None = None,
                    model_task: str = "classify", memory: dict | None = None) -> dict:
    """Turn a leaf-probability dict into the public result (shared by single and combined)."""
    if not leaf_probs:
        return {"label": None, "waste_type": None, "sub_type": None, "confidence": 0.0,
                "type_confidence": 0.0, "subtype_confidence": 0.0, "leaf_confidence": 0.0,
                "is_confident": False, "subtype_is_confident": False, "top_k": [],
                "top_k_subtypes": [], "is_unknown": True, "unknown_reason": "no prediction",
                "best_guess": None, "ood": None, "leaf_probs": {}, "model_task": model_task,
                "model_version": "unknown"}
    hierarchical = any(SEP in k for k in leaf_probs)
    types: dict[str, float] = {}
    for leaf, p in leaf_probs.items():
        wt, _ = split_leaf(leaf)
        types[wt] = types.get(wt, 0.0) + p
    ranked_types = sorted(types.items(), key=lambda kv: kv[1], reverse=True)
    wt, type_conf = ranked_types[0]
    subs = sorted(((split_leaf(l)[1], p) for l, p in leaf_probs.items() if split_leaf(l)[0] == wt),
                  key=lambda kv: kv[1], reverse=True)
    st, leaf_conf = subs[0]
    sub_conf = leaf_conf / type_conf if type_conf > 0 else 0.0
    ranked_leaves = sorted(leaf_probs.items(), key=lambda kv: kv[1], reverse=True)

    if calib and calib.get("min_confidence") is not None:
        min_confidence = calib["min_confidence"]
    status, reasons = "detected", []
    ood = None
    if calib is not None:
        knn_thr = (calib.get("knn_threshold_by_type") or {}).get(wt, calib["knn_threshold"])
        far_thr = (calib.get("knn_far_by_type") or {}).get(wt)
        ood = {"knn_distance": None if knn is None else round(float(knn), 4),
               "knn_threshold": round(knn_thr, 4),
               "knn_far_threshold": None if far_thr is None else round(far_thr, 4),
               "knn_threshold_global": round(calib["knn_threshold"], 4),
               "knn_threshold_by_type": calib.get("knn_threshold_by_type") or {},
               "knn_far_by_type": calib.get("knn_far_by_type") or {},
               "memory_threshold": calib.get("memory_threshold"),
               "msp_threshold": round(calib["msp_threshold"], 4),
               "min_confidence": calib.get("min_confidence")}
        status, reasons = ood_status(type_conf, knn, knn_thr, far_thr, calib["msp_threshold"],
                                     familiar=bool(memory and memory.get("familiar")),
                                     other_mass=(memory or {}).get("other_mass", 0.0))
    is_unknown = status == "unknown"
    best_guess = {"waste_type": wt, "sub_type": st, "confidence": round(type_conf, 4)}
    out_wt = UNKNOWN if is_unknown else wt
    return {
        "label": out_wt,
        "waste_type": out_wt,
        "sub_type": None if is_unknown else st,
        "confidence": round(type_conf, 4),
        "type_confidence": round(type_conf, 4),
        "subtype_confidence": round(sub_conf, 4) if st is not None else None,
        "leaf_confidence": round(leaf_conf, 4),
        "is_confident": bool(type_conf >= min_confidence and status == "detected"),
        "status": status,
        "confirm_reason": "; ".join(reasons) if status == "confirm" else None,
        "subtype_is_confident": bool(st is not None and sub_conf >= DEFAULT_SUBTYPE_MIN_CONFIDENCE),
        "top_k": [{"label": t, "confidence": round(p, 4)} for t, p in ranked_types[:max(1, top_k)]],
        "top_k_subtypes": [{"label": l, "waste_type": split_leaf(l)[0], "sub_type": split_leaf(l)[1],
                            "confidence": round(p, 4)} for l, p in ranked_leaves[:max(1, top_k)]],
        "is_unknown": is_unknown,
        "unknown_reason": "; ".join(reasons) if is_unknown else None,
        "best_guess": best_guess,
        "ood": ood,
        "memory": (memory or {}).get("info"),
        "leaf_probs": {l: round(p, 5) for l, p in leaf_probs.items()},
        "model_task": model_task,
        "model_version": "v2_hierarchical" if hierarchical else "v1_flat",
    }


def predict_batch(images: Iterable[Any], weights: str | Path | None = None, top_k: int = 3,
                  min_confidence: float = DEFAULT_MIN_CONFIDENCE, imgsz: int | None = None,
                  device: str | None = None) -> list[dict]:
    """Classify several images independently. One result dict per image, in order."""
    model = load_model(weights)
    calib = ood_calibration(weights)
    pil_images = [_to_pil(im) for im in images]
    if not pil_images:
        return []
    kwargs: dict[str, Any] = {}
    if imgsz:
        kwargs["imgsz"] = imgsz
    if device:
        kwargs["device"] = device
    with _PREDICT_LOCK:
        results, emb = _features_and_probs(model, pil_images, **kwargs)
    knn = knn_distance(emb, calib["bank"], calib["k"]) if (calib and emb is not None) else [None] * len(results)
    mem = None
    if calib and calib.get("memory_threshold") and emb is not None and is_hierarchical(model):
        key = str((Path(weights) if weights else DEFAULT_WEIGHTS).expanduser().resolve())
        mem = feedback_memory.entries(key, lambda paths: embed_probs(paths, weights)[1])
    out = []
    for i, (r, d) in enumerate(zip(results, knn)):
        probs, memory = _leaf_probs(r, model), None
        if mem is not None:
            probs, memory = apply_memory(probs, emb[i], mem, calib["memory_threshold"])
            if memory:
                model_only = from_leaf_probs(_leaf_probs(r, model), 1, min_confidence, None, None, model.task)
                memory["info"]["model_guess"] = model_only["best_guess"]
        res = from_leaf_probs(probs, top_k, min_confidence, None if d is None else float(d),
                              calib, model.task, memory)
        out.append(small_image_check(res, min(pil_images[i].size) < SMALL_SIDE) if is_hierarchical(model) else res)
    return out


def embed_probs(images: list, weights: str | Path | None = None, batch_size: int = 64,
                progress: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Raw arrays for evaluation/calibration: (probabilities N x C, embeddings N x D, class names)."""
    model = load_model(weights)
    names = [model.names[i] for i in range(len(model.names))]
    P, F = [], []
    for s in range(0, len(images), batch_size):
        chunk = [_to_pil(im) for im in images[s:s + batch_size]]
        with _PREDICT_LOCK:
            results, emb = _features_and_probs(model, chunk)
        if emb is None:
            raise RuntimeError(f"no embeddings captured for images {s}-{s + len(chunk)} "
                               "(classifier head hook did not fire)")
        P.append(np.stack([r.probs.data.float().cpu().numpy() for r in results]))
        F.append(emb)
        if progress and (s // batch_size) % 20 == 0:
            print(f"  {progress}: {min(s + batch_size, len(images))}/{len(images)}", flush=True)
    return np.concatenate(P), np.concatenate(F), names


def predict(image: Any, weights: str | Path | None = None, top_k: int = 3,
            min_confidence: float = DEFAULT_MIN_CONFIDENCE, imgsz: int | None = None,
            device: str | None = None) -> dict:
    return predict_batch([image], weights, top_k, min_confidence, imgsz, device)[0]


def combine_predictions(per_image: list[dict], top_k: int = 3,
                        min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> dict:
    """Soft vote over photos of ONE item: average every leaf's probability across photos.

    The unknown check uses the mean embedding distance, so one odd close-up can't flip it.
    """
    if not per_image:
        raise ValueError("No predictions to combine")
    if all(r.get("leaf_probs") for r in per_image):
        leaves = {l for r in per_image for l in r["leaf_probs"]}
        mean = {l: sum(r["leaf_probs"].get(l, 0.0) for r in per_image) / len(per_image) for l in leaves}
        oods = [r.get("ood") for r in per_image]
        calib = None
        knn = None
        if all(o and o.get("knn_distance") is not None for o in oods):
            calib = {"knn_threshold": oods[0].get("knn_threshold_global", oods[0]["knn_threshold"]),
                     **{k: oods[0].get(k) for k in ("knn_threshold_by_type", "knn_far_by_type",
                                                    "memory_threshold", "msp_threshold", "min_confidence")}}
            knn = float(np.mean([o["knn_distance"] for o in oods]))
        mems = [r.get("memory") for r in per_image if r.get("memory")]
        memory = None
        if mems:  # at least one photo resembles a confirmed feedback photo
            memory = {"familiar": True, "other_mass": 0.0,
                      "info": {"n_similar": sum(m["n_similar"] for m in mems),
                               "nearest_distance": min(m["nearest_distance"] for m in mems),
                               "weight": max(m["weight"] for m in mems),
                               "labels": sorted({l for m in mems for l in m["labels"]}),
                               "note": f"adjusted using feedback on {len(mems)} of {len(per_image)} photos"}}
            if any(m["labels"] == ["other"] for m in mems) and len(mems) == len(per_image):
                memory["other_mass"] = 1.0
        out = from_leaf_probs(mean, top_k, min_confidence, knn, calib,
                              per_image[0].get("model_task", "classify"), memory)
        if out.get("model_version") == "v2_hierarchical":
            out = small_image_check(out, all(r.get("small_image") for r in per_image))
        ref = out["best_guess"]["waste_type"]
    else:  # results from an older predict() without leaf_probs
        labels = {t["label"] for r in per_image for t in r["top_k"]}
        avg = {lab: sum(next((t["confidence"] for t in r["top_k"] if t["label"] == lab), 0.0)
                        for r in per_image) / len(per_image) for lab in labels}
        ranked = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)
        out = {"label": ranked[0][0], "waste_type": ranked[0][0], "sub_type": None,
               "confidence": round(ranked[0][1], 4), "type_confidence": round(ranked[0][1], 4),
               "is_confident": ranked[0][1] >= min_confidence, "is_unknown": False,
               "top_k": [{"label": k, "confidence": round(v, 4)} for k, v in ranked[:max(1, top_k)]],
               "model_task": per_image[0].get("model_task", "classify")}
        ref = out["label"]
    out["n_images"] = len(per_image)
    out["agreement"] = sum((r.get("best_guess") or {}).get("waste_type", r["label"]) == ref for r in per_image)
    out["per_image"] = [{"label": r["label"], "waste_type": r.get("waste_type", r["label"]),
                         "sub_type": r.get("sub_type"), "confidence": r["confidence"],
                         "status": r.get("status"), "is_unknown": r.get("is_unknown", False),
                         "adjusted_by_feedback": bool(r.get("memory"))} for r in per_image]
    return out


def predict_group(images: Iterable[Any], same_item: bool, weights: str | Path | None = None,
                  top_k: int = 3, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> list[dict]:
    """Several photos: one combined result if they show the same item, else one per photo."""
    per_image = predict_batch(images, weights, top_k=top_k, min_confidence=min_confidence)
    if same_item:
        return [combine_predictions(per_image, top_k, min_confidence)]
    return per_image


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Predict waste type + sub-type for photos.")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--same-item", action="store_true", help="combine all photos into one prediction")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        load_model(args.weights)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if args.same_item:
        try:
            res = predict_group(args.images, True, args.weights, args.top_k)[0]
        except (FileNotFoundError, ValueError, TypeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        res.pop("leaf_probs", None)
        print(json.dumps(res, indent=2) if args.json else
              f"{len(args.images)} photos -> {res['waste_type']} / {res.get('sub_type')} "
              f"({res['confidence']:.2f}, {res['agreement']}/{res['n_images']} agree)")
        return 0
    output, failed = [], 0
    for path in args.images:
        try:
            r = predict(path, args.weights, args.top_k, device=args.device)
            r.pop("leaf_probs", None)
            r["image"] = path
        except (FileNotFoundError, ValueError, TypeError) as exc:
            r = {"image": path, "error": str(exc)}
            failed += 1
        output.append(r)
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        for r in output:
            if "error" in r:
                print(f"{r['image']}: ERROR - {r['error']}")
                continue
            if r["is_unknown"]:
                g = r["best_guess"]
                print(f"{r['image']}: UNKNOWN ({r['unknown_reason']}); best guess {g['waste_type']}/"
                      f"{g['sub_type']} {g['confidence']:.2f}")
                continue
            sub = f" / {r['sub_type']} ({r['subtype_confidence']:.2f})" if r["sub_type"] else ""
            flag = "" if r["is_confident"] else "  (low confidence - ask user to confirm)"
            print(f"{r['image']}: {r['waste_type']} ({r['type_confidence']:.2f}){sub}{flag}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
