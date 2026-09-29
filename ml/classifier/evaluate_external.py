"""
evaluate_external.py - independent external validation of the CURRENT classifier on an unseen dataset.

    python -m ml.classifier.evaluate_external                       # data/external_validation -> reports/external_validation/
    python -m ml.classifier.evaluate_external --limit 50            # smoke test (writes to a _smoke folder)

EVALUATION ONLY: no training, no tuning, weights are only read (their sha256 is checked before and after),
the dataset is only read, and nothing is added to any training set.

The dataset is COCO-format instance segmentation of conveyor-belt frames (labels.json per split, frames
in <split>/data, masks in <split>/sem_seg). A frame holds several objects, so there is no single label per
frame. Two evaluation units:
  1. crops (PRIMARY): every annotated object cropped from its bbox with the SAME rule the training set used
     for CODD boxes (10% padding, crops < 24 px skipped) -> one ground-truth label per crop;
  2. frames (secondary): the whole frame, labelled with the category covering the largest annotated area
     (what a user would upload as one photo of a mixed load).

Every image goes through predict_batch() exactly as the app does (same weights, calibration, "Other /
unknown" check and confirmed-feedback memory). "Accepted" = the app's green "Detected" badge =
result["is_confident"] (type confidence >= calibrated min_confidence AND status "detected").
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.classifier.predict import DEFAULT_WEIGHTS, ood_calibration, predict_batch  # noqa: E402
from ml.taxonomy import WASTE_TYPES  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "external_validation"
DEFAULT_OUT = ROOT / "ml" / "classifier" / "reports" / "external_validation"
PAD, MIN_CROP = 0.10, 24  # = train_yolo.py defaults used for the CODD crops of the training set
# dataset category -> (model waste type, model sub-type or None when the model has no equivalent)
GT_MAP = {"rigid_plastic": ("plastic", None), "soft_plastic": ("plastic", None),
          "cardboard": ("paper", "cardboard"), "metal": ("metal", "metal")}
THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95]
BATCH = 64


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash64(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def crop_box(img: Image.Image, bbox: list[float]) -> Image.Image | None:
    x, y, bw, bh = bbox
    w, h = img.size
    x0, y0 = max(0, int(x - bw * PAD)), max(0, int(y - bh * PAD))
    x1, y1 = min(w, int(round(x + bw + bw * PAD))), min(h, int(round(y + bh + bh * PAD)))
    if min(x1 - x0, y1 - y0) < MIN_CROP:
        return None
    return img.crop((x0, y0, x1, y1))


def error_type(r: dict) -> str:
    if r["status"] == "failed":
        return "unreadable"
    if r["status"] == "unknown":
        return "rejected_unknown"
    if r["correct"]:
        return "none" if r["accepted"] else "correct_low_confidence"
    return "confidently_misclassified" if r["accepted"] else "misclassified_uncertain"


def row_from_pred(base: dict, p: dict) -> dict:
    top = p.get("top_k") or []
    tops = [f"{t['label']} ({t['confidence']:.3f})" for t in top[:3]] + [""] * 3
    exp_t, exp_s = base["expected_waste_type"], base.get("expected_sub_type")
    r = {**base,
         "predicted_category": p["waste_type"],
         "predicted_subtype": p.get("sub_type") or "",
         "confidence": round(float(p["type_confidence"]), 4),
         "subtype_confidence": "" if p.get("subtype_confidence") is None else round(float(p["subtype_confidence"]), 4),
         "accepted": bool(p["is_confident"]),
         "correct": p["waste_type"] == exp_t,
         "status": p["status"],
         "best_guess": (p.get("best_guess") or {}).get("waste_type") or "",
         "best_guess_correct": (p.get("best_guess") or {}).get("waste_type") == exp_t,
         "subtype_correct": "" if exp_s is None else (p["waste_type"] == exp_t and p.get("sub_type") == exp_s),
         "reason": p.get("unknown_reason") or p.get("confirm_reason") or "",
         "knn_distance": (p.get("ood") or {}).get("knn_distance"),
         "memory_adjusted": bool(p.get("memory")),
         "top1": tops[0], "top2": tops[1], "top3": tops[2]}
    r["error_type"] = error_type(r)
    return r


def failed_row(base: dict, why: str) -> dict:
    r = {**base, "predicted_category": "", "predicted_subtype": "", "confidence": "", "subtype_confidence": "",
         "accepted": False, "correct": False, "status": "failed", "best_guess": "", "best_guess_correct": False,
         "subtype_correct": "", "reason": why, "knn_distance": "", "memory_adjusted": False,
         "top1": "", "top2": "", "top3": ""}
    r["error_type"] = "unreadable"
    return r


# ------------------------------------------------------------------ run

SPLITS = ("train", "val", "test")
DATA_NOTE = "Never used for training, calibration or tuning."


def run(data: Path, weights: Path, limit: int | None, do_frames: bool):
    audit = {"splits": {}, "formats": Counter(), "corrupt_frames": [], "corrupt_masks": [], "missing_files": [],
             "files_without_json": [], "frames_without_annotations": 0, "annotations_total": 0,
             "annotations_by_category": Counter(), "frames_by_category": Counter(),
             "crops_too_small": Counter(), "iscrowd": 0, "bad_bbox": 0, "image_sizes": Counter()}
    hashes: dict[str, list[str]] = defaultdict(list)
    dh: list[tuple[int, str]] = []
    crop_rows, frame_rows = [], []
    pend_crops: list[tuple[dict, Image.Image]] = []
    pend_frames: list[tuple[dict, Image.Image]] = []

    def flush(pend, rows):
        if not pend:
            return
        preds = predict_batch([im for _, im in pend], weights=weights, top_k=3)
        rows += [row_from_pred(b, p) for (b, _), p in zip(pend, preds)]
        pend.clear()

    t0, n_frames_done = time.time(), 0
    for split in SPLITS:
        sd = data / split
        if not (sd / "labels.json").exists():
            continue
        coco = json.load(open(sd / "labels.json", encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)
        on_disk = {p.name for p in (sd / "data").iterdir() if p.is_file()}
        masks = {p.name for p in (sd / "sem_seg").iterdir() if p.is_file()} if (sd / "sem_seg").exists() else set()
        in_json = {im["file_name"] for im in coco["images"]}
        audit["files_without_json"] += [f"{split}/data/{n}" for n in sorted(on_disk - in_json)]
        audit["splits"][split] = {"frames_in_json": len(coco["images"]), "frame_files": len(on_disk),
                                  "mask_files": len(masks), "annotations": len(coco["annotations"]),
                                  "categories": sorted(cats.values())}
        images = coco["images"][:limit] if limit else coco["images"]
        for im in images:
            rel = f"{split}/data/{im['file_name']}"
            path = sd / "data" / im["file_name"]
            audit["formats"][path.suffix.lower()] += 1
            if not path.exists():
                audit["missing_files"].append(rel)
                continue
            raw = path.read_bytes()
            hashes[hashlib.sha256(raw).hexdigest()].append(rel)
            try:
                frame = Image.open(io.BytesIO(raw))
                frame.load()
                frame = frame.convert("RGB")
            except Exception as exc:  # noqa: BLE001
                audit["corrupt_frames"].append(f"{rel}: {exc}")
                for a in anns.get(im["id"], []):
                    cat = cats[a["category_id"]]
                    crop_rows.append(failed_row(_crop_base(split, im, a, cat), "frame could not be decoded"))
                continue
            mpath = sd / "sem_seg" / im["file_name"]
            if mpath.exists():
                try:
                    with Image.open(mpath) as m:
                        m.verify()
                except Exception as exc:  # noqa: BLE001
                    audit["corrupt_masks"].append(f"{split}/sem_seg/{im['file_name']}: {exc}")
            audit["image_sizes"][f"{frame.size[0]}x{frame.size[1]}"] += 1
            dh.append((dhash64(frame), rel))

            fa = anns.get(im["id"], [])
            area_by_type: Counter = Counter()
            area_by_cat: Counter = Counter()
            for a in fa:
                cat = cats[a["category_id"]]
                audit["annotations_total"] += 1
                audit["annotations_by_category"][cat] += 1
                audit["iscrowd"] += int(bool(a.get("iscrowd")))
                area_by_cat[cat] += float(a.get("area") or a["bbox"][2] * a["bbox"][3])
                area_by_type[GT_MAP[cat][0]] += float(a.get("area") or a["bbox"][2] * a["bbox"][3])
                x, y, bw, bh = a["bbox"]
                if bw <= 0 or bh <= 0 or x < -1 or y < -1 or x + bw > frame.size[0] + 1 or y + bh > frame.size[1] + 1:
                    audit["bad_bbox"] += 1
                c = crop_box(frame, a["bbox"])
                if c is None:
                    audit["crops_too_small"][cat] += 1
                    continue
                pend_crops.append((_crop_base(split, im, a, cat), c))
            for cat in {cats[a["category_id"]] for a in fa}:
                audit["frames_by_category"][cat] += 1
            if not fa:
                audit["frames_without_annotations"] += 1
            if do_frames:
                if fa:
                    dom_cat = area_by_cat.most_common(1)[0][0]
                    dom_type, share = area_by_type.most_common(1)[0][0], area_by_type.most_common(1)[0][1] / sum(area_by_type.values())
                    base = {"unit": "frame", "split": split, "image_path": rel, "annotation_id": "",
                            "bbox": "", "actual_category": dom_cat, "expected_waste_type": GT_MAP[dom_cat][0],
                            "expected_sub_type": GT_MAP[dom_cat][1], "n_objects": len(fa),
                            "dominant_type_area_share": round(share, 3),
                            "categories_in_frame": "|".join(sorted({cats[a['category_id']] for a in fa}))}
                else:
                    base = {"unit": "frame", "split": split, "image_path": rel, "annotation_id": "", "bbox": "",
                            "actual_category": "no_annotation", "expected_waste_type": "none",
                            "expected_sub_type": None, "n_objects": 0, "dominant_type_area_share": "",
                            "categories_in_frame": ""}
                pend_frames.append((base, frame))
            if len(pend_crops) >= BATCH:
                flush(pend_crops, crop_rows)
            if len(pend_frames) >= 16:
                flush(pend_frames, frame_rows)
            n_frames_done += 1
            if n_frames_done % 250 == 0:
                el = time.time() - t0
                print(f"  {n_frames_done} frames, {len(crop_rows)} crops done, {el / 60:.1f} min", flush=True)
    flush(pend_crops, crop_rows)
    flush(pend_frames, frame_rows)

    audit["exact_duplicate_groups"] = [v for v in hashes.values() if len(v) > 1]
    audit["near_duplicates"] = near_duplicates(dh)
    return audit, crop_rows, frame_rows


def _crop_base(split: str, im: dict, a: dict, cat: str) -> dict:
    return {"unit": "crop", "split": split, "image_path": f"{split}/data/{im['file_name']}#ann{a['id']}",
            "annotation_id": a["id"], "bbox": " ".join(f"{v:.0f}" for v in a["bbox"]), "actual_category": cat,
            "expected_waste_type": GT_MAP[cat][0], "expected_sub_type": GT_MAP[cat][1]}


def near_duplicates(dh: list[tuple[int, str]], max_dist: int = 4) -> dict:
    """Pairs of frames whose 64-bit dHash differ in <= max_dist bits (consecutive video frames)."""
    if len(dh) < 2:
        return {"pairs": 0, "frames_involved": 0, "max_hamming": max_dist, "examples": []}
    h = np.array([x for x, _ in dh], dtype=np.uint64)
    names = [n for _, n in dh]
    pairs, involved, examples = 0, set(), []
    for i in range(len(h) - 1):
        x = np.bitwise_xor(h[i + 1:], h[i])
        d = np.array([bin(int(v)).count("1") for v in x]) if len(x) < 64 else _popcount(x)
        for j in np.nonzero(d <= max_dist)[0]:
            pairs += 1
            involved |= {names[i], names[i + 1 + j]}
            if len(examples) < 5:
                examples.append([names[i], names[i + 1 + j], int(d[j])])
    return {"pairs": pairs, "frames_involved": len(involved), "max_hamming": max_dist, "examples": examples}


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    c = np.zeros(x.shape, dtype=np.int64)
    for _ in range(64):
        c += (x & np.uint64(1)).astype(np.int64)
        x >>= np.uint64(1)
    return c


# ------------------------------------------------------------------ metrics

def summarise(rows: list[dict], key: str = "actual_category") -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        valid = [r for r in rs if r["status"] != "failed"]
        conf = [r["confidence"] for r in valid]
        n = len(valid)
        k = sum(r["correct"] for r in valid)
        out[g] = {
            "images": len(rs), "valid": n, "correct": k, "incorrect": n - k,
            "accuracy": k / n if n else math.nan, "accuracy_ci95": wilson(k, n),
            "accepted": sum(r["accepted"] for r in valid),
            "acceptance_rate": sum(r["accepted"] for r in valid) / n if n else math.nan,
            "strong": sum(r["correct"] and r["accepted"] for r in valid),
            "weak": sum(r["correct"] and not r["accepted"] for r in valid),
            "wrong": sum((not r["correct"]) and r["status"] != "unknown" for r in valid),
            "wrong_accepted": sum((not r["correct"]) and r["accepted"] for r in valid),
            "rejected_unknown": sum(r["status"] == "unknown" for r in valid),
            "confirm": sum(r["status"] == "confirm" for r in valid),
            "unreadable": len(rs) - n,
            "best_guess_correct": sum(r["best_guess_correct"] for r in valid),
            "memory_adjusted": sum(r["memory_adjusted"] for r in valid),
            "conf_mean": statistics.fmean(conf) if conf else math.nan,
            "conf_median": statistics.median(conf) if conf else math.nan,
            "conf_min": min(conf) if conf else math.nan, "conf_max": max(conf) if conf else math.nan,
            "ge90": sum(c >= 0.90 for c in conf) / n if n else math.nan,
            "ge80": sum(c >= 0.80 for c in conf) / n if n else math.nan,
            "ge50": sum(c >= 0.50 for c in conf) / n if n else math.nan,
            "conf_correct_mean": statistics.fmean([r["confidence"] for r in valid if r["correct"]]) if k else math.nan,
            "conf_wrong_mean": statistics.fmean([r["confidence"] for r in valid if not r["correct"]]) if n - k else math.nan,
            "confidently_wrong_90": sum((not r["correct"]) and r["status"] != "unknown" and r["confidence"] >= 0.90 for r in valid),
            "subtype_n": sum(r["subtype_correct"] != "" and r["correct"] for r in valid),
            "subtype_correct": sum(r["subtype_correct"] is True for r in valid),
            "predicted": Counter(r["predicted_category"] for r in valid),
            "predicted_subtypes": Counter(f"{r['predicted_category']}/{r['predicted_subtype']}" for r in valid
                                          if r["predicted_category"] != "unknown"),
        }
    return out


def overall(rows: list[dict]) -> dict:
    return summarise([{**r, "_all": "all"} for r in rows], "_all")["all"]


def threshold_table(rows: list[dict], min_conf_prod: float) -> list[dict]:
    """What-if only: vary the confidence threshold, keep every other part of the pipeline as it is."""
    valid = [r for r in rows if r["status"] != "failed"]
    n = len(valid)
    out = []
    for t in [*THRESHOLDS, round(min_conf_prod, 4)]:
        for mode in ("pipeline", "confidence_only"):
            acc = [r for r in valid if r["confidence"] >= t and (mode == "confidence_only" or r["status"] == "detected")]
            ok = sum(r["correct"] or (mode == "confidence_only" and r["best_guess_correct"]) for r in acc) \
                if mode == "confidence_only" else sum(r["correct"] for r in acc)
            out.append({"threshold": t, "mode": mode, "accepted": len(acc), "acceptance_rate": len(acc) / n,
                        "correct_acceptance_rate": ok / n, "false_acceptance_rate": (len(acc) - ok) / n,
                        "precision_of_accepted": ok / len(acc) if acc else math.nan})
    return out


# ------------------------------------------------------------------ report

def pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


CSV_COLS = ["image_path", "actual_category", "predicted_category", "predicted_subtype", "confidence", "accepted",
            "correct", "top1", "top2", "top3", "error_type",
            # extra context
            "unit", "split", "annotation_id", "bbox", "expected_waste_type", "expected_sub_type", "subtype_confidence",
            "subtype_correct", "status", "reason", "best_guess", "best_guess_correct", "knn_distance",
            "memory_adjusted"]


def confusion(rows: list[dict], actual_key: str = "actual_category") -> tuple[list[str], list[str], dict]:
    valid = [r for r in rows if r["status"] != "failed"]
    actual = sorted({r[actual_key] for r in valid})
    predicted = [t for t in WASTE_TYPES if any(r["predicted_category"] == t for r in valid)] + \
                (["unknown"] if any(r["predicted_category"] == "unknown" for r in valid) else [])
    m = {a: Counter(r["predicted_category"] for r in valid if r[actual_key] == a) for a in actual}
    return actual, predicted, m


def plot_confusion(actual, predicted, m, path: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False
    arr = np.array([[m[a][p] for p in predicted] for a in actual], dtype=float)
    norm = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(predicted) + 2.5, 0.8 * len(actual) + 1.8))
    ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(predicted)), predicted, rotation=40, ha="right")
    ax.set_yticks(range(len(actual)), actual)
    for i in range(len(actual)):
        for j in range(len(predicted)):
            if arr[i, j]:
                ax.text(j, i, f"{int(arr[i, j])}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xlabel("predicted (app output)")
    ax.set_ylabel("actual (dataset label)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def contact_sheet(rows: list[dict], data: Path, path: Path, n: int = 24) -> bool:
    """Examples of accepted-but-wrong crops (the most dangerous errors), for a human look."""
    bad = sorted([r for r in rows if r["error_type"] == "confidently_misclassified" and r["unit"] == "crop"],
                 key=lambda r: -r["confidence"])
    by_cat = defaultdict(list)
    for r in bad:
        by_cat[r["actual_category"]].append(r)
    pick = []
    while len(pick) < n and any(by_cat.values()):
        for c in list(by_cat):
            if by_cat[c]:
                pick.append(by_cat[c].pop(0))
    if not pick:
        return False
    from PIL import ImageDraw
    tile, cols = 180, 6
    sheet = Image.new("RGB", (cols * tile, math.ceil(len(pick) / cols) * (tile + 30)), "white")
    for i, r in enumerate(pick):
        frame_rel = r["image_path"].split("#")[0]
        img = Image.open(data / frame_rel).convert("RGB")
        x, y, bw, bh = (float(v) for v in r["bbox"].split())
        c = crop_box(img, [x, y, bw, bh]) or img
        c.thumbnail((tile, tile))
        ox, oy = (i % cols) * tile, (i // cols) * (tile + 30)
        sheet.paste(c, (ox + (tile - c.width) // 2, oy + (tile - c.height) // 2))
        ImageDraw.Draw(sheet).text((ox + 4, oy + tile + 2),
                                   f"{r['actual_category']} -> {r['predicted_category']} {r['confidence']:.2f}", fill="black")
    sheet.save(path)
    return True


def report(out: Path, data: Path, weights: Path, sha: str, audit: dict, crops: list[dict], frames: list[dict],
           minutes: float) -> None:
    calib = ood_calibration(weights)
    min_conf = calib["min_confidence"]
    per_cat = summarise(crops)
    all_c = overall(crops)
    by_type = summarise(crops, "expected_waste_type")
    lab_frames = [r for r in frames if r["actual_category"] != "no_annotation"]
    empty_frames = [r for r in frames if r["actual_category"] == "no_annotation"]
    per_frame_cat = summarise(lab_frames) if lab_frames else {}
    all_f = overall(lab_frames) if lab_frames else None
    pure = [r for r in lab_frames if r["dominant_type_area_share"] != "" and r["dominant_type_area_share"] >= 0.9]
    pure_sum = summarise(pure, "expected_waste_type") if pure else {}
    thr = threshold_table(crops, min_conf)
    actual, predicted, m = confusion(crops)
    has_png = plot_confusion(actual, predicted, m, out / "confusion_crops.png",
                             "External validation: crops (rows normalised)")
    if lab_frames:
        fa, fp, fm = confusion(lab_frames)
        plot_confusion(fa, fp, fm, out / "confusion_frames.png", "External validation: frames (dominant category)")
    has_sheet = contact_sheet(crops, data, out / "confidently_misclassified_examples.png")

    pairs = sorted(((a, p, c) for a in actual for p, c in m[a].items() if p != GT_MAP[a][0]),
                   key=lambda t: -t[2])
    n_valid = all_c["valid"]

    # ---- category classes for the headline lists
    def klass(s: dict) -> str:
        if s["accuracy"] >= 0.8 and s["acceptance_rate"] >= 0.5 and s["strong"] / s["valid"] >= 0.5:
            return "strong"
        if s["accuracy"] >= 0.6:
            return "weak"
        return "misclassified"

    strong = [c for c, s in per_cat.items() if klass(s) == "strong"]
    weak = [c for c, s in per_cat.items() if klass(s) == "weak"]
    miscl = [c for c, s in per_cat.items() if klass(s) == "misclassified"]

    L = []
    w = L.append
    w("# External ML Validation Report\n")
    w(f"**Model:** `ml/classifier/weights/best.pt`: v2 hierarchical YOLOv8n-cls (9 waste types / 27 sub-types), "
      f"sha256 `{sha[:16]}…` (unchanged before/after this run) + `best.ood.npz` calibration + the app's confirmed-"
      f"feedback memory  ")
    w(f"**Dataset:** `data/external_validation`: COCO instance-segmentation of waste on a sorting-line conveyor belt "
      f"(4 categories: rigid_plastic, soft_plastic, cardboard, metal; structure and categories match the public "
      f"ZeroWaste-f dataset). {DATA_NOTE}  ")
    w(f"**Date:** {datetime.now():%Y-%m-%d} · run time {minutes:.1f} min on the laptop CPU  ")
    w(f"**Evaluation unit:** object crops (primary, {n_valid:,} crops from {audit['annotations_total']:,} annotations "
      f"in {sum(s['frames_in_json'] for s in audit['splits'].values()):,} frames); whole frames reported separately (§10)\n")
    w("| Headline (crops) | Value |\n|---|---:|")
    w(f"| Total images evaluated (crops) | {all_c['images']:,} |")
    w(f"| Overall accuracy (waste type) | **{pct(all_c['accuracy'])}** (95% CI {pct(all_c['accuracy_ci95'][0])}–{pct(all_c['accuracy_ci95'][1])}) |")
    w(f"| Overall acceptance rate | **{pct(all_c['acceptance_rate'])}** |")
    w(f"| Correct acceptance rate | **{pct(all_c['strong'] / n_valid)}** |")
    w(f"| False acceptance rate | **{pct(all_c['wrong_accepted'] / n_valid)}** |")
    w(f"| Average confidence | {pct(all_c['conf_mean'])} |")
    w(f"| Flagged \"Other / unknown\" | {pct(all_c['rejected_unknown'] / n_valid)} ({all_c['rejected_unknown']:,}) |")
    w(f"| Best-guess accuracy (ignoring the unknown flag) | {pct(all_c['best_guess_correct'] / n_valid)} |")
    if all_f:
        w(f"| Whole frames (dominant category), accuracy | {pct(all_f['accuracy'])} of {all_f['valid']:,} |")
    w("")
    w(f"Acceptance = what the app shows with the green **Detected** badge: waste-type confidence ≥ "
      f"**{min_conf:.4f}** (calibrated `min_confidence` in best.ood.npz) **and** status `detected` (not \"please "
      f"confirm\" / \"unknown\"). Production thresholds were not changed.\n")
    w("### Strongly Recognized Categories\n")
    w(("\n".join(f"- **{c}**: {pct(per_cat[c]['accuracy'])} accurate, {pct(per_cat[c]['strong'] / per_cat[c]['valid'])} strongly recognized" for c in strong)) or "- none (no category reaches ≥80% accuracy with ≥50% correctly accepted)")
    w("\n### Weak / Uncertain Categories\n")
    w(("\n".join(f"- **{c}**: {pct(per_cat[c]['accuracy'])} accurate, acceptance {pct(per_cat[c]['acceptance_rate'])}" for c in weak)) or "- none")
    w("\n### Frequently Misclassified Categories\n")
    w(("\n".join(f"- **{c}**: {pct(per_cat[c]['accuracy'])} accurate; mostly predicted as "
                 + ", ".join(f"{p} ({n / per_cat[c]['valid']:.0%})" for p, n in per_cat[c]['predicted'].most_common(3))
                 for c in miscl)) or "- none (every category ≥ 60% accurate)")
    w("\n### Unsupported / OOD Categories\n")
    w("- **None of the 4 dataset categories is outside the model's classes** (plastic, paper→cardboard and metal "
      "are all trained classes). What is out-of-distribution is the **domain**: dirty, crumpled, overlapping objects "
      "on a conveyor belt under industrial lighting, vs. the mostly clean single-object photos the model was trained on. "
      "§8 measures how the model handles that shift.")
    if empty_frames:
        s = overall(empty_frames)
        w(f"- **{len(empty_frames)} frames contain no annotated object** (belt / unlabelled residue only), a "
          f"natural \"nothing supported here\" probe: flagged unknown {pct(s['rejected_unknown'] / s['valid'])}, "
          f"accepted as some material {pct(s['acceptance_rate'])} (§8).")
    w("")

    w("## 1. Dataset audit\n")
    w("| Split | Frames (json) | Frame files | Mask files | Annotations |\n|---|---:|---:|---:|---:|")
    for sp, s in audit["splits"].items():
        w(f"| {sp} | {s['frames_in_json']:,} | {s['frame_files']:,} | {s['mask_files']:,} | {s['annotations']:,} |")
    w(f"| **total** | {sum(s['frames_in_json'] for s in audit['splits'].values()):,} | "
      f"{sum(s['frame_files'] for s in audit['splits'].values()):,} | {sum(s['mask_files'] for s in audit['splits'].values()):,} | "
      f"{sum(s['annotations'] for s in audit['splits'].values()):,} |\n")
    w(f"- Categories (identical in every split): {', '.join(next(iter(audit['splits'].values()))['categories'])}")
    w(f"- File formats: {dict(audit['formats'])} frames (+ PNG masks in `sem_seg/`, one `labels.json` per split); "
      f"frame sizes: {dict(audit['image_sizes'])}")
    w(f"- Corrupt / undecodable frames: **{len(audit['corrupt_frames'])}**; corrupt masks: **{len(audit['corrupt_masks'])}**; "
      f"missing files: {len(audit['missing_files'])}; frame files not in labels.json: {len(audit['files_without_json'])}")
    w(f"- Frames without any annotation: {audit['frames_without_annotations']:,}; crowd annotations: {audit['iscrowd']}; "
      f"boxes outside the frame / empty: {audit['bad_bbox']}")
    w(f"- Exact duplicate frames (sha256): **{sum(len(g) - 1 for g in audit['exact_duplicate_groups'])}** extra copies in "
      f"{len(audit['exact_duplicate_groups'])} groups")
    nd = audit["near_duplicates"]
    w(f"- Near-duplicate frames (64-bit dHash ≤ {nd['max_hamming']} bits): {nd['pairs']:,} pairs involving "
      f"{nd['frames_involved']:,} frames, expected, the frames are consecutive video frames (every 10th frame). "
      f"Nothing was removed; per-category numbers therefore contain correlated samples, so the confidence intervals "
      f"below are optimistic.")
    w("\n| Category | Annotations | Frames containing it | Crops evaluated | Crops < 24 px (skipped, as in training) |\n|---|---:|---:|---:|---:|")
    for c in sorted(audit["annotations_by_category"]):
        w(f"| {c} | {audit['annotations_by_category'][c]:,} | {audit['frames_by_category'][c]:,} | "
          f"{per_cat.get(c, {}).get('images', 0):,} | {audit['crops_too_small'][c]:,} |")
    w("\nGround truth → model classes: rigid_plastic → plastic, soft_plastic → plastic, cardboard → paper "
      "(sub-type cardboard), metal → metal. The model's plastic sub-types (plastic, foam) have no rigid/soft split, "
      "so plastic sub-types are not scored.\n")

    w("## 2–3. Category-level results (crops)\n")
    w("| Waste Category | Images | Correct | Incorrect | Accuracy | 95% CI | Accepted | Acceptance Rate | Avg Confidence |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c, s in list(per_cat.items()) + [("**all**", all_c)]:
        w(f"| {c} | {s['images']:,} | {s['correct']:,} | {s['incorrect']:,} | {pct(s['accuracy'])} | "
          f"{pct(s['accuracy_ci95'][0], 0)}–{pct(s['accuracy_ci95'][1], 0)} | {s['accepted']:,} | "
          f"{pct(s['acceptance_rate'])} | {pct(s['conf_mean'])} |")
    w("\n\"Incorrect\" includes crops the app flags as \"Other / unknown\" (it does not output the right material). "
      "By model waste type (rigid + soft plastic merged):\n")
    w("| Model waste type | Crops | Accuracy | Acceptance | Sub-type correct (when type correct) |\n|---|---:|---:|---:|---:|")
    for c, s in by_type.items():
        sub = f"{s['subtype_correct']:,} / {s['subtype_n']:,} ({pct(s['subtype_correct'] / s['subtype_n'])})" if s["subtype_n"] else "not scored"
        w(f"| {c} | {s['images']:,} | {pct(s['accuracy'])} | {pct(s['acceptance_rate'])} | {sub} |")
    w("")

    w("## 4. Recognition summary\n")
    w("| Category | Strongly Recognized | Weak/Uncertain | Incorrect | Failed | Recognition Rate |\n|---|---:|---:|---:|---:|---:|")
    for c, s in list(per_cat.items()) + [("**all**", all_c)]:
        w(f"| {c} | {s['strong']:,} | {s['weak']:,} | {s['wrong']:,} | {s['rejected_unknown'] + s['unreadable']:,} | "
          f"{pct(s['strong'] / s['valid'])} |")
    w("\n- Strongly recognized = correct AND accepted (Detected badge). Weak/uncertain = correct but not accepted "
      "(\"please confirm\" or below the threshold). Incorrect = a wrong material was output. Failed = the app output "
      "\"Other / unknown\" (no valid material) or the image could not be processed "
      f"({all_c['unreadable']} unreadable). Recognition rate = strongly recognized / valid images.\n")

    w("## 5. Confidence analysis (waste-type confidence, crops)\n")
    w("| Category | Mean | Median | Min | Max | ≥ 0.90 | ≥ 0.80 | ≥ 0.50 | Mean when correct | Mean when wrong |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c, s in list(per_cat.items()) + [("**all**", all_c)]:
        w(f"| {c} | {pct(s['conf_mean'])} | {pct(s['conf_median'])} | {pct(s['conf_min'])} | {pct(s['conf_max'])} | "
          f"{pct(s['ge90'])} | {pct(s['ge80'])} | {pct(s['ge50'])} | {pct(s['conf_correct_mean'])} | {pct(s['conf_wrong_mean'])} |")
    w("\nFor crops flagged unknown the confidence is that of the model's best guess.\n")

    w("## 6. Acceptance rate (production threshold)\n")
    w("| Metric | Definition | Value |\n|---|---|---:|")
    w(f"| Acceptance rate | accepted / valid images | **{pct(all_c['acceptance_rate'])}** ({all_c['accepted']:,} / {n_valid:,}) |")
    w(f"| Correct acceptance rate | correct AND accepted / valid images | **{pct(all_c['strong'] / n_valid)}** ({all_c['strong']:,}) |")
    w(f"| False acceptance rate | incorrect BUT accepted / valid images | **{pct(all_c['wrong_accepted'] / n_valid)}** ({all_c['wrong_accepted']:,}) |")
    acc_prec = all_c["strong"] / all_c["accepted"] if all_c["accepted"] else math.nan
    w(f"| Precision of accepted | correct / accepted | {pct(acc_prec)} |")
    w("\n| Category | Acceptance | Correct acceptance | False acceptance | Precision of accepted |\n|---|---:|---:|---:|---:|")
    for c, s in per_cat.items():
        w(f"| {c} | {pct(s['acceptance_rate'])} | {pct(s['strong'] / s['valid'])} | {pct(s['wrong_accepted'] / s['valid'])} | "
          f"{pct(s['strong'] / s['accepted']) if s['accepted'] else 'n/a'} |")
    w("")

    w("## 7. Confusion analysis (crops)\n")
    w("Rows = dataset category, columns = app output (waste type, or `unknown`).\n")
    w("| Actual \\ Predicted | " + " | ".join(predicted) + " | Total |\n|---|" + "---:|" * (len(predicted) + 1))
    for a in actual:
        tot = sum(m[a].values())
        w(f"| {a} | " + " | ".join(f"{m[a][p]:,} ({m[a][p] / tot:.0%})" if m[a][p] else "·" for p in predicted) + f" | {tot:,} |")
    if has_png:
        w("\n![confusion](confusion_crops.png)")
    w("\n**Top confusion pairs** (actual → predicted, excluding correct):\n")
    w("| # | Actual → Predicted | Count | Share of the category | of which accepted |\n|---:|---|---:|---:|---:|")
    for i, (a, p, c) in enumerate(pairs[:10], 1):
        accd = sum(r["actual_category"] == a and r["predicted_category"] == p and r["accepted"] for r in crops)
        w(f"| {i} | {a} → {p} | {c:,} | {c / sum(m[a].values()):.1%} | {accd:,} |")
    w("\nPredicted sub-types per category (top 5):\n")
    for c, s in per_cat.items():
        w(f"- **{c}**: " + ", ".join(f"{k} {v:,}" for k, v in s["predicted_subtypes"].most_common(5)))
    w("")

    w("## 8. Unknown / out-of-distribution behaviour\n")
    w("No category is outside the model's classes, so this section asks: on unfamiliar-looking (domain-shifted) "
      "images, does the model **flag** its doubt, or does it **confidently force** a wrong material?\n")
    w("| Category | Confidently misclassified (wrong + accepted) | Wrong with conf ≥ 0.90 | Wrong but flagged (confirm) | Flagged unknown | Of unknown: best guess was right |")
    w("|---|---:|---:|---:|---:|---:|")
    for c, s in list(per_cat.items()) + [("**all**", all_c)]:
        rs = [r for r in crops if c == "**all**" or r["actual_category"] == c]
        wrong_confirm = sum((not r["correct"]) and r["status"] == "confirm" for r in rs)
        unk_ok = sum(r["status"] == "unknown" and r["best_guess_correct"] for r in rs)
        w(f"| {c} | {s['wrong_accepted']:,} ({pct(s['wrong_accepted'] / s['valid'])}) | {s['confidently_wrong_90']:,} | "
          f"{wrong_confirm:,} | {s['rejected_unknown']:,} ({pct(s['rejected_unknown'] / s['valid'])}) | {unk_ok:,} |")
    wrong = [r for r in crops if not r["correct"] and r["status"] != "failed" and r["status"] != "unknown"]
    caught = sum(not r["accepted"] for r in wrong)
    w(f"\nOf the {len(wrong):,} crops where the app output a wrong material, {caught:,} ({pct(caught / len(wrong) if wrong else math.nan)}) "
      f"were at least marked \"please confirm\" (not accepted) and {len(wrong) - caught:,} were shown as **Detected**: "
      f"these are the confidently misclassified cases. Example: the top-confidence accepted errors are shown in "
      f"`confidently_misclassified_examples.png`." if has_sheet else "")
    knn = [float(r["knn_distance"]) for r in crops if r["knn_distance"] not in ("", None)]
    if knn:
        thr_by = calib.get("knn_threshold_by_type") or {}
        far_by = calib.get("knn_far_by_type") or {}
        unfam = sum(float(r["knn_distance"]) > thr_by.get(r["best_guess"], calib["knn_threshold"]) for r in crops
                    if r["knn_distance"] not in ("", None) and r["best_guess"])
        very = sum(float(r["knn_distance"]) > far_by.get(r["best_guess"], 9) for r in crops
                   if r["knn_distance"] not in ("", None) and r["best_guess"])
        w(f"\nEmbedding familiarity (kNN cosine distance to the training bank): median {statistics.median(knn):.3f}; "
          f"{pct(unfam / len(knn))} of crops are beyond their type's \"unfamiliar\" threshold and {pct(very / len(knn))} "
          f"beyond the \"very unfamiliar\" one (calibrated at the 97.5th / 99.5th percentile of the in-domain "
          f"validation split, i.e. ~2.5% / ~0.5% of normal images exceed them). "
          f"So the OOD detector does see the domain shift, but a large, confident softmax often keeps the result at "
          f"\"confirm\" rather than \"unknown\".")
    if empty_frames:
        s = overall(empty_frames)
        w(f"\n**Frames with no annotated object** ({len(empty_frames)}): output " +
          ", ".join(f"{k} {v}" for k, v in s["predicted"].most_common()) +
          f"; accepted as a material {s['accepted']} ({pct(s['acceptance_rate'])}). These frames still show the belt "
          f"and unlabelled residue, so a material answer is not necessarily wrong; reported for information only.")
    w("")

    w("## 9. Threshold analysis (what-if only, production threshold NOT changed)\n")
    w(f"Production: min_confidence **{min_conf:.4f}** + status `detected`. `pipeline` keeps the status rule and "
      "varies only the confidence threshold; `confidence_only` ignores the unknown/confirm status and accepts the "
      "best guess whenever its confidence ≥ threshold.\n")
    w("| Threshold | Mode | Accepted | Acceptance rate | Correct acceptance | False acceptance | Precision of accepted |")
    w("|---:|---|---:|---:|---:|---:|---:|")
    for t in thr:
        mark = " (production)" if abs(t["threshold"] - round(min_conf, 4)) < 1e-9 and t["mode"] == "pipeline" else ""
        w(f"| {t['threshold']:.4g}{mark} | {t['mode']} | {t['accepted']:,} | {pct(t['acceptance_rate'])} | "
          f"{pct(t['correct_acceptance_rate'])} | {pct(t['false_acceptance_rate'])} | {pct(t['precision_of_accepted'])} |")
    w("")

    if lab_frames:
        w("## 10. Whole frames (secondary)\n")
        w("Each frame is labelled with the category covering the largest annotated area; frames usually contain "
          "several materials plus unlabelled residue, so this is a harder, noisier test that resembles a user "
          "photographing a mixed load.\n")
        w("| Dominant category | Frames | Accuracy | Acceptance | False acceptance | Avg confidence | Predicted as (top 3) |")
        w("|---|---:|---:|---:|---:|---:|---|")
        for c, s in list(per_frame_cat.items()) + [("**all**", all_f)]:
            w(f"| {c} | {s['valid']:,} | {pct(s['accuracy'])} | {pct(s['acceptance_rate'])} | "
              f"{pct(s['wrong_accepted'] / s['valid'])} | {pct(s['conf_mean'])} | "
              + ", ".join(f"{k} {v / s['valid']:.0%}" for k, v in s["predicted"].most_common(3)) + " |")
        if pure_sum:
            w(f"\nFrames where one model waste type covers ≥ 90% of the annotated area ({len(pure):,}): " +
              "; ".join(f"{c} {pct(s['accuracy'])} accurate (n={s['valid']:,})" for c, s in pure_sum.items()) + ".")
        w("")

    w("## Files\n")
    w("- `external_validation_results.csv`: one row per evaluated crop (the required columns first, then context: "
      "split, bbox, status, reason, best guess, kNN distance, memory adjustment)")
    w("- `external_validation_results_frames.csv`: one row per frame (secondary test)")
    w("- `metrics.json`, `dataset_audit.json`, `confusion_crops.png`, `confusion_frames.png`, "
      "`confidently_misclassified_examples.png`, `EXTERNAL_VALIDATION_CONCLUSION.md`")
    w(f"\nReproduce: `python -m ml.classifier.evaluate_external` (reads `{data.relative_to(ROOT)}`, writes a new folder).")
    (out / "EXTERNAL_VALIDATION_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    metrics = {"model": str(weights.relative_to(ROOT)), "weights_sha256": sha, "min_confidence": min_conf,
               "date": f"{datetime.now():%Y-%m-%d}", "crops": _jsonable(all_c), "per_category": _jsonable(per_cat),
               "per_waste_type": _jsonable(by_type), "frames": _jsonable(all_f) if all_f else None,
               "per_frame_category": _jsonable(per_frame_cat), "thresholds": thr,
               "top_confusions": [{"actual": a, "predicted": p, "count": c} for a, p, c in pairs[:10]]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, float) and math.isnan(d):
        return None
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="frames per split (smoke test)")
    ap.add_argument("--no-frames", action="store_true", help="skip the whole-frame test")
    ap.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=None,
                    help="evaluate only these splits (e.g. test, for models trained on the train/val crops)")
    a = ap.parse_args(argv)
    global SPLITS, DATA_NOTE
    if a.splits:
        SPLITS = tuple(a.splits)
        DATA_NOTE = (f"Only split(s) {', '.join(SPLITS)} evaluated. Models trained with build_domain_dataset saw "
                     "crops of the train/val splits only; the test split was never used for training, calibration or tuning.")
    out = a.out or (DEFAULT_OUT.parent / f"{DEFAULT_OUT.name}_smoke" if a.limit else DEFAULT_OUT)
    if (out / "external_validation_results.csv").exists() and not a.limit:
        print(f"{out} already holds results - not overwriting. Pass --out <new folder>.")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    weights = a.weights.resolve()
    sha_before = sha256_file(weights)
    ood_before = sha256_file(weights.with_suffix(".ood.npz"))
    t0 = time.time()
    audit, crops, frames = run(a.data.resolve(), weights, a.limit, not a.no_frames)
    minutes = (time.time() - t0) / 60
    assert sha256_file(weights) == sha_before and sha256_file(weights.with_suffix(".ood.npz")) == ood_before, \
        "weights changed during evaluation!"
    write_csv(out / "external_validation_results.csv", crops, CSV_COLS)
    if frames:
        write_csv(out / "external_validation_results_frames.csv", frames,
                  CSV_COLS + ["n_objects", "dominant_type_area_share", "categories_in_frame"])
    audit_json = {**audit, "formats": dict(audit["formats"]), "annotations_by_category": dict(audit["annotations_by_category"]),
                  "frames_by_category": dict(audit["frames_by_category"]), "crops_too_small": dict(audit["crops_too_small"]),
                  "image_sizes": dict(audit["image_sizes"])}
    (out / "dataset_audit.json").write_text(json.dumps(audit_json, indent=2), encoding="utf-8")
    m = report(out, a.data.resolve(), weights, sha_before, audit, crops, frames, minutes)
    c = m["crops"]
    print(f"done in {minutes:.1f} min -> {out}")
    print(f"crops {c['valid']}: accuracy {c['accuracy']:.3f}, acceptance {c['acceptance_rate']:.3f}, "
          f"false acceptance {c['wrong_accepted'] / c['valid']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
