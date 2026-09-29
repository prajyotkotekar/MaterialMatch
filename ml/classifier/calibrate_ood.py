"""
calibrate_ood.py - Calibrate the "Other / unknown material" check for a trained classifier.

    python -m ml.classifier.calibrate_ood --weights ml/classifier/weights/best_candidate.pt

Writes <weights>.ood.npz, which predict.py loads automatically:
  bank            L2-normalised embeddings of up to --per-class TRAIN images per class (default 800)
  k               neighbours used for the kNN distance
  knn_threshold   global kNN cosine distance above which an image is "unlike any training image";
  knn_threshold_by_type  the same, per predicted waste type (used when present)
  knn_far_by_type 99.5th percentile per type: beyond it a photo is "unknown" even if confident
  msp_threshold   waste-type probability below which no material clearly wins
  min_confidence  type probability above which the UI says "Detected" instead of "Please confirm"
  memory_threshold nearest-neighbour distance below which a neighbour's waste type is right
                  >= 98% of the time; the feedback memory only uses confirmed photos this close
How they are combined: predict.ood_status.

All thresholds come from the in-distribution VALIDATION split only (no out-of-distribution
images are used for tuning): the combined rule keeps --id-retention of validation images, and
min_confidence keeps 95% of correctly classified validation images.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from ml.classifier.predict import embed_probs, knn_distance, split_leaf

ROOT = Path(__file__).resolve().parents[2]


def list_split(ds: Path, split: str, per_class: int | None = None, seed: int = 0):
    out = []
    rng = random.Random(seed)
    for d in sorted(p for p in (ds / split).iterdir() if p.is_dir()):
        files = sorted(f for f in d.glob("*.jpg") if "_rep" not in f.stem)
        if per_class and len(files) > per_class:
            files = rng.sample(files, per_class)
        out += [(f, d.name) for f in files]
    return out


def type_probs(P: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    types = sorted({split_leaf(n)[0] for n in names})
    M = np.zeros((len(names), len(types)), dtype=np.float32)
    for i, n in enumerate(names):
        M[i, types.index(split_leaf(n)[0])] = 1
    return P @ M, types


def calibrate(weights: Path, ds: Path, per_class: int = 800, k: int = 10,
              id_retention: float = 0.95, out: Path | None = None) -> dict:
    bank_items = list_split(ds, "train", per_class)
    _, Fb, names = embed_probs([f for f, _ in bank_items], weights, progress="bank")
    bank = Fb / np.linalg.norm(Fb, axis=1, keepdims=True)
    val = list_split(ds, "val")
    Pv, Fv, _ = embed_probs([f for f, _ in val], weights, progress="val")
    T, types = type_probs(Pv, names)
    true_t = np.array([types.index(split_leaf(c)[0]) for _, c in val])
    conf = T.max(1)
    correct = T.argmax(1) == true_t
    knn = knn_distance(Fv, bank, k)

    # kNN threshold per PREDICTED waste type (visually diverse types such as mixed trash sit further
    # from their neighbours than uniform product shots): each keeps (1 - miss/2) of its val images.
    # The msp threshold then brings the combined rule to id_retention overall.
    miss = 1 - id_retention
    knn_thr = float(np.quantile(knn, 1 - miss / 2))
    pt = T.argmax(1)
    by_type = {t: (float(np.quantile(knn[pt == i], 1 - miss / 2)) if (pt == i).sum() >= 20 else knn_thr)
               for i, t in enumerate(types)}
    thr_each = np.array([by_type[types[i]] for i in pt])
    passed = knn <= thr_each
    target_reject = miss * len(val) - (~passed).sum()
    cand = np.sort(conf[passed])
    msp_thr = float(cand[int(max(0, min(len(cand) - 1, target_reject)))]) if target_reject > 0 else 0.0
    unknown = (knn > thr_each) | (conf < msp_thr)
    min_conf = float(np.quantile(conf[correct], 0.05))

    # "Extremely unfamiliar": beyond the 99.5th percentile of its predicted type's val distances a
    # photo is called unknown even when the model is confident (see predict.ood_status).
    far_all = float(np.quantile(knn, 0.995))
    far_by_type = {t: (float(np.quantile(knn[pt == i], 0.995)) if (pt == i).sum() >= 20 else far_all)
                   for i, t in enumerate(types)}

    # Feedback memory: the nearest-neighbour distance below which a neighbour's waste type is right
    # >= 98% of the time (val image vs its nearest TRAIN embedding). predict.py only lets a confirmed
    # feedback photo influence a new photo that is at least this close.
    bank_t = np.array([types.index(split_leaf(c)[0]) for _, c in bank_items])
    fv = Fv / np.linalg.norm(Fv, axis=1, keepdims=True)
    nn_d, nn_i = np.empty(len(fv)), np.empty(len(fv), dtype=int)
    for s in range(0, len(fv), 512):
        sims = fv[s:s + 512] @ bank.T
        nn_i[s:s + 512] = sims.argmax(1)
        nn_d[s:s + 512] = 1 - sims.max(1)
    agree = bank_t[nn_i] == true_t
    order = np.argsort(nn_d)
    precision = np.cumsum(agree[order]) / np.arange(1, len(order) + 1)
    ok = np.nonzero((precision >= 0.98) & (np.arange(1, len(order) + 1) >= 50))[0]
    mem_thr = float(nn_d[order][ok[-1]]) if len(ok) else float(np.quantile(nn_d, 0.05))
    stats_mem = {"memory_threshold": mem_thr,
                 "memory_threshold_coverage": float((nn_d <= mem_thr).mean()),
                 "memory_threshold_type_precision": float(agree[nn_d <= mem_thr].mean())}

    stats = {"weights": str(weights), "dataset": str(ds), "bank_size": int(len(bank)), "k": k,
             "knn_far_by_type": far_by_type, **stats_mem,
             "val_images": len(val), "val_type_accuracy": float(correct.mean()),
             "knn_threshold": knn_thr, "knn_threshold_by_type": by_type,
             "msp_threshold": msp_thr, "min_confidence": min_conf,
             "val_flagged_unknown_by_type": {t: float(unknown[pt == i].mean()) for i, t in enumerate(types) if (pt == i).any()},
             "val_flagged_unknown": float(unknown.mean()),
             "val_correct_kept_confident": float((conf[correct] >= min_conf).mean()),
             "val_wrong_below_min_confidence": float((conf[~correct] < min_conf).mean()) if (~correct).any() else None}
    out = out or weights.with_suffix(".ood.npz")
    stats["file"] = str(out)
    np.savez_compressed(out, bank=bank.astype(np.float16), k=k,
                        knn_threshold=knn_thr, msp_threshold=msp_thr, min_confidence=min_conf,
                        knn_threshold_by_type=json.dumps(by_type), knn_far_by_type=json.dumps(far_by_type),
                        memory_threshold=mem_thr, meta=json.dumps(stats))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "processed" / "cls_dataset")
    ap.add_argument("--per-class", type=int, default=800)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--id-retention", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=None, help="default: <weights>.ood.npz")
    a = ap.parse_args(argv)
    s = calibrate(a.weights.resolve(), a.dataset.resolve(), a.per_class, a.k, a.id_retention, a.out)
    print(json.dumps(s, indent=2))
    print(f"wrote {s['file']}")


if __name__ == "__main__":
    main()
