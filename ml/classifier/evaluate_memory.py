"""
evaluate_memory.py - Does the feedback memory help, and can it hurt? (simulation on held-out data)

    python -m ml.classifier.evaluate_memory [--weights ml/classifier/weights/best.pt]

Pretends users confirmed photos with their true labels and measures what the memory in
predict.apply_memory does to OTHER photos:
  1. curve selection on the validation split only: half A = "confirmed" photos, half B = new photos;
  2. report on the test split with the whole validation split as "confirmed" photos.
Counts fixed (wrong -> right) and broken (right -> wrong) predictions for each blend curve.
Writes ml/classifier/reports/v2_eval/memory_simulation.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.classifier import predict as P
from ml.classifier.calibrate_ood import list_split

ROOT = Path(__file__).resolve().parents[2]
CURVES = ("linear", "quadratic", "step")


def run(Pq, Fq, yq, Fm, ym, names, thr, curve):
    P.MEMORY_CURVE = curve
    mem = {"emb": Fm / np.linalg.norm(Fm, axis=1, keepdims=True),
           "labels": [P.split_leaf(names[i]) for i in ym]}
    types = sorted({P.split_leaf(n)[0] for n in names})
    tix = np.array([types.index(P.split_leaf(n)[0]) for n in names])
    res = {"n": len(yq), "with_memory_match": 0}
    before_t, after_t, before_l, after_l = [], [], [], []
    for i in range(len(yq)):
        probs = {names[j]: float(Pq[i, j]) for j in range(len(names))}
        new, m = P.apply_memory(probs, Fq[i], mem, thr)
        res["with_memory_match"] += m is not None
        pb = np.array([probs[n] for n in names])
        pa = np.array([new[n] for n in names])
        before_l.append(pb.argmax()); after_l.append(pa.argmax())
        before_t.append(np.bincount(tix, pb).argmax()); after_t.append(np.bincount(tix, pa).argmax())
    bt, at = np.array(before_t) == tix[yq], np.array(after_t) == tix[yq]
    bl, al = np.array(before_l) == yq, np.array(after_l) == yq
    res.update({"type_acc_before": float(bt.mean()), "type_acc_after": float(at.mean()),
                "type_fixed": int((~bt & at).sum()), "type_broken": int((bt & ~at).sum()),
                "leaf_acc_before": float(bl.mean()), "leaf_acc_after": float(al.mean()),
                "leaf_fixed": int((~bl & al).sum()), "leaf_broken": int((bl & ~al).sum())})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "ml" / "classifier" / "weights" / "best.pt")
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "processed" / "cls_dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "ml" / "classifier" / "reports" / "v2_eval" / "memory_simulation.json")
    a = ap.parse_args(argv)
    thr = P.ood_calibration(a.weights)["memory_threshold"]
    val, test = list_split(a.dataset, "val"), list_split(a.dataset, "test")
    Pv, Fv, names = P.embed_probs([f for f, _ in val], a.weights, progress="val")
    Pt, Ft, _ = P.embed_probs([f for f, _ in test], a.weights, progress="test")
    yv = np.array([names.index(c) for _, c in val])
    yt = np.array([names.index(c) for _, c in test])
    A, B = np.arange(len(val)) % 2 == 0, np.arange(len(val)) % 2 == 1
    out = {"memory_threshold": thr, "selection_on_val": {}, "test_with_val_as_memory": {}}
    for c in CURVES:
        out["selection_on_val"][c] = run(Pv[B], Fv[B], yv[B], Fv[A], yv[A], names, thr, c)
    score = {c: r["type_fixed"] - r["type_broken"] + 0.5 * (r["leaf_fixed"] - r["leaf_broken"])
             for c, r in out["selection_on_val"].items()}
    out["chosen_curve"] = max(score, key=score.get)
    for c in CURVES:
        out["test_with_val_as_memory"][c] = run(Pt, Ft, yt, Fv, yv, names, thr, c)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for split in ("selection_on_val", "test_with_val_as_memory"):
        print(f"\n{split}:")
        for c, r in out[split].items():
            print(f"  {c:<9} matched {r['with_memory_match']:>4}/{r['n']}  type {r['type_acc_before']:.4f} -> "
                  f"{r['type_acc_after']:.4f} (fixed {r['type_fixed']}, broken {r['type_broken']})  leaf "
                  f"{r['leaf_acc_before']:.4f} -> {r['leaf_acc_after']:.4f} (fixed {r['leaf_fixed']}, broken {r['leaf_broken']})")
    print(f"\nchosen on validation: {out['chosen_curve']}  -> {a.out}")


if __name__ == "__main__":
    main()
