"""
evaluate_real_photos.py - Compare models on REAL user photos (the feedback log), not dataset images.

    python -m ml.classifier.tools.evaluate_real_photos
    python -m ml.classifier.tools.evaluate_real_photos --weights ml/classifier/weights/best.pt other.pt ...

Uses the latest answer per photo set in data/feedback/classifier_feedback.jsonl whose actual waste
type is one of the model's types (so "other" answers are skipped). Predictions are the RAW model
(no feedback memory, no unknown check): photos a user confirmed are exactly what the memory would
match, so including it would make every model look perfect.

Dataset test splits say little about bales, bulk loads and phone photos; this is the number that
decides whether a candidate model is better for real use. With few photos the error bars are huge:
the report prints a 95% Wilson interval. Default --weights: best.pt + every kaggle_candidate*.pt and
feedback_candidate.pt that exists. Writes ml/classifier/reports/real_photos/<timestamp>.json.

Caution: photos a model was FINE-TUNED on (feedback_candidate.pt) are not a fair test for that model.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from ml.classifier.tools.calibrate_ood import type_probs
from ml.classifier.predict import embed_probs, load_model, split_leaf

ROOT = Path(__file__).resolve().parents[3]
W = ROOT / "ml" / "classifier" / "weights"
OUT = ROOT / "ml" / "classifier" / "reports" / "real_photos"


def feedback_log() -> Path:
    import os
    return Path(os.environ.get("MM_FEEDBACK_DIR", ROOT / "data" / "feedback")) / "classifier_feedback.jsonl"


def photos(types: set[str]) -> list[dict]:
    log = feedback_log()
    latest = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["item_key"]] = r
    out = []
    for r in latest.values():
        if r.get("actual_label") not in types:
            continue
        for f in r["image_files"]:
            # stored relative to the project root; with MM_FEEDBACK_DIR the photo sits in <dir>/images/
            p = ROOT / f if (ROOT / f).exists() else log.parent / "images" / Path(f).name
            if p.exists():
                out.append({"file": p, "type": r["actual_label"], "sub_type": r.get("actual_sub_type")})
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, nargs="*")
    a = ap.parse_args(argv)
    weights = a.weights or [W / "best.pt"] + sorted(W.glob("kaggle_candidate*.pt")) + sorted(W.glob("feedback_candidate.pt"))
    model = load_model(weights[0])
    types = {split_leaf(model.names[i])[0] for i in range(len(model.names))}
    items = photos(types)
    if not items:
        print(f"No usable real photos in {feedback_log()} yet - confirm/correct predictions in the app first.")
        return 1
    print(f"{len(items)} real photos (latest answer per photo set)\n")
    res = {"photos": len(items), "models": {}}
    for w in weights:
        P, _, names = embed_probs([it["file"] for it in items], w)
        T, tnames = type_probs(P, names)
        rows = []
        for it, t, p in zip(items, T, P):
            pt = tnames[int(t.argmax())]
            rows.append({"file": it["file"].name, "true": it["type"], "pred": pt, "conf": round(float(t.max()), 3),
                         "pred_sub": split_leaf(names[int(p.argmax())])[1], "ok": pt == it["type"]})
        k = sum(r["ok"] for r in rows)
        lo, hi = wilson(k, len(rows))
        res["models"][w.name] = {"type_right": k, "n": len(rows), "wilson95": [round(lo, 3), round(hi, 3)], "rows": rows}
        print(f"{w.name:<32} type right {k}/{len(rows)} = {100 * k / len(rows):.0f}%  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)")
        for r in rows:
            if not r["ok"]:
                print(f"    {r['file'][:12]}  true {r['true']:<12} -> {r['pred']} {r['conf']:.0%}")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
