"""
evaluate_resolution.py - Does the model depend on photo resolution? (the "small photo = e-waste" bias)

    python -m ml.classifier.tools.evaluate_resolution --weights ml/classifier/weights/best.pt [more.pt ...]

Takes a fixed sample of the TEST split (--per-class images per sub-type, seed 0), shrinks every image
so its short side is 150 / 100 / 70 px (bilinear, like a small web or chat photo) and reports per
size: waste-type top-1, how many NON-e-waste photos are called e-waste, and e-waste recall.
Raw model probabilities (no unknown check, no small-image guard). Same sample for every model, so
the numbers are directly comparable. Writes reports/resolution/<weights-stem>.json.

Why: every ewaste_small image is 150x150, so a model can learn "small / low-detail -> e-waste".
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ml.classifier.tools.calibrate_ood import type_probs
from ml.classifier.predict import embed_probs, split_leaf

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data" / "processed" / "cls_dataset"
OUT = ROOT / "ml" / "classifier" / "reports" / "resolution"
SIZES = (None, 150, 100, 70)                  # None = original size


def sample(ds: Path, per_class: int, seed: int = 0) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    items = []
    for leaf_dir in sorted(p for p in (ds / "test").iterdir() if p.is_dir()):
        files = sorted(leaf_dir.glob("*.jpg"))
        items += [(f, leaf_dir.name) for f in rng.sample(files, min(per_class, len(files)))]
    return items


def shrink(path: Path, short: int | None) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if short is None or min(im.size) <= short:
        return im
    s = short / min(im.size)
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)


def evaluate(weights: Path, items: list[tuple[Path, str]]) -> dict:
    true_types = np.array([split_leaf(leaf)[0] for _, leaf in items])
    non_ew = true_types != "e_waste"
    res = {"weights": str(weights), "n": len(items), "n_non_e_waste": int(non_ew.sum()), "sizes": {}}
    for short in SIZES:
        P, _, names = embed_probs([shrink(f, short) for f, _ in items], weights)
        T, types = type_probs(P, names)
        pred = np.array([types[i] for i in T.argmax(1)])
        key = "original" if short is None else f"{short}px"
        res["sizes"][key] = {
            "type_top1": float((pred == true_types).mean()),
            "non_e_waste_called_e_waste": float((pred[non_ew] == "e_waste").mean()),
            "e_waste_recall": float((pred[~non_ew] == "e_waste").mean()),
        }
        print(f"  {key:>8}: type top-1 {100 * res['sizes'][key]['type_top1']:.1f}%  "
              f"non-e-waste -> e-waste {100 * res['sizes'][key]['non_e_waste_called_e_waste']:.1f}%  "
              f"e-waste recall {100 * res['sizes'][key]['e_waste_recall']:.1f}%", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--per-class", type=int, default=60)
    a = ap.parse_args(argv)
    items = sample(a.dataset, a.per_class)
    print(f"{len(items)} test images ({a.per_class} per sub-type max)")
    OUT.mkdir(parents=True, exist_ok=True)
    for w in a.weights:
        print(w)
        r = evaluate(w.resolve(), items)
        (OUT / f"{w.stem}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
