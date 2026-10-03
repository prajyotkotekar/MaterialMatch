"""Robustness benchmark: how do models cope with corrupted, cropped and occluded photos?

A fixed sample of the TEST split of the v2 dataset (--per-class images per sub-type, seed 0; the same
images and the same corruption randomness for every model) is scored under 12 conditions:
  clean
  seen      gaussian_blur, gaussian_noise, jpeg, low_resolution, exposure  (= robust_augment training set)
  held-out  motion_blur, haze, salt_pepper, colour_cast                    (never used in training)
  partial   partial_crop (keeps ~50% of the area), occlusion (40% covered by a patch of ANOTHER test image)
all at severity 0.7. Scoring = the app pipeline (predict_batch: status, Detected threshold of that model's
own calibration) with the feedback memory switched OFF (empty MM_FEEDBACK_DIR), so models are compared
on their own. Metrics per condition: type accuracy (best guess), acceptance (Detected badge), correct /
false acceptance, precision of accepted, mean confidence when right / wrong, ECE (15 bins).

--fit-check adds clean accuracy + loss on equal-size samples of the train, val and test splits
(over/underfitting check: a large train >> val gap = overfitting; low train accuracy = underfitting).

    python -m ml.classifier.tools.evaluate_robustness --weights ml\\classifier\\weights\\best.pt other.pt [--fit-check]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("MM_FEEDBACK_DIR", tempfile.mkdtemp(prefix="mm_nomemory_"))   # no memory blending

from ml.classifier.tools import robust_augment as ra  # noqa: E402
from ml.classifier.predict import SEP, predict_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data" / "processed" / "cls_dataset"
OUT = ROOT / "ml" / "classifier" / "reports" / "robustness"
SEVERITY = 0.7
CONDITIONS = (["clean"] + [f"seen:{k}" for k in ra.TRAIN_CORRUPTIONS] + [f"heldout:{k}" for k in ra.HELDOUT_CORRUPTIONS]
              + ["partial:partial_crop", "partial:occlusion"])


def sample(split: str, per_class: int, seed: int = 0) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    items = []
    for leaf_dir in sorted(p for p in (DATASET / split).iterdir() if p.is_dir()):
        files = sorted(f for f in leaf_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")
                       and "_os" not in f.stem)          # no oversampled copies
        items += [(f, leaf_dir.name) for f in rng.sample(files, min(per_class, len(files)))]
    return items


def corrupt(cond: str, im: Image.Image, idx: int, others: list[Path]) -> Image.Image:
    if cond == "clean":
        return im
    group, name = cond.split(":")
    rng = random.Random(f"{cond}|{idx}")                  # same randomness for every model
    if name == "occlusion":
        other = Image.open(others[rng.randrange(len(others))]).convert("RGB")
        return ra.occlusion(im, SEVERITY, rng, patch=other)
    fn = {**ra.TRAIN_CORRUPTIONS, **ra.HELDOUT_CORRUPTIONS, **ra.GEOMETRIC}[name]
    return fn(im, SEVERITY, rng).convert("RGB")


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def score(weights: Path, images: list[Image.Image], leaves: list[str], batch: int = 64) -> dict:
    res = []
    for i in range(0, len(images), batch):
        res += predict_batch(images[i:i + batch], weights, top_k=3)
    truth = np.array([l.split(SEP)[0] for l in leaves])
    guess = np.array([r["best_guess"]["waste_type"] for r in res])
    conf = np.array([r["type_confidence"] for r in res], dtype=float)
    acc_ = np.array([r["is_confident"] for r in res])
    unk = np.array([r["status"] == "unknown" for r in res])
    ok = guess == truth
    # loss on the true type (-log p) from the leaf probabilities
    p_true = np.array([sum(p for l, p in r["leaf_probs"].items() if l.split(SEP)[0] == t) for r, t in zip(res, truth)])
    n = len(res)
    return {"n": n, "accuracy": float(ok.mean()), "acceptance": float(acc_.mean()),
            "correct_acceptance": float((acc_ & ok).mean()), "false_acceptance": float((acc_ & ~ok).mean()),
            "precision_accepted": float(ok[acc_].mean()) if acc_.any() else None,
            "unknown": float(unk.mean()),
            "mean_conf": float(conf.mean()),
            "mean_conf_correct": float(conf[ok].mean()) if ok.any() else None,
            "mean_conf_wrong": float(conf[~ok].mean()) if (~ok).any() else None,
            "ece": ece(conf, ok.astype(float)),
            "nll": float(-np.log(np.clip(p_true, 1e-6, 1)).mean())}


def evaluate(weights: Path, items: list[tuple[Path, str]], fit: dict | None) -> dict:
    base = [Image.open(f).convert("RGB") for f, _ in items]
    leaves = [l for _, l in items]
    others = [f for f, _ in items]
    out = {"weights": str(weights), "severity": SEVERITY, "conditions": {}}
    for cond in CONDITIONS:
        t = time.time()
        imgs = [corrupt(cond, im, i, others) for i, im in enumerate(base)]
        out["conditions"][cond] = m = score(weights, imgs, leaves)
        print(f"  {cond:<24} acc {m['accuracy']:.3f}  accept {m['acceptance']:.3f}  false-acc "
              f"{m['false_acceptance']:.3f}  ece {m['ece']:.3f}  ({time.time() - t:.0f}s)", flush=True)
    for g in ("seen", "heldout", "partial"):
        ms = [v for k, v in out["conditions"].items() if k.startswith(g + ":")]
        out[f"mean_{g}"] = {k: float(np.mean([m[k] for m in ms])) for k in
                            ("accuracy", "acceptance", "correct_acceptance", "false_acceptance", "ece", "mean_conf_correct")}
    if fit:
        out["fit_check"] = {}
        for split, its in fit.items():
            imgs = [Image.open(f).convert("RGB") for f, _ in its]
            m = score(weights, imgs, [l for _, l in its])
            out["fit_check"][split] = {k: m[k] for k in ("n", "accuracy", "nll", "mean_conf", "ece")}
            print(f"  fit {split:<6} acc {m['accuracy']:.3f}  nll {m['nll']:.3f}", flush=True)
    return out


def table(results: list[dict]) -> str:
    names = [Path(r["weights"]).stem for r in results]
    L = ["| Condition | " + " | ".join(f"{n} acc / accept / false-acc" for n in names) + " |",
         "|---|" + "---:|" * len(names)]
    rows = CONDITIONS + ["mean_seen", "mean_heldout", "mean_partial"]
    for c in rows:
        cells = []
        for r in results:
            m = r["conditions"][c] if c in r["conditions"] else r[c]
            cells.append(f"{m['accuracy']:.1%} / {m['acceptance']:.1%} / {m['false_acceptance']:.1%}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "| Calibration (ECE, lower = confidence matches accuracy) | " + " | ".join(names) + " |",
          "|---|" + "---:|" * len(names)]
    for c in ("clean", "mean_seen", "mean_heldout", "mean_partial"):
        L.append(f"| {c} | " + " | ".join(f"{(r['conditions'][c] if c in r['conditions'] else r[c])['ece']:.3f}"
                                           for r in results) + " |")
    if all("fit_check" in r for r in results):
        L += ["", "| Fit check (clean, equal samples) | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for s in ("train", "val", "test"):
            L.append(f"| {s} accuracy / loss | " + " | ".join(
                f"{r['fit_check'][s]['accuracy']:.1%} / {r['fit_check'][s]['nll']:.3f}" for r in results) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--fit-check", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    items = sample("test", a.per_class)
    fit = {s: sample(s, a.per_class, seed=1) for s in ("train", "val", "test")} if a.fit_check else None
    print(f"{len(items)} test images x {len(CONDITIONS)} conditions (severity {SEVERITY})")
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    for w in a.weights:
        print(f"\n== {w}")
        r = evaluate(w, items, fit)
        (a.out / f"{w.stem}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        results.append(r)
    md = a.out / ("compare_" + "_vs_".join(Path(r["weights"]).stem for r in results) + ".md")
    md.write_text(f"# Robustness benchmark\n\n{len(items)} test images per condition, severity {SEVERITY}; "
                  "acc = type accuracy of the best guess, accept = Detected badge, false-acc = wrong but "
                  "Detected. Feedback memory off.\n\n" + table(results), encoding="utf-8")
    print("\n" + table(results) + f"\n-> {md}")


if __name__ == "__main__":
    main()
