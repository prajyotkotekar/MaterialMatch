"""
retrain_with_feedback.py - Fold user feedback into the model (fine-tune + test gate + optional promote).

    python -m ml.classifier.retrain_with_feedback                  # 1. review what would be used
    python -m ml.classifier.retrain_with_feedback --confirm        # 2. fine-tune + compare (no promote)
    python -m ml.classifier.retrain_with_feedback --confirm --promote   # 3. ... and replace best.pt if it passes

Uses the latest answer per photo set in data/feedback/classifier_feedback.jsonl whose waste type is
one of the model's types and whose sub-type is known (given by the user, implied because the type
has a single sub-type, or confirmed with "Yes"). "Other" answers are skipped (no class to learn).

Fine-tuning set (data/processed/cls_feedback_finetune/): the feedback photos repeated --repeat times
+ a replay sample of --replay-per-class original TRAIN images per sub-type (so the model does not
forget the rest), validated on --val-per-class original VAL images per sub-type. Starts from the
current weights with a low learning rate.

Gate: the candidate is evaluated on the FULL standard test split next to the current model; it may
only replace best.pt if waste-type top-1 drops by at most --max-type-drop points and sub-type top-1
by at most --max-leaf-drop points. Promotion backs up the current model and recalibrates the
unknown check. With >= 10 feedback photo sets, 20% are held out as a real-photo test.
Report: ml/classifier/reports/feedback_retrain_<timestamp>/report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ml.classifier.calibrate_ood import calibrate, list_split, type_probs
from ml.classifier.predict import SEP, embed_probs, load_model, split_leaf

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "ml" / "classifier" / "weights"
FEEDBACK_LOG = ROOT / "data" / "feedback" / "classifier_feedback.jsonl"
DATASET = ROOT / "data" / "processed" / "cls_dataset"
FT_DIR = ROOT / "data" / "processed" / "cls_feedback_finetune"


def feedback_items(names: list[str], since: str | None = None) -> tuple[list[dict], Counter]:
    """(usable items, skipped reasons). One item per photo set (latest answer wins)."""
    latest = {}
    if FEEDBACK_LOG.exists():
        for line in FEEDBACK_LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["item_key"]] = r
    leaves = set(names)
    subs = {}
    for n in names:
        wt, st = split_leaf(n)
        subs.setdefault(wt, []).append(st)
    items, skipped = [], Counter()
    for r in latest.values():
        if since and r["created_at"] < since:
            skipped["before --since"] += 1
            continue
        wt, st = r.get("actual_label"), r.get("actual_sub_type")
        if wt not in subs:
            skipped[f"'{wt}' is not a model class (e.g. 'other')"] += 1
            continue
        if not st and len(subs[wt]) == 1:
            st = subs[wt][0]
        if not st and r.get("is_correct") and r.get("predicted_label") == wt:
            st = r.get("predicted_sub_type")
        leaf = f"{wt}{SEP}{st}"
        if leaf not in leaves:
            skipped[f"no sub-type given for multi-sub-type '{wt}'"] += 1
            continue
        files = [ROOT / f for f in r["image_files"] if (ROOT / f).exists()]
        if not files:
            skipped["photo file missing"] += 1
            continue
        items.append({"key": r["item_key"], "leaf": leaf, "files": files, "created_at": r["created_at"],
                      "was": f"{r.get('predicted_label')}/{r.get('predicted_sub_type')}"})
    return items, skipped


def review_sheet(items: list[dict], path: Path) -> None:
    thumbs = [(f, it["leaf"], it["was"]) for it in items for f in it["files"]]
    cols, T = 6, 150
    sheet = Image.new("RGB", (cols * T, ((len(thumbs) + cols - 1) // cols) * (T + 30) or 1), "white")
    d = ImageDraw.Draw(sheet)
    for i, (f, leaf, was) in enumerate(thumbs):
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        im.thumbnail((T, T))
        x, y = (i % cols) * T, (i // cols) * (T + 30)
        sheet.paste(im, (x + (T - im.width) // 2, y))
        d.text((x + 2, y + T + 2), f"{leaf}\nwas {was}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def build_finetune_set(items, names, replay, val_per, repeat, seed) -> Path:
    if FT_DIR.exists():
        shutil.rmtree(FT_DIR)
    rng = random.Random(seed)
    for split, per in (("train", replay), ("val", val_per)):
        for leaf in names:
            files = sorted(f for f in (DATASET / split / leaf).glob("*.jpg") if "_rep" not in f.stem)
            dst = FT_DIR / split / leaf
            dst.mkdir(parents=True, exist_ok=True)
            for f in rng.sample(files, min(per, len(files))):
                shutil.copyfile(f, dst / f.name)
    for it in items:
        for f in it["files"]:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            im.thumbnail((320, 320))
            for k in range(repeat):
                im.save(FT_DIR / "train" / it["leaf"] / f"feedback_{f.stem[:16]}_{k}.jpg", "JPEG", quality=92)
    return FT_DIR


def accuracy(weights: Path, samples: list[tuple[Path, str]], names: list[str]) -> dict:
    if not samples:
        return {"n": 0}
    P, _, n2 = embed_probs([f for f, _ in samples], weights)
    assert n2 == names, "class order changed"
    y = np.array([names.index(c) for _, c in samples])
    T, types = type_probs(P, names)
    yt = np.array([types.index(split_leaf(c)[0]) for _, c in samples])
    return {"n": len(samples), "type_top1": float((T.argmax(1) == yt).mean()),
            "leaf_top1": float((P.argmax(1) == y).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=W / "best.pt")
    ap.add_argument("--since", default=None, help="only feedback created at/after this ISO time")
    ap.add_argument("--confirm", action="store_true", help="after review: build, fine-tune and compare")
    ap.add_argument("--promote", action="store_true", help="replace best.pt if the gate passes")
    ap.add_argument("--repeat", type=int, default=20, help="copies of each feedback photo in the fine-tune set")
    ap.add_argument("--replay-per-class", type=int, default=150)
    ap.add_argument("--val-per-class", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr0", type=float, default=0.0002)
    ap.add_argument("--max-type-drop", type=float, default=0.5, help="allowed test drop, percentage points")
    ap.add_argument("--max-leaf-drop", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    base = a.weights.resolve()
    model = load_model(base)
    names = [model.names[i] for i in range(len(model.names))]
    items, skipped = feedback_items(names, a.since)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "ml" / "classifier" / "reports" / f"feedback_retrain_{stamp}"
    review_sheet(items, out / "review.png") if items else None
    print(f"Usable feedback photo sets: {len(items)}  ({sum(len(i['files']) for i in items)} photos)")
    for it in items:
        print(f"  {it['created_at']}  {it['leaf']:<28} (model said {it['was']})  {len(it['files'])} photo(s)")
    for why, n in skipped.items():
        print(f"  skipped {n}: {why}")
    if not items:
        print("Nothing to learn from yet.")
        return 1
    print(f"Review sheet: {out / 'review.png'}")
    if not a.confirm:
        print("Check the labels above, then rerun with --confirm to fine-tune.")
        return 0

    holdout = []
    if len(items) >= 10:
        hold_keys = {it["key"] for it in items if int(hashlib.md5(it["key"].encode()).hexdigest(), 16) % 5 == 0}
        holdout = [it for it in items if it["key"] in hold_keys]
        items = [it for it in items if it["key"] not in hold_keys]
    print(f"Fine-tuning on {len(items)} photo sets (holdout: {len(holdout)}) ...")
    ds = build_finetune_set(items, names, a.replay_per_class, a.val_per_class, a.repeat, a.seed)

    from ultralytics import YOLO
    y = YOLO(str(base))
    y.train(data=str(ds), epochs=a.epochs, imgsz=224, batch=64, lr0=a.lr0, optimizer="AdamW",
            warmup_epochs=0, patience=a.epochs, device=a.device, workers=4, seed=a.seed, plots=False,
            project=str(ROOT / "ml" / "classifier" / "runs"), name=f"feedback_ft_{stamp}", exist_ok=True,
            verbose=False)
    tr = y.trainer
    cand_src = Path(tr.best) if tr.best and Path(tr.best).exists() else Path(tr.last)
    cand = W / "feedback_candidate.pt"
    shutil.copy2(cand_src, cand)

    test = list_split(DATASET, "test")
    fb = [(f, it["leaf"]) for it in items for f in it["files"]]
    ho = [(f, it["leaf"]) for it in holdout for f in it["files"]]
    res = {name: {"test": accuracy(w, test, names), "feedback_used": accuracy(w, fb, names),
                  "feedback_holdout": accuracy(w, ho, names)}
           for name, w in (("current", base), ("candidate", cand))}
    d_type = 100 * (res["candidate"]["test"]["type_top1"] - res["current"]["test"]["type_top1"])
    d_leaf = 100 * (res["candidate"]["test"]["leaf_top1"] - res["current"]["test"]["leaf_top1"])
    passed = d_type >= -a.max_type_drop and d_leaf >= -a.max_leaf_drop
    promoted = None
    if a.promote and passed:
        backup = W / f"best_before_feedback_{stamp}.pt"
        shutil.copy2(base, backup)
        shutil.copy2(cand, base)
        load_model.__globals__["_MODEL_CACHE"].pop(str(base), None)
        load_model.__globals__["_OOD_CACHE"].pop(str(base), None)
        calibrate(base, DATASET)
        promoted = str(backup)

    L = [f"# Feedback fine-tune {stamp}", "",
         f"Base: `{base}` · feedback photo sets used: {len(items)} (holdout {len(holdout)}) · "
         f"repeat x{a.repeat} · replay {a.replay_per_class}/sub-type · {a.epochs} epochs · lr0 {a.lr0}", "",
         "| | current | candidate |", "|---|---:|---:|"]
    for split in ("test", "feedback_used", "feedback_holdout"):
        c, n = res["current"][split], res["candidate"][split]
        if c["n"]:
            L.append(f"| {split} type top-1 (n={c['n']}) | {100 * c['type_top1']:.1f}% | {100 * n['type_top1']:.1f}% |")
            L.append(f"| {split} sub-type top-1 | {100 * c['leaf_top1']:.1f}% | {100 * n['leaf_top1']:.1f}% |")
    L += ["", f"Gate (type drop ≤ {a.max_type_drop} pt, sub-type drop ≤ {a.max_leaf_drop} pt): "
          f"type {d_type:+.2f} pt, sub-type {d_leaf:+.2f} pt → **{'PASS' if passed else 'FAIL'}**",
          "", f"Promoted: {'yes, previous model backed up to `' + promoted + '`, unknown check recalibrated' if promoted else 'no'}",
          "", "Note: 'feedback_used' photos were trained on, so that row only shows the model learned them; "
              "only 'feedback_holdout' (needs >= 10 photo sets) says anything about NEW real photos.",
          "", f"Skipped feedback: {dict(skipped) or 'none'}", "", "![review](review.png)"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"results": res, "passed": passed, "promoted": promoted,
                                                  "d_type_pt": d_type, "d_leaf_pt": d_leaf}, indent=2), encoding="utf-8")
    print("\n".join(L))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
