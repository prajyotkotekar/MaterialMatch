"""Build a training dataset = the v2 dataset + sorting-line object crops from data/external_validation.

User-approved on 2026-09-30 ("u can use the new dataset for training"). To keep an honest external test:
  * crops come ONLY from the external dataset's `train` split (-> our train) and `val` split (-> our val);
  * its `test` split is never read here; `evaluate_external --splits test` is the external test;
  * train/val frames that near-duplicate a test frame (consecutive video frames; list from the
    cross-split check, --drop-frames) are skipped.
The base dataset is linked, not copied or changed. Our own test split is the unchanged v2 test split.

Crops are made like evaluate_external / the CODD training crops (bbox + 10% pad, short side >= 24 px),
resized to max side 320 and saved as JPEG. Mapping: cardboard -> paper__cardboard, rigid_plastic and
soft_plastic -> plastic__plastic, metal -> metal__metal. Per-category caps keep the new domain from
swamping the rest; leaves stay within the range of the existing oversampled train counts.

    python -m ml.classifier.tools.build_domain_dataset [--drop-frames leak.json]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from ml.classifier.tools.train_yolo import _link

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "processed" / "cls_dataset"
EXT = ROOT / "data" / "external_validation"
OUT = ROOT / "data" / "processed" / "cls_dataset_zw"
LEAF = {"cardboard": "paper__cardboard", "rigid_plastic": "plastic__plastic",
        "soft_plastic": "plastic__plastic", "metal": "metal__metal"}
# crops per source category (train, val); metal has only ~380 objects in total, so all are used and
# the train ones repeated (like the base dataset's oversampling of small leaves, max 3x, train only)
CAPS = {"cardboard": (1400, 300), "soft_plastic": (1000, 200), "rigid_plastic": (700, 150), "metal": (None, None)}
METAL_REPEAT = 3
PAD, MIN_CROP, MAX_SIDE = 0.10, 24, 320


def crops_for(split: str, drop: set[str]) -> dict[str, list[tuple[str, list[float], int]]]:
    j = json.load(open(EXT / split / "labels.json", encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in j["categories"]}
    files = {im["id"]: im["file_name"] for im in j["images"]}
    out = defaultdict(list)
    for a in j["annotations"]:
        fn = files[a["image_id"]]
        if f"{split}/data/{fn}" in drop or f"{split}\\data\\{fn}" in drop:
            continue
        x, y, w, h = a["bbox"]
        if min(w, h) * (1 + 2 * PAD) < MIN_CROP:
            continue
        out[cats[a["category_id"]]].append((fn, a["bbox"], a["id"]))
    return out


def save_crop(split: str, fn: str, bbox, dst: Path, cache: dict) -> None:
    if fn not in cache:
        cache.clear()
        cache[fn] = Image.open(EXT / split / "data" / fn).convert("RGB")
    im = cache[fn]
    x, y, w, h = bbox
    px, py = w * PAD, h * PAD
    box = (max(0, int(x - px)), max(0, int(y - py)), min(im.width, int(x + w + px)), min(im.height, int(y + h + py)))
    c = im.crop(box)
    c.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
    c.save(dst, "JPEG", quality=92)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--drop-frames", type=Path, default=None, help="json with 'drop_frames' (cross-split check)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    drop = set(json.load(open(a.drop_frames))["drop_frames"]) if a.drop_frames else set()
    if a.out.exists():
        shutil.rmtree(a.out)
    # 1. link the unchanged base dataset
    n = 0
    for p in a.base.rglob("*"):
        if p.is_file() and p.suffix != ".cache":
            dst = a.out / p.relative_to(a.base)
            dst.parent.mkdir(parents=True, exist_ok=True)
            _link(p, dst)
            n += 1
    print(f"linked {n} base files")
    # 2. add crops (never from the external test split)
    rng = random.Random(a.seed)
    added = Counter()
    for split, cap_i in (("train", 0), ("val", 1)):
        by_cat = crops_for(split, drop)
        cache: dict = {}
        for cat, items in sorted(by_cat.items()):
            cap = CAPS[cat][cap_i]
            items = sorted(items, key=lambda t: t[2])
            chosen = items if cap is None or len(items) <= cap else rng.sample(items, cap)
            chosen.sort(key=lambda t: t[0])            # frame order -> each frame decoded once
            leaf_dir = a.out / split / LEAF[cat]
            reps = METAL_REPEAT if (cat == "metal" and split == "train") else 1
            for fn, bbox, aid in chosen:
                dst = leaf_dir / f"zerowaste_{split}_{cat}_{aid}.jpg"
                save_crop(split, fn, bbox, dst, cache)
                for r in range(1, reps):
                    _link(dst, dst.with_name(f"{dst.stem}_os{r}.jpg"))
                added[(split, cat)] += reps
            print(f"  {split:<5} {cat:<14} {len(chosen):>5} crops (x{reps}) of {len(items)} -> {LEAF[cat]}")
    counts = {sp: {d.name: sum(1 for _ in d.iterdir()) for d in (a.out / sp).iterdir() if d.is_dir()}
              for sp in ("train", "val", "test")}
    info = {"base": str(a.base), "source": str(EXT), "splits_used": ["train", "val"], "test_split_used": False,
            "frames_dropped_near_test": len(drop), "caps": CAPS, "metal_repeat": METAL_REPEAT,
            "added": {f"{s}/{c}": v for (s, c), v in sorted(added.items())},
            "leaf_counts": counts}
    (a.out / "DOMAIN_ADDITIONS.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    for sp in ("train", "val"):
        c = counts[sp]
        print(sp, {k: c[k] for k in ("paper__cardboard", "plastic__plastic", "metal__metal")}, "total", sum(c.values()))


if __name__ == "__main__":
    main()
