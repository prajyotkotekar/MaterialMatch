"""
train_yolo.py - Build the MaterialMatch image dataset and fine-tune a YOLOv8 classifier.

The model predicts a LEAF class named "<waste_type>__<sub_type>" (e.g. "e_waste__keyboard",
"construction__concrete"). The waste type is the prefix, so predict.py can marginalise the
leaf probabilities to a waste-type probability without a second model.

Which source label becomes which leaf is read from data/processed/class_mapping.csv
(columns: dataset, source_label, canonical_waste_type, canonical_sub_type, status, notes).
Rows with status "unmapped" are excluded. Edit the CSV, not this file, to change the mapping.

Sources (data/Image/):
  codd/          Pascal-VOC XML + jpg scenes. Every annotated object is CROPPED out. Official
                 training/validation/testing split honoured.
  ewaste_small/  train/val/test/<device>/ whole images. Official split honoured.
  garbage_v2/    <class>/ whole images, no split -> deterministic stratified 70/15/15 split.

Leakage control (whole-image sources): exact (md5) and near-duplicate images (rule in
audit_datasets.duplicate_pairs) are grouped; a group lives in exactly one split. When a group
straddles an official split, copies outside the chosen split are dropped (train wins, then
val). Groups whose members map to different leaves are dropped entirely. Exact copies inside
one split are reduced to one file.

Imbalance: leaf train counts are reported; by default (--balance oversample) leaves below the
median train count are topped up by repeating their own train images (at most
--max-oversample x), relying on the trainer's random augmentation. Val/test are never touched.
--max-per-class is still available but off by default.

Examples
--------
python ml/classifier/train_yolo.py --auto --build-only
python ml/classifier/train_yolo.py --dataset data/processed/cls_dataset --epochs 25 --patience 7 \
    --device cpu --workers 4 --batch 64
(the new checkpoint is written to weights/best_candidate.pt; weights/best.pt is not touched)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

try:
    from ml.classifier.audit_datasets import duplicate_pairs, image_features
except ImportError:  # run as a script from ml/classifier
    from audit_datasets import duplicate_pairs, image_features

for _heif_mod in ("pi_heif", "pillow_heif"):
    try:
        __import__(_heif_mod).register_heif_opener()
        break
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_WEIGHTS_DIR = HERE / "weights"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "cls_dataset"
DEFAULT_MAPPING = PROJECT_ROOT / "data" / "processed" / "class_mapping.csv"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "Image"
DEFAULT_CODD_DIR = DEFAULT_IMAGE_ROOT / "codd"
DEFAULT_EWASTE_DIR = DEFAULT_IMAGE_ROOT / "ewaste_small"
DEFAULT_GARBAGE_DIR = DEFAULT_IMAGE_ROOT / "garbage_v2"

SEP = "__"
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}


def leaf_name(waste_type: str, sub_type: str) -> str:
    return f"{waste_type}{SEP}{sub_type}"


@dataclass
class Record:
    leaf: str
    split: str | None            # official split, or None
    src: Path
    box: tuple | None            # (x0, y0, x1, y1) for CODD crops
    name: str                    # output file stem
    dataset: str
    source_label: str
    group: str = ""              # dedup / split unit
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #
def load_mapping(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        raise SystemExit(f"Mapping file not found: {path}")
    rows = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["dataset"].strip(), r["source_label"].strip().lower())
            rows[key] = r
    return rows


def map_label(mapping, dataset: str, label: str, unmapped: Counter) -> str | None:
    r = mapping.get((dataset, label.strip().lower()))
    if r is None:
        unmapped[f"{dataset}/{label} (NOT IN MAPPING)"] += 1
        return None
    if r["status"].strip() == "unmapped" or not r["canonical_waste_type"].strip():
        unmapped[f"{dataset}/{label}"] += 1
        return None
    return leaf_name(r["canonical_waste_type"].strip(), r["canonical_sub_type"].strip())


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def detect_split(path: Path, root: Path) -> str | None:
    for part in path.relative_to(root).parts[:-1]:
        s = SPLIT_ALIASES.get(part.strip().lower())
        if s:
            return s
    return None


def iter_codd(codd_dir: Path, mapping, unmapped: Counter):
    xmls = sorted(codd_dir.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"No .xml annotations under {codd_dir}")
    for xml_path in xmls:
        img = next((xml_path.with_suffix(e) for e in (".jpg", ".jpeg", ".png", ".JPG")
                    if xml_path.with_suffix(e).exists()), None)
        if img is None:
            unmapped["codd/<xml without image>"] += 1
            continue
        split = detect_split(img, codd_dir)
        for i, obj in enumerate(ET.parse(xml_path).iter("object")):
            raw = (obj.findtext("name") or "").strip()
            leaf = map_label(mapping, "codd", raw, unmapped)
            if leaf is None:
                continue
            bb = obj.find("bndbox")
            try:
                box = tuple(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            except (TypeError, ValueError, AttributeError):
                unmapped["codd/<bad box>"] += 1
                continue
            if box[2] <= box[0] or box[3] <= box[1]:
                unmapped["codd/<degenerate box>"] += 1
                continue
            yield Record(leaf, split, img, box, f"codd_{split}_{img.stem}_{i}", "codd", raw,
                         group=f"codd:{img.relative_to(codd_dir).as_posix()}")


def iter_folder(root: Path, dataset: str, mapping, unmapped: Counter):
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        parts = img.relative_to(root).parts[:-1]
        split = next((SPLIT_ALIASES[p.lower()] for p in parts if p.lower() in SPLIT_ALIASES), None)
        label = next((p for p in reversed(parts) if p.lower() not in SPLIT_ALIASES), None)
        if label is None:
            continue
        leaf = map_label(mapping, dataset, label, unmapped)
        if leaf is None:
            continue
        tag = hashlib.md5(img.relative_to(root).as_posix().encode()).hexdigest()[:8]
        yield Record(leaf, split, img, None, f"{dataset}_{img.stem}_{tag}".replace(" ", "_"),
                     dataset, label)


def iter_extra(leaf: str, folder: Path):
    for img in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        tag = hashlib.md5(str(img).encode()).hexdigest()[:8]
        yield Record(leaf, detect_split(img, folder), img, None, f"extra_{img.stem}_{tag}",
                     "extra", str(folder))


# --------------------------------------------------------------------------- #
# Dedup + split
# --------------------------------------------------------------------------- #
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _features(path: Path):
    data = path.read_bytes()
    try:
        with Image.open(path) as im:
            f = image_features(ImageOps.exif_transpose(im))
    except Exception:
        return hashlib.md5(data).hexdigest(), None
    return hashlib.md5(data).hexdigest(), f


def dedup_and_split(records: list[Record], val_fraction: float, test_fraction: float,
                    seed: int, stats: dict) -> list[Record]:
    whole = [r for r in records if r.box is None]
    crops = [r for r in records if r.box is not None]
    print(f"  hashing {len(whole)} whole images for duplicate detection...", flush=True)
    with ThreadPoolExecutor(12) as ex:
        feats = list(ex.map(_features, [r.src for r in whole], chunksize=64))
    for r, (md5, _) in zip(whole, feats):
        r.extra["md5"] = md5
    uf = UnionFind(len(whole))
    by_md5 = defaultdict(list)
    for i, r in enumerate(whole):
        by_md5[r.extra["md5"]].append(i)
    for idx in by_md5.values():
        for j in idx[1:]:
            uf.union(idx[0], j)
    near = duplicate_pairs([f for _, f in feats])
    for a, b in near:
        uf.union(a, b)
    groups = defaultdict(list)
    for i in range(len(whole)):
        groups[uf.find(i)].append(i)

    kept: list[Record] = []
    dropped = Counter()
    unsplit_units = defaultdict(list)  # leaf -> [(group_key, [records])]
    for gid, idx in groups.items():
        members = [whole[i] for i in idx]
        leaves = {m.leaf for m in members}
        if len(leaves) > 1:
            dropped["duplicate group with conflicting labels"] += len(members)
            continue
        # one file per md5 inside the group
        seen, uniq = set(), []
        for m in sorted(members, key=lambda m: (SPLIT_PRIORITY.get(m.split, 9), str(m.src))):
            if m.extra["md5"] in seen:
                dropped["exact duplicate copy"] += 1
                continue
            seen.add(m.extra["md5"])
            uniq.append(m)
        official = sorted({m.split for m in uniq if m.split}, key=SPLIT_PRIORITY.get)
        key = f"dup:{min(str(m.src) for m in uniq)}"
        # Keep ONE image per duplicate group: prefer a member of the chosen official split.
        rep = uniq[0]
        for m in uniq[1:]:
            if official and m.split and m.split != official[0]:
                dropped[f"duplicate across official split ({m.split} copy of a {official[0]} image)"] += 1
            else:
                dropped["near-duplicate copy (kept one per group)"] += 1
        rep.group = key
        if official:
            rep.split = official[0]
            kept.append(rep)
        else:
            unsplit_units[rep.leaf].append((key, [rep]))

    # Deterministic STRATIFIED split per leaf for datasets without an official split
    for leaf, units in unsplit_units.items():
        units.sort(key=lambda u: hashlib.md5(f"{seed}:{u[0]}".encode()).hexdigest())
        n = len(units)
        n_val, n_test = round(n * val_fraction), round(n * test_fraction)
        for k, (_, members) in enumerate(units):
            split = "val" if k < n_val else "test" if k < n_val + n_test else "train"
            for m in members:
                m.split = split
                kept.append(m)

    for r in crops:  # CODD always ships an official split; hash the scene if one ever doesn't
        if r.split is None:
            x = int(hashlib.md5(f"{seed}:{r.group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            r.split = "val" if x < val_fraction else "test" if x < val_fraction + test_fraction else "train"
    stats["duplicates_dropped"] = dict(dropped)
    stats["near_duplicate_pairs"] = len(near)
    stats["duplicate_groups_multi"] = sum(len(v) > 1 for v in groups.values())
    return kept + crops


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def load_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        print(f"  [skip] cannot read {path}: {exc}")
        return None


def crop_box(img, box, pad: float, min_side: int):
    x0, y0, x1, y1 = box
    w, h = img.size
    bw, bh = x1 - x0, y1 - y0
    x0, y0 = max(0, int(x0 - bw * pad)), max(0, int(y0 - bh * pad))
    x1, y1 = min(w, int(round(x1 + bw * pad))), min(h, int(round(y1 + bh * pad)))
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None
    return img.crop((x0, y0, x1, y1))


def save_resized(img, out_path: Path, max_side: int) -> None:
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=92)


def write_images(records: list[Record], out_dir: Path, pad: float, min_crop: int,
                 max_side: int) -> tuple[list[Record], Counter]:
    by_src = defaultdict(list)
    for r in records:
        by_src[r.src].append(r)
    problems = Counter()
    written: list[Record] = []

    def job(item):
        src, recs = item
        img = load_image(src)
        if img is None:
            return [], Counter({"unreadable source": len(recs)})
        ok, bad = [], Counter()
        for r in recs:
            out = img if r.box is None else crop_box(img, r.box, pad, min_crop)
            if out is None:
                bad[f"crop smaller than {min_crop}px"] += 1
                continue
            path = out_dir / r.split / r.leaf / f"{r.name}.jpg"
            save_resized(out, path, max_side)
            r.extra["out"] = path
            ok.append(r)
        return ok, bad

    items = sorted(by_src.items(), key=lambda kv: str(kv[0]))
    with ThreadPoolExecutor(8) as ex:
        for n, (ok, bad) in enumerate(ex.map(job, items), 1):
            written += ok
            problems += bad
            if n % 2000 == 0 or n == len(items):
                print(f"  writing... {n}/{len(items)} source images", flush=True)
    return written, problems


def drop_cross_split_crop_duplicates(written: list[Record]) -> tuple[list[Record], Counter]:
    """CODD's official split puts consecutive shots of the same object in different splits.
    For crops that near-duplicate a crop in another split, keep the train copy (then val) and
    delete the others, so val/test never contain a near-copy of a training image."""
    crops = [r for r in written if r.box is not None]
    print(f"  checking {len(crops)} crops for near-duplicates across splits...", flush=True)
    with ThreadPoolExecutor(12) as ex:
        feats = list(ex.map(lambda r: _features(r.extra["out"])[1], crops, chunksize=64))
    uf = UnionFind(len(crops))
    for a, b in duplicate_pairs(feats):
        uf.union(a, b)
    groups = defaultdict(list)
    for i in range(len(crops)):
        groups[uf.find(i)].append(crops[i])
    removed, dropped = set(), Counter()
    for members in groups.values():
        splits = {m.split for m in members}
        if len(splits) < 2:
            continue
        keep = min(splits, key=SPLIT_PRIORITY.get)
        for m in members:
            if m.split != keep:
                m.extra["out"].unlink(missing_ok=True)
                removed.add(id(m))
                dropped[f"CODD crop near-duplicate across official split ({m.split} copy of a {keep} crop)"] += 1
    return [r for r in written if id(r) not in removed], dropped


def oversample(written: list[Record], out_dir: Path, max_factor: float, seed: int) -> dict:
    train = defaultdict(list)
    for r in written:
        if r.split == "train":
            train[r.leaf].append(r)
    target = int(statistics.median(len(v) for v in train.values()))
    rng = random.Random(seed)
    added = {}
    for leaf, recs in sorted(train.items()):
        n = len(recs)
        want = min(target, int(math.floor(n * max_factor)))
        extra = max(0, want - n)
        pool = []
        while len(pool) < extra:
            batch = recs[:]
            rng.shuffle(batch)
            pool += batch
        for k, r in enumerate(pool[:extra]):
            src = r.extra["out"]
            shutil.copyfile(src, src.with_name(f"{src.stem}_rep{k}.jpg"))
        added[leaf] = extra
    return {"target_train_count": target, "added": added}


def _link(src: Path, dst: Path) -> None:
    """Cheapest way to reuse a file: symlink (posix), hardlink (same volume), else copy."""
    try:
        if os.name != "nt":
            os.symlink(src, dst)
        else:
            os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def lowres_view(dataset_dir: Path, out_dir: Path, fraction: float, min_side: int, max_side: int,
                seed: int) -> tuple[Path, dict]:
    """Derived dataset = dataset_dir (linked, untouched) + low-resolution copies of a random
    `fraction` of every leaf's original TRAIN images whose short side is > max_side.

    Why: every ewaste_small image is 150x150, so "small / low-detail photo" was a shortcut for
    e-waste (measured: non-e-waste test photos shrunk to 100 px -> 15% called e-waste). Low-res
    copies of all other classes remove that shortcut. Val/test are linked unchanged, so test
    numbers stay comparable with models trained without this option."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rng = random.Random(seed)
    added: Counter = Counter()
    for f in sorted(dataset_dir.iterdir()):
        if f.is_file():                                  # hierarchy.json, summaries
            shutil.copyfile(f, out_dir / f.name)
    for split_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for leaf_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            dst = out_dir / split_dir.name / leaf_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in sorted(leaf_dir.glob("*.jpg")):
                _link(img, dst / img.name)
                if split_dir.name != "train" or "_rep" in img.stem or rng.random() >= fraction:
                    continue
                with Image.open(img) as im:
                    im = im.convert("RGB")
                    if min(im.size) <= max_side:
                        continue
                    s = rng.randint(min_side, max_side) / min(im.size)
                    small = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))),
                                      Image.BILINEAR)
                    small.save(dst / f"{img.stem}_lowres.jpg", "JPEG", quality=85)
                added[leaf_dir.name] += 1
    stats = {"fraction": fraction, "short_side_px": [min_side, max_side], "added": dict(sorted(added.items())),
             "total_added": sum(added.values())}
    (out_dir / "LOWRES_COPIES.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Low-res view: {out_dir} (+{stats['total_added']} train copies, short side {min_side}-{max_side} px)")
    return out_dir, stats


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_dataset(args) -> list[str]:
    out_dir: Path = args.out_dataset.resolve()
    mapping = load_mapping(args.mapping)
    unmapped = Counter()
    print("Reading datasets:")
    records: list[Record] = []
    if args.codd_dir:
        records += list(iter_codd(args.codd_dir.resolve(), mapping, unmapped))
    if args.ewaste_dir:
        records += list(iter_folder(args.ewaste_dir.resolve(), "ewaste_small", mapping, unmapped))
    if args.garbage_dir:
        records += list(iter_folder(args.garbage_dir.resolve(), "garbage_v2", mapping, unmapped))
    for leaf, dirs in parse_extras(args.extra).items():
        for d in dirs:
            records += list(iter_extra(leaf, d))
    print(f"  {len(records)} candidate examples; excluded by mapping: {dict(unmapped)}")

    stats: dict = {"excluded_by_mapping": dict(unmapped)}
    records = dedup_and_split(records, args.val_fraction, args.test_fraction, args.seed, stats)

    if args.max_per_class:
        rng = random.Random(args.seed)
        budget = {"train": args.max_per_class,
                  "val": max(1, round(args.max_per_class * args.val_fraction)),
                  "test": max(1, round(args.max_per_class * args.test_fraction))}
        by = defaultdict(list)
        for r in records:
            by[(r.leaf, r.split)].append(r)
        records = []
        for (leaf, split), lst in sorted(by.items()):
            rng.shuffle(lst)
            if len(lst) > budget[split]:
                stats.setdefault("capped", {})[f"{leaf}/{split}"] = len(lst) - budget[split]
            records += lst[:budget[split]]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    written, problems = write_images(records, out_dir, args.pad, args.min_crop, args.max_side)
    stats["write_problems"] = dict(problems)
    written, crop_dups = drop_cross_split_crop_duplicates(written)
    stats["duplicates_dropped"].update(crop_dups)

    counts = Counter((r.split, r.leaf) for r in written)
    leaves = sorted({r.leaf for r in written})
    dropped_leaves = [l for l in leaves
                      if counts[("train", l)] + counts[("val", l)] + counts[("test", l)] < args.min_per_class
                      or not counts[("train", l)] or not counts[("val", l)]]
    for l in dropped_leaves:
        print(f"  [warn] leaf '{l}' has too few images / no train or val -> removed")
        for s in SPLITS:
            shutil.rmtree(out_dir / s / l, ignore_errors=True)
    leaves = [l for l in leaves if l not in dropped_leaves]
    written = [r for r in written if r.leaf in leaves]
    for s in SPLITS:
        for l in leaves:
            (out_dir / s / l).mkdir(parents=True, exist_ok=True)

    over = {"target_train_count": None, "added": {}}
    if args.balance == "oversample":
        over = oversample(written, out_dir, args.max_oversample, args.seed)

    write_summary(out_dir, written, leaves, counts, over, stats, mapping, args)
    hierarchy = defaultdict(list)
    for l in leaves:
        wt, st = l.split(SEP, 1)
        hierarchy[wt].append(st)
    (out_dir / "hierarchy.json").write_text(json.dumps(hierarchy, indent=2), encoding="utf-8")
    return leaves


def write_summary(out_dir, written, leaves, counts, over, stats, mapping, args) -> None:
    src = defaultdict(Counter)
    for r in written:
        src[r.leaf][f"{r.dataset}/{r.source_label}"] += 1
    by_type = defaultdict(lambda: Counter())
    for (s, l), n in counts.items():
        if l in leaves:
            by_type[l.split(SEP)[0]][s] += n
    L = ["MaterialMatch hierarchical classification dataset",
         f"built from   : {DEFAULT_IMAGE_ROOT}",
         f"mapping      : {args.mapping}",
         f"layout       : <split>/<waste_type>{SEP}<sub_type>/  (YOLO classification folders)",
         f"val/test frac: {args.val_fraction}/{args.test_fraction} (stratified, only for sources without an official split)   seed: {args.seed}",
         f"max_side {args.max_side}px  crop pad {args.pad}  min_crop {args.min_crop}px  max_per_class {args.max_per_class}  balance {args.balance} (max x{args.max_oversample})",
         "", f"WASTE TYPES: {len(by_type)}    SUB-TYPES (leaf classes): {len(leaves)}", ""]
    tot = [sum(by_type[w][s] for w in by_type) for s in SPLITS]
    L.append(f"{'waste_type':<14}{'train':>8}{'val':>8}{'test':>8}{'total':>8}")
    for w in sorted(by_type, key=lambda w: -sum(by_type[w].values())):
        c = by_type[w]
        L.append(f"{w:<14}{c['train']:>8}{c['val']:>8}{c['test']:>8}{sum(c.values()):>8}")
    L.append(f"{'ALL':<14}{tot[0]:>8}{tot[1]:>8}{tot[2]:>8}{sum(tot):>8}")
    L += ["", "Leaf classes (train = unique images; +oversampled copies in train only):",
          f"{'leaf':<32}{'train':>7}{'+over':>7}{'val':>6}{'test':>6}  sources"]
    for l in leaves:
        L.append(f"{l:<32}{counts[('train', l)]:>7}{over['added'].get(l, 0):>7}{counts[('val', l)]:>6}"
                 f"{counts[('test', l)]:>6}  {dict(src[l])}")
    tr = {l: counts[("train", l)] for l in leaves}
    ty = {w: sum(by_type[w].values()) for w in by_type}
    L += ["", "Imbalance:",
          f"  leaf train count min {min(tr.values())} ({min(tr, key=tr.get)}), max {max(tr.values())} "
          f"({max(tr, key=tr.get)}), ratio {max(tr.values()) / min(tr.values()):.1f}x, median {statistics.median(tr.values()):.0f}",
          f"  waste-type total min {min(ty.values())} ({min(ty, key=ty.get)}), max {max(ty.values())} "
          f"({max(ty, key=ty.get)}), ratio {max(ty.values()) / min(ty.values()):.1f}x",
          f"  oversampling target (median leaf train count): {over['target_train_count']}",
          "", "Excluded by mapping (status unmapped / not in mapping):"]
    L += [f"  {k}: {v}" for k, v in stats["excluded_by_mapping"].items()] or ["  none"]
    L += ["", f"Duplicates: {stats['near_duplicate_pairs']} near-duplicate pairs, "
          f"{stats['duplicate_groups_multi']} multi-image groups (each group kept in ONE split)"]
    L += [f"  dropped - {k}: {v}" for k, v in stats["duplicates_dropped"].items()]
    L += [f"  write problems - {k}: {v}" for k, v in stats["write_problems"].items()]
    if stats.get("capped"):
        L += [f"  capped - {k}: {v}" for k, v in stats["capped"].items()]
    L += ["", "Mapping used (dataset, source_label -> waste_type/sub_type [status]):"]
    for (ds, lab), r in sorted(mapping.items()):
        tgt = f"{r['canonical_waste_type']}/{r['canonical_sub_type']}" if r["canonical_waste_type"] else "EXCLUDED"
        L.append(f"  {ds:<13}{r['source_label']:<17}-> {tgt:<30}[{r['status']}]")
    text = "\n".join(L) + "\n"
    (out_dir / "DATASET_SUMMARY.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


def check_existing_dataset(ds: Path) -> list[str]:
    if not (ds / "train").exists() or not (ds / "val").exists():
        raise SystemExit(f"{ds} must contain train/<class>/ and val/<class>/")
    tr = sorted(p.name for p in (ds / "train").iterdir() if p.is_dir())
    va = sorted(p.name for p in (ds / "val").iterdir() if p.is_dir())
    if tr != va:
        raise SystemExit(f"train classes {tr} != val classes {va}")
    return tr


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(dataset_dir: Path, args) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.model)
    if model.task != "classify":
        raise SystemExit(f"--model {args.model} is a '{model.task}' model; use a -cls checkpoint.")
    if args.lowres_fraction > 0:
        out = Path(args.lowres_dir) if args.lowres_dir else DEFAULT_DATASET_DIR.parent / "cls_dataset_lowres"
        dataset_dir, _ = lowres_view(dataset_dir, out.resolve(), args.lowres_fraction,
                                     args.lowres_min_side, args.lowres_max_side, args.seed)
    kw = dict(data=str(dataset_dir), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              patience=args.patience, workers=args.workers, seed=args.seed,
              project=str(Path(args.project).resolve()), name=args.name, exist_ok=True,
              plots=True, verbose=False, optimizer=args.optimizer, warmup_epochs=args.warmup_epochs)
    if args.lr0 is not None:
        kw["lr0"] = args.lr0
    if args.robust_aug > 0:
        from ml.classifier import robust_augment
        robust_augment.install(args.robust_aug)
        print(f"Robust augmentation on: {robust_augment.RandomCorruption(args.robust_aug)}")
    if args.crop_scale is not None:
        kw["scale"] = args.crop_scale                  # RandomResizedCrop keeps (1 - scale) .. 100% of the area
    if args.cos_lr:
        kw["cos_lr"] = True
    if args.device:
        kw["device"] = args.device
    model.train(**kw)
    trainer = model.trainer
    best = Path(trainer.best) if trainer.best and Path(trainer.best).exists() else Path(trainer.last)
    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / args.weights_name
    shutil.copy2(best, dest)
    if (dataset_dir / "hierarchy.json").exists():
        shutil.copy2(dataset_dir / "hierarchy.json", dest.with_suffix(".hierarchy.json"))
    print("\n" + "=" * 60)
    print(f"Best checkpoint : {best}\nCopied to       : {dest}")
    print(f"Classes ({len(YOLO(str(dest)).names)}): {list(YOLO(str(dest)).names.values())}")
    print("Evaluate with   : python -m ml.classifier.evaluate_classifier --weights", dest)
    print("=" * 60)
    return dest


def parse_extras(values) -> dict[str, list[Path]]:
    extras = defaultdict(list)
    for v in values or []:
        if "=" not in v or SEP not in v.split("=", 1)[0]:
            raise SystemExit(f"--extra must look like waste_type{SEP}sub_type=folder, got '{v}'")
        leaf, folder = v.split("=", 1)
        extras[leaf.strip()].append(Path(folder).expanduser().resolve())
    return extras


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("data source")
    src.add_argument("--auto", action="store_true", help="use all three datasets under data/Image")
    src.add_argument("--codd-dir", type=Path)
    src.add_argument("--ewaste-dir", type=Path)
    src.add_argument("--garbage-dir", type=Path)
    src.add_argument("--extra", action="append", metavar=f"WASTE_TYPE{SEP}SUB_TYPE=FOLDER")
    src.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    src.add_argument("--dataset", type=Path, help="existing built dataset (skips building)")
    src.add_argument("--out-dataset", type=Path, default=DEFAULT_DATASET_DIR)
    b = ap.add_argument_group("dataset building")
    b.add_argument("--val-fraction", type=float, default=0.15)
    b.add_argument("--test-fraction", type=float, default=0.15)
    b.add_argument("--pad", type=float, default=0.10)
    b.add_argument("--min-crop", type=int, default=24)
    b.add_argument("--max-side", type=int, default=320, help="saved images are resized to this max side")
    b.add_argument("--max-per-class", type=int, default=None, help="optional cap per leaf (off by default)")
    b.add_argument("--min-per-class", type=int, default=30)
    b.add_argument("--balance", choices=("oversample", "none"), default="oversample")
    b.add_argument("--max-oversample", type=float, default=3.0)
    b.add_argument("--build-only", action="store_true")
    t = ap.add_argument_group("training")
    t.add_argument("--model", default="yolov8n-cls.pt")
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--imgsz", type=int, default=224)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--patience", type=int, default=7)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--device", default=None)
    t.add_argument("--seed", type=int, default=42)
    # optimizer=auto picks lr = 0.002*5/(4+nc): 0.00125 for the 4-class v1 but only 0.00032 for 27
    # classes, which under-trains in a short CPU run. Default to v1's proven setting explicitly.
    t.add_argument("--optimizer", default="AdamW")
    t.add_argument("--lr0", type=float, default=0.00125)
    t.add_argument("--warmup-epochs", type=float, default=1.0)
    t.add_argument("--lowres-fraction", type=float, default=0.0,
                   help="add low-res copies of this fraction of train images (fixes the small-photo -> "
                        "e-waste shortcut); trains on a derived dataset, the original is untouched")
    t.add_argument("--robust-aug", type=float, default=0.0,
                   help="probability of 1-2 random corruptions (blur, noise, JPEG, low-res, exposure) per "
                        "training image; see robust_augment.py. 0 = off (as for v2)")
    t.add_argument("--crop-scale", type=float, default=None,
                   help="ultralytics 'scale': random crops keep (1 - scale)..100%% of the area "
                        "(default 0.5 -> 50-100%%; 0.75 -> 25-100%%)")
    t.add_argument("--cos-lr", action="store_true", help="cosine learning-rate schedule")
    t.add_argument("--lowres-min-side", type=int, default=48)
    t.add_argument("--lowres-max-side", type=int, default=150)
    t.add_argument("--lowres-dir", default=None, help="where the derived dataset goes "
                                                      "(default data/processed/cls_dataset_lowres)")
    t.add_argument("--project", default=str(HERE / "runs"))
    t.add_argument("--name", default="materialmatch_hier")
    t.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    t.add_argument("--weights-name", default="best_candidate.pt",
                   help="output file; best.pt is only replaced after evaluation, by hand")
    args = ap.parse_args(argv)
    if args.val_fraction + args.test_fraction >= 1:
        ap.error("--val-fraction + --test-fraction must be < 1")
    if args.auto:
        args.codd_dir = args.codd_dir or DEFAULT_CODD_DIR
        args.ewaste_dir = args.ewaste_dir or DEFAULT_EWASTE_DIR
        args.garbage_dir = args.garbage_dir or DEFAULT_GARBAGE_DIR
    if args.dataset:
        dataset_dir = args.dataset.resolve()
        print(f"Using {dataset_dir}: {len(check_existing_dataset(dataset_dir))} classes")
    else:
        if not (args.codd_dir or args.ewaste_dir or args.garbage_dir or args.extra):
            ap.error("give --auto (or dataset flags), or --dataset")
        dataset_dir = args.out_dataset.resolve()
        build_dataset(args)
    if args.build_only:
        print("--build-only given; skipping training.")
        return None
    return train(dataset_dir, args)


if __name__ == "__main__":
    main()
