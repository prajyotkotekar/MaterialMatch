"""
validate_dataset.py - Independent leakage / integrity check of a built classification dataset.

    python ml/classifier/validate_dataset.py [--dataset data/processed/cls_dataset]

Checks the OUTPUT folders (including CODD crops): every split has the same classes, class
names parse as <waste_type>__<sub_type> and exist in class_mapping.csv, no file name is reused
across splits, no exact (md5) duplicate and no near-duplicate (audit_datasets rule) spans two
splits, and oversampled "_rep" copies only live in train. Writes LEAKAGE_REPORT.txt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_datasets import duplicate_pairs, image_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def feat(p: Path):
    data = p.read_bytes()
    with Image.open(p) as im:
        return hashlib.md5(data).hexdigest(), image_features(im)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "processed" / "cls_dataset")
    ap.add_argument("--mapping", type=Path, default=ROOT / "data" / "processed" / "class_mapping.csv")
    a = ap.parse_args(argv)
    ds = a.dataset.resolve()
    failures, L = [], []

    classes = {s: sorted(p.name for p in (ds / s).iterdir() if p.is_dir()) for s in ("train", "val", "test")}
    if not (classes["train"] == classes["val"] == classes["test"]):
        failures.append("splits have different class folders")
    with a.mapping.open(encoding="utf-8") as f:
        allowed = {f"{r['canonical_waste_type']}__{r['canonical_sub_type']}" for r in csv.DictReader(f)
                   if r["canonical_waste_type"]}
    bad = [c for c in classes["train"] if "__" not in c or c not in allowed]
    if bad:
        failures.append(f"class folders not in mapping: {bad}")
    L.append(f"classes: {len(classes['train'])} (identical in train/val/test: {not failures})")

    files = [p for s in ("train", "val", "test") for p in (ds / s).rglob("*.jpg")]
    rep_outside_train = [p for p in files if "_rep" in p.stem and p.parts[-3] != "train"]
    if rep_outside_train:
        failures.append(f"{len(rep_outside_train)} oversampled copies outside train")
    uniq = [p for p in files if "_rep" not in p.stem]
    stems = Counter(p.stem for p in uniq)
    collisions = [s for s, n in stems.items() if n > 1]
    if collisions:
        failures.append(f"{len(collisions)} file names used more than once, e.g. {collisions[:3]}")
    L.append(f"files: {len(files)} total, {len(uniq)} unique (non-oversampled), "
             f"{len(files) - len(uniq)} oversampled copies (all in train: {not rep_outside_train})")
    L.append(f"file-name collisions across splits/classes: {len(collisions)}")

    print(f"hashing {len(uniq)} images...", flush=True)
    with ThreadPoolExecutor(12) as ex:
        feats = list(ex.map(feat, uniq, chunksize=64))
    split = [p.parts[-3] for p in uniq]
    cls = [p.parts[-2] for p in uniq]
    md5s = Counter()
    by_md5 = {}
    exact_cross = 0
    for i, (m, _) in enumerate(feats):
        if m in by_md5 and split[by_md5[m]] != split[i]:
            exact_cross += 1
        by_md5.setdefault(m, i)
        md5s[m] += 1
    exact_within = sum(n - 1 for n in md5s.values() if n > 1)
    pairs = duplicate_pairs([f for _, f in feats])
    cross = Counter()
    examples = []
    for x, y in pairs:
        if split[x] != split[y]:
            key = f"{split[x]}~{split[y]}  {cls[x]}" + ("" if cls[x] == cls[y] else f" ~ {cls[y]}")
            cross[key] += 1
            if len(examples) < 10:
                examples.append(f"{uniq[x].relative_to(ds)}  <->  {uniq[y].relative_to(ds)}")
    same = sum(split[x] == split[y] for x, y in pairs)
    L.append(f"exact duplicates across splits: {exact_cross}   (within one split: {exact_within})")
    L.append(f"near-duplicate pairs across splits: {sum(cross.values())}   (within one split: {same})")
    L += [f"  {k}: {v}" for k, v in cross.most_common()]
    L += ["  e.g. " + e for e in examples]
    if exact_cross:
        failures.append(f"{exact_cross} exact duplicates across splits")
    L.append("")
    L.append("RESULT: " + ("PASS" if not failures else "FAIL - " + "; ".join(failures)))
    text = "\n".join(L) + "\n"
    (ds / "LEAKAGE_REPORT.txt").write_text(text, encoding="utf-8")
    print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
