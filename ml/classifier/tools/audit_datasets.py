"""
audit_datasets.py - Inventory every image dataset under data/Image before building a training set.

    python ml/classifier/tools/audit_datasets.py            # writes data/processed/dataset_audit/

Reads the actual files (never a hard-coded mapping) and reports, per dataset and label:
image counts per split, annotation format, image sizes, unreadable files, CODD object/box
statistics, and exact (md5) + near-duplicate (see duplicate_pairs) images
within and across datasets/splits.
"""

from __future__ import annotations

import hashlib
import json
import statistics as st
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
IMAGE_ROOT = ROOT / "data" / "Image"
OUT = ROOT / "data" / "processed" / "dataset_audit"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}
# Near-duplicate rule (calibrated by eye on 2026-09-28 sample sheets): a 64-bit dHash alone
# pairs unrelated product shots on white backgrounds, so a candidate (dHash64 <= 10 bits) is
# only a duplicate if the 256-bit dHash differs by <= 22 bits AND 24x24 RGB thumbnails differ
# by <= 8 grey levels on average.
CAND_BITS, D256_BITS, THUMB_MAD = 10, 22, 8.0


def dhash(img: Image.Image, size: int = 8) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def image_features(img: Image.Image) -> dict:
    b64 = dhash(img, 8)
    return {"dhash": int("".join("1" if b else "0" for b in b64), 2),
            "d256": np.packbits(dhash(img, 16)),
            "thumb": np.asarray(img.convert("RGB").resize((24, 24), Image.Resampling.BOX),
                                dtype=np.uint8)}


def duplicate_pairs(feats: list[dict]) -> list[tuple[int, int]]:
    """Index pairs (i, j) of exact or near-duplicate images, using the rule above."""
    ok = [i for i, f in enumerate(feats) if f is not None]
    h = np.array([feats[i]["dhash"] for i in ok], dtype=np.uint64)
    d256 = np.stack([feats[i]["d256"] for i in ok])
    th = np.stack([feats[i]["thumb"] for i in ok]).reshape(len(ok), -1).astype(np.float32)
    pairs = []
    for a in range(len(ok)):
        cand = np.nonzero(np.bitwise_count(h[a + 1:] ^ h[a]) <= CAND_BITS)[0] + a + 1
        if not len(cand):
            continue
        dd = np.bitwise_count(d256[cand] ^ d256[a]).sum(1)
        mad = np.abs(th[cand] - th[a]).mean(1)
        for b in cand[(dd <= D256_BITS) & (mad <= THUMB_MAD)]:
            pairs.append((ok[a], ok[int(b)]))
    return pairs


def inspect_file(path: Path) -> dict:
    rec = {"path": str(path.relative_to(IMAGE_ROOT)).replace("\\", "/")}
    data = path.read_bytes()
    rec["md5"] = hashlib.md5(data).hexdigest()
    rec["bytes"] = len(data)
    try:
        with Image.open(path) as im:
            rec["format"], rec["mode"] = im.format, im.mode
            im = ImageOps.exif_transpose(im)
            rec["w"], rec["h"] = im.size
            rec.update(image_features(im))
    except Exception as exc:
        rec["error"] = str(exc)[:120]
    return rec


def describe_sizes(recs: list[dict]) -> dict:
    ok = [r for r in recs if "w" in r]
    if not ok:
        return {}
    ws, hs = [r["w"] for r in ok], [r["h"] for r in ok]
    common = Counter(f"{r['w']}x{r['h']}" for r in ok).most_common(3)
    return {"min_side": min(min(r["w"], r["h"]) for r in ok), "median_w": int(st.median(ws)),
            "median_h": int(st.median(hs)), "max_side": max(max(r["w"], r["h"]) for r in ok),
            "most_common": common, "modes": dict(Counter(r["mode"] for r in ok))}


def folder_records(root: Path, dataset: str) -> list[dict]:
    """Whole-image datasets: label = last non-split folder; split = split folder if any."""
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = p.relative_to(root).parts[:-1]
        split = next((SPLIT_ALIASES[x.lower()] for x in parts if x.lower() in SPLIT_ALIASES), None)
        label = next((x for x in reversed(parts) if x.lower() not in SPLIT_ALIASES), None)
        out.append({"dataset": dataset, "label": label, "split": split, "file": p})
    return out


def codd_audit(root: Path) -> tuple[list[dict], dict]:
    xmls = sorted(root.rglob("*.xml"))
    scenes, objects = [], []
    problems = Counter()
    fields = Counter()
    for x in xmls:
        split = SPLIT_ALIASES.get(x.parent.name.lower())
        img = next((x.with_suffix(e) for e in (".jpg", ".jpeg", ".png", ".JPG") if x.with_suffix(e).exists()), None)
        if img is None:
            problems["xml_without_image"] += 1
            continue
        try:
            tree = ET.parse(x)
        except ET.ParseError:
            problems["malformed_xml"] += 1
            continue
        for el in tree.getroot():
            fields[el.tag] += 1
        labels = []
        for obj in tree.iter("object"):
            name = (obj.findtext("name") or "").strip()
            bb = obj.find("bndbox")
            try:
                x0, x1 = float(bb.findtext("xmin")), float(bb.findtext("xmax"))
                y0, y1 = float(bb.findtext("ymin")), float(bb.findtext("ymax"))
            except Exception:
                problems["missing_box"] += 1
                continue
            if x1 <= x0 or y1 <= y0:
                problems["degenerate_box"] += 1
                continue
            labels.append(name)
            objects.append({"label": name, "split": split, "w": x1 - x0, "h": y1 - y0,
                            "has_polygon": obj.find("polygon") is not None,
                            "difficult": (obj.findtext("difficult") or "0").strip() == "1",
                            "truncated": (obj.findtext("truncated") or "0").strip() == "1",
                            "occluded": (obj.findtext("occluded") or "0").strip() == "1"})
        scenes.append({"dataset": "codd", "label": "__scene__", "split": split, "file": img,
                       "objects": labels})
    # images with no xml
    imgs = {p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    problems["image_without_xml"] = len(imgs - {s["file"] for s in scenes})
    per_label = defaultdict(lambda: Counter())
    sizes = defaultdict(list)
    flags = defaultdict(Counter)
    for o in objects:
        per_label[o["label"]][o["split"]] += 1
        sizes[o["label"]].append(min(o["w"], o["h"]))
        for f in ("has_polygon", "difficult", "truncated", "occluded"):
            flags[o["label"]][f] += o[f]
    cooc = Counter()
    for s in scenes:
        u = sorted(set(s["objects"]))
        for i, a in enumerate(u):
            for b in u[i + 1:]:
                cooc[f"{a}+{b}"] += 1
    labels_per_scene = Counter(len(set(s["objects"])) for s in scenes)
    summary = {
        "annotation_files": len(xmls), "scenes": len(scenes), "objects": len(objects),
        "xml_root_fields": dict(fields), "problems": dict(problems),
        "labels": {lab: {"objects": sum(c.values()), "by_split": dict(c),
                         "scenes_containing": sum(lab in s["objects"] for s in scenes),
                         "box_short_side_median": round(st.median(sizes[lab]), 1),
                         "box_short_side_p10": round(float(np.percentile(sizes[lab], 10)), 1),
                         "boxes_short_side_lt_24px": sum(v < 24 for v in sizes[lab]),
                         "boxes_short_side_lt_64px": sum(v < 64 for v in sizes[lab]),
                         **{k: int(v) for k, v in flags[lab].items()}}
                   for lab, c in sorted(per_label.items(), key=lambda kv: -sum(kv[1].values()))},
        "distinct_labels_per_scene": dict(sorted(labels_per_scene.items())),
        "top_label_cooccurrence": cooc.most_common(12),
    }
    return scenes, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Scanning datasets...")
    codd_scenes, codd_summary = codd_audit(IMAGE_ROOT / "codd")
    records = codd_scenes + folder_records(IMAGE_ROOT / "ewaste_small", "ewaste_small") \
        + folder_records(IMAGE_ROOT / "garbage_v2", "garbage_v2")
    other = sorted({p.parent.relative_to(IMAGE_ROOT).parts[0] for p in IMAGE_ROOT.rglob("*")
                    if p.is_file()} - {"codd", "ewaste_small", "garbage_v2"})
    print(f"  {len(records)} images to inspect; other top-level folders: {other or 'none'}")
    with ThreadPoolExecutor(12) as ex:
        info = list(ex.map(inspect_file, [r["file"] for r in records], chunksize=64))
    for r, i in zip(records, info):
        r.update(i)

    # ---- per dataset / label inventory
    inventory = []
    by_key = defaultdict(list)
    for r in records:
        by_key[(r["dataset"], r["label"])].append(r)
    for (ds, lab), rs in sorted(by_key.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        inventory.append({"dataset": ds, "label": lab, "images": len(rs),
                          "by_split": dict(Counter(r["split"] or "none" for r in rs)),
                          "unreadable": sum("error" in r for r in rs),
                          "sizes": describe_sizes(rs)})

    # ---- exact duplicates
    by_md5 = defaultdict(list)
    for i, r in enumerate(records):
        by_md5[r["md5"]].append(i)
    exact_groups = [g for g in by_md5.values() if len(g) > 1]
    exact = Counter()
    exact_examples = []
    for g in exact_groups:
        keys = {(records[i]["dataset"], records[i]["label"], records[i]["split"]) for i in g}
        kind = ("cross_dataset" if len({k[0] for k in keys}) > 1 else
                "cross_label" if len({k[1] for k in keys}) > 1 else
                "cross_split" if len({k[2] for k in keys}) > 1 else "same_folder")
        exact[kind] += 1
        if len(exact_examples) < 25 and kind != "same_folder":
            exact_examples.append([records[i]["path"] for i in g])
    exact_by_label = Counter()
    for g in exact_groups:
        for i in g[1:]:
            exact_by_label[f"{records[i]['dataset']}/{records[i]['label']}"] += 1

    # ---- near duplicates (whole images only: CODD scenes share one studio background, and are
    # cropped anyway - crop-level leakage is checked on the built dataset instead)
    print("  near-duplicate search...")
    whole = [i for i, r in enumerate(records) if r["dataset"] != "codd"]
    pairs = duplicate_pairs([records[i] if "dhash" in records[i] else None for i in whole])
    near = Counter()
    near_examples = defaultdict(list)
    for a, b in pairs:
        ra, rb = records[whole[a]], records[whole[b]]
        if ra["md5"] == rb["md5"]:
            continue
        if ra["dataset"] != rb["dataset"]:
            kind = f"cross_dataset {ra['dataset']}/{ra['label']} ~ {rb['dataset']}/{rb['label']}"
        elif ra["label"] != rb["label"]:
            kind = f"cross_label {ra['dataset']}: {ra['label']} ~ {rb['label']}"
        elif ra["split"] != rb["split"]:
            kind = f"cross_split {ra['dataset']}/{ra['label']}"
        else:
            kind = f"same_folder {ra['dataset']}/{ra['label']}"
        near[kind] += 1
        if len(near_examples[kind]) < 4:
            near_examples[kind].append([ra["path"], rb["path"]])
    report = {
        "image_root": str(IMAGE_ROOT), "total_images": len(records),
        "other_top_level_folders": other,
        "inventory": inventory, "codd": codd_summary,
        "exact_duplicates": {"groups": len(exact_groups), "by_kind": dict(exact),
                             "redundant_copies_by_folder": dict(exact_by_label.most_common()),
                             "examples": exact_examples},
        "near_duplicates": {"rule": f"dHash64<={CAND_BITS} & dHash256<={D256_BITS} & thumbMAD<={THUMB_MAD}",
                            "pairs_excluding_exact": sum(near.values()),
                            "by_kind": dict(near.most_common()), "examples": dict(near_examples)},
        "unreadable_files": [r["path"] for r in records if "error" in r],
    }
    (OUT / "audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    # per-file hashes, reused by the dataset builder for leakage control
    with (OUT / "file_hashes.tsv").open("w", encoding="utf-8") as f:
        f.write("path\tdataset\tlabel\tsplit\tmd5\tdhash\tw\th\n")
        for r in records:
            f.write(f"{r['path']}\t{r['dataset']}\t{r['label']}\t{r['split'] or ''}\t{r['md5']}\t"
                    f"{r.get('dhash', '')}\t{r.get('w', '')}\t{r.get('h', '')}\n")
    write_markdown(report)
    print(f"Wrote {OUT / 'audit.json'} and file_hashes.tsv")


def write_markdown(a: dict) -> None:
    L = ["# Dataset audit (generated by ml/classifier/tools/audit_datasets.py)", "",
         f"Total images: {a['total_images']} · unreadable: {len(a['unreadable_files'])} · "
         f"other folders under data/Image: {a['other_top_level_folders'] or 'none'}", "",
         "## Folder datasets", "", "| dataset | label | images | train/val/test (none = no official split) | size (median) | min side |",
         "|---|---|---:|---|---|---:|"]
    for r in a["inventory"]:
        if r["label"] == "__scene__":
            continue
        s = r["sizes"]
        L.append(f"| {r['dataset']} | {r['label']} | {r['images']} | {r['by_split']} | "
                 f"{s.get('median_w')}x{s.get('median_h')} | {s.get('min_side')} |")
    c = a["codd"]
    L += ["", "## CODD (Pascal-VOC XML, bounding box + polygon per object)", "",
          f"{c['annotation_files']} XML files, {c['scenes']} scenes (1920x1200), {c['objects']} objects. "
          f"Problems: {c['problems']}.", "",
          "| label | objects | train/val/test | scenes | median box short side | boxes <24px | boxes <64px |",
          "|---|---:|---|---:|---:|---:|---:|"]
    for lab, v in c["labels"].items():
        L.append(f"| {lab} | {v['objects']} | {v['by_split']} | {v['scenes_containing']} | "
                 f"{v['box_short_side_median']} | {v['boxes_short_side_lt_24px']} | {v['boxes_short_side_lt_64px']} |")
    L += ["", f"Distinct labels per scene: {c['distinct_labels_per_scene']}",
          f"Most frequent label co-occurrence: {c['top_label_cooccurrence'][:6]}", "",
          "## Duplicates", "", f"Exact (md5): {a['exact_duplicates']['groups']} groups {a['exact_duplicates']['by_kind']}; "
          f"redundant copies: {a['exact_duplicates']['redundant_copies_by_folder']}", "",
          f"Near duplicates ({a['near_duplicates']['rule']}), excluding exact: "
          f"{a['near_duplicates']['pairs_excluding_exact']} pairs", ""]
    L += [f"- {k}: {v}" for k, v in a["near_duplicates"]["by_kind"].items()]
    (OUT / "DATASET_AUDIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
