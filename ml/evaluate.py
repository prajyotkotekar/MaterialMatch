"""
ml/evaluate.py - Accuracy / match-quality numbers for the pitch.

    python -m ml.evaluate                 # everything
    python -m ml.evaluate --skip-classifier
    python -m ml.evaluate --json results.json

1. Classifier: top-1 + per-class accuracy + confusion matrix on the held-out TEST split
   (data/processed/cls_dataset/test), which training never sees.
2. Listing search: precision@10 / MRR for 16 hand-written paraphrase queries (one per
   sub_type, deliberately avoiding the sub_type's own name) against the SYNTHETIC listings.
3. Recycler matching: coverage and distance to the top match for every synthetic listing.
4. Impact totals over the synthetic listings (potential, not actual, diversion).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ml.carbon import co2e_saved
from ml.embeddings import load_listings
from ml.matcher import embedding_backend, find_listings, load_recyclers, match_listing

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "processed" / "cls_dataset" / "test"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EVAL_QUERIES = {
    "cotton_scrap": "leftover cotton pieces from garment cutting",
    "polyester_waste": "synthetic polyester fabric waste",
    "fabric_offcuts": "mixed cloth offcuts from tailoring units",
    "textile_fiber": "loose textile fibres for spinning",
    "HDPE_film": "high density polyethylene packaging film",
    "LDPE_plastic": "low density polyethylene bags and wrap",
    "PP_waste": "polypropylene containers and woven sacks",
    "PET_bottles": "used soda and water bottles",
    "concrete_debris": "broken concrete slabs from demolition",
    "brick_rubble": "crushed bricks and masonry rubble",
    "wood_waste": "timber offcuts and old wooden pallets",
    "steel_scrap": "rebar and structural steel scrap",
    "circuit_boards": "printed circuit boards from old electronics",
    "metal_components": "metal parts salvaged from appliances",
    "cable_scrap": "copper wires and electrical cables",
    "electronic_components": "chips, capacitors and small electronic parts",
    # newer waste types (synthetic listings added 2026-09-29)
    "paper": "waste office paper and printing offcuts for pulping",
    "cardboard": "flattened corrugated boxes from warehouses",
    "glass": "clean glass bottles crushed to cullet",
    "metal": "aluminium and steel turnings from machine shops",
    "biological": "canteen food scraps for composting",
}


def eval_classifier(batch: int = 64) -> dict:
    from ml.classifier.predict import DEFAULT_WEIGHTS, predict_batch

    if not DEFAULT_WEIGHTS.exists():
        return {"skipped": f"no weights at {DEFAULT_WEIGHTS}"}
    if not TEST_DIR.exists():
        return {"skipped": f"no test split at {TEST_DIR}"}

    # test folders are "<waste_type>__<sub_type>" (v2) or "<waste_type>" (v1)
    items = [(p, d.name) for d in sorted(TEST_DIR.iterdir()) if d.is_dir()
             for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
    classes = sorted({c.split("__")[0] for _, c in items})
    confusion = {t: Counter() for t in classes}
    leaf_ok = unknown = 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        for (_, folder), res in zip(chunk, predict_batch([p for p, _ in chunk], top_k=1)):
            guess = res.get("best_guess") or {"waste_type": res["label"], "sub_type": None}
            confusion[folder.split("__")[0]][guess["waste_type"]] += 1
            leaf_ok += "__" in folder and folder == f"{guess['waste_type']}__{guess['sub_type']}"
            unknown += bool(res.get("is_unknown"))

    total = sum(sum(c.values()) for c in confusion.values())
    correct = sum(confusion[c][c] for c in classes)
    return {
        "split": "test (held out)",
        "n_images": total,
        "top1_accuracy": round(correct / total, 4),
        "subtype_top1_accuracy": round(leaf_ok / total, 4) if any("__" in c for _, c in items) else None,
        "flagged_unknown": round(unknown / total, 4),
        "per_class_accuracy": {c: round(confusion[c][c] / max(1, sum(confusion[c].values())), 4)
                               for c in classes},
        "confusion_matrix": {t: dict(confusion[t]) for t in classes},
        "note": "waste-type accuracy uses the best guess even when the photo is flagged unknown; "
                "full report: python -m ml.classifier.tools.evaluate_classifier",
    }


def eval_search(k: int = 10) -> dict:
    listings = load_listings()
    per_query, precisions, type_precisions, rr = {}, [], [], []
    for sub_type, query in EVAL_QUERIES.items():
        waste_type = listings.loc[listings["sub_type"] == sub_type, "waste_type"].iloc[0]
        ranked = find_listings(query, top_k=len(listings))
        top = ranked[:k]
        p = sum(r["sub_type"] == sub_type for r in top) / k
        tp = sum(r["waste_type"] == waste_type for r in top) / k
        first = next((i for i, r in enumerate(ranked, 1) if r["sub_type"] == sub_type), None)
        precisions.append(p)
        type_precisions.append(tp)
        rr.append(1 / first if first else 0.0)
        per_query[sub_type] = {"query": query, f"precision@{k}": p,
                               f"waste_type_precision@{k}": tp, "first_relevant_rank": first}
    return {
        "backend": embedding_backend(),
        "n_queries": len(EVAL_QUERIES),
        "data": "SYNTHETIC listings; hand-written paraphrase queries",
        f"mean_precision@{k}": round(float(np.mean(precisions)), 3),
        f"mean_waste_type_precision@{k}": round(float(np.mean(type_precisions)), 3),
        "mrr": round(float(np.mean(rr)), 3),
        "per_query": per_query,
    }


def eval_recycler_matching() -> dict:
    listings = load_listings()
    covered, dists, specialist_top = 0, [], 0
    by_type = defaultdict(list)
    for wid, wt in zip(listings["waste_id"], listings["waste_type"]):
        m = match_listing(int(wid), top_k=1)
        if m:
            covered += 1
            dists.append(m[0]["distance_km"])
            by_type[wt].append(m[0]["distance_km"])
            specialist_top += "," not in m[0]["waste_types"]
    n = len(listings)
    return {
        "n_listings": n,
        "coverage": round(covered / n, 3),
        "top_match_is_specialist": round(specialist_top / max(1, covered), 3),
        "median_km_to_top_match": round(float(np.median(dists)), 1) if dists else None,
        "median_km_by_waste_type": {k: round(float(np.median(v)), 1) for k, v in sorted(by_type.items())},
        "note": f"{len(load_recyclers())} sample recycler profiles; no real partnerships",
    }


def impact_totals() -> dict:
    listings = load_listings()
    rows = [co2e_saved(r.sub_type, r.quantity_kg, waste_type=r.waste_type) for r in listings.itertuples()]
    with_factor = [r for r in rows if r["factor_available"]]  # e.g. mixed trash has no carbon factor
    total_kg = float(listings["quantity_kg"].sum())
    total_co2 = sum(r["co2e_saved_kg"] for r in with_factor)
    proxy_co2 = sum(r["co2e_saved_kg"] for r in with_factor if r["is_proxy"])
    return {
        "data": "SYNTHETIC listings - potential if all were reused, not actual diversion",
        "total_kg": total_kg,
        "listings_without_carbon_factor": len(rows) - len(with_factor),
        "total_co2e_saved_tonnes": round(total_co2 / 1000, 2),
        "share_of_co2e_from_proxy_factors": round(proxy_co2 / total_co2, 3) if total_co2 else 0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--skip-classifier", action="store_true")
    ap.add_argument("--json", type=Path, help="also write results to this JSON file")
    args = ap.parse_args(argv)

    results = {}
    if not args.skip_classifier:
        results["classifier"] = eval_classifier()
    results["listing_search"] = eval_search()
    results["recycler_matching"] = eval_recycler_matching()
    results["impact"] = impact_totals()

    for section, res in results.items():
        print(f"\n== {section} ==")
        for key, val in res.items():
            if key == "per_query":
                for st_, q in val.items():
                    print(f"    {st_:<22} P@10={q['precision@10']:.1f}  "
                          f"type-P@10={q['waste_type_precision@10']:.1f}  "
                          f"first={q['first_relevant_rank']}  '{q['query']}'")
            else:
                print(f"  {key}: {val}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
