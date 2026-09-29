"""
ml/taxonomy.py - The waste_type -> sub_type hierarchy shared by the classifier, API and UI.

Two kinds of sub-type exist and are kept apart on purpose:
  * photo sub-types: what the image model can recognise (data/processed/class_mapping.csv),
    e.g. construction/tile, e_waste/keyboard;
  * marketplace sub-types: rows of data/raw/carbon_factors.csv / waste_listings.csv,
    e.g. concrete_debris, circuit_boards.
A photo sub-type is translated to a marketplace sub-type ONLY where they are the same material
(PHOTO_TO_MARKET). Everything else keeps its own name and has no carbon factor.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING_CSV = ROOT / "data" / "processed" / "class_mapping.csv"

# The four types the marketplace (synthetic listings) was built around.
MARKET_WASTE_TYPES = ("textile", "plastic", "construction", "e_waste")
# Types the image model can also recognise. Sample recycler profiles exist for all of them
# (2026-09-29); carbon factors for paper/cardboard, glass, metal, biological (not trash);
# no synthetic listings.
EXTRA_WASTE_TYPES = ("paper", "glass", "metal", "biological", "trash")
WASTE_TYPES = MARKET_WASTE_TYPES + EXTRA_WASTE_TYPES
UNKNOWN = "unknown"

WASTE_LABELS = {
    "textile": "Textile", "plastic": "Plastic", "construction": "Construction", "e_waste": "E-waste",
    "paper": "Paper & cardboard", "glass": "Glass", "metal": "Metal", "biological": "Organic / food",
    "trash": "Mixed trash", UNKNOWN: "Other / unknown material",
}

SUB_LABELS = {
    "pcb": "PCB / circuit board", "media_player": "Media player (CD/record/radio)",
    "gypsum_board": "Gypsum board", "washing_machine": "Washing machine",
    "foam": "Foam (EPS)", "biological": "Food / organic", "trash": "Mixed trash",
}

# Same material, different name: photo sub-type -> marketplace sub-type (carbon factor row).
PHOTO_TO_MARKET = {
    ("construction", "concrete"): "concrete_debris",
    ("construction", "brick"): "brick_rubble",
    ("construction", "wood"): "wood_waste",
    ("e_waste", "pcb"): "circuit_boards",
}


@lru_cache(maxsize=1)
def photo_hierarchy() -> dict[str, list[str]]:
    """waste_type -> photo sub-types, from class_mapping.csv (empty if the file is missing)."""
    out: dict[str, list[str]] = {}
    if MAPPING_CSV.exists():
        with MAPPING_CSV.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                wt, st = r["canonical_waste_type"].strip(), r["canonical_sub_type"].strip()
                if wt and st and st not in out.setdefault(wt, []):
                    out[wt].append(st)
    return out


def single_subtype(waste_type: str) -> bool:
    """True when the photo model has no finer label than the waste type (glass, metal, ...)."""
    return photo_hierarchy().get(waste_type, []) in ([waste_type], [])


def to_market_subtype(waste_type: str, sub_type: str | None) -> str | None:
    if not sub_type:
        return None
    return PHOTO_TO_MARKET.get((waste_type, sub_type), sub_type)


def market_to_photo(waste_type: str, sub_type: str | None) -> str | None:
    rev = {v: k[1] for k, v in PHOTO_TO_MARKET.items() if k[0] == waste_type}
    return rev.get(sub_type, sub_type) if sub_type else None


def waste_label(t: str | None) -> str:
    return WASTE_LABELS.get(t, t or "Unknown")


def sub_label(s: str | None) -> str:
    if not s or s == UNKNOWN:
        return "Not sure"
    if s in SUB_LABELS:
        return SUB_LABELS[s]
    text = s.replace("_", " ")
    return text[0].upper() + text[1:]
