"""
ml/carbon.py - Embodied-carbon savings for MaterialMatch.

    CO2e saved = (virgin - recycled emissions per kg) x kg = quantity_kg * co2e_saved_kg_per_kg

`co2e_saved_kg_per_kg` in carbon_factors.csv is already that virgin-minus-recycled difference.

Factors come from data/raw/carbon_factors.csv (mostly US EPA WARM recycling factors; food /
organic waste uses the WARM composting factor).
They are demonstration / conservative estimates, NOT India-specific LCA results.
Every result carries `is_proxy`; any UI must show proxy results as estimates.

    from ml.carbon import co2e_saved
    co2e_saved("cotton_scrap", 2500)
    co2e_saved(None, 2500, waste_type="textile")   # photo only gives waste_type
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from ml.taxonomy import to_market_subtype

ROOT = Path(__file__).resolve().parents[1]
FACTORS_CSV = ROOT / "data" / "raw" / "carbon_factors.csv"

DISCLAIMER = ("Carbon figures are demonstration estimates based mainly on US EPA WARM "
              "recycling factors, not India-specific LCA results.")

# Concrete/brick have tiny CO2e factors; their real benefit is avoided quarrying and landfill.
LOW_CARBON_NOTE = ("Low direct CO2e saving: the main benefits of reusing this material are "
                   "avoided quarrying and landfill space.")
LOW_CARBON_THRESHOLD = 0.05


def load_factors(path: str | Path = FACTORS_CSV) -> pd.DataFrame:
    """Carbon factors by sub_type; re-read automatically when the CSV changes on disk."""
    try:
        version = Path(path).stat().st_mtime_ns
    except OSError:
        version = 0
    return _read_factors(str(path), version)


@lru_cache(maxsize=2)
def _read_factors(path: str, _version: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["is_proxy"] = df["is_proxy"].astype(str).str.strip().str.lower() == "true"
    return df.set_index("sub_type")


UNAVAILABLE_LABEL = "no carbon factor available"


def _unavailable(reason: str) -> dict:
    return {"factor": None, "is_proxy": None, "available": False, "basis": reason, "source": ""}


def _waste_type_factor(waste_type: str) -> dict:
    df = load_factors()
    rows = df[df["waste_type"] == waste_type]
    if rows.empty:
        return _unavailable(f"No carbon factors for waste type '{waste_type}' in carbon_factors.csv yet")
    return {
        "factor": float(rows["co2e_saved_kg_per_kg"].mean()),
        "is_proxy": True,  # an average across sub_types is always an estimate
        "available": True,
        "basis": f"Mean of {len(rows)} {waste_type} sub_type factors (sub_type unknown)",
        "source": "; ".join(sorted(set(rows["source"].dropna()))),
    }


def get_factor(sub_type: str | None = None, waste_type: str | None = None) -> dict:
    """Factor for a sub_type; the waste_type average (proxy) only when the sub_type is NOT known.

    A known sub-type without its own factor (e.g. photo sub-types like tile or keyboard) and a
    waste type without any factors (paper, glass, ...) return available=False - never a factor
    borrowed from a different material.
    """
    df = load_factors()
    known = sub_type not in (None, "", "unknown")
    market = to_market_subtype(waste_type, sub_type) if (known and waste_type) else sub_type
    if known and market in df.index:
        r = df.loc[market]
        return {"factor": float(r["co2e_saved_kg_per_kg"]), "is_proxy": bool(r["is_proxy"]),
                "available": True, "basis": r["factor_basis"], "source": r["source"]}
    if known:
        if not waste_type:
            raise KeyError(f"Unknown sub_type '{sub_type}' and no waste_type given")
        return _unavailable(f"No carbon factor for sub-type '{sub_type}' in carbon_factors.csv")
    if waste_type:
        return _waste_type_factor(waste_type)
    raise KeyError("Give a sub_type or a waste_type")


def co2e_saved(sub_type: str | None, quantity_kg: float, waste_type: str | None = None) -> dict:
    if quantity_kg < 0:
        raise ValueError("quantity_kg must be non-negative")
    f = get_factor(sub_type, waste_type)
    base = {"sub_type": sub_type, "waste_type": waste_type, "quantity_kg": quantity_kg,
            "factor_available": f["available"], "basis": f["basis"], "source": f["source"]}
    if not f["available"]:
        return {**base, "co2e_saved_kg_per_kg": None, "co2e_saved_kg": None, "co2e_saved_tonnes": None,
                "is_proxy": None, "factor_label": UNAVAILABLE_LABEL, "note": ""}
    saved = quantity_kg * f["factor"]
    note = LOW_CARBON_NOTE if f["factor"] < LOW_CARBON_THRESHOLD else ""
    label = "proxy / conservative estimate" if f["is_proxy"] else "sourced factor (still an estimate)"
    return {
        **base,
        "co2e_saved_kg_per_kg": f["factor"],
        "co2e_saved_kg": round(saved, 2),
        "co2e_saved_tonnes": round(saved / 1000, 3),
        "is_proxy": f["is_proxy"],
        "factor_label": label,
        "note": note,
    }

if __name__ == "__main__":
    for st, q in [("cotton_scrap", 2500), ("concrete_debris", 10000), ("cable_scrap", 500)]:
        r = co2e_saved(st, q)
        print(f"{st:<16} {q:>6} kg -> {r['co2e_saved_kg']:>9} kg CO2e  ({r['factor_label']}) {r['note']}")
    r = co2e_saved(None, 1000, waste_type="plastic")
    print(f"plastic (avg)     1000 kg -> {r['co2e_saved_kg']:>9} kg CO2e  ({r['factor_label']})")
    for st, wt in [("concrete", "construction"), ("tile", "construction"), ("keyboard", "e_waste"),
                   ("glass", "glass"), (None, "paper")]:
        r = co2e_saved(st, 1000, waste_type=wt)
        print(f"{wt}/{st}: {r['co2e_saved_kg']} kg CO2e ({r['factor_label']}; {r['basis']})")
    print(DISCLAIMER)
