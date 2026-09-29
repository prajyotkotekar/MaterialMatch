"""
ml/matcher.py - score = similarity + distance + capacity.

  1. Waste generator -> ranked recyclers
         match_recyclers("textile", quantity_kg=2500, lat=12.98, lon=77.70)
         match_listing(waste_id=1)
  2. Manufacturer feedstock need -> ranked waste listings
         find_listings("clean PET bottles for flakes", lat=12.97, lon=77.59)

All listings are SYNTHETIC demo data (`is_synthetic`) and recyclers are sample profiles
(`is_sample_profile`). Carbon numbers come from ml.carbon and carry `is_proxy`.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ml.carbon import co2e_saved
from ml.embeddings import file_version, listing_index, listing_text, load_listings

ROOT = Path(__file__).resolve().parents[1]
RECYCLERS_CSV = ROOT / "data" / "raw" / "recyclers.csv"

DISTANCE_SCALE_KM = 25.0
QUALITY_SCORE = {"good": 1.0, "fair": 0.7, "poor": 0.4}

RECYCLER_WEIGHTS = {"distance": 0.5, "capacity": 0.3, "specialist": 0.2}
LISTING_WEIGHTS = {"similarity": 0.6, "distance": 0.25, "quality": 0.15}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def distance_score(km):
    return np.exp(-np.asarray(km, dtype=float) / DISTANCE_SCALE_KM)


def load_recyclers() -> pd.DataFrame:
    return _read_recyclers(file_version(RECYCLERS_CSV))


@lru_cache(maxsize=1)
def _read_recyclers(_version: int) -> pd.DataFrame:
    df = pd.read_csv(RECYCLERS_CSV)
    df["waste_type_list"] = df["waste_types"].str.split(",").apply(lambda xs: [x.strip() for x in xs])
    return df


NO_RECYCLER_MESSAGE = "No matching recycler profile available for this material yet."


def match_recyclers(waste_type: str, quantity_kg: float, lat: float, lon: float,
                    sub_type: str | None = None, top_k: int = 5) -> list[dict]:
    """Rank recyclers that accept `waste_type`, by distance, monthly capacity and specialisation.

    Returns [] when no recycler profile in recyclers.csv lists the waste type; recyclers are
    never invented at runtime.
    """
    rec = load_recyclers()
    rec = rec[rec["waste_type_list"].apply(lambda ts: waste_type in ts)].copy()
    if rec.empty:
        return []

    rec["distance_km"] = haversine_km(lat, lon, rec["lat"].values, rec["lon"].values)
    rec["distance_score"] = distance_score(rec["distance_km"])
    rec["capacity_score"] = np.minimum(1.0, rec["capacity_kg_per_month"] / max(quantity_kg, 1))
    rec["specialist_score"] = 1.0 / rec["waste_type_list"].apply(len)
    w = RECYCLER_WEIGHTS
    rec["score"] = sum(w[k] * rec[f"{k}_score"] for k in w)
    rec = rec.sort_values("score", ascending=False).head(top_k)

    carbon = co2e_saved(sub_type, quantity_kg, waste_type=waste_type)
    return [{
        "recycler_id": int(r["recycler_id"]),
        "name": r["name"],
        "waste_types": r["waste_types"],
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
        "distance_km": round(float(r["distance_km"]), 1),
        "capacity_kg_per_month": int(r["capacity_kg_per_month"]),
        "can_absorb_in_one_month": bool(r["capacity_kg_per_month"] >= quantity_kg),
        "score": round(float(r["score"]), 3),
        "score_breakdown": {k: round(float(r[f"{k}_score"]), 3) for k in w},
        "co2e_saved_kg": carbon["co2e_saved_kg"],
        "carbon_is_proxy": carbon["is_proxy"],
        "is_sample_profile": True,
    } for _, r in rec.iterrows()]


def match_listing(waste_id: int, top_k: int = 5) -> list[dict]:
    df = load_listings()
    row = df[df["waste_id"] == waste_id]
    if row.empty:
        raise KeyError(f"No listing with waste_id {waste_id}")
    r = row.iloc[0]
    return match_recyclers(r["waste_type"], float(r["quantity_kg"]), float(r["location_lat"]),
                           float(r["location_lon"]), sub_type=r["sub_type"], top_k=top_k)


def embedding_backend() -> str:
    return listing_index().backend


LISTING_COLUMNS = ["waste_id", "waste_type", "sub_type", "quantity_kg", "quality", "location_name", "location_lat",
                   "location_lon", "seller_name", "seller_contact", "price_per_kg", "description"]


def browse_weights() -> dict:
    """Listing weights without the text-similarity term (blank search): distance and quality, renormalised."""
    rest = {k: v for k, v in LISTING_WEIGHTS.items() if k != "similarity"}
    total = sum(rest.values())
    return {k: v / total for k, v in rest.items()}


def _session_frame(extra_listings: list[dict]) -> pd.DataFrame:
    """Listings a user published in this session (not in the CSV), shaped like the demo listings."""
    df = pd.DataFrame(extra_listings)
    for col in [*LISTING_COLUMNS, "pickup_location", "pickup_address"]:
        if col not in df:
            df[col] = ""
    df["sub_type"] = df["sub_type"].fillna("unknown")
    return df


def find_listings(query: str = "", waste_type: str | None = None, lat: float | None = None,
                  lon: float | None = None, min_quantity_kg: float = 0,
                  quality: list[str] | None = None, top_k: int = 10,
                  sub_type: str | None = None, extra_listings: list[dict] | None = None) -> list[dict]:
    """Rank waste listings for a manufacturer's free-text feedstock need.

    An empty `query` is "browse": no text similarity, so listings are ranked by distance and quality only
    (the similarity weight is dropped and the other two are renormalised). `min_quantity_kg` keeps only
    listings that hold at least that much. `extra_listings` are listings published in this session (page 1);
    they are ranked together with the synthetic demo listings and returned with `is_synthetic=False`.
    `co2e_saved_kg` in each result is for the listing's WHOLE batch; `co2e_per_kg` is the factor, for sizing
    it to the amount a buyer needs."""
    has_query = bool((query or "").strip())
    df = load_listings().copy()
    df["source"] = "Demo"
    df["pickup_location"] = ""
    df["pickup_address"] = ""
    df["is_synthetic"] = True
    df["similarity"] = listing_index().similarity(query) if has_query else 0.0

    if extra_listings:
        mine = _session_frame(extra_listings)
        mine["source"] = "Your listing"
        mine["is_synthetic"] = False
        texts = [listing_text({**r, "sub_type": "" if r["sub_type"] == "unknown" else r["sub_type"]})
                 for _, r in mine.iterrows()]
        mine["similarity"] = listing_index().similarity_texts(texts, query) if has_query else 0.0
        df = pd.concat([df, mine[[*LISTING_COLUMNS, "source", "pickup_location", "pickup_address",
                                  "is_synthetic", "similarity"]]], ignore_index=True)

    # Scale the text similarity over the WHOLE catalogue, before any filter, so a listing's match % does not
    # change with the quality/type/quantity filters (min-max over the filtered subset made the top hit always 100%).
    sim = df["similarity"]
    df["similarity_score"] = (sim - sim.min()) / (sim.max() - sim.min()) if sim.max() > sim.min() else 1.0

    mask = df["quantity_kg"] >= min_quantity_kg
    if waste_type:
        mask &= df["waste_type"] == waste_type
    if sub_type:
        mask &= df["sub_type"] == sub_type
    if quality:
        mask &= df["quality"].isin(quality)
    df = df[mask].copy()
    if df.empty:
        return []

    if lat is not None and lon is not None:
        df["distance_km"] = haversine_km(lat, lon, df["location_lat"].astype(float).values,
                                         df["location_lon"].astype(float).values)
        df["distance_score"] = distance_score(df["distance_km"])
    else:
        df["distance_km"] = math.nan
        df["distance_score"] = 1.0
    df["quality_score"] = df["quality"].map(QUALITY_SCORE).fillna(0.5)
    w = LISTING_WEIGHTS if has_query else browse_weights()
    df["score"] = sum(w[k] * df[f"{k}_score"] for k in w)
    # ties (no location, same quality, ...) fall back to the larger batch, then a stable id order
    df = df.sort_values(["score", "quantity_kg", "waste_id"], ascending=[False, False, True]).head(top_k)

    out = []
    for _, r in df.iterrows():
        c = co2e_saved(r["sub_type"], r["quantity_kg"], waste_type=r["waste_type"])
        out.append({
            "waste_id": int(r["waste_id"]),
            "waste_type": r["waste_type"],
            "sub_type": r["sub_type"],
            "quantity_kg": int(r["quantity_kg"]),
            "quality": r["quality"],
            "location_name": r["location_name"],
            "lat": float(r["location_lat"]),
            "lon": float(r["location_lon"]),
            "distance_km": None if math.isnan(r["distance_km"]) else round(float(r["distance_km"]), 1),
            "seller_name": r["seller_name"],
            "seller_contact": "" if pd.isna(r["seller_contact"]) else str(r["seller_contact"]),
            "price_per_kg": int(r["price_per_kg"]),
            "description": r["description"],
            "source": r["source"],
            "pickup_location": r["pickup_location"],
            "pickup_address": r["pickup_address"],
            "has_query": has_query,
            "similarity": round(float(r["similarity"]), 3),
            "score": round(float(r["score"]), 3),
            "score_weights": w,
            "score_breakdown": {k: round(float(r[f"{k}_score"]), 3) for k in w},
            "co2e_saved_kg": c["co2e_saved_kg"],  # whole batch; None when the material has no factor
            "co2e_per_kg": c["co2e_saved_kg_per_kg"],
            "carbon_is_proxy": c["is_proxy"],
            "is_synthetic": bool(r["is_synthetic"]),
        })
    return out


if __name__ == "__main__":
    print("Recyclers for listing #1 (SYNTHETIC):")
    for m in match_listing(1):
        print(f"  {m['score']:.3f}  {m['name']:<30} {m['distance_km']:>5} km  CO2e {m['co2e_saved_kg']} kg")
    q = "clean PET bottles to make recycled flakes"
    print(f"\nListings for '{q}' [backend: {embedding_backend()}] (SYNTHETIC):")
    for m in find_listings(q, lat=12.9756, lon=77.6050, top_k=5):
        print(f"  {m['score']:.3f}  #{m['waste_id']:<4} {m['sub_type']:<16} {m['quantity_kg']:>6} kg")
