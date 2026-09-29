"""POST /match -> top-k recyclers; POST /match/listings -> feedstock search."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend import db
from backend.schemas import FeedstockQuery, MatchRequest, MatchResponse
from ml.carbon import co2e_saved
from ml.matcher import NO_RECYCLER_MESSAGE, find_listings, match_recyclers

router = APIRouter(prefix="/match", tags=["match"])


@router.post("", response_model=MatchResponse)
def match(body: MatchRequest):
    if body.waste_id is not None:
        row = db.get_listing(body.waste_id)
        if row is None:
            raise HTTPException(404, f"No listing with waste_id {body.waste_id}")
        waste_type, sub_type = row["waste_type"], row.get("sub_type")
        qty, lat, lon = float(row["quantity_kg"]), float(row["location_lat"]), float(row["location_lon"])
    else:
        missing = [f for f in ("waste_type", "quantity_kg", "lat", "lon") if getattr(body, f) is None]
        if missing:
            raise HTTPException(422, f"Give waste_id, or all of: {', '.join(missing)}")
        waste_type, sub_type, qty, lat, lon = body.waste_type, body.sub_type, body.quantity_kg, body.lat, body.lon

    try:
        carbon = co2e_saved(sub_type, qty, waste_type=waste_type)
    except KeyError as exc:
        raise HTTPException(422, str(exc)) from exc
    matches = match_recyclers(waste_type, qty, lat, lon, sub_type=sub_type, top_k=body.top_k)
    return {
        "waste_type": waste_type,
        "quantity_kg": qty,
        "co2e_saved_kg": carbon["co2e_saved_kg"],
        "carbon_is_proxy": carbon["is_proxy"],
        "carbon_factor_available": carbon["factor_available"],
        "carbon_note": carbon["note"] or ("" if carbon["factor_available"] else carbon["basis"]),
        "matches": matches,
        "message": None if matches else NO_RECYCLER_MESSAGE,
    }


@router.post("/listings")
def match_listings(body: FeedstockQuery):
    return find_listings(body.query, waste_type=body.waste_type, lat=body.lat, lon=body.lon,
                         min_quantity_kg=body.min_quantity_kg, quality=body.quality, top_k=body.top_k,
                         sub_type=body.sub_type)
