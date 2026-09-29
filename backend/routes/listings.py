"""GET/POST waste listings."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend import db
from backend.schemas import Listing, ListingIn, Quality, WasteType

router = APIRouter(prefix="/listings", tags=["listings"])


@router.get("", response_model=list[Listing])
def list_listings(waste_type: WasteType | None = None, quality: Quality | None = None,
                  min_quantity_kg: int = Query(0, ge=0),
                  limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    rows = [r for r in db.all_listings()
            if (waste_type is None or r["waste_type"] == waste_type)
            and (quality is None or r["quality"] == quality)
            and r["quantity_kg"] >= min_quantity_kg]
    return rows[offset:offset + limit]


@router.get("/{waste_id}", response_model=Listing)
def get_listing(waste_id: int):
    row = db.get_listing(waste_id)
    if row is None:
        raise HTTPException(404, f"No listing with waste_id {waste_id}")
    return row


@router.post("", response_model=Listing, status_code=201)
def create_listing(body: ListingIn):
    return db.add_listing(body.model_dump())
