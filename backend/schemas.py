"""backend/schemas.py - Pydantic request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The 4 marketplace types plus the types the v2 photo model recognises (see ml/taxonomy.py).
# All 9 have sample recycler profiles; only trash has no carbon factor.
WasteType = Literal["textile", "plastic", "construction", "e_waste",
                    "paper", "glass", "metal", "biological", "trash"]
Quality = Literal["good", "fair", "poor"]


class ListingIn(BaseModel):
    waste_type: WasteType
    sub_type: str | None = Field(default=None, max_length=60)
    quantity_kg: int = Field(gt=0, le=100_000_000)
    quality: Quality = "fair"
    location_name: str = Field(min_length=1, max_length=120)
    location_lat: float = Field(ge=-90, le=90)
    location_lon: float = Field(ge=-180, le=180)
    seller_name: str = Field(min_length=1, max_length=120)
    seller_contact: str | None = Field(default=None, max_length=40)
    price_per_kg: int | None = Field(default=None, ge=0, le=1_000_000)
    description: str = Field(default="", max_length=2000)


class Listing(ListingIn):
    waste_id: int
    is_synthetic: bool
    source: str


class MatchRequest(BaseModel):
    waste_id: int | None = Field(default=None, description="match an existing listing, or give the fields below")
    waste_type: WasteType | None = None
    sub_type: str | None = Field(default=None, max_length=60)
    quantity_kg: float | None = Field(default=None, gt=0, le=100_000_000)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    top_k: int = Field(default=5, ge=1, le=50)


class RecyclerMatch(BaseModel):
    recycler_id: int
    name: str
    waste_types: str
    lat: float
    lon: float
    distance_km: float
    capacity_kg_per_month: int
    can_absorb_in_one_month: bool
    score: float
    score_breakdown: dict[str, float]
    co2e_saved_kg: float | None = Field(description="None when no carbon factor exists for this material")
    carbon_is_proxy: bool | None
    is_sample_profile: bool


class MatchResponse(BaseModel):
    waste_type: WasteType
    quantity_kg: float
    co2e_saved_kg: float | None
    carbon_is_proxy: bool | None
    carbon_factor_available: bool = True
    carbon_note: str
    matches: list[RecyclerMatch]
    message: str | None = Field(default=None, description="set when no recycler profile accepts this material")


class FeedstockQuery(BaseModel):
    query: str = Field(default="", max_length=500,
                       description="free-text feedstock need; empty = browse (ranked by distance and quality)")
    waste_type: WasteType | None = None
    sub_type: str | None = Field(default=None, max_length=60, description="listing sub-type, e.g. cotton_scrap")
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    min_quantity_kg: float = Field(default=0, ge=0, le=100_000_000)
    quality: list[Quality] | None = None
    top_k: int = Field(default=10, ge=1, le=500)


class ClassifyResponse(BaseModel):
    """label/confidence/top_k keep their v1 meaning (waste type); the rest is v2 (sub-types, unknown)."""
    label: str | None = Field(description="waste type, or 'unknown' when the photo is rejected")
    confidence: float = Field(description="waste-type confidence (same as type_confidence)")
    is_confident: bool
    top_k: list[dict] = Field(description="waste-type level: [{label, confidence}]")
    model_task: str
    waste_type: str | None = None
    sub_type: str | None = None
    type_confidence: float | None = None
    subtype_confidence: float | None = Field(default=None, description="P(sub_type | waste_type)")
    leaf_confidence: float | None = None
    subtype_is_confident: bool | None = None
    top_k_subtypes: list[dict] | None = Field(default=None, description="[{label, waste_type, sub_type, confidence}]")
    status: Literal["detected", "confirm", "unknown"] | None = Field(
        default=None, description="detected | confirm (best guess, please confirm) | unknown")
    confirm_reason: str | None = None
    memory: dict | None = Field(default=None, description="set when confirmed user feedback adjusted this result")
    is_unknown: bool = False
    unknown_reason: str | None = None
    best_guess: dict | None = Field(default=None, description="the model's best guess, also when rejected as unknown")
    ood: dict | None = None
    model_version: str | None = None


class GroupPrediction(ClassifyResponse):
    n_images: int | None = None
    agreement: int | None = Field(default=None, description="photos whose own top label equals the combined label")
    per_image: list[dict] | None = None


class MultiClassifyResponse(BaseModel):
    mode: Literal["same_item", "different_items"]
    results: list[GroupPrediction]
