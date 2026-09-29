"""
backend/db.py - Data access. Reads the CSVs in data/raw (read-only).

Listings created through POST /listings are kept in memory for the life of the process;
the CSV files are never modified. Swap this module for SQLite later without touching routes.
"""

from __future__ import annotations

import threading

import pandas as pd

from ml.carbon import load_factors
from ml.embeddings import load_listings
from ml.matcher import load_recyclers

_lock = threading.Lock()
_user_listings: list[dict] = []


def _csv_listings() -> list[dict]:
    df = load_listings().copy()
    df["source"] = "synthetic_demo"
    return df.to_dict(orient="records")


def all_listings() -> list[dict]:
    with _lock:
        return _csv_listings() + list(_user_listings)


def get_listing(waste_id: int) -> dict | None:
    return next((r for r in all_listings() if r["waste_id"] == waste_id), None)


def add_listing(data: dict) -> dict:
    with _lock:
        next_id = max([int(load_listings()["waste_id"].max())]
                      + [r["waste_id"] for r in _user_listings]) + 1
        row = {"waste_id": next_id, **data, "is_synthetic": False, "source": "user_submitted"}
        _user_listings.append(row)
        return row


def recyclers() -> pd.DataFrame:
    return load_recyclers()


def carbon_factors() -> pd.DataFrame:
    return load_factors()
