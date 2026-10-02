"""
ml/embeddings.py - Text embeddings for waste listings (sentence-transformers).

    python -m ml.embeddings      # precompute -> data/processed/listings_with_embeddings.npz

At runtime the cache is used if it exists and matches the current listings; otherwise
embeddings are computed in memory. If the model can't be loaded (e.g. offline), a TF-IDF
backend is used instead so search still works.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LISTINGS_CSV = ROOT / "data" / "raw" / "waste_listings.csv"
CACHE_NPZ = ROOT / "data" / "processed" / "listings_with_embeddings.npz"  # plain arrays, never pickle
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


ABBREVIATIONS = {
    "PP": "PP polypropylene",
    "PET": "PET polyethylene terephthalate",
    "HDPE": "HDPE high-density polyethylene",
    "LDPE": "LDPE low-density polyethylene",
    "PCB": "PCB printed circuit board",
}


def expand_abbreviations(text: str) -> str:
    return " ".join(ABBREVIATIONS.get(w, w) for w in text.split())


def listing_text(r) -> str:
    return expand_abbreviations(
        f"{r['sub_type'].replace('_', ' ')} {r['waste_type'].replace('_', ' ')} waste, "
        f"{r['quality']} quality, {r['location_name']}. {r['description']}")


def file_version(path: Path) -> int:
    """Modification time of a data file; used as a cache key so an edited CSV is re-read without a restart."""
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return 0


def load_listings() -> pd.DataFrame:
    return _read_listings(file_version(LISTINGS_CSV))


@lru_cache(maxsize=1)
def _read_listings(_version: int) -> pd.DataFrame:
    df = pd.read_csv(LISTINGS_CSV)
    df["is_synthetic"] = True
    return df


def _texts_hash(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


@lru_cache(maxsize=1)
def sentence_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    return sentence_model().encode(texts, normalize_embeddings=True)


class ListingIndex:
    """Similarity search over listing texts."""

    def __init__(self, texts: list[str], use_cache: bool = True):
        self.texts = texts
        self.backend = "tfidf"
        try:
            self.matrix = self._cached(texts) if use_cache else None
            if self.matrix is None:
                self.matrix = embed(texts)
            self.backend = "sentence-transformers"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self.matrix = self.tfidf.fit_transform(texts)

    @staticmethod
    def _cached(texts: list[str]):
        if not CACHE_NPZ.exists():
            return None
        with np.load(CACHE_NPZ, allow_pickle=False) as z:
            if str(z["model"]) == EMBED_MODEL and str(z["texts_hash"]) == _texts_hash(texts):
                return z["embeddings"]
        return None

    def similarity(self, query: str) -> np.ndarray:
        query = expand_abbreviations(query)
        if self.backend == "sentence-transformers":
            return (self.matrix @ embed([query]).T).ravel()
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(self.tfidf.transform([query]), self.matrix).ravel()

    def similarity_texts(self, texts: list[str], query: str) -> np.ndarray:
        """Similarity of `query` to texts that are NOT in the index (e.g. listings published this session)."""
        if not texts:
            return np.zeros(0)
        query = expand_abbreviations(query)
        if self.backend == "sentence-transformers":
            return (embed(texts) @ embed([query]).T).ravel()
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(self.tfidf.transform([query]), self.tfidf.transform(texts)).ravel()


def listing_index() -> ListingIndex:
    return _listing_index(file_version(LISTINGS_CSV))


@lru_cache(maxsize=1)
def _listing_index(_version: int) -> ListingIndex:
    return ListingIndex([listing_text(r) for _, r in load_listings().iterrows()])


def build_cache() -> Path:
    df = load_listings()
    texts = [listing_text(r) for _, r in df.iterrows()]
    CACHE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_NPZ, model=np.array(EMBED_MODEL), texts_hash=np.array(_texts_hash(texts)),
             waste_ids=df["waste_id"].to_numpy(), texts=np.array(texts), embeddings=embed(texts))
    return CACHE_NPZ


if __name__ == "__main__":
    print(f"Wrote {build_cache()}")
