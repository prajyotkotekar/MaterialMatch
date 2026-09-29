"""Compatibility wrapper - the implementation now lives in ml/matcher.py and ml/embeddings.py."""

from ml.embeddings import load_listings  # noqa: F401
from ml.matcher import *  # noqa: F401,F403
from ml.matcher import (embedding_backend, find_listings, load_recyclers,  # noqa: F401
                        match_listing, match_recyclers)

if __name__ == "__main__":
    import runpy

    runpy.run_module("ml.matcher", run_name="__main__")
