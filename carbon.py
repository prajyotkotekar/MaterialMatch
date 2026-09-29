"""Compatibility wrapper - the implementation now lives in ml/carbon.py."""

from ml.carbon import *  # noqa: F401,F403
from ml.carbon import DISCLAIMER, co2e_saved, get_factor, load_factors  # noqa: F401

if __name__ == "__main__":
    import runpy

    runpy.run_module("ml.carbon", run_name="__main__")
