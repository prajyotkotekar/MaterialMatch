"""
dashboard.py - launches the MaterialMatch app in frontend/.

    streamlit run dashboard.py        # same as: streamlit run frontend/app.py
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "frontend" / "app.py"), run_name="__main__")
