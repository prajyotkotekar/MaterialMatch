"""
frontend/app.py - MaterialMatch Streamlit entry point.

    streamlit run frontend/app.py         # from the project root (dashboard.py also launches this)
"""

import html
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib  # noqa: E402

import streamlit as st  # noqa: E402


def _reload_changed(names: tuple[str, ...]) -> None:
    """Streamlit reruns page scripts after an edit but keeps imported modules cached, so a running server
    would mix new pages with old helpers (ImportError on every page). Reload ours when their file changed."""
    stale = False  # once one module reloads, later ones (which import from it) reload too
    for name in names:
        mod = sys.modules.get(name)
        if mod is None or not getattr(mod, "__file__", None):
            continue
        mtime = Path(mod.__file__).stat().st_mtime
        if stale or getattr(mod, "_mm_mtime", mtime) != mtime:
            mod, stale = importlib.reload(mod), True
        mod._mm_mtime = mtime


_reload_changed(("common", "visuals"))  # order matters: visuals imports from common

from backend import feedback_store  # noqa: E402
from common import inject_css  # noqa: E402
from ml.carbon import DISCLAIMER  # noqa: E402

st.set_page_config(page_title="MaterialMatch", page_icon=str(HERE / "assets" / "mark.svg"), layout="wide")
inject_css()
st.logo(str(HERE / "assets" / "logo.svg"), size="large")


@st.cache_resource(show_spinner="Restoring saved feedback...")
def _restore_feedback() -> int:
    """Once per server process, before any prediction reads the feedback memory (see backend/feedback_sync.py)."""
    return feedback_store.restore()


_restore_feedback()

if "my_listings" not in st.session_state:
    st.session_state.my_listings = []
# Streamlit drops widget state for widgets not rendered this run; re-assigning keeps form values across pages.
for _k in list(st.session_state.keys()):
    if (_k in ("wt", "sub", "qty", "loc", "photo_mode", "feed_query", "feed_wt", "feed_sub", "feed_minq", "feed_qual", "feed_loc")
            or _k.startswith(("wt_", "sub_", "qty_"))):
        st.session_state[_k] = st.session_state[_k]

# Not called "pages": a folder with that name next to the entry script switches Streamlit to its legacy
# multipage mode, which shows "Page not found" when a page URL is opened directly.
PAGES = HERE / "app_pages"
# Pages link to each other through these objects (st.session_state.nav): a "pages/..." path would be relative
# to the entrypoint, which differs between `streamlit run frontend/app.py` and `streamlit run dashboard.py`.
st.session_state.nav = {
    "waste": st.Page(PAGES / "1_List_Waste.py", title="I have waste", url_path="waste", default=True),
    "feedstock": st.Page(PAGES / "2_Find_Matches.py", title="I need feedstock", url_path="feedstock"),
    # the carbon factors are a tab on the Impact page
    "impact": st.Page(PAGES / "3_Impact_Dashboard.py", title="Impact", url_path="impact"),
}
page = st.navigation(list(st.session_state.nav.values()), position="top")

page.run()

st.markdown(f'<p class="mm-foot">{html.escape(DISCLAIMER)}</p>', unsafe_allow_html=True)
_reload_changed(("common", "visuals"))  # records the file times of modules first imported on this run
