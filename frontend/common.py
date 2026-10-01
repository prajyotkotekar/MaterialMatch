"""Shared data helpers and UI building blocks for the Streamlit pages."""

from __future__ import annotations

import html
import math
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from ml import taxonomy
from ml.carbon import co2e_saved
from ml.embeddings import file_version, load_listings
from ml.matcher import DISTANCE_SCALE_KM, RECYCLER_WEIGHTS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WASTE_TYPES = ["textile", "plastic", "construction", "e_waste"]  # marketplace types (listings, recyclers, factors)
ALL_WASTE_TYPES = list(taxonomy.WASTE_TYPES)  # + types the photo model recognises (paper, glass, ...)
WASTE_LABELS = taxonomy.WASTE_LABELS
# One earth-toned colour per material, used for tags, card edges, the result card, charts and map pins
# (and nothing else). Same order as chartCategoricalColors in .streamlit/config.toml.
WASTE_COLOR = {"construction": "#B48F6E", "e_waste": "#8BA8D4", "plastic": "#E3965F", "textile": "#C29AD4",
               "paper": "#D6BC78", "glass": "#79C3BA", "metal": "#A9B1B6", "biological": "#A9BE6E", "trash": "#D88580"}
# Design tokens, exposed to style.css as --mm-<name>. Lime is the single accent: live/AI states and main actions.
TOKENS = {"ground": "#121512", "panel": "#191D19", "raised": "#20251F", "line": "#2B312A", "line-strong": "#3E463C",
          "text": "#E9ECE4", "muted": "#99A296", "faint": "#6E776B", "bone": "#ECEBE3", "ink": "#121512",
          "lime": "#BEF04A", "amber": "#F0A43A", "red": "#E8776E"}
SESSION_ID_START = 10_000
CO2E_EXPLAINER = "Estimated impact from diverting this material to reuse/recycling."
RAW_DIR = ROOT / "data" / "raw"
PROFILES_CSV = RAW_DIR / "recycler_profiles.csv"


# ---------- formatting ----------

waste_label = taxonomy.waste_label
sub_label = taxonomy.sub_label


def material_color(t: str | None) -> str:
    return WASTE_COLOR.get(t or "", "#6E776B")


def type_tag(t: str) -> str:
    """Material tag: colour swatch + name (HTML, render with unsafe_allow_html=True)."""
    return f'<span class="mm-tag" style="--c:{material_color(t)}"><i></i>{html.escape(waste_label(t))}</span>'


def quality_tag(q: str) -> str:
    q = str(q).lower()
    return (f'<span class="mm-q mm-q-{html.escape(q)}" title="{html.escape(q.capitalize())} quality">'
            f'<b><i></i><i></i><i></i></b>{html.escape(q.capitalize())}</span>')


def carbon_label(is_proxy: bool) -> str:
    return "Proxy estimate" if is_proxy else "Sourced factor"


def fmt_kg(kg: float) -> str:
    return f"{kg / 1000:,.1f} t" if kg >= 10_000 else f"{kg:,.0f} kg"


def tip(content: str, tooltip: str, below: bool = False) -> str:
    """Inline HTML with a hover/focus tooltip (render with unsafe_allow_html=True)."""
    cls = "mm-tip mm-tip-below" if below else "mm-tip"
    return (f'<span class="{cls}" tabindex="0" data-tip="{html.escape(tooltip, quote=True)}" '
            f'aria-label="{html.escape(tooltip, quote=True)}">{content}</span>')


def icon(name: str) -> str:
    return f'<span class="mm-ico" aria-hidden="true">{name}</span>'


# ---------- styling ----------

def inject_css() -> None:
    tokens = ";".join(f"--mm-{k}:{v}" for k, v in TOKENS.items())
    # result cards get their material colour as the left edge (card keys end in "__<waste_type>")
    edges = "".join(f'[class*="st-key-card_"][class*="__{t}"]{{--mm-edge:{c}}}' for t, c in WASTE_COLOR.items())
    st.html(f"<style>:root{{{tokens}}}{edges}{(HERE / 'style.css').read_text(encoding='utf-8')}</style>")


def page_header(title: str, lede: str) -> None:
    st.markdown(f"# {title}")
    st.markdown(f'<p class="mm-lede">{html.escape(lede)}</p>', unsafe_allow_html=True)


def step_title(n: int, text: str, note: str = "") -> None:
    """Numbered step heading (only for pages that are a real sequence)."""
    extra = f'<span class="mm-step-note">{html.escape(note)}</span>' if note else ""
    st.markdown(f'<div class="mm-step"><span class="mm-step-n">{n:02d}</span><span>{html.escape(text)}</span>{extra}</div>',
                unsafe_allow_html=True)


def log_classification(pred: dict, n_photos: int, source: str) -> None:
    """Session history of new predictions (for the dashboard and Present page). Never written to disk."""
    log = st.session_state.setdefault("class_log", [])
    log.append({"time": pd.Timestamp.now(), "waste_type": pred.get("label"), "sub_type": pred.get("sub_type"),
                "confidence": float(pred.get("confidence") or 0), "status": pred.get("status") or
                ("unknown" if pred.get("is_unknown") else "detected"), "photos": n_photos, "source": source})


# ---------- data ----------

@st.cache_resource(show_spinner="Loading the material classifier…")
def get_classifier():
    try:
        from ml.classifier.predict import load_model, predict
        load_model()
        return predict, None
    except Exception as exc:  # weights missing or ultralytics unavailable
        return None, str(exc)


@st.cache_data(max_entries=64, show_spinner=False)  # the page shows its own scan state
def classify_bytes(data: bytes, memory_version: str = "") -> dict | None:
    """Prediction for one photo, with the probability of every class (needed to combine photos).

    memory_version (feedback_memory.version()) is only part of the cache key: new feedback must
    not be hidden behind a cached prediction."""
    predict, _ = get_classifier()
    return predict(data, top_k=99) if predict else None


def data_version() -> tuple[int, ...]:
    """Changes whenever a CSV in data/raw is edited: the cached tables below are keyed on it, so the
    pages always show the current data without restarting the server."""
    return tuple(file_version(p) for p in sorted(RAW_DIR.glob("*.csv")))


def zones() -> pd.DataFrame:
    return _zones(data_version())


@st.cache_data(max_entries=4)
def _zones(_version: tuple) -> pd.DataFrame:
    return (load_listings().groupby("location_name")[["location_lat", "location_lon"]]
            .mean().sort_index())


def zone_coords(name: str) -> tuple[float, float]:
    z = zones()
    return float(z.loc[name, "location_lat"]), float(z.loc[name, "location_lon"])


def recycler_profiles() -> dict[int, dict]:
    return _recycler_profiles(data_version())


@st.cache_data(max_entries=4)
def _recycler_profiles(_version: tuple) -> dict[int, dict]:
    df = pd.read_csv(PROFILES_CSV)
    return {int(r["recycler_id"]): r for r in df.to_dict(orient="records")}


def synthetic_listings_with_carbon() -> pd.DataFrame:
    return _synthetic_listings_with_carbon(data_version())


@st.cache_data(max_entries=4)
def _synthetic_listings_with_carbon(_version: tuple) -> pd.DataFrame:
    df = load_listings().copy()
    carbon = [co2e_saved(r.sub_type, r.quantity_kg, waste_type=r.waste_type) for r in df.itertuples()]
    df["co2e_saved_kg"] = [c["co2e_saved_kg"] for c in carbon]
    df["carbon_is_proxy"] = [bool(c["is_proxy"]) for c in carbon]  # None (no factor, e.g. trash) -> False
    df["source"] = "Demo"
    return df


def all_listings() -> pd.DataFrame:
    """Demo listings + listings published in this session."""
    base = synthetic_listings_with_carbon()
    mine = pd.DataFrame(st.session_state.get("my_listings", []))
    if mine.empty:
        return base
    # materials without a carbon factor are stored as None: keep them out of CO2e totals, not 0-valued
    mine["co2e_saved_kg"] = pd.to_numeric(mine["co2e_saved_kg"], errors="coerce")
    mine["carbon_is_proxy"] = mine["carbon_is_proxy"].fillna(False).astype(bool)
    return pd.concat([mine, base], ignore_index=True)


# ---------- components ----------

def section_title(text: str) -> None:
    st.markdown(f'<div class="mm-section">{html.escape(text)}</div>', unsafe_allow_html=True)


def kpi(key: str, label: str, value: str, note: str | None = None, help: str | None = None) -> None:
    with st.container(border=True, key=f"card_{key}", gap="xxsmall", height="stretch"):
        st.metric(label, value, help=help)
        if note:
            st.caption(note)


def cost_by_quality_kpi(top_by_quality: dict, need_kg: float, help: str | None = None) -> None:
    """KPI card: what `need_kg` would cost per quality grade, from the best-ranked listing of each.
    top_by_quality maps 'good'/'fair'/'poor' to a listing dict (or None when there is none)."""
    with st.container(border=True, key="card_cost", gap="xsmall", height="stretch"):
        st.markdown(":small[Cost for your quantity]", help=help)
        # one horizontal row per grade: grade on the left, total + ₹/kg on the right
        for q in ("good", "fair", "poor"):
            r = top_by_quality.get(q)
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(quality_tag(q), unsafe_allow_html=True, width="content")
                st.space("stretch")
                if r is None:
                    st.markdown(":gray[no listing]", width="content")
                elif not r["price_per_kg"]:
                    st.markdown(":gray[price not set]", width="content")
                elif need_kg > 0:
                    st.markdown(f"**₹{need_kg * r['price_per_kg']:,.0f}** &nbsp;:gray[₹{r['price_per_kg']:,}/kg]",
                                width="content")
                else:
                    st.markdown(f"**₹{r['price_per_kg']:,}** &nbsp;:gray[per kg]", width="content")


def recycler_logo(name: str, waste_types: str) -> str:
    """Monogram tile (no real logos: these are sample profiles)."""
    initials = "".join(w[0] for w in name.replace("&", " ").split() if w[0].isalpha())[:2].upper()
    color = material_color(waste_types.split(",")[0].strip())
    return tip(f'<span class="mm-logo" style="--c:{color}">{html.escape(initials)}</span>',
               "Sample recycler profile, not a verified partner")


def meta_row(*items: str) -> str:
    """Inline facts separated by space. Items are escaped HTML or Streamlit markdown (a <span>, not a <div>,
    so badges and :material/ icons inside are still rendered)."""
    return '<span class="mm-meta">' + "".join(f"<span>{i}</span>" for i in items if i) + "</span>"


def absorb_info(m: dict, quantity_kg: float) -> tuple[str, str]:
    """Badge + tooltip, computed from the listed monthly capacity and the entered quantity."""
    cap = m["capacity_kg_per_month"]
    if m["can_absorb_in_one_month"]:
        return (":green-badge[:material/check: Can absorb this batch]",
                f"Yes, this recycler's listed monthly processing capacity ({cap:,} kg/month) is at least "
                f"the quantity you entered ({quantity_kg:,.0f} kg), so the whole batch fits within one "
                "month of capacity. Based on the sample profile's listed capacity; current spare "
                "capacity isn't known.")
    months = math.ceil(quantity_kg / cap)
    return (f":orange-badge[:material/schedule: ~{months} months of capacity]",
            f"Your {quantity_kg:,.0f} kg is more than this recycler's listed monthly capacity "
            f"({cap:,} kg/month), so processing it would take about {months} months of their "
            "capacity, or split the batch across recyclers.")


def recycler_card(m: dict, rank: int, quantity_kg: float, waste_type: str) -> None:
    prof = recycler_profiles().get(m["recycler_id"], {})
    edge = m["waste_types"].split(",")[0].strip()
    key = f"card_top__{edge}" if rank == 0 else f"card_rec_{rank}__{edge}"
    with st.container(border=True, key=key, gap="small"):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(recycler_logo(m["name"], m["waste_types"]), unsafe_allow_html=True, width="content")
            with st.container(gap="xxsmall"):
                tag = '<span class="mm-flag">Best match</span>' if rank == 0 else f'<span class="mm-rank">{rank + 1}.</span>'
                st.markdown(f"{tag}<strong>{html.escape(m['name'])}</strong>", unsafe_allow_html=True)
                accepts = "".join(type_tag(t.strip()) for t in m["waste_types"].split(","))
                st.markdown(f"{accepts}<span class='mm-area'>{html.escape(prof.get('area', ''))}</span>",
                            unsafe_allow_html=True)
            st.markdown(
                tip(f'<span class="mm-score">{m["score"]:.0%}</span><span class="mm-score-label">match</span>',
                    "Overall score: distance (50%), capacity fit (30%) and specialisation (20%). "
                    "Open details for the breakdown.",
                    below=True),
                unsafe_allow_html=True, width="content")
        st.progress(min(max(m["score"], 0.0), 1.0))
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            badge, why = absorb_info(m, quantity_kg)
            st.markdown(meta_row(
                tip(f"{icon('near_me')} {m['distance_km']:.1f} km",
                    "Straight-line distance between your location and this recycler."),
                tip(f"{icon('factory')} {m['capacity_kg_per_month']:,} kg/month",
                    "Monthly processing capacity listed in this sample profile."),
                badge + " " + tip(icon("info"), why)),
                unsafe_allow_html=True, width="content")
            st.space("stretch")
            if st.button("Details & contact", key=f"details_{rank}_{m['recycler_id']}", icon=":material/contact_page:",
                         type="tertiary"):
                recycler_dialog(m, quantity_kg, waste_type)


@st.dialog("Recycler details", width="medium")
def recycler_dialog(m: dict, quantity_kg: float, waste_type: str) -> None:
    prof = recycler_profiles().get(m["recycler_id"], {})
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(recycler_logo(m["name"], m["waste_types"]), unsafe_allow_html=True, width="content")
        with st.container(gap="xxsmall"):
            st.markdown(f'<span class="mm-dlg-name">{html.escape(m["name"])}</span>', unsafe_allow_html=True)
            st.markdown(":gray-badge[Sample profile, not a verified partner]")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f":material/location_on: **Address**  \n{prof.get('address', 'Not listed')}")
        st.markdown(f":material/call: **Phone**  \n{prof.get('phone', 'Not listed')}")
        st.markdown(f":material/mail: **Email**  \n{prof.get('email', 'Not listed')}")
        if prof.get("website"):
            st.markdown(f":material/language: **Website**  \n"
                        f"[{prof['website'].removeprefix('https://')}]({prof['website']})")
    with c2:
        accepts = "".join(type_tag(t.strip()) for t in m["waste_types"].split(","))
        st.markdown(f":material/category: **Accepts**  \n{accepts}", unsafe_allow_html=True)
        st.markdown(f":material/factory: **Monthly capacity**  \n{m['capacity_kg_per_month']:,} kg/month")
        st.markdown(f":material/near_me: **Distance**  \n{m['distance_km']:.1f} km from your location")
        st.markdown(f":material/insights: **Match score**  \n{m['score']:.0%}")

    with st.container(border=True, gap="small"):
        st.markdown("**Why this match?**")
        b = m["score_breakdown"]
        w = RECYCLER_WEIGHTS
        n_types = len(m["waste_types"].split(","))
        st.markdown(f":green[:material/check_circle:] **Material**: accepts {waste_label(waste_type).lower()}. "
                    "Only recyclers that list your waste type are ranked.")
        rows = [
            ("Proximity", "distance",
             f"{m['distance_km']:.1f} km away. The score halves roughly every "
             f"{DISTANCE_SCALE_KM * math.log(2):.0f} km."),
            ("Capacity fit", "capacity",
             f"{m['capacity_kg_per_month']:,} kg/month for your {quantity_kg:,.0f} kg "
             "(capacity ÷ quantity, capped at 100%)."),
            ("Specialisation", "specialist",
             f"Accepts {n_types} waste type{'s' if n_types > 1 else ''}; specialists score higher (1 ÷ types)."),
        ]
        for label, k, text in rows:
            st.progress(b[k], text=f"{label}: {b[k]:.0%}, weight {w[k]:.0%}")
            st.caption(text)
        st.caption("Score = " + " + ".join(f"{w[k]:.0%} × {b[k]:.0%}" for _, k, _ in rows)
                   + f" = **{m['score']:.0%}**. Sub-type and quality don't affect ranking; the sub-type is "
                   "used for the CO₂e estimate.")

    with st.container(horizontal=True, gap="small"):
        if prof.get("email"):
            subject = f"MaterialMatch enquiry: {quantity_kg:,.0f} kg of {waste_label(waste_type).lower()}"
            st.link_button("Email", f"mailto:{prof['email']}?subject={quote(subject)}",
                           icon=":material/mail:", type="primary")
        if prof.get("phone"):
            st.link_button("Call", "tel:" + prof["phone"].replace(" ", ""), icon=":material/call:")
    st.caption("Demo contact details (reserved .example domains, placeholder numbers). "
               "Replace with verified partner data before real use.")


def listing_carbon_text(r: dict, need_kg: float = 0) -> str:
    """CO2e for what the buyer needs (not the seller's whole batch); per-kg factor when no amount is given."""
    per_kg = r.get("co2e_per_kg")
    if per_kg is None:
        return ":material/eco: no carbon factor for this material"
    label = carbon_label(r["carbon_is_proxy"]).lower()
    if need_kg > 0:
        return f":material/eco: {need_kg * per_kg:,.1f} kg CO₂e saved on your {need_kg:,.0f} kg ({label})"
    return f":material/eco: {per_kg:.2f} kg CO₂e saved per kg ({label})"


def seller_display(name: str) -> str:
    """'Factory_173' -> 'Factory 173' (the synthetic seller names come from ids)."""
    return " ".join(str(name).replace("_", " ").split()) or "Seller"


def listing_email(r: dict) -> str:
    """Placeholder e-mail for DEMO sellers (reserved .example domain). Sellers who published a listing
    themselves get none: we only show what they entered."""
    if not r.get("is_synthetic", True):
        return ""
    slug = "".join(c if c.isalnum() else "." for c in seller_display(r["seller_name"]).lower()).strip(".")
    return f"{slug}@demo-seller.example"


def price_text(r: dict, need_kg: float = 0) -> str:
    if not r["price_per_kg"]:
        return "price not set"
    text = f"₹{r['price_per_kg']:,}/kg"
    if need_kg > 0:
        text += f" (≈ ₹{need_kg * r['price_per_kg']:,.0f} for your {need_kg:,.0f} kg)"
    return text


def source_badge(r: dict) -> str:
    return ('<span class="mm-src mm-src-mine">Your listing</span>' if r.get("source") == "Your listing"
            else '<span class="mm-src">Demo listing</span>')


def listing_card(r: dict, rank: int, need_kg: float = 0) -> None:
    key = f"card_top_l__{r['waste_type']}" if rank == 0 else f"card_lst_{rank}__{r['waste_type']}"
    has_query = r.get("has_query", True)
    with st.container(border=True, key=key, gap="xsmall"):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            with st.container(gap="xxsmall"):
                tag = f'<span class="mm-flag">{"Best match" if has_query else "Top result"}</span>' if rank == 0 else ""
                st.markdown(f"{tag}<strong>{html.escape(sub_label(r['sub_type']))}</strong>"
                            f"<span class='mm-area'>{r['quantity_kg']:,} kg available</span>", unsafe_allow_html=True)
                st.markdown(type_tag(r["waste_type"]) + quality_tag(r["quality"]) + " " + source_badge(r),
                            unsafe_allow_html=True)
            st.markdown(
                tip(f'<span class="mm-score">{r["score"]:.0%}</span>'
                    f'<span class="mm-score-label">{"match" if has_query else "score"}</span>',
                    "Text match, distance and quality combined. Open details for the breakdown." if has_query else
                    "No search text entered: ranked by distance and quality only.", below=True),
                unsafe_allow_html=True, width="content")
        dist = f", {r['distance_km']:.1f} km" if r.get("distance_km") is not None else ""
        carbon = listing_carbon_text(r, need_kg).replace(":material/eco: ", "", 1)
        st.markdown(meta_row(f"{icon('location_on')} {html.escape(r['location_name'])}{dist}",
                             f"{icon('sell')} {html.escape(price_text(r, need_kg))}",
                             f'<span class="mm-carbon">{icon("eco")} {html.escape(carbon)}</span>'),
                    unsafe_allow_html=True)
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown(meta_row(f"{icon('storefront')} {html.escape(seller_display(r['seller_name']))}",
                                 f"{icon('call')} {html.escape(r['seller_contact'])}" if r.get("seller_contact") else ""),
                        unsafe_allow_html=True, width="content")
            st.space("stretch")
            if st.button("Details & contact", key=f"contact_{rank}_{r['waste_id']}", icon=":material/contact_page:",
                         type="tertiary"):
                listing_dialog(r, need_kg)


@st.dialog("Listing details", width="medium")
def listing_dialog(r: dict, need_kg: float = 0) -> None:
    seller = seller_display(r["seller_name"])
    email = listing_email(r)
    with st.container(gap="xxsmall"):
        st.markdown(f'<span class="mm-dlg-name">{html.escape(sub_label(r["sub_type"]))}, {r["quantity_kg"]:,} kg</span>',
                    unsafe_allow_html=True)
        st.markdown(type_tag(r["waste_type"]) + quality_tag(r["quality"]) + " " + source_badge(r),
                    unsafe_allow_html=True)
    if r.get("description"):
        st.write(r["description"])

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f":material/storefront: **Seller**  \n{seller}")
        st.markdown(f":material/call: **Phone**  \n{r.get('seller_contact') or 'Not listed'}")
        if email:
            st.markdown(f":material/mail: **Email**  \n{email}")
        pickup = r.get("pickup_location") or r["location_name"]
        address = f"  \n{r['pickup_address']}" if r.get("pickup_address") else ""
        st.markdown(f":material/location_on: **Pickup / drop-off**  \n{pickup}{address}")
    with c2:
        st.markdown(f":material/scale: **Available**  \n{r['quantity_kg']:,} kg")
        st.markdown(f":material/sell: **Price**  \n{price_text(r, need_kg)}")
        if r.get("distance_km") is not None:
            st.markdown(f":material/near_me: **Distance**  \n{r['distance_km']:.1f} km from your location "
                        f"({r['location_name']})")
        else:
            st.markdown(f":material/near_me: **Location**  \n{r['location_name']}")
        st.markdown(":material/eco: **CO₂e**  \n" + listing_carbon_text(r, need_kg).replace(":material/eco: ", "", 1))

    with st.container(border=True, gap="small"):
        has_query = r.get("has_query", True)
        st.markdown(f"**Why this {'match' if has_query else 'result'}?**")
        b, w = r["score_breakdown"], r["score_weights"]
        rows = []
        if "similarity" in b:
            rows.append(("Text match", "similarity",
                         "How close the listing text is to what you typed (relative to the other listings)."))
        rows += [
            ("Proximity", "distance",
             (f"{r['distance_km']:.1f} km away. The score halves roughly every {DISTANCE_SCALE_KM * math.log(2):.0f} km."
              if r.get("distance_km") is not None else "No location chosen, so every listing scores the same.")),
            ("Quality", "quality", f"{r['quality'].capitalize()} quality (good 100%, fair 70%, poor 40%)."),
        ]
        for label, k, text in rows:
            st.progress(min(max(b[k], 0.0), 1.0), text=f"{label}: {b[k]:.0%}, weight {w[k]:.0%}")
            st.caption(text)
        st.caption("Score = " + " + ".join(f"{w[k]:.0%} × {b[k]:.0%}" for _, k, _ in rows) + f" = **{r['score']:.0%}**."
                   + ("" if has_query else " No search text was entered, so text match is not part of the score."))

    with st.container(horizontal=True, gap="small"):
        if email:
            subject = f"MaterialMatch enquiry: {sub_label(r['sub_type'])}"
            st.link_button("Email", f"mailto:{email}?subject={quote(subject)}", icon=":material/mail:", type="primary")
        if r.get("seller_contact"):
            st.link_button("Call", "tel:" + "".join(c for c in r["seller_contact"] if c.isdigit() or c == "+"),
                           icon=":material/call:", type="secondary" if email else "primary")
    st.caption("Demo listing: the seller, phone and email are placeholders, not a real business."
               if r.get("is_synthetic", True) else
               "Published by you in this session; the contact details are the ones you entered.")


def results_map(origin: tuple[float, float], points: list[dict], height: int = 380,
                origin_label: str = "Your waste", points_label: str = "Recyclers") -> None:
    """Map with the origin (yellow square-ish pin) and ranked points in their material colour, the best one larger.
    points: {lat, lon, name, detail, waste_type}."""
    import pydeck as pdk

    def rgb(hex_: str, a: int) -> list[int]:
        return [int(hex_[i:i + 2], 16) for i in (1, 3, 5)] + [a]

    rows = [{"lat": origin[0], "lon": origin[1], "name": origin_label, "detail": "", "fill": rgb("#F2C200", 255),
             "line": rgb("#1C2529", 255), "r": 300}]
    rows += [{**p, "fill": rgb(material_color(p.get("waste_type")), 240 if i == 0 else 190),
              "line": [255, 255, 255, 230], "r": 320 if i == 0 else 200}
             for i, p in enumerate(points)]
    lats = [r["lat"] for r in rows]
    lons = [r["lon"] for r in rows]
    span = max(max(lats) - min(lats), max(lons) - min(lons), 0.01)
    zoom = min(13.0, math.floor(math.log2(360 * height * 0.6 / (512 * span)) * 2) / 2)  # deck.gl: 512px tiles
    layer = pdk.Layer(
        "ScatterplotLayer", pd.DataFrame(rows), get_position="[lon, lat]", get_fill_color="fill",
        get_radius="r", radius_min_pixels=5, radius_max_pixels=14, stroked=True,
        get_line_color="line", line_width_min_pixels=1.5, pickable=True,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=(max(lats) + min(lats)) / 2,
                                         longitude=(max(lons) + min(lons)) / 2, zoom=zoom),
        tooltip={"html": "<b>{name}</b><br/>{detail}",
                 "style": {"fontSize": "12px", "borderRadius": "3px", "padding": "6px 8px"}},
        map_style=None,
    )
    st.pydeck_chart(deck, height=height)
    st.markdown(meta_row('<span class="mm-pin" style="--c:#F2C200"></span>' + html.escape(origin_label),
                         '<span class="mm-pin" style="--c:var(--mm-muted)"></span>'
                         f"{html.escape(points_label)}, largest is the best match"),
                unsafe_allow_html=True)
