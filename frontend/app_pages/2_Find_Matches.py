import html

import streamlit as st

import visuals as vz
from common import (ALL_WASTE_TYPES, all_listings, cost_by_quality_kpi, fmt_kg, kpi, listing_card, listing_carbon_text,
                    listing_dialog, price_text, results_map, section_title, seller_display, step_title, sub_label,
                    waste_label, zone_coords, zones)
from ml.embeddings import load_listings
from ml.matcher import embedding_backend, find_listings

ss = st.session_state
ss.setdefault("feed_query", "")  # empty = browse all listings, nearest and best quality first
ss.setdefault("feed_minq", 0)  # no pre-filled amount: the buyer enters the quantity they need
ss.setdefault("feed_qual", ["good", "fair", "poor"])
mine = list(ss.get("my_listings", []))  # listings published on "I have waste" this session


def sub_types_for(waste_type: str) -> list[str]:
    """Sub-types that actually have listings (demo or yours) for this waste type."""
    found = set(load_listings().query("waste_type == @waste_type")["sub_type"])
    found |= {l["sub_type"] for l in mine if l["waste_type"] == waste_type}
    return sorted(found)


# ---------------------------------------------------------------- hero: what is listed right now
supply = all_listings().groupby("waste_type")["quantity_kg"].sum()
supply = supply.reindex([t for t in ALL_WASTE_TYPES if t in supply.index])
st.html(vz.hero_html(
    "Find the feedstock that's already out there.",
    "Describe the material you need in your own words. Listings are ranked by how well they match, how close "
    "they are and their quality, with the cost and CO₂e for the amount you need.",
    f"Demo marketplace: {len(all_listings())} listings across Bengaluru",
    ["Describe the need", "Filter", "Compare listings", "Contact the seller"],
    vz.wheel_html([(t, float(kg), f"{waste_label(t)} {kg / 1000:,.0f} t") for t, kg in supply.items()],
                  fmt_kg(supply.sum()), "listed now, by material (demo data)")))

# ---------------------------------------------------------------- 01 what you need
with st.container(border=True, key="card_search", gap="small"):
    step_title(1, "What you need")
    query = st.text_input("What feedstock do you need?", icon=":material/search:", max_chars=500,
                          key="feed_query",
                          placeholder="e.g. cotton offcuts for recycled yarn, or leave empty to browse")
    f1, f2, f3 = st.columns(3, gap="small")
    wt = f1.selectbox("Waste type", ["any"] + ALL_WASTE_TYPES, key="feed_wt",
                      format_func=lambda t: "Any type" if t == "any" else waste_label(t))
    sub_opts = ["any"] + (sub_types_for(wt) if wt != "any" else [])
    if ss.get("feed_sub") not in sub_opts:  # the sub-type list follows the waste type
        ss.feed_sub = "any"
    sub = f2.selectbox("Sub-type", sub_opts, key="feed_sub", disabled=wt == "any",
                       format_func=lambda s: "Any sub-type" if s == "any" else sub_label(s),
                       help="Choose a waste type first." if wt == "any" else None)
    need = f3.number_input("Quantity needed (kg)", min_value=0, step=100, key="feed_minq",
                           help="Only listings holding at least this much are shown, and the CO₂e and cost "
                                "estimates are for this amount. Leave at 0 to see every listing.")
    g1, g2 = st.columns([1, 1], gap="small")
    with g1:
        qual = st.pills("Quality", ["good", "fair", "poor"], selection_mode="multi",
                        format_func=str.capitalize, key="feed_qual")
    loc = g2.selectbox("Deliver to", zones().index, key="feed_loc")

lat, lon = zone_coords(loc)
has_query = bool(query.strip())
SHOWN = 10  # cards / map pins; the KPIs below count EVERY listing that passes the filters, not just these
matches = find_listings(query, waste_type=None if wt == "any" else wt, sub_type=None if sub == "any" else sub,
                        lat=lat, lon=lon, min_quantity_kg=need, quality=qual or None, top_k=10_000,
                        extra_listings=mine)
results = matches[:SHOWN]

# ---------------------------------------------------------------- 02 best match
st.space("medium")
with st.container(horizontal=True, vertical_alignment="bottom"):
    with st.container(gap="xxsmall"):
        step_title(2, "Best match" if has_query else "Top result")
        if has_query:  # no caption when browsing (removed at the user's request)
            st.caption(f"Ranked for “{query.strip()}”, delivered to {loc}")
    n_mine = sum(not r["is_synthetic"] for r in results)
    if mine:
        st.badge(f"{n_mine} of yours" if n_mine else "None of yours shown", color="blue")

if not results:
    st.html('<div class="mm-empty mm-gridbg"><b>No listings match these filters</b><p>Try another sub-type, a lower '
            "quantity, or a wider quality range.</p></div>")
    st.stop()

top = results[0]
if top["co2e_per_kg"] is None:
    co2_value, co2_note = "n/a", f"No carbon factor for {sub_label(top['sub_type']).lower()} yet"
elif need > 0:
    co2_value = f"{need * top['co2e_per_kg']:,.1f} kg"
    co2_note = f"By recycling {need:,.0f} kg of {sub_label(top['sub_type']).lower()}"
else:
    co2_value, co2_note = "n/a", "Enter a quantity to see it"

# Cost per quality grade: the best-ranked listing of each grade FOR THE SAME MATERIAL as the top match, so the
# three prices are comparable (the quality pills only narrow the list below, all three grades stay visible).
top_by_quality = {}
for q in ("good", "fair", "poor"):
    best = find_listings(query, waste_type=top["waste_type"], sub_type=top["sub_type"], lat=lat, lon=lon,
                         min_quantity_kg=need, quality=[q], top_k=1, extra_listings=mine)
    top_by_quality[q] = best[0] if best else None

with st.container(horizontal=True, gap="small", key="kpis_feed"):
    kpi("matches", "Listings found", f"{len(matches)}",
        f"Best {len(results)} shown, top {'match' if has_query else 'result'} {top['score']:.0%}"
        if len(matches) > len(results) else f"Top {'match' if has_query else 'result'} {top['score']:.0%}")
    kpi("avail", "Total available", fmt_kg(sum(r["quantity_kg"] for r in matches)),
        f"Across all {len(matches)} listings" if len(matches) > 1 else "In this listing")
    cost_by_quality_kpi(top_by_quality, need,
                        help="Quantity needed × price per kg of the best-ranked listing of each quality, for the "
                             "same material as the top match. Seller prices are indicative, not a quote.")
    kpi("co2", "CO₂e saved on your quantity", co2_value, co2_note,
        help="Quantity needed × the carbon factor of the top match (kg CO₂e avoided per kg vs. virgin "
             "material). It is for the amount you need, not for the sellers' whole batches.")

spot_col, map_col = st.columns([7, 5], gap="medium")
with spot_col:
    reveal = ss.get("p2_top") != top["waste_id"]  # animate only when the top listing changes
    ss.p2_top = top["waste_id"]
    dist = f", {top['distance_km']:.1f} km away" if top.get("distance_km") is not None else ""
    yours = top.get("source") == "Your listing"
    st.html(vz.spotlight_html(
        f"Best match for “{query.strip()}”" if has_query else "Nearest, best-quality listing",
        sub_label(top["sub_type"]),
        f"{top['quantity_kg']:,} kg from {seller_display(top['seller_name'])}, {top['location_name']}{dist}",
        vz.chip("Your listing") if yours else vz.chip("Demo listing", "off"),
        float(top["score"]), "match" if has_query else "score", vz.material_color(top["waste_type"]), reveal))
    st.html(vz.cells_html([
        ("Price", price_text(top, need), "Indicative, not a quote"),
        ("CO₂e", listing_carbon_text(top, need).replace(":material/eco: ", "", 1), "vs. virgin material"),
        ("Quality", top["quality"].capitalize(), waste_label(top["waste_type"])),
        ("Contact", top.get("seller_contact") or "Not listed", "Demo placeholder" if not yours else "As entered"),
    ]))
    if st.button("Details & contact", key=f"spot_contact_{top['waste_id']}", icon=":material/contact_page:",
                 type="primary"):
        listing_dialog(top, need)
with map_col:
    with st.container(border=True, key="card_map", gap="small"):
        section_title("Map")
        results_map((lat, lon),
                    [{"lat": r["lat"], "lon": r["lon"], "name": f"{sub_label(r['sub_type'])}, {r['quantity_kg']:,} kg",
                      "waste_type": r["waste_type"],
                      "detail": f"Rank {i + 1}, {r['score']:.0%} {'match' if has_query else 'score'}, "
                                f"{seller_display(r['seller_name'])}, {r['location_name']}"}
                     for i, r in enumerate(results)],
                    height=360, origin_label=f"Your site at {loc}", points_label="Listings")
        st.caption(f"Matched on meaning, not exact words ({html.escape(embedding_backend())})." if has_query
                   else "Browsing: sorted by distance and quality.")

# ---------------------------------------------------------------- 03 compare the rest
if len(results) > 1:
    st.space("medium")
    step_title(3, "Compare the rest", f"next {len(results) - 1} of {len(matches)}")
    cols = st.columns(2, gap="medium")
    for rank, r in enumerate(results[1:], 1):
        with cols[(rank - 1) % 2]:
            listing_card(r, rank, need)
