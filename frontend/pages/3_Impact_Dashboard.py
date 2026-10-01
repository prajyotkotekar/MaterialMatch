import html

import altair as alt
import pandas as pd
import streamlit as st

from backend import feedback_store
from common import (ALL_WASTE_TYPES, all_listings, fmt_kg, kpi, material_color, page_header, section_title,
                    sub_label, type_tag, waste_label)
from ml.carbon import load_factors


def material_scale(types) -> alt.Scale:
    """Bars keep each material's own colour (same as tags, card edges and map pins)."""
    return alt.Scale(domain=[waste_label(t) for t in types], range=[material_color(t) for t in types])


with st.container(horizontal=True, vertical_alignment="bottom"):
    with st.container(gap="xxsmall"):
        page_header("Impact", "What could be saved if every listed batch were reused instead of landfilled, "
                              "and the carbon factors behind each estimate.")
    st.badge("Demo listings", color="gray")

def history_rows(rows: list[dict]) -> str:
    """Recent predictions as ruled rows: time, material, confidence bar, status."""
    out = []
    for r in rows:
        wt = r["waste_type"]
        name = type_tag(wt) if wt in ALL_WASTE_TYPES else '<span class="mm-tag">Unknown</span>'
        sub = (f'<span class="mm-area">{html.escape(sub_label(r["sub_type"]))}</span>'
               if r.get("sub_type") and r["sub_type"] != wt else "")
        status = {"detected": "Detected", "confirm": "Please confirm"}.get(r["status"], "Other / unknown")
        out.append(f'<div><time>{r["time"]:%H:%M:%S}</time><span>{name}{sub}</span>'
                   f'<span class="mm-bar" style="--c:{material_color(wt)}" title="{r["confidence"]:.0%}">'
                   f'<i style="width:{r["confidence"]:.0%}"></i></span>'
                   f'<span class="mm-st mm-st-{html.escape(str(r["status"]))}">{status}, {r["confidence"]:.0%}</span></div>')
    return '<div class="mm-hist">' + "".join(out) + "</div>"


# The carbon factors (formerly their own page) live in a tab, so each view stays uncluttered.
overview_tab, class_tab, factors_tab = st.tabs(["Overview", "Classifications", "Carbon factors"], key="impact_tab")

with class_tab:
    log = st.session_state.get("class_log", [])
    answered = feedback_store.load_feedback()
    fs = feedback_store.summary()
    st.caption("What the photo classifier has seen. Session numbers count this browser session only; "
               "answered photos come from the saved feedback log.")
    with st.container(horizontal=True, gap="small", key="kpis_class"):
        if log:
            counts = pd.Series([r["waste_type"] for r in log]).value_counts()
            kpi("c_n", "Classified this session", f"{len(log)}",
                f"{sum(r['photos'] for r in log)} photo{'s' if sum(r['photos'] for r in log) != 1 else ''}")
            kpi("c_top", "Most detected", waste_label(counts.index[0]) if counts.index[0] in ALL_WASTE_TYPES
                else "Unknown", f"{counts.iloc[0]} of {len(log)}")
            kpi("c_conf", "Average confidence", f"{sum(r['confidence'] for r in log) / len(log):.0%}",
                f"{sum(r['status'] == 'detected' for r in log)} detected without a doubt flag")
        else:
            kpi("c_n", "Classified this session", "0", "Add a photo on I have waste")
        kpi("c_fb", "Answered by users", f"{fs['n_items']}",
            f"{fs['accuracy_on_feedback']:.0%} confirmed correct" if fs["n_items"] else "No answers saved yet",
            help="Photos where someone answered 'Was this prediction correct?'. Saved across sessions.")

    left, right = st.columns([7, 5], gap="medium")
    with left:
        with st.container(border=True, key="card_history"):
            section_title("Recent predictions")
            if log:
                st.html(history_rows(list(reversed(log))[:12]))
            else:
                st.html('<div class="mm-empty mm-gridbg"><b>No predictions yet</b><p>Classify a photo on '
                        "I have waste or Present, and it shows up here.</p></div>")
                st.page_link("pages/1_List_Waste.py", label="Classify a photo", icon=":material/arrow_forward:")
    with right:
        with st.container(border=True, key="card_answered"):
            section_title("Answered photos by predicted material")
            if answered:
                fb = pd.DataFrame([{"material": waste_label(r["predicted_label"]),
                                    "answer": "Confirmed" if r["is_correct"] else "Corrected",
                                    "wt": r["predicted_label"]} for r in answered])
                st.altair_chart(
                    alt.Chart(fb).mark_bar().encode(
                        x=alt.X("count():Q", title=None, axis=alt.Axis(tickMinStep=1)),
                        y=alt.Y("material:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200, labelOverlap=False)),
                        color=alt.Color("answer:N", title=None, legend=alt.Legend(orient="bottom"),
                                        scale=alt.Scale(domain=["Confirmed", "Corrected"],
                                                        range=["#BEF04A", "#F0A43A"])),
                        tooltip=["material", "answer", "count()"],
                    ).properties(height=40 + 30 * fb["material"].nunique(), background="transparent"))
            else:
                st.caption("No answered photos yet.")

with overview_tab:
    df = all_listings()
    options = ALL_WASTE_TYPES + sorted(set(df["waste_type"]) - set(ALL_WASTE_TYPES))
    types = st.pills("Waste types", options, selection_mode="multi", default=options, key="impact_types",
                     format_func=waste_label, label_visibility="collapsed")
    df = df[df["waste_type"].isin(types or options)]

    if df.empty:
        st.info("No listings for this selection.")
    else:
        total_co2 = df["co2e_saved_kg"].sum()
        proxy_share = (df.loc[df["carbon_is_proxy"].fillna(False).astype(bool), "co2e_saved_kg"].sum() / total_co2
                       if total_co2 else 0)

        with st.container(horizontal=True, gap="small", key="kpis_impact"):
            kpi("co2", "Potential CO₂e saved", f"{total_co2 / 1000:,.1f} t", "Estimated, vs. virgin production")
            kpi("kg", "Waste that could be diverted", fmt_kg(df["quantity_kg"].sum()), f"{len(df):,} listings")
            residual = df.loc[df["waste_type"] == "trash", "quantity_kg"].sum()
            kpi("recov", "Recoverable share", f"{1 - residual / df['quantity_kg'].sum():.0%}",
                f"{fmt_kg(residual)} is mixed trash (residual)",
                help="By material type: every listed material except mixed trash has a recovery route. "
                     "It does not measure contamination within a batch.")
            kpi("proxy", "From proxy factors", f"{proxy_share:.0%}", "Share of CO₂e using surrogate factors",
                help="Share of the CO₂e total that relies on proxy (surrogate) emission factors. "
                     "The Carbon factors tab lists which materials use them.")

        present = [t for t in options if t in set(df["waste_type"])]
        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True, key="card_zone_chart"):
                section_title("Waste by zone, tonnes")
                by_zone = (df.assign(tonnes=df["quantity_kg"] / 1000, material=df["waste_type"].map(waste_label))
                           .groupby(["location_name", "material"], as_index=False)["tonnes"].sum())
                zone_order = (by_zone.groupby("location_name")["tonnes"].sum()
                              .sort_values(ascending=False).index.tolist())
                st.altair_chart(
                    alt.Chart(by_zone).mark_bar().encode(
                        x=alt.X("sum(tonnes):Q", title=None),
                        y=alt.Y("location_name:N", sort=zone_order, title=None, axis=alt.Axis(labelLimit=220)),
                        color=alt.Color("material:N", scale=material_scale(present), title="Material",
                                        legend=alt.Legend(orient="bottom", columns=3)),
                        tooltip=["location_name", "material", alt.Tooltip("tonnes:Q", format=",.1f")],
                    ).properties(height=440, background="transparent"))
        with right:
            with st.container(border=True, key="card_sub_chart"):
                section_title("CO₂e saved by material, tonnes")
                by_sub = (df.dropna(subset=["co2e_saved_kg"])  # materials without a carbon factor have no bar
                          .groupby(["sub_type", "waste_type"], as_index=False)["co2e_saved_kg"].sum()
                          .assign(tonnes=lambda d: d["co2e_saved_kg"] / 1000,
                                  sub=lambda d: d["sub_type"].map(sub_label),
                                  material=lambda d: d["waste_type"].map(waste_label)))
                with_co2 = [t for t in present if t in set(by_sub["waste_type"])]
                st.altair_chart(
                    alt.Chart(by_sub).mark_bar().encode(
                        x=alt.X("tonnes:Q", title=None),
                        y=alt.Y("sub:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                        color=alt.Color("material:N", scale=material_scale(with_co2), title="Material",
                                        legend=alt.Legend(orient="bottom", columns=3)),
                        tooltip=["sub", "material", alt.Tooltip("tonnes:Q", format=",.1f")],
                    ).properties(height=440, background="transparent"))

        with st.container(border=True, key="card_zone_table"):
            section_title("By zone")
            zone_table = (df.groupby("location_name")
                          .agg(listings=("waste_id", "count"), kg=("quantity_kg", "sum"),
                               co2e_kg=("co2e_saved_kg", "sum"))
                          .sort_values("co2e_kg", ascending=False)
                          .reset_index())
            zone_table = zone_table.assign(t=zone_table["kg"] / 1000, co2e_t=zone_table["co2e_kg"] / 1000)
            st.dataframe(
                zone_table[["location_name", "listings", "t", "co2e_t"]],
                hide_index=True,
                column_config={
                    "location_name": "Zone",
                    "listings": "Listings",
                    "t": st.column_config.NumberColumn("Waste", format="%.1f t"),
                    "co2e_t": st.column_config.ProgressColumn("Potential CO₂e saved", format="%.1f t", min_value=0,
                                                              max_value=float(zone_table["co2e_t"].max())),
                },
            )
        st.caption("Concrete and brick have very small CO₂e factors; their main benefit is avoided "
                   "quarrying and landfill space, which these charts do not capture. Mixed trash has no carbon "
                   "factor, so it counts towards waste but not towards CO₂e.")

with factors_tab:
    f = load_factors().reset_index()
    st.caption("kg CO₂e avoided per kg of material recycled instead of produced from virgin resources. "
               "Every CO₂e figure in the app is quantity × this factor.")

    with st.container(horizontal=True, gap="small", key="kpis_factors"):
        kpi("f_n", "Material factors", f"{len(f)}", "One per sub-type")
        kpi("f_sourced", "Sourced", f"{int((~f['is_proxy']).sum())}", "Direct factor from a cited source")
        kpi("f_proxy", "Proxy estimates", f"{int(f['is_proxy'].sum())}", "Surrogate factor, conservative")
        top = f.loc[f["co2e_saved_kg_per_kg"].idxmax()]
        kpi("f_max", "Highest saving", f"{top['co2e_saved_kg_per_kg']:.2f}",
            f"kg CO₂e per kg, {sub_label(top['sub_type']).lower()}")

    with st.container(border=True, key="card_factors", gap="small"):
        section_title("Factor by material")
        # every waste type that has at least one factor (trash has none, so it is not offered)
        with_factors = [t for t in ALL_WASTE_TYPES if t in set(f["waste_type"])]
        ftypes = st.pills("Waste types with factors", with_factors, selection_mode="multi", default=with_factors,
                          key="factor_types", format_func=waste_label, label_visibility="collapsed")
        view = f[f["waste_type"].isin(ftypes or with_factors)].sort_values("co2e_saved_kg_per_kg", ascending=False)
        st.dataframe(
            view.assign(sub=view["sub_type"].map(sub_label), type=view["waste_type"].map(waste_label),
                        basis=view["is_proxy"].map({True: "Proxy estimate", False: "Sourced"}),
                        # derived rows (fabric offcuts, textile fibre) have no link of their own: blank, not "None"
                        source_url=view["source_url"].str.split(";").str[0].str.strip().fillna(""),
                        notes=view["notes"].fillna(""))
            [["sub", "type", "co2e_saved_kg_per_kg", "basis", "factor_basis", "source_url", "notes"]],
            hide_index=True,
            height="content",
            column_config={
                "sub": "Material",
                "type": "Waste type",
                "co2e_saved_kg_per_kg": st.column_config.ProgressColumn(
                    "kg CO₂e / kg", format="%.3f", min_value=0, max_value=float(f["co2e_saved_kg_per_kg"].max())),
                "basis": "Basis",
                "factor_basis": st.column_config.TextColumn("Method", width="medium"),
                "source_url": st.column_config.LinkColumn("Source", display_text=r"https?://(?:www\.)?([^/]+)",
                                                          help="Website of the cited source; blank = derived "
                                                               "from other rows (see Method)."),
                "notes": st.column_config.TextColumn("Notes", width="medium"),
            },
        )
    st.caption("Sources: US EPA WARM recycling and composting chapters and cited LCAs. "
               "Proxy estimates borrow the closest WARM category and are conservative, not India-specific.")
