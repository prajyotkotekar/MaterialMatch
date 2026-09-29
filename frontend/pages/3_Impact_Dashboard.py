import streamlit as st

from common import ALL_WASTE_TYPES, all_listings, fmt_kg, kpi, section_title, sub_label, waste_label
from ml.carbon import load_factors

with st.container(horizontal=True, vertical_alignment="bottom"):
    with st.container(gap="xxsmall"):
        st.markdown("## Impact overview")
        st.caption("Potential impact if every listed batch were reused instead of sent to landfill, "
                   "and the carbon factors behind the estimates.")
    st.badge("Demo listings", icon=":material/science:", color="gray")

# The carbon factors (formerly their own page) live in a second tab, so each view stays uncluttered.
overview_tab, factors_tab = st.tabs([":material/insights: Overview", ":material/co2: Carbon factors"],
                                    key="impact_tab")

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
            kpi("zones", "Industrial zones", f"{df['location_name'].nunique()}", "Across Bengaluru")
            kpi("proxy", "From proxy factors", f"{proxy_share:.0%}", "Share of CO₂e using surrogate factors",
                help="Share of the CO₂e total that relies on proxy (surrogate) emission factors. "
                     "The Carbon factors tab lists which materials use them.")

        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True, key="card_zone_chart"):
                section_title("Waste by zone (tonnes)", "location_on")
                by_zone = (df.assign(tonnes=df["quantity_kg"] / 1000, type=df["waste_type"].map(waste_label))
                           .pivot_table(index="location_name", columns="type", values="tonnes",
                                        aggfunc="sum", fill_value=0))
                st.bar_chart(by_zone, horizontal=True, sort=False, height=440, x_label="", y_label="")
        with right:
            with st.container(border=True, key="card_sub_chart"):
                section_title("CO₂e saved by sub-type (tonnes)", "eco")
                by_sub = (df.dropna(subset=["co2e_saved_kg"])  # materials without a carbon factor have no bar
                          .groupby(["sub_type", "waste_type"], as_index=False)["co2e_saved_kg"].sum()
                          .assign(tonnes=lambda d: d["co2e_saved_kg"] / 1000,
                                  sub=lambda d: d["sub_type"].map(sub_label),
                                  **{"Waste type": lambda d: d["waste_type"].map(waste_label)}))
                st.bar_chart(by_sub, x="sub", y="tonnes", color="Waste type", horizontal=True, sort="-tonnes",
                             height=440, x_label="", y_label="")

        with st.container(border=True, key="card_zone_table"):
            section_title("By zone", "table_chart")
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
        st.caption(":material/info: Concrete and brick have very small CO₂e factors; their main benefit is avoided "
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
            f"kg CO₂e per kg · {sub_label(top['sub_type'])}")

    with st.container(border=True, key="card_factors", gap="small"):
        section_title("Factor by material", "table_chart")
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
    st.caption(":material/info: Sources: US EPA WARM recycling and composting chapters and cited LCAs. "
               "Proxy estimates borrow the closest WARM category and are conservative, not India-specific.")
