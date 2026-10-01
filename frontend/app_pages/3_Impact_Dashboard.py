import pandas as pd
import streamlit as st

import visuals as vz
from common import ALL_WASTE_TYPES, all_listings, fmt_kg, kpi, material_color, section_title, sub_label, waste_label
from ml.carbon import load_factors

_all = all_listings()
_co2 = _all.dropna(subset=["co2e_saved_kg"]).groupby("waste_type")["co2e_saved_kg"].sum()
_co2 = _co2.reindex([t for t in ALL_WASTE_TYPES if t in _co2.index])
st.html(vz.hero_html(
    "What this marketplace could keep out of landfill.",
    "Potential impact if every listed batch were reused instead of landfilled, where that waste sits, "
    "and the carbon factors behind each estimate.",
    f"Demo listings: {len(_all)} batches, {fmt_kg(_all['quantity_kg'].sum())}",
    ["Listed waste", "Carbon factor", "CO₂e avoided", "Where it comes from"],
    vz.wheel_html([(t, float(kg), f"{waste_label(t)} {kg / 1000:,.0f} t") for t, kg in _co2.items()],
                  f"{_co2.sum() / 1000:,.0f} t", "CO₂e potential by material (estimate)")))


def lever(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """CO2e potential per material (by='waste_type') or sub-type (by='sub_type'), largest first."""
    d = df.dropna(subset=["co2e_saved_kg"]).assign(
        proxy_co2=lambda x: x["co2e_saved_kg"].where(x["carbon_is_proxy"].fillna(False).astype(bool), 0))
    keys = ["sub_type", "waste_type"] if by == "sub_type" else ["waste_type"]
    return (d.groupby(keys).agg(co2=("co2e_saved_kg", "sum"), kg=("quantity_kg", "sum"), n=("waste_id", "count"),
                                proxy_co2=("proxy_co2", "sum"))
            .reset_index().sort_values(["co2", keys[0]], ascending=[False, True]))


def basis_chip(proxy_share: float) -> str:
    if proxy_share >= 0.999:
        return vz.chip("Proxy estimate", "warn")
    if proxy_share <= 0.001:
        return vz.chip("Sourced factor")
    return vz.chip(f"{proxy_share:.0%} from proxy factors", "warn")


# The carbon factors (formerly their own page) live in a tab, so each view stays uncluttered.
overview_tab, factors_tab = st.tabs(["Overview", "Carbon factors"], key="impact_tab")

with overview_tab:
    df = all_listings()
    # Mixed trash is residual waste with no recovery route or carbon factor, so the impact view leaves it out.
    df = df[df["waste_type"] != "trash"]
    options = [t for t in ALL_WASTE_TYPES if t != "trash"]
    options += sorted(set(df["waste_type"]) - set(options))
    types = st.pills("Waste types", options, selection_mode="multi", default=options, key="impact_types",
                     format_func=waste_label, label_visibility="collapsed")
    df = df[df["waste_type"].isin(types or options)]

    if df.empty:
        st.info("No listings for this selection.")
    else:
        rated = df.dropna(subset=["co2e_saved_kg"])
        total_co2 = rated["co2e_saved_kg"].sum()
        proxy_share = (rated.loc[rated["carbon_is_proxy"].fillna(False).astype(bool), "co2e_saved_kg"].sum() / total_co2
                       if total_co2 else 0)

        with st.container(horizontal=True, gap="small", key="kpis_impact"):
            kpi("co2", "Potential CO₂e saved", f"{total_co2 / 1000:,.1f} t", "Estimated, vs. virgin production")
            kpi("kg", "Waste that could be diverted", fmt_kg(df["quantity_kg"].sum()), f"{len(df):,} listings")
            kpi("avg", "Average saving", f"{total_co2 / rated['quantity_kg'].sum():.2f} kg" if len(rated) else "None",
                "CO₂e per kg of waste listed",
                help="Potential CO₂e divided by the listed waste that has a carbon factor.")
            kpi("proxy", "From proxy factors", f"{proxy_share:.0%}", "Share of CO₂e using surrogate factors",
                help="Share of the CO₂e total that relies on proxy (surrogate) emission factors. "
                     "The Carbon factors tab lists which materials use them.")

        # Biggest lever: with several materials selected, the material with the most CO2e potential;
        # with one material, its sub-type with the most.
        single = df["waste_type"].nunique() == 1
        board = lever(df, "sub_type" if single else "waste_type")
        if not board.empty and total_co2:
            top = board.iloc[0]
            wt = top["waste_type"]
            rows = df[df["sub_type"] == top["sub_type"]] if single else df[df["waste_type"] == wt]
            zone = rows.groupby("location_name")["quantity_kg"].sum().sort_values(ascending=False).index[0]
            name = sub_label(top["sub_type"]) if single else waste_label(wt)
            kicker = (f"Biggest lever within {waste_label(wt).lower()}" if single
                      else f"Biggest lever of the {df['waste_type'].nunique()} materials selected")
            st.space("small")
            st.html(vz.spotlight_html(
                kicker, name,
                f"{fmt_kg(top['kg'])} listed in {top['n']} batches, about {top['co2'] / 1000:,.1f} t CO₂e potential",
                basis_chip(top["proxy_co2"] / top["co2"]),
                float(top["co2"] / total_co2), "of CO₂e", material_color(wt), size="sm"))
            if single:
                nxt = board.iloc[1] if len(board) > 1 else None
                last = (("Runner-up", sub_label(nxt["sub_type"]), f"{nxt['co2'] / 1000:,.1f} t CO₂e") if nxt is not None
                        else ("Material", waste_label(wt), "Only sub-type with a factor"))
            else:
                subs = lever(rows, "sub_type")
                last = ("Top sub-type", sub_label(subs.iloc[0]["sub_type"]),
                        f"{subs.iloc[0]['co2'] / 1000:,.1f} t CO₂e of {len(subs)} sub-types")
            st.html(vz.cells_html([
                ("Carbon factor", f"{top['co2'] / top['kg']:.2f} kg CO₂e per kg",
                 "From the factor table" if single else "Average over its sub-types"),
                ("Share of potential", f"{top['co2'] / total_co2:.0%} of {total_co2 / 1000:,.0f} t", "This selection"),
                ("Most listed in", zone, "Industrial zone"),
                last,
            ]))
            st.space("small")

        present = [t for t in options if t in set(df["waste_type"])]
        left, right = st.columns([7, 5], gap="medium")
        with left:
            with st.container(border=True, key="card_zone_board"):
                section_title("Waste by zone")
                st.caption("Ranked by tonnes listed. Bars share one scale and split by material; hover a segment "
                           "for its share.")
                zones = []
                for name, g in df.groupby("location_name"):
                    zones.append({"name": name, "listings": len(g), "kg": float(g["quantity_kg"].sum()),
                                  "co2": float(g["co2e_saved_kg"].sum()),
                                  "mix": g.groupby("waste_type")["quantity_kg"].sum().to_dict()})
                zones.sort(key=lambda z: (-z["kg"], z["name"]))
                st.html(vz.legend_html(present) + vz.zone_board_html(zones))
        with right:
            with st.container(border=True, key="card_co2_rank"):
                section_title("CO₂e potential by sub-type")
                ranks = lever(df, "sub_type")
                st.caption(f"{len(ranks)} sub-types with a carbon factor, largest first.")
                st.html(vz.co2_rank_html([
                    {"label": sub_label(r.sub_type), "waste_type": r.waste_type, "co2": r.co2, "kg": r.kg,
                     "proxy": r.proxy_co2 > 0.5 * r.co2} for r in ranks.itertuples()])
                    + vz.legend_html([t for t in present if t in set(ranks["waste_type"])], proxy_key=True))

        st.caption("Concrete and brick have very small CO₂e factors; their main benefit is avoided quarrying and "
                   "landfill space, which these figures do not capture. Mixed trash is left out of this view: it "
                   "has no recovery route and no carbon factor.")

with factors_tab:
    f = load_factors().reset_index()
    st.caption("kg CO₂e avoided per kg of material recycled instead of produced from virgin resources. "
               "Every CO₂e figure in the app is quantity × this factor.")

    with st.container(horizontal=True, gap="small", key="kpis_factors"):
        kpi("f_n", "Material factors", f"{len(f)}", "One per sub-type")
        kpi("f_sourced", "Sourced", f"{int((~f['is_proxy']).sum())}", "Direct factor from a cited source")
        kpi("f_proxy", "Proxy estimates", f"{int(f['is_proxy'].sum())}", "Surrogate factor, conservative")
        best = f.loc[f["co2e_saved_kg_per_kg"].idxmax()]
        kpi("f_max", "Highest saving", f"{best['co2e_saved_kg_per_kg']:.2f}",
            f"kg CO₂e per kg, {sub_label(best['sub_type']).lower()}")

    with st.container(border=True, key="card_factors", gap="small"):
        section_title("Factor by material")
        # every waste type that has at least one factor (trash has none, so it is not offered)
        with_factors = [t for t in ALL_WASTE_TYPES if t in set(f["waste_type"])]
        ftypes = st.pills("Waste types with factors", with_factors, selection_mode="multi", default=with_factors,
                          key="factor_types", format_func=waste_label, label_visibility="collapsed")
        view = f[f["waste_type"].isin(ftypes or with_factors)].sort_values(
            ["co2e_saved_kg_per_kg", "sub_type"], ascending=[False, True])
        st.markdown(vz.factor_table_html([
            # derived rows (fabric offcuts, textile fibre) have no link of their own
            {"sub": sub_label(r.sub_type), "waste_type": r.waste_type, "factor": float(r.co2e_saved_kg_per_kg),
             "proxy": bool(r.is_proxy), "method": r.factor_basis if isinstance(r.factor_basis, str) else "",
             "url": r.source_url.split(";")[0].strip() if isinstance(r.source_url, str) else "",
             "notes": r.notes if isinstance(r.notes, str) else ""}
            for r in view.itertuples()]), unsafe_allow_html=True)
    st.caption("Sources: US EPA WARM recycling and composting chapters and cited LCAs. "
               "Proxy estimates borrow the closest WARM category and are conservative, not India-specific.")
