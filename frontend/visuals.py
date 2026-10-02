"""HTML/CSS building blocks for the identification flow: hero, scan state, result card, recovery loop.

Everything shown is either model output, data from the project (carbon factors, recyclers) or general
recovery guidance per material (ROUTES), never invented measurements."""

from __future__ import annotations

import base64
import html
import io
import math

import streamlit as st
from PIL import Image

from common import WASTE_COLOR, material_color, sub_label, waste_label
from ml import taxonomy
from ml.carbon import co2e_saved
from ml.classifier.predict import SMALL_SIDE, check_pixels
from ml.matcher import load_recyclers

MODEL_INPUT = 224  # the classifier sees 224 x 224 px
STAGES = ["Waste", "Sorting", "Recovery", "Recycling", "New product"]

# General recovery guidance per material (static information, not model output).
ROUTES = {
    "plastic": ("Recyclable when sorted by polymer and clean", "Keep it dry and clean, bag or bale by type.",
                ["Separated by polymer (PET, HDPE, PP)", "Shredded, washed and dried", "Melted into pellets or flakes",
                 "Fibre, film, pipes, containers"]),
    "paper": ("Widely recycled while dry", "Keep it dry, flatten and bale cardboard.",
              ["Graded into cardboard and mixed paper", "Pulped in water", "Screened, de-inked and pressed",
               "Cardboard, paperboard, tissue"]),
    "glass": ("Recyclable many times without losing quality", "Keep it apart from ceramics and stones.",
              ["Sorted by colour, caps removed", "Crushed into cullet", "Remelted in a furnace",
               "Bottles, jars, glass wool"]),
    "metal": ("Recyclable many times", "Separate iron and steel from other metals if you can.",
              ["Split by magnet and eddy current", "Shredded and cleaned", "Remelted and cast",
               "Steel and aluminium stock"]),
    "textile": ("Reusable or recyclable, blends are harder", "Keep it dry and set aside wearable items.",
                ["Graded for reuse or recycling", "Sorted by fibre, trims removed", "Shredded back to fibre",
                 "Yarn, insulation, wiping cloths"]),
    "e_waste": ("Recoverable through authorised recyclers", "Do not dismantle it; hand it to an authorised e-waste recycler.",
                ["Received by an authorised recycler", "Batteries and hazardous parts removed",
                 "Metals recovered from boards and cables", "Copper, aluminium, precious metals"]),
    "construction": ("Largely recoverable as aggregate", "Keep it free of plaster, soil and mixed trash.",
                     ["Concrete, brick and wood separated", "Crushed and screened", "Graded as recycled aggregate",
                      "Road base, blocks, fill"]),
    "biological": ("Compostable", "Keep wet waste apart from dry waste.",
                   ["Kept apart from dry waste", "Composted or digested", "Matured into compost or biogas",
                    "Soil conditioner, cooking gas"]),
    "trash": ("Mostly residual", "Pull out anything recyclable before disposal.",
              ["Picked for any recyclables", "Residue has no material route", "Co-processing or landfill",
               "Reduce it at source"]),
}
BROKEN_FROM = {"trash": 2}  # loop stages from this index on are not a recovery route


def esc(text) -> str:
    return html.escape(str(text), quote=True)


# ---------- photo info ----------

def image_info(data: bytes) -> dict | None:
    """Size, format and a quality verdict measured against what the classifier actually uses."""
    try:
        im = Image.open(io.BytesIO(data))
        w, h = im.size
        fmt = im.format or "image"
    except Exception:
        return None
    short = min(w, h)
    if short < SMALL_SIDE:
        q, note = "low", f"Under {SMALL_SIDE} px: predictions are less reliable"
    elif short < MODEL_INPUT:
        q, note = "fair", f"Below the model's {MODEL_INPUT} px input size"
    else:
        q, note = "good", f"Sharp enough for the model ({MODEL_INPUT} px input)"
    return {"w": w, "h": h, "format": fmt, "kb": len(data) / 1024, "quality": q, "note": note}


@st.cache_data(max_entries=64, show_spinner=False)
def thumb_uri(data: bytes, side: int = 480) -> str:
    try:
        im = Image.open(io.BytesIO(data))
        check_pixels(im)
        im = im.convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def file_rows(files) -> str:
    rows = []
    for f in files:
        data = f.getvalue()
        info = image_info(data)
        meta = (f"{info['w']} × {info['h']} px, {info['format']}, {info['kb']:,.0f} KB" if info else "Unreadable file")
        chip = (f'<span class="mm-iq mm-iq-{info["quality"]}" title="{esc(info["note"])}">{info["quality"].capitalize()}'
                f' quality</span>' if info else '<span class="mm-iq mm-iq-low">Can\'t read</span>')
        rows.append(f'<div class="mm-file"><img src="{thumb_uri(data, 160)}" alt=""><div><b>{esc(f.name)}</b>'
                    f'<span>{meta}</span><span class="mm-file-note">{esc(info["note"]) if info else ""}</span></div>'
                    f'{chip}</div>')
    return '<div class="mm-files">' + "".join(rows) + "</div>"


# ---------- hero ----------

def _polar(deg: float, radius_pct: float) -> tuple[float, float]:
    """Point on a circle in % of the box; 0 deg = top, clockwise (same as conic-gradient)."""
    r = math.radians(deg)
    return 50 + radius_pct * math.sin(r), 50 - radius_pct * math.cos(r)


def _segments(colors: list[str], gap: float, weights: list[float] | None = None) -> str:
    """conic-gradient stops: one segment per colour (sized by weight), with a transparent gap between them."""
    weights = weights or [1.0] * len(colors)
    total = sum(weights) or 1.0
    stops, start, end = [], 0.0, 0.0
    for c, w in zip(colors, weights):
        span = 360 * w / total
        a0, a1 = start + gap / 2, start + span - gap / 2
        stops.append(f"transparent {end:.1f}deg {a0:.1f}deg, {c} {a0:.1f}deg {max(a0, a1):.1f}deg")
        end, start = max(a0, a1), start + span
    return "conic-gradient(" + ", ".join(stops) + f", transparent {end:.1f}deg 360deg)"


def wheel_html(parts: list[tuple[str, float, str]], big: str, small: str) -> str:
    """The material wheel: parts = (waste_type, weight, label). Segments are sized by weight; very small
    ones keep a minimum size so every material stays visible, and their labels are dropped if crowded."""
    total = sum(w for _, w, _ in parts) or 1.0
    weights = [max(w / total, 0.025) for _, w, _ in parts]
    shown_total = sum(weights)
    labels, start = [], 0.0
    for (t, _, text), w in zip(parts, weights):
        span = 360 * w / shown_total
        if span >= 22:
            x, y = _polar(start + span / 2, 41)
            side = "r" if x > 56 else "l" if x < 44 else "c"  # anchor away from the ring
            dy = ("-0.9rem" if y < 50 else "0.9rem") if side == "c" else "0rem"
            labels.append(f'<span class="mm-wheel-l mm-wl-{side}" style="left:{x:.1f}%;top:{y:.1f}%;--dy:{dy}">'
                          f'{esc(text)}</span>')
        start += span
    wheel = _segments([WASTE_COLOR.get(t, "#8F988D") for t, _, _ in parts], 3, weights)
    return (f'<div class="mm-hero-art" aria-hidden="true"><div class="mm-wheel" style="background:{wheel}"></div>'
            f'<div class="mm-wheel-in"></div><div class="mm-sweep"></div>{"".join(labels)}'
            f'<div class="mm-hero-core"><b>{esc(big)}</b><span>{esc(small)}</span></div></div>')


def hero_html(title: str, lede: str, status: str, flow: list[str], art: str) -> str:
    """Page opener: status line, headline, lede, the page's steps, and a visual (usually the wheel)."""
    steps = "".join(f"<li><b>{i:02d}</b>{esc(s)}</li>" for i, s in enumerate(flow, 1))
    return (f'<section class="mm-hero"><div class="mm-hero-copy"><div class="mm-status"><i></i>{esc(status)}</div>'
            f'<h1 class="mm-hero-h">{esc(title)}</h1><p class="mm-hero-p">{esc(lede)}</p>'
            f'<ol class="mm-flow">{steps}</ol></div>{art}</section>')


def classifier_hero_html() -> str:
    """Hero of 'I have waste': the 9 materials the model knows."""
    types = list(taxonomy.WASTE_TYPES)
    n_sub = sum(len(v) for v in taxonomy.photo_hierarchy().values())
    return hero_html(
        "Your waste is someone's raw material.",
        "Photograph it. The model names the material, and MaterialMatch finds the Bengaluru recyclers who can "
        "use it, with the CO₂e that saves compared with new material.",
        f"Photo classifier: {len(types)} materials, {n_sub} sub-types",
        ["Photo in", "Model reads it", "Material named", "Recycler found"],
        wheel_html([(t, 1.0, waste_label(t)) for t in types], str(n_sub), "sub-types it can tell apart"))


# ---------- scan state (shown while the model is actually running) ----------

def scan_html(files) -> str:
    img = thumb_uri(files[0].getvalue())
    more = f'<span class="mm-scan-more">+{len(files) - 1} more</span>' if len(files) > 1 else ""
    steps = "".join(f"<li style='--d:{i * 0.35:.2f}s'>{esc(s)}</li>" for i, s in enumerate(
        ["Reading the photo", "Comparing it with 27 material sub-types", "Checking how familiar the photo looks"]))
    return f"""
<div class="mm-scan" role="status" aria-live="polite">
  <div class="mm-scan-img"><img src="{img}" alt="Photo being analysed"><div class="mm-scan-grid"></div>
    <div class="mm-scan-beam"></div>{more}</div>
  <div class="mm-scan-txt"><b>Identifying the material</b><ol>{steps}</ol></div>
</div>"""


# ---------- result card ----------

def _ring(pct: float, color: str, size: str, label: str = "confidence") -> str:
    """Dial (CSS conic-gradient); --p drives both the arc and the number, so they animate together."""
    p = max(0.0, min(1.0, pct))
    return (f'<div class="mm-ring mm-ring-{size}" style="--p:{round(p * 100)};--c:{color}"><div class="mm-ring-dial"></div>'
            f'<span class="mm-ring-n"></span><span class="mm-ring-l">{esc(label)}</span></div>')


def spotlight_html(kicker: str, name: str, line: str, chip: str, pct: float, ring_label: str, color: str,
                   reveal: bool = False, size: str = "lg") -> str:
    """Off-white spotlight card: the one thing a page wants you to look at (a result, a top listing, ...)."""
    cls = f"mm-res mm-res-{size}" + (" mm-reveal" if reveal else "")
    return (f'<div class="{cls}" style="--c:{color}"><div class="mm-res-band"></div><div class="mm-res-body">'
            f'<div class="mm-res-main"><span class="mm-res-k">{esc(kicker)}</span>'
            f'<span class="mm-res-name">{esc(name)}</span>'
            + (f'<span class="mm-res-sub">{esc(line)}</span>' if line else "")
            + f"{chip}</div>{_ring(pct, color, size, ring_label)}</div></div>")


def match_html(flag: str, source: str, seller: str, offer: str, where: str, checks: list[tuple[str, str, str, bool]],
               pct: float, ring_label: str, color: str, reveal: bool = False) -> str:
    """The top listing on 'I need feedstock' as a seller card: who sells it, what and where, the match score,
    and one check per buyer criterion. checks = (criterion, value, note, met)."""
    items = "".join(
        f'<li class="{"" if met else "mm-open"}"><span class="mm-ico" aria-hidden="true">{"check" if met else "remove"}</span>'
        f'<span>{esc(k)}</span><b>{esc(v)}</b><small>{esc(n)}</small></li>' for k, v, n, met in checks)
    cls = "mm-match" + (" mm-reveal" if reveal else "")
    return (f'<div class="{cls}" style="--c:{color}">'
            f'<div class="mm-match-top"><span class="mm-match-flag"><span class="mm-ico" aria-hidden="true">verified</span>'
            f'{esc(flag)}</span><span class="mm-match-src">{esc(source)}</span></div>'
            f'<div class="mm-match-body"><div class="mm-match-main"><b class="mm-match-name">{esc(seller)}</b>'
            f'<span class="mm-match-offer">{esc(offer)}</span>'
            f'<span class="mm-match-where"><span class="mm-ico" aria-hidden="true">location_on</span>{esc(where)}</span></div>'
            f'{_ring(pct, color, "lg", ring_label)}</div>'
            f'<ul class="mm-match-checks" aria-label="How this listing meets your criteria">{items}</ul></div>')


def chip(text: str, kind: str = "ok") -> str:
    return f'<span class="mm-chip mm-chip-{kind}">{esc(text)}</span>'


def result_html(pred: dict, reveal: bool, size: str = "lg", sub_note: str = "") -> str:
    """The identification 'moment': material name, confidence dial, status. sub_note = sub-type line."""
    if pred.get("is_unknown"):
        g = pred.get("best_guess") or {}
        conf = float(g.get("confidence") or 0)
        return spotlight_html("Not identified", "Unknown material",
                              f"Closest guess {waste_label(g.get('waste_type'))} at {conf:.0%}, not reliable enough to use",
                              chip("Other / unknown", "off"), conf, "confidence", "#8F988D", reveal, size)
    wt = pred["label"]
    status = chip("Detected") if pred.get("is_confident") else chip("Please confirm", "warn")
    return spotlight_html("Material identified", waste_label(wt), sub_note, status, float(pred["confidence"]),
                          "confidence", material_color(wt), reveal, size)


def cells_html(cells: list[tuple[str, str, str]]) -> str:
    """Row of small fact cells: (label, value, note)."""
    return '<div class="mm-facts">' + "".join(
        f'<div><span>{esc(k)}</span><b>{esc(v)}</b><small>{esc(n)}</small></div>' for k, v, n in cells) + "</div>"


def facts_html(waste_type: str | None, sub_type: str | None = None) -> str:
    """Supporting facts around the result: recovery route, recyclability, impact, handling, recyclers."""
    if waste_type not in ROUTES:
        return ""
    recyclability, action, loop = ROUTES[waste_type]
    carbon = co2e_saved(sub_type, 1.0, waste_type=waste_type)
    if carbon.get("factor_available") and carbon.get("co2e_saved_kg") is not None:
        impact = f"{carbon['co2e_saved_kg']:.2f} kg CO₂e saved per kg recycled"
        impact_note = "Proxy estimate" if carbon.get("is_proxy") else "Sourced factor"
    else:
        impact, impact_note = "No sourced carbon factor", "Shown as unavailable, not borrowed"
    rec = load_recyclers()
    n_rec = int(rec["waste_type_list"].apply(lambda ts: waste_type in ts).sum())
    cells = [
        ("Recovery route", f"{loop[1]}, then {loop[2][0].lower() + loop[2][1:]}", "General guidance"),
        ("Recyclability", recyclability, "General guidance"),
        ("Environmental impact", impact, impact_note),
        ("Handling", action, "Before pickup"),
        ("Recyclers in the network", f"{n_rec} accept {waste_label(waste_type).lower()}",
         "Sample profiles, Bengaluru"),
    ]
    return cells_html(cells)


# ---------- circular recovery loop ----------

def loop_html(waste_type: str | None, sub_type: str | None = None) -> str:
    wt = waste_type if waste_type in ROUTES else "plastic"
    _, _, steps = ROUTES[wt]
    color = material_color(wt)
    broken = BROKEN_FROM.get(wt, 99)
    first = sub_label(sub_type) if sub_type else waste_label(wt)
    texts = [f"Your {first.lower()}"] + steps
    # segment i runs from stage i to stage i+1; from `broken` on there is no recovery route
    ring = _segments([color if i < broken else "#363B36" for i in range(5)], 14)
    nodes = "".join(
        f'<span class="mm-loop-node{" off" if i > broken else ""}" style="left:{x:.1f}%;top:{y:.1f}%">{i + 1}</span>'
        for i in range(5) for x, y in [_polar(i * 72, 48.5)])
    legend = "".join(
        f'<li class="{"off" if i > broken else ""}"><b>{i + 1}</b><div><span>{esc(STAGES[i])}</span>{esc(t)}</div></li>'
        for i, t in enumerate(texts))
    core = "No recovery loop" if wt in BROKEN_FROM else "Back into use"
    return f"""
<div class="mm-loop" style="--c:{color}">
  <div class="mm-loop-art" aria-hidden="true">
    <div class="mm-loop-ring" style="background:{ring}"></div>
    <div class="mm-orbit"><i></i></div>
    {nodes}
    <div class="mm-loop-core"><b>{esc(waste_label(wt))}</b><span>{core}</span></div>
  </div>
  <ol class="mm-loop-list">{legend}</ol>
</div>"""


# ---------- impact page: zone board, CO2e ranking, factor table ----------

# Stacking/legend order chosen so neighbouring colours stay apart (checked for colour-blind separation).
SEGMENT_ORDER = ["biological", "textile", "paper", "e_waste", "construction", "glass", "plastic", "metal"]


def mass(kg: float) -> str:
    return f"{kg / 1000:,.1f} t" if kg >= 1000 else f"{kg:,.0f} kg"


def legend_html(types, proxy_key: bool = False) -> str:
    """Material swatches in SEGMENT_ORDER; proxy_key adds the solid / hatched key of the CO2e bars."""
    order = [t for t in SEGMENT_ORDER if t in types] + [t for t in types if t not in SEGMENT_ORDER]
    items = "".join(f'<span style="--c:{material_color(t)}"><i></i>{esc(waste_label(t))}</span>' for t in order)
    if proxy_key:
        items += ('<span class="mm-key" style="--c:#A4ABA2"><i></i>Sourced factor</span>'
                  '<span class="mm-key" style="--c:#A4ABA2"><i class="mm-px"></i>Proxy estimate</span>')
    return f'<div class="mm-legend">{items}</div>'


def zone_board_html(zones: list[dict]) -> str:
    """One row per zone: listings, waste, CO2e potential, and a bar split by material.
    zones = [{name, listings, kg, co2, mix: {waste_type: kg}}], already sorted; bars share one scale."""
    top = max((z["kg"] for z in zones), default=0) or 1
    rows = []
    for i, z in enumerate(zones):
        segs = "".join(
            f'<i style="--c:{material_color(t)};flex:{kg:.3f}" tabindex="0" '
            f'data-tip="{esc(waste_label(t))}: {mass(kg)}, {kg / z["kg"]:.0%} of this zone"></i>'
            for t in SEGMENT_ORDER + [t for t in z["mix"] if t not in SEGMENT_ORDER]
            if (kg := z["mix"].get(t, 0)) > 0)
        rows.append(
            f'<div class="mm-zr" style="--d:{i * 70}ms"><span class="mm-zr-i">{i + 1}</span>'
            f'<b class="mm-zr-name">{esc(z["name"])}</b><span class="mm-zr-n">{z["listings"]}</span>'
            f'<span class="mm-zr-kg">{mass(z["kg"])}</span><span class="mm-zr-co2">{mass(z["co2"])}</span>'
            f'<div class="mm-zr-bar" style="width:{100 * z["kg"] / top:.1f}%">{segs}</div></div>')
    head = ('<div class="mm-zr mm-zr-h"><span class="mm-zr-i"></span><span class="mm-zr-name">Zone</span>'
            '<span class="mm-zr-n">Listings</span><span class="mm-zr-kg">Waste</span>'
            '<span class="mm-zr-co2">CO₂e potential</span></div>')
    return f'<div class="mm-board">{head}{"".join(rows)}</div>'


def co2_rank_html(rows: list[dict]) -> str:
    """Ranked CO2e bars per sub-type: rows = [{label, waste_type, co2, kg, proxy}], sorted, largest first.
    Proxy-based estimates are hatched so they never pass for sourced figures."""
    top = max((r["co2"] for r in rows), default=0) or 1
    out = []
    for i, r in enumerate(rows):
        basis = "proxy estimate" if r["proxy"] else "sourced factor"
        tip = f'{waste_label(r["waste_type"])}, {mass(r["kg"])} listed, {basis}'
        out.append(
            f'<div class="mm-rk" style="--c:{material_color(r["waste_type"])};--d:{i * 35}ms">'
            f'<span class="mm-rk-l"><i></i>{esc(r["label"])}</span>'
            f'<span class="mm-rk-track"><i class="{"mm-px" if r["proxy"] else ""}" tabindex="0" data-tip="{esc(tip)}" '
            f'style="width:max(3px, {100 * r["co2"] / top:.2f}%)"></i></span><b>{mass(r["co2"])}</b></div>')
    return f'<div class="mm-rank-list">{"".join(out)}</div>'


def factor_table_html(rows: list[dict]) -> str:
    """Carbon factors as rows: material, factor bar, basis, source link, then method and notes.
    rows = [{sub, waste_type, factor, proxy, method, url, notes}]; render with st.markdown (keeps link targets)."""
    top = max((r["factor"] for r in rows), default=0) or 1
    out = []
    for r in rows:
        url = r["url"]
        domain = url.split("//", 1)[-1].split("/", 1)[0].removeprefix("www.") if url else ""
        source = (f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(domain)}</a>' if url
                  else '<span class="mm-ft-none" title="Derived from other rows, see the method">Derived</span>')
        method, notes = r["method"].strip(), r["notes"].strip()
        if method and notes and not method.endswith("."):
            method += "."
        note = " ".join(x for x in (method, notes) if x)
        out.append(
            f'<div class="mm-ft-r" style="--c:{material_color(r["waste_type"])}">'
            f'<div class="mm-ft-m"><b>{esc(r["sub"])}</b><span><i></i>{esc(waste_label(r["waste_type"]))}</span></div>'
            f'<div class="mm-ft-v"><b>{r["factor"]:.3f}</b><span class="mm-rk-track"><i class="{"mm-px" if r["proxy"] else ""}" '
            f'style="width:max(3px, {100 * r["factor"] / top:.2f}%)"></i></span></div>'
            f'<div class="mm-ft-b"><span class="mm-basis{" mm-basis-px" if r["proxy"] else ""}">'
            f'{"Proxy estimate" if r["proxy"] else "Sourced"}</span></div>'
            f'<div class="mm-ft-s">{source}</div><p class="mm-ft-note">{esc(note)}</p></div>')
    head = ('<div class="mm-ft-r mm-ft-h"><span>Material</span><span>kg CO₂e saved per kg</span><span>Basis</span>'
            '<span>Source</span></div>')
    return f'<div class="mm-ft">{head}{"".join(out)}</div>'