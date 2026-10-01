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
from ml.classifier.predict import SMALL_SIDE
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
        im = Image.open(io.BytesIO(data)).convert("RGB")
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


def _segments(colors: list[str], gap: float) -> str:
    """conic-gradient stops: equal segments with a transparent gap between them."""
    step = 360 / len(colors)
    stops, end = [], 0.0
    for i, c in enumerate(colors):
        a0, a1 = i * step + gap / 2, (i + 1) * step - gap / 2
        stops.append(f"transparent {end:.1f}deg {a0:.1f}deg, {c} {a0:.1f}deg {a1:.1f}deg")
        end = a1
    return "conic-gradient(" + ", ".join(stops) + f", transparent {end:.1f}deg 360deg)"


def hero_html() -> str:
    """Hero: headline + the sorting wheel (the 9 materials the model knows, swept by a scan line).
    Drawn with CSS, because st.html strips inline SVG."""
    types = list(taxonomy.WASTE_TYPES)
    n_sub = sum(len(v) for v in taxonomy.photo_hierarchy().values())
    step = 360 / len(types)
    labels = "".join(
        f'<span class="mm-wheel-l" style="left:{x:.1f}%;top:{y:.1f}%">{esc(waste_label(t))}</span>'
        for i, t in enumerate(types) for x, y in [_polar((i + 0.5) * step, 49)])
    flow = "".join(f"<li><b>{i:02d}</b>{esc(s)}</li>" for i, s in
                   enumerate(["Photo in", "Model reads it", "Material named", "Recycler found"], 1))
    wheel = _segments([WASTE_COLOR[t] for t in types], 3)
    return f"""
<section class="mm-hero">
  <div class="mm-hero-copy">
    <div class="mm-status"><i></i>Photo classifier: {len(types)} materials, {n_sub} sub-types</div>
    <h1 class="mm-hero-h">Your waste is someone's raw material.</h1>
    <p class="mm-hero-p">Photograph it. The model names the material, and MaterialMatch finds the Bengaluru
    recyclers who can use it, with the CO₂e that saves compared with new material.</p>
    <ol class="mm-flow">{flow}</ol>
  </div>
  <div class="mm-hero-art" aria-hidden="true">
    <div class="mm-wheel" style="background:{wheel}"></div>
    <div class="mm-wheel-in"></div>
    <div class="mm-sweep"></div>
    {labels}
    <div class="mm-hero-core"><b>{n_sub}</b><span>sub-types it can tell apart</span></div>
  </div>
</section>"""


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

def _ring(pct: float, color: str, size: str) -> str:
    """Confidence dial (CSS conic-gradient); --p drives both the arc and the number, so they animate together."""
    p = max(0.0, min(1.0, pct))
    return (f'<div class="mm-ring mm-ring-{size}" style="--p:{round(p * 100)};--c:{color}"><div class="mm-ring-dial"></div>'
            f'<span class="mm-ring-n"></span><span class="mm-ring-l">confidence</span></div>')


def result_html(pred: dict, reveal: bool, size: str = "lg", sub_note: str = "") -> str:
    """The identification 'moment': material name, confidence ring, status. sub_note = sub-type line."""
    unknown = pred.get("is_unknown")
    if unknown:
        g = pred.get("best_guess") or {}
        wt, conf = g.get("waste_type"), float(g.get("confidence") or 0)
        name, color = "Unknown material", "#6E776B"
        status = '<span class="mm-chip mm-chip-off">Other / unknown</span>'
        line = f"Closest guess {waste_label(wt)} at {conf:.0%}, not reliable enough to use"
        kicker = "Not identified"
    else:
        wt, conf = pred["label"], float(pred["confidence"])
        name, color = waste_label(wt), material_color(wt)
        status = ('<span class="mm-chip mm-chip-ok">Detected</span>' if pred.get("is_confident")
                  else '<span class="mm-chip mm-chip-warn">Please confirm</span>')
        line = sub_note
        kicker = "Material identified"
    cls = f"mm-res mm-res-{size}" + (" mm-reveal" if reveal else "")
    return (f'<div class="{cls}" style="--c:{color}"><div class="mm-res-band"></div><div class="mm-res-body">'
            f'<div class="mm-res-main"><span class="mm-res-k">{kicker}</span>'
            f'<span class="mm-res-name">{esc(name)}</span>'
            + (f'<span class="mm-res-sub">{esc(line)}</span>' if line else "")
            + f"{status}</div>{_ring(conf, color, size)}</div></div>")


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
    return '<div class="mm-facts">' + "".join(
        f'<div><span>{esc(k)}</span><b>{esc(v)}</b><small>{esc(n)}</small></div>' for k, v, n in cells) + "</div>"


# ---------- circular recovery loop ----------

def loop_html(waste_type: str | None, sub_type: str | None = None) -> str:
    wt = waste_type if waste_type in ROUTES else "plastic"
    _, _, steps = ROUTES[wt]
    color = material_color(wt)
    broken = BROKEN_FROM.get(wt, 99)
    first = sub_label(sub_type) if sub_type else waste_label(wt)
    texts = [f"Your {first.lower()}"] + steps
    # segment i runs from stage i to stage i+1; from `broken` on there is no recovery route
    ring = _segments([color if i < broken else "#3E463C" for i in range(5)], 14)
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
