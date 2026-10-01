"""Presentation mode: one photo in, a large result, its recovery loop and this session's recent predictions.
Uses the same classifier call as "I have waste"; it saves nothing and leaves that page's form alone."""

import time

import streamlit as st

import visuals as vz
from common import classify_bytes, get_classifier, log_classification, material_color, waste_label
from ml.classifier import feedback_memory
from ml.classifier.predict import combine_predictions

ss = st.session_state
ss.setdefault("present_seen", set())

with st.container(key="present", gap="medium"):
    st.html('<h1 class="mm-present-h">Live classification</h1>')
    left, right = st.columns([4, 8], gap="large")
    with left:
        photos = st.file_uploader("Photo of the waste", type=["jpg", "jpeg", "png", "webp"],
                                  accept_multiple_files=True, key="present_photos") or []
        photos = photos[:4]
        if photos:
            st.html(vz.file_rows(photos))
        st.caption("Several photos are treated as one item seen from different angles.")

    with right:
        slot = st.empty()
        if not photos:
            slot.html('<div class="mm-empty mm-gridbg" style="min-height:22rem"><b>Drop a photo to start</b>'
                      "<p>The model names the material, shows how sure it is, and where that material goes next."
                      "</p></div>")
            pred = None
        else:
            classify, err = get_classifier()
            sig = "|".join(f.file_id for f in photos)
            fresh = sig not in ss.present_seen
            pred = None
            if err:
                slot.html('<div class="mm-empty"><b>The classifier is not available</b></div>')
            else:
                started = time.monotonic()
                if fresh:
                    slot.html(vz.scan_html(photos))
                try:
                    mv = feedback_memory.version()
                    results = [classify_bytes(f.getvalue(), mv) for f in photos]
                    pred = combine_predictions(results) if len(results) > 1 else results[0]
                except (ValueError, TypeError):
                    slot.html('<div class="mm-empty"><b>Couldn\'t read this image</b><p>Try a JPG, PNG or WEBP photo.'
                              "</p></div>")
                if pred:
                    if fresh:
                        time.sleep(max(0.0, 1.2 - (time.monotonic() - started)))
                        ss.present_seen.add(sig)
                        log_classification(pred, len(photos), "Present")
                    sub = (f"Sub-type {vz.sub_label(pred['sub_type'])}"
                           if pred.get("sub_type") and pred["sub_type"] != pred["label"] and not pred.get("is_unknown")
                           else "")
                    with slot.container(gap="small"):
                        st.html(vz.result_html(pred, reveal=fresh, size="lg", sub_note=sub))
                        if not pred.get("is_unknown"):
                            st.html(vz.facts_html(pred["label"]))

    if pred and not pred.get("is_unknown"):
        with st.container(border=True, key="card_ploop"):
            st.html(f'<div class="mm-section">Where does {vz.esc(waste_label(pred["label"]).lower())} go next?</div>')
            st.html(vz.loop_html(pred["label"]))

    log = ss.get("class_log", [])
    if log:
        st.html('<div class="mm-section">This session</div>')
        chips = "".join(
            f'<span class="mm-tag" style="--c:{material_color(r["waste_type"])}"><i></i>'
            f'{vz.esc(waste_label(r["waste_type"]) if r["status"] != "unknown" else "Unknown")} '
            f'{r["confidence"]:.0%}</span>' for r in reversed(log[-10:]))
        st.html(f"<div>{chips}</div>")
