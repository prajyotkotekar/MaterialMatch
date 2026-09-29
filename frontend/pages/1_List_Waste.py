import hashlib

import pandas as pd
import streamlit as st

from backend import feedback_store
from common import (ALL_WASTE_TYPES, CO2E_EXPLAINER, SESSION_ID_START, all_listings, carbon_label,
                    classify_bytes, fmt_kg, get_classifier, kpi, recycler_card, results_map, section_title,
                    recycler_profiles, seller_display, sub_label, waste_label, zone_coords, zones)
from ml.embeddings import load_listings
from ml.carbon import co2e_saved, load_factors
from ml.classifier import feedback_memory
from ml.classifier.predict import UNKNOWN, combine_predictions
from ml.matcher import NO_RECYCLER_MESSAGE, load_recyclers, match_recyclers
from ml.taxonomy import photo_hierarchy, to_market_subtype

MAX_PHOTOS = 4
SAME, DIFFERENT = "same", "different"
WASTE_TYPES = ALL_WASTE_TYPES
FEEDBACK_CHOICES = [*WASTE_TYPES, feedback_store.OTHER]

ss = st.session_state
ss.setdefault("wt", WASTE_TYPES[0])
ss.setdefault("sub", "unknown")
ss.setdefault("qty", 0)  # no pre-filled amount: the user enters their own quantity
ss.setdefault("loc", zones().index[0])
ss.setdefault("photo_mode", SAME)
ss.setdefault("show_results", False)
ss.setdefault("pred_cache", {})


def photo_subs(waste_type: str) -> list[str]:
    """Photo sub-types that say more than the waste type itself (glass/glass, plastic/plastic excluded)."""
    return [s for s in photo_hierarchy().get(waste_type, []) if s != waste_type]


def sub_options(waste_type: str) -> list[str]:
    """Marketplace sub-types (with carbon factors) + photo sub-types, one entry per material."""
    market = load_factors().query("waste_type == @waste_type").index.tolist()
    photo = [to_market_subtype(waste_type, s) for s in photo_subs(waste_type)]
    return ["unknown"] + market + [s for s in photo if s not in market]


def form_keys(item: str) -> tuple[str, str, str]:
    """Widget keys for one item. The single-item form keeps the original wt/sub/qty keys."""
    return ("wt", "sub", "qty") if item == "main" else (f"wt_{item}", f"sub_{item}", f"qty_{item}")


def fb_label(choice: str) -> str:
    return "Other material (not in the list)" if choice == feedback_store.OTHER else waste_label(choice)


def pred_text(pred: dict | None) -> str:
    if not pred:
        return ""
    if pred.get("is_unknown"):
        return f"Unknown material {pred['confidence']:.0%}"
    sub = f" / {sub_label(pred['sub_type'])}" if pred.get("sub_type") and pred["sub_type"] != pred["label"] else ""
    return f"{waste_label(pred['label'])}{sub} {pred['confidence']:.0%}"


SUBTYPE_AUTOFILL_MIN = 0.85  # the sub-type is filled in automatically only at or above this confidence


def predicted_sub(pred: dict) -> str | None:
    """The predicted sub-type as a value of the sub-type dropdown (None if the dropdown has no such entry).

    Includes materials whose only sub-type is the type itself (glass, metal, paper, organic) when the
    carbon table lists it - those were previously never filled in."""
    st_ = pred.get("sub_type")
    if not st_ or pred["label"] not in WASTE_TYPES:
        return None
    sub = to_market_subtype(pred["label"], st_)
    return sub if sub in sub_options(pred["label"]) and sub != "unknown" else None


def sub_confidence(pred: dict) -> float:
    """Confidence shown next to the sub-type: P(sub | type); for one-sub-type materials the type confidence."""
    if photo_subs(pred["label"]):
        return float(pred.get("subtype_confidence") or 0)
    return float(pred.get("confidence") or 0)


def form_subtype(pred: dict) -> str:
    """Form value for the predicted sub-type: only at >= SUBTYPE_AUTOFILL_MIN, else 'Not sure' (user picks)."""
    sub = predicted_sub(pred)
    return sub if sub and sub_confidence(pred) >= SUBTYPE_AUTOFILL_MIN else "unknown"


def load_listing(waste_id: int) -> None:
    row = all_listings().set_index("waste_id").loc[waste_id]
    ss.wt, ss.qty, ss.loc = row["waste_type"], int(row["quantity_kg"]), row["location_name"]
    ss.sub = row["sub_type"] if row["sub_type"] in sub_options(row["waste_type"]) else "unknown"
    ss.selected_listing_id = waste_id
    ss.show_results = True


def save_feedback(sig: str) -> None:
    info = ss.pred_cache.get(sig)
    answer = ss.get(f"fb_{sig}")
    if info is None or answer is None:
        return
    pred = info["pred"]
    predicted, predicted_sub = pred["label"], pred.get("sub_type")
    if answer == "Yes":
        correct, actual, actual_sub = True, None, None
    else:
        actual = ss.get(f"fbfix_{sig}")
        if actual is None:  # wait until the user picks the real material
            ss.pop(f"fbsaved_{sig}", None)
            return
        actual_sub = ss.get(f"fbsub_{sig}")
        if actual_sub not in photo_hierarchy().get(actual, []):
            actual_sub = None
        if actual_sub is None and len(photo_hierarchy().get(actual, [])) == 1:
            actual_sub = photo_hierarchy()[actual][0]  # glass -> glass, metal -> metal, ...
        correct = actual == predicted and (actual_sub is None or actual_sub == predicted_sub)
    rec = feedback_store.save_feedback(
        info["images"], predicted, pred["confidence"], correct, actual, info["mode"],
        probabilities=pred.get("top_k_subtypes") or pred["top_k"], source="streamlit",
        predicted_sub_type=predicted_sub, actual_sub_type=actual_sub)
    ss[f"fbsaved_{sig}"] = rec["actual_label"]
    ss[f"fbsavedsub_{sig}"] = rec.get("actual_sub_type")
    if rec["actual_label"] in WASTE_TYPES:  # the confirmed/corrected material drives the form
        ss[info["wt_key"]] = rec["actual_label"]
        sub = rec.get("actual_sub_type")
        ss[info["sub_key"]] = (to_market_subtype(rec["actual_label"], sub)
                               if sub and sub != rec["actual_label"] else "unknown")


def prediction_block(sig: str, pred: dict | None, error: str | None, per_photo: list[dict] | None) -> None:
    if error:
        st.markdown(f":red-badge[:material/broken_image: {error}]")
        st.caption("Try a JPG, PNG or WEBP photo, or choose the waste type manually.")
        return
    if pred is None:
        st.markdown(":gray-badge[:material/model_training: Model not ready]")
        st.caption("Choose the waste type manually for now.")
        return
    about = ("AI suggestion from the photo. The model knows 9 material types and 27 sub-types "
             "(e.g. e-waste → keyboard, construction → concrete). 'Please confirm' = the photo looks "
             "unfamiliar or two materials are close; 'Other / unknown' = both, or very unfamiliar. Photos "
             "similar to ones users corrected are adjusted using that feedback. Always check it.")
    if pred.get("is_unknown"):
        g = pred.get("best_guess") or {}
        st.markdown(":gray-badge[:material/help: Other / unknown material]", help=about)
        st.caption(f"Closest match: {waste_label(g.get('waste_type'))} {g.get('confidence', 0):.0%}. "
                   f"Not reliable enough to use ({pred.get('unknown_reason') or 'unfamiliar photo'}). "
                   "Pick the material yourself if you know it.")
    else:
        badge = ("green-badge[:material/auto_awesome: Detected]" if pred["is_confident"]
                 else "orange-badge[:material/help: Please confirm]")
        st.markdown(f":{badge} **{waste_label(pred['label'])}** · {pred['confidence']:.0%} confidence", help=about)
        if pred.get("status") == "confirm" and pred.get("confirm_reason"):
            if "low-resolution" in pred["confirm_reason"]:
                why = "Small, low-resolution photo. The model over-predicts e-waste for these. Please check it."
            elif "different" in pred["confirm_reason"]:
                why = "Unfamiliar-looking photo. The best guess is filled in, please check it."
            elif len(pred["top_k"]) > 1:
                second = pred["top_k"][1]
                why = f"Close call with {waste_label(second['label'])} {second['confidence']:.0%}. Please check it."
            else:
                why = "Close call. Please check it."
            st.caption(f":orange[:material/info:] {why}")
        psub = predicted_sub(pred)
        if psub:
            sub_conf = sub_confidence(pred)
            auto = sub_conf >= SUBTYPE_AUTOFILL_MIN
            st.markdown(f"Sub-type: **{sub_label(pred['sub_type'])}** · {sub_conf:.0%}"
                        + (" · filled in" if auto else " · please choose it yourself"),
                        help=f"Confidence of the sub-type within the detected material. At {SUBTYPE_AUTOFILL_MIN:.0%} "
                             "or more it is filled into the form automatically; below that the form keeps "
                             "'Not sure' so the carbon estimate isn't based on a guess.")
    if pred.get("memory"):
        m, g = pred["memory"], (pred["memory"].get("model_guess") or {})
        st.caption(f":material/history: Adjusted using {m['n_similar']} similar photo"
                   f"{'s' if m['n_similar'] > 1 else ''} confirmed by users",
                   help=f"Without that feedback the model would say {waste_label(g.get('waste_type'))} "
                        f"{g.get('confidence', 0):.0%}. Feedback weight {m['weight']:.0%} (closer photos count more)."
                   if g else "Confirmed feedback photos that look very similar were used.")
    others = " · ".join(f"{waste_label(t['label'])} {t['confidence']:.0%}" for t in pred["top_k"][1:3])
    if per_photo:
        n = len(per_photo)
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.caption(f"{pred['agreement']} of {n} photos agree · {others}")
            with st.popover("Per photo", type="tertiary", icon=":material/photo_library:"):
                for i, r in enumerate(per_photo, 1):
                    st.markdown(f"Photo {i}: **{pred_text(r)}**")
                st.caption("Combined by averaging each sub-type's probability across the photos.")
    else:
        st.caption(others)

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption("Was this prediction correct?")
        st.segmented_control("Was this prediction correct?", ["Yes", "No"], key=f"fb_{sig}",
                             on_change=save_feedback, args=(sig,), label_visibility="collapsed")
    if ss.get(f"fb_{sig}") == "No":
        c1, c2 = st.columns(2, gap="small")
        c1.selectbox("Actual waste type", FEEDBACK_CHOICES, index=None, key=f"fbfix_{sig}",
                     placeholder="Select the material", format_func=fb_label,
                     on_change=save_feedback, args=(sig,))
        subs = photo_subs(ss.get(f"fbfix_{sig}") or "")
        if subs:
            c2.selectbox("Actual sub-type", subs, index=None, key=f"fbsub_{sig}", placeholder="Not sure",
                         format_func=sub_label, on_change=save_feedback, args=(sig,))
    saved = ss.get(f"fbsaved_{sig}")
    if saved == feedback_store.OTHER:
        st.caption(":material/info: Saved, thanks. MaterialMatch can't match this material yet: "
                   "remove the photo or pick the closest supported type.")
    elif saved:
        st.caption(":material/check: Saved, thanks. Similar photos uploaded from now on will use this answer; "
                   "it also goes into the next retraining.")

# ---------------------------------------------------------------- page
st.markdown("## Match your waste to the right recycler")
st.caption("Snap a photo, confirm the material, and get ranked recyclers with an estimate of the CO₂e saved.")

upload_col, details_col = st.columns([5, 7], gap="medium")

with upload_col:
    with st.container(border=True, key="card_upload", gap="small"):
        section_title("Waste photos", "add_a_photo")
        st.caption(f"Up to {MAX_PHOTOS} photos · AI detects the material → you confirm it")
        photos = st.file_uploader("Upload waste photos", type=["jpg", "jpeg", "png", "webp"],
                                  accept_multiple_files=True, key="photos", label_visibility="collapsed") or []
        if len(photos) > MAX_PHOTOS:
            st.caption(f":orange[:material/warning:] Using the first {MAX_PHOTOS} of {len(photos)} photos.")
            photos = photos[:MAX_PHOTOS]

        mode = SAME
        if len(photos) >= 2:
            mode = st.segmented_control(
                "What do these photos show?", [SAME, DIFFERENT], key="photo_mode", width="stretch", required=True,
                format_func={SAME: ":material/filter_center_focus: One item, several angles",
                             DIFFERENT: ":material/category: Different items"}.get,
                help="One item: the photos are combined into a single prediction and one listing. "
                     "Different items: each photo becomes its own item in one submission.") or SAME

        groups = ([(p.file_id, [p]) for p in photos] if mode == DIFFERENT
                  else [("main", photos)])
        store_mode = ("different_items" if mode == DIFFERENT else "same_item" if len(photos) > 1 else "single")
        classify, model_err = get_classifier() if photos else (None, None)

        item_preds: dict[str, dict | None] = {}
        item_sigs: dict[str, str] = {}
        for item, files in groups:
            wt_key, sub_key, qty_key = form_keys(item)
            ss.setdefault(sub_key, "unknown")
            ss.setdefault(qty_key, 0)
            if not files:
                ss.setdefault(wt_key, WASTE_TYPES[0])
                continue
            sig = hashlib.sha1("|".join(f.file_id for f in files).encode()).hexdigest()[:12]
            item_sigs[item] = sig
            pred, error, per_photo = None, None, None
            if sig in ss.pred_cache:  # keep the prediction the user is answering, even after new feedback
                pred = ss.pred_cache[sig]["pred"]
                per_photo = pred.get("per_image") if len(files) > 1 else None
            elif not model_err:
                try:
                    mv = feedback_memory.version()
                    results = [classify_bytes(f.getvalue(), mv) for f in files]
                    pred = combine_predictions(results) if len(results) > 1 else results[0]
                    per_photo = pred.get("per_image") if len(results) > 1 else None
                except (ValueError, TypeError):
                    error = "Couldn't read this image" if len(files) == 1 else "Couldn't read one of the photos"
            if pred:
                ss.pred_cache[sig] = {"images": [f.getvalue() for f in files], "pred": pred,
                                      "mode": store_mode, "wt_key": wt_key, "sub_key": sub_key}
                if ss.get(f"applied_{item}") != sig:  # new photo(s): pre-fill type + sub-type once
                    ss[f"applied_{item}"] = sig
                    if not ss.get(f"fbsaved_{sig}") and not pred.get("is_unknown") and pred["label"] in WASTE_TYPES:
                        ss[wt_key] = pred["label"]
                        ss[sub_key] = form_subtype(pred)
            ss.setdefault(wt_key, WASTE_TYPES[0])
            item_preds[item] = pred

            side_by_side = mode == DIFFERENT or len(files) == 1
            with st.container(horizontal=True, vertical_alignment="top", gap="small"):
                for f in files:
                    st.image(f, width=72 if side_by_side else 64)
                if side_by_side:
                    with st.container(gap="xxsmall"):
                        prediction_block(sig, pred, error, None)
            if not side_by_side:
                prediction_block(sig, pred, error, per_photo)

        if not photos:
            st.caption(":material/info: Optional: skip it if you already know the material. "
                       "Several photos of one item improve the prediction.")

with details_col:
    with st.container(border=True, key="card_details", gap="small"):
        multi = len(groups) > 1
        with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
            section_title(f"Waste details · {len(groups)} items" if multi else "Waste details", "inventory_2")
            with st.popover("Use a listing", icon=":material/list_alt:", type="tertiary", disabled=multi,
                            help="Switch to a single item to load an existing listing." if multi else None):
                listings = all_listings()
                # no internal waste_id in the label (as on "I need feedstock"): material, amount, place, seller
                labels = {int(r.waste_id): f"{sub_label(r.sub_type)} · {r.quantity_kg:,} kg · {r.location_name} · "
                                           + (f"{r.seller_name} (yours)" if r.source != "Demo"
                                              else seller_display(r.seller_name))
                          for r in listings.itertuples()}
                pick = st.selectbox("Marketplace listing", list(labels), format_func=labels.get)
                st.button("Match this listing", icon=":material/bolt:", on_click=load_listing, args=(pick,),
                          width="stretch")

        if not multi:
            c1, c2 = st.columns(2, gap="small")
            c1.selectbox("Waste type", WASTE_TYPES, key="wt", format_func=waste_label)
            opts = sub_options(ss.wt)
            if ss.sub not in opts:
                ss.sub = "unknown"
            c2.selectbox("Sub-type", opts, key="sub", format_func=sub_label)
            c3, c4 = st.columns(2, gap="small")
            c3.number_input("Quantity (kg)", min_value=0, step=100, key="qty")
            c4.selectbox("Location", zones().index, key="loc")
        else:
            for i, (item, files) in enumerate(groups):
                wt_key, sub_key, qty_key = form_keys(item)
                vis = "visible" if i == 0 else "collapsed"
                c0, c1, c2, c3 = st.columns([1, 3, 3, 2], gap="small", vertical_alignment="bottom")
                c0.image(files[0], width=40)
                c1.selectbox("Waste type", WASTE_TYPES, key=wt_key, format_func=waste_label, label_visibility=vis)
                opts = sub_options(ss[wt_key])
                if ss[sub_key] not in opts:
                    ss[sub_key] = "unknown"
                c2.selectbox("Sub-type", opts, key=sub_key, format_func=sub_label, label_visibility=vis)
                c3.number_input("Quantity (kg)", min_value=0, step=100, key=qty_key, label_visibility=vis)
            st.selectbox("Location (all items)", zones().index, key="loc")

        # Both actions are always visible; the Publish popover is added below, once `items` exists.
        actions = st.container(horizontal=True, gap="small")
        with actions:
            if st.button("Find matching recyclers", type="primary", icon=":material/travel_explore:",
                         width="stretch"):
                ss.show_results = True

# Always built from the CURRENT form values: results and publishing can't use stale input.
items = []
for item, files in groups:
    wt_key, sub_key, qty_key = form_keys(item)
    sub = ss[sub_key]
    sig = item_sigs.get(item)
    items.append({"item": item, "waste_type": ss[wt_key], "sub_type": None if sub == "unknown" else sub,
                  "quantity_kg": float(ss[qty_key]), "photos": [f.name for f in files],
                  "prediction": item_preds.get(item), "feedback": ss.get(f"fbsaved_{sig}") if sig else None,
                  "feedback_sub": ss.get(f"fbsavedsub_{sig}") if sig else None})
loc = ss.loc
lat, lon = zone_coords(loc)
for it in items:
    it["label"] = f"{waste_label(it['waste_type'])} · {sub_label(it['sub_type'])}"
    it["carbon"] = co2e_saved(it["sub_type"], it["quantity_kg"], waste_type=it["waste_type"])

# Pickup / drop-off choices = ONLY the recyclers in the database that accept a listed waste type.
_rec = load_recyclers()
_wanted = {i["waste_type"] for i in items}
_rec = _rec[_rec["waste_type_list"].apply(lambda ts: bool(_wanted & set(ts)))].sort_values("recycler_id")
pickup_profiles = recycler_profiles()
pickup_sites = [int(r) for r in _rec["recycler_id"]]
pickup_labels = {int(r.recycler_id): f"{r.name}, {pickup_profiles.get(int(r.recycler_id), {}).get('area', '')}"
                 for r in _rec.itertuples()}


def valid_phone(text: str) -> bool:
    """8-15 digits, optionally with a leading + and spaces, dashes or brackets."""
    digits = [c for c in text if c.isdigit()]
    return 8 <= len(digits) <= 15 and all(c.isdigit() or c in " +-()" for c in text.strip()) \
        and "+" not in text.strip()[1:]


def price_help(quality: str) -> str:
    """Help for the price field, with what demo listings of the same material and quality ask (indicative)."""
    text = ("Buyers on “I need feedstock” see this price and what their quantity would cost. "
            "Leave 0 if it isn't decided yet.")
    df = load_listings()
    ranges = []
    for sub in dict.fromkeys(i["sub_type"] for i in items if i["sub_type"]):
        p = df[(df["sub_type"] == sub) & (df["quality"] == quality)]["price_per_kg"]
        if len(p):
            lo, hi = int(p.min()), int(p.max())
            ranges.append(f"{sub_label(sub).lower()} ₹{lo}/kg" if lo == hi else f"{sub_label(sub).lower()} ₹{lo}–{hi}/kg")
    if ranges:
        text += (f" For reference, demo listings in {quality} quality ask: {'; '.join(ranges)} "
                 "(demo values, not a market quote).")
    return text


def publish_type_changed(wt_key: str, sub_key: str, pub_wt: str) -> None:
    ss[wt_key] = ss[pub_wt]
    ss[sub_key] = "unknown"


def publish_sub_changed(sub_key: str, pub_sub: str) -> None:
    ss[sub_key] = ss[pub_sub]


def publish_qty_changed(qty_key: str, pub_qty: str) -> None:
    ss[qty_key] = ss[pub_qty]


with actions:  # next to "Find matching recyclers", available from the start
    with st.popover(f"Publish {len(items)} listings" if len(items) > 1 else "Publish as listing",
                    icon=":material/storefront:", width="stretch"):
        multi_pub = len(groups) > 1
        quality_opts = ["good", "fair", "poor"]
        quality = "fair"
        # 1) Material: type / sub-type / quantity mirror the "Waste details" form (edit either, both stay in step).
        for n, (item, _files) in enumerate(groups):
            wt_key, sub_key, qty_key = form_keys(item)
            pub_wt, pub_sub, pub_qty = f"pub_wt_{item}", f"pub_sub_{item}", f"pub_qty_{item}"
            ss[pub_wt] = ss[wt_key]
            opts = sub_options(ss[wt_key])
            ss[pub_sub] = ss[sub_key] if ss[sub_key] in opts else "unknown"
            ss[pub_qty] = int(ss[qty_key])
            vis = "visible" if n == 0 else "collapsed"
            suffix = f" (item {n + 1})" if multi_pub else ""
            if multi_pub:
                p1, p2, p3 = st.columns([3, 3, 2], gap="small")
            else:
                (p1, p2), (p3, p4, p5) = st.columns(2, gap="small"), st.columns(3, gap="small")
            p1.selectbox("Waste type" + suffix, WASTE_TYPES, key=pub_wt, format_func=waste_label,
                         on_change=publish_type_changed, args=(wt_key, sub_key, pub_wt), label_visibility=vis)
            p2.selectbox("Sub-type", opts, key=pub_sub, format_func=sub_label, on_change=publish_sub_changed,
                         args=(sub_key, pub_sub), label_visibility=vis)
            p3.number_input("Quantity (kg)", min_value=0, step=100, key=pub_qty, on_change=publish_qty_changed,
                            args=(qty_key, pub_qty), label_visibility=vis)
            if not multi_pub:
                quality = p4.selectbox("Quality", quality_opts, index=1, format_func=str.capitalize)
                price = p5.number_input("Price (₹/kg)", min_value=0, value=0, step=1, key="pub_price",
                                        help=price_help(quality))
        # 2) Where the buyer collects / delivers: only recyclers that accept the chosen type(s).
        pickup_id = st.selectbox(
            "Pickup / drop-off location", pickup_sites, index=None, placeholder="Choose a recycler location",
            format_func=lambda i: pickup_labels[i],
            help="Only recyclers in the network (data/raw/recyclers.csv) that accept your waste type are "
                 "listed, so the list changes with the waste type. Buyers see this location.")
        # 3) Quality / price (several photos share them), then who is selling and how buyers reach them.
        if multi_pub:
            q1, q2 = st.columns(2, gap="small")
            quality = q1.selectbox("Quality", quality_opts, index=1, format_func=str.capitalize)
            price = q2.number_input("Price (₹/kg)", min_value=0, value=0, step=1, key="pub_price",
                                    help=price_help(quality))
        s1, s2 = st.columns(2, gap="small")
        seller = s1.text_input("Business name", max_chars=120)
        contact = s2.text_input("Contact number", max_chars=20, placeholder="e.g. +91 98450 12345",
                                help="Shown to buyers on the “I need feedstock” page.")
        description = st.text_area("Description (optional)", max_chars=2000, height=80,
                                   placeholder="e.g. clean cotton offcuts, baled, weekly supply")
        if st.button("Publish", type="primary", icon=":material/storefront:", width="stretch"):
            if any(i["quantity_kg"] <= 0 for i in items):
                st.error("Please enter a quantity greater than 0 kg.")
            elif not seller.strip():
                st.error("Please enter a business name.")
            elif not valid_phone(contact):
                st.error("Please enter a valid contact number (8–15 digits, e.g. +91 98450 12345).")
            elif pickup_id is None:
                st.error("Please choose a pickup / drop-off location.")
            else:
                site = pickup_profiles.get(pickup_id, {})
                submission = f"S{len({l['submission_id'] for l in ss.my_listings}) + 1:03d}"
                for it in items:
                    new_id = SESSION_ID_START + len(ss.my_listings) + 1
                    pred = it["prediction"]
                    ss.my_listings.append({
                        "waste_id": new_id, "submission_id": submission,
                        "waste_type": it["waste_type"], "sub_type": it["sub_type"] or "unknown",
                        "quantity_kg": int(it["quantity_kg"]), "quality": quality,
                        "location_name": loc, "location_lat": lat, "location_lon": lon,
                        "pickup_location": pickup_labels[pickup_id],
                        "pickup_address": site.get("address", ""),
                        "seller_name": seller.strip(), "seller_contact": " ".join(contact.split()),
                        "price_per_kg": int(price),
                        "description": description.strip(),
                        "co2e_saved_kg": it["carbon"]["co2e_saved_kg"],
                        "carbon_is_proxy": it["carbon"]["is_proxy"], "source": "This session",
                        "photos": ", ".join(it["photos"]),
                        "photo_prediction": pred_text(pred),
                        "prediction_feedback": (fb_label(it["feedback"]) + (
                            f" / {sub_label(it['feedback_sub'])}" if it["feedback_sub"]
                            and it["feedback_sub"] != it["feedback"] else "")) if it["feedback"] else "",
                    })
                ss.selected_listing_id = SESSION_ID_START + len(ss.my_listings)
                ss.just_published = submission
                st.success(f"Published as submission {submission}. See “Your published listings” "
                           "at the bottom of the page.", icon=":material/check_circle:")
                st.toast(f"Submission {submission} published ({len(items)} listing"
                         f"{'s' if len(items) > 1 else ''})", icon=":material/check_circle:")

no_quantity = any(i["quantity_kg"] <= 0 for i in items)
if ss.show_results and no_quantity:
    st.warning("Enter the quantity (kg) of your waste to see matching recyclers and the CO₂e estimate.",
               icon=":material/scale:")

if ss.show_results and not no_quantity:
    st.space("small")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        with st.container(gap="xxsmall"):
            st.markdown("### Recommended recyclers")
            if len(items) == 1:
                it = items[0]
                st.caption(f"{it['quantity_kg']:,.0f} kg of {waste_label(it['waste_type']).lower()}"
                           f"{' · ' + sub_label(it['sub_type']) if it['sub_type'] else ''} from {loc}")
            else:
                total_kg = sum(i["quantity_kg"] for i in items)
                total_co2 = sum(i["carbon"]["co2e_saved_kg"] or 0 for i in items)
                no_factor = sum(i["carbon"]["co2e_saved_kg"] is None for i in items)
                st.caption(f"{len(items)} items · {fmt_kg(total_kg)} from {loc} · "
                           f"≈ {total_co2:,.0f} kg CO₂e saved in total"
                           + (f" ({no_factor} item{'s' if no_factor > 1 else ''} without a carbon factor)"
                              if no_factor else ""))

    sel = items[0]
    if len(items) > 1:
        keys = [i["item"] for i in items]
        if ss.get("result_item") not in keys:
            ss.result_item = keys[0]
        chosen = st.segmented_control("Show recyclers for", keys, key="result_item", width="stretch", required=True,
                                      format_func=lambda k: next(i["label"] for i in items if i["item"] == k))
        sel = next(i for i in items if i["item"] == (chosen or keys[0]))

    matches = match_recyclers(sel["waste_type"], sel["quantity_kg"], lat, lon, sub_type=sel["sub_type"], top_k=5)
    carbon = sel["carbon"]
    n_compatible = int(load_recyclers()["waste_type_list"].apply(lambda ts: sel["waste_type"] in ts).sum())

    has_factor = carbon["factor_available"]
    with st.container(horizontal=True, gap="small", key="kpis_waste"):
        kpi("co2", "Estimated CO₂e saved", f"{carbon['co2e_saved_kg']:,.0f} kg" if has_factor else "n/a",
            CO2E_EXPLAINER if has_factor else "No carbon factor for this material yet",
            help="Quantity × the carbon factor for this sub-type (the waste type's average if the sub-type "
                 "is 'Not sure'), vs. making the material from virgin resources. Materials without a "
                 "sourced factor show no estimate rather than a borrowed one.")
        kpi("factor", "Carbon factor (kg CO₂e/kg)", f"{carbon['co2e_saved_kg_per_kg']:.3f}" if has_factor else "n/a",
            f":{'orange' if carbon['is_proxy'] else 'green'}-badge[{carbon_label(carbon['is_proxy'])}]"
            if has_factor else ":gray-badge[Factor unavailable]",
            help=f"{carbon['basis']} (source: {carbon['source']})" if has_factor else carbon["basis"])
        kpi("compat", "Compatible recyclers", f"{n_compatible}", "Accept this waste type",
            help="Sample recycler profiles in the network that list this waste type.")
        kpi("best", "Best match", f"{matches[0]['score']:.0%}" if matches else "n/a",
            matches[0]["name"] if matches else "No recycler profile yet")
    if carbon["note"]:
        st.caption(f":material/info: {carbon['note']}")

    if matches:
        list_col, map_col = st.columns([7, 5], gap="medium")
        with list_col:
            for rank, m in enumerate(matches):
                recycler_card(m, rank, sel["quantity_kg"], sel["waste_type"])
        with map_col:
            with st.container(border=True, key="card_map", gap="small"):
                section_title("Bengaluru", "map")
                results_map((lat, lon),
                            [{"lat": m["lat"], "lon": m["lon"], "name": m["name"],
                              "detail": f"#{i + 1} · {m['score']:.0%} match · {m['distance_km']:.1f} km"}
                             for i, m in enumerate(matches)],
                            origin_label=f"Your waste · {loc}")
    else:
        st.info(NO_RECYCLER_MESSAGE, icon=":material/search_off:")

if ss.my_listings:
    with st.expander(f"Your published listings ({len(ss.my_listings)})", icon=":material/storefront:",
                     expanded=bool(ss.pop("just_published", None))):
        published = pd.DataFrame(ss.my_listings[::-1])
        published["price_per_kg"] = published["price_per_kg"].where(published["price_per_kg"] > 0)  # 0 = not set
        st.dataframe(
            published.reindex(columns=[
                "submission_id", "waste_id", "waste_type", "sub_type", "quantity_kg", "location_name",
                "seller_name", "seller_contact", "pickup_location", "pickup_address", "quality", "price_per_kg",
                "photo_prediction", "prediction_feedback", "co2e_saved_kg"]),
            hide_index=True,
            column_config={
                "submission_id": "Submission", "waste_id": "Listing",
                "waste_type": st.column_config.TextColumn("Type"), "sub_type": "Sub-type",
                "quantity_kg": st.column_config.NumberColumn("Quantity", format="%d kg"),
                "location_name": "Location", "seller_name": "Business", "seller_contact": "Contact",
                "pickup_location": "Pickup / drop-off", "pickup_address": "Address", "quality": "Quality",
                "price_per_kg": st.column_config.NumberColumn("Price", format="₹%d/kg"),
                "photo_prediction": "Photo AI", "prediction_feedback": "Confirmed as",
                "co2e_saved_kg": st.column_config.NumberColumn("CO₂e saved", format="%.0f kg"),
            },
        )
