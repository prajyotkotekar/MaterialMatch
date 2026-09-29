"""
evaluate_classifier.py - Full evaluation of the hierarchical classifier (+ comparison with v1).

    python -m ml.classifier.evaluate_classifier --weights ml/classifier/weights/best_candidate.pt

Writes ml/classifier/reports/<name>/: report.md, metrics.json, confusion_type.csv/png,
confusion_leaf.csv/png, confidence.png, per_image_test.csv.

Measured on the held-out TEST split only:
  * leaf (sub-type) top-1 / top-3, waste-type top-1 / top-3 (type = sum of its leaves)
  * per waste type and per sub-type precision / recall / F1, sub-type accuracy given the right type
  * accuracy per source dataset, confidence distribution and calibration table
  * unknown handling: false-unknown rate on the test split, rejection rate on probes that are
    NOT in the training data (CODD general_w crops, Windows wallpapers + ultralytics sample
    photos, synthetic noise/colour images) - thresholds are never tuned on these probes
  * v1 (4-class) vs v2 on the same test images, excluding images the v1 model trained on
  * open-set experiment with v1: materials v1 never saw (paper, cardboard, glass, metal, food,
    trash) - does the kNN check catch them better than a confidence threshold?
"""

from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from ml.classifier.calibrate_ood import calibrate, list_split, type_probs
from ml.classifier.predict import embed_probs, knn_distance, split_leaf

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "ml" / "classifier" / "weights"
OLD_TYPES = ["construction", "e_waste", "plastic", "textile"]


def source_of(path: Path) -> str:
    s = path.stem
    return "codd" if s.startswith("codd_") else "ewaste_small" if s.startswith("ewaste_small_") else \
        "garbage_v2" if s.startswith("garbage_v2_") else "other"


def load_ood(weights: Path):
    f = weights.with_suffix(".ood.npz")
    if not f.exists():
        return None
    z = np.load(f)
    out = {k: (z[k].astype(np.float32) if k == "bank" else float(z[k])) for k in
           ("bank", "k", "knn_threshold", "msp_threshold", "min_confidence")}
    out["by_type"] = json.loads(str(z["knn_threshold_by_type"])) if "knn_threshold_by_type" in z else {}
    out["far_by_type"] = json.loads(str(z["knn_far_by_type"])) if "knn_far_by_type" in z else {}
    return out


def unknown_mask(P, F, names, cal):
    """Same rule as predict.ood_status (without feedback memory). Returns unknown, knn, conf, confirm."""
    T, types = type_probs(P, names)
    knn = knn_distance(F, cal["bank"], int(cal["k"]))
    pt, conf = T.argmax(1), T.max(1)
    thr = np.array([cal["by_type"].get(types[i], cal["knn_threshold"]) for i in pt])
    far = np.array([cal["far_by_type"].get(types[i], np.inf) for i in pt])
    unfamiliar, very, close = knn > thr, knn > far, conf < cal["msp_threshold"]
    unknown = very | (unfamiliar & close)
    return unknown, knn, conf, ~unknown & (unfamiliar | close)


def auroc(id_scores, ood_scores):
    y = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    return float(roc_auc_score(y, np.r_[id_scores, ood_scores])) if len(ood_scores) else None


def plot_confusion(cm, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(labels)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.42), max(5, n * 0.4)))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5 if n > 12 else 7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"), ax.set_ylabel("true"), ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def probes(seed=0):
    """Images from outside the training data. Returns {name: [PIL or path]}."""
    rng = random.Random(seed)
    out = {}
    crops = []
    xmls = sorted((ROOT / "data" / "Image" / "codd").rglob("*.xml"))
    rng.shuffle(xmls)
    for x in xmls:
        objs = [o for o in ET.parse(x).iter("object") if o.findtext("name") == "general_w"]
        if not objs:
            continue
        img = Image.open(x.with_suffix(".jpg")).convert("RGB")
        for o in objs:
            b = [float(o.find("bndbox").findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            if min(b[2] - b[0], b[3] - b[1]) >= 24:
                crops.append(img.crop(b))
        if len(crops) >= 400:
            break
    out["codd_general_w (excluded label, mostly plastic litter)"] = crops[:400]
    natural = [p for p in Path(r"C:\Windows\Web").rglob("*") if p.suffix.lower() in (".jpg", ".png")]
    natural += list((Path(__import__("ultralytics").__file__).parent / "assets").glob("*.jpg"))
    out["non-waste photos (Windows wallpapers + ultralytics samples)"] = natural
    syn, r = [], np.random.default_rng(seed)
    for i in range(20):
        syn.append(Image.fromarray(r.integers(0, 256, (224, 224, 3), dtype=np.uint8)))
        syn.append(Image.new("RGB", (224, 224), tuple(int(v) for v in r.integers(0, 256, 3))))
        g = np.linspace(0, 255, 224, dtype=np.uint8)
        syn.append(Image.fromarray(np.stack([np.tile(g, (224, 1))] * 3, -1).astype(np.uint8)))
    out["synthetic (noise / solid colour / gradient)"] = syn
    return out


def old_train_stems(old_ds: Path) -> set[str]:
    """garbage_v2 original file stems the v1 model trained or validated on."""
    stems = set()
    for split in ("train", "val"):
        for f in (old_ds / split).rglob("garbage_*.jpg"):
            m = re.match(r"garbage_(.+)_[0-9a-f]{8}$", f.stem)
            if m:
                stems.add(m.group(1))
    return stems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=W / "best_candidate.pt")
    ap.add_argument("--old-weights", type=Path, default=W / "best_v1_4class.pt")
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "processed" / "cls_dataset")
    ap.add_argument("--old-dataset", type=Path, default=ROOT / "data" / "processed" / "cls_dataset_v1_4class")
    ap.add_argument("--out", type=Path, default=ROOT / "ml" / "classifier" / "reports" / "v2_eval")
    ap.add_argument("--skip-old", action="store_true")
    a = ap.parse_args(argv)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    M: dict = {"weights": str(a.weights)}

    # ------------------------------------------------------------------ test split
    test = list_split(a.dataset, "test")
    paths = [f for f, _ in test]
    P, F, names = embed_probs(paths, a.weights, progress="test")
    y_leaf = np.array([names.index(c) for _, c in test])
    T, types = type_probs(P, names)
    y_type = np.array([types.index(split_leaf(c)[0]) for _, c in test])
    pl, pt = P.argmax(1), T.argmax(1)
    top3l = (np.argsort(-P, 1)[:, :3] == y_leaf[:, None]).any(1)
    top3t = (np.argsort(-T, 1)[:, :3] == y_type[:, None]).any(1)
    M["test_images"] = len(test)
    M["n_waste_types"], M["n_sub_types"] = len(types), len(names)
    M["leaf_top1"], M["leaf_top3"] = float((pl == y_leaf).mean()), float(top3l.mean())
    M["type_top1"], M["type_top3"] = float((pt == y_type).mean()), float(top3t.mean())
    tc = T.max(1)

    pr, rc, f1, sup = precision_recall_fscore_support(y_type, pt, labels=range(len(types)), zero_division=0)
    M["per_type"] = {t: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                         "support": int(sup[i])} for i, t in enumerate(types)}
    M["type_macro_f1"] = float(f1.mean())
    pr, rc, f1, sup = precision_recall_fscore_support(y_leaf, pl, labels=range(len(names)), zero_division=0)
    per_leaf = {}
    for i, n in enumerate(names):
        m = y_leaf == i
        right_type = m & (pt == y_type)
        per_leaf[n] = {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                       "support": int(sup[i]),
                       "type_accuracy": float((pt[m] == y_type[m]).mean()),
                       "subtype_acc_given_right_type": float((pl[right_type] == i).mean()) if right_type.any() else None}
    M["per_leaf"] = per_leaf
    M["leaf_macro_f1"] = float(f1.mean())
    src = np.array([source_of(p) for p in paths])
    M["per_source"] = {s: {"n": int((src == s).sum()), "type_top1": float((pt[src == s] == y_type[src == s]).mean()),
                           "leaf_top1": float((pl[src == s] == y_leaf[src == s]).mean())} for s in sorted(set(src))}
    M["per_leaf_source"] = {f"{names[i]} [{s}]": {"n": int(((y_leaf == i) & (src == s)).sum()),
                                                  "leaf_top1": float((pl[(y_leaf == i) & (src == s)] == i).mean())}
                            for i in range(len(names)) for s in sorted(set(src)) if ((y_leaf == i) & (src == s)).sum()}

    cm_t = confusion_matrix(y_type, pt, labels=range(len(types)))
    cm_l = confusion_matrix(y_leaf, pl, labels=range(len(names)))
    np.savetxt(out / "confusion_type.csv", cm_t, fmt="%d", delimiter=",", header=",".join(types), comments="")
    np.savetxt(out / "confusion_leaf.csv", cm_l, fmt="%d", delimiter=",", header=",".join(names), comments="")
    plot_confusion(cm_t, types, out / "confusion_type.png", f"Waste type - test (n={len(test)})")
    plot_confusion(cm_l, names, out / "confusion_leaf.png", f"Sub-type (leaf) - test (n={len(test)})")
    conf_pairs = Counter((types[y_type[i]], types[pt[i]]) for i in np.nonzero(pt != y_type)[0])
    M["top_type_confusions"] = [[f"{a} -> {b}", n] for (a, b), n in conf_pairs.most_common(10)]
    leaf_pairs = Counter((names[y_leaf[i]], names[pl[i]]) for i in np.nonzero(pl != y_leaf)[0])
    M["top_leaf_confusions"] = [[f"{a} -> {b}", n] for (a, b), n in leaf_pairs.most_common(15)]

    bins = [0, .5, .7, .9, .97, .99, 1.0001]
    M["confidence_bins"] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (tc >= lo) & (tc < hi)
        M["confidence_bins"].append({"range": f"{lo:.2f}-{min(hi, 1):.2f}", "n": int(m.sum()),
                                     "accuracy": float((pt[m] == y_type[m]).mean()) if m.any() else None})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist([tc[pt == y_type], tc[pt != y_type]], bins=20, range=(0, 1), label=["correct", "wrong"],
            color=["#10B981", "#EF4444"], log=True)
    ax.set_xlabel("waste-type confidence"), ax.set_ylabel("test images (log)"), ax.legend()
    fig.tight_layout(), fig.savefig(out / "confidence.png", dpi=130), plt.close(fig)

    # ------------------------------------------------------------------ unknown handling
    cal = load_ood(a.weights)
    if cal:
        unk, knn_id, _, confirm = unknown_mask(P, F, names, cal)
        M["unknown"] = {"thresholds": {k: cal[k] for k in ("knn_threshold", "msp_threshold", "min_confidence")},
                        "knn_threshold_by_type": cal["by_type"],
                        "test_false_unknown_rate": float(unk.mean()),
                        "test_accuracy_on_accepted": float((pt[~unk] == y_type[~unk]).mean()),
                        "test_wrong_predictions_flagged_unknown": float(unk[pt != y_type].mean()),
                        "test_confident_share": float(((tc >= cal["min_confidence"]) & ~unk & ~confirm).mean()),
                        "test_confirm_share": float(confirm.mean()),
                        "test_accuracy_on_confirm": float((pt[confirm] == y_type[confirm]).mean()) if confirm.any() else None,
                        "test_accuracy_on_detected": float((pt[~unk & ~confirm] == y_type[~unk & ~confirm]).mean()),
                        "probes": {}}
        for pname, imgs in probes().items():
            if not imgs:
                continue
            Pp, Fp, _ = embed_probs(imgs, a.weights)
            u, knn_p, conf_p, conf_mask = unknown_mask(Pp, Fp, names, cal)
            Tp, _ = type_probs(Pp, names)
            M["unknown"]["probes"][pname] = {
                "n": len(imgs), "rejected_as_unknown": float(u.mean()), "shown_as_confirm": float(conf_mask.mean()),
                "auroc_knn": auroc(knn_id, knn_p), "auroc_msp": auroc(-tc, -conf_p),
                "accepted_as": dict(Counter(types[i] for i in Tp.argmax(1)[~u]).most_common())}
        leaf_unk = {n: float(unk[y_leaf == i].mean()) for i, n in enumerate(names)}
        M["unknown"]["false_unknown_by_leaf"] = dict(sorted(leaf_unk.items(), key=lambda kv: -kv[1]))
    else:
        M["unknown"] = None

    with open(out / "per_image_test.csv", "w", encoding="utf-8") as f:
        f.write("path,true_leaf,pred_leaf,true_type,pred_type,type_conf\n")
        for i, p in enumerate(paths):
            f.write(f"{p.relative_to(a.dataset).as_posix()},{names[y_leaf[i]]},{names[pl[i]]},"
                    f"{types[y_type[i]]},{types[pt[i]]},{tc[i]:.4f}\n")

    # ------------------------------------------------------------------ v1 comparison
    if not a.skip_old and a.old_weights.exists():
        Po, Fo, onames = embed_probs(paths, a.old_weights, progress="v1 on test")
        seen = old_train_stems(a.old_dataset) if a.old_dataset.exists() else set()
        stem = [re.sub(r"^garbage_v2_(.+)_[0-9a-f]{8}$", r"\1", p.stem) for p in paths]
        clean = np.array([not (source_of(p) == "garbage_v2" and s in seen) for p, s in zip(paths, stem)])
        in_old = np.isin(np.array([types[i] for i in y_type]), OLD_TYPES)
        m = in_old & clean
        old_pred = np.array([str(onames[i]) for i in Po.argmax(1)], dtype=object)
        true_t = np.array([types[i] for i in y_type])
        M["v1_comparison"] = {
            "subset": "test images whose type is one of v1's 4 classes, excluding garbage_v2 images v1 trained/validated on",
            "n": int(m.sum()), "excluded_as_seen_by_v1": int((in_old & ~clean).sum()),
            "v1_type_top1": float((old_pred[m] == true_t[m]).mean()),
            "v2_type_top1": float((pt[m] == y_type[m]).mean()),
            "v2_type_top1_all_test": M["type_top1"],
            "v1_on_new_types": {t: {str(k): int(v) for k, v in Counter(old_pred[true_t == t]).most_common()}
                                for t in types if t not in OLD_TYPES},
            "v2_on_new_types": {t: float((pt[true_t == t] == types.index(t)).mean())
                                for t in types if t not in OLD_TYPES},
            "per_type": {t: {"v1": float((old_pred[m & (true_t == t)] == t).mean()),
                             "v2": float((pt[m & (true_t == t)] == types.index(t)).mean()),
                             "n": int((m & (true_t == t)).sum())} for t in OLD_TYPES}}

        # open-set experiment: v1 never saw these materials
        if a.old_dataset.exists():
            cal_old = calibrate(a.old_weights, a.old_dataset, out=out / "v1_open_set.ood.npz")
            z = np.load(out / "v1_open_set.ood.npz")
            bank, k = z["bank"].astype(np.float32), int(z["k"])
            vi = list_split(a.old_dataset, "test")
            Pi, Fi, _ = embed_probs([f for f, _ in vi], a.old_weights, progress="v1 ID test")
            rng = random.Random(1)
            unseen = {}
            for fld in ("paper", "cardboard", "glass", "metal", "biological", "trash"):
                fs = sorted((ROOT / "data" / "Image" / "garbage_v2" / fld).glob("*"))
                unseen[fld] = rng.sample(fs, min(250, len(fs)))
            conf_i, knn_i = Pi.max(1), knn_distance(Fi, bank, k)
            res = {"id_test_n": len(vi),
                   "id_false_unknown_combined": float(((knn_i > cal_old["knn_threshold"]) | (conf_i < cal_old["msp_threshold"])).mean()),
                   "thresholds": {k2: cal_old[k2] for k2 in ("knn_threshold", "msp_threshold")},
                   "msp_only_threshold_95pct_id": float(np.quantile(conf_i, 0.05)), "materials": {}}
            msp95 = res["msp_only_threshold_95pct_id"]
            for fld, fs in unseen.items():
                Pu, Fu, _ = embed_probs(fs, a.old_weights)
                cu, ku = Pu.max(1), knn_distance(Fu, bank, k)
                res["materials"][fld] = {
                    "n": len(fs),
                    "rejected_msp_only_at_95pct_id": float((cu < msp95).mean()),
                    "rejected_combined_rule": float(((ku > cal_old["knn_threshold"]) | (cu < cal_old["msp_threshold"])).mean()),
                    "auroc_msp": auroc(-conf_i, -cu), "auroc_knn": auroc(knn_i, ku)}
            M["v1_open_set_experiment"] = res

    (out / "metrics.json").write_text(json.dumps(M, indent=2), encoding="utf-8")
    write_report(M, out)
    print((out / "report.md").read_text(encoding="utf-8"))


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def write_report(M, out):
    L = ["# Classifier evaluation (held-out test split)", "", f"Weights: `{M['weights']}`", "",
         f"**{M['n_waste_types']} waste types, {M['n_sub_types']} sub-types, {M['test_images']} test images**", "",
         "| metric | value |", "|---|---:|",
         f"| waste-type top-1 | {pct(M['type_top1'])} |", f"| waste-type top-3 | {pct(M['type_top3'])} |",
         f"| waste-type macro F1 | {pct(M['type_macro_f1'])} |",
         f"| sub-type (leaf) top-1 | {pct(M['leaf_top1'])} |", f"| sub-type (leaf) top-3 | {pct(M['leaf_top3'])} |",
         f"| sub-type macro F1 | {pct(M['leaf_macro_f1'])} |", "",
         "## Per waste type", "", "| type | precision | recall | F1 | test n |", "|---|---:|---:|---:|---:|"]
    L += [f"| {t} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} | {v['support']} |"
          for t, v in M["per_type"].items()]
    L += ["", "## Per sub-type", "", "| sub-type | precision | recall (= sub-type accuracy) | F1 | right type | sub-type acc. given right type | test n |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    L += [f"| {n} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} | {pct(v['type_accuracy'])} | "
          f"{pct(v['subtype_acc_given_right_type'])} | {v['support']} |" for n, v in M["per_leaf"].items()]
    L += ["", "## Per source dataset", "", "| source | n | type top-1 | sub-type top-1 |", "|---|---:|---:|---:|"]
    L += [f"| {s} | {v['n']} | {pct(v['type_top1'])} | {pct(v['leaf_top1'])} |" for s, v in M["per_source"].items()]
    L += ["", "Mixed-source leaves: " + ", ".join(f"{k}: {pct(v['leaf_top1'])} (n={v['n']})"
                                                 for k, v in M["per_leaf_source"].items()
                                                 if k.startswith(("plastic__plastic", "e_waste__battery")))]
    L += ["", "## Most frequent confusions", "", "Type: " + "; ".join(f"{a} ({n})" for a, n in M["top_type_confusions"]),
          "", "Sub-type: " + "; ".join(f"{a} ({n})" for a, n in M["top_leaf_confusions"]),
          "", "## Confidence (waste type)", "", "| confidence | n | accuracy |", "|---|---:|---:|"]
    L += [f"| {b['range']} | {b['n']} | {pct(b['accuracy'])} |" for b in M["confidence_bins"]]
    L += ["", "![confidence](confidence.png)", "", "Confusion matrices: `confusion_type.png`, `confusion_leaf.png` (+ CSV)."]
    U = M.get("unknown")
    if U:
        L += ["", "## Other / unknown material", "",
              f"Thresholds (from validation only): kNN distance > the predicted type's threshold "
              f"(global {U['thresholds']['knn_threshold']:.4f}; per type: "
              f"{', '.join(f'{t} {v:.3f}' for t, v in U.get('knn_threshold_by_type', {}).items())}) or "
              f"type confidence < {U['thresholds']['msp_threshold']:.4f}. Unknown = extremely unfamiliar "
              f"(per-type far threshold) or unfamiliar AND low confidence; one of the two alone = 'please "
              f"confirm' (best guess shown). 'Detected' badge at confidence >= {U['thresholds']['min_confidence']:.4f}.", "",
              f"- test images wrongly flagged unknown: {pct(U['test_false_unknown_rate'])}",
              f"- test images shown as 'please confirm': {pct(U.get('test_confirm_share'))} "
              f"(best guess right {pct(U.get('test_accuracy_on_confirm'))}); accuracy when 'detected': "
              f"{pct(U.get('test_accuracy_on_detected'))}",
              f"- accuracy on test images that were accepted: {pct(U['test_accuracy_on_accepted'])}",
              f"- share of wrong test predictions flagged unknown: {pct(U['test_wrong_predictions_flagged_unknown'])}",
              "", "| probe (not in training data) | n | rejected as unknown | shown as 'please confirm' | AUROC kNN | AUROC confidence | not rejected, labelled as |",
              "|---|---:|---:|---:|---:|---:|---|"]
        L += [f"| {k} | {v['n']} | {pct(v['rejected_as_unknown'])} | {pct(v.get('shown_as_confirm'))} | {v['auroc_knn']:.3f} | {v['auroc_msp']:.3f} | {v['accepted_as']} |"
              for k, v in U["probes"].items()]
    V = M.get("v1_comparison")
    if V:
        L += ["", "## v1 (4-class) vs v2", "", f"Subset: {V['subset']} (n={V['n']}; {V['excluded_as_seen_by_v1']} excluded).", "",
              "| | v1 | v2 |", "|---|---:|---:|", f"| type top-1 on the shared subset | {pct(V['v1_type_top1'])} | {pct(V['v2_type_top1'])} |"]
        L += [f"| {t} (n={v['n']}) | {pct(v['v1'])} | {pct(v['v2'])} |" for t, v in V["per_type"].items()]
        L += [f"| {t} (new type) | 0% by construction (predicted {V['v1_on_new_types'][t]}) | {pct(V['v2_on_new_types'][t])} |"
              for t in V["v2_on_new_types"]]
    X = M.get("v1_open_set_experiment")
    if X:
        L += ["", "## Open-set experiment on v1 (materials it never saw)", "",
              f"Thresholds calibrated on v1's own validation split; v1 ID test wrongly flagged: {pct(X['id_false_unknown_combined'])}.", "",
              "| unseen material | n | rejected: confidence only (95% ID kept) | rejected: kNN + confidence rule | AUROC confidence | AUROC kNN |",
              "|---|---:|---:|---:|---:|---:|"]
        L += [f"| {m} | {v['n']} | {pct(v['rejected_msp_only_at_95pct_id'])} | {pct(v['rejected_combined_rule'])} | "
              f"{v['auroc_msp']:.3f} | {v['auroc_knn']:.3f} |" for m, v in X["materials"].items()]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
