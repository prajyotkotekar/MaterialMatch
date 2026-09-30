"""
kaggle_kernel.py - The script that runs ON Kaggle's GPU. Do not run it locally.

ml/classifier/kaggle_runner.py fills in CONFIG_B64 (job settings) and PAYLOAD_B64 (a zip of the
project's ml/ code + class_mapping.csv), pushes it as a private Kaggle script, and downloads what
this script writes to /kaggle/working:

    run_info.json        run id, GPU, versions, per-step exit codes and durations
    weights/             best_candidate.pt (+ .hierarchy.json, .ood.npz)  |  feedback_candidate.pt (+ .ood.npz)
    runs/                ultralytics training run (curves, confusion matrix)          [train job]
    reports/             feedback_retrain_<stamp>/ report.md + results.json           [feedback job]

The data comes from private Kaggle datasets attached to the kernel. Each carries a manifest.json;
the script refuses to run on a dataset whose fingerprint / run id differs from what the runner
uploaded (i.e. a stale dataset version).
"""

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

CONFIG_B64 = "__CONFIG_B64__"
PAYLOAD_B64 = "__PAYLOAD_B64__"

CONFIG = json.loads(base64.b64decode(CONFIG_B64))
ROOT = Path("/tmp/mm")                      # project root on Kaggle (NOT in the downloaded output)
OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PY = sys.executable
info = {"run_id": CONFIG["run_id"], "job": CONFIG["job"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": [], "ok": False}


def save_info():
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def fail(msg):
    print("FAILED:", msg, flush=True)
    info["error"] = msg
    save_info()
    sys.exit(1)


def run(name, cmd):
    print(f"\n=== {name}: {' '.join(map(str, cmd))}", flush=True)
    t = time.time()
    rc = subprocess.call([str(c) for c in cmd], cwd=ROOT)
    info["steps"].append({"name": name, "rc": rc, "seconds": round(time.time() - t)})
    save_info()
    return rc


def dataset_root(ref):
    """Folder of the attached dataset `ref`, found through its manifest.json."""
    for m in INPUT.rglob("manifest.json"):
        try:
            man = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        if man.get("dataset") == ref:
            return m.parent, man
    fail(f"dataset {ref} is not attached (no manifest.json for it under /kaggle/input)")


def unpack(root, zip_name, marker, dest):
    """Kaggle may or may not extract uploaded zips: return the folder that contains `marker`."""
    hits = [p.parent for p in root.rglob(marker)]
    if hits:
        return hits[0]
    zips = list(root.rglob(zip_name))
    if not zips:
        fail(f"neither {marker} nor {zip_name} found in {root}")
    print(f"extracting {zips[0]} -> {dest}", flush=True)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(dest)
    hits = [p.parent for p in Path(dest).rglob(marker)]
    if not hits:
        fail(f"{zip_name} does not contain {marker}")
    return hits[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save_info()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True)
    info["gpu"] = smi.stdout.strip() or "none"
    print("GPU:", info["gpu"], flush=True)

    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD_B64))) as z:
        z.extractall(ROOT)

    if run("pip install", [PY, "-m", "pip", "install", "-q", f"ultralytics=={CONFIG['ultralytics']}"]) != 0:
        fail("pip install ultralytics failed (is internet enabled for this kernel?)")
    import torch
    info["torch"] = torch.__version__
    info["cuda"] = bool(torch.cuda.is_available())
    save_info()
    if not info["cuda"]:
        fail("no CUDA GPU in this session - check the kernel's accelerator / the weekly GPU quota")

    # training images
    ds_root, man = dataset_root(CONFIG["dataset_ref"])
    if man.get("fingerprint") != CONFIG["dataset_fingerprint"]:
        fail(f"attached dataset is a different version (fingerprint {man.get('fingerprint')} != "
             f"{CONFIG['dataset_fingerprint']}); wait for the new version to finish processing and rerun")
    cls = unpack(ds_root, "cls_dataset.zip", "hierarchy.json", "/tmp/cls_dataset")
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    os.symlink(cls, ROOT / "data" / "processed" / "cls_dataset")
    dataset = ROOT / "data" / "processed" / "cls_dataset"
    a = CONFIG["args"]

    if CONFIG["job"] == "train":
        (OUT / "weights").mkdir(exist_ok=True)
        rc = run("train", [PY, "-m", "ml.classifier.train_yolo", "--dataset", dataset, "--device", "0",
                           "--epochs", a["epochs"], "--patience", a["patience"], "--batch", a["batch"],
                           "--workers", a["workers"], "--imgsz", a["imgsz"], "--lr0", a["lr0"],
                           "--optimizer", a["optimizer"], "--warmup-epochs", a["warmup_epochs"],
                           "--model", a["model"], "--seed", a["seed"],
                           "--lowres-fraction", a.get("lowres_fraction", 0.0),
                           "--lowres-dir", "/tmp/cls_dataset_lowres",
                           "--robust-aug", a.get("robust_aug", 0.0),
                           *(["--crop-scale", a["crop_scale"]] if a.get("crop_scale") is not None else []),
                           *(["--cos-lr"] if a.get("cos_lr") else []),
                           "--project", OUT / "runs", "--name", a["name"],
                           "--weights-dir", OUT / "weights", "--weights-name", "best_candidate.pt"])
        if rc != 0:
            fail(f"training exited with {rc}")
        if run("calibrate", [PY, "-m", "ml.classifier.calibrate_ood", "--weights",
                             OUT / "weights" / "best_candidate.pt", "--dataset", dataset]) != 0:
            fail("calibration failed")

    elif CONFIG["job"] == "feedback":
        in_root, man = dataset_root(CONFIG["inputs_ref"])
        if man.get("run_id") != CONFIG["run_id"]:
            fail(f"feedback inputs are from run {man.get('run_id')}, expected {CONFIG['run_id']} "
                 "(the new dataset version was not ready yet); rerun")
        overlay = unpack(in_root, "inputs.zip", "classifier_feedback.jsonl", "/tmp/mm_inputs").parents[1]
        shutil.copytree(overlay, ROOT, dirs_exist_ok=True)
        rc = run("retrain_with_feedback", [PY, "-m", "ml.classifier.retrain_with_feedback", "--confirm",
                                           "--device", "0", *a["retrain_args"]])
        info["gate_passed"] = rc == 0
        w = ROOT / "ml" / "classifier" / "weights" / "feedback_candidate.pt"
        (OUT / "weights").mkdir(exist_ok=True)
        if w.exists():
            shutil.copy2(w, OUT / "weights" / w.name)
        for rep in (ROOT / "ml" / "classifier" / "reports").glob("feedback_retrain_*"):
            shutil.copytree(rep, OUT / "reports" / rep.name, dirs_exist_ok=True)
        if rc not in (0, 3):                        # 3 = trained fine but the test gate failed
            fail(f"retrain_with_feedback exited with {rc}")
        if rc == 0 and run("calibrate", [PY, "-m", "ml.classifier.calibrate_ood", "--weights",
                                         OUT / "weights" / w.name, "--dataset", dataset]) != 0:
            fail("calibration failed")
    elif CONFIG["job"] == "eval":
        in_root, man = dataset_root(CONFIG["inputs_ref"])
        if man.get("run_id") != CONFIG["run_id"]:
            fail(f"eval inputs are from run {man.get('run_id')}, expected {CONFIG['run_id']}; rerun")
        overlay = unpack(in_root, "inputs.zip", "classifier_feedback.jsonl", "/tmp/mm_inputs").parents[1]
        shutil.copytree(overlay, ROOT, dirs_exist_ok=True)
        w = ROOT / "ml" / "classifier" / "weights"
        for stem in a["weights"]:
            if run(f"evaluate {stem}", [PY, "-m", "ml.classifier.evaluate_classifier", "--weights", w / f"{stem}.pt",
                                        "--skip-old", "--dataset", dataset, "--out", OUT / "reports" / f"eval_{stem}"]):
                fail(f"evaluate_classifier failed for {stem}")
        if run("real photos", [PY, "-m", "ml.classifier.evaluate_real_photos", "--weights", w / "best.pt",
                               *[w / f"{s}.pt" for s in a["weights"]]]) == 0:
            shutil.copytree(ROOT / "ml" / "classifier" / "reports" / "real_photos", OUT / "reports" / "real_photos",
                            dirs_exist_ok=True)
    else:
        fail(f"unknown job {CONFIG['job']}")

    info["ok"] = True
    info["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_info()
    print("\nDONE", json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
