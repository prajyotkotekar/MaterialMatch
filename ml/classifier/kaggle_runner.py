"""
kaggle_runner.py - Run the heavy training jobs on Kaggle's free GPU instead of this laptop.

    python -m ml.classifier.kaggle_runner check                  # API connection + weekly GPU quota
    python -m ml.classifier.kaggle_runner upload-dataset         # data/processed/cls_dataset -> private Kaggle dataset
    python -m ml.classifier.kaggle_runner train [--epochs 25]    # full training on the GPU, then OOD calibration
    python -m ml.classifier.kaggle_runner train --tag lowres --lowres-fraction 0.35   # separate, parallel run
    python -m ml.classifier.kaggle_runner feedback [--promote]   # retrain_with_feedback --confirm on the GPU
    python -m ml.classifier.kaggle_runner status [train|feedback]
    python -m ml.classifier.kaggle_runner fetch  train|feedback  # download results of the last run again

Needs the official Kaggle API token at %USERPROFILE%\\.kaggle\\kaggle.json (never inside the project,
never committed). The account must be phone-verified for GPU + internet.

What goes to Kaggle (all PRIVATE):
  * dataset <user>/materialmatch-cls-dataset   - the built training set (uploaded once; re-uploaded
    only when data/processed/cls_dataset changes; `train` does this automatically)
  * dataset <user>/materialmatch-feedback-inputs - `feedback` only: current best.pt + the feedback log
    and the user's feedback photos from data/feedback/
  * kernels <user>/materialmatch-train, <user>/materialmatch-feedback - ml/classifier/kaggle_kernel.py
    with the project's ml/ code embedded (no separate code upload)

Results land in ml/classifier/kaggle_runs/<job>_<run_id>/ and the checkpoint is copied to
    train    -> weights/kaggle_candidate.pt (+ .ood.npz, .hierarchy.json)   never best.pt
    feedback -> weights/feedback_candidate.pt (+ .ood.npz, .hierarchy.json); best.pt is replaced only
                with --promote AND if the test gate passed (backup: best_before_feedback_<stamp>.*)
Evaluation (evaluate_classifier) still runs locally: its probes use local folders.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "ml" / "classifier" / "weights"
DATASET = ROOT / "data" / "processed" / "cls_dataset"
FEEDBACK_DIR = ROOT / "data" / "feedback"
RUNS = ROOT / "ml" / "classifier" / "kaggle_runs"
STATE = RUNS / "state.json"
KERNEL_TEMPLATE = Path(__file__).with_name("kaggle_kernel.py")
# staging outside OneDrive so the 400 MB zip is not synced
STAGE = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "materialmatch_kaggle"
DATASET_SLUG = "materialmatch-cls-dataset"
INPUTS_SLUG = "materialmatch-feedback-inputs"
DONE = ("complete", "error", "cancel")


# --------------------------------------------------------------------------- #
# Kaggle CLI helpers
# --------------------------------------------------------------------------- #
def kaggle_exe() -> str:
    exe = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    found = str(exe) if exe.exists() else shutil.which("kaggle")
    if not found:
        raise SystemExit("kaggle CLI not found: pip install kaggle (inside .venv)")
    return found


def kaggle(*args, check=True) -> str:
    # UTF-8 mode: the CLI writes the kernel log with the console code page otherwise and crashes on
    # non-ASCII characters (progress bars) on Windows
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run([kaggle_exe(), *map(str, args)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    out = (p.stdout + p.stderr).strip()
    if check and p.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(map(str, args))} failed:\n{out}")
    return out


def username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not cfg.exists():
        raise SystemExit(f"No Kaggle token at {cfg}. Kaggle -> Settings -> API -> Create New Token, "
                         "save the downloaded kaggle.json there (not in the project folder).")
    return json.loads(cfg.read_text(encoding="utf-8"))["username"]   # only the name; the key is never read


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"runs": {}}


def save_state(s: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2), encoding="utf-8")


def dataset_status(ref: str) -> str | None:
    out = kaggle("datasets", "status", ref, check=False).lower()
    if "404" in out or "403" in out or "not found" in out:     # a dataset that doesn't exist gives 403
        return None
    for s in ("ready", "pending", "error"):
        if s in out:
            return s
    return out


def wait_dataset_ready(ref: str, timeout_s: int = 3600) -> None:
    t0 = time.time()
    while True:
        s = dataset_status(ref)
        if s == "ready":
            print(f"  dataset {ref}: ready")
            return
        if s == "error" or time.time() - t0 > timeout_s:
            raise SystemExit(f"dataset {ref} did not become ready (status {s})")
        print(f"  dataset {ref}: {s} ... ({int(time.time() - t0)} s)")
        time.sleep(20)


def publish_dataset(folder: Path, ref: str, notes: str) -> None:
    """Create the private dataset, or add a new version if it exists; then wait until it is ready."""
    if dataset_status(ref) is None:
        print(f"Creating PRIVATE dataset {ref} ...")
        out = kaggle("datasets", "create", "-p", folder, "--dir-mode", "skip")
    else:
        print(f"Uploading a new version of {ref} ...")
        out = kaggle("datasets", "version", "-p", folder, "-m", notes, "--dir-mode", "skip")
    print("  " + out.splitlines()[-1] if out else "")
    if "error" in out.lower() and "successfully" not in out.lower():
        raise SystemExit(out)
    time.sleep(10)
    wait_dataset_ready(ref)


# --------------------------------------------------------------------------- #
# Uploads
# --------------------------------------------------------------------------- #
def fingerprint(folder: Path) -> tuple[str, int, int]:
    h, n, size = hashlib.sha1(), 0, 0
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix != ".cache":
            st = p.stat()
            h.update(f"{p.relative_to(folder).as_posix()}|{st.st_size}\n".encode())
            n, size = n + 1, size + st.st_size
    return h.hexdigest()[:16], n, size


def metadata(folder: Path, ref: str, title: str, description: str) -> None:
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": ref, "licenses": [{"name": "other"}], "description": description},
        indent=2), encoding="utf-8")


def dataset_slug(folder: Path) -> str:
    """The default cls_dataset keeps its original Kaggle dataset; any other built dataset folder gets
    its own private Kaggle dataset, so uploading it never replaces the original."""
    folder = Path(folder).resolve()
    if folder == DATASET.resolve():
        return DATASET_SLUG
    return DATASET_SLUG + "-" + re.sub(r"[^a-z0-9]+", "-", folder.name.lower()).strip("-")


def upload_dataset(force: bool = False, folder: Path = DATASET) -> dict:
    folder = Path(folder).resolve()
    if not (folder / "train").is_dir():
        raise SystemExit(f"{folder} is missing - build it: python ml/classifier/train_yolo.py --auto --build-only")
    slug = dataset_slug(folder)
    ref = f"{username()}/{slug}"
    key = "dataset" if slug == DATASET_SLUG else f"dataset:{slug}"
    fp, n, size = fingerprint(folder)
    state = load_state()
    cur = state.get(key, {})
    if not force and cur.get("fingerprint") == fp and dataset_status(ref) == "ready":
        print(f"Dataset {ref} is up to date ({n} files, fingerprint {fp}).")
        return cur
    stage = STAGE / "dataset"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    print(f"Zipping {n} files ({size / 1e6:.0f} MB) -> {stage / 'cls_dataset.zip'} ...")
    with zipfile.ZipFile(stage / "cls_dataset.zip", "w", zipfile.ZIP_STORED) as z:   # JPEGs: no gain from deflate
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix != ".cache":
                z.write(p, p.relative_to(folder).as_posix())
    man = {"dataset": ref, "fingerprint": fp, "files": n, "bytes": size,
           "uploaded_at": datetime.now().isoformat(timespec="seconds")}
    (stage / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    metadata(stage, ref, "MaterialMatch classifier dataset",
             "Private working copy of MaterialMatch's built training set (27 waste sub-types). Sources: "
             "CODD (Demetriou et al., Mendeley Data, doi:10.17632/wds85kt64j.3, CC BY 4.0); Garbage "
             "Dataset v2 (Suman Kunwar, Kaggle, MIT); E Waste Image Dataset (Akshat Tamrakar, Kaggle, "
             "Apache 2.0).")
    t = time.time()
    publish_dataset(stage, ref, f"cls_dataset {fp}")
    print(f"Upload + processing took {(time.time() - t) / 60:.1f} min.")
    shutil.rmtree(stage, ignore_errors=True)
    state[key] = man
    save_state(state)
    return man


def upload_feedback_inputs(run_id: str) -> str:
    log = FEEDBACK_DIR / "classifier_feedback.jsonl"
    if not log.exists():
        raise SystemExit(f"No feedback yet ({log} does not exist).")
    ref = f"{username()}/{INPUTS_SLUG}"
    stage = STAGE / "inputs"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    files = [W / "best.pt", W / "best.hierarchy.json", log]
    files += sorted(p for p in (FEEDBACK_DIR / "images").rglob("*") if p.is_file())
    with zipfile.ZipFile(stage / "inputs.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if p.exists():
                z.write(p, p.relative_to(ROOT).as_posix())
    n_img = sum(1 for p in files if "images" in p.parts)
    print(f"Feedback inputs: best.pt + feedback log + {n_img} feedback photo(s) -> PRIVATE dataset {ref}")
    (stage / "manifest.json").write_text(json.dumps({"dataset": ref, "run_id": run_id}), encoding="utf-8")
    metadata(stage, ref, "MaterialMatch feedback inputs",
             "Private: current MaterialMatch classifier weights + user-corrected photos for fine-tuning.")
    publish_dataset(stage, ref, f"run {run_id}")
    shutil.rmtree(stage, ignore_errors=True)
    return ref


# --------------------------------------------------------------------------- #
# Kernel push / wait / fetch
# --------------------------------------------------------------------------- #
def code_payload() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "ml").rglob("*.py")):
            if "__pycache__" not in p.parts:
                z.write(p, p.relative_to(ROOT).as_posix())
        z.write(ROOT / "data" / "processed" / "class_mapping.csv", "data/processed/class_mapping.csv")
    return base64.b64encode(buf.getvalue()).decode()


def push_kernel(job: str, config: dict, sources: list[str], accelerator: str | None) -> str:
    ref = f"{username()}/materialmatch-{job}"
    folder = STAGE / f"kernel_{job}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    script = KERNEL_TEMPLATE.read_text(encoding="utf-8")
    script = script.replace("__CONFIG_B64__", base64.b64encode(json.dumps(config).encode()).decode(), 1)
    script = script.replace("__PAYLOAD_B64__", code_payload(), 1)
    (folder / "run.py").write_text(script, encoding="utf-8")
    meta = {"id": ref, "title": f"materialmatch-{job}", "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": sources, "competition_sources": [], "kernel_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    args = ["kernels", "push", "-p", folder] + (["--accelerator", accelerator] if accelerator else [])
    out = kaggle(*args)
    print("  " + out.replace("\n", "\n  "))
    if "error" in out.lower() and "successfully" not in out.lower():
        raise SystemExit("Kernel push failed.")
    return ref


def kernel_status(ref: str) -> str:
    out = kaggle("kernels", "status", ref, check=False)
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?([A-Za-z_]+)', out)
    return (m.group(1) if m else out).lower()


def fetch(job: str, run: dict) -> Path:
    dest = RUNS / f"{job}_{run['run_id']}"
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", run["kernel"], "-p", dest, "-o", "-q")
    return dest


def wait_and_fetch(job: str, run: dict, poll_s: int = 60, timeout_h: float = 11.5) -> tuple[Path, dict]:
    t0, last, stale = time.time(), None, 0
    print(f"Waiting for {run['kernel']} (https://www.kaggle.com/code/{run['kernel']}) ...")
    while True:
        s = kernel_status(run["kernel"])
        if s != last:
            print(f"  [{datetime.now():%H:%M:%S}] {s}", flush=True)
            last = s
        if any(d in s for d in DONE):
            dest = fetch(job, run)
            info_f = dest / "run_info.json"
            info = json.loads(info_f.read_text(encoding="utf-8")) if info_f.exists() else {}
            if info.get("run_id") == run["run_id"]:
                return dest, info
            stale += 1                               # status still describes the previous version
            if stale > 10:
                raise SystemExit(f"Kernel finished but run_info.json is not from run {run['run_id']}; "
                                 f"see {dest} and the Kaggle page.")
        if time.time() - t0 > timeout_h * 3600:
            raise SystemExit("Timed out waiting; rerun later with: python -m ml.classifier.kaggle_runner "
                             f"fetch {job}")
        time.sleep(poll_s)


def summarize(info: dict, dest: Path) -> None:
    print(f"\nRun {info.get('run_id')} on {info.get('gpu')} (torch {info.get('torch')}): "
          f"{'OK' if info.get('ok') else 'FAILED - ' + str(info.get('error'))}")
    for s in info.get("steps", []):
        print(f"  {s['name']:<22} exit {s['rc']}  {s['seconds'] / 60:.1f} min")
    logs = list(dest.glob("*.log"))
    if logs:
        print(f"Full log: {logs[0]}")


def candidate_stem(name: str) -> str:
    """train -> kaggle_candidate, train-lowres -> kaggle_candidate_lowres"""
    tag = name.split("-", 1)[1] if "-" in name else ""
    return "kaggle_candidate" + (f"_{tag}" if tag else "")


def install_train(dest: Path, run_id: str, name: str = "train") -> None:
    src, stem = dest / "weights", candidate_stem(name)
    for suffix in (".pt", ".ood.npz", ".hierarchy.json"):
        f = src / f"best_candidate{suffix}"
        if f.exists():
            shutil.copy2(f, W / f"{stem}{suffix}")
    print(f"\nCandidate: {W / (stem + '.pt')} (best.pt untouched)\nNext:")
    print(f"  python -m ml.classifier.evaluate_classifier --weights {W / (stem + '.pt')} --skip-old "
          f"--out ml\\classifier\\reports\\kaggle_eval_{run_id}")
    print(f"  # read the report; to promote, copy {stem}.pt/.ood.npz/.hierarchy.json over best.*"
          " (back up best.* first), then restart Streamlit")


def install_feedback(dest: Path, promote: bool) -> None:
    src = dest / "weights" / "feedback_candidate.pt"
    if not src.exists():
        raise SystemExit("No feedback_candidate.pt in the Kaggle output.")
    shutil.copy2(src, W / "feedback_candidate.pt")
    if (src.with_suffix(".ood.npz")).exists():
        shutil.copy2(src.with_suffix(".ood.npz"), W / "feedback_candidate.ood.npz")
    shutil.copy2(W / "best.hierarchy.json", W / "feedback_candidate.hierarchy.json")
    reports = sorted((dest / "reports").glob("feedback_retrain_*"))
    res = {}
    for rep in reports:
        local = ROOT / "ml" / "classifier" / "reports" / rep.name
        shutil.copytree(rep, local, dirs_exist_ok=True)
        print((rep / "report.md").read_text(encoding="utf-8") if (rep / "report.md").exists() else "")
        print(f"Report: {local / 'report.md'}")
        if (rep / "results.json").exists():
            res = json.loads((rep / "results.json").read_text(encoding="utf-8"))
    if not promote:
        print("Not promoted (use --promote to replace best.pt when the gate passes).")
        return
    if not res.get("passed") or not (W / "feedback_candidate.ood.npz").exists():
        print("NOT promoted: the test gate failed (or calibration is missing). best.pt unchanged.")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in (".pt", ".ood.npz"):
        shutil.copy2(W / f"best{suffix}", W / f"best_before_feedback_{stamp}{suffix}")
        shutil.copy2(W / f"feedback_candidate{suffix}", W / f"best{suffix}")
    print(f"PROMOTED: best.pt replaced (backup best_before_feedback_{stamp}.pt/.ood.npz). Restart Streamlit.")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_check(a) -> int:
    print(f"Kaggle user : {username()}")
    print(kaggle("quota"))
    print(f"Dataset     : {DATASET_SLUG} -> {dataset_status(f'{username()}/{DATASET_SLUG}') or 'not uploaded yet'}")
    return 0


def run_name(job: str, tag: str | None) -> str:
    if tag and not re.fullmatch(r"[a-z0-9]+", tag):
        raise SystemExit("--tag must be lowercase letters/digits")
    return f"{job}-{tag}" if tag else job


def start(job: str, args: dict, sources: list[str], extra: dict, accelerator: str | None,
          name: str | None = None) -> dict:
    """Push one run. `name` (job or job-tag) = Kaggle kernel slug suffix + key in state.json, so
    runs with different tags are separate kernels and can run at the same time."""
    name = name or job
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {"job": job, "run_id": run_id, "args": args,
              "ultralytics": importlib.metadata.version("ultralytics"), **extra}
    if job == "feedback":
        config["inputs_ref"] = upload_feedback_inputs(run_id)
        sources = sources + [config["inputs_ref"]]
    print(f"Starting Kaggle job '{name}' (run {run_id}) ...")
    kernel = push_kernel(name, config, sources, accelerator)
    run = {"run_id": run_id, "kernel": kernel, "pushed_at": datetime.now().isoformat(timespec="seconds"),
           "config": {k: v for k, v in config.items()}}
    state = load_state()
    state["runs"][name] = run
    save_state(state)
    return run


def cmd_train(a) -> int:
    man = upload_dataset(folder=a.dataset_dir)
    name = run_name("train", a.tag)
    args = {k: getattr(a, k) for k in ("epochs", "patience", "batch", "workers", "imgsz", "lr0", "optimizer",
                                       "warmup_epochs", "model", "seed", "name", "lowres_fraction",
                                       "robust_aug", "crop_scale", "cos_lr")}
    run = start("train", args, [man["dataset"]], {"dataset_ref": man["dataset"],
                                                  "dataset_fingerprint": man["fingerprint"]}, a.accelerator, name)
    if a.no_wait:
        print(f"Not waiting. Later: python -m ml.classifier.kaggle_runner fetch {name}")
        return 0
    dest, info = wait_and_fetch(name, run)
    summarize(info, dest)
    if info.get("ok"):
        install_train(dest, run["run_id"], name)
    return 0 if info.get("ok") else 1


def cmd_feedback(a) -> int:
    man = upload_dataset()
    retrain = ["--repeat", a.repeat, "--replay-per-class", a.replay_per_class, "--val-per-class",
               a.val_per_class, "--epochs", a.epochs, "--lr0", a.lr0, "--max-type-drop", a.max_type_drop,
               "--max-leaf-drop", a.max_leaf_drop] + (["--since", a.since] if a.since else [])
    run = start("feedback", {"retrain_args": [str(x) for x in retrain]}, [man["dataset"]],
                {"dataset_ref": man["dataset"], "dataset_fingerprint": man["fingerprint"]}, a.accelerator)
    if a.no_wait:
        print("Not waiting. Later: python -m ml.classifier.kaggle_runner fetch feedback [--promote]")
        return 0
    dest, info = wait_and_fetch("feedback", run)
    summarize(info, dest)
    if info.get("ok"):
        install_feedback(dest, a.promote)
    return 0 if info.get("ok") else 1


def cmd_status(a) -> int:
    runs = load_state()["runs"]
    for name in ([a.job] if a.job else runs):
        if name in runs:
            r = runs[name]
            print(f"{name:<14} run {r['run_id']}  {r['kernel']}  -> {kernel_status(r['kernel'])}")
    if not runs:
        print("No Kaggle runs started yet.")
    return 0


def cmd_fetch(a) -> int:
    runs = load_state()["runs"]
    run = runs.get(a.job)
    if not run:
        raise SystemExit(f"No '{a.job}' run recorded (known: {', '.join(runs) or 'none'}).")
    dest, info = wait_and_fetch(a.job, run)
    summarize(info, dest)
    if info.get("ok"):
        if run["config"]["job"] == "train":
            install_train(dest, run["run_id"], a.job)
        else:
            install_feedback(dest, a.promote)
    return 0 if info.get("ok") else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    up = sub.add_parser("upload-dataset")
    up.add_argument("--force", action="store_true")
    up.add_argument("--dataset-dir", type=Path, default=DATASET)
    up.set_defaults(fn=lambda a: upload_dataset(a.force, a.dataset_dir) and 0)

    def common(p):
        p.add_argument("--accelerator", default="NvidiaTeslaT4", help="Kaggle machine shape ('' = Kaggle default)")
        p.add_argument("--no-wait", action="store_true", help="start the job and return")

    t = sub.add_parser("train", help="train_yolo.py on the Kaggle GPU (+ calibrate_ood)")
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--patience", type=int, default=7)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--imgsz", type=int, default=224)
    t.add_argument("--lr0", type=float, default=0.00125)
    t.add_argument("--optimizer", default="AdamW")
    t.add_argument("--warmup-epochs", type=float, default=1.0)
    t.add_argument("--model", default="yolov8n-cls.pt")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--name", default="materialmatch_hier_kaggle")
    t.add_argument("--lowres-fraction", type=float, default=0.0,
                   help="train_yolo --lowres-fraction (low-res copies against the small-photo bias)")
    t.add_argument("--robust-aug", type=float, default=0.0, help="train_yolo --robust-aug (corruption probability)")
    t.add_argument("--crop-scale", type=float, default=None, help="train_yolo --crop-scale")
    t.add_argument("--cos-lr", action="store_true")
    t.add_argument("--dataset-dir", type=Path, default=DATASET,
                   help="built dataset to train on (a non-default folder is uploaded as its own private dataset)")
    t.add_argument("--tag", default=None, help="separate kernel + candidate name, e.g. lowres -> "
                                               "materialmatch-train-lowres / weights/kaggle_candidate_lowres.pt")
    common(t)
    t.set_defaults(fn=cmd_train)

    f = sub.add_parser("feedback", help="retrain_with_feedback --confirm on the Kaggle GPU (uploads feedback photos)")
    f.add_argument("--promote", action="store_true", help="replace best.pt if the test gate passes")
    f.add_argument("--since", default=None)
    f.add_argument("--repeat", type=int, default=20)
    f.add_argument("--replay-per-class", type=int, default=150)
    f.add_argument("--val-per-class", type=int, default=60)
    f.add_argument("--epochs", type=int, default=3)
    f.add_argument("--lr0", type=float, default=0.0002)
    f.add_argument("--max-type-drop", type=float, default=0.5)
    f.add_argument("--max-leaf-drop", type=float, default=1.0)
    common(f)
    f.set_defaults(fn=cmd_feedback)

    s = sub.add_parser("status")
    s.add_argument("job", nargs="?", help="run name: train, feedback or train-<tag>")
    s.set_defaults(fn=cmd_status)
    fe = sub.add_parser("fetch")
    fe.add_argument("job", help="run name: train, feedback or train-<tag>")
    fe.add_argument("--promote", action="store_true")
    fe.set_defaults(fn=cmd_fetch)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
