"""
backend/feedback_sync.py - Keep data/feedback in a private Hugging Face dataset.

Hosts such as Streamlit Community Cloud wipe the disk on every reboot, which loses the feedback log,
its photos and with them the instant feedback memory. With MM_FEEDBACK_REPO and HF_TOKEN set (env vars,
or root-level Streamlit secrets, which Streamlit also exports as env vars) the folder is downloaded once
at startup and every new answer is uploaded in the background. Unset = local disk only, as before.

    MM_FEEDBACK_REPO = "<hf-user>/materialmatch-feedback"   # created private if missing
    HF_TOKEN = "hf_..."                                     # token with write access

The photos are user uploads, so a repo that is not private is refused.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

RETRY_SECONDS = 60
_lock = threading.Lock()
_pending: set[Path] = set()
_wake = threading.Event()
_state: dict = {"restored": False, "blocked": None, "worker": None}


def _config() -> tuple[str, str] | None:
    repo, token = os.environ.get("MM_FEEDBACK_REPO"), os.environ.get("HF_TOKEN")
    return (repo, token) if repo and token and not _state["blocked"] else None


def enabled() -> bool:
    return _config() is not None


def _check_private(api, repo: str) -> bool:
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:
        _state["blocked"] = f"{repo} is public; feedback photos are only synced to a private dataset"
        log.error(_state["blocked"])
        return False
    return True


def restore(log_path: Path) -> int:
    """Download the saved feedback once per process if this disk has none. Returns files restored."""
    cfg = _config()
    with _lock:
        if cfg is None or _state["restored"]:
            return 0
        _state["restored"] = True
    if log_path.exists():  # local data wins (dev machine, or the same container after a rerun)
        return 0
    repo, token = cfg
    folder = log_path.parent
    try:
        from huggingface_hub import HfApi, snapshot_download

        if not _check_private(HfApi(token=token), repo):
            return 0
        snapshot_download(repo, repo_type="dataset", local_dir=folder, token=token,
                          allow_patterns=[log_path.name, "memory_state.json", "images/*"])
    except Exception as exc:  # the app must still start without the remote copy
        log.warning("Feedback restore from %s failed: %s", repo, exc)
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and ".cache" not in p.parts)


def push(log_path: Path, files: list[Path]) -> None:
    """Queue the log and new files for upload; a failed upload is retried every RETRY_SECONDS."""
    if not enabled():
        return
    with _lock:
        _pending.update([log_path, *files])
        if _state["worker"] is None:
            _state["worker"] = threading.Thread(target=_upload_loop, args=(log_path.parent,),
                                                name="feedback-sync", daemon=True)
            _state["worker"].start()
    _wake.set()


def flush(timeout: float = 30) -> bool:
    """Wait until the queue is uploaded (tests, shutdown). True when nothing is left."""
    _wake.set()
    for _ in range(int(timeout * 10)):
        with _lock:
            if not _pending and not _state.get("busy"):
                return True
        time.sleep(0.1)
    return False


def _upload_loop(folder: Path) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    checked = False
    while True:
        _wake.wait(RETRY_SECONDS)
        _wake.clear()
        cfg = _config()
        with _lock:
            batch = sorted(_pending) if cfg else []
            _pending.clear()
            _state["busy"] = bool(batch)
        if not batch:
            continue
        repo, token = cfg
        api = HfApi(token=token)
        try:
            if not checked:
                if not _check_private(api, repo):
                    continue
                checked = True
            # read now, so the log is a consistent snapshot even while new answers are appended
            ops = [CommitOperationAdd(path_in_repo=p.relative_to(folder).as_posix(), path_or_fileobj=p.read_bytes())
                   for p in batch if p.exists()]
            api.create_commit(repo, repo_type="dataset", operations=ops,
                              commit_message=f"Add feedback ({len(ops)} files)")
        except Exception as exc:
            log.warning("Feedback upload to %s failed, retrying: %s", repo, exc)
            with _lock:
                _pending.update(batch)
        finally:
            with _lock:
                _state["busy"] = False
