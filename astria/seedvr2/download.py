"""
Utilities to fetch the SeedVR2-7B checkpoint from Hugging Face.

- Public repo:  ByteDance-Seed/SeedVR2-7B
- File name:    seedvr2_ema_7b_sharp.pth (default/“sharp” variant)

Usage:

    from astria.seedvr2.projects.seedvr2_download import ensure_seedvr2_7b_checkpoint

    ckpt_path = ensure_seedvr2_7b_checkpoint("./ckpts/seedvr2_ema_7b.pth")
    # => returns the local path where the checkpoint is available
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Tuple, Optional

from astria_utils import CACHE_DIR
from filelock import FileLock, Timeout
from torch.hub import download_url_to_file
from huggingface_hub import snapshot_download, hf_hub_download


DEFAULT_REPO_ID = "ByteDance-Seed/SeedVR2-7B"
# the upstream filename we want to fetch
DEFAULT_FILENAME_SHARP = "seedvr2_ema_7b_sharp.pth"
DEFAULT_FILENAME_REGULAR = "seedvr2_ema_7b.pth"
# if you ever need a non-sharp variant, change this or add a param
DEFAULT_FILENAME = DEFAULT_FILENAME_REGULAR

_POS_URL = "https://github.com/ByteDance-Seed/SeedVR/raw/refs/heads/main/pos_emb.pt"
_NEG_URL = "https://github.com/ByteDance-Seed/SeedVR/raw/refs/heads/main/neg_emb.pt"
VAE_FILENAME = "ema_vae.pth"


def _existing_and_nonempty(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 1024
    except Exception:
        return False


def ensure_seedvr2_text_embeddings(
    dest_dir: str,
    *,
    prefix: str = "seedvr2_",
    repo_id: str = DEFAULT_REPO_ID,
    prefer_snapshot: bool = True,
    token: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Ensure {dest_dir}/{prefix}pos_emb.pt and {dest_dir}/{prefix}neg_emb.pt exist.
    - If legacy files (pos_emb.pt / neg_emb.pt) already exist, copy them to prefixed names.
    - If nothing exists, download from HF (snapshot first, then direct URLs).
    - Creates legacy symlinks (or copies) to keep old code working.

    Returns: (prefixed_pos_path, prefixed_neg_path)
    """
    d = Path(dest_dir)
    d.mkdir(parents=True, exist_ok=True)

    pref_pos = d / f"{prefix}pos_emb.pt"
    pref_neg = d / f"{prefix}neg_emb.pt"
    leg_pos  = d / "pos_emb.pt"
    leg_neg  = d / "neg_emb.pt"

    # Already there?
    if pref_pos.exists() and pref_neg.exists():
        _ensure_legacy_links(pref_pos, leg_pos)
        _ensure_legacy_links(pref_neg, leg_neg)
        return str(pref_pos), str(pref_neg)

    # If legacy exists, promote to prefixed
    if leg_pos.exists() and not pref_pos.exists():
        shutil.copy2(leg_pos, pref_pos)
    if leg_neg.exists() and not pref_neg.exists():
        shutil.copy2(leg_neg, pref_neg)
    if pref_pos.exists() and pref_neg.exists():
        _ensure_legacy_links(pref_pos, leg_pos)
        _ensure_legacy_links(pref_neg, leg_neg)
        return str(pref_pos), str(pref_neg)

    # Try snapshot_download (pull only the two files)
    if prefer_snapshot:
        cache_dir = d / ".hf_cache_seedvr2_embeds"
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["pos_emb.pt", "neg_emb.pt"],
            resume_download=True,
            token=token,
        )
        if (cache_dir / "pos_emb.pt").exists():
            shutil.copy2(cache_dir / "pos_emb.pt", pref_pos)
        if (cache_dir / "neg_emb.pt").exists():
            shutil.copy2(cache_dir / "neg_emb.pt", pref_neg)

    # Fallback: direct URLs
    if not pref_pos.exists():
        download_url_to_file(_POS_URL, str(pref_pos))
    if not pref_neg.exists():
        download_url_to_file(_NEG_URL, str(pref_neg))

    if not (pref_pos.exists() and pref_neg.exists()):
        raise RuntimeError("Failed to fetch SeedVR2 text embeddings.")

    _ensure_legacy_links(pref_pos, leg_pos)
    _ensure_legacy_links(pref_neg, leg_neg)
    return str(pref_pos), str(pref_neg)


def _ensure_legacy_links(src: Path, dst: Path) -> None:
    """
    Make 'pos_emb.pt' and 'neg_emb.pt' point at the prefixed files (best-effort).
    On platforms without symlink perms, we copy instead.
    """
    if dst.exists():
        return
    try:
        os.symlink(src, dst)  # works on Linux/macOS; may need perms on Windows
    except Exception:
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def ensure_seedvr2_vae_checkpoint(
    target_dir: str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = VAE_FILENAME,
    token: Optional[str] = None,
    lock_timeout: int = 300,
) -> str:
    """
    Ensure the SeedVR2 VAE checkpoint (ema_vae.pth) exists under `target_dir`.
    Returns the absolute local path.
    """
    target_dir = os.path.abspath(target_dir or ".")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)
    if _existing_and_nonempty(target_path):
        return target_path

    lock_path = os.path.join(target_dir, "seedvr2_vae.ckpt.lock")
    try:
        with FileLock(lock_path, timeout=lock_timeout):
            if _existing_and_nonempty(target_path):
                return target_path

            # Try snapshot first (fetch only the single file)
            try:
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[filename],
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                    token=token,
                    resume_download=True,
                )
                downloaded = os.path.join(target_dir, filename)
            except Exception:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                    token=token,
                    resume_download=True,
                )

            if not _existing_and_nonempty(downloaded):
                raise RuntimeError(f"Downloaded VAE not found or empty: {downloaded}")

            if os.path.abspath(downloaded) != os.path.abspath(target_path):
                shutil.copy2(downloaded, target_path)

            return target_path
    except Timeout:
        if _existing_and_nonempty(target_path):
            return target_path
        raise


def ensure_seedvr2_7b_checkpoint(
    target_dir: str = CACHE_DIR,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
    lock_timeout: int = 300,
) -> tuple[str, str, str]:
    """
    Make sure the SeedVR2-7B checkpoint exists locally.

    - If `target_path` exists and is non-empty, returns it immediately.
    - Otherwise downloads {repo_id}/{filename} into `local_dir` (defaults to the
      directory of `target_path`) using `snapshot_download` and/or `hf_hub_download`.
    - Copies the downloaded file to `target_path` if the names differ.

    Returns:
        str: absolute path to the usable checkpoint file.
    """
    target_path = f'{target_dir}/{filename}'
    target_path = os.path.abspath(target_path)
    pos_path, neg_path = ensure_seedvr2_text_embeddings(target_dir)
    if _existing_and_nonempty(target_path):
        return target_path,pos_path, neg_path

    # Figure out where to place the downloaded file
    local_dir = os.path.abspath(local_dir or target_dir or ".")
    os.makedirs(local_dir, exist_ok=True)

    lock_path = os.path.join(local_dir, "seedvr2_7b.ckpt.lock")
    try:
        with FileLock(lock_path, timeout=lock_timeout):
            # Re-check inside the lock in case another process finished meanwhile
            if _existing_and_nonempty(target_path):
                return target_path, pos_path, neg_path

            # First try snapshot_download to put the file directly into local_dir
            try:
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[filename],
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    token=token,
                    resume_download=True,
                )
                downloaded = os.path.join(local_dir, filename)
            except Exception:
                # Fallback to a direct single-file fetch
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    token=token,
                    resume_download=True,
                )

            if not _existing_and_nonempty(downloaded):
                raise RuntimeError(
                    f"Downloaded file not found or empty: {downloaded}"
                )

            # If caller expects a specific name (e.g., .../seedvr2_ema_7b.pth), copy it.
            if os.path.abspath(downloaded) != target_path:
                shutil.copy2(downloaded, target_path)

            return target_path, pos_path, neg_path

    except Timeout:
        # If we couldn't get the lock, best effort: if another process finished, use it.
        if _existing_and_nonempty(target_path):
            return target_path, pos_path, neg_path
        raise Timeout(f"Timed out acquiring lock {lock_path} while downloading SeedVR2-7B checkpoint.")


__all__ = ["ensure_seedvr2_7b_checkpoint"]
