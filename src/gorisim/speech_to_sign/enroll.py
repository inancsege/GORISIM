"""Voice profile enrollment: average ECAPA embeddings across reference clips."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import numpy as np

from gorisim.config import get_settings


class _EmbedProtocol(Protocol):
    def embed(self, audio_path: Path) -> np.ndarray: ...


_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _profiles_dir() -> Path:
    d = get_settings().data_dir / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise ValueError(f"Invalid profile name: {name!r}")


def enroll_user(*, name: str, wav_paths: list[Path], verifier: _EmbedProtocol) -> Path:
    _validate_name(name)
    if not wav_paths:
        raise ValueError("At least one wav file required")
    embeds = [verifier.embed(p) for p in wav_paths]
    avg = np.mean(np.stack(embeds, axis=0), axis=0).astype(np.float32)
    target = _profiles_dir() / f"{name}.npy"
    np.save(target, avg)
    return target


def list_profiles() -> list[str]:
    return sorted(p.stem for p in _profiles_dir().glob("*.npy"))


def load_profile(name: str) -> np.ndarray:
    _validate_name(name)
    return np.load(_profiles_dir() / f"{name}.npy")


def delete_profile(name: str) -> None:
    _validate_name(name)
    target = _profiles_dir() / f"{name}.npy"
    if target.exists():
        target.unlink()
