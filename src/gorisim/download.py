"""Bootstrap downloader for all GORISIM weights and data."""
from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gorisim.config import get_settings


@dataclass(frozen=True)
class Asset:
    name: str
    target: Path
    url: str | None  # None => fetched via huggingface_hub
    sha256: str | None
    fetcher: Callable[[Asset], None] | None = None


def ensure_dirs(models_dir: Path, data_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sign_clips").mkdir(parents=True, exist_ok=True)
    (data_dir / "profiles").mkdir(parents=True, exist_ok=True)


def is_present(asset: Asset) -> bool:
    if not asset.target.exists():
        return False
    if asset.sha256 is None:
        return True
    return _sha256(asset.target) == asset.sha256


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Manifest is built lazily so tests can monkeypatch settings paths.
def _build_manifest() -> list[Asset]:
    s = get_settings()
    m = s.models_dir
    d = s.data_dir
    return [
        Asset(
            name="hrnet_wholebody",
            target=m / "hrnet_w48_coco_wholebody_384x288.pth",
            url="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth",
            sha256=None,
        ),
        Asset(
            name="rgb_final_finetuned",
            target=m / "rgb_final_finetuned.pth",
            # Hosted by jackyjsy/CVPR21Chal-SLR — exact URL filled in download_url_resolver
            url="https://github.com/jackyjsy/CVPR21Chal-SLR/releases/download/v1.0/rgb_final_finetuned.pth",
            sha256=None,
        ),
        Asset(
            name="pyannote_diarization_3_1",
            target=m / "pyannote",
            url=None,  # huggingface_hub
            sha256=None,
        ),
        Asset(
            name="speechbrain_ecapa",
            target=m / "speechbrain" / "spkrec-ecapa-voxceleb",
            url=None,
            sha256=None,
        ),
        Asset(
            name="faster_whisper",
            target=m / "faster-whisper",
            url=None,
            sha256=None,
        ),
        Asset(
            name="autsl_signlist_csv",
            target=d / "SignList_ClassId_TR_EN.csv",
            url="https://raw.githubusercontent.com/jackyjsy/data-prepare/main/SignList_ClassId_TR_EN.csv",
            sha256=None,
        ),
    ]


MANIFEST = _build_manifest()


def main() -> int:
    """CLI entrypoint — implemented in Tasks 6-7."""
    print("download.py — see Tasks 6-7 for the full implementation", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
