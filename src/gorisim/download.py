"""Bootstrap downloader for all GORISIM weights and data."""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]
from tqdm import tqdm

from gorisim.config import get_settings


@dataclass(frozen=True)
class Asset:
    name: str
    target: Path
    url: str | None  # None => fetched via huggingface_hub
    sha256: str | None


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


# `_build_manifest()` is exposed so callers can rebuild the manifest after settings change;
# the module-level `MANIFEST` is a snapshot taken at import.
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


def require_hf_token() -> str:
    s = get_settings()
    if not s.hf_token:
        print(
            "ERROR: HF_TOKEN is not set.\n"
            "1. Create a token at https://huggingface.co/settings/tokens\n"
            "2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "3. Put it in your .env file: HF_TOKEN=hf_...",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return s.hf_token


def fetch_http(*, url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", "0"))
        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(target)


def fetch_hf_repo(*, repo_id: str, target: Path, allow_patterns: list[str] | None = None) -> None:
    token = require_hf_token()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        token=token,
        allow_patterns=allow_patterns,
    )


def fetch_hf_whisper_ct2(*, model_size: str, target: Path) -> None:
    """faster-whisper model published by Systran/community as CTranslate2 repos."""
    token = require_hf_token()
    repo_id = f"Systran/faster-whisper-{model_size}"
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target), token=token)


def main() -> int:
    s = get_settings()
    ensure_dirs(s.models_dir, s.data_dir)

    manifest = _build_manifest()
    for asset in manifest:
        if is_present(asset):
            print(f"[skip] {asset.name} already present")
            continue
        print(f"[fetch] {asset.name}")
        if asset.name == "pyannote_diarization_3_1":
            fetch_hf_repo(repo_id="pyannote/speaker-diarization-3.1", target=asset.target)
        elif asset.name == "speechbrain_ecapa":
            fetch_hf_repo(repo_id="speechbrain/spkrec-ecapa-voxceleb", target=asset.target)
        elif asset.name == "faster_whisper":
            fetch_hf_whisper_ct2(model_size=s.whisper_model, target=asset.target)
        elif asset.url is not None:
            fetch_http(url=asset.url, target=asset.target)
        else:
            raise RuntimeError(f"No fetcher for asset: {asset.name}")
    print("All assets present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
