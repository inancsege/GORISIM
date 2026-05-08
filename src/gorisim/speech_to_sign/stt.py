"""Whisper-based Turkish ASR via faster-whisper (CTranslate2)."""
from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from gorisim.config import get_settings


class TurkishASR:
    def __init__(self, model_dir: Path | None = None):
        s = get_settings()
        model_dir = model_dir or (s.models_dir / "faster-whisper")
        if not model_dir.exists() or not any(model_dir.iterdir()):
            raise FileNotFoundError(
                f"Whisper model directory is empty: {model_dir}. "
                "Run `python -m gorisim.download` first."
            )
        device = "cuda" if s.device == "cuda" else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path) -> str:
        segments, _info = self.model.transcribe(  # pyright: ignore[reportUnknownMemberType]
            str(audio_path), language="tr", beam_size=5
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
