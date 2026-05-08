"""pyannote 3.1 speaker diarization wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from pyannote.audio import Pipeline

from gorisim.config import get_settings
from gorisim.devices import pick_device


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


class Diarizer:
    def __init__(self, model_dir: Path | None = None):
        s = get_settings()
        model_dir = model_dir or (s.models_dir / "pyannote")
        # pyannote 3.1 publishes multiple files; load via its config.yaml when present
        config = model_dir / "config.yaml"
        if config.exists():
            self.pipeline = Pipeline.from_pretrained(str(config))
        else:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=s.hf_token,
            )
        device = pick_device(s.device)
        self.pipeline.to(torch.device(str(device).replace("mps", "cpu")))

    def diarize(
        self,
        audio_path: Path,
        *,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[Turn]:
        params: dict[str, int] = {}
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers
        annotation = self.pipeline(str(audio_path), **params)
        return [
            Turn(start=float(seg.start), end=float(seg.end), speaker=str(label))
            for seg, _, label in annotation.itertracks(yield_label=True)
        ]
