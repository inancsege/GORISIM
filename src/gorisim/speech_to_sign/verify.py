"""speechbrain ECAPA speaker verification."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import SpeakerRecognition

from gorisim.config import get_settings


class SpeakerVerifier:
    def __init__(self, model_dir: Path | None = None):
        s = get_settings()
        model_dir = model_dir or (s.models_dir / "speechbrain" / "spkrec-ecapa-voxceleb")
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(model_dir),
        )

    def embed(self, audio_path: Path) -> np.ndarray:
        signal, _fs = self.model.load_audio(str(audio_path))
        embedding = self.model.encode_batch(signal.unsqueeze(0))
        return embedding.squeeze().detach().cpu().numpy()

    def matches(self, audio_path: Path, profile_embedding: np.ndarray, *, threshold: float = 0.25) -> bool:
        clip_embedding = self.embed(audio_path)
        a = torch.from_numpy(profile_embedding)
        b = torch.from_numpy(clip_embedding)
        sim = float(torch.cosine_similarity(a, b, dim=0))
        return sim >= threshold
