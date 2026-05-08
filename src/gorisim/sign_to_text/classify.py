"""R(2+1)D-18 sign classifier inference."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from gorisim.config import get_settings
from gorisim.devices import pick_device
from gorisim.sign_to_text.models.resnet2plus1d import r2plus1d_18

NUM_CLASSES = 226
INPUT_SIZE = (100, 100)
CROP_BOX = (16, 16, 240, 240)


class SignClassifier:
    def __init__(self, weights_path: Path | None = None):
        s = get_settings()
        self.device = pick_device(s.device)
        weights_path = weights_path or (s.models_dir / "rgb_final_finetuned.pth")
        self.model = r2plus1d_18(pretrained=False, num_classes=NUM_CLASSES)
        ckpt = torch.load(weights_path, map_location="cpu")
        new_sd = OrderedDict()
        for k, v in ckpt.items():
            new_sd[k[len("module."):] if k.startswith("module.") else k] = v
        self.model.load_state_dict(new_sd, strict=False)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def classify(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Run the classifier on a list of cropped 256x256 BGR frames; return logits (NUM_CLASSES,)."""
        if not frames:
            raise ValueError("No frames given to classifier")
        prepared: list[np.ndarray] = []
        for img in frames:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil = pil.crop(CROP_BOX)
            arr = np.float32(pil)
            arr = cv2.resize(arr, INPUT_SIZE, interpolation=cv2.INTER_AREA)
            arr = arr / 255.0
            prepared.append(arr[np.newaxis, ...])
        # (1, T, H, W, 3) -> (1, 3, T, H, W)
        x = torch.from_numpy(np.stack(prepared, axis=0)).unsqueeze(0)
        x = x.permute(0, 4, 1, 2, 3).to(self.device)
        return self.model(x).squeeze(0)  # (NUM_CLASSES,)

    @staticmethod
    def top_k(logits: torch.Tensor, k: int = 5) -> list[tuple[int, float]]:
        probs = torch.softmax(logits, dim=0)
        scores, idx = torch.topk(probs, k=k)
        return [(int(i), float(s)) for i, s in zip(idx.tolist(), scores.tolist(), strict=True)]
