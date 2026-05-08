"""HRNet whole-body 2D keypoint extraction.

Pure PyTorch — no cupy. Handles multi-scale forward + horizontal-flip merge
following the original CVPR21Chal-SLR pipeline, but with merge_hm fixed
(original code reduced the wrong tensor).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from gorisim.config import get_settings
from gorisim.devices import pick_device
from gorisim.sign_to_text.models.hrnet_config import cfg
from gorisim.sign_to_text.models.pose_hrnet import get_pose_net
from gorisim.sign_to_text.utils import pose_process

# 133 joint indices used to mirror keypoints across the horizontal flip.
INDEX_MIRROR = (
    np.concatenate(
        [
            [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16],
            [21, 22, 23, 18, 19, 20],
            np.arange(40, 23, -1),
            np.arange(50, 40, -1),
            np.arange(51, 55),
            np.arange(59, 54, -1),
            [69, 68, 67, 66, 71, 70],
            [63, 62, 61, 60, 65, 64],
            np.arange(78, 71, -1),
            np.arange(83, 78, -1),
            [88, 87, 86, 85, 84, 91, 90, 89],
            np.arange(113, 134),
            np.arange(92, 113),
        ]
    )
    - 1
)
assert INDEX_MIRROR.shape[0] == 133

MULTI_SCALES = (512, 640)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def merge_hm(hms_list: list[torch.Tensor]) -> torch.Tensor:
    """Average heatmaps across scales and the flipped pair.

    Each input tensor has shape (2, 133, H, W) where the first dim is
    [original, horizontally-flipped]. We re-flip the flipped pair using
    INDEX_MIRROR and take the mean over (scales × 2).
    """
    assert isinstance(hms_list, list)
    fixed: list[torch.Tensor] = []
    for hms in hms_list:
        hms = hms.clone()
        hms[1] = torch.flip(hms[1, INDEX_MIRROR, :, :], dims=[2])
        fixed.append(hms)
    hm = torch.cat(fixed, dim=0)  # (2 * N_scales, 133, H, W)
    return hm.mean(dim=0)  # (133, H, W)


def _norm_to_tensor(img_stacked: np.ndarray) -> torch.Tensor:
    """img_stacked: (2, H, W, 3) uint8 RGB. Returns (2, 3, H, W) float32 normalized."""
    img = img_stacked.astype(np.float32) / 255.0
    for i in range(3):
        img[..., i] = (img[..., i] - MEAN[i]) / STD[i]
    return torch.from_numpy(img).permute(0, 3, 1, 2).contiguous()


def _stack_with_flip(img: np.ndarray) -> np.ndarray:
    return np.stack([img, cv2.flip(img, 1)], axis=0)


class PoseExtractor:
    def __init__(self, weights_path: Path | None = None):
        s = get_settings()
        self.device = pick_device(s.device)
        weights_path = weights_path or (s.models_dir / "hrnet_w48_coco_wholebody_384x288.pth")
        config_yaml = (
            Path(__file__).parent / "models" / "hrnet_config" / "wholebody_w48_384x288.yaml"
        )
        cfg.merge_from_file(str(config_yaml))
        self.model = get_pose_net(cfg, is_train=False)

        ckpt = torch.load(weights_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        new_sd: OrderedDict[str, torch.Tensor] = OrderedDict()
        for k, v in sd.items():
            if "backbone." in k:
                new_sd[k[len("backbone.") :]] = v
            elif "keypoint_head." in k:
                new_sd[k[len("keypoint_head.") :]] = v
            else:
                new_sd[k] = v
        self.model.load_state_dict(new_sd, strict=False)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def extract(self, video_path: Path) -> np.ndarray:
        """Return keypoints array of shape (T, 133, 3) — (x, y, score)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        out: list[np.ndarray] = []
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                h, w = img.shape[:2]
                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hms_list: list[torch.Tensor] = []
                for scale in MULTI_SCALES:
                    img_t = img if scale == 512 else cv2.resize(img, (scale, scale))
                    stacked = _stack_with_flip(img_t)
                    tensor = _norm_to_tensor(stacked).to(self.device)
                    hms = self.model(tensor)
                    if scale != 512:
                        hms = F.interpolate(
                            hms, (w // 4, h // 4), mode="bilinear", align_corners=False
                        )
                    hms_list.append(hms)
                merged = merge_hm(hms_list)
                pred = self._heatmap_to_xy(merged, w, h)
                out.append(pred)
        finally:
            cap.release()
        if not out:
            raise ValueError(f"No frames decoded from: {video_path}")
        return np.stack(out, axis=0)

    @staticmethod
    def _heatmap_to_xy(hm: torch.Tensor, frame_w: int, frame_h: int) -> np.ndarray:
        flat = hm.reshape(133, -1)
        idx = torch.argmax(flat, dim=1).cpu().numpy()
        y = idx // (frame_w // 4)
        x = idx % (frame_w // 4)
        pred = np.zeros((133, 3), dtype=np.float32)
        pred[:, 0] = x
        pred[:, 1] = y
        hm_np = hm.cpu().numpy().reshape((133, frame_h // 4, frame_w // 4))
        pred = pose_process(pred, hm_np)
        pred[:, :2] *= 4.0
        return pred
