"""Signer-centered crop computation given full-body keypoints."""
from __future__ import annotations

import cv2
import numpy as np

# Indices into the 133-keypoint COCO-WholeBody output, preserved from
# the original CVPR21Chal-SLR pipeline.
SELECTED_JOINTS = np.concatenate((
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [91, 95, 96, 99, 100, 103, 104, 107, 108, 111],
    [112, 116, 117, 120, 121, 124, 125, 128, 129, 132],
), axis=0)


def compute_crop_window(
    keypoints: np.ndarray,
    *,
    frame_size: int = 512,
    pad: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Given (T, 133, 3) keypoints, return (center, radius) for a square crop.

    The crop is centered between the maximum and minimum coords of the
    selected joints across all frames.
    """
    if keypoints.ndim != 3 or keypoints.shape[1] != 133:
        raise ValueError(f"keypoints must have shape (T, 133, 3), got {keypoints.shape}")
    sub = keypoints[:, SELECTED_JOINTS, :2]  # (T, 31, 2)
    sub = sub.copy()
    sub[..., 0] = frame_size - sub[..., 0]  # match original mirror logic
    xy_max = sub.max(axis=1).max(axis=0)  # (2,)
    xy_min = sub.min(axis=1).min(axis=0)
    center = (xy_max + xy_min) / 2 - pad
    radius = (xy_max - center).max()
    return center.astype(np.float32), np.array([radius, radius], dtype=np.float32)


def crop_frame(
    image: np.ndarray,
    center: np.ndarray,
    radius: np.ndarray,
    *,
    frame_size: int = 512,
    scale: float = 1.3,
) -> np.ndarray:
    """Crop and pad a single frame to a square around the signer."""
    radius_crop = (radius * scale).astype(np.int32)
    center_crop = center.astype(np.int32)
    rect = (
        max(0, (center_crop - radius_crop)[0]),
        max(0, (center_crop - radius_crop)[1]),
        min(frame_size, (center_crop + radius_crop)[0]),
        min(frame_size, (center_crop + radius_crop)[1]),
    )
    image = image[rect[1]:rect[3], rect[0]:rect[2], :]
    h, w = image.shape[:2]
    if h < w:
        top = abs(h - w) // 2
        bottom = abs(h - w) - top
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif h > w:
        left = abs(h - w) // 2
        right = abs(h - w) - left
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return image
