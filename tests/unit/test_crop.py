import numpy as np
from gorisim.sign_to_text.crop import SELECTED_JOINTS, compute_crop_window


def test_selected_joints_count():
    assert len(SELECTED_JOINTS) == 31  # 11 body + 10 right hand + 10 left hand


def test_compute_crop_window_returns_center_and_radius():
    keypoints = np.zeros((10, 133, 3), dtype=np.float32)
    keypoints[:, :, 0] = 100  # all x = 100
    keypoints[:, :, 1] = 100  # all y = 100
    center, radius = compute_crop_window(keypoints, frame_size=512)
    assert isinstance(center, np.ndarray)
    assert isinstance(radius, np.ndarray)
    assert center.shape == (2,)
    assert radius.shape == (2,) or radius.ndim == 0


def test_compute_crop_window_handles_motion():
    keypoints = np.zeros((10, 133, 3), dtype=np.float32)
    keypoints[:, :, 0] = np.linspace(50, 200, 10).reshape(-1, 1)
    keypoints[:, :, 1] = np.linspace(50, 200, 10).reshape(-1, 1)
    center, _ = compute_crop_window(keypoints, frame_size=512)
    # After the pipeline's mirror (frame_size - x), x range [50, 200] maps to
    # [312, 462], so center_x is the midpoint minus pad ≈ 367.
    assert 300 < center[0] < 400
