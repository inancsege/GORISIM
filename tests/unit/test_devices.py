from unittest.mock import patch

import torch

from gorisim.devices import pick_device


def test_explicit_cpu_override():
    assert pick_device("cpu") == torch.device("cpu")


def test_auto_picks_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert pick_device("auto") == torch.device("cuda")


def test_auto_picks_mps_when_no_cuda():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True):
        assert pick_device("auto") == torch.device("mps")


def test_auto_falls_back_to_cpu():
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        assert pick_device("auto") == torch.device("cpu")


def test_invalid_override_raises():
    import pytest
    with pytest.raises(ValueError):
        pick_device("tpu")
