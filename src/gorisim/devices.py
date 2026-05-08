import torch
from torch import device as _device  # pyright: ignore[reportPrivateImportUsage]

_VALID = {"auto", "cpu", "mps", "cuda"}


def pick_device(override: str = "auto") -> _device:
    if override not in _VALID:
        raise ValueError(f"Invalid device override: {override!r}. Allowed: {sorted(_VALID)}")
    if override != "auto":
        return _device(override)
    if torch.cuda.is_available():
        return _device("cuda")
    if torch.backends.mps.is_available():
        return _device("mps")
    return _device("cpu")
