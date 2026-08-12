"""Config loading and device selection."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device(requested: str = "auto") -> torch.device:
    """Resolve 'auto' to cuda > mps > cpu, or honor an explicit request."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
