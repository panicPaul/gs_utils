"""Random number generation helpers."""

import random

import numpy as np
import torch

_NUMPY_RNG = np.random.default_rng()


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    global _NUMPY_RNG

    random.seed(seed)
    _NUMPY_RNG = np.random.default_rng(seed)
    torch.manual_seed(seed)


def get_numpy_rng() -> np.random.Generator:
    """Return the shared NumPy random generator used by the examples."""
    return _NUMPY_RNG
