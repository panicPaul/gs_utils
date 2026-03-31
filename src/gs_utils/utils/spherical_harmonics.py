"""Spherical harmonics helper functions."""

from torch import Tensor


def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert RGB values in `[0, 1]` to the DC spherical harmonics coefficient."""
    spherical_harmonic_dc_constant = 0.28209479177387814
    return (rgb - 0.5) / spherical_harmonic_dc_constant
