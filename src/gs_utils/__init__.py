"""Composable utilities and contracts for Gaussian-splatting-style projects."""

from gs_utils.contracts import (
    DifferentiableDepth,
    DifferentiableNormals,
    RenderInput,
    RenderMode,
    RenderOutput,
    RendersAlpha,
    RendersDepth,
    RendersNormals,
    Scene,
    Splat2DGS,
    Splat3DGS,
)

__all__ = [
    "DifferentiableDepth",
    "DifferentiableNormals",
    "RenderInput",
    "RenderMode",
    "RenderOutput",
    "RendersAlpha",
    "RendersDepth",
    "RendersNormals",
    "Scene",
    "Splat2DGS",
    "Splat3DGS",
]
