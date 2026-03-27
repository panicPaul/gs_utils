"""Shared contracts for scenes, capabilities, and render I/O."""

from gs_utils.contracts.capabilities import (
    DifferentiableDepth,
    DifferentiableNormals,
    RendersAlpha,
    RendersDepth,
    RendersNormals,
    Splat2DGS,
    Splat3DGS,
)
from gs_utils.contracts.render import RenderInput, RenderMode, RenderOutput
from gs_utils.contracts.scene import Scene

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
