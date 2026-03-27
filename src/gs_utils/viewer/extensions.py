"""Capability-gated viewer extensions."""

from dataclasses import dataclass, field
from typing import Protocol

from gs_utils.contracts import (
    RenderMode,
    RendersAlpha,
    RendersDepth,
    RendersNormals,
    Splat2DGS,
    Splat3DGS,
)
from gs_utils.contracts.scene import Scene


class ViewerExtension(Protocol):
    """Minimal viewer extension contract."""

    name: str

    def supports(self, scene: Scene) -> bool: ...

    def supported_modes(self, scene: Scene) -> set[RenderMode]: ...


@dataclass(slots=True)
class GeometryViewerExtension:
    """Enable geometry-specific tools."""

    name: str = "geometry"
    modes: set[RenderMode] = field(default_factory=lambda: {RenderMode.RGB})

    def supports(self, scene: Scene) -> bool:
        return isinstance(scene, (Splat3DGS, Splat2DGS))

    def supported_modes(self, scene: Scene) -> set[RenderMode]:
        return set(self.modes)


@dataclass(slots=True)
class DepthViewerExtension:
    """Enable depth-specific viewer features."""

    name: str = "depth"

    def supports(self, scene: Scene) -> bool:
        return isinstance(scene, RendersDepth)

    def supported_modes(self, scene: Scene) -> set[RenderMode]:
        return {
            RenderMode.DEPTH,
            RenderMode.RGB_DEPTH,
            RenderMode.RGB_DEPTH_NORMALS,
            RenderMode.RGB_DEPTH_ALPHA,
            RenderMode.RGB_DEPTH_NORMALS_ALPHA,
        }


@dataclass(slots=True)
class NormalsViewerExtension:
    """Enable normals-specific viewer features."""

    name: str = "normals"

    def supports(self, scene: Scene) -> bool:
        return isinstance(scene, RendersNormals)

    def supported_modes(self, scene: Scene) -> set[RenderMode]:
        return {
            RenderMode.NORMALS,
            RenderMode.RGB_NORMALS,
            RenderMode.RGB_DEPTH_NORMALS,
            RenderMode.RGB_NORMALS_ALPHA,
            RenderMode.RGB_DEPTH_NORMALS_ALPHA,
        }


@dataclass(slots=True)
class AlphaViewerExtension:
    """Enable alpha-specific viewer features."""

    name: str = "alpha"

    def supports(self, scene: Scene) -> bool:
        return isinstance(scene, RendersAlpha)

    def supported_modes(self, scene: Scene) -> set[RenderMode]:
        return {
            RenderMode.ALPHA,
            RenderMode.RGB_ALPHA,
            RenderMode.RGB_DEPTH_ALPHA,
            RenderMode.RGB_NORMALS_ALPHA,
            RenderMode.RGB_DEPTH_NORMALS_ALPHA,
        }
