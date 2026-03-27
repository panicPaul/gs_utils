"""Minimal capability-driven viewer host scaffold."""

from dataclasses import dataclass, field

from gs_utils.contracts import RenderMode
from gs_utils.contracts.scene import Scene
from gs_utils.viewer.extensions import (
    AlphaViewerExtension,
    DepthViewerExtension,
    GeometryViewerExtension,
    NormalsViewerExtension,
    ViewerExtension,
)


@dataclass
class ViewerHost:
    """Minimal viewer host that filters render modes by scene capabilities."""

    scene: Scene
    extensions: list[ViewerExtension] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.extensions:
            self.extensions = [
                GeometryViewerExtension(),
                DepthViewerExtension(),
                NormalsViewerExtension(),
                AlphaViewerExtension(),
            ]

    def supported_extensions(self) -> list[ViewerExtension]:
        return [ext for ext in self.extensions if ext.supports(self.scene)]

    def supported_render_modes(self) -> list[RenderMode]:
        modes = set(RenderMode.supported_for_scene(self.scene))
        for extension in self.supported_extensions():
            modes |= extension.supported_modes(self.scene)
        return sorted(modes, key=lambda mode: mode.value)
