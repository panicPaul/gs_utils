"""Default geometry viewer extension."""

from dataclasses import dataclass

from gs_utils.contracts import RenderMode, Splat2DGS, Splat3DGS


@dataclass
class GeometryViewerRenderState:
    """Render state for the geometry viewer extension."""


class GeometryViewer:
    """Default viewer extension for splat geometry scenes."""

    def _viewer_supported_render_modes(self) -> set[RenderMode]:
        """Return geometry render modes supported by the current scene."""
        if isinstance(self.scene, (Splat3DGS, Splat2DGS)):
            return {RenderMode.RGB}
        return set()
