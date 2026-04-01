"""Concrete viewer for the vanilla 3DGS example."""

from dataclasses import dataclass

from gs_utils.viewer import (
    AlphaViewer,
    AlphaViewerRenderState,
    DepthViewer,
    DepthViewerRenderState,
    GeometryViewer,
    GeometryViewerRenderState,
    Viewer,
    ViewerRenderState,
)


@dataclass
class VanillaViewerRenderState(
    ViewerRenderState,
    GeometryViewerRenderState,
    DepthViewerRenderState,
    AlphaViewerRenderState,
):
    """Composed render state for the vanilla 3DGS viewer."""


class VanillaViewer(Viewer, GeometryViewer, DepthViewer, AlphaViewer):
    """Viewer for the vanilla 3DGS example."""

    render_state_type = VanillaViewerRenderState
