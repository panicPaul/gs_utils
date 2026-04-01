"""Default viewer extensions."""

from gs_utils.viewer.extensions.alpha import (
    AlphaViewer,
    AlphaViewerRenderState,
)
from gs_utils.viewer.extensions.depth import (
    DepthViewer,
    DepthViewerRenderState,
)
from gs_utils.viewer.extensions.geometry import (
    GeometryViewer,
    GeometryViewerRenderState,
)
from gs_utils.viewer.extensions.normals import (
    NormalsViewer,
    NormalsViewerRenderState,
)

__all__ = [
    "AlphaViewer",
    "AlphaViewerRenderState",
    "DepthViewer",
    "DepthViewerRenderState",
    "GeometryViewer",
    "GeometryViewerRenderState",
    "NormalsViewer",
    "NormalsViewerRenderState",
]
