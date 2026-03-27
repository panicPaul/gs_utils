"""Capability-driven viewer scaffolding."""

from gs_utils.viewer.extensions import (
    AlphaViewerExtension,
    DepthViewerExtension,
    GeometryViewerExtension,
    NormalsViewerExtension,
    ViewerExtension,
)
from gs_utils.viewer.host import ViewerHost

__all__ = [
    "AlphaViewerExtension",
    "DepthViewerExtension",
    "GeometryViewerExtension",
    "NormalsViewerExtension",
    "ViewerExtension",
    "ViewerHost",
]
