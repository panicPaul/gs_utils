"""Typed parsed-scene metadata and dataset samples."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from gs_utils.contracts import RenderInput


def _identity_transform() -> Float[Tensor, "4 4"]:
    return torch.eye(4, dtype=torch.float32)


@dataclass(slots=True)
class PointCloud:
    """Optional sparse scene geometry metadata."""

    positions: Float[Tensor, "num_points 3"]
    colors: Float[Tensor, "num_points 3"] | None = None


@dataclass(slots=True)
class SceneFrame:
    """Frame-level parsed-scene metadata."""

    render_input: RenderInput
    image_path: Path
    camera_id: int | None = None
    mask: Bool[Tensor, "height width"] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedScene:
    """Backend-agnostic parsed scene metadata."""

    frames: list[SceneFrame]
    scene_scale: float = 1.0
    normalization_transform: Float[Tensor, "4 4"] = field(
        default_factory=_identity_transform
    )
    point_cloud: PointCloud | None = None


@dataclass(slots=True, kw_only=True)
class DataSample:
    """Base typed sample for training and evaluation datasets."""

    render_input: RenderInput
    image: Float[Tensor, "height width 3"]
    image_path: Path | None = None
    camera_id: int | None = None
    mask: Bool[Tensor, "height width"] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class DepthSample(DataSample):
    """Dataset sample with depth supervision."""

    depth: Float[Tensor, "height width 1"]


@dataclass(slots=True, kw_only=True)
class NormalSample(DataSample):
    """Dataset sample with normal supervision."""

    normals: Float[Tensor, "height width 3"]


@dataclass(slots=True, kw_only=True)
class DepthNormalSample(DepthSample):
    """Dataset sample with both depth and normal supervision."""

    normals: Float[Tensor, "height width 3"]
