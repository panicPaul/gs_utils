"""Shared helpers for scene initialization."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from gs_utils.contracts import (
    SphericalHarmonics2DGS,
    SphericalHarmonics3DGS,
    Splat2DGS,
    Splat3DGS,
)
from gs_utils.data.contract import PointCloud
from gs_utils.initialization.config import InitializationConfig
from gs_utils.utils.neighbors import knn
from gs_utils.utils.spherical_harmonics import rgb_to_sh


@dataclass(slots=True, frozen=True)
class InitContext:
    """Shared inputs available to initialization strategies."""

    point_cloud: PointCloud | None = None
    scene_scale: float = 1.0

InitScene = Splat3DGS | Splat2DGS
InitFn = Callable[[InitScene, InitializationConfig, InitContext], None]


def random_points(
    count: int, extent: float, device: torch.device
) -> torch.Tensor:
    """Sample uniformly distributed random 3D points within the init extent."""
    return (torch.rand((count, 3), device=device) * 2.0 - 1.0) * extent


def random_quats(count: int, device: torch.device) -> torch.Tensor:
    """Sample random quaternion parameters for initialization."""
    return torch.rand((count, 4), device=device)


def load_checkpoint_into_scene(scene: InitScene, path: str) -> None:
    """Load a checkpoint state dict into a scene module."""
    checkpoint = torch.load(path, map_location=next(scene.parameters()).device)
    state_dict = checkpoint.get("scene", checkpoint)
    scene.load_state_dict(state_dict)


def require_point_cloud(point_cloud: PointCloud | None) -> PointCloud:
    """Require a point cloud for initialization strategies that depend on one."""
    if point_cloud is None:
        raise ValueError(
            "point_cloud is required for sfm/point_cloud initialization."
        )
    return point_cloud


def init_common_from_points(
    scene: Splat3DGS | Splat2DGS,
    points: torch.Tensor,
    colors: torch.Tensor | None,
    config: InitializationConfig,
) -> None:
    """Initialize the shared scene parameters from point and color inputs."""
    scene.means.data = points
    scene.logit_opacities.data.fill_(
        torch.logit(
            torch.tensor(
                config.init_opacity,
                device=scene.means.device,
                dtype=scene.means.dtype,
            )
        )
    )
    scene.unnormalized_quats.data = random_quats(
        points.shape[0],
        points.device,
    )
    if colors is not None and isinstance(
        scene, SphericalHarmonics3DGS | SphericalHarmonics2DGS
    ):
        scene.sh_0.data = (
            rgb_to_sh(colors)
            .unsqueeze(1)
            .to(
                device=scene.sh_0.device,
                dtype=scene.sh_0.dtype,
            )
        )
        scene.sh_N.data.zero_()


def point_cloud_inputs(
    point_cloud: PointCloud | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Move point-cloud positions and colors onto the target device."""
    required = require_point_cloud(point_cloud)
    points = required.positions.to(device=device)
    colors = (
        None if required.colors is None else required.colors.to(device=device)
    )
    return points, colors


def compute_knn_log_scales(
    points: torch.Tensor, config: InitializationConfig, dims: int
) -> torch.Tensor:
    """Compute KNN-based log-scales with the requested output dimensionality."""
    average_squared_neighbor_distance = (
        knn(points.detach(), 4)[:, 1:] ** 2
    ).mean(dim=-1)
    average_neighbor_distance = torch.sqrt(average_squared_neighbor_distance)
    return (
        torch.log(average_neighbor_distance * config.init_scale)
        .unsqueeze(-1)
        .repeat(1, dims)
    )
