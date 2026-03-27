"""Shared helpers for scene initialization."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from gs_utils.config.models import InitializationConfig
from gs_utils.contracts.scene import Scene
from gs_utils.data.contract import PointCloud
from gs_utils.utils import knn, rgb_to_sh


@dataclass(slots=True, frozen=True)
class InitContext:
    """Shared inputs available to initialization strategies."""

    point_cloud: PointCloud | None = None
    scene_scale: float = 1.0


InitFn = Callable[[Scene, InitializationConfig, InitContext], None]


def random_points(
    count: int, extent: float, device: torch.device
) -> torch.Tensor:
    """Sample uniformly distributed random 3D points within the init extent."""
    return (torch.rand((count, 3), device=device) * 2.0 - 1.0) * extent


def random_quats(count: int, device: torch.device) -> torch.Tensor:
    """Sample random quaternion parameters for initialization."""
    return torch.rand((count, 4), device=device)


def load_checkpoint_into_scene(scene: Scene, path: str) -> None:
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
    scene: Scene,
    points: torch.Tensor,
    colors: torch.Tensor | None,
    config: InitializationConfig,
) -> None:
    """Initialize the shared scene parameters from point and color inputs."""
    scene.means.data = points
    scene.opacities.data.fill_(torch.logit(torch.tensor(config.init_opacity)))
    scene.quats.data = random_quats(points.shape[0], points.device)
    if colors is not None and hasattr(scene, "sh0"):
        scene.sh0.data = rgb_to_sh(colors)
    if hasattr(scene, "shN"):
        scene.shN.data.zero_()


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
    dist2_avg = (knn(points.detach(), 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    return torch.log(dist_avg * config.init_scale).unsqueeze(-1).repeat(1, dims)
