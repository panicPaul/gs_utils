"""Built-in geometry-aware scene initializers."""

import math

import torch

from gs_utils.config.models import InitializationConfig
from gs_utils.contracts.scene import Scene
from gs_utils.data.contract import ParsedScene
from gs_utils.initialization.base import SceneInitializer


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def _random_points(
    count: int, extent: float, device: torch.device
) -> torch.Tensor:
    return (torch.rand((count, 3), device=device) * 2.0 - 1.0) * extent


class _BaseSplatInitializer(SceneInitializer):
    def _load_checkpoint(self, scene: Scene, path: str) -> None:
        checkpoint = torch.load(
            path, map_location=next(scene.parameters()).device
        )
        state_dict = checkpoint.get("scene", checkpoint)
        scene.load_state_dict(state_dict)

    def _init_common_from_points(
        self,
        scene: Scene,
        points: torch.Tensor,
        colors: torch.Tensor | None,
        config: InitializationConfig,
    ) -> None:
        scene.means.data = points
        scene.opacities.data.fill_(
            torch.logit(torch.tensor(config.init_opacity))
        )
        scene.quats.data.zero_()
        scene.quats.data[:, 0] = 1.0
        if colors is not None and hasattr(scene, "sh0"):
            scene.sh0.data = _rgb_to_sh(colors)
        if hasattr(scene, "shN"):
            scene.shN.data.zero_()


class Splat3DGSInitializer(_BaseSplatInitializer):
    """Built-in initializer for 3DGS-style scenes."""

    def initialize(
        self,
        scene: Scene,
        config: InitializationConfig,
        parsed_scene: ParsedScene | None = None,
    ) -> None:
        device = scene.means.device
        if config.strategy == "checkpoint":
            self._load_checkpoint(scene, config.checkpoint_path or "")
            return
        if config.strategy in {"sfm", "point_cloud"}:
            if parsed_scene is None:
                raise ValueError(
                    "parsed_scene is required for sfm/point_cloud initialization."
                )
            if parsed_scene.point_cloud is None:
                raise ValueError(
                    "parsed_scene.point_cloud is required for sfm/point_cloud initialization."
                )
            points = parsed_scene.point_cloud.positions.to(device=device)
            colors = (
                None
                if parsed_scene.point_cloud.colors is None
                else parsed_scene.point_cloud.colors.to(device=device)
            )
            self._init_common_from_points(scene, points, colors, config)
            scene.log_scales.data.fill_(math.log(config.init_scale))
            return
        if config.strategy == "random":
            points = _random_points(
                config.init_num_points, config.init_extent, device
            )
            self._init_common_from_points(scene, points, None, config)
            scene.log_scales.data.fill_(math.log(config.init_scale))
            return
        raise ValueError(
            f"Unsupported initialization strategy: {config.strategy}"
        )


class Splat2DGSInitializer(_BaseSplatInitializer):
    """Built-in initializer for 2DGS-style scenes."""

    def initialize(
        self,
        scene: Scene,
        config: InitializationConfig,
        parsed_scene: ParsedScene | None = None,
    ) -> None:
        device = scene.means.device
        if config.strategy == "checkpoint":
            self._load_checkpoint(scene, config.checkpoint_path or "")
            return
        if config.strategy in {"sfm", "point_cloud"}:
            if parsed_scene is None:
                raise ValueError(
                    "parsed_scene is required for sfm/point_cloud initialization."
                )
            if parsed_scene.point_cloud is None:
                raise ValueError(
                    "parsed_scene.point_cloud is required for sfm/point_cloud initialization."
                )
            points = parsed_scene.point_cloud.positions.to(device=device)
            colors = (
                None
                if parsed_scene.point_cloud.colors is None
                else parsed_scene.point_cloud.colors.to(device=device)
            )
            self._init_common_from_points(scene, points, colors, config)
            scene.log_scales.data[:, :2].fill_(math.log(config.init_scale))
            scene.log_scales.data[:, 2].zero_()
            return
        if config.strategy == "random":
            points = _random_points(
                config.init_num_points, config.init_extent, device
            )
            self._init_common_from_points(scene, points, None, config)
            scene.log_scales.data[:, :2].fill_(math.log(config.init_scale))
            scene.log_scales.data[:, 2].zero_()
            return
        raise ValueError(
            f"Unsupported initialization strategy: {config.strategy}"
        )
