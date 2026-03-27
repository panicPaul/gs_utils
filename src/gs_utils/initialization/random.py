"""Random initialization strategies."""

from gs_utils.config.models import InitializationConfig
from gs_utils.contracts import Splat2DGS, Splat3DGS
from gs_utils.contracts.scene import Scene
from gs_utils.initialization.common import (
    InitContext,
    compute_knn_log_scales,
    init_common_from_points,
    random_points,
)
from gs_utils.initialization.registry import register_init_fn


@register_init_fn("random", scene_type=Splat3DGS)
def init_random_3dgs(
    scene: Scene,
    config: InitializationConfig,
    context: InitContext,
) -> None:
    """Initialize a 3DGS scene from randomly sampled points."""
    points = random_points(
        config.init_num_points,
        config.init_extent * context.scene_scale,
        scene.means.device,
    )
    init_common_from_points(scene, points, None, config)
    scene.log_scales.data = compute_knn_log_scales(points, config, dims=3).to(
        device=scene.log_scales.device, dtype=scene.log_scales.dtype
    )


@register_init_fn("random", scene_type=Splat2DGS)
def init_random_2dgs(
    scene: Scene,
    config: InitializationConfig,
    context: InitContext,
) -> None:
    """Initialize a 2DGS scene from randomly sampled points."""
    points = random_points(
        config.init_num_points,
        config.init_extent * context.scene_scale,
        scene.means.device,
    )
    init_common_from_points(scene, points, None, config)
    scene.log_scales.data = compute_knn_log_scales(points, config, dims=2).to(
        device=scene.log_scales.device, dtype=scene.log_scales.dtype
    )
