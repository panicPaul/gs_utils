"""Random initialization strategies."""

from gs_utils.contracts import Splat2DGS, Splat3DGS
from gs_utils.initialization.common import (
    InitContext,
    compute_knn_log_scales,
    init_common_from_points,
    random_points,
)
from gs_utils.initialization.config import InitializationConfig
from gs_utils.initialization.registry import register_init_fn


@register_init_fn("random", scene_type=Splat3DGS)
def init_random_3dgs(
    scene: Splat3DGS,
    config: InitializationConfig,
    context: InitContext,
) -> None:
    """Initialize a 3DGS scene from randomly sampled points."""
    random_point_positions = random_points(
        config.init_num_points,
        config.init_extent * context.scene_scale,
        scene.means.device,
    )
    init_common_from_points(scene, random_point_positions, None, config)
    computed_log_scales = compute_knn_log_scales(
        random_point_positions,
        config,
        dims=3,
    )
    scene.log_scales.data = computed_log_scales.to(
        device=scene.log_scales.device,
        dtype=scene.log_scales.dtype,
    )


@register_init_fn("random", scene_type=Splat2DGS)
def init_random_2dgs(
    scene: Splat2DGS,
    config: InitializationConfig,
    context: InitContext,
) -> None:
    """Initialize a 2DGS scene from randomly sampled points."""
    random_point_positions = random_points(
        config.init_num_points,
        config.init_extent * context.scene_scale,
        scene.means.device,
    )
    init_common_from_points(scene, random_point_positions, None, config)
    computed_log_scales = compute_knn_log_scales(
        random_point_positions,
        config,
        dims=2,
    )
    scene.log_scales.data = computed_log_scales.to(
        device=scene.log_scales.device,
        dtype=scene.log_scales.dtype,
    )
