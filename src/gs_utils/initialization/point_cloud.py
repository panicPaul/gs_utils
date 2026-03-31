"""Point-cloud-based initialization strategies."""

from gs_utils.contracts import Splat2DGS, Splat3DGS
from gs_utils.initialization.common import (
    InitContext,
    compute_knn_log_scales,
    init_common_from_points,
    point_cloud_inputs,
)
from gs_utils.initialization.config import InitializationConfig
from gs_utils.initialization.registry import register_init_fn


@register_init_fn("point_cloud", scene_type=Splat3DGS)
def init_3dgs_from_point_cloud(
    scene: Splat3DGS,
    config: InitializationConfig,
    context: InitContext,
) -> None:
    """Initialize a 3DGS scene from a point cloud."""
    points, colors = point_cloud_inputs(context.point_cloud, scene.means.device)
    init_common_from_points(scene, points, colors, config)
    computed_log_scales = compute_knn_log_scales(points, config, dims=3)
    scene.log_scales.data = computed_log_scales.to(
        device=scene.log_scales.device,
        dtype=scene.log_scales.dtype,
    )


@register_init_fn("point_cloud", scene_type=Splat2DGS)
def init_2dgs_from_point_cloud(
    scene: Splat2DGS,
    config: InitializationConfig,
    context: InitContext,
) -> None:
    """Initialize a 2DGS scene from a point cloud."""
    points, colors = point_cloud_inputs(context.point_cloud, scene.means.device)
    init_common_from_points(scene, points, colors, config)
    computed_log_scales = compute_knn_log_scales(points, config, dims=2)
    scene.log_scales.data = computed_log_scales.to(
        device=scene.log_scales.device,
        dtype=scene.log_scales.dtype,
    )
