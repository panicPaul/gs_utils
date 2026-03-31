"""Checkpoint-based initialization strategies."""

from gs_utils.contracts import Splat2DGS, Splat3DGS
from gs_utils.initialization.common import (
    InitContext,
    load_checkpoint_into_scene,
)
from gs_utils.initialization.config import InitializationConfig
from gs_utils.initialization.registry import register_init_fn


@register_init_fn("checkpoint", scene_type=Splat3DGS)
@register_init_fn("checkpoint", scene_type=Splat2DGS)
def init_from_checkpoint(
    scene: Splat3DGS | Splat2DGS,
    config: InitializationConfig,
    _context: InitContext,
) -> None:
    """Initialize a scene from a checkpoint."""
    load_checkpoint_into_scene(scene, str(config.checkpoint_path or ""))
