"""Public initialization dispatch helpers."""

from gs_utils.config.models import InitializationConfig
from gs_utils.contracts.scene import Scene
from gs_utils.data.contract import PointCloud
from gs_utils.initialization import checkpoint as _checkpoint  # noqa: F401
from gs_utils.initialization import point_cloud as _point_cloud  # noqa: F401
from gs_utils.initialization import random as _random  # noqa: F401
from gs_utils.initialization.common import InitContext
from gs_utils.initialization.registry import INIT_FNS


def initialize_scene(
    scene: Scene,
    config: InitializationConfig,
    point_cloud: PointCloud | None = None,
    scene_scale: float = 1.0,
) -> None:
    """Initialize a scene in place based on its geometry contract."""
    registrations = INIT_FNS.get(config.strategy)
    if registrations is None:
        raise ValueError(
            f"Unsupported initialization strategy: {config.strategy}"
        )
    context = InitContext(point_cloud=point_cloud, scene_scale=scene_scale)
    for registration in registrations:
        if isinstance(scene, registration.scene_type):
            registration.init_fn(scene, config, context)
            return
    raise TypeError(
        f"Initialization strategy {config.strategy!r} does not support "
        f"scene type {type(scene)!r}."
    )
