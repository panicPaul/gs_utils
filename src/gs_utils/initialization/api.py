"""Public initialization dispatch helpers."""

from gs_utils.config.models import InitializationConfig
from gs_utils.contracts import Splat2DGS, Splat3DGS
from gs_utils.contracts.scene import Scene
from gs_utils.data.contract import ParsedScene
from gs_utils.initialization.builtins import (
    Splat2DGSInitializer,
    Splat3DGSInitializer,
)


def initialize_scene(
    scene: Scene,
    config: InitializationConfig,
    parsed_scene: ParsedScene | None = None,
) -> None:
    """Initialize a scene in place based on its geometry contract."""
    if isinstance(scene, Splat3DGS):
        Splat3DGSInitializer().initialize(scene, config, parsed_scene)
        return
    if isinstance(scene, Splat2DGS):
        Splat2DGSInitializer().initialize(scene, config, parsed_scene)
        return
    raise TypeError(
        f"Unsupported scene type {type(scene)!r} for shared initialization."
    )
