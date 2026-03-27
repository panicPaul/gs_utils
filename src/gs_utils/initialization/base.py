"""Base contracts for scene initializers."""

from abc import ABC, abstractmethod

from gs_utils.config.models import InitializationConfig
from gs_utils.contracts.scene import Scene
from gs_utils.data.contract import ParsedScene


class SceneInitializer(ABC):
    """Base in-place scene initializer."""

    @abstractmethod
    def initialize(
        self,
        scene: Scene,
        config: InitializationConfig,
        parsed_scene: ParsedScene | None = None,
    ) -> None:
        """Mutate a scene in place according to the initialization config."""
