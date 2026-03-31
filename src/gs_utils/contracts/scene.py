"""Base scene abstraction."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import torch
import torch.nn as nn
from pydantic import BaseModel

from gs_utils.contracts.capabilities import RendersRGB
from gs_utils.data.contract import PointCloud
from gs_utils.contracts.render import RenderInput, RenderMode, RenderOutput

ConfigT = TypeVar("ConfigT", bound=BaseModel)
OptimizerT = TypeVar("OptimizerT", bound=torch.optim.Optimizer)
SchedulerT = TypeVar(
    "SchedulerT",
    bound=torch.optim.lr_scheduler.LRScheduler,
)
DensificationConfigT = TypeVar(
    "DensificationConfigT",
    bound=BaseModel,
)


class Scene(
    nn.Module,
    RendersRGB,
    ABC,
    Generic[ConfigT, OptimizerT, SchedulerT, DensificationConfigT],
):
    """Base scene contract for shared GS utilities."""

    @abstractmethod
    def render(
        self,
        render_input: RenderInput,
        render_mode: RenderMode = RenderMode.RGB,
    ) -> RenderOutput:
        """Render the scene for the provided camera state."""

    def forward(
        self,
        render_input: RenderInput,
        render_mode: RenderMode = RenderMode.RGB,
    ) -> RenderOutput:
        """Alias for render."""
        return self.render(render_input, render_mode)

    def save(self, path: Path | str) -> None:
        """Save the scene to disk."""
        path = Path(path)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path | str) -> "Scene":
        """Load the scene from disk."""
        path = Path(path)
        state_dict = torch.load(path)
        instance = cls()
        instance.load_state_dict(state_dict)
        return instance

    def initialize(
        self,
        config: BaseModel,
        point_cloud: PointCloud | None = None,
        scene_scale: float = 1.0,
    ) -> None:
        """Initialize the scene through the shared initialization registry."""
        from gs_utils.initialization.api import initialize_scene
        from gs_utils.initialization.config import (
            InitializationConfig as SharedInitializationConfig,
        )

        if isinstance(config, SharedInitializationConfig):
            initialization_config = config
        elif hasattr(config, "method") and hasattr(config, "config"):
            method_name = getattr(config, "method")
            method_config = getattr(config, "config")
            if not isinstance(method_name, str):
                raise TypeError("Initialization method must be a string.")
            if not isinstance(method_config, BaseModel):
                raise TypeError(
                    "Initialization config payload must be a pydantic model."
                )
            initialization_config = SharedInitializationConfig(
                strategy=method_name,
                **method_config.model_dump(),
            )
        else:
            raise TypeError(
                "Initialization config must be a shared initialization config "
                "or an example wrapper with `method` and `config` fields."
            )

        initialize_scene(
            self,
            initialization_config,
            point_cloud=point_cloud,
            scene_scale=scene_scale,
        )

    @abstractmethod
    def initialize_optimizers(
        self,
        config: ConfigT,
    ) -> dict[str, OptimizerT]:
        """Initialize optimizers for the scene."""

    @abstractmethod
    def initialize_densification(
        self,
        config: DensificationConfigT,
        scene_scale: float = 1.0,
    ) -> None:
        """Initialize runtime densification state for the scene."""

    @abstractmethod
    def initialize_schedulers(
        self,
        optimizers: dict[str, OptimizerT],
        config: ConfigT,
    ) -> dict[str, SchedulerT]:
        """Initialize learning rate schedulers for the scene."""

    def optimizer_step(
        self,
        optimizers: dict[str, OptimizerT],
        schedulers: dict[str, SchedulerT],
    ) -> None:
        """Perform an optimization step for all optimizers and schedulers associated with the scene."""
        for name, optimizer in optimizers.items():
            scheduler = schedulers.get(name)
            if scheduler is not None:
                scheduler.step()
            optimizer.step()

    def optimizer_zero_grad(
        self,
        optimizers: dict[str, OptimizerT],
    ) -> None:
        """Zero the gradients of all optimizers associated with the scene."""
        for optimizer in optimizers.values():
            optimizer.zero_grad()

    @abstractmethod
    def densification_step_pre_backward(
        self,
        iteration: int,
        optimizers: dict[str, OptimizerT],
    ) -> None:
        """Pre-backward step for densification."""

    @abstractmethod
    def densification_step_post_backward(
        self,
        iteration: int,
        optimizers: dict[str, OptimizerT],
    ) -> None:
        """Post-backward step for densification."""
