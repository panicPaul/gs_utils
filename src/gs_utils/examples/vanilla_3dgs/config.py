"""Config for vanilla 3DGS example."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gs_utils.data.config import DatasetConfig, DataSourceConfig
from gs_utils.initialization.api import INIT_FNS
from gs_utils.initialization.config import (
    InitializationConfig as SharedInitializationConfig,
)


class _ConfigModel(BaseModel):
    """Base config model with strict field validation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class ExponentialDecaySchedulerConfig(_ConfigModel):
    """Configuration for an exponential decay learning-rate scheduler."""

    final_ratio: float = 0.01
    max_steps: int = 30_000

    def gamma(self) -> float:
        """Return the exponential decay factor per optimization step."""
        return self.final_ratio ** (1.0 / self.max_steps)

    def scale_by_factor(self, scale_factor: float) -> None:
        """Scale the scheduler horizon by the provided factor."""
        self.max_steps = max(1, round(self.max_steps * scale_factor))


class OptimizationConfig(_ConfigModel):
    """Optimizer and scheduler configuration for vanilla 3DGS."""

    means_lr: float = 1.6e-4
    log_scales_lr: float = 5e-3
    logit_opacities_lr: float = 5e-2
    unnormalized_quats_lr: float = 1e-3
    sh_0_lr: float = 2.5e-3
    sh_N_lr: float = 2.5e-3 / 20.0
    means_scheduler: ExponentialDecaySchedulerConfig = Field(
        default_factory=ExponentialDecaySchedulerConfig
    )

    def scale_to_max_steps(self, new_max_steps: int) -> None:
        """Scale all schedule horizons to a new optimization horizon."""
        if new_max_steps <= 0:
            raise ValueError("new_max_steps must be positive.")
        scale_factor = new_max_steps / self.means_scheduler.max_steps
        self.means_scheduler.scale_by_factor(scale_factor)


class DensificationConfig(_ConfigModel):
    """Configuration for vanilla 3DGS densification."""

    enabled: bool = True
    reference_training_steps: int = 30_000
    prune_opacity_threshold: float = 0.005
    image_plane_gradient_magnitude_threshold: float = 0.0002
    duplicate_max_normalized_scale_3d: float = 0.01
    split_max_normalized_radius_2d: float = 0.05
    prune_max_normalized_scale_3d: float = 0.1
    prune_max_normalized_radius_2d: float = 0.15
    screen_space_refinement_stop_iteration: int = 0
    refinement_start_iteration: int = 500
    refinement_stop_iteration: int = 15_000
    opacity_reset_interval: int = 3_000
    refinement_interval: int = 100
    refinement_pause_after_opacity_reset: int = 0
    use_absolute_image_plane_gradients: bool = False
    use_revised_opacity_after_split: bool = False
    verbose: bool = False

    def scale_to_max_steps(self, new_max_steps: int) -> None:
        """Scale densification cadence values to a new training horizon."""
        if new_max_steps <= 0:
            raise ValueError("new_max_steps must be positive.")
        scale_factor = new_max_steps / self.reference_training_steps
        self.refinement_start_iteration = max(
            1, round(self.refinement_start_iteration * scale_factor)
        )
        self.refinement_stop_iteration = max(
            1, round(self.refinement_stop_iteration * scale_factor)
        )
        self.opacity_reset_interval = max(
            1, round(self.opacity_reset_interval * scale_factor)
        )
        self.refinement_interval = max(
            1, round(self.refinement_interval * scale_factor)
        )
        self.refinement_pause_after_opacity_reset = max(
            0,
            round(self.refinement_pause_after_opacity_reset * scale_factor),
        )
        self.screen_space_refinement_stop_iteration = max(
            0,
            round(self.screen_space_refinement_stop_iteration * scale_factor),
        )
        self.reference_training_steps = new_max_steps


class RandomInitializationConfig(_ConfigModel):
    """Configuration for random point-based initialization."""

    init_num_points: int = 100_000
    init_extent: float = 3.0
    init_opacity: float = 0.1
    init_scale: float = 1.0


class PointCloudInitializationConfig(_ConfigModel):
    """Configuration for sparse point-cloud initialization."""

    init_opacity: float = 0.1
    init_scale: float = 1.0


class CheckpointInitializationConfig(_ConfigModel):
    """Configuration for checkpoint-based initialization."""

    checkpoint_path: Path


ValidInitializationConfig = (
    RandomInitializationConfig
    | PointCloudInitializationConfig
    | CheckpointInitializationConfig
)


class InitializationConfig(_ConfigModel):
    """Selected initialization method and its corresponding config."""

    method: Literal["random", "point_cloud", "checkpoint"] = "point_cloud"
    config: ValidInitializationConfig = Field(
        default_factory=PointCloudInitializationConfig
    )

    @model_validator(mode="after")
    def validate_method_and_config(self) -> "InitializationConfig":
        """Ensure the selected method is registered and matches the config."""
        if self.method not in INIT_FNS:
            raise ValueError(
                f"Initialization method {self.method!r} is not registered."
            )

        expected_config_types: dict[str, type[ValidInitializationConfig]] = {
            "random": RandomInitializationConfig,
            "point_cloud": PointCloudInitializationConfig,
            "checkpoint": CheckpointInitializationConfig,
        }
        expected_config_type = expected_config_types[self.method]
        if not isinstance(self.config, expected_config_type):
            raise ValueError(
                f"Initialization method {self.method!r} requires "
                f"{expected_config_type.__name__}, got "
                f"{type(self.config).__name__}."
            )
        return self

    def to_shared_initialization_config(self) -> SharedInitializationConfig:
        """Convert the example initialization wrapper to the shared init config."""
        return SharedInitializationConfig(
            strategy=self.method,
            **self.config.model_dump(),
        )


class TrainingConfig(_ConfigModel):
    """Training-loop configuration for the vanilla 3DGS example."""

    result_dir: Path = Path("results/vanilla_3dgs")
    seed: int = 42
    device: str = "cuda"
    max_steps: int = 30_000
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    log_every: int = 100
    eval_every: int = 1_000
    save_at_steps: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_training_config(self) -> "TrainingConfig":
        """Validate the training-loop configuration."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive.")
        if any(save_step <= 0 for save_step in self.save_at_steps):
            raise ValueError("save_at_steps must contain only positive steps.")
        self.save_at_steps = sorted(set(self.save_at_steps))
        return self


class Config(_ConfigModel):
    """Top-level configuration for the vanilla 3DGS example."""

    data: DataSourceConfig
    train_dataset: DatasetConfig = Field(
        default_factory=lambda: DatasetConfig(split="train")
    )
    val_dataset: DatasetConfig = Field(
        default_factory=lambda: DatasetConfig(split="val")
    )
    init: InitializationConfig = Field(default_factory=InitializationConfig)
    densification: DensificationConfig = Field(
        default_factory=DensificationConfig
    )
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="after")
    def scale_configs_to_training_horizon(self) -> "Config":
        """Scale step-based configs to the configured training horizon."""
        self.optimization.scale_to_max_steps(self.training.max_steps)
        self.densification.scale_to_max_steps(self.training.max_steps)
        return self
