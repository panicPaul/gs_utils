"""Pydantic config models for scene initialization."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class _ConfigModel(BaseModel):
    """Base config model with strict field validation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class InitializationConfig(_ConfigModel):
    """Configuration for geometry-aware scene initialization."""

    strategy: Literal["sfm", "random", "point_cloud", "checkpoint"] = "sfm"
    checkpoint_path: Path | None = None
    init_num_points: int = 100_000
    init_extent: float = 3.0
    init_opacity: float = 0.1
    init_scale: float = 1.0

    @model_validator(mode="after")
    def validate_checkpoint(self) -> "InitializationConfig":
        """Require a checkpoint path when using checkpoint initialization."""
        if self.strategy == "checkpoint" and not self.checkpoint_path:
            raise ValueError(
                "`checkpoint_path` is required when strategy='checkpoint'."
            )
        return self
