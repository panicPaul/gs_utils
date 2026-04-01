"""Config models for offline evaluation."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class _ConfigModel(BaseModel):
    """Base config model with strict field validation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class EvaluationConfig(_ConfigModel):
    """Configuration for offline run-directory evaluation."""

    path: Path
    checkpoint_step: int | None = None
    write_test_images: bool = False
    overwrite: bool = False
