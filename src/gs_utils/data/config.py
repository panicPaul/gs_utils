"""Pydantic config models for data loading and dataset construction."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _ConfigModel(BaseModel):
    """Base config model with strict field validation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class DataSourceConfig(_ConfigModel):
    """Configuration for loading a parsed scene from a concrete data source."""

    type: Literal["colmap"] = "colmap"
    path: Path
    factor: int = 1
    normalize: bool = False
    test_every: int = 8
    load_exposure: bool = False


class DatasetConfig(_ConfigModel):
    """Configuration for building supervised datasets from parsed scene data."""

    split: Literal["train", "val", "test", "all"] = "train"
    patch_size: int | None = None
    downsample_factor: int = 1
    preload: bool = False
    depth_dir: Path | None = None
    normals_dir: Path | None = None
