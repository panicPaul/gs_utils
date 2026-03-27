"""Pydantic config models for gs_utils subsystems."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class DataSourceConfig(_ConfigModel):
    type: Literal["colmap"] = "colmap"
    path: Path
    factor: int = 1
    normalize: bool = False
    test_every: int = 8
    load_exposure: bool = False


class DatasetConfig(_ConfigModel):
    split: Literal["train", "val", "test", "all"] = "train"
    patch_size: int | None = None
    downsample_factor: int = 1
    preload: bool = False
    depth_dir: Path | None = None
    normals_dir: Path | None = None


class SceneConfig(_ConfigModel):
    geometry: Literal["3dgs", "2dgs"]
    sh_degree: int = 3
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    device: str = "cuda"


class InitializationConfig(_ConfigModel):
    strategy: Literal["sfm", "random", "point_cloud", "checkpoint"] = "sfm"
    checkpoint_path: Path | None = None
    init_num_points: int = 100_000
    init_extent: float = 3.0
    init_opacity: float = 0.1
    init_scale: float = 1.0

    @model_validator(mode="after")
    def validate_checkpoint(self) -> "InitializationConfig":
        if self.strategy == "checkpoint" and not self.checkpoint_path:
            raise ValueError(
                "`checkpoint_path` is required when strategy='checkpoint'."
            )
        return self


class PreprocessingConfig(_ConfigModel):
    enabled: bool = True
    stages: list[str] = Field(default_factory=list)


class PostprocessingConfig(_ConfigModel):
    enabled: bool = True
    stages: list[str] = Field(default_factory=list)


class ViewerConfig(_ConfigModel):
    enabled: bool = True
    port: int = 8080
    default_render_mode: str = "rgb"


class ExampleRuntimeConfig(_ConfigModel):
    batch_size: int = 1
    max_steps: int = 30_000
    eval_every: int = 7_000
    save_every: int = 7_000
