"""Checkpoint and run-directory helper functions."""

from pathlib import Path

import torch
import yaml

from gs_utils.contracts.scene import Scene


def prepare_run_directory(result_dir: Path, overwrite: bool) -> None:
    """Create the run directory or fail if it already exists."""
    if result_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Run directory {result_dir} already exists. "
            "Pass overwrite=True to reuse it."
        )
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (result_dir / "tensorboard").mkdir(parents=True, exist_ok=True)


def save_config_yaml(config: object, path: Path) -> None:
    """Serialize a pydantic config-like object to YAML."""
    if not hasattr(config, "model_dump"):
        raise TypeError("Config object must provide model_dump().")
    with path.open("w") as config_file:
        yaml.safe_dump(
            config.model_dump(mode="json"),
            config_file,
            sort_keys=False,
        )


def save_scene_checkpoint(
    scene: Scene,
    iteration: int,
    checkpoint_path: Path,
) -> None:
    """Persist the current scene state and training step."""
    torch.save(
        {
            "step": iteration,
            "scene": scene.state_dict(),
        },
        checkpoint_path,
    )
