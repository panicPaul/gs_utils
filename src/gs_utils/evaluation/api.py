"""Offline evaluation entrypoints for gs_utils runs."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
import yaml
from einops import rearrange
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from gs_utils.data.datasets import build_dataloader, get_dataset
from gs_utils.examples.vanilla_3dgs.config import Config as VanillaConfig
from gs_utils.examples.vanilla_3dgs.scene import Vanilla3DGS

from .config import EvaluationConfig


@dataclass
class EvaluateCommand:
    """Evaluate a gs_utils run directory or grouped experiment directory."""

    path: Path
    checkpoint_step: int | None = None
    write_test_images: bool = False
    overwrite: bool = False

    def __call__(self) -> None:
        """Run offline evaluation and write JSON reports."""
        evaluate_path(
            EvaluationConfig(
                path=self.path,
                checkpoint_step=self.checkpoint_step,
                write_test_images=self.write_test_images,
                overwrite=self.overwrite,
            )
        )


def evaluate_path(config: EvaluationConfig) -> dict[str, Any]:
    """Evaluate a single run, dataset directory, or experiment directory."""
    evaluation_path = config.path
    if _is_run_directory(evaluation_path):
        return evaluate_run_directory(evaluation_path, config)
    if _is_dataset_directory(evaluation_path):
        return evaluate_dataset_directory(evaluation_path, config)
    return evaluate_experiment_directory(evaluation_path, config)


def evaluate_run_directory(
    run_directory: Path,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Evaluate one gs_utils run directory and write its JSON report."""
    run_config = _load_run_config(run_directory)
    checkpoint_path, checkpoint_step = _resolve_checkpoint_path(
        run_directory / "checkpoints",
        requested_step=config.checkpoint_step,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    scene = Vanilla3DGS()
    scene.load_state_dict(checkpoint["scene"])
    device = torch.device(run_config.training.device)
    scene.to(device)
    scene.eval()

    train_dataset, _ = get_dataset(
        source_config=run_config.data,
        dataset_config=run_config.train_dataset.model_copy(
            update={"patch_size": None}
        ),
    )
    validation_dataset, _ = get_dataset(
        source_config=run_config.data,
        dataset_config=run_config.val_dataset.model_copy(
            update={"patch_size": None}
        ),
    )

    train_report = _evaluate_dataset_split(
        scene=scene,
        dataset=train_dataset,
        split_name=run_config.train_dataset.split,
        device=device,
        write_images=False,
        images_output_directory=None,
        num_workers=run_config.training.num_workers,
        pin_memory=run_config.training.pin_memory,
        persistent_workers=run_config.training.persistent_workers,
    )
    validation_images_directory = (
        run_directory
        / "evaluation"
        / f"step_{checkpoint_step:06d}_images"
        / run_config.val_dataset.split
    )
    validation_report = _evaluate_dataset_split(
        scene=scene,
        dataset=validation_dataset,
        split_name=run_config.val_dataset.split,
        device=device,
        write_images=config.write_test_images,
        images_output_directory=validation_images_directory,
        num_workers=max(1, run_config.training.num_workers // 2),
        pin_memory=run_config.training.pin_memory,
        persistent_workers=run_config.training.persistent_workers,
    )

    run_report = {
        "type": "run_evaluation",
        "run_dir": str(run_directory),
        "checkpoint": {
            "path": str(checkpoint_path),
            "step": checkpoint_step,
        },
        "splits": {
            train_report["split"]: train_report,
            validation_report["split"]: validation_report,
        },
    }
    output_directory = run_directory / "evaluation"
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / f"step_{checkpoint_step:06d}.json"
    _write_json(report_path, run_report, overwrite=config.overwrite)
    return run_report


def evaluate_dataset_directory(
    dataset_directory: Path,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Evaluate all scene runs inside one dataset directory."""
    scene_run_directories = sorted(
        path for path in dataset_directory.iterdir() if _is_run_directory(path)
    )
    if not scene_run_directories:
        raise ValueError(
            f"Dataset directory {dataset_directory} does not contain run directories."
        )

    scene_reports = [
        evaluate_run_directory(scene_run_directory, config)
        for scene_run_directory in scene_run_directories
    ]
    dataset_report = {
        "type": "dataset_evaluation",
        "dataset_name": dataset_directory.name,
        "dataset_dir": str(dataset_directory),
        "checkpoint_selection": _checkpoint_selection_label(config),
        "scene_reports": [
            {
                "scene_name": Path(scene_report["run_dir"]).name,
                "run_dir": scene_report["run_dir"],
                "checkpoint": scene_report["checkpoint"],
                "splits": {
                    split_name: {"aggregate": split_report["aggregate"]}
                    for split_name, split_report in scene_report[
                        "splits"
                    ].items()
                },
            }
            for scene_report in scene_reports
        ],
        "aggregates": _aggregate_run_reports(scene_reports),
    }
    output_directory = dataset_directory / "evaluation"
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = (
        output_directory / f"summary_{_checkpoint_selection_label(config)}.json"
    )
    _write_json(report_path, dataset_report, overwrite=config.overwrite)
    return dataset_report


def evaluate_experiment_directory(
    experiment_directory: Path,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Evaluate all dataset directories inside one experiment directory."""
    dataset_directories = sorted(
        path
        for path in experiment_directory.iterdir()
        if path.is_dir() and _is_dataset_directory(path)
    )
    if not dataset_directories:
        raise ValueError(
            f"Experiment directory {experiment_directory} does not contain dataset directories."
        )

    dataset_reports = [
        evaluate_dataset_directory(dataset_directory, config)
        for dataset_directory in dataset_directories
    ]
    experiment_report = {
        "type": "experiment_evaluation",
        "experiment_dir": str(experiment_directory),
        "checkpoint_selection": _checkpoint_selection_label(config),
        "dataset_reports": [
            {
                "dataset_name": dataset_report["dataset_name"],
                "dataset_dir": dataset_report["dataset_dir"],
                "aggregates": dataset_report["aggregates"],
            }
            for dataset_report in dataset_reports
        ],
        "aggregates": _aggregate_dataset_reports(dataset_reports),
    }
    output_directory = experiment_directory / "evaluation"
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = (
        output_directory / f"summary_{_checkpoint_selection_label(config)}.json"
    )
    _write_json(report_path, experiment_report, overwrite=config.overwrite)
    return experiment_report


def _evaluate_dataset_split(
    scene: Vanilla3DGS,
    dataset: Any,
    split_name: str,
    device: torch.device,
    write_images: bool,
    images_output_directory: Path | None,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> dict[str, Any]:
    """Evaluate one dataset split and return per-image plus aggregate metrics."""
    dataloader = build_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    lpips_module = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=True,
    ).to(device)
    ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_module = PeakSignalNoiseRatio(data_range=1.0).to(device)

    per_image_reports: list[dict[str, Any]] = []
    for image_index, sample in enumerate(dataloader):
        sample = sample.to(device)

        # Render the scene and evaluate the current image independently.
        with torch.no_grad():
            render_output = scene.render(sample.render_input)
        rendered_image = torch.clamp(render_output.image, 0.0, 1.0)
        target_image = torch.clamp(sample.image, 0.0, 1.0)
        rendered_image_nchw = rearrange(
            rendered_image,
            "batch height width channels -> batch channels height width",
        )
        target_image_nchw = rearrange(
            target_image,
            "batch height width channels -> batch channels height width",
        )

        lpips_module.reset()
        ssim_module.reset()
        psnr_module.reset()
        image_metrics = {
            "psnr": float(
                psnr_module(rendered_image_nchw, target_image_nchw).item()
            ),
            "ssim": float(
                ssim_module(rendered_image_nchw, target_image_nchw).item()
            ),
            "lpips": float(
                lpips_module(rendered_image_nchw, target_image_nchw).item()
            ),
        }
        image_report = {
            "index": image_index,
            "image_path": _batched_image_path(sample.metadata),
            "camera_id": _batched_camera_id(sample.metadata),
            "metadata": _make_json_safe(sample.metadata),
            "metrics": image_metrics,
        }
        per_image_reports.append(image_report)

        # Optionally write rendered test images for later visual inspection.
        if write_images and images_output_directory is not None:
            images_output_directory.mkdir(parents=True, exist_ok=True)
            image_file_name = _rendered_image_name(
                image_index=image_index,
                image_path=_batched_image_path_as_path(sample.metadata),
            )
            imageio.imwrite(
                images_output_directory / image_file_name,
                _to_uint8_image(rendered_image[0]),
            )

    aggregate_metrics = _aggregate_image_reports(per_image_reports)
    return {
        "split": split_name,
        "aggregate": aggregate_metrics,
        "images": per_image_reports,
    }


def _load_run_config(run_directory: Path) -> VanillaConfig:
    """Load the saved vanilla run config from a run directory."""
    config_path = run_directory / "config.yaml"
    with config_path.open("r") as config_file:
        loaded_config = yaml.safe_load(config_file)
    return VanillaConfig.model_validate(loaded_config)


def _resolve_checkpoint_path(
    checkpoints_directory: Path,
    requested_step: int | None,
) -> tuple[Path, int]:
    """Resolve the checkpoint path for the requested or latest step."""
    checkpoint_paths = sorted(
        checkpoints_directory.glob("ckpt_*.pt"),
        key=_checkpoint_step_from_path,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoints_directory}."
        )

    if requested_step is None:
        checkpoint_path = checkpoint_paths[-1]
    else:
        matching_checkpoint_paths = [
            checkpoint_path
            for checkpoint_path in checkpoint_paths
            if _checkpoint_step_from_path(checkpoint_path) == requested_step
        ]
        if not matching_checkpoint_paths:
            raise FileNotFoundError(
                f"No checkpoint found for step {requested_step} in "
                f"{checkpoints_directory}."
            )
        checkpoint_path = matching_checkpoint_paths[0]

    return checkpoint_path, _checkpoint_step_from_path(checkpoint_path)


def _checkpoint_step_from_path(checkpoint_path: Path) -> int:
    """Extract the training step from a checkpoint file name."""
    return int(checkpoint_path.stem.removeprefix("ckpt_"))


def _checkpoint_selection_label(config: EvaluationConfig) -> str:
    """Return a stable label for the requested checkpoint selection."""
    return (
        "latest"
        if config.checkpoint_step is None
        else f"step_{config.checkpoint_step:06d}"
    )


def _aggregate_image_reports(
    image_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-image metrics into one split summary."""
    metric_names = ["psnr", "ssim", "lpips"]
    num_images = len(image_reports)
    aggregate_metrics = {
        metric_name: (
            sum(
                float(image_report["metrics"][metric_name])
                for image_report in image_reports
            )
            / num_images
            if num_images > 0
            else 0.0
        )
        for metric_name in metric_names
    }
    aggregate_metrics["num_images"] = num_images
    return aggregate_metrics


def _aggregate_run_reports(run_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate run-level split summaries across scenes."""
    split_names = {
        split_name
        for run_report in run_reports
        for split_name in run_report["splits"]
    }
    return {
        split_name: _aggregate_split_summaries(
            [
                run_report["splits"][split_name]["aggregate"]
                for run_report in run_reports
                if split_name in run_report["splits"]
            ]
        )
        for split_name in split_names
    }


def _aggregate_dataset_reports(
    dataset_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate dataset-level split summaries across datasets."""
    split_names = {
        split_name
        for dataset_report in dataset_reports
        for split_name in dataset_report["aggregates"]
    }
    return {
        split_name: _aggregate_split_summaries(
            [
                dataset_report["aggregates"][split_name]
                for dataset_report in dataset_reports
                if split_name in dataset_report["aggregates"]
            ]
        )
        for split_name in split_names
    }


def _aggregate_split_summaries(
    split_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate split summaries with image-count weighting."""
    metric_names = ["psnr", "ssim", "lpips"]
    total_images = sum(
        int(split_summary["num_images"]) for split_summary in split_summaries
    )
    if total_images == 0:
        return {metric_name: 0.0 for metric_name in metric_names} | {
            "num_images": 0
        }

    return {
        metric_name: sum(
            float(split_summary[metric_name]) * int(split_summary["num_images"])
            for split_summary in split_summaries
        )
        / total_images
        for metric_name in metric_names
    } | {"num_images": total_images}


def _is_run_directory(path: Path) -> bool:
    """Return whether the directory matches the expected run layout."""
    return (
        path.is_dir()
        and (path / "config.yaml").is_file()
        and (path / "checkpoints").is_dir()
    )


def _is_dataset_directory(path: Path) -> bool:
    """Return whether the directory directly contains scene run directories."""
    if not path.is_dir():
        return False
    child_directories = [
        child
        for child in path.iterdir()
        if child.is_dir() and child.name != "evaluation"
    ]
    return bool(child_directories) and all(
        _is_run_directory(child_directory)
        for child_directory in child_directories
    )


def _batched_image_path(metadata: dict[str, Any]) -> str | None:
    """Extract the original image path from batch metadata."""
    image_paths = metadata.get("image_paths")
    if not isinstance(image_paths, list) or not image_paths:
        return None
    image_path = image_paths[0]
    return None if image_path is None else str(image_path)


def _batched_image_path_as_path(metadata: dict[str, Any]) -> Path | None:
    """Extract the original image path as a Path object from batch metadata."""
    image_path = _batched_image_path(metadata)
    return None if image_path is None else Path(image_path)


def _batched_camera_id(metadata: dict[str, Any]) -> int | None:
    """Extract the original camera id from batch metadata."""
    camera_ids = metadata.get("camera_ids")
    if not isinstance(camera_ids, list) or not camera_ids:
        return None
    camera_id = camera_ids[0]
    return camera_id if isinstance(camera_id, int) else None


def _rendered_image_name(image_index: int, image_path: Path | None) -> str:
    """Return a stable output file name for a rendered evaluation image."""
    if image_path is None:
        return f"{image_index:06d}.png"
    return f"{image_index:06d}_{image_path.stem}.png"


def _to_uint8_image(image: torch.Tensor) -> Any:
    """Convert a float image in [0, 1] to uint8 image data."""
    return (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(
        "uint8"
    )


def _write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    """Write a JSON file, optionally refusing to overwrite an existing file."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Evaluation output {path} already exists. Pass overwrite=True to reuse it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as report_file:
        json.dump(payload, report_file, indent=2, sort_keys=False)


def _make_json_safe(value: Any) -> Any:
    """Convert a nested value into a JSON-serializable structure."""
    if isinstance(value, dict):
        return {
            str(key): _make_json_safe(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list | tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    return str(value)
