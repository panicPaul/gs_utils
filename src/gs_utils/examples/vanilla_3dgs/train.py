"""Training script for the vanilla 3DGS scene."""

from dataclasses import dataclass, field

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

from gs_utils.data.config import DatasetConfig, DataSourceConfig
from gs_utils.data.datasets import (
    build_dataloader,
    get_dataset,
)
from gs_utils.utils.checkpoints import (
    prepare_run_directory,
    save_config_yaml,
    save_scene_checkpoint,
)
from gs_utils.utils.random import set_random_seed

from .config import (
    Config,
    DensificationConfig,
    InitializationConfig,
    OptimizationConfig,
    TrainingConfig,
)
from .scene import Vanilla3DGS


def train(config: Config) -> None:
    """Run the first-pass vanilla 3DGS training loop."""
    # Set up deterministic run state and persist the resolved config.
    training_config = config.training
    set_random_seed(training_config.seed)
    save_config_yaml(config, training_config.result_dir / "config.yaml")

    # Construct the scene on the requested device.
    device = torch.device(training_config.device)
    scene = Vanilla3DGS()
    scene.to(device)

    # Build the training and validation datasets and their loaders.
    train_dataset, point_cloud = get_dataset(
        source_config=config.data,
        dataset_config=config.train_dataset,
    )
    validation_dataset, _ = get_dataset(
        source_config=config.data,
        dataset_config=config.val_dataset,
    )
    scene_scale = train_dataset.parsed_scene.scene_scale

    train_loader = build_dataloader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers,
    )
    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, training_config.num_workers // 2),
        pin_memory=training_config.pin_memory,
        persistent_workers=training_config.persistent_workers,
    )

    # Initialize scene state, then create the optimizer and scheduler objects.
    scene.initialize(
        config.init.to_shared_initialization_config(),
        point_cloud=point_cloud,
        scene_scale=scene_scale,
    )
    scene.initialize_densification(
        config.densification,
        scene_scale=scene_scale,
    )
    optimizers = scene.initialize_optimizers(config.optimization)
    schedulers = scene.initialize_schedulers(
        optimizers,
        config.optimization,
    )

    # Create logging and evaluation modules once for the full training run.
    writer = SummaryWriter(log_dir=training_config.result_dir / "tensorboard")
    lpips_module = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=True,
    ).to(device)
    ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_module = PeakSignalNoiseRatio(data_range=1.0).to(device)

    # Run the training loop with periodic evaluation and checkpointing.
    save_steps = set(training_config.save_at_steps)
    train_loader_iterator = iter(train_loader)
    progress_bar = tqdm(range(1, training_config.max_steps + 1), desc="train")
    for iteration in progress_bar:
        try:
            training_sample = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_loader)
            training_sample = next(train_loader_iterator)

        training_sample = training_sample.to(device)
        scene.train_step(
            iteration=iteration,
            render_input=training_sample.render_input,
            gt_image=training_sample.image,
            optimizers=optimizers,
            schedulers=schedulers,
            logger=writer
            if iteration % training_config.log_every == 0
            else None,
        )

        if (
            iteration % training_config.eval_every == 0
            or iteration == training_config.max_steps
        ):
            validation_metrics = scene.eval_loop(
                iteration=iteration,
                validation_loader=validation_loader,
                lpips_module=lpips_module,
                ssim_module=ssim_module,
                psnr_module=psnr_module,
                logger=writer,
            )
            progress_bar.set_postfix(
                {
                    metric_name: f"{metric_value:.4f}"
                    for metric_name, metric_value in validation_metrics.items()
                }
            )

        if iteration in save_steps or iteration == training_config.max_steps:
            save_scene_checkpoint(
                scene,
                iteration,
                training_config.result_dir
                / "checkpoints"
                / f"ckpt_{iteration:06d}.pt",
            )

    writer.close()


@dataclass
class TrainCommand:
    """Train the vanilla 3DGS example."""

    data: DataSourceConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    train_dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(split="train")
    )
    val_dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(split="val")
    )
    init: InitializationConfig = field(default_factory=InitializationConfig)
    densification: DensificationConfig = field(
        default_factory=DensificationConfig
    )
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    overwrite: bool = False
    scale_steps_to: int | None = None

    def __call__(self) -> None:
        """Run the vanilla 3DGS training loop."""
        resolved_training = (
            self.training
            if self.scale_steps_to is None
            else self.training.model_copy(
                update={"max_steps": self.scale_steps_to}
            )
        )

        resolved_config = Config(
            data=self.data,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            init=self.init,
            densification=self.densification,
            optimization=self.optimization,
            training=resolved_training,
        )
        prepare_run_directory(
            resolved_config.training.result_dir,
            self.overwrite,
        )
        train(resolved_config)


if __name__ == "__main__":
    tyro.cli(TrainCommand)
