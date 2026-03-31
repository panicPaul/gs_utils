"""Training script for the vanilla 3DGS scene."""

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

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

from .config import TrainingConfig
from .scene import Vanilla3DGS


def train(config: TrainingConfig) -> None:
    """Run the first-pass vanilla 3DGS training loop."""
    set_random_seed(config.seed)
    prepare_run_directory(config.result_dir, config.overwrite)
    save_config_yaml(config, config.result_dir / "config.yaml")

    device = torch.device(config.device)
    scene = Vanilla3DGS()
    scene.to(device)

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
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )
    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )

    scene.initialize(
        config.init,
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

    writer = SummaryWriter(log_dir=config.result_dir / "tensorboard")
    lpips_module = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=True,
    ).to(device)
    ssim_module = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_module = PeakSignalNoiseRatio(data_range=1.0).to(device)

    train_loader_iterator = iter(train_loader)
    progress_bar = tqdm(range(1, config.max_steps + 1), desc="train")
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
            logger=writer if iteration % config.log_every == 0 else None,
        )

        if iteration % config.eval_every == 0 or iteration == config.max_steps:
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

        if iteration % config.save_every == 0 or iteration == config.max_steps:
            save_scene_checkpoint(
                scene,
                iteration,
                config.result_dir / "checkpoints" / f"ckpt_{iteration:06d}.pt",
            )

    writer.close()


def main() -> None:
    """Parse CLI args and run the vanilla 3DGS trainer."""
    config = tyro.cli(TrainingConfig)
    train(config)


if __name__ == "__main__":
    main()
