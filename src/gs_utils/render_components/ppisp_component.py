"""PPISP-based post-processing component."""

from pathlib import Path

import torch
from jaxtyping import Float
from ppisp import PPISP, PPISPConfig

from gs_utils.render_components.config import PPISPComponentConfig


class PPISPComponent(torch.nn.Module):
    """Wrap the external PPISP package behind a local nn.Module surface."""

    def __init__(
        self,
        num_cameras: int,
        num_frames: int,
        config: PPISPComponentConfig | None = None,
    ) -> None:
        """Initialize the wrapped PPISP module."""
        super().__init__()
        self.config = config or PPISPComponentConfig()

        ppisp_config = PPISPConfig(
            use_controller=self.config.use_controller,
            controller_distillation=self.config.controller_distillation,
            controller_activation_ratio=self.config.controller_activation_ratio,
        )
        self.ppisp_module = PPISP(
            num_cameras=num_cameras,
            num_frames=num_frames,
            config=ppisp_config,
        )

    def forward(
        self,
        image: Float[torch.Tensor, "batch height width 3"],
        pixel_coords: Float[torch.Tensor, "height width 2"],
        resolution: tuple[int, int],
        camera_idx: int | None = None,
        frame_idx: int | None = None,
        exposure_prior: Float[torch.Tensor, "batch"] | None = None,
    ) -> Float[torch.Tensor, "batch height width 3"]:
        """Apply PPISP post-processing to rendered RGB images."""
        return self.ppisp_module(
            rgb=image,
            pixel_coords=pixel_coords,
            resolution=resolution,
            camera_idx=camera_idx,
            frame_idx=frame_idx,
            exposure_prior=exposure_prior,
        )

    def create_optimizers(self) -> list[torch.optim.Optimizer]:
        """Create PPISP-owned optimizers."""
        return self.ppisp_module.create_optimizers()

    def create_schedulers(
        self,
        optimizers: list[torch.optim.Optimizer],
        max_optimization_iters: int,
    ) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """Create PPISP-owned learning-rate schedulers."""
        return self.ppisp_module.create_schedulers(
            optimizers,
            max_optimization_iters=max_optimization_iters,
        )

    def get_regularization_loss(self) -> torch.Tensor:
        """Return the PPISP regularization loss."""
        return self.ppisp_module.get_regularization_loss()

    @torch.no_grad()
    def export_report(
        self,
        frames_per_camera: list[int],
        output_dir: Path,
        camera_names: list[str] | None = None,
    ) -> list[Path]:
        """Export PPISP visualization reports and parameter summaries."""
        from ppisp.report import export_ppisp_report

        return export_ppisp_report(
            self.ppisp_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
