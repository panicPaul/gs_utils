"""Vanilla style 3DGS."""

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import torch
from einops import rearrange
from gsplat import rasterization
from jaxtyping import Float
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler
from torch.profiler import record_function
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from gs_utils import RendersDepth
from gs_utils.contracts import (
    RenderInput,
    RenderMode,
    RenderOutput,
    RendersAlpha,
    RendersRGB,
    Scene,
    SphericalHarmonics3DGS,
)
from gs_utils.data.contract import DataSample
from gs_utils.utils.erank import compute_effective_rank
from gs_utils.utils.losses import photometric_loss
from gs_utils.utils.optimizers import (
    build_append_selected_parameter_factory,
    build_append_zero_optimizer_state_factory,
    build_keep_selected_optimizer_state_factory,
    build_keep_selected_parameter_factory,
    build_zero_optimizer_state_factory,
    remap_parameters_and_optimizer_state,
)
from gs_utils.utils.rotations import (
    normalized_quaternion_to_rotation_matrix,
)

from .config import (
    DensificationConfig,
    OptimizationConfig,
)


@dataclass(slots=True)
class Vanilla3DGSDensificationState:
    """Runtime densification state for the vanilla 3DGS scene."""

    scene_scale: float
    config: DensificationConfig
    accumulated_image_plane_gradient_norms: torch.Tensor | None = None
    visibility_counts: torch.Tensor | None = None
    max_screen_space_radii: torch.Tensor | None = None
    latest_render_metadata: "Vanilla3DGSRenderMetadata | None" = None


@dataclass(slots=True)
class Vanilla3DGSRenderMetadata:
    """Typed render metadata needed by vanilla 3DGS densification."""

    image_width: int
    image_height: int
    num_cameras: int
    image_plane_means: torch.Tensor
    projected_radii: torch.Tensor
    gaussian_ids: torch.Tensor


class Vanilla3DGS(
    Scene[
        OptimizationConfig,
        Optimizer,
        LRScheduler,
        DensificationConfig,
    ],
    SphericalHarmonics3DGS,
    RendersRGB,
    RendersAlpha,
    RendersDepth,
):
    """Vanilla style 3DGS."""

    def __init__(self):
        """Initialize an empty vanilla 3DGS scene."""
        super().__init__()
        self.means = nn.Parameter(torch.empty((0, 3)))
        self.unnormalized_quats = nn.Parameter(torch.empty((0, 4)))
        self.sh_0 = nn.Parameter(torch.empty((0, 1, 3)))
        self.sh_N = nn.Parameter(torch.empty((0, 15, 3)))
        self.log_scales = nn.Parameter(torch.empty((0, 3)))
        self.logit_opacities = nn.Parameter(torch.empty(0))
        self.densification_state: Vanilla3DGSDensificationState | None = None

    @property
    def colors(self) -> Float[torch.Tensor, "num_splats 3"]:
        """Return RGB colors derived from the zeroth SH band."""
        return self.sh_0.squeeze(1)

    # <------------> Initialization <------------>

    def initialize_densification(
        self,
        config: DensificationConfig,
        scene_scale: float = 1.0,
    ) -> None:
        """Initialize the runtime densification state for the scene."""
        if not config.enabled:
            self.densification_state = None
            return

        self.densification_state = Vanilla3DGSDensificationState(
            scene_scale=scene_scale,
            config=config,
        )

    def initialize_optimizers(
        self,
        config: OptimizationConfig,
    ) -> dict[str, Optimizer]:
        """Initialize all optimizers for the vanilla 3DGS parameters."""
        optimizers: dict[str, Optimizer] = {
            "means": Adam([self.means], lr=config.means_lr),
            "log_scales": Adam(
                [self.log_scales],
                lr=config.log_scales_lr,
            ),
            "logit_opacities": Adam(
                [self.logit_opacities],
                lr=config.logit_opacities_lr,
            ),
            "unnormalized_quats": Adam(
                [self.unnormalized_quats],
                lr=config.unnormalized_quats_lr,
            ),
            "sh_0": Adam([self.sh_0], lr=config.sh_0_lr),
            "sh_N": Adam([self.sh_N], lr=config.sh_N_lr),
        }
        return optimizers

    def initialize_schedulers(
        self,
        optimizers: dict[str, Optimizer],
        config: OptimizationConfig,
    ) -> dict[str, LRScheduler]:
        """Initialize schedulers for the provided optimizers and config."""
        return {
            "means": ExponentialLR(
                optimizers["means"],
                gamma=config.means_scheduler.gamma(),
            )
        }

    # <------------> Rendering <------------>

    def render(
        self,
        render_input: RenderInput,
        render_mode: RenderMode = RenderMode.RGB,
        sh_degree: int | None = None,
    ) -> RenderOutput:
        """Render the scene for the provided camera state."""
        # Activate internal attributes
        spherical_harmonics = torch.cat([self.sh_0, self.sh_N], dim=1)
        scales = torch.exp(self.log_scales)
        opacities = torch.sigmoid(self.logit_opacities)

        if sh_degree is None:
            num_sh_coeffs = self.sh_0.shape[1] + 1
            sh_degree = int(math.sqrt(num_sh_coeffs) - 1)

        # TODO: should we check for supported render modes?
        match render_mode:
            case RenderMode.RGB | RenderMode.ALPHA:
                gsplat_render_mode = "RGB"
            case _:
                gsplat_render_mode = "RGB+ED"

        densification_state = self.densification_state
        use_absolute_image_plane_gradients = (
            densification_state is not None
            and densification_state.config.use_absolute_image_plane_gradients
        )

        rendered_image, rendered_alphas, raw_render_metadata = rasterization(
            means=self.means,
            quats=self.unnormalized_quats,  # normalization is fused into the backend
            scales=scales,
            opacities=opacities,
            colors=spherical_harmonics,
            viewmats=render_input.cam_to_world,
            Ks=render_input.get_intrinsics(),
            height=render_input.height,
            width=render_input.width,
            backgrounds=render_input.background,
            sh_degree=sh_degree,
            render_mode=gsplat_render_mode,
            absgrad=use_absolute_image_plane_gradients,
        )
        render_metadata = Vanilla3DGSRenderMetadata(
            image_width=raw_render_metadata["width"],
            image_height=raw_render_metadata["height"],
            num_cameras=raw_render_metadata["n_cameras"],
            image_plane_means=raw_render_metadata["means2d"],
            projected_radii=raw_render_metadata["radii"],
            gaussian_ids=raw_render_metadata["gaussian_ids"],
        )

        rendered_rgb = rendered_image[..., :3]
        rendered_depth = (
            rendered_image[..., 3:] if gsplat_render_mode == "RGB+ED" else None
        )

        if render_input.background is not None:
            rendered_rgb = rendered_rgb * (
                rendered_alphas
            ) + render_input.background * (1 - rendered_alphas)

        if densification_state is not None:
            densification_state.latest_render_metadata = render_metadata

        return RenderOutput(
            image=rendered_rgb,
            depth=rendered_depth,
            alpha=rendered_alphas,
            aux=render_metadata,
        )

    # <------------> Training <------------>
    @record_function("train_step")
    def train_step(
        self,
        iteration: int,
        render_input: RenderInput,
        gt_image: Float[torch.Tensor, "batch height width 3"],
        optimizers: dict[str, Optimizer],
        schedulers: dict[str, LRScheduler],
        logger: SummaryWriter | None = None,
    ) -> None:
        """Perform a training step for the scene."""
        self.optimizer_zero_grad(optimizers)

        with record_function("forward"):
            render_output = self.render(render_input)

        with record_function("densification_step_pre_backward"):
            self.densification_step_pre_backward(iteration, optimizers)

        with record_function("loss_computation"):
            loss = photometric_loss(render_output.image, gt_image)

        with record_function("backward"):
            loss.backward()

        with record_function("optimizer_step"):
            self.optimizer_step(optimizers, schedulers)
        self.optimizer_zero_grad(optimizers)

        with record_function("densification_step_post_backward"):
            self.densification_step_post_backward(iteration, optimizers)

        with record_function("logging"):
            if logger is not None:
                logger.add_scalar("train/loss", loss.item(), iteration)
                effective_rank = compute_effective_rank(self)
                logger.add_scalar("train/erank", effective_rank, iteration)

    # <------------> Densification <------------>
    def densification_step_pre_backward(
        self,
        iteration: int,
        optimizers: dict[str, Optimizer],
    ) -> None:
        """Run the pre-backward densification hook."""
        del iteration, optimizers
        densification_state = self.densification_state
        if (
            densification_state is None
            or densification_state.latest_render_metadata is None
        ):
            return

        densification_state.latest_render_metadata.image_plane_means.retain_grad()

    def densification_step_post_backward(
        self,
        iteration: int,
        optimizers: dict[str, Optimizer],
    ) -> None:
        """Run the post-backward densification hook."""
        densification_state = self.densification_state
        if (
            densification_state is None
            or densification_state.latest_render_metadata is None
        ):
            return

        densification_config = densification_state.config
        if iteration >= densification_config.refinement_stop_iteration:
            densification_state.latest_render_metadata = None
            return

        self._update_densification_statistics(densification_state)

        if (
            iteration > densification_config.refinement_start_iteration
            and iteration % densification_config.refinement_interval == 0
            and iteration % densification_config.opacity_reset_interval
            >= densification_config.refinement_pause_after_opacity_reset
        ):
            self._duplicate_and_split_gaussians(
                densification_state=densification_state,
                optimizers=optimizers,
                iteration=iteration,
            )
            self._prune_gaussians(
                densification_state=densification_state,
                optimizers=optimizers,
                iteration=iteration,
            )
            self._reset_densification_statistics(densification_state)
            if self.means.is_cuda:
                torch.cuda.empty_cache()

        if (
            iteration > 0
            and iteration % densification_config.opacity_reset_interval == 0
        ):
            self._reset_opacities(
                optimizers=optimizers,
                post_sigmoid_opacity_value=(
                    densification_config.prune_opacity_threshold * 2.0
                ),
            )

        densification_state.latest_render_metadata = None

    @torch.no_grad()
    def _update_densification_statistics(
        self,
        densification_state: Vanilla3DGSDensificationState,
    ) -> None:
        """Update running densification statistics from the latest render."""
        densification_config = densification_state.config

        render_metadata = densification_state.latest_render_metadata
        if render_metadata is None:
            raise RuntimeError("Missing render metadata for densification.")

        if densification_config.use_absolute_image_plane_gradients:
            image_plane_gradients = (
                render_metadata.image_plane_means.absgrad.clone()
            )
        else:
            image_plane_gradients = (
                render_metadata.image_plane_means.grad.clone()
            )
        image_plane_gradients[..., 0] *= (
            render_metadata.image_width / 2.0 * render_metadata.num_cameras
        )
        image_plane_gradients[..., 1] *= (
            render_metadata.image_height / 2.0 * render_metadata.num_cameras
        )

        number_of_gaussians = self.means.shape[0]
        if densification_state.accumulated_image_plane_gradient_norms is None:
            densification_state.accumulated_image_plane_gradient_norms = (
                torch.zeros(
                    number_of_gaussians,
                    device=image_plane_gradients.device,
                )
            )
        if densification_state.visibility_counts is None:
            densification_state.visibility_counts = torch.zeros(
                number_of_gaussians,
                device=image_plane_gradients.device,
            )
        if (
            densification_config.screen_space_refinement_stop_iteration > 0
            and densification_state.max_screen_space_radii is None
        ):
            densification_state.max_screen_space_radii = torch.zeros(
                number_of_gaussians,
                device=image_plane_gradients.device,
            )

        visible_in_all_cameras = (render_metadata.projected_radii > 0.0).all(
            dim=-1
        )
        visible_gaussian_indices = torch.where(visible_in_all_cameras)[1]
        visible_image_plane_gradients = image_plane_gradients[
            visible_in_all_cameras
        ]
        visible_screen_space_radii = (
            render_metadata.projected_radii[visible_in_all_cameras]
            .max(dim=-1)
            .values
        )

        densification_state.accumulated_image_plane_gradient_norms.index_add_(
            0,
            visible_gaussian_indices,
            visible_image_plane_gradients.norm(dim=-1),
        )
        densification_state.visibility_counts.index_add_(
            0,
            visible_gaussian_indices,
            torch.ones_like(
                visible_gaussian_indices,
                dtype=torch.float32,
            ),
        )
        if densification_state.max_screen_space_radii is not None:
            max_image_extent = float(
                max(
                    render_metadata.image_width,
                    render_metadata.image_height,
                )
            )
            densification_state.max_screen_space_radii[
                visible_gaussian_indices
            ] = torch.maximum(
                densification_state.max_screen_space_radii[
                    visible_gaussian_indices
                ],
                visible_screen_space_radii / max_image_extent,
            )

    @torch.no_grad()
    def _duplicate_and_split_gaussians(
        self,
        densification_state: Vanilla3DGSDensificationState,
        optimizers: dict[str, Optimizer],
        iteration: int,
    ) -> None:
        """Duplicate and split Gaussians based on accumulated statistics."""
        densification_config = densification_state.config
        if (
            densification_state.accumulated_image_plane_gradient_norms is None
            or densification_state.visibility_counts is None
        ):
            return

        average_image_plane_gradient_norms = (
            densification_state.accumulated_image_plane_gradient_norms
            / densification_state.visibility_counts.clamp_min(1)
        )
        exceeds_gradient_threshold = (
            average_image_plane_gradient_norms
            > densification_config.image_plane_gradient_magnitude_threshold
        )
        gaussian_scales = torch.exp(self.log_scales).max(dim=-1).values
        is_small_enough_for_duplication = (
            gaussian_scales
            <= densification_config.duplicate_max_normalized_scale_3d
            * densification_state.scene_scale
        )
        should_duplicate = (
            exceeds_gradient_threshold & is_small_enough_for_duplication
        )
        number_of_duplicates = int(should_duplicate.sum().item())

        should_split = (
            exceeds_gradient_threshold & ~is_small_enough_for_duplication
        )
        if (
            iteration
            < densification_config.screen_space_refinement_stop_iteration
            and densification_state.max_screen_space_radii is not None
        ):
            should_split |= (
                densification_state.max_screen_space_radii
                > densification_config.split_max_normalized_radius_2d
            )
        number_of_splits = int(should_split.sum().item())

        if number_of_duplicates > 0:
            self._duplicate_gaussians(should_duplicate, optimizers)

        if number_of_duplicates > 0:
            should_split = torch.cat(
                [
                    should_split,
                    torch.zeros(
                        number_of_duplicates,
                        dtype=torch.bool,
                        device=should_split.device,
                    ),
                ]
            )

        if number_of_splits > 0:
            self._split_gaussians(
                should_split,
                optimizers,
                densification_config.use_revised_opacity_after_split,
            )

        if densification_config.verbose and (
            number_of_duplicates > 0 or number_of_splits > 0
        ):
            print(
                f"Iteration {iteration}: duplicated {number_of_duplicates} "
                f"({number_of_duplicates / self.means.shape[0] * 100:.1f}%) "
                f"Gaussians and split {number_of_splits} "
                f"({number_of_splits / self.means.shape[0] * 100:.1f}%) Gaussians. "
                f"Scene now has {self.means.shape[0]} Gaussians."
            )

    @torch.no_grad()
    def _prune_gaussians(
        self,
        densification_state: Vanilla3DGSDensificationState,
        optimizers: dict[str, Optimizer],
        iteration: int,
    ) -> None:
        """Prune Gaussians based on opacity and scale thresholds."""
        densification_config = densification_state.config

        should_prune = (
            torch.sigmoid(self.logit_opacities.flatten())
            < densification_config.prune_opacity_threshold
        )
        if iteration > densification_config.opacity_reset_interval:
            exceeds_scene_scale_limit = (
                torch.exp(self.log_scales).max(dim=-1).values
                > densification_config.prune_max_normalized_scale_3d
                * densification_state.scene_scale
            )
            should_prune |= exceeds_scene_scale_limit
            if (
                iteration
                < densification_config.screen_space_refinement_stop_iteration
                and densification_state.max_screen_space_radii is not None
            ):
                should_prune |= (
                    densification_state.max_screen_space_radii
                    > densification_config.prune_max_normalized_radius_2d
                )

        number_of_pruned_gaussians = int(should_prune.sum().item())
        if number_of_pruned_gaussians > 0:
            self._remove_gaussians(should_prune, optimizers)

        if densification_config.verbose and number_of_pruned_gaussians > 0:
            print(
                f"Iteration {iteration}: pruned "
                f"{number_of_pruned_gaussians} Gaussians "
                f"({number_of_pruned_gaussians / self.means.shape[0] * 100:.1f}%)."
                f"Scene now has {self.means.shape[0]} Gaussians."
            )

    def _reset_densification_statistics(
        self,
        densification_state: Vanilla3DGSDensificationState,
    ) -> None:
        """Reset running densification statistics after a refine step."""
        if (
            densification_state.accumulated_image_plane_gradient_norms
            is not None
        ):
            densification_state.accumulated_image_plane_gradient_norms.zero_()
        if densification_state.visibility_counts is not None:
            densification_state.visibility_counts.zero_()
        if densification_state.max_screen_space_radii is not None:
            densification_state.max_screen_space_radii.zero_()

    def _scene_parameters(self) -> dict[str, nn.Parameter]:
        """Return the trainable scene parameters affected by densification."""
        return {
            "means": self.means,
            "unnormalized_quats": self.unnormalized_quats,
            "sh_0": self.sh_0,
            "sh_N": self.sh_N,
            "log_scales": self.log_scales,
            "logit_opacities": self.logit_opacities,
        }

    def _assign_scene_parameters(
        self,
        scene_parameters_by_name: dict[str, nn.Parameter],
    ) -> None:
        """Write an updated scene parameter mapping back onto the scene."""
        self.means = scene_parameters_by_name["means"]
        self.unnormalized_quats = scene_parameters_by_name["unnormalized_quats"]
        self.sh_0 = scene_parameters_by_name["sh_0"]
        self.sh_N = scene_parameters_by_name["sh_N"]
        self.log_scales = scene_parameters_by_name["log_scales"]
        self.logit_opacities = scene_parameters_by_name["logit_opacities"]

    def _build_split_parameter_factory(
        self,
        split_indices: torch.Tensor,
        kept_indices: torch.Tensor,
        split_offsets: torch.Tensor,
        selected_scales: torch.Tensor,
        use_revised_opacity_after_split: bool,
    ) -> Callable[[str, torch.Tensor], nn.Parameter]:
        """Build the parameter remap used when replacing each selected Gaussian with two children."""

        def split_parameter_rows(
            parameter_name: str,
            current_parameter: torch.Tensor,
        ) -> nn.Parameter:
            """Replace selected parameter rows with two split rows each."""
            repeat_shape = [2] + [1] * (current_parameter.dim() - 1)
            if parameter_name == "means":
                split_parameter_values = (
                    current_parameter[split_indices] + split_offsets
                ).reshape(-1, 3)
            elif parameter_name == "log_scales":
                split_parameter_values = torch.log(
                    selected_scales / 1.6
                ).repeat(2, 1)
            elif (
                parameter_name == "logit_opacities"
                and use_revised_opacity_after_split
            ):
                selected_opacities = torch.sigmoid(
                    current_parameter[split_indices]
                )
                revised_opacities = 1.0 - torch.sqrt(1.0 - selected_opacities)
                split_parameter_values = torch.logit(revised_opacities).repeat(
                    *repeat_shape
                )
            else:
                split_parameter_values = current_parameter[
                    split_indices
                ].repeat(*repeat_shape)
            updated_parameter_values = torch.cat(
                [
                    current_parameter[kept_indices],
                    split_parameter_values,
                ]
            )
            return nn.Parameter(
                updated_parameter_values,
                requires_grad=current_parameter.requires_grad,
            )

        return split_parameter_rows

    def _build_split_optimizer_state_factory(
        self,
        split_indices: torch.Tensor,
        kept_indices: torch.Tensor,
    ) -> Callable[[str, torch.Tensor], torch.Tensor]:
        """Build the optimizer-state remap used by the Gaussian split operation."""

        def split_optimizer_state_rows(
            optimizer_state_name: str,
            optimizer_state_tensor: torch.Tensor,
        ) -> torch.Tensor:
            """Keep existing state for surviving rows and zero-initialize split rows."""
            del optimizer_state_name
            split_state_values = torch.zeros(
                (
                    2 * split_indices.numel(),
                    *optimizer_state_tensor.shape[1:],
                ),
                device=optimizer_state_tensor.device,
            )
            return torch.cat(
                [optimizer_state_tensor[kept_indices], split_state_values]
            )

        return split_optimizer_state_rows

    @torch.no_grad()
    def _duplicate_gaussians(
        self,
        duplicate_mask: torch.Tensor,
        optimizers: dict[str, Optimizer],
    ) -> None:
        """Duplicate the selected Gaussians and extend optimizer state."""
        densification_state = self.densification_state
        if densification_state is None:
            raise RuntimeError("Densification state has not been initialized.")
        duplicate_indices = torch.where(duplicate_mask)[0]
        if duplicate_indices.numel() == 0:
            return

        # Append duplicated Gaussian parameters and extend the optimizer state with zeros.
        updated_scene_parameters = remap_parameters_and_optimizer_state(
            parameters_by_name=self._scene_parameters(),
            optimizers_by_parameter_name=optimizers,
            updated_parameter_factory=build_append_selected_parameter_factory(
                duplicate_indices
            ),
            updated_optimizer_state_factory=build_append_zero_optimizer_state_factory(
                duplicate_indices
            ),
        )
        self._assign_scene_parameters(updated_scene_parameters)

        # Keep the per-Gaussian densification statistics aligned with the resized scene tensors.
        if (
            densification_state.accumulated_image_plane_gradient_norms
            is not None
        ):
            densification_state.accumulated_image_plane_gradient_norms = torch.cat(
                [
                    densification_state.accumulated_image_plane_gradient_norms,
                    densification_state.accumulated_image_plane_gradient_norms[
                        duplicate_indices
                    ],
                ]
            )
        if densification_state.visibility_counts is not None:
            densification_state.visibility_counts = torch.cat(
                [
                    densification_state.visibility_counts,
                    densification_state.visibility_counts[duplicate_indices],
                ]
            )
        if densification_state.max_screen_space_radii is not None:
            densification_state.max_screen_space_radii = torch.cat(
                [
                    densification_state.max_screen_space_radii,
                    densification_state.max_screen_space_radii[
                        duplicate_indices
                    ],
                ]
            )

    @torch.no_grad()
    def _split_gaussians(
        self,
        split_mask: torch.Tensor,
        optimizers: dict[str, Optimizer],
        use_revised_opacity_after_split: bool,
    ) -> None:
        """Replace selected Gaussians with two split Gaussians each."""
        densification_state = self.densification_state
        if densification_state is None:
            raise RuntimeError("Densification state has not been initialized.")
        split_indices = torch.where(split_mask)[0]
        if split_indices.numel() == 0:
            return
        kept_indices = torch.where(~split_mask)[0]

        # Sample two child Gaussian offsets in each selected Gaussian's local frame.
        selected_scales = torch.exp(self.log_scales[split_indices])
        normalized_quaternions = F.normalize(
            self.unnormalized_quats[split_indices],
            dim=-1,
        )
        rotation_matrices = normalized_quaternion_to_rotation_matrix(
            normalized_quaternions
        )
        random_offsets_in_local_frame = torch.randn(
            2,
            selected_scales.shape[0],
            3,
            device=selected_scales.device,
        )
        split_offsets = torch.einsum(
            "num_gaussians row col, num_gaussians col, branch num_gaussians col -> branch num_gaussians row",
            rotation_matrices,
            selected_scales,
            random_offsets_in_local_frame,
        )

        # Replace each selected Gaussian with two children and reset their optimizer state.
        updated_scene_parameters = remap_parameters_and_optimizer_state(
            parameters_by_name=self._scene_parameters(),
            optimizers_by_parameter_name=optimizers,
            updated_parameter_factory=self._build_split_parameter_factory(
                split_indices=split_indices,
                kept_indices=kept_indices,
                split_offsets=split_offsets,
                selected_scales=selected_scales,
                use_revised_opacity_after_split=use_revised_opacity_after_split,
            ),
            updated_optimizer_state_factory=self._build_split_optimizer_state_factory(
                split_indices=split_indices,
                kept_indices=kept_indices,
            ),
        )
        self._assign_scene_parameters(updated_scene_parameters)

        # Mirror the same keep-and-repeat transformation on the densification statistics.
        if (
            densification_state.accumulated_image_plane_gradient_norms
            is not None
        ):
            repeated_gradient_norms = (
                densification_state.accumulated_image_plane_gradient_norms[
                    split_indices
                ].repeat(2)
            )
            densification_state.accumulated_image_plane_gradient_norms = torch.cat(
                [
                    densification_state.accumulated_image_plane_gradient_norms[
                        kept_indices
                    ],
                    repeated_gradient_norms,
                ]
            )
        if densification_state.visibility_counts is not None:
            repeated_visibility_counts = densification_state.visibility_counts[
                split_indices
            ].repeat(2)
            densification_state.visibility_counts = torch.cat(
                [
                    densification_state.visibility_counts[kept_indices],
                    repeated_visibility_counts,
                ]
            )
        if densification_state.max_screen_space_radii is not None:
            repeated_screen_space_radii = (
                densification_state.max_screen_space_radii[
                    split_indices
                ].repeat(2)
            )
            densification_state.max_screen_space_radii = torch.cat(
                [
                    densification_state.max_screen_space_radii[kept_indices],
                    repeated_screen_space_radii,
                ]
            )

    @torch.no_grad()
    def _remove_gaussians(
        self,
        prune_mask: torch.Tensor,
        optimizers: dict[str, Optimizer],
    ) -> None:
        """Remove the selected Gaussians and shrink optimizer state."""
        densification_state = self.densification_state
        if densification_state is None:
            raise RuntimeError("Densification state has not been initialized.")
        kept_indices = torch.where(~prune_mask)[0]

        # Keep only the surviving Gaussian parameters and optimizer state rows.
        updated_scene_parameters = remap_parameters_and_optimizer_state(
            parameters_by_name=self._scene_parameters(),
            optimizers_by_parameter_name=optimizers,
            updated_parameter_factory=build_keep_selected_parameter_factory(
                kept_indices
            ),
            updated_optimizer_state_factory=build_keep_selected_optimizer_state_factory(
                kept_indices
            ),
        )
        self._assign_scene_parameters(updated_scene_parameters)

        # Apply the same filtering to all per-Gaussian densification statistics.
        if (
            densification_state.accumulated_image_plane_gradient_norms
            is not None
        ):
            densification_state.accumulated_image_plane_gradient_norms = (
                densification_state.accumulated_image_plane_gradient_norms[
                    kept_indices
                ]
            )
        if densification_state.visibility_counts is not None:
            densification_state.visibility_counts = (
                densification_state.visibility_counts[kept_indices]
            )
        if densification_state.max_screen_space_radii is not None:
            densification_state.max_screen_space_radii = (
                densification_state.max_screen_space_radii[kept_indices]
            )

    @torch.no_grad()
    def _reset_opacities(
        self,
        optimizers: dict[str, Optimizer],
        post_sigmoid_opacity_value: float,
    ) -> None:
        """Clamp opacities to a lower value and reset opacity optimizer state."""
        max_logit_opacity = torch.logit(
            torch.tensor(
                post_sigmoid_opacity_value,
                device=self.logit_opacities.device,
            )
        ).item()

        def updated_parameter_factory(
            parameter_name: str,
            current_parameter: torch.Tensor,
        ) -> nn.Parameter:
            """Clamp opacity logits while leaving all other parameter names invalid."""
            if parameter_name != "logit_opacities":
                raise ValueError(
                    f"Unexpected parameter for opacity reset: {parameter_name!r}."
                )
            return nn.Parameter(
                torch.clamp(current_parameter, max=max_logit_opacity),
                requires_grad=current_parameter.requires_grad,
            )

        # Clamp opacity logits in place and zero the matching optimizer state.
        updated_scene_parameters = remap_parameters_and_optimizer_state(
            parameters_by_name=self._scene_parameters(),
            optimizers_by_parameter_name=optimizers,
            updated_parameter_factory=updated_parameter_factory,
            updated_optimizer_state_factory=build_zero_optimizer_state_factory(),
            parameter_names=("logit_opacities",),
        )
        self._assign_scene_parameters(updated_scene_parameters)

    # <------------> Evaluation <------------>

    @record_function("eval_step")
    def _eval_step(
        self,
        iteration: int,
        render_input: RenderInput,
        gt_image: Float[torch.Tensor, "batch height width 3"],
        lpips_module: LearnedPerceptualImagePatchSimilarity,
        ssim_module: StructuralSimilarityIndexMeasure,
        psnr_module: PeakSignalNoiseRatio,
    ) -> dict[str, torch.Tensor]:
        """Perform a single evaluation step for the scene."""
        render_output = self.render(render_input)
        rendered_image = torch.clamp(render_output.image, 0.0, 1.0)
        target_image = torch.clamp(gt_image, 0.0, 1.0)
        rendered_image_nchw = rearrange(
            rendered_image,
            "batch height width channels -> batch channels height width",
        )
        target_image_nchw = rearrange(
            target_image,
            "batch height width channels -> batch channels height width",
        )
        return {
            "psnr": psnr_module(rendered_image_nchw, target_image_nchw),
            "ssim": ssim_module(rendered_image_nchw, target_image_nchw),
            "lpips": lpips_module(rendered_image_nchw, target_image_nchw),
        }

    @torch.no_grad()
    def eval_loop(
        self,
        iteration: int,
        validation_loader: DataLoader[DataSample],
        lpips_module: LearnedPerceptualImagePatchSimilarity,
        ssim_module: StructuralSimilarityIndexMeasure,
        psnr_module: PeakSignalNoiseRatio,
        logger: SummaryWriter | None = None,
    ) -> dict[str, float]:
        """Perform an evaluation loop for the scene."""
        metrics_by_name: dict[str, list[torch.Tensor]] = defaultdict(list)
        device = self.means.device
        self.eval()
        lpips_module.eval()
        ssim_module.eval()
        psnr_module.eval()
        for validation_sample in validation_loader:
            validation_sample = validation_sample.to(device)
            validation_metrics = self._eval_step(
                iteration=iteration,
                render_input=validation_sample.render_input,
                gt_image=validation_sample.image,
                lpips_module=lpips_module,
                ssim_module=ssim_module,
                psnr_module=psnr_module,
            )
            for metric_name, metric_value in validation_metrics.items():
                metrics_by_name[metric_name].append(metric_value.detach())

        reduced_metrics = {
            metric_name: torch.stack(metric_values).mean().item()
            for metric_name, metric_values in metrics_by_name.items()
        }
        if logger is not None:
            for metric_name, metric_value in reduced_metrics.items():
                logger.add_scalar(
                    f"val/{metric_name}",
                    metric_value,
                    iteration,
                )
        self.train()
        return reduced_metrics
