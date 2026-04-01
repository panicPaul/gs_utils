"""Viewer support for gsplat-compatible PLY files."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization
from jaxtyping import Float
from plyfile import PlyData
from pydantic import BaseModel
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from gs_utils.contracts import (
    RenderInput,
    RenderMode,
    RenderOutput,
    RendersAlpha,
    RendersDepth,
    RendersRGB,
    Scene,
    SphericalHarmonics3DGS,
)
from gs_utils.viewer.extensions.alpha import AlphaViewer, AlphaViewerRenderState
from gs_utils.viewer.extensions.depth import DepthViewer, DepthViewerRenderState
from gs_utils.viewer.extensions.geometry import (
    GeometryViewer,
    GeometryViewerRenderState,
)
from gs_utils.viewer.host import Viewer, ViewerRenderState


@dataclass
class PlyViewerRenderState(
    ViewerRenderState,
    GeometryViewerRenderState,
    DepthViewerRenderState,
    AlphaViewerRenderState,
):
    """Composed render state for the gsplat PLY viewer."""


class GsplatPlyScene(
    Scene[BaseModel, Optimizer, LRScheduler, BaseModel],
    SphericalHarmonics3DGS,
    RendersRGB,
    RendersAlpha,
    RendersDepth,
):
    """Minimal 3DGS scene loaded from a gsplat-compatible PLY file."""

    def __init__(self) -> None:
        """Initialize an empty PLY-backed 3DGS scene."""
        super().__init__()
        self.means = nn.Parameter(torch.empty((0, 3)))
        self.unnormalized_quats = nn.Parameter(torch.empty((0, 4)))
        self.sh_0 = nn.Parameter(torch.empty((0, 1, 3)))
        self.sh_N = nn.Parameter(torch.empty((0, 15, 3)))
        self.log_scales = nn.Parameter(torch.empty((0, 3)))
        self.logit_opacities = nn.Parameter(torch.empty((0,)))

    @property
    def colors(self) -> Float[torch.Tensor, "num_splats 3"]:
        """Return RGB colors derived from the zeroth SH band."""
        return self.sh_0.squeeze(1)

    @classmethod
    def from_ply(cls, path: Path | str) -> "GsplatPlyScene":
        """Construct a scene from a gsplat-compatible spherical-harmonics PLY file."""
        scene = cls()
        scene._load_ply(path)
        return scene

    def _load_ply(self, path: Path | str) -> None:
        """Load the scene parameters from a gsplat-compatible PLY file."""
        ply_path = Path(path)
        vertex_data = PlyData.read(ply_path)["vertex"].data

        # Read the core gsplat geometry fields from the structured vertex array.
        means = _stack_vertex_fields(vertex_data, ["x", "y", "z"])
        log_scales = _stack_vertex_fields(
            vertex_data,
            ["scale_0", "scale_1", "scale_2"],
        )
        unnormalized_quaternions = _stack_vertex_fields(
            vertex_data,
            ["rot_0", "rot_1", "rot_2", "rot_3"],
        )
        logit_opacities = _read_vertex_field(vertex_data, "opacity")

        # Reconstruct the SH tensors from the gsplat PLY coefficient layout.
        zeroth_band_coefficients = _stack_vertex_fields(
            vertex_data,
            ["f_dc_0", "f_dc_1", "f_dc_2"],
        )[:, None, :]
        higher_band_names = _sorted_higher_band_names(vertex_data.dtype.names)
        higher_band_coefficients = _reconstruct_higher_band_coefficients(
            vertex_data,
            higher_band_names,
        )

        self.means = nn.Parameter(torch.from_numpy(means).float())
        self.log_scales = nn.Parameter(torch.from_numpy(log_scales).float())
        self.unnormalized_quats = nn.Parameter(
            torch.from_numpy(unnormalized_quaternions).float()
        )
        self.logit_opacities = nn.Parameter(
            torch.from_numpy(logit_opacities).float()
        )
        self.sh_0 = nn.Parameter(
            torch.from_numpy(zeroth_band_coefficients).float()
        )
        self.sh_N = nn.Parameter(
            torch.from_numpy(higher_band_coefficients).float()
        )

    def render(
        self,
        render_input: RenderInput,
        render_mode: RenderMode = RenderMode.RGB,
    ) -> RenderOutput:
        """Render the loaded PLY scene for the provided camera state."""
        spherical_harmonics = torch.cat([self.sh_0, self.sh_N], dim=1)
        scales = torch.exp(self.log_scales)
        opacities = torch.sigmoid(self.logit_opacities)
        max_spherical_harmonic_degree = int(
            math.sqrt(spherical_harmonics.shape[1]) - 1
        )

        # Depth uses the expected-depth path, while RGB and alpha use the lighter RGB path.
        gsplat_render_mode = (
            "RGB+ED" if render_mode == RenderMode.DEPTH else "RGB"
        )
        rendered_image, rendered_alphas, _ = rasterization(
            means=self.means,
            quats=self.unnormalized_quats,
            scales=scales,
            opacities=opacities,
            colors=spherical_harmonics,
            viewmats=render_input.cam_to_world,
            Ks=render_input.get_intrinsics(),
            height=render_input.height,
            width=render_input.width,
            backgrounds=render_input.background,
            sh_degree=max_spherical_harmonic_degree,
            render_mode=gsplat_render_mode,
        )

        rendered_rgb = rendered_image[..., :3]
        rendered_depth = (
            rendered_image[..., 3:] if gsplat_render_mode == "RGB+ED" else None
        )
        if render_input.background is not None:
            rendered_rgb = rendered_rgb * rendered_alphas + (
                render_input.background * (1 - rendered_alphas)
            )

        return RenderOutput(
            image=rendered_rgb,
            depth=rendered_depth,
            alpha=rendered_alphas,
        )

    def initialize_optimizers(self, config: BaseModel) -> dict[str, Optimizer]:
        """PLY viewer scenes do not expose training optimizers."""
        del config
        raise NotImplementedError("PLY viewer scenes do not support training.")

    def initialize_densification(
        self,
        config: BaseModel,
        scene_scale: float = 1.0,
    ) -> None:
        """PLY viewer scenes do not expose densification."""
        del config, scene_scale
        raise NotImplementedError(
            "PLY viewer scenes do not support densification."
        )

    def initialize_schedulers(
        self,
        optimizers: dict[str, Optimizer],
        config: BaseModel,
    ) -> dict[str, LRScheduler]:
        """PLY viewer scenes do not expose learning-rate schedulers."""
        del optimizers, config
        raise NotImplementedError("PLY viewer scenes do not support training.")

    def densification_step_pre_backward(
        self,
        iteration: int,
        optimizers: dict[str, Optimizer],
    ) -> None:
        """PLY viewer scenes do not expose training hooks."""
        del iteration, optimizers
        raise NotImplementedError("PLY viewer scenes do not support training.")

    def densification_step_post_backward(
        self,
        iteration: int,
        optimizers: dict[str, Optimizer],
    ) -> None:
        """PLY viewer scenes do not expose training hooks."""
        del iteration, optimizers
        raise NotImplementedError("PLY viewer scenes do not support training.")


class GsplatPlyViewer(Viewer, GeometryViewer, DepthViewer, AlphaViewer):
    """Viewer for arbitrary gsplat-compatible PLY scenes."""

    render_state_type = PlyViewerRenderState


@dataclass
class ViewPlyCommand:
    """Load a gsplat-compatible PLY file and launch the generic viewer."""

    ply: Path
    host: str = "0.0.0.0"
    port: int = 8080

    def __call__(self) -> None:
        """Launch the viewer for the provided PLY file."""
        scene = GsplatPlyScene.from_ply(self.ply)
        GsplatPlyViewer(
            scene=scene,
            output_dir=self.ply.parent,
            host=self.host,
            port=self.port,
        ).launch()


def _read_vertex_field(
    vertex_data: np.ndarray,
    field_name: str,
) -> np.ndarray:
    """Read one float-valued field from the PLY vertex array."""
    if field_name not in vertex_data.dtype.names:
        raise ValueError(f"PLY file is missing required field {field_name!r}.")
    return np.asarray(vertex_data[field_name], dtype=np.float32)


def _stack_vertex_fields(
    vertex_data: np.ndarray,
    field_names: list[str],
) -> np.ndarray:
    """Read multiple float-valued vertex fields and stack them column-wise."""
    return np.stack(
        [
            _read_vertex_field(vertex_data, field_name)
            for field_name in field_names
        ],
        axis=1,
    )


def _sorted_higher_band_names(field_names: tuple[str, ...] | None) -> list[str]:
    """Return higher-band SH field names sorted by their numeric suffix."""
    if field_names is None:
        raise ValueError("PLY file does not contain any vertex fields.")
    higher_band_names = [
        field_name
        for field_name in field_names
        if field_name.startswith("f_rest_")
    ]
    if not higher_band_names:
        raise ValueError("PLY file is missing higher-band SH fields.")
    return sorted(
        higher_band_names,
        key=lambda field_name: int(field_name.removeprefix("f_rest_")),
    )


def _reconstruct_higher_band_coefficients(
    vertex_data: np.ndarray,
    higher_band_names: list[str],
) -> np.ndarray:
    """Reconstruct the higher-band SH tensor from flattened gsplat PLY fields."""
    flattened_higher_band_coefficients = _stack_vertex_fields(
        vertex_data,
        higher_band_names,
    )
    if flattened_higher_band_coefficients.shape[1] % 3 != 0:
        raise ValueError(
            "PLY file has an invalid number of higher-band SH coefficients."
        )

    num_higher_band_coefficients = (
        flattened_higher_band_coefficients.shape[1] // 3
    )
    num_spherical_harmonic_coefficients = 1 + num_higher_band_coefficients
    spherical_harmonic_degree = int(
        math.isqrt(num_spherical_harmonic_coefficients) - 1
    )
    if (
        spherical_harmonic_degree + 1
    ) ** 2 != num_spherical_harmonic_coefficients:
        raise ValueError(
            "PLY file does not contain a square number of SH coefficients."
        )

    return flattened_higher_band_coefficients.reshape(
        flattened_higher_band_coefficients.shape[0],
        3,
        num_higher_band_coefficients,
    ).transpose(0, 2, 1)
