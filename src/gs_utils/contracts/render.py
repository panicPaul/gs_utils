"""Typed render requests and outputs shared across scenes and viewers."""

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import torch
from jaxtyping import Float

from gs_utils.contracts.capabilities import (
    RendersAlpha,
    RendersDepth,
    RendersNormals,
    RendersRGB,
)


class RenderMode(StrEnum):
    """Shared render modes gated by scene capabilities."""

    RGB = "rgb"
    DEPTH = "depth"
    NORMALS = "normals"
    ALPHA = "alpha"
    RGB_DEPTH = "rgb+depth"
    RGB_NORMALS = "rgb+normals"
    RGB_ALPHA = "rgb+alpha"
    RGB_DEPTH_NORMALS = "rgb+depth+normals"
    RGB_DEPTH_ALPHA = "rgb+depth+alpha"
    RGB_NORMALS_ALPHA = "rgb+normals+alpha"
    RGB_DEPTH_NORMALS_ALPHA = "rgb+depth+normals+alpha"

    def check_is_supported(self, scene: object) -> bool:
        """Return whether the scene supports this render mode by contract."""
        match self:
            case RenderMode.RGB:
                return isinstance(scene, RendersRGB)
            case RenderMode.DEPTH:
                return isinstance(scene, RendersDepth)
            case RenderMode.NORMALS:
                return isinstance(scene, RendersNormals)
            case RenderMode.ALPHA:
                return isinstance(scene, RendersAlpha)
            case RenderMode.RGB_DEPTH:
                return isinstance(scene, RendersDepth)
            case RenderMode.RGB_NORMALS:
                return isinstance(scene, RendersNormals)
            case RenderMode.RGB_ALPHA:
                return isinstance(scene, RendersAlpha)
            case RenderMode.RGB_DEPTH_NORMALS:
                return isinstance(scene, RendersDepth) and isinstance(
                    scene, RendersNormals
                )
            case RenderMode.RGB_DEPTH_ALPHA:
                return isinstance(scene, RendersDepth) and isinstance(
                    scene, RendersAlpha
                )
            case RenderMode.RGB_NORMALS_ALPHA:
                return isinstance(scene, RendersNormals) and isinstance(
                    scene, RendersAlpha
                )
            case RenderMode.RGB_DEPTH_NORMALS_ALPHA:
                return (
                    isinstance(scene, RendersDepth)
                    and isinstance(scene, RendersNormals)
                    and isinstance(scene, RendersAlpha)
                )

    @classmethod
    def supported_for_scene(cls, scene: object) -> list["RenderMode"]:
        """Return all shared render modes supported by a given scene."""
        return [mode for mode in cls if mode.check_is_supported(scene)]


@dataclass(slots=True, frozen=True)
class RenderInput:
    """Canonical scene render input."""

    cam_to_world: Float[torch.Tensor, "... 4 4"]
    width: int
    height: int
    intrinsics: Float[torch.Tensor, "... 3 3"] | None = None
    fov: float | torch.Tensor | None = None
    render_mode: RenderMode = RenderMode.RGB
    background: Float[torch.Tensor, "3"] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_intrinsics = self.intrinsics is not None
        has_fov = self.fov is not None
        if has_intrinsics == has_fov:
            raise ValueError(
                "RenderInput requires exactly one of `intrinsics` or `fov`."
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("RenderInput width and height must be positive.")

    def get_intrinsics(self) -> Float[torch.Tensor, "... 3 3"]:
        """Return intrinsics, computing them lazily from the vertical FOV if needed."""
        if self.intrinsics is not None:
            return self.intrinsics

        vertical_field_of_view = self.get_fov()
        focal_length = (
            0.5 * self.height / torch.tan(vertical_field_of_view / 2.0)
        )
        intrinsics = torch.zeros(
            (*vertical_field_of_view.shape, 3, 3),
            dtype=vertical_field_of_view.dtype,
            device=vertical_field_of_view.device,
        )
        intrinsics[..., 0, 0] = focal_length
        intrinsics[..., 1, 1] = focal_length
        intrinsics[..., 0, 2] = self.width / 2.0
        intrinsics[..., 1, 2] = self.height / 2.0
        intrinsics[..., 2, 2] = 1.0
        return intrinsics

    def get_fov(self) -> torch.Tensor:
        """Return the vertical FOV, computing it lazily from intrinsics if needed."""
        if self.fov is not None:
            if isinstance(self.fov, torch.Tensor):
                return self.fov
            return torch.tensor(
                self.fov,
                dtype=self.cam_to_world.dtype,
                device=self.cam_to_world.device,
            )

        intrinsics_matrix = self.intrinsics
        if intrinsics_matrix is None:
            raise ValueError("RenderInput requires intrinsics or fov.")
        return 2.0 * torch.atan(
            torch.tensor(
                self.height,
                dtype=intrinsics_matrix.dtype,
                device=intrinsics_matrix.device,
            )
            / (2.0 * intrinsics_matrix[..., 1, 1])
        )

    def to(self, device: torch.device) -> "RenderInput":
        """Move tensor fields in the render input to the target device."""
        return replace(
            self,
            cam_to_world=self.cam_to_world.to(device),
            intrinsics=(
                None if self.intrinsics is None else self.intrinsics.to(device)
            ),
            fov=(
                None
                if self.fov is None
                else (
                    self.fov.to(device)
                    if isinstance(self.fov, torch.Tensor)
                    else self.fov
                )
            ),
            background=(
                None if self.background is None else self.background.to(device)
            ),
        )


@dataclass(slots=True)
class RenderOutput:
    """Canonical scene render output."""

    image: Float[torch.Tensor, "height width 3"]
    depth: Float[torch.Tensor, "height width 1"] | None = None
    normals: Float[torch.Tensor, "height width 3"] | None = None
    alpha: Float[torch.Tensor, "height width 1"] | None = None
    aux: Any = field(default_factory=dict)
