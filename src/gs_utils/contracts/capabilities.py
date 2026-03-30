"""Capability contracts for geometry and render outputs."""

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class Splat3DGS(Protocol):
    """Geometry contract for 3D Gaussian splats."""

    @property
    def means(self) -> Float[Tensor, "num_splats 3"]:
        """Mean positions of the splats."""

    @property
    def quats(self) -> Float[Tensor, "num_splats 4"]:
        """Quaternion rotations of the splats."""

    @property
    def scales(self) -> Float[Tensor, "num_splats 3"]:
        """Scales of the splats."""

    @property
    def opacities(self) -> Float[Tensor, "num_splats"]:
        """Opacities of the splats."""

    @property
    def colors(self) -> Float[Tensor, "num_splats num_bases 3"]:
        """Spherical harmonic coefficients of the splats."""


@runtime_checkable
class ConfidenceBased3DGS(Splat3DGS, Protocol):
    """Geometry contract for confidence-based 3D Gaussian splats."""

    @property
    def confidences(self) -> Float[Tensor, "num_splats"]:
        """Confidences of the splats."""


@runtime_checkable
class Splat2DGS(Protocol):
    """Geometry contract for 2D Gaussian splats."""

    @property
    def means(self) -> Float[Tensor, "num_splats 3"]:
        """Mean positions of the splats."""

    @property
    def quats(self) -> Float[Tensor, "num_splats 4"]:
        """Quaternion rotations of the splats."""

    @property
    def scales(self) -> Float[Tensor, "num_splats 2"]:
        """Scales of the splats."""

    @property
    def opacities(self) -> Float[Tensor, "num_splats"]:
        """Opacities of the splats."""

    @property
    def colors(self) -> Float[Tensor, "num_splats num_bases 3"]:
        """Spherical harmonic coefficients of the splats."""


@runtime_checkable
class RendersRGB(Protocol):
    """Renderability contract for RGB output."""


@runtime_checkable
class RendersDepth(Protocol):
    """Renderability contract for depth output."""


@runtime_checkable
class RendersNormals(Protocol):
    """Renderability contract for normals output."""


@runtime_checkable
class RendersAlpha(Protocol):
    """Renderability contract for alpha output."""


@runtime_checkable
class DifferentiableDepth(RendersDepth, Protocol):
    """Differentiability contract for depth output."""


@runtime_checkable
class DifferentiableNormals(RendersNormals, Protocol):
    """Differentiability contract for normals output."""
