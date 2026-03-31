"""Capability contracts for geometry and render outputs."""

from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch.nn import Parameter


@runtime_checkable
class Splat3DGS(Protocol):
    """Geometry contract for 3D Gaussian splats."""

    means: Float[Parameter, "num_splats 3"]
    unnormalized_quats: Float[Parameter, "num_splats 4"]
    log_scales: Float[Parameter, "num_splats 3"]
    logit_opacities: Float[Parameter, "num_splats"]

    @property
    def colors(self) -> Float[torch.Tensor, "num_splats feature_dim"]:
        """Features of the splats, usually in RGB format."""
        raise NotImplementedError


@runtime_checkable
class ConfidenceBased3DGS(Splat3DGS, Protocol):
    """Geometry contract for confidence-based 3D Gaussian splats."""

    log_confidences: Float[Parameter, "num_splats"]


@runtime_checkable
class SphericalHarmonics3DGS(Splat3DGS, Protocol):
    """Geometry contract for spherical harmonics-based 3D Gaussian splats."""

    sh_0: Float[Parameter, "num_splats 1 3"]
    sh_N: Float[Parameter, "num_splats num_coeffs-1 3"]

    def colors(self) -> Float[torch.Tensor, "num_splats feature_dim"]:
        """Zeroth band spherical harmonics coefficients."""
        return self.sh_0.squeeze(1)


@runtime_checkable
class Splat2DGS(Protocol):
    """Geometry contract for 2D Gaussian splats."""

    means: Float[Parameter, "num_splats 2"]
    unnormalized_quats: Float[Parameter, "num_splats 4"]
    log_scales: Float[Parameter, "num_splats 2"]
    logit_opacities: Float[Parameter, "num_splats"]

    @property
    def colors(self) -> Float[Parameter, "num_splats num_features"]:
        """Features of the splats, usually in RGB format."""
        raise NotImplementedError


@runtime_checkable
class SphericalHarmonics2DGS(Splat2DGS, Protocol):
    """Geometry contract for spherical harmonics-based 2D Gaussian splats."""

    sh_0: Float[Parameter, "num_splats 1 3"]
    sh_N: Float[Parameter, "num_splats num_coeffs-1 3"]

    def colors(self) -> Float[torch.Tensor, "num_splats feature_dim"]:
        """Zeroth band spherical harmonics coefficients."""
        return self.sh_0.squeeze(1)


@runtime_checkable
class ConfidenceBased2DGS(Splat2DGS, Protocol):
    """Geometry contract for confidence-based 3D Gaussian splats."""

    log_confidences: Float[Parameter, "num_splats"]


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
