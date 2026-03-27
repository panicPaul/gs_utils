"""Contracts and helpers for derived depth and normal supervision."""
# NOTE: nothing more than a rough draft at the moment

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor

from gs_utils.data.contract import DataSample


@runtime_checkable
class DepthEstimator(Protocol):
    """Contract for depth prediction over data samples."""

    def estimate_depth(
        self, sample: DataSample
    ) -> Float[Tensor, "height width 1"]:
        """Estimate depth for a given data sample."""


@runtime_checkable
class NormalEstimator(Protocol):
    """Contract for normal prediction over data samples."""

    def estimate_normals(
        self, sample: DataSample
    ) -> Float[Tensor, "height width 3"]:
        """Estimate normals for a given data sample."""
