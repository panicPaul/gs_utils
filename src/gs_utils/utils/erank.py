"""Effective Rank."""

import torch
from jaxtyping import Float

from gs_utils.contracts import Splat3DGS


def compute_effective_rank(
    scene: Splat3DGS,
) -> Float[torch.Tensor, " num_splats"]:
    """Compute the effective rank."""
    squared_scales = torch.exp(2 * scene.log_scales)
    normed_squared_scales = squared_scales / squared_scales.sum(
        dim=-1, keepdim=True
    )
    entropy = -torch.sum(
        normed_squared_scales * torch.log(normed_squared_scales), dim=-1
    )
    return torch.exp(entropy)
