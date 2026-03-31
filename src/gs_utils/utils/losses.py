"""Utility functions for training losses."""

from typing import Literal

import torch
import torch.nn as nn
from fused_ssim import FusedSSIMMap
from jaxtyping import Float


def ssim_loss(
    pred: Float[torch.Tensor, "batch height width 3"],
    target: Float[torch.Tensor, "batch height width 3"],
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Float[torch.Tensor, ""] | Float[torch.Tensor, "batch height width"]:
    """Computes the SSIM loss between predicted and target tensors.

    Args:
        pred: The predicted RGB tensor. [batch, height, width, 3]
        target: The target RGB tensor. [batch, height, width, 3]
        reduction: The reduction method to apply to the loss.

    Returns:
        The computed SSIM loss.
    """
    C1 = 0.01**2
    C2 = 0.03**2

    pred = pred.contiguous()
    map = FusedSSIMMap.apply(
        C1,
        C2,
        pred,
        target,
        padding="same",
        train=True,
        spatial_dims=2,
    )
    match reduction:
        case "mean":
            return 1.0 - torch.mean(map)
        case "sum":
            return 1.0 - torch.sum(map)
        case "none":
            return 1.0 - map
        case _:
            raise ValueError(f"Invalid reduction method: {reduction}")


def photometric_loss(
    pred: Float[torch.Tensor, "batch height width 3"],
    target: Float[torch.Tensor, "batch height width 3"],
    reduction: Literal["mean", "sum", "none"] = "mean",
    lambda_ssim: float = 0.2,
) -> Float[torch.Tensor, ""] | Float[torch.Tensor, "batch height width"]:
    """Computes the photometric loss between predicted and target tensors.

    Args:
        pred: The predicted RGB tensor. [batch, height, width, 3]
        target: The target RGB tensor. [batch, height, width, 3]
        reduction: The reduction method to apply to the loss.
        lambda_ssim: The weight of the SSIM loss.

    Returns:
        The computed photometric loss.
    """
    assert 0.0 <= lambda_ssim <= 1.0, "lambda_ssim must be between 0.0 and 1.0"
    l1 = nn.functional.l1_loss(pred, target, reduction=reduction)
    ssim = ssim_loss(pred, target, reduction=reduction)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * ssim


def confidence_loss(
    pred: Float[torch.Tensor, "batch height width 3"],
    target: Float[torch.Tensor, "batch height width 3"],
    confidences: Float[torch.Tensor, "batch height width"],
    beta: float = 7.5e-2,
    confidence_clamping: tuple[float, float] | None = (0.001, 5.0),
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Float[torch.Tensor, ""] | Float[torch.Tensor, "batch height width"]:
    """Computes the confidence loss between predicted and target tensors.

    https://arxiv.org/pdf/2603.24725#page=19.24

    NOTE: you should train without confidence loss for a few hounred iterations before adding it back.
          The original authors trained with the photometric loss only for the first 500 iterations, then
          added the confidence loss.

    Args:
        pred: The predicted RGB tensor. [batch, height, width, 3]
        target: The target RGB tensor. [batch, height, width, 3]
        confidences: The alpha-blended confidence tensor. [batch, height, width]
        beta: The confidence loss weight.
        confidence_clamping: The confidence clamping range.
        reduction: The reduction method to apply to the loss.

    Returns:
        The computed confidence loss.
    """
    photometric_part = nn.functional.l1_loss(pred, target, reduction=reduction)

    if confidence_clamping is not None:
        confidences = torch.clamp(
            confidences, min=confidence_clamping[0], max=confidence_clamping[1]
        )
    confidence_part = -beta * torch.log(confidences)

    match reduction:
        case "mean":
            return photometric_part + confidence_part.mean()
        case "sum":
            return photometric_part + confidence_part.sum()
        case "none":
            return photometric_part + confidence_part
