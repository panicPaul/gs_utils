"""Visualization and colormap helpers."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from torch import Tensor


def colormap(img: np.ndarray, cmap: str = "jet") -> Tensor:
    """Render a Matplotlib colormap legend image for the input array."""
    width, height = img.shape[:2]
    dots_per_inch = 300
    figure, axis = plt.subplots(
        1,
        figsize=(height / dots_per_inch, width / dots_per_inch),
        dpi=dots_per_inch,
    )
    rendered_image = axis.imshow(img, cmap=cmap)
    axis.set_axis_off()
    figure.colorbar(rendered_image, ax=axis)
    figure.tight_layout()
    figure.canvas.draw()
    buffer = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    buffer = buffer.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    output_image: torch.Tensor = (
        torch.from_numpy(buffer).float().permute(2, 0, 1)
    )
    plt.close()
    return output_image


def apply_float_colormap(
    image: torch.Tensor, colormap: str = "turbo"
) -> torch.Tensor:
    """Convert a single-channel tensor into a colored image."""
    image = torch.nan_to_num(image, 0)
    if colormap == "gray":
        return image.repeat(1, 1, 3)
    image_as_indices = (image * 255).long()
    image_index_min = torch.min(image_as_indices)
    image_index_max = torch.max(image_as_indices)
    assert image_index_min >= 0, f"the min value is {image_index_min}"
    assert image_index_max <= 255, f"the max value is {image_index_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=image.device,
    )[image_as_indices[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    accumulation: torch.Tensor | None = None,
    near_plane: float | None = None,
    far_plane: float | None = None,
) -> torch.Tensor:
    """Convert a depth image to a color image for visualization."""
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    image = apply_float_colormap(depth, colormap="turbo")
    if accumulation is not None:
        image = image * accumulation + (1.0 - accumulation)
    return image
