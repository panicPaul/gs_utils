"""Shared helper modules and utilities for example training scripts."""

# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

_NUMPY_RNG = np.random.default_rng()


def knn(x: Tensor, K: int = 4) -> Tensor:
    """Return Euclidean distances to the `K` nearest neighbors for each point."""
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert RGB values in `[0, 1]` to the DC spherical harmonics coefficient."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    global _NUMPY_RNG

    random.seed(seed)
    _NUMPY_RNG = np.random.default_rng(seed)
    torch.manual_seed(seed)


def get_numpy_rng() -> np.random.Generator:
    """Return the shared NumPy random generator used by the examples."""
    return _NUMPY_RNG


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img: np.ndarray, cmap: str = "jet") -> Tensor:
    """Render a Matplotlib colormap legend image for the input array."""
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img: torch.Tensor = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(
    img: torch.Tensor, colormap: str = "turbo"
) -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float | None = None,
    far_plane: float | None = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img
