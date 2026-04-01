"""PLY export for splat geometry contracts."""

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from gs_utils.contracts import (
    SphericalHarmonics2DGS,
    SphericalHarmonics3DGS,
    Splat2DGS,
    Splat3DGS,
)


def _to_three_dimensional_means(
    geometry: Splat3DGS | Splat2DGS,
) -> np.ndarray:
    """Return xyz positions for 2D or 3D splat geometry."""
    means = geometry.means.detach().cpu().numpy()
    if means.shape[1] == 3:
        return means
    if means.shape[1] == 2:
        zero_depth = np.zeros((means.shape[0], 1), dtype=means.dtype)
        return np.concatenate([means, zero_depth], axis=1)
    raise ValueError(f"Unsupported means shape {means.shape}.")


def _to_three_dimensional_log_scales(
    geometry: Splat3DGS | Splat2DGS,
) -> np.ndarray:
    """Return three log-scale channels for 2D or 3D splat geometry."""
    log_scales = geometry.log_scales.detach().cpu().numpy()
    if log_scales.shape[1] == 3:
        return log_scales
    if log_scales.shape[1] == 2:
        zero_log_scale = np.zeros(
            (log_scales.shape[0], 1),
            dtype=log_scales.dtype,
        )
        return np.concatenate([log_scales, zero_log_scale], axis=1)
    raise ValueError(f"Unsupported log-scales shape {log_scales.shape}.")


def _to_spherical_harmonics(
    geometry: Splat3DGS | Splat2DGS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return SH coefficients for export, promoting RGB colors when needed."""
    if isinstance(geometry, (SphericalHarmonics3DGS, SphericalHarmonics2DGS)):
        zeroth_band_coefficients = geometry.sh_0.detach().cpu().numpy()
        higher_band_coefficients = geometry.sh_N.detach().cpu().numpy()
        return zeroth_band_coefficients, higher_band_coefficients

    colors = geometry.colors.detach().cpu().numpy()
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise ValueError(
            "Non-spherical-harmonics splat export requires RGB colors with "
            "shape [num_splats, 3]."
        )

    zeroth_band_coefficients = colors[:, :3][:, None, :]
    higher_band_coefficients = np.zeros(
        (colors.shape[0], 15, 3),
        dtype=colors.dtype,
    )
    return zeroth_band_coefficients, higher_band_coefficients


def _flatten_higher_band_coefficients(
    higher_band_coefficients: np.ndarray,
) -> np.ndarray:
    """Flatten higher-band SH coefficients to the gsplat PLY layout."""
    num_splats = higher_band_coefficients.shape[0]
    num_flattened_features = (
        higher_band_coefficients.shape[1] * higher_band_coefficients.shape[2]
    )
    return higher_band_coefficients.transpose(0, 2, 1).reshape(
        num_splats,
        num_flattened_features,
    )


def _vertex_dtype(num_higher_band_features: int) -> list[tuple[str, str]]:
    """Return the structured dtype for gsplat-compatible PLY vertices."""
    names: list[str] = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    ]
    names.extend(
        f"f_rest_{feature_index}"
        for feature_index in range(num_higher_band_features)
    )
    names.append("opacity")
    names.extend(f"scale_{axis_index}" for axis_index in range(3))
    names.extend(f"rot_{quat_index}" for quat_index in range(4))
    return [(name, "f4") for name in names]


def export_ply(
    geometry: Splat3DGS | Splat2DGS,
    path: Path | str,
) -> None:
    """Export splat geometry to gsplat-compatible spherical-harmonics PLY."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    means = _to_three_dimensional_means(geometry)
    normals = np.zeros_like(means)
    zeroth_band_coefficients, higher_band_coefficients = (
        _to_spherical_harmonics(geometry)
    )
    flattened_higher_band_coefficients = _flatten_higher_band_coefficients(
        higher_band_coefficients
    )
    logit_opacities = geometry.logit_opacities.detach().cpu().numpy()[:, None]
    log_scales = _to_three_dimensional_log_scales(geometry)
    unnormalized_quaternions = (
        geometry.unnormalized_quats.detach().cpu().numpy()
    )

    attributes = np.concatenate(
        [
            means,
            normals,
            zeroth_band_coefficients[:, 0, :],
            flattened_higher_band_coefficients,
            logit_opacities,
            log_scales,
            unnormalized_quaternions,
        ],
        axis=1,
    )

    vertex_array = np.empty(
        means.shape[0],
        dtype=_vertex_dtype(flattened_higher_band_coefficients.shape[1]),
    )
    vertex_array[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(vertex_array, "vertex")]).write(path)
