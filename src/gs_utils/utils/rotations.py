"""Rotation math helpers."""

import torch
from jaxtyping import Float


def normalized_quaternion_to_rotation_matrix(
    normalized_quaternions: Float[torch.Tensor, "num_gaussians 4"],
) -> Float[torch.Tensor, "num_gaussians 3 3"]:
    """Convert normalized quaternions in wxyz convention to rotation matrices."""
    quaternion_w, quaternion_x, quaternion_y, quaternion_z = torch.unbind(
        normalized_quaternions,
        dim=-1,
    )
    rotation_matrix_entries = torch.stack(
        [
            1 - 2 * (quaternion_y**2 + quaternion_z**2),
            2 * (quaternion_x * quaternion_y - quaternion_w * quaternion_z),
            2 * (quaternion_x * quaternion_z + quaternion_w * quaternion_y),
            2 * (quaternion_x * quaternion_y + quaternion_w * quaternion_z),
            1 - 2 * (quaternion_x**2 + quaternion_z**2),
            2 * (quaternion_y * quaternion_z - quaternion_w * quaternion_x),
            2 * (quaternion_x * quaternion_z - quaternion_w * quaternion_y),
            2 * (quaternion_y * quaternion_z + quaternion_w * quaternion_x),
            1 - 2 * (quaternion_x**2 + quaternion_y**2),
        ],
        dim=-1,
    )
    return rotation_matrix_entries.reshape(
        normalized_quaternions.shape[:-1] + (3, 3)
    )
