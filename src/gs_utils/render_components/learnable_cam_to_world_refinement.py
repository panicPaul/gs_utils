"""Learnable camera-to-world pose refinement component."""

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


def _rotation_6d_to_matrix(
    rotation_6d: Float[torch.Tensor, "... 6"],
) -> Float[torch.Tensor, "... 3 3"]:
    """Convert a 6D rotation representation to a rotation matrix."""
    first_basis_input = rotation_6d[..., :3]
    second_basis_input = rotation_6d[..., 3:]
    first_basis_vector = F.normalize(first_basis_input, dim=-1)
    second_basis_vector = (
        second_basis_input
        - (first_basis_vector * second_basis_input).sum(-1, keepdim=True)
        * first_basis_vector
    )
    second_basis_vector = F.normalize(second_basis_vector, dim=-1)
    third_basis_vector = torch.cross(
        first_basis_vector,
        second_basis_vector,
        dim=-1,
    )
    return torch.stack(
        (first_basis_vector, second_basis_vector, third_basis_vector),
        dim=-2,
    )


class LearnableCamToWorldRefinement(torch.nn.Module):
    """Learn per-camera SE(3)-like pose deltas in camera-to-world space."""

    def __init__(self, num_cameras: int) -> None:
        """Initialize learnable pose deltas for a fixed camera set."""
        super().__init__()
        self.embeds = torch.nn.Embedding(num_cameras, 9)
        self.register_buffer(
            "identity",
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )

    def zero_init(self) -> None:
        """Initialize all pose deltas to zero."""
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float) -> None:
        """Initialize pose deltas from a zero-mean normal distribution."""
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(
        self,
        cam_to_world: Float[torch.Tensor, "... 4 4"],
        embed_ids: Int[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "... 4 4"]:
        """Apply learnable camera-to-world refinements."""
        if cam_to_world.shape[:-2] != embed_ids.shape:
            raise ValueError(
                "cam_to_world batch dimensions must match embed_ids shape."
            )
        batch_dims = cam_to_world.shape[:-2]
        pose_deltas = self.embeds(embed_ids)
        translation_deltas = pose_deltas[..., :3]
        rotation_deltas = pose_deltas[..., 3:]
        refined_rotation = _rotation_6d_to_matrix(
            rotation_deltas + self.identity.expand(*batch_dims, -1)
        )
        refinement_transform = torch.eye(
            4,
            dtype=pose_deltas.dtype,
            device=pose_deltas.device,
        ).repeat((*batch_dims, 1, 1))
        refinement_transform[..., :3, :3] = refined_rotation
        refinement_transform[..., :3, 3] = translation_deltas
        return torch.matmul(cam_to_world, refinement_transform)
