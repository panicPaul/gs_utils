"""View-dependent color prediction component."""

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from gs_utils.render_components.config import ViewDependentColorMLPConfig


class ViewDependentColorMLP(torch.nn.Module):
    """Predict per-view color residuals from features, embeddings, and SH bases."""

    def __init__(
        self,
        num_embeddings: int,
        feature_dim: int,
        config: ViewDependentColorMLPConfig | None = None,
    ) -> None:
        """Initialize appearance embeddings and the view-dependent color MLP."""
        super().__init__()
        self.config = config or ViewDependentColorMLPConfig()
        self.feature_dim = feature_dim
        self.embeds = torch.nn.Embedding(num_embeddings, self.config.embed_dim)

        input_dimension = (
            self.config.embed_dim
            + feature_dim
            + (self.config.max_sh_degree + 1) ** 2
        )
        mlp_layers: list[torch.nn.Module] = [
            torch.nn.Linear(input_dimension, self.config.mlp_width),
            torch.nn.ReLU(inplace=True),
        ]
        for _ in range(self.config.mlp_depth - 1):
            mlp_layers.append(
                torch.nn.Linear(
                    self.config.mlp_width,
                    self.config.mlp_width,
                )
            )
            mlp_layers.append(torch.nn.ReLU(inplace=True))
        mlp_layers.append(torch.nn.Linear(self.config.mlp_width, 3))
        self.color_head = torch.nn.Sequential(*mlp_layers)
        self.zero_init_output_head()

    def zero_init_output_head(self) -> None:
        """Initialize the final color-head layer to zero."""
        last = self.color_head[-1]
        if not isinstance(last, torch.nn.Linear):
            raise TypeError("Expected final color_head layer to be Linear.")
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    def forward(
        self,
        features: Float[torch.Tensor, "num_splats feature_dim"],
        embed_ids: Int[torch.Tensor, "num_cameras"] | None,
        dirs: Float[torch.Tensor, "num_cameras num_splats 3"],
        sh_degree: int,
    ) -> Float[torch.Tensor, "num_cameras num_splats 3"]:
        """Predict view-dependent color residuals for each camera and splat."""
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        if sh_degree > self.config.max_sh_degree:
            raise ValueError(
                f"Requested sh_degree={sh_degree} exceeds "
                f"max_sh_degree={self.config.max_sh_degree}."
            )

        num_cameras, num_splats = dirs.shape[:2]
        if features.shape != (num_splats, self.feature_dim):
            raise ValueError("features must match the configured feature_dim.")
        if embed_ids is not None and embed_ids.shape != (num_cameras,):
            raise ValueError("embed_ids must have shape (num_cameras,).")

        if embed_ids is None:
            camera_embeddings = torch.zeros(
                num_cameras,
                self.config.embed_dim,
                device=features.device,
                dtype=features.dtype,
            )
        else:
            camera_embeddings = self.embeds(embed_ids)
        camera_embeddings = camera_embeddings[:, None, :].expand(
            -1,
            num_splats,
            -1,
        )

        expanded_features = features[None, :, :].expand(num_cameras, -1, -1)
        normalized_directions = F.normalize(dirs, dim=-1)

        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.config.max_sh_degree + 1) ** 2
        spherical_harmonic_bases = torch.zeros(
            num_cameras,
            num_splats,
            num_bases,
            device=features.device,
            dtype=features.dtype,
        )
        spherical_harmonic_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(
            num_bases_to_use,
            normalized_directions,
        )

        if self.config.embed_dim > 0:
            color_head_inputs = torch.cat(
                [
                    camera_embeddings,
                    expanded_features,
                    spherical_harmonic_bases,
                ],
                dim=-1,
            )
        else:
            color_head_inputs = torch.cat(
                [expanded_features, spherical_harmonic_bases],
                dim=-1,
            )
        return self.color_head(color_head_inputs)
