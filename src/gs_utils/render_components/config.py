"""Pydantic config models for learnable render components."""

from pydantic import BaseModel, ConfigDict


class _ConfigModel(BaseModel):
    """Base config model with strict field validation."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class ViewDependentColorMLPConfig(_ConfigModel):
    """Configuration for the view-dependent color prediction component."""

    embed_dim: int = 16
    max_sh_degree: int = 3
    mlp_width: int = 64
    mlp_depth: int = 2


class PPISPComponentConfig(_ConfigModel):
    """Configuration for the PPISP post-processing component."""

    use_controller: bool = True
    controller_distillation: bool = True
    controller_activation_ratio: float = 1.0
