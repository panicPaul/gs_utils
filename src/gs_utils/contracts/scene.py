"""Base scene abstraction."""

from abc import ABC, abstractmethod

import torch.nn as nn

from gs_utils.contracts.capabilities import RendersRGB
from gs_utils.contracts.render import RenderInput, RenderOutput


class Scene(nn.Module, RendersRGB, ABC):
    """Base scene contract for shared GS utilities."""

    @abstractmethod
    def render(self, render_input: RenderInput) -> RenderOutput:
        """Render the scene for the provided camera state."""

    def forward(self, render_input: RenderInput) -> RenderOutput:
        """Alias for render."""
        return self.render(render_input)
