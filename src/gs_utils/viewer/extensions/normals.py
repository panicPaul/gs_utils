"""Default normals viewer extension."""

from dataclasses import dataclass

import numpy as np

from gs_utils.contracts import RenderMode, RendersNormals
from gs_utils.contracts.render import RenderOutput


@dataclass
class NormalsViewerRenderState:
    """Render state for the normals viewer extension."""


class NormalsViewer:
    """Default viewer extension for normal-capable scenes."""

    def _viewer_supported_render_modes(self) -> set[RenderMode]:
        """Return normals render modes supported by the current scene."""
        if isinstance(self.scene, RendersNormals):
            return {RenderMode.NORMALS}
        return set()

    def _viewer_display_image(
        self,
        render_output: RenderOutput,
    ) -> np.ndarray | None:
        """Convert normal output into a viewer image when normals mode is active."""
        if self.render_tab_state.render_mode != RenderMode.NORMALS:
            return None
        if render_output.normals is None:
            raise ValueError("Normals render mode requires normal output.")

        normals_image = (
            render_output.normals[0]
            if render_output.normals.ndim == 4
            else render_output.normals
        )
        display_image = 0.5 * (normals_image + 1.0)
        return display_image.detach().cpu().numpy()
