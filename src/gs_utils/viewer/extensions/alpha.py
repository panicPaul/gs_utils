"""Default alpha viewer extension."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from gs_utils.contracts import RenderMode, RendersAlpha
from gs_utils.contracts.render import RenderOutput
from gs_utils.utils.visualization import apply_float_colormap


@dataclass
class AlphaViewerRenderState:
    """Render state for the alpha viewer extension."""

    alpha_colormap: Literal[
        "turbo",
        "viridis",
        "magma",
        "inferno",
        "cividis",
        "gray",
    ] = "turbo"


class AlphaViewer:
    """Default viewer extension for alpha-capable scenes."""

    def _viewer_supported_render_modes(self) -> set[RenderMode]:
        """Return alpha render modes supported by the current scene."""
        if isinstance(self.scene, RendersAlpha):
            return {RenderMode.ALPHA}
        return set()

    def _viewer_populate_rendering_tab(self) -> None:
        """Add alpha-specific GUI controls."""
        if not isinstance(self.scene, RendersAlpha):
            return

        self._alpha_colormap_dropdown = self.server.gui.add_dropdown(
            "Alpha Colormap",
            ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
            initial_value=self.render_tab_state.alpha_colormap,
            disabled=True,
            hint="Colormap used for alpha visualization.",
        )

        @self._alpha_colormap_dropdown.on_update
        def _(_: object) -> None:
            self.render_tab_state.alpha_colormap = (
                self._alpha_colormap_dropdown.value
            )
            self.rerender(_)

    def _viewer_sync_control_states(self) -> None:
        """Enable or disable alpha controls for the active render mode."""
        if not hasattr(self, "_alpha_colormap_dropdown"):
            return

        is_alpha_mode = self.render_tab_state.render_mode == RenderMode.ALPHA
        self._alpha_colormap_dropdown.disabled = not is_alpha_mode

    def _viewer_display_image(
        self,
        render_output: RenderOutput,
    ) -> np.ndarray | None:
        """Convert alpha output into a viewer image when alpha mode is active."""
        if self.render_tab_state.render_mode != RenderMode.ALPHA:
            return None
        if render_output.alpha is None:
            raise ValueError("Alpha render mode requires alpha output.")

        alpha_image = (
            render_output.alpha[0]
            if render_output.alpha.ndim == 4
            else render_output.alpha
        )
        display_image = apply_float_colormap(
            alpha_image,
            self.render_tab_state.alpha_colormap,
        )
        return display_image.detach().cpu().numpy()
