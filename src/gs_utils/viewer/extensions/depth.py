"""Default depth viewer extension."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from gs_utils.contracts import RenderMode, RendersDepth
from gs_utils.contracts.render import RenderOutput
from gs_utils.utils.visualization import apply_float_colormap


@dataclass
class DepthViewerRenderState:
    """Render state for the depth viewer extension."""

    normalize_with_near_far: bool = True
    invert: bool = False
    depth_colormap: Literal[
        "turbo",
        "viridis",
        "magma",
        "inferno",
        "cividis",
        "gray",
    ] = "turbo"


class DepthViewer:
    """Default viewer extension for depth-capable scenes."""

    def _viewer_supported_render_modes(self) -> set[RenderMode]:
        """Return depth render modes supported by the current scene."""
        if isinstance(self.scene, RendersDepth):
            return {RenderMode.DEPTH}
        return set()

    def _viewer_populate_rendering_tab(self) -> None:
        """Add depth-specific GUI controls."""
        if not isinstance(self.scene, RendersDepth):
            return

        self._depth_normalize_checkbox = self.server.gui.add_checkbox(
            "Normalize Near/Far",
            initial_value=self.render_tab_state.normalize_with_near_far,
            disabled=True,
            hint="Normalize depth with the current near and far range.",
        )

        @self._depth_normalize_checkbox.on_update
        def _(_: object) -> None:
            self.render_tab_state.normalize_with_near_far = (
                self._depth_normalize_checkbox.value
            )
            self.rerender(_)

        self._depth_invert_checkbox = self.server.gui.add_checkbox(
            "Invert Depth",
            initial_value=self.render_tab_state.invert,
            disabled=True,
            hint="Invert depth colors after normalization.",
        )

        @self._depth_invert_checkbox.on_update
        def _(_: object) -> None:
            self.render_tab_state.invert = self._depth_invert_checkbox.value
            self.rerender(_)

        self._depth_colormap_dropdown = self.server.gui.add_dropdown(
            "Depth Colormap",
            ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
            initial_value=self.render_tab_state.depth_colormap,
            disabled=True,
            hint="Colormap used for depth visualization.",
        )

        @self._depth_colormap_dropdown.on_update
        def _(_: object) -> None:
            self.render_tab_state.depth_colormap = (
                self._depth_colormap_dropdown.value
            )
            self.rerender(_)

    def _viewer_sync_control_states(self) -> None:
        """Enable or disable depth controls for the active render mode."""
        if not hasattr(self, "_depth_normalize_checkbox"):
            return

        is_depth_mode = self.render_tab_state.render_mode == RenderMode.DEPTH
        self._depth_normalize_checkbox.disabled = not is_depth_mode
        self._depth_invert_checkbox.disabled = not is_depth_mode
        self._depth_colormap_dropdown.disabled = not is_depth_mode

    def _viewer_display_image(
        self,
        render_output: RenderOutput,
    ) -> np.ndarray | None:
        """Convert depth output into a viewer image when depth mode is active."""
        if self.render_tab_state.render_mode != RenderMode.DEPTH:
            return None
        if render_output.depth is None:
            raise ValueError("Depth render mode requires depth output.")

        depth_image = (
            render_output.depth[0]
            if render_output.depth.ndim == 4
            else render_output.depth
        )

        # Normalize the scalar depth image before applying the configured colormap.
        if self.render_tab_state.normalize_with_near_far:
            normalized_depth = depth_image - torch.min(depth_image)
            normalized_depth = normalized_depth / (
                torch.max(normalized_depth) + 1e-10
            )
        else:
            normalized_depth = depth_image

        display_image = apply_float_colormap(
            normalized_depth,
            self.render_tab_state.depth_colormap,
        )
        if self.render_tab_state.invert:
            display_image = 1.0 - display_image
        return display_image.detach().cpu().numpy()
