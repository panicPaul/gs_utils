"""Minimal base viewer with extension hooks."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import viser
from nerfview import CameraState, RenderTabState
from nerfview import Viewer as NerfviewViewer

from gs_utils.contracts import RenderInput, RenderMode, RenderOutput, Scene


@dataclass
class ViewerRenderState(RenderTabState):
    """Shared viewer render state."""

    render_mode: RenderMode = RenderMode.RGB


class Viewer(NerfviewViewer):
    """Minimal viewer base that composes extension hooks by inheritance."""

    render_state_type = ViewerRenderState

    def __init__(
        self,
        scene: Scene,
        output_dir: Path | None = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        mode: Literal["rendering", "training"] = "rendering",
    ) -> None:
        """Initialize the viewer server and bind it to a scene."""
        self.scene = scene.eval()
        self.server = viser.ViserServer(host=host, port=port)
        super().__init__(
            server=self.server,
            render_fn=self._render_viewer_frame,
            output_dir=output_dir,
            mode=mode,
        )
        self.server.gui.set_panel_label("gs_utils viewer")

    def launch(self) -> None:
        """Keep the viewer process alive until interrupted."""
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            return

    def _iter_viewer_hooks(self, hook_name: str) -> list[object]:
        """Return bound hook methods defined by the viewer class hierarchy."""
        hook_methods: list[object] = []
        for viewer_class in type(self).__mro__:
            hook_function = viewer_class.__dict__.get(hook_name)
            if hook_function is None:
                continue
            if viewer_class is Viewer:
                continue
            hook_methods.append(hook_function.__get__(self, type(self)))
        return hook_methods

    def supported_render_modes(self) -> list[RenderMode]:
        """Return the render modes available for the current scene and viewer."""
        supported_render_modes: set[RenderMode] = {RenderMode.RGB}
        for hook in self._iter_viewer_hooks("_viewer_supported_render_modes"):
            supported_render_modes |= hook()
        return [mode for mode in RenderMode if mode in supported_render_modes]

    def _init_rendering_tab(self) -> None:
        """Initialize the rendering tab state used by the viewer."""
        self.render_tab_state = self.render_state_type()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self) -> None:
        """Populate the shared rendering tab and extension-specific controls."""
        super()._populate_rendering_tab()

        supported_render_mode_values = tuple(
            mode.value for mode in self.supported_render_modes()
        )
        if (
            self.render_tab_state.render_mode.value
            not in supported_render_mode_values
        ):
            self.render_tab_state.render_mode = self.supported_render_modes()[0]

        with self._rendering_folder:
            self._render_mode_dropdown = self.server.gui.add_dropdown(
                "Render Mode",
                supported_render_mode_values,
                initial_value=self.render_tab_state.render_mode.value,
                hint="Rendered output shown in the viewer.",
            )

            @self._render_mode_dropdown.on_update
            def _(_: object) -> None:
                self.render_tab_state.render_mode = RenderMode(
                    self._render_mode_dropdown.value
                )
                self._sync_extension_controls()
                self.rerender(_)

            for hook in self._iter_viewer_hooks(
                "_viewer_populate_rendering_tab"
            ):
                hook()

        self._sync_extension_controls()

    def _sync_extension_controls(self) -> None:
        """Synchronize extension GUI state with the active render mode."""
        for hook in self._iter_viewer_hooks("_viewer_sync_control_states"):
            hook()

    def _render_viewer_frame(
        self,
        camera_state: CameraState,
        render_tab_state: ViewerRenderState,
    ) -> np.ndarray:
        """Render one viewer frame for the current camera and render state."""
        render_input = self._build_render_input(camera_state, render_tab_state)
        render_mode = render_tab_state.render_mode

        with torch.no_grad():
            render_output = self.scene.render(
                render_input=render_input,
                render_mode=render_mode,
            )

        for hook in self._iter_viewer_hooks("_viewer_display_image"):
            display_image = hook(render_output)
            if display_image is not None:
                return self._to_uint8_image(display_image)

        return self._to_uint8_image(self._default_display_image(render_output))

    def _build_render_input(
        self,
        camera_state: CameraState,
        render_tab_state: ViewerRenderState,
    ) -> RenderInput:
        """Build a typed render request from the current viewer camera state."""
        scene_device = self._scene_device()
        image_width = int(render_tab_state.viewer_width)
        image_height = int(render_tab_state.viewer_height)
        camera_to_world = (
            torch.from_numpy(camera_state.c2w)
            .float()
            .unsqueeze(0)
            .to(scene_device)
        )
        intrinsics_matrix = (
            torch.from_numpy(camera_state.get_K((image_width, image_height)))
            .float()
            .unsqueeze(0)
            .to(scene_device)
        )
        return RenderInput(
            cam_to_world=camera_to_world,
            width=image_width,
            height=image_height,
            intrinsics=intrinsics_matrix,
            render_mode=render_tab_state.render_mode,
        )

    def _scene_device(self) -> torch.device:
        """Return the device used by the current scene."""
        try:
            first_parameter = next(self.scene.parameters())
        except StopIteration:
            return torch.device("cpu")
        return first_parameter.device

    def _default_display_image(
        self,
        render_output: RenderOutput,
    ) -> np.ndarray | torch.Tensor:
        """Return the default RGB viewer image for a render output."""
        rendered_image = render_output.image
        if rendered_image.ndim == 4:
            rendered_image = rendered_image[0]
        return rendered_image.detach().cpu().numpy()

    def _to_uint8_image(
        self,
        display_image: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Convert a float image in [0, 1] to uint8 viewer output."""
        if isinstance(display_image, torch.Tensor):
            display_image = display_image.detach().cpu().numpy()
        display_image = np.clip(display_image, 0.0, 1.0)
        return (display_image * 255.0).astype(np.uint8)
