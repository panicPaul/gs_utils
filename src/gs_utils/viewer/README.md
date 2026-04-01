# viewer

Viewer base classes and shared extensions.

This package contains the reusable viewer host, default viewer extensions, and
generic viewers such as the gsplat-compatible PLY viewer.

The base viewer is meant to stay thin. Concrete viewers are expected to compose
the shared extensions they need and then add package- or example-specific
viewer behavior on top.

## CLI

- `gs_utils view-ply --ply path/to/scene.ply`
  Launch the generic viewer for a gsplat-compatible PLY file.

Example-specific viewers, such as the vanilla 3DGS viewer, are exposed through
their own example packages rather than this shared package directly.
