# gs_utils

Shared utilities and scaffolding for splatting-style research code.

This package is meant to hold the infrastructure that tends to be reused across
projects even when the underlying representation changes. That includes things
like contracts, data loading, initialization, viewers, evaluation, export, and
small reusable render-time components.

The main idea is to keep the shared boilerplate here while leaving room for
concrete example packages and backend-specific implementations to evolve
quickly.

## CLI

Current entrypoints include:

- `gs_utils view-ply --ply path/to/scene.ply`
  Launch the generic viewer for a gsplat-compatible PLY file.
- `gs_utils evaluate --path path/to/run_or_experiment`
  Run offline checkpoint evaluation on a run directory or grouped experiment
  directory.
- `gs_utils vanilla_3dgs train ...`
  Train the reference vanilla 3DGS example.
- `gs_utils vanilla_3dgs visualize --checkpoint path/to/ckpt.pt`
  Launch the vanilla example viewer from a saved checkpoint.
- `gs_utils vanilla_3dgs export-ply --checkpoint path/to/ckpt.pt`
  Export a saved checkpoint scene to gsplat-compatible PLY.
