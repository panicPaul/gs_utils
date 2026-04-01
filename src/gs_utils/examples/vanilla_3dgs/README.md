# vanilla_3dgs

Reference 3D Gaussian Splatting example.

This package contains the current first-pass vanilla 3DGS scene, training
loop, viewer, config, and CLI wiring built on top of the shared `gs_utils`
infrastructure.

It is both a usable example and the current integration target for shared
subsystems such as initialization, evaluation, export, and viewer support.

## CLI

- `gs_utils vanilla_3dgs train ...`
  Train the example scene from a parsed data source.
- `gs_utils vanilla_3dgs visualize --checkpoint path/to/ckpt.pt`
  Open the example viewer from a saved checkpoint.
- `gs_utils vanilla_3dgs export-ply --checkpoint path/to/ckpt.pt`
  Export a saved checkpoint to gsplat-compatible PLY.
