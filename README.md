# gs_utils

`gs_utils` is a shared substrate for splatting-style research code.

The point is to move as much recurring boilerplate as possible out of individual
projects and into a reusable package.

This includes things like:

- render and camera contracts
- data loading
- initialization from SfM / point clouds / checkpoints
- viewer plumbing
- shared pre/post processing models such as PPISP or bilateral grids
- preprocessing / postprocessing hooks
- small shared utilities

The main reason for this package is that backend-specific assumptions change
often. Classical 3DGS, beta splatting, or voxel-based methods do not share the
same geometry model, but they still share a lot of surrounding infrastructure.

`gs_utils` is meant to keep that shared infrastructure in one place without
forcing everything into one representation.

The current focus is:

- shared contracts
- typed data flow
- initialization utilities
- extension points for downstream code
