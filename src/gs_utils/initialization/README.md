# initialization

Scene initialization utilities.

This package contains shared initialization dispatch and strategy logic for
constructing scenes from point clouds, checkpoints, and other supported
sources.

It is intentionally separate from concrete scene classes so initialization
methods can be shared across compatible geometry contracts without forcing one
scene family or backend structure.
