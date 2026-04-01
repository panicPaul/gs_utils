# contracts

Shared capability and interface contracts.

This package defines the stable typed surfaces that other `gs_utils` code works
against, such as scene contracts, render I/O, and geometry/render
capabilities.

The goal here is to keep the shared interfaces more stable than the concrete
implementations. Different backends can change geometry, rendering, or training
internals, but they should still plug into the same contracts where that makes
sense.
