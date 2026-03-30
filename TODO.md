# TODO

## Phase 1

- Implement concrete `3dgs` and `2dgs` scene modules on top of the current contracts.
- Add optimizer construction.
- Keep parameter ownership on the scene.
- Build a modular training loop that uses the shared data and initialization pipeline.
- Add the densification / refinement strategy layer needed for actual 3DGS-style training.
- Implement a base viewer host with clean extension points.
- Implement an evaluation module:
  - dataset iteration
  - rendering
  - metrics
  - checkpoint-based evaluation entrypoints
- Implement checkpoint save/load as a first-class module.
- Checkpoints should store scene state, not optimizer state.
- Implement export modules for point-based formats such as `.ply` and PLAS.

## Phase 2

- Add a dedicated video creation module.
- Add mesh extraction.
- Add richer viewer extensions for specific geometry / render capability families.
- Add higher-level example applications once the core modules are stable enough.
