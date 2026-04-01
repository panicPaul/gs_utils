# evaluation

Offline checkpoint evaluation utilities.

This package evaluates saved run directories, writes per-image and aggregated
metrics, and provides the basis for future evaluation tasks such as mesh-based
scoring.

The current focus is checkpoint-based image evaluation. Over time this package
should also become the home for broader experiment aggregation and future
evaluation families such as mesh extraction and mesh scoring.

## CLI

- `gs_utils evaluate --path path/to/scene_dir`
  Evaluate a single saved scene directory.
- `gs_utils evaluate --path path/to/experiment_root`
  Evaluate a grouped experiment laid out by dataset and scene.

The current evaluator writes one JSON report per scene with both per-image and
aggregated stats, and can also write rendered test images when requested.
