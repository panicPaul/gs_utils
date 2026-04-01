# data

Typed data loading and dataset construction.

This package turns concrete data sources such as COLMAP outputs into parsed
scene metadata, datasets, dataloaders, and typed samples used by training,
evaluation, and viewers.

It is meant to absorb the repetitive data plumbing around scene parsing,
splitting, batching, and typed sample creation, while keeping the concrete data
source adapters local to this package.
