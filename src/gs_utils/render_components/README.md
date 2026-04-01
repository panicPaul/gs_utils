# render_components

Reusable render-time components.

This package contains scene-attached `nn.Module` building blocks such as
view-dependent color prediction, camera refinement, and PPISP integration.

These components are meant to stay smaller than full scenes or trainers. They
are the reusable learnable pieces that participate in rendering without owning
the whole pipeline.
