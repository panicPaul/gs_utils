# utils

Small shared implementation utilities.

This package contains low-level helpers that are reused across `gs_utils`,
such as optimizer remapping, rotations, visualization, randomness, and similar
support code.

The boundary here should stay narrow: utilities belong here when they are
reusable implementation details, not when they define scene policy or package
architecture.
