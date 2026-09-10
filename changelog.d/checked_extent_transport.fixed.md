Computed reshape result claims and broadcast unit preconditions survive inlined
calls, aliases and discarded results. Explicit checked extents retain the required
witness independently of result metadata, so mismatches report a Domain failure
before shape-dependent allocation or element access.

Computed remainder targets retain result claims on Eval and C, including HIP host-C entry selection. Signed remainder preserves exact minimum-integer behavior and DivZero diagnostics.
