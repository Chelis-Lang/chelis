Computed reshape result claims and broadcast unit preconditions survive inlined
calls, aliases and discarded results. Explicit checked extents retain the required
witness independently of result metadata, so mismatches report a Domain failure
before shape-dependent allocation or element access.
