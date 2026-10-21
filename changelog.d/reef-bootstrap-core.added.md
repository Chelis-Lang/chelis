Add dependency-free `chelis-reef-bootstrap-core` with `no_std` and `alloc` for checked bootstrap graph order. Reef retains typed version requirements, canonical release checks, diagnostics, and original release selection. Manifest access, default discovery, artifact verification, and installation remain host operations. The acceptance command checks isolated target consumers, exact graph behavior, and real host bootstrap fixtures. Target compilation does not prove device execution or allocator capacity. See [#3984](https://github.com/Chelis-Lang/chelis/issues/3984).

The standalone pipeline rejection fixture retains a locked graph that includes the new core.
Its compiler checks retain all original diagnostics.
