Move exact certificate arithmetic to `chelis-certificate-core` with unconditional `no_std` and `alloc`. Consumers import `poly`, `exact`, and `linalg` from that crate without compatibility re-exports. Clarabel, native BLAS, goal extraction, solver layouts, and proof admission remain in `chelis-prove`. Host admission rejects noncanonical block counts, matrix dimensions, and interval multipliers. The acceptance command checks target consumers, exact fixed-revision parity, native production certificates, and mandatory failure controls. See [#3989](https://github.com/Chelis-Lang/chelis/issues/3989).

Standalone acceptance executes both Python failure-control suites before its PASS marker.

The core inherits workspace lint rules. Runtime solver-isolation checks replace source-text setup checks.
