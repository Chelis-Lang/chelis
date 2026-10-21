Use `chelis-shell-core` for CHB 8 metadata and complete validation with unconditional `no_std` and `alloc`. Metadata imports move directly from `chelis_shell` to `chelis_shell_core`. Validation now returns typed errors and needs explicit `StackServices`. `chelis-shell` retains codecs and filesystem adapters. Reef retains archive and requested-package identity checks.

CHB fields, bytes, diagnostics, first-error order, and embedded bundle digests remain unchanged. The acceptance command checks isolated target consumers, immutable-baseline parity, and real host artifact rejection. Linux integration support and macOS nightly validation execute the complete command. Target compilation does not prove device execution or allocator capacity. See [#3986](https://github.com/Chelis-Lang/chelis/issues/3986).

Matching incomplete metadata snapshots fail acceptance.
