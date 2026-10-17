Each Nix crate build gets only its own directory and the paths that its compile
reads, without `tests/`, `benches/`, and `examples/`. An edit to `chelis-cli`
now changes 1 crate derivation, not 8, and a test-only edit changes none. The
Nix package toolchain is the minimal profile of the pinned Rust version, so a
builder no longer downloads rust-docs, rustfmt, clippy, or rustc-dev. See
[#3810](https://github.com/Chelis-Lang/chelis/issues/3810).
