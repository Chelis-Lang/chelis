# Reef package-cache concurrent-write race

## Symptom

`chelis-cli::phase_a_item8_autofetch_build::phaseA_item8_two_concurrent_builds_serialize`
fails intermittently in CI:

```
error: failed to parse <reef-home>/cache/<hash>/reef.toml: TOML parse error at line 1, column 1
thread 'phaseA_item8_two_concurrent_builds_serialize' panicked at
crates/chelis-cli/tests/phase_a_item8_autofetch_build.rs:782: build A panicked: Any { .. }
```

The test passes on rerun and passes on `main` with the same code, so it
is a genuine intermittent race, not a hard failure.

## Root cause

`load_registry_package` in `crates/chelis-reef/src/lib.rs` populated the
shared package cache directory `$CHELIS_REEF_HOME/cache/<archive_sha256>/`
non-atomically:

```rust
let cache_root = registry_root.join("cache").join(&archive_sha256);
if !cache_root.exists() {
    fs::create_dir_all(&cache_root)?;
    extract_archive(&archive_path, &cache_root)?; // unpacks files one by one, in place
}
```

`extract_archive` calls `tar::Archive::unpack`, which writes each entry
(`reef.toml`, `src/main.ch`, ...) into `cache_root` one file at a time.
The `if !cache_root.exists()` guard goes false the instant
`create_dir_all` runs, well before `unpack` has written `reef.toml`.

The reef-home `flock` does not cover this path. In the concurrent-build
test:

1. Build A wins the lock, runs auto-fetch + install (updates
   `index.json`), releases the lock, then re-calls
   `load_registry_package` which extracts into `cache/<hash>/`.
2. Build B lost the lock race; after acquiring it, its double-checked
   `load_registry_package` now finds the package in `index.json` and
   returns via the same `load_registry_package` path, also extracting
   into the same `cache/<hash>/`.

Both builds extract into the same directory concurrently (or one reads
while the other is mid-extract). A reader doing
`read_manifest(cache_root.join("reef.toml"))` observes an empty or
truncated `reef.toml`, hence the `TOML parse error at line 1, column 1`.

## Pre-existing vs introduced

Pre-existing. The non-atomic extract pattern has existed since the
original Item 8 auto-fetch feature (`crates/chelis-reef/src/lib.rs`,
commits `7295dbf` / `56e0fda`, merged in `3061dd3`). The recent
test-infra workstream added an XDG cache-dir fallback and the
`CompiledContext` identity work, which touch a *different* cache (the
compiled-context typecheck cache), not the reef package cache under
`$CHELIS_REEF_HOME/cache/`. `git log -L` on the extract lines confirms
they were untouched by that workstream.

## Fix

New `extract_archive_atomic` helper: extract into a unique sibling
`.extract-<pid>-<seq>.tmp` staging directory, then `fs::rename` it onto
`cache/<hash>/`. Same-directory rename is atomic on every supported FS,
so a concurrent reader sees either no `cache/<hash>/` directory or the
fully populated one, never an intermediate state. A lost rename race
(another process published `cache/<hash>/` first) is treated as success
because the cache key is the archive's own content hash, so the two
extracted trees are byte-identical; the loser discards its staging dir.

## Reproduction

The race window between `create_dir_all` and `unpack` finishing is
sub-millisecond, so the flake does not reproduce reliably on a fast
workstation even under CPU stress (the test's two builds are also
partly serialized by the reef-home flock). The CI evidence plus the
unambiguous code path are the basis for the fix. A unit-level
reproduction and guard lives in
`extract_archive_atomic_concurrent_writers_never_tear_reef_toml`: 16
threads extract and immediately read back `reef.toml` from one shared
cache directory; without the atomic rename a reader observes a torn
file.

## Verification

- `cargo test -p chelis-reef --lib extract_archive_atomic` — 2 passed.
- `phaseA_item8_two_concurrent_builds_serialize` run 12x clean, plus
  20x clean under `stress-ng --cpu 0` together with
  `phaseA_item8_concurrent_one_no_auto_fetch_does_not_deadlock`.
- Full gate: `cargo build --workspace --all-targets`,
  `cargo nextest run --workspace` (3102 passed),
  `cargo test --workspace --lib`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo fmt --all -- --check`,
  `chelis lint --check .` (exit 0) all green.
