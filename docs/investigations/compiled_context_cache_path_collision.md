# Compiled-Context Disk Cache: Path-Collision Bug

Red-team finding against PR #127 (cross-process chelis-std typecheck
cache / compiled-context disk cache). This note records the bug, the
reproduction, and the owning manual gate for the `#[ignore]`-d repro
tests.

## Symptom

`cargo nextest run --workspace` is non-deterministic on a developer
machine: the `crates/chelis-cli/tests/phase3t_*` integration tests
(`phase3t_compiled_context_bridge`, `phase3t_subprocess_isolation`,
`phase3t_test_smoke`) pass on a cold `~/.cache/chelis/compiled/` and
fail once that cache has been warmed by an earlier run. The failure is:

```
FAIL (worker exited 2: error: CHELIS_TEST_COMPILED_CONTEXT package_root
`/tmp/.tmpAAA/<pkg>` does not match worker cwd `/tmp/.tmpBBB/<pkg>`;
refusing to run tests with a mismatched library context)
```

It also fails with `--test-threads=1`, so it is not a parallelism
flake -- it is a deterministic, order-dependent cache poisoning.

## Root cause

`chelis_compiler_api::load_or_compile_for_package`
(`crates/chelis-compiler-api/src/context.rs`) keys the compiled-context
disk cache on:

```
(root_package_name, root_package_version, source_hash)
```

where `source_hash` is `ContextHash::from_digests(...)` over the
*contents* of every source file in the package graph. The package's
location on disk is not part of the key.

`CompiledContext::load_if_fresh` then validates the envelope version,
the recomputed `source_hash`, and the payload SHA-256 -- but it does
**not** check that the decoded `ctx.reef_state().package_root` matches
the live `package_dir` it was asked to load for, and it does not
re-bind that field.

So two reef packages with the same `name`/`version` in `reef.toml` and
byte-identical sources, built from two different directories, share one
cache entry. The second build gets a cache HIT whose
`CompiledContext.reef_state().package_root` still points at the first
package's directory.

This collides with the Phase H `chelis test` worker, which has a
correct path-equality guard (`crates/chelis-cli/src/main.rs`, the
`does not match worker cwd` error): the worker refuses to run tests
against a `CompiledContext` whose `package_root` does not match its
cwd. The guard is doing its job; the cache handed it a stale context.

The `phase3t_*` test helpers (`make_minimal_reef_with_test` and
friends) write byte-identical sources with a fixed package name every
run, into a fresh `tempdir()` -- the exact shape that triggers the
collision.

## Reproduction

Hermetic, deterministic repro (does not touch the real user cache):

```sh
cargo test -p chelis-cli \
  --test redteam_compiled_context_cache_collision -- --ignored
```

Each test points `XDG_CACHE_HOME` at its own private tempdir, builds +
`chelis test`s a content-identical package twice from two different
directories, and asserts the second run still passes. Both tests
currently FAIL, reproducing the bug.

Manual repro against the real cache:

```sh
rm -rf ~/.cache/chelis/compiled
cargo nextest run -p chelis-cli --test phase3t_compiled_context_bridge  # passes (cold)
cargo nextest run -p chelis-cli --test phase3t_compiled_context_bridge  # fails (warm)
```

## Fix direction

Either is sufficient; the cache-key option is the more robust:

1. Include the canonicalized package directory in the cache key
   (`cache_file_name` / the `source_hash` input), so content-identical
   packages at different paths get distinct entries; or
2. In `CompiledContext::load_if_fresh`, after decode, compare the
   decoded `reef_state().package_root` against the live `package_dir`
   (canonicalized) and treat a mismatch as a clean cache miss
   (`Ok(None)`), the same way a `source_hash` mismatch is handled.

## Owning manual gate

Until the fix lands, the two repro tests in
`crates/chelis-cli/tests/redteam_compiled_context_cache_collision.rs`
are `#[ignore]`-d. Manual gate command:

```sh
cargo test -p chelis-cli \
  --test redteam_compiled_context_cache_collision -- --ignored
```

Expected success condition once fixed: both tests pass. When they do,
remove the `#[ignore]` attributes so the invariant stays locked on the
per-PR gate.
