# Compiled-context cache package-identity collision

## Severity

HIGH release blocker. Found independently by two red-team passes on
`main` (HEAD `7b3d5de`, regressed by PR #127 `8fb56a4`).

## Symptom

A cold `cargo nextest run --workspace` passes. Every subsequent (warm)
run fails ~7-20 `phase3t_*` integration tests. CI stays green only
because fresh CI runners always have a cold cache.

Deterministic reproduction:

```
rm -f ~/.cache/chelis/compiled/*.ctx
cargo nextest run --workspace -E 'binary(/phase3t_/)' --test-threads=1   # cold: pass
cargo nextest run --workspace -E 'binary(/phase3t_/)' --test-threads=1   # warm: fail
```

On the `chelis test` path the worker refuses with an error containing
`does not match worker cwd` (its `package_root` guard firing). On the
`chelis eval --file` / `run_eval_in_context` path there is no such
guard, so the same collision is a silent wrong-result risk.

## Root cause

PR #127 added an XDG cache-dir fallback for the cross-process
typecheck cache. As a side effect, `load_or_compile_for_package` in
`crates/chelis-compiler-api/src/context.rs` no longer bypasses the
Phase K `CompiledContext` disk cache when `CHELIS_REEF_HOME` is unset;
it now routes through `~/.cache/chelis/compiled/`.

The Phase K cache file name keyed only on
`(package_name, package_version, source_hash)`, and `load_if_fresh`
re-verified only `source_hash`. `source_hash` is a pure *content*
check: two distinct on-disk packages with identical name+version and
byte-identical source files hash identically. They therefore collided
on one cache file. The second package silently loaded the first's
`CompiledContext`, including its `package_root`.

The `phase3t_*` tests build packages in fresh temp dirs but reuse
deterministic package names (`phase3t-iso-abort`, etc.) and identical
source bytes, so every warm run reads a stale entry pointing at a
since-deleted temp dir.

A related MEDIUM finding shares the bug class: neither the Phase K
`ContextHash` nor the `stdlib_cache_key` folded in compiler build
identity. A chelis binary built from different compiler source but the
same bundled chelis-std produced the same key, so a newer binary could
read an older binary's cached context and apply stale semantics.
`STDLIB_CACHE_FORMAT_VERSION` only guards the on-disk struct shape, not
semantics.

## Fix

Add package + build identity to the Phase K cache. `CacheIdentity`
holds the canonicalized `package_root` plus `COMPILER_VERSION`.

- It is stored on `CompiledContext` and in the on-disk `CacheEnvelope`.
- `cache_file_name` / `cache_path_for` fold an identity fingerprint
  into the file name, so two checkouts that share name+version+source
  land on separate cache files.
- `load_if_fresh` recomputes the identity from the live package and the
  running binary; a mismatch is a clean miss (`Ok(None)`), never a
  stale hit. A belt-and-braces inner-vs-envelope identity check rejects
  tampered bytes as `CacheError::IdentityMismatch`.
- The cache format version and magic bumped from `3` to `4`, so
  pre-existing `V3` entries are rejected as `UnsupportedVersion` and
  recompiled.

`stdlib_cache_key` folds `COMPILER_VERSION` in directly.

The fix does not re-gate the cache behind `CHELIS_REEF_HOME` (that
would discard the test-worker speedup PR #127 added) and does not make
the `phase3t_*` tests set an isolated `CHELIS_REEF_HOME` (that would
leave the `chelis eval --file` silent-result hole open). The
`run_eval_in_context` path is correct because `load_or_compile_for_package`
itself is correct.

## Regression coverage

- `phase_i_disk_cache.rs::two_packages_same_name_version_source_but_different_root_do_not_collide`
- `phase_i_disk_cache.rs::cache_entry_from_a_different_compiler_build_is_a_clean_miss`
- `stdlib_cache.rs::tests::cache_key_depends_on_the_compiler_version`

All three are on the per-PR `ci` profile (cache key / identity logic
only, no real build).
