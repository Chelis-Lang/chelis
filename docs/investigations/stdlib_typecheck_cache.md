# chelis-std Typecheck Cache Diagnosis and Design

Investigation of the per-invocation cost of re-parsing and re-typechecking the
entire chelis-std import graph on every `chelis check` / `chelis build`, and the
design of the cross-process content-addressed cache that removes it.

## Symptom

Every `chelis check` and `chelis build` on a file that imports chelis-std
re-parses and re-typechecks the *entire* chelis-std transitive import graph from
scratch. Measured on this workstation (`chelis check
packages/chelis-std/src/nn/linear.ch`, debug binary):

| Run | Wall clock |
|---|---|
| 1 | 4.60s |
| 2 | 4.53s |
| 3 | 4.53s |

Fully repeatable: there is zero cross-process or on-disk typecheck cache for the
`check` / `build` path today. `cargo nextest` spawns ~200 separate test
processes, a large fraction of which `chelis check` or `chelis build` a
stdlib-importing file and therefore each pay this cost independently.

## Root cause

Tracing the `check` entry point:

- `cmd_check_one` (`crates/chelis-cli/src/main.rs:1191`) calls
  `load_check_build_decls` -> `chelis_reef::prepare_program_for_file`, which
  resolves the package graph, links every library module (chelis-std + deps +
  the user package's own modules) and the entry module into one merged `Decl`
  list, then desugars + macro-expands it to one flat `Vec<deep::Expr>`.
- `cmd_check_one` then runs `chelis_types::check_ir_fitness(&deep_exprs)` and
  `chelis_types::check_typed_program(&deep_exprs)` over that *whole merged
  program*. Both are full Hindley-Milner inference passes over the entire
  chelis-std graph plus the entry file.
- `cmd_build` (`main.rs:1506`) similarly runs `checked_program_with_effects`
  (which calls `check_ir_program`, i.e. `check_ir_with_context` against an
  *empty* context) over the whole merged program, then lowers it monolithically.

`link_graph` (`crates/chelis-reef/src/lib.rs:4438`) does **not** typecheck — it
only rewrites Surf `Decl`s with internal names. There is no typecheck seam at
`link_graph`; the typecheck happens later in the CLI over the merged program.

## Why a chelis-std sub-context cache is correct

The chelis-std bundle ships inside the chelis binary via `include_bytes!`
(`crates/chelis-std-bundle/src/lib.rs`): `CHELIS_STD_ARCHIVE`, `CHELIS_STD_SHELL`,
and `BUNDLED_CHELIS_STD_VERSION` are compile-time constants, immutable for a
given binary.

Crucially, `link_graph` iterates `graph.packages` (a `BTreeMap`) and rewrites
internal names with `internal_name(package_name, module_name, symbol)` — a pure
function of chelis-std's *own* package/module/symbol names, with **zero**
dependence on the user package. chelis-std imports nothing external. Therefore
the linked + internal-name-rewritten chelis-std `Decl`s, their
desugared/macro-expanded Deep form, and everything derived from them (the
typechecked `TypeEnv`, the library `CheckedProgram`, the `LoweredLibrary`) are
**bit-identical for every fixture that depends on the bundled stdlib**.

That property is the foundation of the cache: the chelis-std sub-context can be
computed once, content-addressed, and reused across every process and every
fixture.

## Chosen design

A *hybrid* layered cache. Two on-disk artifacts, one envelope format, one
write path.

### Cache key

The chelis-std sub-context is keyed, content-addressed, on:

```
chelis_std_typecheck_v{N} || BUNDLED_CHELIS_STD_VERSION || archive_sha256() || shell_sha256()
```

`archive_sha256()` / `shell_sha256()` (`chelis-std-bundle/src/lib.rs:75,84`) hash
the `include_bytes!`'d bundle. The key self-invalidates on any stdlib
regeneration; no manual bust. `v{N}` is an internal struct-format-version prefix
so a shape change forces a clean miss rather than a bad decode.

### Cache location

`$CHELIS_REEF_HOME/.cache/typecheck/` when `CHELIS_REEF_HOME` is set; otherwise
the XDG fallback `$XDG_CACHE_HOME/chelis/typecheck/` ->
`~/.cache/chelis/typecheck/`. The XDG fallback is mandatory: `cargo nextest` test
workers do not set `CHELIS_REEF_HOME`, and they are exactly the workload being
optimized. A cache that only works with `CHELIS_REEF_HOME` set is a dead cache
for CI.

### Layering

The cache is built in three layers:

1. **Layer 1 (cached, chelis-std sub-key):** the chelis-std linked decls,
   desugared + macro-expanded, run through `build_compiled_library_context` +
   effects + linearity + `lower_program_to_library`. Cached as a `StdLibContext`
   keyed on the content-addressed sub-key above. A hit skips all of this.
2. **Layer 2 (per-package, whole-graph key):** the *non-chelis-std* library
   decls (the user package's own modules + path-deps) checked `_with_context`
   against Layer 1. This produces the existing `CompiledContext`, whose builder
   is re-implemented to layer on `StdLibContext` instead of re-checking the whole
   graph monolithically.
3. **Layer 3 (entry):** the user's entry file checked `_with_context` against
   Layer 2.

For a fixture corpus that all imports the same bundled stdlib, Layer 1 is one
shared cross-process hit; Layer 2 is small (just the user package); Layer 3 is
the entry file. That is the ~200-worker win.

### Concurrency

~200 nextest processes race on the first miss. Writes are atomic: a temp file in
the same directory plus `fs::rename` (POSIX rename is atomic within a
filesystem). A torn or corrupt read falls through to recompute; it never panics.
No lockfile. This reuses `CompiledContext`'s existing `save()` / `load_if_fresh()`
machinery (magic header, version envelope, payload SHA-256, torn-write
rejection).

### check_ir_fitness

`cmd_check_one` runs a *second* whole-program inference pass,
`check_ir_fitness`, for the JSON `score` / `typed_nodes` / `total_nodes` /
`components`. Two coupled changes make the `chelis check` oracle actually speed
up while staying byte-identical:

1. A context-aware `check_ir_fitness_with_context` runs the fitness pass
   `_with_context` against the cached layers so it does not re-infer stdlib.
2. The in-context fitness report reconstitutes the **whole-program** metrics.
   Since chelis#858, `typed_nodes` / `total_nodes` are the inference product's
   honest checker-visit counters rather than structural AST counts. Each
   serialized `CheckedProgram` retains those counters, and the layered report
   adds them across the stdlib / non-stdlib partition. `count_nodes` and
   `structure_score` remain separate pure structural walks whose cached
   partition sums reconstruct only the `structure` component. This keeps the
   layered and monolithic JSON byte-identical without fabricating either metric
   (chelis#973).

## Acceptance oracle

`crates/chelis-cli/tests/stdlib_typecheck_cache_oracle.rs` and
`stdlib_typecheck_cache_concurrency.rs`. Two non-negotiable byte-identical
properties:

- **cold-vs-warm:** a cold-cache run and a warm-cache run of `chelis check` /
  `chelis build` over the stdlib-importing corpus produce byte-identical stdout.
- **monolithic-vs-in-context:** the monolithic typecheck path
  (`CHELIS_STDLIB_CACHE_DISABLE=1`) and the layered `_with_context` path produce
  byte-identical stdout, independent of the cache. A divergence here is a
  compiler-correctness bug in a `_with_context` variant, not a cache bug, and is
  an escalation, not a test to relax.

Plus the stale-key negative (a chelis-std source byte change flips the key) and
the concurrency / corrupt-cache-fall-through stress tests.

## Relationship to the Phase K `CompiledContext` cache

The Phase K cache (`crates/chelis-compiler-api/src/context.rs:426`,
`load_or_compile_for_package`) is a content-addressed, atomically-written,
version-enveloped disk cache of a whole reef package's `CompiledContext`. It is
keyed on `source_hash` over **every** source file in the resolved graph
(chelis-std + deps + user package), and before this change it was bypassed
entirely when `CHELIS_REEF_HOME` was unset. It is wired only into `chelis test`.

This change does **not** merge the two caches and does **not** change
`CompiledContext`'s public key or `chelis test`'s contract. The relationship is:

- `StdLibContext` is a new, narrower sub-layer keyed on the chelis-std content
  sub-key. It is the cross-fixture-reuse artifact: every fixture's whole-graph
  `CompiledContext` key differs, but they all share the same `StdLibContext`
  key.
- `CompiledContext`'s builder is re-implemented to consume `StdLibContext` as
  Layer 1 instead of re-checking chelis-std monolithically. Its public key
  (whole-graph `source_hash`) and its `chelis test` consumers are unchanged.
- The XDG cache-dir resolution helper is shared by both caches, which is what
  un-gates the existing `CompiledContext` cache for the no-`CHELIS_REEF_HOME`
  test-worker case as a side effect.

They remain two cache files with two keys (chelis-std content key, whole-graph
key) but one envelope format and one write path. They are layered, not merged.
