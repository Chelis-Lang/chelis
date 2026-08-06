## Context

The pipeline-core extraction (PR #1155) sealed semantic transitions behind typed
artifacts and moved the Reef parity harness onto frozen fixtures. Review after the
`harden-pipeline-core-cache-and-lowering` pass found the residuals this change owns.

The accepted parity leg re-executes the test binary so `SOURCE_DATE_EPOCH` is set in a
fresh process (in-process `std::env::set_var` is racy under the parallel test harness,
so the child-spawn shape is correct). When the leg gained `#[ignore]` for chelis#1198,
nobody revisited the child's argument list. libtest's behavior makes the combination
vacuous: `--exact <name>` selects an ignored test but does not run it, the run reports
`0 passed; 1 ignored`, and the process exits 0. The parent's only check is
`status.success()`.

Separately, `build_checked_library_layered` in `crates/chelis-compiler-api/src/context.rs`
returns `Some(Err(..))` for `LibraryRejection::ContextMismatch` while the chelis#930
comment above it still promises that the layered path folds ANY failure into `None`.

## Goals / Non-Goals

**Goals:**

- An explicit invocation of the deferred parity leg either executes the leg's assertions
  or fails; it can never report green while running nothing.
- The layered fallback contract in the spec and the code comment agree with the shipped
  behavior.
- The warm-decode cost, the root-alignment order contract, and the gate command list are
  recorded where their readers look.

**Non-Goals:**

- No archive-determinism fix and no relative-oracle conversion (chelis#1198).
- No `LibraryProofId` derivation change (`canonicalize-library-proof-identity`).
- No cache-decode behavior change; measurement only.

## Decisions

### D1: Non-vacuous runner via `--include-ignored` plus an executed-test assertion

The child re-exec adds `--include-ignored`, and the parent switches from
`Command::status()` to `Command::output()` and asserts the child report shows exactly one
executed test before asserting success.

The flag alone is insufficient: a future rename that breaks the `--exact` filter would
again select zero tests and exit 0. The output assertion makes "zero tests ran" a failure
regardless of cause, which is the actual invariant. The exit-status assertion stays; a
child that runs the leg and fails its assertions must still fail the parent.

Consequence to state plainly: after this fix, the manual invocation surfaces the leg's
real outcome. On a machine whose archive hash differs from the frozen baseline the leg
FAILS — that is the honest state of an ignored-because-nondeterministic test and is why
the leg stays `#[ignore]`d until chelis#1198 replaces the frozen baseline with a
relative oracle. The invariant this change installs (an invoked deferred leg executes or
fails) is independent of that conversion and survives it.

### D2: The layered proof-bind mismatch stays loud

Keep the `Some(Err(..))` behavior and record it, rather than restoring fold-to-`None`.

The chelis#930 fold exists so a user program that fails a semantic stage on the layered
path gets the byte-identical monolithic rejection. `ContextMismatch` is not that: it
means the freshly composed library and its type environment disagree on the proof
identity, which is only reachable through an internal proof-threading bug. Folding it
into the monolithic fallback would produce a correct user-visible result while
permanently hiding the invariant failure — exactly the silent-fallback class the
extraction exists to eliminate. The cost is bounded: the guard is unreachable through
the public API (the type-state design prevents constructing the mismatch from outside),
so no working program can newly fail.

Coverage is honest about that unreachability: the loud path is exercised at the cache
decode boundary (`context_decode_rejects_a_foreign_type_environment` and the
`validate_cached_library` rejection tests), while the in-process layered arm is covered
by the spec carve-out and the corrected comment, not by a synthetic test that would need
to break the core's own privacy to construct the state.

### D3: Warm-decode cost is evidence, not a behavior change

Decode reruns effects and linearity and re-lowers the library by design (the
`harden-pipeline-core-cache-and-lowering` decision). What is missing is the number. This
change measures cold vs warm `compile_reef_context` on the standard fixture and records
it in `docs/investigations/compiler_pipeline_inventory.md` next to the existing
cache-boundary notes, so the maintainer deciding between the current rerun-everything
decode and a future recompute-the-identity decode (`canonicalize-library-proof-identity`)
has the trade-off in front of them. If the measurement shows the warm path regressing
below the cold path, that is a finding for a dedicated change, not something to patch
here.

### D4: Order contract and clamp are documented, not re-armored

`NamedRoots::aligned` zips declared tensor names with DAG roots positionally and checks
only the counts. This is sound because both sequences derive from the same top-level
declaration order — the checker collects root names and the lowerer emits roots in that
shared order — so a permutation cannot arise without one side breaking its own
derivation. A structural name-to-root verification would need the DAG to carry per-root
provenance it does not have; adding that is out of scope. The contract gets a doc
comment naming the shared-order premise so a future reordering change knows what it
must preserve.

The contextual `root_start` clamp (`library_root_count.min(dag.roots().len())`) is a
defensive slice guard: a healthy composition always has at least the library's roots.
The comment states that a composed DAG with fewer roots than its library indicates an
upstream lowering defect and that the case is caught by the root-count alignment unless
the new code declares no tensor names.

## Risks / Trade-offs

- **D1 makes a previously-green manual command fail on most machines.** Intended; the
  green was vacuous. BASELINE.md already states the frozen values are reference-only.
- **D2 forgoes a fallback that would have kept a hypothetical broken build working.**
  Accepted: masking an internal invariant failure costs more than the loud error.
- The AGENTS.md snippet sync only covers the lines this extraction added; the
  pre-existing wording drift on the workspace-nextest line predates PR #1155 and is left
  to a docs-only follow-up if wanted.

## Open Questions

None. The archive-nondeterminism root cause and the relative-oracle shape remain with
chelis#1198; the identity derivation remains with `canonicalize-library-proof-identity`.
