## Why

A fresh review of the pipeline-core extraction (PR #1155) found residual defects that
none of the active pipeline-core changes own. Two are contract bugs; the rest are
evidence and documentation gaps that would otherwise rot silently.

1. **The deferred parity leg's manual runner is vacuous.** The accepted parity leg
   (`accepted_package_outputs_match_the_pre_migration_baseline`) re-executes itself in a
   child process to pin `SOURCE_DATE_EPOCH`. The leg is now `#[ignore]`d pending
   chelis#1198, but the child invocation passes only `--exact <name>` — no
   `--include-ignored` — so libtest selects the test, skips it as ignored, runs zero
   tests, and exits 0. The parent asserts only on the exit status, so every explicit
   `--ignored` invocation reports green while executing no assertions. An invoker
   verifying the chelis#1198 fix would trust a result that asserted nothing.

2. **The layered fallback contract changed without a record.** The chelis#930 comment on
   the layered library build says the path "folds ANY failure into `None`" (monolithic
   fallback). The extraction added `LibraryRejection::ContextMismatch`, which now returns
   a loud check error instead of folding. The loud error is the right call — a proof-bind
   mismatch is an internal invariant failure, not a user-program rejection — but the
   decision exists only in code, contradicting the comment above it and the
   "monolithic error fallback" language in the capability spec.

3. **Unmeasured warm-path cost.** Cache decode now reruns effect and linearity checks and
   re-lowers the whole library, using the cached lowered payload only as a comparison
   artifact. The hardening is deliberate, but no measured warm-decode number exists, and
   the cost is in tension with the chelis#1168/chelis#830 build-lane cache work.

4. **Undocumented order contract.** `NamedRoots::aligned` validates only the tensor-name
   and DAG-root counts and trusts positional order; the contextual root slice clamps
   `root_start` to the composed root count. Both are sound today for reasons that live
   nowhere.

5. **Gate documentation drift.** The `gate.py --list` snippet in the agent contract omits
   the three pipeline-core guard commands the gate actually runs.

6. **Stale issue citations.** The parity fixtures cite chelis#1194, which was closed as
   replaced by chelis#1198; the repointing edits exist only as uncommitted worktree
   state.

## What Changes

- Make the deferred parity leg's manual invocation non-vacuous: the child re-exec passes
  `--include-ignored`, and the parent asserts that the child actually executed exactly
  one test rather than trusting the exit status alone.
- Record the layered fallback carve-out in the capability spec and fix the stale
  chelis#930 comment: semantic rejections still fold into the monolithic fallback; a
  library proof-bind mismatch surfaces as a loud check error.
- Measure the warm cache-decode path (cold vs warm `compile_reef_context`) and record
  the numbers in the current-state inventory. Evidence only; no decode behavior change.
- Document the root-alignment order contract on `NamedRoots::aligned` and the defensive
  clamp in the contextual root slice.
- Sync the agent contract's `gate.py --list` snippet with the script's actual output.
- Commit the chelis#1194 → chelis#1198 citation repointing in the parity fixtures.

### Non-Goals

- Do not fix the package-archive nondeterminism and do not convert the accepted leg to a
  relative monolithic-vs-layered oracle; chelis#1198 owns both.
- Do not remove the `#[ignore]` from the accepted parity leg.
- Do not change cache-decode behavior, the `LibraryProofId` derivation, or the
  composition path; the `canonicalize-library-proof-identity` change owns the identity
  derivation.
- Do not change accepted-path language, compiler, CLI, backend, runtime, or package
  behavior.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `compiler-pipeline-architecture`:
  - "Reef uses canonical semantic transitions" — a deferred parity leg executes its
    assertions when explicitly invoked; a harness child that runs zero tests fails the
    invocation.
  - "Context and cache parity" — semantic rejections fold into the monolithic fallback;
    a library proof-bind mismatch is a loud error, never a silent fallback.

## Impact

- `crates/chelis-reef/tests/pipeline_parity.rs` — child re-exec gains
  `--include-ignored` and an executed-test assertion; chelis#1194 citations become
  chelis#1198.
- `crates/chelis-reef/tests/fixtures/pipeline_parity/BASELINE.md` — citation repointing.
- `crates/chelis-compiler-api/src/context.rs` — the chelis#930 fold-to-`None` comment
  states the proof-bind carve-out.
- `crates/chelis-pipeline-core/src/artifacts.rs` and `src/lower.rs` — order-contract and
  clamp documentation only.
- `docs/investigations/compiler_pipeline_inventory.md` — warm-decode measurement record.
- `AGENTS.md` — gate command snippet regenerated from `gate.py --list`.
- `openspec/specs/compiler-pipeline-architecture/spec.md` (delta).

This change repairs verification honesty and records an already-shipped decision. The
numbered specifications retain authority for language, compiler, serialization, backend,
runtime, and package behavior.
