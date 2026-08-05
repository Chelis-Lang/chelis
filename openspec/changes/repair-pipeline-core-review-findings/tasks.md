# Tasks: repair-pipeline-core-review-findings

## 0. Verification probes

- [x] 0.1 Reproduced the vacuous runner on the pre-fix child argument list: the child
  (`--exact` only) reported `test result: ok. 0 passed; 0 failed; 1 ignored` while the
  parent reported `1 passed` and exited 0 — a vacuous pass asserting nothing.
- [x] 0.2 The layered `ContextMismatch` arm is unreachable through the public API: the
  core exposes no constructor that pairs a foreign `TypeEnv` with a context-checked
  extension (`ContextCheckedCompilation` retains the exact `&CheckedLibrary` that
  produced it, and `compose` takes no replacement). The state is reachable only through
  the cache-decode boundary, where it is already covered. D2 therefore needs no synthetic
  in-process test — the spec carve-out plus the corrected comment are the record.

## 1. Spec-first failing tests (non-vacuous runner)

- [x] 1.1 The parent branch captures the child with `Command::output()` and asserts the
  child report contains `1 passed` or `1 failed` (never `0 passed`) before asserting exit
  success.
- [x] 1.2 1.1 fails red against the pre-fix child argument list (probe 0.1: `0 passed;
  1 ignored`, no executed test).

## 2. Runner repair (D1)

- [x] 2.1 Added `--include-ignored` to the child argument list so the re-executed child
  runs the deferred leg under the pinned `SOURCE_DATE_EPOCH`.
- [x] 2.2 Kept the `#[ignore = "nondeterministic package archive hash; see chelis#1198"]`
  attribute; this change repairs the manual invocation, not the deferral.
- [x] 2.3 On this macOS-local machine the child EXECUTED the leg and honestly FAILED on
  the chelis#1198 nondeterministic `archive_sha256` (`99174816…` this run vs frozen
  `c440b136…`; every other schema field byte-identical). The child report showed
  `0 passed; 1 failed` — an executed assertion, not a skip. That is the non-vacuous
  contract holding; only the pre-fix skip violated it.
- [x] 2.4 `cargo nextest run -p chelis-reef --test pipeline_parity`: 2 passed, 1 skipped
  (`baseline_records_the_pre_migration_revision` and the rejected leg pass; the accepted
  leg stays ignored).

## 3. Layered fallback record (D2)

- [x] 3.1 Corrected the chelis#930 comment on `build_checked_library_layered` (and the
  paired fold-to-`None` comment in `compile_reef_context`): semantic rejections and
  upstream build failures fold into `None`; `LibraryRejection::ContextMismatch` returns
  `Some(Err(..))` because a proof-bind mismatch is an internal invariant failure the
  fallback must not mask.
- [x] 3.2 Loud-path coverage green: `context_decode_rejects_a_foreign_type_environment`,
  `context_decode_rejects_a_foreign_lowered_library`, and
  `context_decode_rejects_a_changed_lowered_payload_with_the_same_identity` all pass, plus
  the `chelis-pipeline-core` `validate_cached_library` rejection tests (24/24).

## 4. Evidence and documentation

- [x] 4.1 Measured cold `compile_reef_context` vs warm `CompiledContext::decode` on the
  `path_dep_fixture` (macOS local, 10 iters, 31,735-byte payload): ~30.0 ms cold vs
  ~5.2 ms warm, ratio ~0.17. Recorded in the cache-boundary section of
  `docs/investigations/compiler_pipeline_inventory.md`. Measured with a temporary
  `#[ignore]` timer that was not retained; no decode behavior change (D3).
- [x] 4.2 Documented the shared-declaration-order premise on `NamedRoots::aligned`
  (`artifacts.rs`) and the defensive `root_start` clamp (`lower.rs`). Comments only.
- [x] 4.3 Added the three pipeline-core guard commands to the `gate.py --list` snippet in
  `AGENTS.md`. Left the pre-existing workspace-nextest wording drift alone; verified it
  predates PR #1155 (base `7db8c271` already emits `--workspace  # full gate; CI coverage
  split`), so it is out of scope here (design Risks).
- [x] 4.4 The chelis#1194 → chelis#1198 repointing in `pipeline_parity.rs` and
  `BASELINE.md` is staged for commit with this change.

## 5. Validation

- [x] 5.1 `cargo nextest run -p chelis-reef --test pipeline_parity`: 2 passed, 1 skipped.
- [x] 5.2 Manual deferred-leg command
  `cargo test -p chelis-reef --test pipeline_parity -- --ignored` on macOS local: the
  child EXECUTED one test (`0 passed; 1 failed`) and the parent surfaced the child report
  — never a vacuous pass. The failure is the chelis#1198 nondeterministic hash, the
  honest state of the deferred leg. Pre-fix the same command exited 0 with `1 passed`.
- [x] 5.3 `scripts/compiler_pipeline_oracle.py`: `compiler pipeline oracle: PASS` (the
  authoritative completion oracle) in `CARGO_TARGET_DIR=target/agents/repair-findings`.
- [x] 5.4 `openspec validate repair-pipeline-core-review-findings --strict
  --no-interactive` passes; `openspec validate --all --strict` 36/36.
- [ ] 5.5 Red team: a fresh local subagent must confirm the manual invocation cannot
  report green with zero executed tests (adversarially: rename the test so `--exact`
  misses, drop `--include-ignored`, force a skip) and that the layered comment matches
  the shipped fallback. NOT DONE from this session: fresh local subagent execution is
  unavailable from the current toolset. Per the Red Team Protocol this must run before
  the change is counted as red-teamed; do not mark phase-complete without it.

### Validation record (2026-08-05)

Run in `CARGO_TARGET_DIR=target/agents/repair-findings` (isolated per the concurrency
contract).

- `cargo fmt --all -- --check`: clean.
- `cargo clippy -p chelis-compiler-api -p chelis-pipeline-core -p chelis-reef
  --all-targets -- -D warnings`: clean.
- `chelis-pipeline-core`: 24/24 lib, 9/9 doctests. Dependency and documentation guards
  PASS.
- `chelis-compiler-api`: facade compile-fail doctests intact; the three
  `context_decode_rejects_*` loud-path tests and the bincode round-trip pass.
- `chelis-reef` `pipeline_parity` default: 2 passed, 1 skipped. Manual `--ignored`: child
  executed the leg (`0 passed; 1 failed` on the chelis#1198 hash), parent surfaced it —
  non-vacuous.
- Authoritative oracle `scripts/compiler_pipeline_oracle.py`: PASS.
- `openspec validate --all --strict --no-interactive`: 36/36.

Residual: 5.5 (fresh-subagent red team) is unrun — fresh local subagent execution is
unavailable from this session's toolset. Every other task is complete and evidenced
above.
