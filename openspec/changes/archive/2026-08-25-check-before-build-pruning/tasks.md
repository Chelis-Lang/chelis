## 1. Normative contract and failing fixtures

- [x] 1.1 Amend `spec/04-type-system.md` §2.5 so build checks every definition in the selected program before build-specific pruning.
- [x] 1.2 Amend `spec/05-risc-primitives.md` §3.7 and `[05-HOST-2]` so eval-only rejection uses the retained compile target after semantic success.
- [x] 1.3 Preserve the separate whole-program `tensor_scan` rule in `spec/05-risc-primitives.md` §3.6.
- [x] 1.4 Extend `library_cache_oracle.rs` helpers to run cache-disabled, cold, warm, and `chelis check` cases for one Reef fixture.
- [x] 1.5 Add a direct eval-only-tainted type-error fixture that build and check must reject with the same kind and source location.
- [x] 1.6 Add depth-three eval-only wrapper fixtures for effect and linearity errors that all build cache modes must reject.
- [x] 1.7 Require byte-identical build stderr across cache-disabled, cold, and warm rejection paths.
- [x] 1.8 Extend the well-typed depth-three wrapper fixture so every cache mode accepts and emits byte-identical C.
- [x] 1.9 Keep negative coverage for an unreachable ordinary type error, a reachable eval-only call, and an entry-unreachable `tensor_scan` call.
- [x] 1.10 Add a source-selection fixture that proves a file outside the selected target remains outside the build semantic gate.
- [x] 1.11 Record the pre-fix red result: build accepts each invalid eval-only-tainted fixture while `chelis check` rejects it.
- [x] 1.12 No structured-suggestion implementation exists in this revision.

## 2. Build pipeline implementation

- [x] 2.1 Retain the expanded full input as `selected_deep_exprs` until its complete semantic gate succeeds.
- [x] 2.2 Use `check_layered_for_build` success as the selected-program proof on a clean cache path.
- [x] 2.3 Route `Ok(None)` and cache-disabled builds through `checked_compilation_with_effects(&selected_deep_exprs)`.
- [x] 2.4 Move `drop_unreachable_eval_only_defs` and general reachability pruning after selected-program semantic success.
- [x] 2.5 Reuse the selected `CheckedCompilation` only when neither removal step changes the program.
- [x] 2.6 Recheck the exact retained program when removal changes it, and pass only that checked product to backend gates.
- [x] 2.7 Preserve the transitive eval-only closure, paired `defsig` removal, and reachable-call rejection behavior.
- [x] 2.8 Update `check_layered_for_build` and `cmd_build` comments to state that monolithic fallback receives the selected pre-prune program.
- [x] 2.9 Keep all cache keys, cache envelopes, and cache format versions unchanged.
- [x] 2.10 Preserve `Ok(None)` as a fallback request and reserve `Err(error)` for operational failure.
- [x] 2.11 Run the new fixtures green without weakening an expected diagnostic or cache-parity assertion.

## 3. Documentation and review routing

- [x] 3.1 Replace the chelis#1184 known-issue entry in `CHANGELOG.md` with a breaking-fix note and migration text.
- [x] 3.2 Update `docs/CHELIS_SURFACE.md` so eval-only backend scope matches amended `[05-HOST-2]`.
- [x] 3.3 Update JSON, CSV, and `process_run` test comments that describe eval-only rejection as unconditionally whole-program.
- [x] 3.4 Replace the divergence note on `drop_unreachable_eval_only_defs` with the new pre-prune semantic-gate invariant.
- [x] 3.5 Review `capture-type-system` and `capture-risc-primitives` for stale scope text before either capture change archives.
- [x] 3.6 Review executable examples and record that no valid example output changes.
- [x] 3.7 Keep chelis#1184 open until the authoritative completion oracle and fresh red-team pass are green.

## 4. Strict local validation

- [x] 4.1 Run `cargo fmt --all -- --check`.
- [x] 4.2 Run `openspec validate check-before-build-pruning --strict`.
- [x] 4.3 Run the authoritative oracle: `CARGO_TARGET_DIR=target/agents/check-before-build-pruning cargo nextest run -p chelis-cli --test library_cache_oracle --no-fail-fast`.
- [x] 4.4 Run the existing reachable `process_run` and whole-program `tensor_scan` test suites as supporting evidence.
- [x] 4.5 No structured-suggestion implementation exists in this revision.
- [x] 4.6 No SNAFU migration exists in this revision.
- [x] 4.7 Run `python3 scripts/gate.py --local` with an isolated target directory.

## 5. Adversarial validation

- [x] 5.1 Start a fresh local red-team subagent after the focused oracle is green.
- [x] 5.2 Confirm that a post-drop fallback mutation makes the new direct and transitive rejection fixtures fail.
- [x] 5.3 Confirm that a one-hop eval-only removal mutation makes the direct transitive-closure unit fixture fail.
- [x] 5.4 Confirm that a pre-prune eval-only backend-gate mutation makes the well-typed unused eval-only fixture fail.
- [x] 5.5 Confirm that the clean warm path uses layered success and does not add a monolithic full-program check.
- [x] 5.6 Record the red-team commands, results, and residual findings in the change review evidence.

## 6. Hosted acceptance and issue closure

- [x] 6.1 Obtain a green required `Integration Tests (Linux)` check for the final commit.
- [x] 6.2 Confirm that hosted diagnostics and generated C match the local oracle evidence.
- [x] 6.3 Confirm that all active cross-change oracles pass on the same revision.
- [x] 6.4 Link the implementation and hosted evidence to chelis#1184. Keep the issue open until the integration branch reaches `main`.
