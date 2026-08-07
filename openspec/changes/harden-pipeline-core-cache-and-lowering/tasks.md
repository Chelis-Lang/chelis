# Tasks: harden-pipeline-core-cache-and-lowering

## 1. Spec-first failing tests (contextual host lowering)

- [x] 1.1 In `crates/chelis-pipeline-core/src/lower.rs` tests, add a contextual case that
  drives the root-binding tail with `AllowHostBackend` and a nonfatal lower rejection and
  asserts an empty `NamedRoots` host result (fails red: the pre-fix code returned a lower
  error). (`contextual_host_backend_accepts_a_nonfatal_lower_rejection`)
- [x] 1.2 Add the negative parity: `AllowHostOnly` with a declared tensor root name and a
  nonfatal rejection still returns the lower error.
  (`contextual_host_only_rejects_a_nonfatal_rejection_with_tensor_names`)
- [x] 1.3 Record that 1.1 fails red against the pre-fix `lower_checked_with_context`.

## 2. Contextual host lowering parity (D2)

- [x] 2.1 Extract the contextual root-binding tail into `finish_contextual_lowering`
  (mirroring `finish_isolated_lowering`), mapping `ComposedLowering` into the existing
  `LoweredProgram` carrier so the arm is unit-testable with a synthetic diagnostic.
- [x] 2.2 Add the `AllowHostBackend` nonfatal-rejection arm to `finish_contextual_lowering`,
  matching the isolated path's `RootBindingMode::AcceptedNonfatalRejection` result.
- [x] 2.3 Run 1.1-1.2 green; confirm the isolated `AllowHostBackend` tests and the
  contextual `AllowHostOnly`/replacement-library tests still pass
  (`chelis-pipeline-core` 24/24).

## 3. Cache proof recompute (D1) - Option C (document the residual)

- [x] 3.1 Recompute-from-source is infeasible at the current identity derivation: the id
  is bound pre-effects while the cached program is post-effects, and a layered build's
  cached library is a composed program whose id is the extension's id over the base
  context, not a hash of the composed `exprs()`. A naive recompute rejects legitimate
  caches (design D1).
- [x] 3.2 Option C: documented the residual on `validate_cached_library`
  (`crates/chelis-pipeline-core/src/semantic.rs`) - the stored id is a self-consistency
  check, not a source-recompute - and in the cache-boundary section of
  `docs/investigations/compiler_pipeline_inventory.md`. No cache-decode or `chelis-types`
  behavior change.
- [x] 3.3 Option A (canonical post-effects identity + cache recompute) tracked as the
  `canonicalize-library-proof-identity` follow-up OpenSpec change.

## 4. Docs and spec sync

- [x] 4.1 Spec delta reduced to the shippable "Contextual host lowering matches isolated
  host lowering" requirement; the cache recompute is recorded in design D1 as a decision,
  not a normative requirement.
- [ ] 4.2 If Option A/B is chosen later, update the pipeline extraction current-state notes
  and add the cache-recompute requirement in the follow-up change.

## 5. Validation and oracle

- [x] 5.1 `cargo nextest run -p chelis-pipeline-core` green in an isolated
  `CARGO_TARGET_DIR` (24/24, including the two new contextual host-lowering tests).
- [ ] 5.2 `cargo test -p chelis-pipeline-core --doc` and
  `.venv/bin/python scripts/compiler_pipeline_oracle.py` green (the authoritative
  completion oracle) - run before merge.
- [x] 5.3 `openspec validate harden-pipeline-core-cache-and-lowering --strict` passes.
- [ ] 5.4 Red team: a fresh local subagent confirms the isolated-vs-contextual
  `AllowHostBackend` parity holds and that `AllowHostOnly` keeps its tensor-name guard on
  both paths.

### Validation record (2026-08-04)

- `chelis-pipeline-core` 24/24 green (was 22; +2 contextual host-lowering tests) in
  `CARGO_TARGET_DIR=target/agents/review-flaws`.
- Flaw #3 (cache proof recompute): implementation surfaced the pre-effect binding and
  composition-lossy identity obstacles. Resolved via Option C (documented the residual on
  the owning function and in the inventory); Option A (canonical post-effects derivation)
  is the `canonicalize-library-proof-identity` follow-up change.