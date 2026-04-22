# Phase 2 Wave 2 Status — Preservation And Sorry Inventory

Snapshot on branch `chelis-proof` after closing the DB-backed preservation wrappers and the captured-context preservation cases.

## Acceptance Oracle

Use `cd proof/lean && lake build` as the authoritative branch-health check.

`proof/scripts/check_toolchain.py` is useful for workstation setup diagnostics, but it is not the proof-status oracle for this branch because it currently depends on extra local Vibe/Leanstral configuration that is not required for a repo-local Lean build.

## Current State

- Verified baseline before the `SubstitutionDB.lean` fix: 5 active `sorry` declarations
  - `LaCaDiLE/AdjointTyping.lean:213`
  - `LaCaDiLE/Preservation.lean:392`
  - `LaCaDiLE/Substitution.lean:516`
  - `LaCaDiLE/Substitution.lean:1098`
  - `LaCaDiLE/SubstitutionDB.lean:1751`
- Current branch state is materially different from that baseline:
  - `TranslationDB.lean`'s four DB-backed preservation wrappers are closed
  - `Preservation.lean` closes the substitution-dependent cases, `handleOpCtx`, `handleOpCtxs`, and `ctx`
  - the preservation theorem now carries an explicit `RuntimeLinear` premise for closed runtime terms
  - the only remaining `Preservation.lean` admissions are the agreed calculus-level blockers `tgrad` and `tvmap`

The branch still contains admitted lemmas outside those two cases, notably in `Substitution.lean`, `AdjointTyping.lean`, `SubstitutionDB.lean`, and `Translation.lean`. The critical-path meaning of the inventory has changed: the main blocker is no longer generic substitution-dependent preservation, but the runtime-linearity loop plus the tensor-specific upstream lemmas.

## What Closed Recently

- `LinearitySoundness` is fully proved.
- The AddDim multiset refactor landed, including the `Quotient.sound` proof path.
- `plug_preserves_typing` is fully proved across all 19 evaluation-context cases.
- `Progress` was restructured around the three-way outcome (`value / steps / stuck-on-perform`) and is down to a single remaining admission outside this file.
- Store weakening infrastructure and the effect-scoping lemma `hasType_perform_eff_mem` are proved.
- `SubstitutionDB.lean`'s last remaining admission, `hasTypeDB_cap_weaken_tgrad_body`, is now closed by composing capability weakening with a new adjacent-capability swap lemma.
- The old `SlotCorr` / `SlotKillsOnly` bridge has been deleted; `CtxCorr` is now the only translation correspondence layer.
- `OpSigMatch` has been tightened to a functional operation-signature witness, which closed the direct handler wrapper in `TranslationDB.lean`.
- The operational semantics now mint a fresh captured continuation binder instead of hardcoding `_kArg`.
- `Preservation.lean` now closes the generic `ctx` case by using frame-local store agreement over untouched context locations instead of global store monotonicity.

## Remaining Critical Path

1. `RuntimeLinear` propagation
   `Preservation.lean` is now honest but conditional: it proves one-step type preservation for closed runtime-linear programs. To iterate that theorem over a reduction sequence, the branch still needs a theorem that checked runtime configurations preserve `RuntimeLinear` across one step, or an equivalent reachability theorem from checked source terms.

2. `AdjointTyping.lean`
   This remains the upstream blocker for the `tgrad` preservation case and the eventual AD-correctness wave.

3. `AddDim.lean` / `tvmap`
   The dimension-representation side is still the upstream blocker for the `tvmap` preservation case.

## Next Moves

1. Prove the runtime-linearity invariant needed to re-establish the strengthened preservation premise after each step.
2. Sync the paper-facing workstreams so §5 states preservation with the `RuntimeLinear` premise and explains why checked source programs still satisfy it along execution.
3. Return to `AdjointTyping.lean` for the `tgrad` blocker once the runtime-linearity loop is closed.
4. Leave `tvmap` behind the dimension-representation / `AddDim` work rather than pushing `Preservation.lean` directly.
