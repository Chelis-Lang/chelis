# Phase 2 Wave 2 Status — Verified Sorry Inventory

Snapshot on branch `chelis-proof` after closing the remaining `SubstitutionDB.lean` admission.

## Acceptance Oracle

Use `cd proof/lean && lake build` as the authoritative branch-health check.

`proof/scripts/check_toolchain.py` is useful for workstation setup diagnostics, but it is not the proof-status oracle for this branch because it currently depends on extra local Vibe/Leanstral configuration that is not required for a repo-local Lean build.

## Baseline Versus Current State

- Verified baseline before the `SubstitutionDB.lean` fix: 5 active `sorry` declarations
  - `LaCaDiLE/AdjointTyping.lean:213`
  - `LaCaDiLE/Preservation.lean:392`
  - `LaCaDiLE/Substitution.lean:516`
  - `LaCaDiLE/Substitution.lean:1098`
  - `LaCaDiLE/SubstitutionDB.lean:1751`
- Current branch state after closing the remaining `SubstitutionDB` hole: 4 active `sorry` declarations
  - `LaCaDiLE/AdjointTyping.lean:213`
  - `LaCaDiLE/Preservation.lean:392`
  - `LaCaDiLE/Substitution.lean:516`
  - `LaCaDiLE/Substitution.lean:1098`

In other words: `SubstitutionDB.lean` is now `sorry`-free under the branch oracle, and the active proof debt has moved back to `Substitution.lean`, `Preservation.lean`, and `AdjointTyping.lean`.

## What Closed Recently

- `LinearitySoundness` is fully proved.
- The AddDim multiset refactor landed, including the `Quotient.sound` proof path.
- `plug_preserves_typing` is fully proved across all 19 evaluation-context cases.
- `Progress` was restructured around the three-way outcome (`value / steps / stuck-on-perform`) and is down to a single remaining admission outside this file.
- Store weakening infrastructure and the effect-scoping lemma `hasType_perform_eff_mem` are proved.
- `SubstitutionDB.lean`'s last remaining admission, `hasTypeDB_cap_weaken_tgrad_body`, is now closed by composing capability weakening with a new adjacent-capability swap lemma.

## Remaining Critical Path

1. `Substitution.lean`
   The branch still carries two admitted lemmas here, and they remain the main blocker for the substitution-dependent preservation cases.

2. `Preservation.lean`
   One admission remains. The substitution-dependent cases should be revisited immediately after the `Substitution.lean` debt is resolved or deleted.

3. `AdjointTyping.lean`
   One admission remains. This is still the calculus-level blocker for the `grad` preservation path and the eventual AD-correctness wave.

## Next Moves

1. Finish or retire the remaining `Substitution.lean` admissions, then rerun the branch oracle.
2. Use that to clear the blocked `Preservation.lean` cases and re-evaluate whether `DimSafety` and `EffectCorrectness` can now collapse to corollaries cleanly.
3. Return to the remaining `AdjointTyping.lean` admission once the substitution and preservation path is stable.
