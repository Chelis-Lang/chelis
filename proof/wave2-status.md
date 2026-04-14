# Phase 2 Wave 2 Status — Sorry Inventory

Snapshot at commit `04cdb50`. Build green, 8 sorry declarations.

## Theorems proven this wave

- **LinearitySoundness** — full theorem (Wave 1 close)
- Inversion lemmas (12 in Preservation.lean)
- `HasType.weaken_eff`
- `StoreWf.extend_fresh`, `StoreWf.remove`, `StoreWf.remove_extend`
- `storeFreshLoc_ne`
- `addDimStoreTyp_lookup`
- `canonical_forms_tensor`, `canonical_forms_pair`, `canonical_forms_arrow`
- `value_preserves_closed_context` (via `has_type_linear_shrinks` oracle)
- `AddDim.loc` case (internal)
- `Preservation.tconst/copy/tadd/tmul/tsum/texpand/tuniformLike` cases (internal)
- `Progress.var/pair-closure` cases (internal)

## Remaining 8 sorries

### Tractable with 1–2 dedicated turns each

1. **`Progress.has_type_linear_shrinks`** — mechanical 25-case `HasType.rec` with `DomSub` motive. Each case is short (1–8 lines) but simp invocations on `linearCtxDom` unfolding have been detail-sensitive. Three attempts so far with `simp only [linearCtxDom, List.map_append, List.mem_append, List.map_cons, List.mem_cons, List.map_nil]` have produced type mismatches on the decomposed membership hypotheses. Closing this single lemma is pre-requisite for value_preserves closing and for the fst/snd/handleRet Preservation cases.

2. **`AddDim.tvmap`** — structural obstacle: `addDim d_out (addDim d_in τ) ≠ addDim d_in (addDim d_out τ)` because `addDim` prepends the new dim. Cannot be closed without one of: (a) multiset-valued tensor dims, (b) canonical sort order on Dim, (c) dependent tensor shapes, or (d) rewriting `T-Vmap`'s result type to not nest `addDim`.

### Blocked on cross-cutting helpers

3. **`Substitution.weakening_tail`**
4. **`Substitution.exchange_tail`**
5. **`Substitution.subst_preserves_typing`** — all three need position-indexed reformulation with existential output split. Agent handoff document already written in-file; the `weakening_insert` statement has been identified, with a `DomSub`-style binder handling and a `freshInTerm` freshness predicate (already defined in Syntax.lean). ~300 lines of `HasType.rec` proof across ~25 cases plus two filter-commutation lemmas.

6. **`AdjointTyping.adjoint_preserves_typing`** — needs `opSignature accum` refined from `(unit, unit)` to a tensor-shaped signature. Requires either dependent types or a type-indexed `HasType.perform` (e.g., `perform op τ e` with explicit `τ`). Ripples into `T-Perform` everywhere.

7. **`Preservation.preservation`** — 10+ internal sorries, partitioned by helper dependency:
   - `beta`, `letBind`, `letpair`, `handleOpDirect` → need `subst_preserves_typing` (blocked by #3–#5)
   - `fst`, `snd`, `handleRet` → need `value_preserves_closed_context` + SubEffRow widening on `pair_inv`'s effect row witness
   - `tgrad` → needs `adjoint_preserves_typing` (blocked by #6)
   - `tvmap` → needs `addDim_preserves_typing`'s tvmap case (blocked by #2)
   - `ctx` → needs a `plug_preserves_typing` replacement lemma for evaluation contexts. Proof is by induction on `EvalCtx` with one case per constructor; self-contained.

8. **`Progress.progress_aux`** — sub-term cases partitioned by helper dependency:
   - `pair` (partial) → second sub-case needs non-pair-type branch ruled out
   - `app`, `letBind`, `letpair`, `perform` → need new inversion lemmas (`HasType.app_inv`, `letBind_inv`, `letpair_inv`, `perform_inv`) in Preservation.lean
   - `copy`, `sum`, `expand`, `uniformLike` → need `StoreWf` threading through `progress_aux` to produce runtime tensor values, plus `canonical_forms_tensor` as already-available oracle
   - `add`, `mul` → need two-argument dispatch with `value_preserves_closed_context` + `StoreWf`

## Wave 3 plan

Priority 1 (unblocks most): `has_type_linear_shrinks`. This alone lets `value_preserves_closed_context` become load-bearing rather than oracle-dependent, unblocking Preservation fst/snd/handleRet and Progress pair/app/etc.

Priority 2 (unblocks substitution cluster): position-indexed `weakening_insert`, which closes Substitution × 3 and then beta/letBind/letpair/handleOpDirect in Preservation.

Priority 3 (unblocks AD): `opSignature` tensor refinement (calculus change).

Priority 4 (unblocks AddDim): Phase 1 T7 dim-representation refactor.
