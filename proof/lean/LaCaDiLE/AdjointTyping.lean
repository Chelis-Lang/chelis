-- LaCaDiLE/AdjointTyping.lean — adjoint typing lemma (Phase 2 proof).
--
-- WS2.2 target: the adjoint transformation preserves typing. Per the
-- E-Grad reduction (figures/opsem.tex), `grad(λx:τ.e)` reduces to
--    λx:τ. λgs:τ_out. handle[{Accum}] (adjoint e x gs) with h_accum
-- so the *body of the handle* is `adjoint e x gs`. The handle's clause
-- assembles the parameter gradient and returns it as tensor[ds]; the
-- adjoint body itself is a sequence of `perform accum (...)` calls
-- whose head term has type `unit` and effect row `{accum}` (plus the
-- forwarded effects of the original body, which must lie in
-- DiffCompat).
--
-- Wave 3 calculus refinement originally expected an `OpSigMatch`
-- witness that lets `perform accum` take a tensor argument. The
-- current global `OpSigMatch` in `Syntax.lean` instead fixes `accum`
-- at `unit -> unit`, so the adjoint leaf skeleton now consumes the
-- tensor seed via a trivial `letBind` and emits `perform accum unit`.
-- This keeps the linear-context threading honest while the richer
-- tensor-carrying accum surface remains future work.
--
-- Wave 4 Track B: the helper `adjoint_typed_aux` is counter-threaded
-- (via `adjointFrom` rather than `adjoint`) and carries a freshness
-- premise `AdjointNamesFresh n Γ_s Γ_s'` stating that every
-- counter-indexed fresh name the adjoint transform could mint at
-- counter ≥ n is disjoint from both linear contexts. The freshness
-- discipline lets the `add` and `mul` cases close by building typed
-- `letpair` / `letBind` / `copy` derivations over sub-calls at the
-- incremented counter.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.AdjointTransform
import LaCaDiLE.StringHelpers
import LaCaDiLE.Operational
import LaCaDiLE.Substitution
import LaCaDiLE.TranslationDB

namespace LaCaDiLE

/-- The nine base names the adjoint transform mints. Used to state the
    freshness premise of `adjoint_typed_aux` uniformly across cases. -/
def adjointBases : List String :=
  ["gA", "gB", "a", "aTape", "b", "bTape", "y", "adjA", "adjHb"]

/-- Every base name used by `adjointFrom` is hash-free, so
    `freshName_ne_of_base_ne` applies whenever two bases differ. -/
theorem adjointBases_noHash :
    ∀ b ∈ adjointBases, NoHash b := by
  intro b hb
  simp [adjointBases] at hb
  rcases hb with h | h | h | h | h | h | h | h | h <;>
    (subst h; show '#' ∉ _; decide)

/-- Freshness predicate: every counter-indexed fresh name
    `freshName base m` for `m ≥ n` and `base ∈ adjointBases` is
    disjoint from `Γ`. Monotone in `n`. -/
def AdjointNamesFresh (n : Nat) (Γ : LinearCtx) : Prop :=
  ∀ m, m ≥ n → ∀ base, base ∈ adjointBases →
    freshName base m ∉ linearCtxDom Γ

/-- Freshness is monotone in the counter: if all names at counters ≥ n
    are fresh, so are all names at counters ≥ n' for any n' ≥ n. -/
theorem AdjointNamesFresh.mono {n n' : Nat} {Γ : LinearCtx}
    (h : AdjointNamesFresh n Γ) (hle : n ≤ n') : AdjointNamesFresh n' Γ :=
  fun m hm base hb => h m (Nat.le_trans hle hm) base hb

/-- Adding a binder whose name is a `freshName` at a *strictly smaller*
    counter preserves freshness at the current counter. Used to extend
    `Γ_s'` with `(freshName b k, t)` when recursing on an inner
    sub-term at counter ≥ k+1. -/
theorem AdjointNamesFresh.cons_freshName
    {n k : Nat} {Γ : LinearCtx} {b : String} {t : Option Typ}
    (h : AdjointNamesFresh n Γ)
    (hb_adj : b ∈ adjointBases) (hk : k < n) :
    AdjointNamesFresh n (Γ ++ [(freshName b k, t)]) := by
  intro m hm base hbase
  have hne_m_k : m ≠ k := by
    intro heq
    exact (Nat.not_lt.mpr (heq ▸ hm) hk)
  have hne_name : freshName base m ≠ freshName b k := by
    by_cases hbb : base = b
    · subst hbb
      exact freshName_ne_of_nat_ne base m k hne_m_k
    · exact freshName_ne_of_base_ne base b m k
        (adjointBases_noHash base hbase)
        (adjointBases_noHash b hb_adj) hbb
  intro hmem
  -- hmem : freshName base m ∈ linearCtxDom (Γ ++ [(freshName b k, t)])
  simp only [linearCtxDom, List.map_append, List.map_cons, List.map_nil,
             List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hmem
  rcases hmem with hIn | hIn
  · exact h m hm base hbase (by simpa [linearCtxDom] using hIn)
  · exact hne_name hIn

/-- Term-side freshness companion to `AdjointNamesFresh`: every adjoint
    `freshName` at counter `m ≥ n` is fresh in the source term. This is
    the reusable hypothesis needed to weaken forward-replayed source
    subterms past generated seed / tape binders. -/
def AdjointTermFresh (n : Nat) (e : Term) : Prop :=
  ∀ m, m ≥ n → ∀ base, base ∈ adjointBases →
    freshInTerm (freshName base m) e

theorem AdjointTermFresh.mono {n n' : Nat} {e : Term}
    (h : AdjointTermFresh n e) (hle : n ≤ n') : AdjointTermFresh n' e := by
  intro m hm base hb
  exact h m (Nat.le_trans hle hm) base hb

theorem AdjointTermFresh.add {n : Nat} {e1 e2 : Term}
    (h : AdjointTermFresh n (Term.add e1 e2)) :
    AdjointTermFresh n e1 ∧ AdjointTermFresh n e2 := by
  constructor <;>
    intro m hm base hb
  · exact (freshInTerm_add (h m hm base hb)).1
  · exact (freshInTerm_add (h m hm base hb)).2

theorem AdjointTermFresh.mul {n : Nat} {e1 e2 : Term}
    (h : AdjointTermFresh n (Term.mul e1 e2)) :
    AdjointTermFresh n e1 ∧ AdjointTermFresh n e2 := by
  constructor <;>
    intro m hm base hb
  · exact (freshInTerm_mul (h m hm base hb)).1
  · exact (freshInTerm_mul (h m hm base hb)).2

theorem AdjointTermFresh.pair {n : Nat} {e1 e2 : Term}
    (h : AdjointTermFresh n (Term.pair e1 e2)) :
    AdjointTermFresh n e1 ∧ AdjointTermFresh n e2 := by
  constructor <;>
    intro m hm base hb
  · exact (freshInTerm_pair (h m hm base hb)).1
  · exact (freshInTerm_pair (h m hm base hb)).2

theorem AdjointTermFresh.copy {n : Nat} {e : Term}
    (h : AdjointTermFresh n (Term.copy e)) :
    AdjointTermFresh n e := by
  intro m hm base hb
  exact freshInTerm_copy (h m hm base hb)

theorem AdjointTermFresh.fst {n : Nat} {tRight : Typ} {e : Term}
    (h : AdjointTermFresh n (Term.fst tRight e)) :
    AdjointTermFresh n e := by
  intro m hm base hb
  exact freshInTerm_fst (h m hm base hb)

theorem AdjointTermFresh.snd {n : Nat} {tLeft : Typ} {e : Term}
    (h : AdjointTermFresh n (Term.snd tLeft e)) :
    AdjointTermFresh n e := by
  intro m hm base hb
  exact freshInTerm_snd (h m hm base hb)

theorem AdjointTermFresh.sum {n : Nat} {e : Term} {d : Dim}
    (h : AdjointTermFresh n (Term.sum e d)) :
    AdjointTermFresh n e := by
  intro m hm base hb
  exact freshInTerm_sum (h m hm base hb)

theorem AdjointTermFresh.expand {n : Nat} {e : Term} {d : Dim}
    (h : AdjointTermFresh n (Term.expand e d)) :
    AdjointTermFresh n e := by
  intro m hm base hb
  exact freshInTerm_expand (h m hm base hb)

theorem AdjointTermFresh.perform {n : Nat} {op : EffectLabel} {e : Term}
    (h : AdjointTermFresh n (Term.perform op e)) :
    AdjointTermFresh n e := by
  intro m hm base hb
  exact freshInTerm_perform (h m hm base hb)

theorem AdjointTermFresh.uniformLike {n : Nat} {e : Term} {lo hi : Float}
    (h : AdjointTermFresh n (Term.uniformLike e lo hi)) :
    AdjointTermFresh n e := by
  intro m hm base hb
  exact freshInTerm_uniformLike (h m hm base hb)

theorem AdjointTermFresh.abs {n : Nat} {x : String} {tx : Typ} {body : Term}
    (h : AdjointTermFresh n (Term.abs x tx body)) :
    AdjointTermFresh n body := by
  intro m hm base hb
  exact (freshInTerm_abs (h m hm base hb)).2

theorem AdjointTermFresh.grad {n : Nat} {x : String} {tx tOut : Typ} {body : Term}
    (h : AdjointTermFresh n (Term.grad x tx tOut body)) :
    AdjointTermFresh n body := by
  intro m hm base hb
  exact (freshInTerm_grad (h m hm base hb)).2

theorem AdjointTermFresh.vmap {n : Nat} {x : String} {tx : Typ} {d : Dim} {body : Term}
    (h : AdjointTermFresh n (Term.vmap x tx d body)) :
    AdjointTermFresh n body := by
  intro m hm base hb
  exact (freshInTerm_vmap (h m hm base hb)).2

theorem AdjointTermFresh.handle_body
    {n : Nat} {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    (h : AdjointTermFresh n (Term.handle epsH body clauses)) :
    AdjointTermFresh n body := by
  intro m hm base hb
  exact freshInTerm_handle_body (h m hm base hb)

theorem AdjointTermFresh.letBind {n : Nat} {x : String} {e1 e2 : Term}
    (h : AdjointTermFresh n (Term.letBind x e1 e2)) :
    AdjointTermFresh n e1 ∧ AdjointTermFresh n e2 := by
  constructor <;>
    intro m hm base hb
  · exact (freshInTerm_letBind (h m hm base hb)).2.1
  · exact (freshInTerm_letBind (h m hm base hb)).2.2

theorem AdjointTermFresh.letpair {n : Nat} {x y : String} {e1 e2 : Term}
    (h : AdjointTermFresh n (Term.letpair x y e1 e2)) :
    AdjointTermFresh n e1 ∧ AdjointTermFresh n e2 := by
  constructor <;>
    intro m hm base hb
  · exact (freshInTerm_letpair (h m hm base hb)).2.2.1
  · exact (freshInTerm_letpair (h m hm base hb)).2.2.2

theorem adjointFreeCtxSupported.add
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : AdjointFreeCtxSupported Gamma (Term.add e1 e2)) :
    AdjointFreeCtxSupported Gamma e1 ∧ AdjointFreeCtxSupported Gamma e2 := by
  constructor <;> intro z t hz hLook
  · exact h z t (by simp [freeVars, List.mem_append, hz]) hLook
  · exact h z t (by simp [freeVars, List.mem_append, hz]) hLook

theorem adjointFreeCtxSupported.mul
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : AdjointFreeCtxSupported Gamma (Term.mul e1 e2)) :
    AdjointFreeCtxSupported Gamma e1 ∧ AdjointFreeCtxSupported Gamma e2 := by
  constructor <;> intro z t hz hLook
  · exact h z t (by simp [freeVars, List.mem_append, hz]) hLook
  · exact h z t (by simp [freeVars, List.mem_append, hz]) hLook

theorem adjointFreeCtxSupported.pair
    {Gamma : LinearCtx} {e1 e2 : Term}
    (h : AdjointFreeCtxSupported Gamma (Term.pair e1 e2)) :
    AdjointFreeCtxSupported Gamma e1 ∧ AdjointFreeCtxSupported Gamma e2 := by
  constructor <;> intro z t hz hLook
  · exact h z t (by simp [freeVars, List.mem_append, hz]) hLook
  · exact h z t (by simp [freeVars, List.mem_append, hz]) hLook

theorem adjointFreeCtxSupported.copy
    {Gamma : LinearCtx} {e : Term}
    (h : AdjointFreeCtxSupported Gamma (Term.copy e)) :
    AdjointFreeCtxSupported Gamma e := by
  intro z t hz hLook
  exact h z t (by simpa [freeVars] using hz) hLook

theorem adjointFreeCtxSupported.fst
    {Gamma : LinearCtx} {tRight : Typ} {e : Term}
    (h : AdjointFreeCtxSupported Gamma (Term.fst tRight e)) :
    AdjointFreeCtxSupported Gamma e := by
  intro z t hz hLook
  exact h z t (by simpa [freeVars] using hz) hLook

theorem adjointFreeCtxSupported.snd
    {Gamma : LinearCtx} {tLeft : Typ} {e : Term}
    (h : AdjointFreeCtxSupported Gamma (Term.snd tLeft e)) :
    AdjointFreeCtxSupported Gamma e := by
  intro z t hz hLook
  exact h z t (by simpa [freeVars] using hz) hLook

theorem adjointFreeCtxSupported.sum
    {Gamma : LinearCtx} {e : Term} {d : Dim}
    (h : AdjointFreeCtxSupported Gamma (Term.sum e d)) :
    AdjointFreeCtxSupported Gamma e := by
  intro z t hz hLook
  exact h z t (by simpa [freeVars] using hz) hLook

theorem adjointFreeCtxSupported.expand
    {Gamma : LinearCtx} {e : Term} {d : Dim}
    (h : AdjointFreeCtxSupported Gamma (Term.expand e d)) :
    AdjointFreeCtxSupported Gamma e := by
  intro z t hz hLook
  exact h z t (by simpa [freeVars] using hz) hLook

theorem adjointFreeCtxSupported.perform
    {Gamma : LinearCtx} {op : EffectLabel} {e : Term}
    (h : AdjointFreeCtxSupported Gamma (Term.perform op e)) :
    AdjointFreeCtxSupported Gamma e := by
  intro z t hz hLook
  exact h z t (by simpa [freeVars] using hz) hLook

theorem adjointFreeCtxSupported.uniformLike
    {Gamma : LinearCtx} {e : Term} {lo hi : Float}
    (h : AdjointFreeCtxSupported Gamma (Term.uniformLike e lo hi)) :
    AdjointFreeCtxSupported Gamma e := by
  intro z t hz hLook
  exact h z t (by simpa [freeVars] using hz) hLook

theorem adjointFreeCtxSupported.handle_body
    {Gamma : LinearCtx} {epsH : EffectRow} {body : Term}
    {clauses : List (EffectLabel × String × String × Term)}
    (h : AdjointFreeCtxSupported Gamma (Term.handle epsH body clauses)) :
    AdjointFreeCtxSupported Gamma body := by
  intro z t hz hLook
  exact h z t (by simp [freeVars, List.mem_append, hz]) hLook

mutual

private theorem hasType_drop_diff_supported
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType (Capability.diff :: Delta) Sigma Gamma e t eps Gamma')
    (hSupp : AdjointSupported e) :
    HasType Delta Sigma Gamma e t eps Gamma' := by
  have hMain :
      ∀ {Delta0 : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
        {e : Term} {t : Typ} {eps : EffectRow},
        HasType Delta0 Sigma Gamma e t eps Gamma' →
        ∀ Delta, Delta0 = Capability.diff :: Delta →
          AdjointSupported e →
          HasType Delta Sigma Gamma e t eps Gamma' := by
    intro Delta0 Sigma Gamma Gamma' e t eps hTyped
    induction hTyped using HasType.rec
      (motive_2 := fun Delta0 Sigma Gamma2 Gamma3 t epsR cls _ =>
        ∀ Delta, Delta0 = Capability.diff :: Delta →
          AdjointSupportedClauses cls →
          ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR cls) with
    | var Delta0 Sigma GammaPre GammaPost x tx =>
        intro Delta hEq _hSupp
        cases hEq
        exact HasType.var Delta Sigma GammaPre GammaPost x tx
    | unit Delta0 Sigma Gamma =>
        intro Delta hEq _hSupp
        cases hEq
        exact HasType.unit Delta Sigma Gamma
    | abs =>
        intro _Delta _hEq hSupp
        cases hSupp
    | app =>
        intro _Delta _hEq hSupp
        cases hSupp
    | letBind Delta0 Sigma Gamma1 Gamma2 Gamma3 x e1 e2 t1 t2 eps1 eps2 slot
        h1 h2 ih1 ih2 =>
        intro Delta hEq hSupp
        cases hEq
        rcases hSupp with ⟨hSupp1, hSupp2⟩
        exact HasType.letBind Delta Sigma Gamma1 Gamma2 Gamma3
          x e1 e2 t1 t2 eps1 eps2 slot
          (ih1 Delta rfl hSupp1)
          (ih2 Delta rfl hSupp2)
    | copy Delta0 Sigma Gamma1 Gamma2 e ds eps hBody ih =>
        intro Delta hEq hSupp
        cases hEq
        exact HasType.copy Delta Sigma Gamma1 Gamma2 e ds eps
          (ih Delta rfl hSupp)
    | letpair Delta0 Sigma Gamma1 Gamma2 Gamma3 x z e1 e2 t1 t2 t eps1 eps2 slotX slotY
        h1 h2 ih1 ih2 =>
        intro Delta hEq hSupp
        cases hEq
        rcases hSupp with ⟨hSupp1, hSupp2⟩
        exact HasType.letpair Delta Sigma Gamma1 Gamma2 Gamma3
          x z e1 e2 t1 t2 t eps1 eps2 slotX slotY
          (ih1 Delta rfl hSupp1)
          (ih2 Delta rfl hSupp2)
    | tpair Delta0 Sigma Gamma1 Gamma2 Gamma3 e1 e2 t1 t2 eps1 eps2
        h1 h2 ih1 ih2 =>
        intro Delta hEq hSupp
        cases hEq
        rcases hSupp with ⟨hSupp1, hSupp2⟩
        exact HasType.tpair Delta Sigma Gamma1 Gamma2 Gamma3
          e1 e2 t1 t2 eps1 eps2
          (ih1 Delta rfl hSupp1)
          (ih2 Delta rfl hSupp2)
    | fst Delta0 Sigma Gamma1 Gamma2 e t1 t2 eps hBody ih =>
        intro Delta hEq hSupp
        cases hEq
        exact HasType.fst Delta Sigma Gamma1 Gamma2 e t1 t2 eps
          (ih Delta rfl hSupp)
    | snd Delta0 Sigma Gamma1 Gamma2 e t1 t2 eps hBody ih =>
        intro Delta hEq hSupp
        cases hEq
        exact HasType.snd Delta Sigma Gamma1 Gamma2 e t1 t2 eps
          (ih Delta rfl hSupp)
    | const Delta0 Sigma Gamma c ds =>
        intro Delta hEq _hSupp
        cases hEq
        exact HasType.const Delta Sigma Gamma c ds
    | tadd Delta0 Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2
        h1 h2 ih1 ih2 =>
        intro Delta hEq hSupp
        cases hEq
        rcases hSupp with ⟨hSupp1, hSupp2⟩
        exact HasType.tadd Delta Sigma Gamma1 Gamma2 Gamma3
          e1 e2 ds eps1 eps2
          (ih1 Delta rfl hSupp1)
          (ih2 Delta rfl hSupp2)
    | tmul Delta0 Sigma Gamma1 Gamma2 Gamma3 e1 e2 ds eps1 eps2
        h1 h2 ih1 ih2 =>
        intro Delta hEq hSupp
        cases hEq
        rcases hSupp with ⟨hSupp1, hSupp2⟩
        exact HasType.tmul Delta Sigma Gamma1 Gamma2 Gamma3
          e1 e2 ds eps1 eps2
          (ih1 Delta rfl hSupp1)
          (ih2 Delta rfl hSupp2)
    | tsum Delta0 Sigma Gamma1 Gamma2 e ds d eps hBody hmem ih =>
        intro Delta hEq hSupp
        cases hEq
        exact HasType.tsum Delta Sigma Gamma1 Gamma2 e ds d eps
          (ih Delta rfl hSupp) hmem
    | texpand Delta0 Sigma Gamma1 Gamma2 e ds d eps hBody ih =>
        intro Delta hEq hSupp
        cases hEq
        exact HasType.texpand Delta Sigma Gamma1 Gamma2 e ds d eps
          (ih Delta rfl hSupp)
    | uniformLike Delta0 Sigma Gamma1 Gamma2 e ds lo hi eps hBody ih =>
        intro Delta hEq hSupp
        cases hSupp
    | perform Delta0 Sigma Gamma1 Gamma2 op e tArg tRet eps hBody hsig ih =>
        intro Delta hEq hSupp
        cases hEq
        rcases hSupp with ⟨_hop, hSuppBody⟩
        exact HasType.perform Delta Sigma Gamma1 Gamma2
          op e tArg tRet eps
          (ih Delta rfl hSuppBody) hsig
    | handle Delta0 Sigma Gamma1 Gamma2 Gamma3 body clauses t epsH epsB
        hBody hOpsIn hClsIn hCover hClauses ihBody ihClauses =>
        intro Delta hEq hSupp
        cases hEq
        rcases hSupp with ⟨hSuppBody, hSuppClauses⟩
        exact HasType.handle Delta Sigma Gamma1 Gamma2 Gamma3
          body clauses t epsH epsB
          (ihBody Delta rfl hSuppBody)
          hOpsIn hClsIn hCover
          (ihClauses Delta rfl hSuppClauses)
    | tgrad Delta0 Sigma Gamma x ds dsOut e eps slot hBody hsub hSuppBody hCtxSupp ih =>
        intro Delta hEq hSupp
        cases hSupp
    | tvmap Delta0 Sigma Gamma x t1 t2 e eps d slot hBody ih =>
        intro Delta hEq hSupp
        cases hSupp
    | loc Delta0 Sigma Gamma ell t hlook =>
        intro Delta hEq _hSupp
        cases hEq
        exact HasType.loc Delta Sigma Gamma ell t hlook
    | subEff Delta0 Sigma Gamma Gamma' e t eps eps' hBody hsub ih =>
        intro Delta hEq hSupp
        cases hEq
        exact HasType.subEff Delta Sigma Gamma Gamma' e t eps eps'
          (ih Delta rfl hSupp) hsub
    | nil Delta0 Sigma Gamma2 t epsR Delta hEq _hSupp =>
        cases hEq
        exact ClausesTyped.nil Delta Sigma Gamma2 t epsR
    | cons Delta0 Sigma Gamma2 Gamma3 t tArg tRet epsR op x k hb rest slotX slotK
        hMatch hBody hRest ihBody ihRest Delta hEq hSupp =>
        cases hEq
        rcases hSupp with ⟨hSuppBody, hSuppRest⟩
        exact ClausesTyped.cons Delta Sigma Gamma2 Gamma3
          t tArg tRet epsR op x k hb rest slotX slotK
          hMatch
          (ihBody Delta rfl hSuppBody)
          (ihRest Delta rfl hSuppRest)
  exact hMain h Delta rfl hSupp

private theorem clausesTyped_drop_diff_supported
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma2 Gamma3 : LinearCtx}
    {t : Typ} {epsR : EffectRow}
    {cls : List (EffectLabel × String × String × Term)}
    (h : ClausesTyped (Capability.diff :: Delta) Sigma Gamma2 Gamma3 t epsR cls)
    (hSupp : AdjointSupportedClauses cls) :
    ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR cls := by
  match h with
  | ClausesTyped.nil _ _ Gamma2 t epsR =>
      exact ClausesTyped.nil Delta Sigma Gamma2 t epsR
  | ClausesTyped.cons _ _ Gamma2 Gamma3 t tArg tRet epsR op x k hb rest slotX slotK
      hMatch hBody hRest =>
      rcases hSupp with ⟨hSuppBody, hSuppRest⟩
      exact ClausesTyped.cons Delta Sigma Gamma2 Gamma3
        t tArg tRet epsR op x k hb rest slotX slotK hMatch
        (hasType_drop_diff_supported hBody hSuppBody)
        (clausesTyped_drop_diff_supported hRest hSuppRest)

private theorem hasType_suffix_weaken_exact
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma e t eps Gamma')
    (suffix : LinearCtx)
    (hSuffixFreshCtx : ∀ y, y ∈ linearCtxDom suffix → y ∉ linearCtxDom Gamma)
    (hSuffixNoDup : NoDupNames suffix)
    (hFreshTerm : ∀ y, y ∈ linearCtxDom suffix → freshInTerm y e) :
    HasType Delta Sigma (Gamma ++ suffix) e t eps (Gamma' ++ suffix) := by
  induction suffix generalizing Gamma Gamma' with
  | nil =>
      simpa using h
  | cons hd rest ih =>
      rcases hd with ⟨y, slotY⟩
      have hNoDupDom : List.Nodup (y :: linearCtxDom rest) := by
        simpa [NoDupNames, linearCtxDom] using hSuffixNoDup
      have hyRest : y ∉ linearCtxDom rest := (List.nodup_cons.1 hNoDupDom).1
      have hRestNoDup : NoDupNames rest := by
        simpa [NoDupNames, linearCtxDom] using (List.nodup_cons.1 hNoDupDom).2
      have hyGamma : y ∉ linearCtxDom Gamma := by
        exact hSuffixFreshCtx y (by simp [linearCtxDom])
      have hyFresh : freshInTerm y e := hFreshTerm y (by simp [linearCtxDom])
      have hHead :
          HasType Delta Sigma
            (Gamma ++ [(y, slotY)])
            e t eps
            (Gamma' ++ [(y, slotY)]) := by
        have hWeak :=
          weakening_beforeSuffix Delta Sigma y slotY h 0
            (by simp)
            hyGamma
            hyFresh
        simpa [insertBeforeSuffix_eq_split] using hWeak
      have hRestFreshCtx :
          ∀ z, z ∈ linearCtxDom rest → z ∉ linearCtxDom (Gamma ++ [(y, slotY)]) := by
        intro z hz
        have hzSuffix : z ∈ linearCtxDom ((y, slotY) :: rest) := by
          simpa [linearCtxDom] using List.mem_cons_of_mem y hz
        have hzGamma : z ∉ linearCtxDom Gamma := by
          exact hSuffixFreshCtx z hzSuffix
        have hzy : z ≠ y := by
          intro hEq
          subst z
          exact hyRest hz
        intro hzIn
        have hzSplit : z ∈ linearCtxDom Gamma ∨ z = y := by
          simpa [linearCtxDom] using hzIn
        cases hzSplit with
        | inl hzGammaIn => exact hzGamma hzGammaIn
        | inr hEq => exact hzy hEq
      have hRestFreshTerm :
          ∀ z, z ∈ linearCtxDom rest → freshInTerm z e := by
        intro z hz
        have hzSuffix : z ∈ linearCtxDom ((y, slotY) :: rest) := by
          simpa [linearCtxDom] using List.mem_cons_of_mem y hz
        exact hFreshTerm z hzSuffix
      simpa [List.append_assoc] using
        (ih
          (Gamma := Gamma ++ [(y, slotY)])
          (Gamma' := Gamma' ++ [(y, slotY)])
          hHead
          hRestFreshCtx
          hRestNoDup
          hRestFreshTerm)

private theorem clausesTyped_suffix_weaken_exact
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma2 Gamma3 : LinearCtx}
    {t : Typ} {epsR : EffectRow}
    {cls : List (EffectLabel × String × String × Term)}
    (h : ClausesTyped Delta Sigma Gamma2 Gamma3 t epsR cls)
    (suffix : LinearCtx)
    (hSuffixFreshCtx : ∀ y, y ∈ linearCtxDom suffix → y ∉ linearCtxDom Gamma2)
    (hSuffixNoDup : NoDupNames suffix)
    (hFreshClauses :
      ∀ y, y ∈ linearCtxDom suffix →
        ∀ op x k hb, (op, x, k, hb) ∈ cls →
          y ≠ x ∧ y ≠ k ∧ freshInTerm y hb) :
    ClausesTyped Delta Sigma (Gamma2 ++ suffix) (Gamma3 ++ suffix) t epsR cls := by
  induction suffix generalizing Gamma2 Gamma3 with
  | nil =>
      simpa using h
  | cons hd rest ih =>
      rcases hd with ⟨y, slotY⟩
      have hNoDupDom : List.Nodup (y :: linearCtxDom rest) := by
        simpa [NoDupNames, linearCtxDom] using hSuffixNoDup
      have hyRest : y ∉ linearCtxDom rest := (List.nodup_cons.1 hNoDupDom).1
      have hRestNoDup : NoDupNames rest := by
        simpa [NoDupNames, linearCtxDom] using (List.nodup_cons.1 hNoDupDom).2
      have hyGamma : y ∉ linearCtxDom Gamma2 := by
        exact hSuffixFreshCtx y (by simp [linearCtxDom])
      have hHead :
          ClausesTyped Delta Sigma
            (Gamma2 ++ [(y, slotY)])
            (Gamma3 ++ [(y, slotY)])
            t epsR cls := by
        have hWeak :=
          weakening_beforeSuffix_clauses Delta Sigma y slotY h 0
            (by simp)
            hyGamma
            (by
              intro op x k hb hmem
              exact hFreshClauses y (by simp [linearCtxDom]) op x k hb hmem)
        simpa [insertBeforeSuffix_eq_split] using hWeak
      have hRestFreshCtx :
          ∀ z, z ∈ linearCtxDom rest → z ∉ linearCtxDom (Gamma2 ++ [(y, slotY)]) := by
        intro z hz
        have hzSuffix : z ∈ linearCtxDom ((y, slotY) :: rest) := by
          simpa [linearCtxDom] using List.mem_cons_of_mem y hz
        have hzGamma : z ∉ linearCtxDom Gamma2 := by
          exact hSuffixFreshCtx z hzSuffix
        have hzy : z ≠ y := by
          intro hEq
          subst z
          exact hyRest hz
        intro hzIn
        have hzSplit : z ∈ linearCtxDom Gamma2 ∨ z = y := by
          simpa [linearCtxDom] using hzIn
        cases hzSplit with
        | inl hzGammaIn => exact hzGamma hzGammaIn
        | inr hEq => exact hzy hEq
      have hRestFreshClauses :
          ∀ z, z ∈ linearCtxDom rest →
            ∀ op x k hb, (op, x, k, hb) ∈ cls →
              z ≠ x ∧ z ≠ k ∧ freshInTerm z hb := by
        intro z hz op x k hb hmem
        have hzSuffix : z ∈ linearCtxDom ((y, slotY) :: rest) := by
          simpa [linearCtxDom] using List.mem_cons_of_mem y hz
        exact hFreshClauses z hzSuffix op x k hb hmem
      simpa [List.append_assoc] using
        (ih
          (Gamma2 := Gamma2 ++ [(y, slotY)])
          (Gamma3 := Gamma3 ++ [(y, slotY)])
          hHead
          hRestFreshCtx
          hRestNoDup
          hRestFreshClauses)

end

private theorem maxStringLength_append_mono_right
    (pre suffix : List String) :
    maxStringLength suffix ≤ maxStringLength (pre ++ suffix) := by
  induction pre with
  | nil =>
      simp
  | cons hd tl ih =>
      simp [maxStringLength]
      exact Nat.le_trans ih (Nat.le_max_right _ _)

private theorem freshName_length_gt_used
    {used : List String} {base : String} {m : Nat}
    (hgt : maxStringLength used < m) :
    maxStringLength used < (freshName base m).toList.length := by
  rw [freshName_toList, List.length_append, List.length_cons, List.length_replicate]
  omega

private theorem freshName_not_mem_of_length_bound
    {used : List String} {base : String} {m : Nat}
    (hgt : maxStringLength used < m) :
    freshName base m ∉ used := by
  intro hmem
  have hle :
      (freshName base m).toList.length ≤ maxStringLength used :=
    mem_maxStringLength (used := used) (s := freshName base m) hmem
  have hlen :
      maxStringLength used < (freshName base m).toList.length :=
    freshName_length_gt_used (used := used) hgt
  exact Nat.not_lt_of_ge hle hlen

private theorem freshName_freshInTerm_of_length_bound
    {base : String} {m : Nat} {e : Term}
    (hgt : maxStringLength (freeVars e ++ boundVars e) < m) :
    freshInTerm (freshName base m) e := by
  refine ⟨?_, ?_⟩
  · intro hmem
    exact
      (freshName_not_mem_of_length_bound
        (used := freeVars e ++ boundVars e) (base := base) (m := m) hgt)
        (List.mem_append_left _ hmem)
  · intro hmem
    exact
      (freshName_not_mem_of_length_bound
        (used := freeVars e ++ boundVars e) (base := base) (m := m) hgt)
        (List.mem_append_right _ hmem)

theorem adjointTermFresh_of_length_bound
    {n : Nat} {e : Term}
    (hgt : maxStringLength (freeVars e ++ boundVars e) < n) :
    AdjointTermFresh n e := by
  intro m hm base _hb
  exact freshName_freshInTerm_of_length_bound (e := e)
    (Nat.lt_of_lt_of_le hgt hm)

theorem adjointTermFresh_of_gradAdjointCounter
    (x gs : String) (e : Term) :
    AdjointTermFresh (gradAdjointCounter x gs e) e := by
  apply adjointTermFresh_of_length_bound
  unfold gradAdjointCounter
  have hle :
      maxStringLength (freeVars e ++ boundVars e) ≤
        maxStringLength (gs :: gradPrimalName x e :: x :: (freeVars e ++ boundVars e)) := by
    simpa [List.append_assoc] using
      (maxStringLength_append_mono_right
        [gs, gradPrimalName x e, x]
        (freeVars e ++ boundVars e))
  exact Nat.lt_of_le_of_lt hle (Nat.lt_succ_self _)

-- linearCtx_filter_fresh_eq and linearCtx_filter_fresh_two_eq
-- deleted: obsolete under tombstone-style contexts (no more .filter
-- on output contexts).

/-- `SubEffRow` from `union eps [accum, accum]`-shaped rows back to
    `union [accum] eps`. Elementwise membership. -/
private theorem subEff_accum_accum_to_accum (epsSeed : EffectRow) :
    SubEffRow
      (EffectRow.union epsSeed
        (EffectRow.union
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))))
      (EffectRow.union [EffectLabel.accum] epsSeed) := by
  -- Both sides are sets equal to `{accum} ∪ epsSeed`. Reduce to a
  -- direct case split on whether `op = accum` or `op ∈ epsSeed`.
  intro op hop
  have hcases : op = EffectLabel.accum ∨ op ∈ epsSeed := by
    simp [EffectRow.union, List.mem_append, List.mem_filter] at hop
    rcases hop with hL | hR
    · exact Or.inr hL
    · exact Or.inl hR.1
  -- Goal: op ∈ union [accum] epsSeed = [accum] ++ epsSeed.filter ...
  show op ∈ EffectRow.union [EffectLabel.accum] epsSeed
  -- It suffices to show `op ∈ [accum] ∨ op ∈ epsSeed`.
  have : op ∈ [EffectLabel.accum] ∨ op ∈ epsSeed := by
    rcases hcases with h | h
    · exact Or.inl (by subst h; simp)
    · exact Or.inr h
  unfold EffectRow.union
  rcases this with hL | hR
  · exact List.mem_append_left _ hL
  · by_cases hmem : op ∈ ([EffectLabel.accum] : EffectRow)
    · exact List.mem_append_left _ hmem
    · apply List.mem_append_right
      refine List.mem_filter.mpr ⟨hR, ?_⟩
      have hop_ne : op ≠ EffectLabel.accum := by
        intro heq
        exact hmem (by subst heq; exact List.mem_singleton.mpr rfl)
      cases op <;> first | rfl | exact absurd rfl hop_ne

/-- SubEffRow for the outer `copy gSeed` union with the inner add body. -/
private theorem subEff_letpair_add (epsSeed : EffectRow) :
    SubEffRow
      (EffectRow.union epsSeed
        (EffectRow.union
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))))
      (EffectRow.union [EffectLabel.accum] epsSeed) :=
  subEff_accum_accum_to_accum epsSeed

private theorem subEff_accum_twice :
    SubEffRow
      (EffectRow.union
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow)) := by
  simpa [EffectRow.union] using
    (subEff_letpair_add ([] : EffectRow))

/-- SubEffRow for the leaf skeleton
    `let _ = gSeed in perform accum unit`. -/
private theorem subEff_seed_accum (epsSeed : EffectRow) :
    SubEffRow
      (EffectRow.union epsSeed
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
      (EffectRow.union [EffectLabel.accum] epsSeed) := by
  intro op hop
  unfold EffectRow.union at hop ⊢
  simp only [List.mem_append, List.mem_filter, List.mem_singleton] at hop ⊢
  rcases hop with hop | hop
  · by_cases hEq : op = EffectLabel.accum
    · left
      exact hEq
    · right
      exact ⟨hop, by simp [List.mem_singleton, hEq]⟩
  · left
    simpa using hop.1

/-- `adjoint_typed_aux` instantiated with a pure seed variable emits
    at least `accum`, so the row can always be widened to the outer
    grad target row `eps ∪ {accum}`. -/
private theorem subEff_accum_into_grad (eps : EffectRow) :
    SubEffRow
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      (EffectRow.union eps [EffectLabel.accum]) := by
  intro op hop
  have hopAccum : op = EffectLabel.accum := by
    simp [EffectRow.union] at hop
    simpa using hop
  subst hopAccum
  unfold EffectRow.union
  by_cases hmem : EffectLabel.accum ∈ eps
  · exact List.mem_append_left _ hmem
  · apply List.mem_append_right
    simp [hmem]

/-- Generic typing lemma for the unit-valued `adjointLeaf` skeleton. It
    linearly consumes an arbitrary supported cotangent seed, then emits
    `perform accum unit`, so the overall result is `unit` with exactly
    the incoming seed effects plus `accum`. -/
private theorem adjointLeaf_typed
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma_s Gamma_s' : LinearCtx)
    (seedTy : Typ) (epsSeed : EffectRow) (n : Nat) (gSeed : Term)
    (h_seed : HasType Delta Sigma Gamma_s gSeed seedTy epsSeed Gamma_s') :
    HasType Delta Sigma Gamma_s
      (adjointLeaf gSeed n)
      Typ.unit
      (EffectRow.union [EffectLabel.accum] epsSeed)
      Gamma_s' := by
  have hAccumSig : OpSigMatch EffectLabel.accum Typ.unit Typ.unit := by
    simp [OpSigMatch, opArgType, opRetType]
  have hPerform :
      HasType Delta Sigma
        (Gamma_s' ++ [(freshName "adjA" n, some seedTy)])
        (Term.perform EffectLabel.accum Term.unit)
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma_s' ++ [(freshName "adjA" n, some seedTy)]) := by
    exact HasType.perform Delta Sigma
      (Gamma_s' ++ [(freshName "adjA" n, some seedTy)])
      (Gamma_s' ++ [(freshName "adjA" n, some seedTy)])
      EffectLabel.accum Term.unit Typ.unit Typ.unit []
      (HasType.unit Delta Sigma _)
      hAccumSig
  have hLet :
      HasType Delta Sigma Gamma_s
        (adjointLeaf gSeed n)
        Typ.unit
        (EffectRow.union epsSeed
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
        Gamma_s' := by
    simpa [adjointLeaf] using
      (HasType.letBind Delta Sigma Gamma_s Gamma_s' Gamma_s'
        (freshName "adjA" n) gSeed
        (Term.perform EffectLabel.accum Term.unit)
        seedTy Typ.unit epsSeed
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (some seedTy) h_seed hPerform)
  exact HasType.subEff Delta Sigma Gamma_s Gamma_s'
    (adjointLeaf gSeed n) Typ.unit
    (EffectRow.union epsSeed
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
    (EffectRow.union [EffectLabel.accum] epsSeed)
    hLet
    (subEff_seed_accum epsSeed)

private theorem adjointTyped_seq_typed
    (Delta : CapCtx) (Sigma : StoreTyp)
    (Gamma1 Gamma2 Gamma3 : LinearCtx)
    (adjA : String) (e1 e2 : Term) (slotA : Option Typ)
    (h1 : HasType Delta Sigma Gamma1 e1 Typ.unit
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow)) Gamma2)
    (h2 : HasType Delta Sigma
      (Gamma2 ++ [(adjA, some Typ.unit)]) e2 Typ.unit
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      (Gamma3 ++ [(adjA, slotA)])) :
    HasType Delta Sigma Gamma1
      (Term.letBind adjA e1 e2)
      Typ.unit
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      Gamma3 := by
  have hLet :
      HasType Delta Sigma Gamma1
        (Term.letBind adjA e1 e2)
        Typ.unit
        (EffectRow.union
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
        Gamma3 := by
    simpa using
      (HasType.letBind Delta Sigma Gamma1 Gamma2 Gamma3
        adjA e1 e2 Typ.unit Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        slotA h1 h2)
  exact HasType.subEff Delta Sigma Gamma1 Gamma3
    (Term.letBind adjA e1 e2) Typ.unit
    (EffectRow.union
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow)))
    (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
    hLet
    subEff_accum_twice

/-- Zero cotangent seeds are well-typed in any linear context at the
    structural cotangent type chosen by `AdjointTransform`. -/
private theorem zeroCotangent_typed
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx) :
    ∀ t, HasType Delta Sigma Gamma (zeroCotangent t) (cotangentType t) [] Gamma
  | Typ.tensor ds =>
      by simpa [zeroCotangent, cotangentType] using
        (HasType.const Delta Sigma Gamma 0 ds)
  | Typ.pair t1 t2 =>
      by
        have h1 := zeroCotangent_typed Delta Sigma Gamma t1
        have h2 := zeroCotangent_typed Delta Sigma Gamma t2
        simpa [zeroCotangent, cotangentType] using
          (HasType.tpair Delta Sigma Gamma Gamma Gamma
            (zeroCotangent t1) (zeroCotangent t2)
            (cotangentType t1) (cotangentType t2) [] [] h1 h2)
  | Typ.unit =>
      by simpa [zeroCotangent, cotangentType] using
        (HasType.unit Delta Sigma Gamma)
  | Typ.arrow _ _ _ =>
      by simpa [zeroCotangent, cotangentType] using
        (HasType.unit Delta Sigma Gamma)
  | Typ.tyVar _ =>
      by simpa [zeroCotangent, cotangentType] using
        (HasType.unit Delta Sigma Gamma)

/-- CPS typing lemma for `splitCotangentSeedFrom` on pure seeds. This
    is the internal bridge needed by the typed clause transform:
    callers supply the typing proof for the continuation at the split
    point, and the theorem reconstructs the surrounding seed split. -/
private theorem splitCotangentSeedFrom_typed
    (Delta : CapCtx) (Sigma : StoreTyp)
    (Gamma_s Gamma_s' : LinearCtx)
    (t : Typ) (gSeed : Term) (n : Nat)
    (k : Nat → Term → Term → Term)
    (epsK : EffectRow) (Gamma_out : LinearCtx)
    (h_seed : HasType Delta Sigma Gamma_s gSeed (cotangentType t) [] Gamma_s')
    (h_k :
      match t with
      | Typ.tensor ds =>
          ∃ slotA slotB,
            HasType Delta Sigma
              (Gamma_s' ++
                [(freshName "gA" n, some (Typ.tensor ds)),
                 (freshName "gB" n, some (Typ.tensor ds))])
              (k (n + 2)
                (Term.var (freshName "gA" n))
                (Term.var (freshName "gB" n)))
              Typ.unit
              epsK
              (Gamma_out ++
                [(freshName "gA" n, slotA),
                 (freshName "gB" n, slotB)])
      | Typ.pair t1 t2 =>
          ∃ slotA slotB,
            HasType Delta Sigma
              (Gamma_s' ++
                [(freshName "gA" n, some (cotangentType t1)),
                 (freshName "gB" n, some (cotangentType t2))])
              (splitCotangentSeedFrom t1 (Term.var (freshName "gA" n)) (n + 2)
                (fun n' gA1 gA2 =>
                  splitCotangentSeedFrom t2 (Term.var (freshName "gB" n)) n'
                    (fun n'' gB1 gB2 =>
                      k n''
                        (Term.pair gA1 gB1)
                        (Term.pair gA2 gB2))))
              Typ.unit
              epsK
              (Gamma_out ++
                [(freshName "gA" n, slotA),
                 (freshName "gB" n, slotB)])
      | Typ.unit =>
          ∃ slotA,
            HasType Delta Sigma
              (Gamma_s' ++ [(freshName "adjA" n, some Typ.unit)])
              (k (n + 1) Term.unit Term.unit)
              Typ.unit
              epsK
              (Gamma_out ++ [(freshName "adjA" n, slotA)])
      | Typ.arrow _ _ _ =>
          ∃ slotA,
            HasType Delta Sigma
              (Gamma_s' ++ [(freshName "adjA" n, some Typ.unit)])
              (k (n + 1) Term.unit Term.unit)
              Typ.unit
              epsK
              (Gamma_out ++ [(freshName "adjA" n, slotA)])
      | Typ.tyVar _ =>
          ∃ slotA,
            HasType Delta Sigma
              (Gamma_s' ++ [(freshName "adjA" n, some Typ.unit)])
              (k (n + 1) Term.unit Term.unit)
              Typ.unit
              epsK
              (Gamma_out ++ [(freshName "adjA" n, slotA)])) :
    HasType Delta Sigma Gamma_s
      (splitCotangentSeedFrom t gSeed n k)
      Typ.unit
      epsK
      Gamma_out := by
  cases t with
  | tensor ds =>
      rcases h_k with ⟨slotA, slotB, hBody⟩
      simpa [splitCotangentSeedFrom] using
        (HasType.letpair Delta Sigma Gamma_s Gamma_s' Gamma_out
          (freshName "gA" n) (freshName "gB" n)
          (Term.copy gSeed)
          (k (n + 2)
            (Term.var (freshName "gA" n))
            (Term.var (freshName "gB" n)))
          (Typ.tensor ds) (Typ.tensor ds) Typ.unit
          []
          epsK
          slotA slotB
          (HasType.copy Delta Sigma Gamma_s Gamma_s' gSeed ds [] h_seed)
          hBody)
  | pair t1 t2 =>
      rcases h_k with ⟨slotA, slotB, hBody⟩
      simpa [splitCotangentSeedFrom, cotangentType] using
        (HasType.letpair Delta Sigma Gamma_s Gamma_s' Gamma_out
          (freshName "gA" n) (freshName "gB" n)
          gSeed
          (splitCotangentSeedFrom t1 (Term.var (freshName "gA" n)) (n + 2)
            (fun n' gA1 gA2 =>
              splitCotangentSeedFrom t2 (Term.var (freshName "gB" n)) n'
                (fun n'' gB1 gB2 =>
                  k n''
                    (Term.pair gA1 gB1)
                    (Term.pair gA2 gB2))))
          (cotangentType t1) (cotangentType t2) Typ.unit
          []
          epsK
          slotA slotB
          h_seed
          hBody)
  | unit =>
      rcases h_k with ⟨slotA, hBody⟩
      simpa [splitCotangentSeedFrom] using
        (HasType.letBind Delta Sigma Gamma_s Gamma_s' Gamma_out
          (freshName "adjA" n) gSeed
          (k (n + 1) Term.unit Term.unit)
          Typ.unit Typ.unit [] epsK slotA
          h_seed hBody)
  | arrow _ _ _ =>
      rcases h_k with ⟨slotA, hBody⟩
      simpa [splitCotangentSeedFrom] using
        (HasType.letBind Delta Sigma Gamma_s Gamma_s' Gamma_out
          (freshName "adjA" n) gSeed
          (k (n + 1) Term.unit Term.unit)
          Typ.unit Typ.unit [] epsK slotA
          h_seed hBody)
  | tyVar _ =>
      rcases h_k with ⟨slotA, hBody⟩
      simpa [splitCotangentSeedFrom] using
        (HasType.letBind Delta Sigma Gamma_s Gamma_s' Gamma_out
          (freshName "adjA" n) gSeed
          (k (n + 1) Term.unit Term.unit)
          Typ.unit Typ.unit [] epsK slotA
          h_seed hBody)

private theorem splitCotangentSeedFrom_slot_typed
    (Delta : CapCtx) (Sigma : StoreTyp)
    {t : Typ}
    {Gamma suffixIn suffixOut : LinearCtx} {x : String}
    {slotIn : Option Typ}
    (gSeed : Term) (n : Nat)
    (k : Nat → Term → Term → Term)
    (epsK : EffectRow)
    (h_seed :
      HasType Delta Sigma
        (Gamma ++ [(x, slotIn)] ++ suffixIn)
        gSeed
        (cotangentType t)
        []
        (Gamma ++ [(x, slotIn)] ++ suffixOut))
    (h_k :
      match t with
      | Typ.tensor ds =>
          ∃ slotOut slotA slotB,
            (slotOut = none ∨ slotOut = slotIn) ∧
            HasType Delta Sigma
              (Gamma ++ [(x, slotIn)] ++ suffixOut ++
                [(freshName "gA" n, some (Typ.tensor ds)),
                 (freshName "gB" n, some (Typ.tensor ds))])
              (k (n + 2)
                (Term.var (freshName "gA" n))
                (Term.var (freshName "gB" n)))
              Typ.unit
              epsK
              (Gamma ++ [(x, slotOut)] ++ suffixOut ++
                [(freshName "gA" n, slotA),
                 (freshName "gB" n, slotB)])
      | Typ.pair t1 t2 =>
          ∃ slotOut slotA slotB,
            (slotOut = none ∨ slotOut = slotIn) ∧
            HasType Delta Sigma
              (Gamma ++ [(x, slotIn)] ++ suffixOut ++
                [(freshName "gA" n, some (cotangentType t1)),
                 (freshName "gB" n, some (cotangentType t2))])
              (splitCotangentSeedFrom t1 (Term.var (freshName "gA" n)) (n + 2)
                (fun n' gA1 gA2 =>
                  splitCotangentSeedFrom t2 (Term.var (freshName "gB" n)) n'
                    (fun n'' gB1 gB2 =>
                      k n''
                        (Term.pair gA1 gB1)
                        (Term.pair gA2 gB2))))
              Typ.unit
              epsK
              (Gamma ++ [(x, slotOut)] ++ suffixOut ++
                [(freshName "gA" n, slotA),
                 (freshName "gB" n, slotB)])
      | Typ.unit =>
          ∃ slotOut slotA,
            (slotOut = none ∨ slotOut = slotIn) ∧
            HasType Delta Sigma
              (Gamma ++ [(x, slotIn)] ++ suffixOut ++
                [(freshName "adjA" n, some Typ.unit)])
              (k (n + 1) Term.unit Term.unit)
              Typ.unit
              epsK
              (Gamma ++ [(x, slotOut)] ++ suffixOut ++
                [(freshName "adjA" n, slotA)])
      | Typ.arrow _ _ _ =>
          ∃ slotOut slotA,
            (slotOut = none ∨ slotOut = slotIn) ∧
            HasType Delta Sigma
              (Gamma ++ [(x, slotIn)] ++ suffixOut ++
                [(freshName "adjA" n, some Typ.unit)])
              (k (n + 1) Term.unit Term.unit)
              Typ.unit
              epsK
              (Gamma ++ [(x, slotOut)] ++ suffixOut ++
                [(freshName "adjA" n, slotA)])
      | Typ.tyVar _ =>
          ∃ slotOut slotA,
            (slotOut = none ∨ slotOut = slotIn) ∧
            HasType Delta Sigma
              (Gamma ++ [(x, slotIn)] ++ suffixOut ++
                [(freshName "adjA" n, some Typ.unit)])
              (k (n + 1) Term.unit Term.unit)
              Typ.unit
              epsK
              (Gamma ++ [(x, slotOut)] ++ suffixOut ++
                [(freshName "adjA" n, slotA)])) :
    ∃ slotOut,
      (slotOut = none ∨ slotOut = slotIn) ∧
      HasType Delta Sigma
        (Gamma ++ [(x, slotIn)] ++ suffixIn)
        (splitCotangentSeedFrom t gSeed n k)
        Typ.unit
        epsK
        (Gamma ++ [(x, slotOut)] ++ suffixOut) := by
  cases t with
  | tensor ds =>
      rcases h_k with ⟨slotOut, slotA, slotB, hslotOut, hBody⟩
      refine ⟨slotOut, hslotOut, ?_⟩
      simpa [splitCotangentSeedFrom] using
        (HasType.letpair Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          (Gamma ++ [(x, slotOut)] ++ suffixOut)
          (freshName "gA" n) (freshName "gB" n)
          (Term.copy gSeed)
          (k (n + 2)
            (Term.var (freshName "gA" n))
            (Term.var (freshName "gB" n)))
          (Typ.tensor ds) (Typ.tensor ds) Typ.unit
          []
          epsK
          slotA slotB
          (HasType.copy Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (Gamma ++ [(x, slotIn)] ++ suffixOut)
            gSeed
            ds
            []
            h_seed)
          hBody)
  | pair t1 t2 =>
      rcases h_k with ⟨slotOut, slotA, slotB, hslotOut, hBody⟩
      refine ⟨slotOut, hslotOut, ?_⟩
      simpa [splitCotangentSeedFrom, cotangentType] using
        (HasType.letpair Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          (Gamma ++ [(x, slotOut)] ++ suffixOut)
          (freshName "gA" n) (freshName "gB" n)
          gSeed
          (splitCotangentSeedFrom t1 (Term.var (freshName "gA" n)) (n + 2)
            (fun n' gA1 gA2 =>
              splitCotangentSeedFrom t2 (Term.var (freshName "gB" n)) n'
                (fun n'' gB1 gB2 =>
                  k n''
                    (Term.pair gA1 gB1)
                    (Term.pair gA2 gB2))))
          (cotangentType t1) (cotangentType t2) Typ.unit
          []
          epsK
          slotA slotB
          h_seed
          hBody)
  | unit =>
      rcases h_k with ⟨slotOut, slotA, hslotOut, hBody⟩
      refine ⟨slotOut, hslotOut, ?_⟩
      simpa [splitCotangentSeedFrom] using
        (HasType.letBind Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          (Gamma ++ [(x, slotOut)] ++ suffixOut)
          (freshName "adjA" n)
          gSeed
          (k (n + 1) Term.unit Term.unit)
          Typ.unit
          Typ.unit
          []
          epsK
          slotA
          h_seed
          hBody)
  | arrow tArg tRet epsT =>
      rcases h_k with ⟨slotOut, slotA, hslotOut, hBody⟩
      refine ⟨slotOut, hslotOut, ?_⟩
      simpa [splitCotangentSeedFrom] using
        (HasType.letBind Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          (Gamma ++ [(x, slotOut)] ++ suffixOut)
          (freshName "adjA" n)
          gSeed
          (k (n + 1) Term.unit Term.unit)
          Typ.unit
          Typ.unit
          []
          epsK
          slotA
          h_seed
          hBody)
  | tyVar alpha =>
      rcases h_k with ⟨slotOut, slotA, hslotOut, hBody⟩
      refine ⟨slotOut, hslotOut, ?_⟩
      simpa [splitCotangentSeedFrom] using
        (HasType.letBind Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          (Gamma ++ [(x, slotOut)] ++ suffixOut)
          (freshName "adjA" n)
          gSeed
          (k (n + 1) Term.unit Term.unit)
          Typ.unit
          Typ.unit
          []
          epsK
          slotA
          h_seed
          hBody)

/-- On quotient-based dimension multisets, re-inserting an erased member
    recovers the original multiset. This is the key shape fact for the
    typed `sum` adjoint branch. -/
private theorem ins_rem_eq_of_mem
    {ds : DimList} {d : Dim}
    (h : d ∈ ds) :
    ins (rem ds d) d = ds := by
  refine Quotient.inductionOn ds ?_ h
  intro l hmem
  have hmem' : d ∈ l := by
    simpa [DimList.mem] using hmem
  show Quotient.mk ListDimSetoid (d :: l.erase d) = Quotient.mk ListDimSetoid l
  exact Quotient.sound (s := ListDimSetoid) (List.perm_cons_erase hmem').symm

/-- Erasing the freshly inserted dimension cancels propositionally on the
    permutation quotient. This is the companion fact for `expand`. -/
private theorem rem_ins_eq
    {ds : DimList} {d : Dim} :
    rem (ins ds d) d = ds := by
  refine Quotient.inductionOn ds ?_
  intro l
  show Quotient.mk ListDimSetoid ((d :: l).erase d) = Quotient.mk ListDimSetoid l
  simp [rem, ins, DimList.erase, DimList.cons, List.erase_cons]

private theorem linearCtx_append2_aux
    (Gamma : LinearCtx)
    (x1 : String) (slot1 : Option Typ)
    (x2 : String) (slot2 : Option Typ) :
    ((Gamma ++ [(x1, slot1)]) ++ [(x2, slot2)]) =
      (Gamma ++ [(x1, slot1), (x2, slot2)]) := by
  simp [List.append_assoc]

private theorem linearCtx_append3_aux
    (Gamma : LinearCtx)
    (x1 : String) (slot1 : Option Typ)
    (x2 : String) (slot2 : Option Typ)
    (x3 : String) (slot3 : Option Typ) :
    (((Gamma ++ [(x1, slot1)]) ++ [(x2, slot2)]) ++ [(x3, slot3)]) =
      (Gamma ++ [(x1, slot1), (x2, slot2), (x3, slot3)]) := by
  simp [List.append_assoc]

private theorem linearCtx_suffix_pair
    (Gamma suffix : LinearCtx)
    (x1 : String) (slot1 : Option Typ)
    (x2 : String) (slot2 : Option Typ) :
    (Gamma ++ suffix ++ [(x1, slot1)] ++ [(x2, slot2)]) =
      (Gamma ++ (suffix ++ [(x1, slot1), (x2, slot2)])) := by
  simp [List.append_assoc]

private theorem adjointTyped_copy_typed
    (Delta : CapCtx) (Sigma : StoreTyp)
    {ds : DimList} {e : Term}
    (ih :
      ∀ {Gamma_s Gamma_s' suffix : LinearCtx} {x : String} {gSeed : Term} {n : Nat},
        HasType Delta Sigma
          (Gamma_s ++ suffix) gSeed (cotangentType (Typ.tensor ds)) []
          (Gamma_s' ++ suffix) →
        HasType Delta Sigma
          (Gamma_s ++ suffix)
          (adjointTypedFrom e (Typ.tensor ds) x gSeed n)
          Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix))
    {Gamma_s Gamma_s' suffix : LinearCtx} {x : String} {gSeed : Term} {n : Nat}
    (h_seed :
      HasType Delta Sigma
        (Gamma_s ++ suffix)
        gSeed
        (cotangentType (Typ.pair (Typ.tensor ds) (Typ.tensor ds)))
        []
        (Gamma_s' ++ suffix)) :
    HasType Delta Sigma
      (Gamma_s ++ suffix)
      (adjointTypedFrom (Term.copy e)
        (Typ.pair (Typ.tensor ds) (Typ.tensor ds)) x gSeed n)
      Typ.unit
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      (Gamma_s' ++ suffix) := by
  let gA := freshName "gA" n
  let gB := freshName "gB" n
  have hVarA :
      HasType Delta Sigma
        (Gamma_s' ++ suffix ++
          [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))])
        (Term.var gA)
        (Typ.tensor ds)
        []
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, some (Typ.tensor ds))]) := by
    simpa [gA, gB, List.append_assoc, linearCtx_append2_aux] using
      (HasType.var Delta Sigma
        (Gamma_s' ++ suffix)
        [(gB, some (Typ.tensor ds))]
        gA
        (Typ.tensor ds))
  have hVarB :
      HasType Delta Sigma
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, some (Typ.tensor ds))])
        (Term.var gB)
        (Typ.tensor ds)
        []
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, none)]) := by
    refine Eq.mp ?_
      (HasType.var Delta Sigma
        (((Gamma_s' ++ suffix) ++ [(gA, none)]))
        []
        gB
        (Typ.tensor ds))
    simp [gA, gB, List.append_assoc, linearCtx_suffix_pair]
  have hAddSeed :
      HasType Delta Sigma
        (Gamma_s' ++ suffix ++
          [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))])
        (Term.add (Term.var gA) (Term.var gB))
        (Typ.tensor ds)
        []
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, none)]) := by
    exact HasType.tadd Delta Sigma
      (Gamma_s' ++ suffix ++
        [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))])
      (Gamma_s' ++ suffix ++
        [(gA, none), (gB, some (Typ.tensor ds))])
      (Gamma_s' ++ suffix ++
        [(gA, none), (gB, none)])
      (Term.var gA) (Term.var gB) ds [] [] hVarA hVarB
  have hAdj :
      HasType Delta Sigma
        (Gamma_s' ++ suffix ++
          [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))])
        (adjointTypedFrom e (Typ.tensor ds) x
          (Term.add (Term.var gA) (Term.var gB)) (n + 3))
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, none)]) := by
    have hAddSeed0 :
        HasType Delta Sigma
          ((Gamma_s' ++ suffix ++
            [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))]) ++ [])
          (Term.add (Term.var gA) (Term.var gB))
          (Typ.tensor ds)
          []
          ((Gamma_s' ++ suffix ++
            [(gA, none), (gB, none)]) ++ []) := by
      simpa using hAddSeed
    refine Eq.mp ?_
      (ih
        (Gamma_s := (Gamma_s' ++ suffix ++
          [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))]))
        (Gamma_s' := (Gamma_s' ++ suffix ++
          [(gA, none), (gB, none)]))
        (suffix := ([] : LinearCtx))
        (x := x)
        (gSeed := Term.add (Term.var gA) (Term.var gB))
        (n := n + 3)
        (by simpa [cotangentType] using hAddSeed0))
    simp [cotangentType, gA, gB, List.append_assoc]
  simpa [adjointTypedFrom, gA, gB, List.append_assoc] using
    (HasType.letpair Delta Sigma
      (Gamma_s ++ suffix) (Gamma_s' ++ suffix) (Gamma_s' ++ suffix)
      gA gB
      gSeed
      (adjointTypedFrom e (Typ.tensor ds) x
        (Term.add (Term.var gA) (Term.var gB)) (n + 3))
      (Typ.tensor ds) (Typ.tensor ds) Typ.unit
      []
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      none none
      h_seed
      hAdj)

private theorem adjointTyped_pair_typed
    (Delta : CapCtx) (Sigma : StoreTyp)
    {t1 t2 : Typ} {e1 e2 : Term}
    (ih1 :
      ∀ {Gamma_s Gamma_s' suffix : LinearCtx} {x : String} {gSeed : Term} {n : Nat},
        HasType Delta Sigma
          (Gamma_s ++ suffix) gSeed (cotangentType t1) []
          (Gamma_s' ++ suffix) →
        HasType Delta Sigma
          (Gamma_s ++ suffix)
          (adjointTypedFrom e1 t1 x gSeed n)
          Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix))
    (ih2 :
      ∀ {Gamma_s Gamma_s' suffix : LinearCtx} {x : String} {gSeed : Term} {n : Nat},
        HasType Delta Sigma
          (Gamma_s ++ suffix) gSeed (cotangentType t2) []
          (Gamma_s' ++ suffix) →
        HasType Delta Sigma
          (Gamma_s ++ suffix)
          (adjointTypedFrom e2 t2 x gSeed n)
          Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix))
    {Gamma_s Gamma_s' suffix : LinearCtx} {x : String} {gSeed : Term} {n : Nat}
    (h_seed :
      HasType Delta Sigma
        (Gamma_s ++ suffix)
        gSeed
        (cotangentType (Typ.pair t1 t2))
        []
        (Gamma_s' ++ suffix)) :
    HasType Delta Sigma
      (Gamma_s ++ suffix)
      (adjointTypedFrom (Term.pair e1 e2) (Typ.pair t1 t2) x gSeed n)
      Typ.unit
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      (Gamma_s' ++ suffix) := by
  let gA := freshName "gA" n
  let gB := freshName "gB" n
  let adjA := freshName "adjA" n
  have hVarA :
      HasType Delta Sigma
        (Gamma_s' ++ suffix ++
          [(gA, some (cotangentType t1)), (gB, some (cotangentType t2))])
        (Term.var gA)
        (cotangentType t1)
        []
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, some (cotangentType t2))]) := by
    simpa [gA, gB, List.append_assoc, linearCtx_append2_aux] using
      (HasType.var Delta Sigma
        (Gamma_s' ++ suffix)
        [(gB, some (cotangentType t2))]
        gA
        (cotangentType t1))
  have hAdj1 :
      HasType Delta Sigma
        (Gamma_s' ++ suffix ++
          [(gA, some (cotangentType t1)), (gB, some (cotangentType t2))])
        (adjointTypedFrom e1 t1 x (Term.var gA) (n + 3))
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, some (cotangentType t2))]) := by
    have hVarA0 :
        HasType Delta Sigma
          ((Gamma_s' ++ suffix ++
            [(gA, some (cotangentType t1)), (gB, some (cotangentType t2))]) ++ [])
          (Term.var gA)
          (cotangentType t1)
          []
          ((Gamma_s' ++ suffix ++
            [(gA, none), (gB, some (cotangentType t2))]) ++ []) := by
      simpa using hVarA
    simpa [cotangentType, gA, gB] using
      (ih1
        (Gamma_s := (Gamma_s' ++ suffix ++
          [(gA, some (cotangentType t1)), (gB, some (cotangentType t2))]))
        (Gamma_s' := (Gamma_s' ++ suffix ++
          [(gA, none), (gB, some (cotangentType t2))]))
        (suffix := ([] : LinearCtx))
        (x := x)
        (gSeed := Term.var gA)
        (n := n + 3)
        hVarA0)
  have hVarB :
      HasType Delta Sigma
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, some (cotangentType t2))] ++
          [(adjA, some Typ.unit)])
        (Term.var gB)
        (cotangentType t2)
        []
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, none)] ++
          [(adjA, some Typ.unit)]) := by
    refine Eq.mp ?_
      (HasType.var Delta Sigma
        (((Gamma_s' ++ suffix) ++ [(gA, none)]))
        [(adjA, some Typ.unit)]
        gB
        (cotangentType t2))
    simp [gA, gB, adjA, List.append_assoc, linearCtx_suffix_pair, linearCtx_append3_aux]
  have hAdj2 :
      HasType Delta Sigma
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, some (cotangentType t2))] ++
          [(adjA, some Typ.unit)])
        (adjointTypedFrom e2 t2 x (Term.var gB) (n + 3))
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, none)] ++
          [(adjA, some Typ.unit)]) := by
    simpa [cotangentType, gA, gB, adjA] using
      (ih2
        (Gamma_s := (Gamma_s' ++ suffix ++
          [(gA, none), (gB, some (cotangentType t2))]))
        (Gamma_s' := (Gamma_s' ++ suffix ++
          [(gA, none), (gB, none)]))
        (suffix := [(adjA, some Typ.unit)])
        (x := x)
        (gSeed := Term.var gB)
        (n := n + 3)
        hVarB)
  have hSeq :
      HasType Delta Sigma
        (Gamma_s' ++ suffix ++
          [(gA, some (cotangentType t1)), (gB, some (cotangentType t2))])
        (Term.letBind adjA
          (adjointTypedFrom e1 t1 x (Term.var gA) (n + 3))
          (adjointTypedFrom e2 t2 x (Term.var gB) (n + 3)))
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma_s' ++ suffix ++
          [(gA, none), (gB, none)]) := by
    exact adjointTyped_seq_typed Delta Sigma
      (Gamma_s' ++ suffix ++
        [(gA, some (cotangentType t1)), (gB, some (cotangentType t2))])
      (Gamma_s' ++ suffix ++
        [(gA, none), (gB, some (cotangentType t2))])
      (Gamma_s' ++ suffix ++
        [(gA, none), (gB, none)])
      adjA
      (adjointTypedFrom e1 t1 x (Term.var gA) (n + 3))
      (adjointTypedFrom e2 t2 x (Term.var gB) (n + 3))
      (some Typ.unit)
      hAdj1
      hAdj2
  simpa [adjointTypedFrom, cotangentType, gA, gB, adjA, List.append_assoc] using
    (HasType.letpair Delta Sigma
      (Gamma_s ++ suffix) (Gamma_s' ++ suffix) (Gamma_s' ++ suffix)
      gA gB
      gSeed
      (Term.letBind adjA
        (adjointTypedFrom e1 t1 x (Term.var gA) (n + 3))
        (adjointTypedFrom e2 t2 x (Term.var gB) (n + 3)))
      (cotangentType t1) (cotangentType t2) Typ.unit
      []
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      none none
      h_seed
      hSeq)

mutual

/-- Structural type-shape witness for `adjointTypedFrom`. This records
    exactly the local result-type information the typed transform needs
    to avoid falling back to the legacy tensor-only surface. The only
    constructor carrying extra typing payload is `mul`, where the
    transform embeds the source operands directly via `copy e1` / `copy e2`.
    Everything else is syntax-directed. -/
inductive AdjointTypedShape (Delta : CapCtx) (Sigma : StoreTyp) : Typ → Term → Prop
  | var {t : Typ} {x : String} :
      AdjointTypedShape Delta Sigma t (Term.var x)
  | const {t : Typ} {v : Float} {ds : DimList} :
      AdjointTypedShape Delta Sigma t (Term.const v ds)
  | unit {t : Typ} :
      AdjointTypedShape Delta Sigma t Term.unit
  | loc {t : Typ} {ell : Nat} :
      AdjointTypedShape Delta Sigma t (Term.loc ell)
  | letBind {t : Typ} {x : String} {e1 e2 : Term} :
      AdjointTypedShape Delta Sigma t e2 →
      AdjointTypedShape Delta Sigma t (Term.letBind x e1 e2)
  | letpair {t : Typ} {x y : String} {e1 e2 : Term} :
      AdjointTypedShape Delta Sigma t e2 →
      AdjointTypedShape Delta Sigma t (Term.letpair x y e1 e2)
  | add {ds : DimList} {e1 e2 : Term} :
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e1 →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e2 →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) (Term.add e1 e2)
  | mul {ds : DimList} {e1 e2 : Term} :
      (∃ Γ1 Γ2 Γ3 eps1 eps2,
         HasType Delta Sigma Γ1 e1 (Typ.tensor ds) eps1 Γ2 ∧
         HasType Delta Sigma Γ2 e2 (Typ.tensor ds) eps2 Γ3) →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e1 →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e2 →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) (Term.mul e1 e2)
  | sum {ds : DimList} {d : Dim} {e : Term} :
      d ∈ ds →
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e →
      AdjointTypedShape Delta Sigma (Typ.tensor (rem ds d)) (Term.sum e d)
  | expand {ds : DimList} {d : Dim} {e : Term} :
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e →
      AdjointTypedShape Delta Sigma (Typ.tensor (ins ds d)) (Term.expand e d)
  | copy {ds : DimList} {e : Term} :
      AdjointTypedShape Delta Sigma (Typ.tensor ds) e →
      AdjointTypedShape Delta Sigma (Typ.pair (Typ.tensor ds) (Typ.tensor ds)) (Term.copy e)
  | pair {t1 t2 : Typ} {e1 e2 : Term} :
      AdjointTypedShape Delta Sigma t1 e1 →
      AdjointTypedShape Delta Sigma t2 e2 →
      AdjointTypedShape Delta Sigma (Typ.pair t1 t2) (Term.pair e1 e2)
  | fst {t1 t2 : Typ} {e : Term} :
      AdjointTypedShape Delta Sigma (Typ.pair t1 t2) e →
      AdjointTypedShape Delta Sigma t1 (Term.fst t2 e)
  | snd {t1 t2 : Typ} {e : Term} :
      AdjointTypedShape Delta Sigma (Typ.pair t1 t2) e →
      AdjointTypedShape Delta Sigma t2 (Term.snd t1 e)
  | handle {t : Typ} {epsH : EffectRow} {body : Term}
      {clauses : List (EffectLabel × String × String × Term)} :
      AdjointTypedShape Delta Sigma t body →
      AdjointTypedClausesShape Delta Sigma t clauses →
      AdjointTypedShape Delta Sigma t (Term.handle epsH body clauses)
  | perform {op : EffectLabel} {e : Term} :
      AdjointTypedShape Delta Sigma (opArgType op) e →
      AdjointTypedShape Delta Sigma Typ.unit (Term.perform op e)

/-- Clause companion to `AdjointTypedShape`. Each handler clause body is
    transformed against the handled result type. -/
inductive AdjointTypedClausesShape
    (Delta : CapCtx) (Sigma : StoreTyp) :
    Typ → List (EffectLabel × String × String × Term) → Prop
  | nil :
      AdjointTypedClausesShape Delta Sigma t []
  | cons {op : EffectLabel} {x k : String} {hb : Term}
      {rest : List (EffectLabel × String × String × Term)} :
      AdjointTypedShape Delta Sigma t hb →
      AdjointTypedClausesShape Delta Sigma t rest →
      AdjointTypedClausesShape Delta Sigma t ((op, x, k, hb) :: rest)

end

mutual

/-- Extract the typed-transform shape witness from a source typing
    derivation plus the supported-fragment premise required by `T-Grad`. -/
private theorem adjointTypedShape_of_typed
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 e t eps Gamma2)
    (hSupp : AdjointSupported e) :
    AdjointTypedShape Delta Sigma t e := by
  induction h using HasType.rec
    (motive_2 := fun Delta Sigma Gamma2 Gamma3 t epsR cls _hcls =>
      AdjointSupportedClauses cls → AdjointTypedClausesShape Delta Sigma t cls) with
  | var =>
      exact .var
  | unit =>
      exact .unit
  | fst _ _ _ _ e t1 t2 _ hBody ih =>
      exact .fst (ih hSupp)
  | snd _ _ _ _ e t1 t2 _ hBody ih =>
      exact .snd (ih hSupp)
  | const =>
      exact .const
  | tadd _ _ _ _ _ e1 e2 ds _ _ h1 h2 ih1 ih2 =>
      rcases hSupp with ⟨hSupp1, hSupp2⟩
      exact .add (ih1 hSupp1) (ih2 hSupp2)
  | tmul _ _ _ _ _ e1 e2 ds _ _ h1 h2 ih1 ih2 =>
      rcases hSupp with ⟨hSupp1, hSupp2⟩
      exact .mul ⟨_, _, _, _, _, h1, h2⟩ (ih1 hSupp1) (ih2 hSupp2)
  | tsum _ _ _ _ e ds d _ hBody hmem ih =>
      exact .sum hmem (ih hSupp)
  | texpand _ _ _ _ e ds d _ hBody ih =>
      exact .expand (ih hSupp)
  | uniformLike _ _ _ _ _ _ _ _ _ hBody ih =>
      cases hSupp
  | copy _ _ _ _ e ds _ hBody ih =>
      exact .copy (ih hSupp)
  | letBind _ _ _ _ _ _ e1 e2 _ _ _ _ _ h1 h2 ih1 ih2 =>
      rcases hSupp with ⟨_hSupp1, hSupp2⟩
      exact .letBind (ih2 hSupp2)
  | letpair _ _ _ _ _ _ _ e1 e2 _ _ _ _ _ _ _ h1 h2 ih1 ih2 =>
      rcases hSupp with ⟨_hSupp1, hSupp2⟩
      exact .letpair (ih2 hSupp2)
  | tpair _ _ _ _ _ e1 e2 _ _ _ _ h1 h2 ih1 ih2 =>
      rcases hSupp with ⟨hSupp1, hSupp2⟩
      exact .pair (ih1 hSupp1) (ih2 hSupp2)
  | loc =>
      exact .loc
  | perform _ _ _ _ op e tArg tRet _ hBody hMatch ih =>
      rcases hSupp with ⟨_hop, hSuppBody⟩
      have hArgEq : tArg = opArgType op := hMatch.1
      have hRetEq : tRet = Typ.unit := by
        cases op <;> simpa [OpSigMatch, opRetType] using hMatch.2
      cases hArgEq
      cases hRetEq
      exact .perform (ih hSuppBody)
  | handle _ _ _ _ _ body clauses _ _ _ hBody _ _ _ hClauses ihBody ihClauses =>
      rcases hSupp with ⟨hSuppBody, hSuppClauses⟩
      exact .handle (ihBody hSuppBody) (ihClauses hSuppClauses)
  | tgrad =>
      cases hSupp
  | tvmap =>
      cases hSupp
  | abs =>
      cases hSupp
  | app =>
      cases hSupp
  | subEff _ _ _ _ _ _ _ _ hBody _ ih =>
      exact ih hSupp
  | nil _ _ _ _ _ hSupp =>
      exact AdjointTypedClausesShape.nil
  | cons _ _ _ _ _ _ _ _ _ _ _ hb rest _ _ hMatch hBody hRest ihBody ihRest hSupp =>
      rcases hSupp with ⟨hSuppBody, hSuppRest⟩
      exact AdjointTypedClausesShape.cons (ihBody hSuppBody) (ihRest hSuppRest)

end

/-- A pair term can never type-check at a tensor result type. Used by
    the structured-seed counterexample below. -/
private theorem hasType_pair_tensor_absurd
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e1 e2 : Term} {ds : DimList} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.pair e1 e2) (Typ.tensor ds) eps Gamma2) :
    False := by
  generalize heq : Term.pair e1 e2 = e_in at h
  generalize hteq : Typ.tensor ds = t_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tpair _ _ _ _ _ _ _ _ _ _ _ _ _ =>
      cases heq
      cases hteq
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq hteq
  | _ =>
      (try cases heq) <;> (try cases hteq) <;>
        first | exact True.intro | (exfalso; contradiction)

/-- Local let-pair inversion used by the handled product-seed
    counterexample. `Progress.lean` has the public theorem, but
    importing it here would create a cycle through `Preservation`. -/
private theorem hasType_letpair_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 GammaOut : LinearCtx}
    {x y : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letpair x y e1 e2) t eps GammaOut) :
    ∃ (Gamma2 Gamma3 : LinearCtx) (t1 t2 : Typ) (eps1 eps2 : EffectRow)
      (slotX slotY : Option Typ),
      HasType Delta Sigma Gamma1 e1 (Typ.pair t1 t2) eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1), (y, some t2)]) e2 t eps2
              (Gamma3 ++ [(x, slotX), (y, slotY)]) ∧
      GammaOut = Gamma3 := by
  generalize heq : Term.letpair x y e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letpair _ _ _ Γ2 Γ3 _ _ _ _ t1 t2 _ eps1 eps2 slotX slotY h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, Γ3, t1, t2, eps1, eps2, slotX, slotY, h1, h2, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ =>
      (try cases heq) <;>
        first | exact True.intro | (exfalso; contradiction)

/-- Local copy inversion used by the handled product-seed
    counterexample. `Preservation.lean` has the public theorem, but
    importing it here would create a cycle. -/
private theorem hasType_copy_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.copy e) t eps Gamma2) :
    ∃ ds, t = Typ.pair (Typ.tensor ds) (Typ.tensor ds) ∧
          HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 := by
  generalize heq : Term.copy e = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | copy _ _ _ _ _ ds _ h' _ =>
      cases heq
      exact ⟨ds, rfl, h'⟩
  | subEff Δ S Γ Γ' _ t' eps0 eps' _ hSub ih =>
      obtain ⟨ds, hteq, hInv⟩ := ih heq
      refine ⟨ds, hteq, ?_⟩
      exact HasType.subEff Δ S Γ Γ' e (Typ.tensor ds) eps0 eps' hInv hSub
  | _ =>
      (try cases heq) <;>
        first | exact True.intro | (exfalso; contradiction)

/-- Local variable inversion used by the higher-order product witness.
    Importing the public inversion surface from `Preservation.lean`
    would create a cycle. -/
private theorem hasType_var_ctx_inv
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma GammaOut : LinearCtx}
    {x : String} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma (Term.var x) t eps GammaOut) :
    ∃ GammaPre GammaPost,
      Gamma = GammaPre ++ [(x, some t)] ++ GammaPost ∧
      GammaOut = GammaPre ++ [(x, none)] ++ GammaPost := by
  generalize heq : Term.var x = e_in at h
  generalize hteq : t = t_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | var _ _ GammaPre GammaPost y ty =>
      cases heq
      cases hteq
      exact ⟨GammaPre, GammaPost, rfl, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq hteq
  | _ =>
      (try cases heq) <;> (try cases hteq) <;>
        first | exact True.intro | (exfalso; contradiction)

private theorem hasType_letBind_inv_local
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma_out : LinearCtx}
    {x : String} {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.letBind x e1 e2) t eps Gamma_out) :
    ∃ (Gamma2 Gamma3 : LinearCtx) (t1 : Typ) (eps1 eps2 : EffectRow)
      (slot : Option Typ),
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma (Gamma2 ++ [(x, some t1)]) e2 t eps2
              (Gamma3 ++ [(x, slot)]) ∧
      Gamma_out = Gamma3 := by
  generalize heq : Term.letBind x e1 e2 = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | letBind _ _ _ Γ2 Γ3 _ _ _ t1 _ eps1 eps2 slot h1 h2 _ _ =>
      cases heq
      exact ⟨Γ2, Γ3, t1, eps1, eps2, slot, h1, h2, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq
  | _ =>
      (try cases heq) <;>
        first | exact True.intro | (exfalso; contradiction)

private theorem hasType_pair_inv_local
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 GammaOut : LinearCtx}
    {e1 e2 : Term} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.pair e1 e2) t eps GammaOut) :
    ∃ (Gamma2 Gamma3 : LinearCtx) (t1 t2 : Typ) (eps1 eps2 : EffectRow),
      HasType Delta Sigma Gamma1 e1 t1 eps1 Gamma2 ∧
      HasType Delta Sigma Gamma2 e2 t2 eps2 Gamma3 ∧
      t = Typ.pair t1 t2 ∧
      GammaOut = Gamma3 := by
  generalize heq : Term.pair e1 e2 = e_in at h
  generalize hteq : t = t_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tpair _ _ _ Gamma2 Gamma3 _ _ t1 t2 eps1 eps2 h1 h2 _ _ =>
      cases heq
      cases hteq
      exact ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, h1, h2, rfl, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      obtain ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, h1, h2, ht, hOut⟩ := ih heq hteq
      exact ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, h1, h2, ht, hOut⟩
  | _ =>
      (try cases heq) <;> (try cases hteq) <;>
        first | exact True.intro | (exfalso; contradiction)

private theorem hasType_sum_inv_local
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma1 Gamma2 : LinearCtx}
    {e : Term} {d : Dim} {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma1 (Term.sum e d) t eps Gamma2) :
    ∃ ds, t = Typ.tensor (rem ds d) ∧ d ∈ ds ∧
          HasType Delta Sigma Gamma1 e (Typ.tensor ds) eps Gamma2 := by
  generalize heq : Term.sum e d = e_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | tsum _ _ _ _ _ ds _ _ h' hmem _ =>
      cases heq
      exact ⟨ds, rfl, hmem, h'⟩
  | subEff Δ S Γ Γ' _ _ eps0 eps' _hSub hSub ih =>
      obtain ⟨ds, hteq, hmem, hInv⟩ := ih heq
      refine ⟨ds, hteq, hmem, ?_⟩
      exact HasType.subEff Δ S Γ Γ' e (Typ.tensor ds) eps0 eps' hInv hSub
  | _ =>
      (try cases heq) <;>
        first | exact True.intro | (exfalso; contradiction)

private def adjointHigherOrderGapTensorT : Typ :=
  Typ.tensor DimList.empty

private def adjointHigherOrderGapFnT : Typ :=
  Typ.arrow adjointHigherOrderGapTensorT adjointHigherOrderGapTensorT []

private def adjointHigherOrderGapBody : Term :=
  Term.snd adjointHigherOrderGapFnT
    (Term.pair
      (Term.abs "y" adjointHigherOrderGapTensorT
        (Term.add (Term.var "x") (Term.var "y")))
      (Term.const 0 DimList.empty))

private theorem adjointHigherOrderGapBody_typed
    {Sigma : StoreTyp} :
    HasType (Capability.diff :: []) Sigma
      ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
      adjointHigherOrderGapBody
      adjointHigherOrderGapTensorT
      []
      ([("x", none)] : LinearCtx) := by
  have hVarX :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (Term.var "x")
        adjointHigherOrderGapTensorT
        []
        ([("x", none), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx) := by
    simpa [adjointHigherOrderGapTensorT, List.append_assoc] using
      (HasType.var (Capability.diff :: []) Sigma
        ([] : LinearCtx)
        ([("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        "x"
        adjointHigherOrderGapTensorT)
  have hVarY :
      HasType (Capability.diff :: []) Sigma
        ([("x", none), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (Term.var "y")
        adjointHigherOrderGapTensorT
        []
        ([("x", none), ("y", none)] : LinearCtx) := by
    simpa [adjointHigherOrderGapTensorT, List.append_assoc] using
      (HasType.var (Capability.diff :: []) Sigma
        ([("x", none)] : LinearCtx)
        ([] : LinearCtx)
        "y"
        adjointHigherOrderGapTensorT)
  have hAdd :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (Term.add (Term.var "x") (Term.var "y"))
        adjointHigherOrderGapTensorT
        []
        ([("x", none), ("y", none)] : LinearCtx) := by
    simpa [EffectRow.union, adjointHigherOrderGapTensorT] using
      (HasType.tadd (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        ([("x", none), ("y", some adjointHigherOrderGapTensorT)] : LinearCtx)
        ([("x", none), ("y", none)] : LinearCtx)
        (Term.var "x")
        (Term.var "y")
        DimList.empty
        []
        []
        hVarX
        hVarY)
  have hAbs :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (Term.abs "y" adjointHigherOrderGapTensorT
          (Term.add (Term.var "x") (Term.var "y")))
        adjointHigherOrderGapFnT
        []
        ([("x", none)] : LinearCtx) := by
    simpa [adjointHigherOrderGapFnT, adjointHigherOrderGapTensorT] using
      (HasType.abs (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
        ([("x", none)] : LinearCtx)
        "y"
        adjointHigherOrderGapTensorT
        adjointHigherOrderGapTensorT
        []
        (Term.add (Term.var "x") (Term.var "y"))
        none
        hAdd)
  have hConst :
      HasType (Capability.diff :: []) Sigma
        ([("x", none)] : LinearCtx)
        (Term.const 0 DimList.empty)
        adjointHigherOrderGapTensorT
        []
        ([("x", none)] : LinearCtx) := by
    simpa [adjointHigherOrderGapTensorT] using
      (HasType.const (Capability.diff :: []) Sigma
        ([("x", none)] : LinearCtx)
        0
        DimList.empty)
  have hPair :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (Term.pair
          (Term.abs "y" adjointHigherOrderGapTensorT
            (Term.add (Term.var "x") (Term.var "y")))
          (Term.const 0 DimList.empty))
        (Typ.pair adjointHigherOrderGapFnT adjointHigherOrderGapTensorT)
        []
        ([("x", none)] : LinearCtx) := by
    simpa [EffectRow.union, adjointHigherOrderGapFnT, adjointHigherOrderGapTensorT] using
      (HasType.tpair (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
        ([("x", none)] : LinearCtx)
        ([("x", none)] : LinearCtx)
        (Term.abs "y" adjointHigherOrderGapTensorT
          (Term.add (Term.var "x") (Term.var "y")))
        (Term.const 0 DimList.empty)
        adjointHigherOrderGapFnT
        adjointHigherOrderGapTensorT
        []
        []
        hAbs
        hConst)
  simpa [adjointHigherOrderGapBody, adjointHigherOrderGapFnT, adjointHigherOrderGapTensorT] using
    (HasType.snd (Capability.diff :: []) Sigma
      ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
      ([("x", none)] : LinearCtx)
      (Term.pair
        (Term.abs "y" adjointHigherOrderGapTensorT
          (Term.add (Term.var "x") (Term.var "y")))
        (Term.const 0 DimList.empty))
      adjointHigherOrderGapFnT
      adjointHigherOrderGapTensorT
      []
      hPair)

private theorem adjointHigherOrderGap_head :
    adjointTypedFrom adjointHigherOrderGapBody adjointHigherOrderGapTensorT "x"
      (Term.var "gs") 0 =
      Term.letpair (freshName "gA" 0) (freshName "gB" 0)
        (Term.pair (zeroCotangent adjointHigherOrderGapFnT) (Term.var "gs"))
        (Term.letBind (freshName "adjA" 0)
          (adjointTypedFrom
            (Term.abs "y" adjointHigherOrderGapTensorT
              (Term.add (Term.var "x") (Term.var "y")))
            adjointHigherOrderGapFnT
            "x"
            (Term.var (freshName "gA" 0))
            (0 + 3))
          (adjointTypedFrom
            (Term.const 0 DimList.empty)
            adjointHigherOrderGapTensorT
            "x"
            (Term.var (freshName "gB" 0))
            (0 + 3))) := by
  simp [adjointHigherOrderGapBody, adjointHigherOrderGapFnT,
    adjointHigherOrderGapTensorT, adjointTypedFrom, zeroCotangent]

private theorem adjointHigherOrderGap_abs_head :
    adjointTypedFrom
      (Term.abs "y" adjointHigherOrderGapTensorT
        (Term.add (Term.var "x") (Term.var "y")))
      adjointHigherOrderGapFnT
      "x"
      (Term.var (freshName "gA" 0))
      (0 + 3) =
      Term.letpair (freshName "gA" (0 + 3)) (freshName "gB" (0 + 3))
        (Term.copy (Term.var (freshName "gA" 0)))
        (Term.letBind (freshName "adjA" ((0 + 3) + 2))
          (adjointTypedFrom
            (Term.var "x")
            adjointHigherOrderGapTensorT
            "x"
            (Term.var (freshName "gA" (0 + 3)))
            (((0 + 3) + 2) + 1))
          (adjointTypedFrom
            (Term.var "y")
            adjointHigherOrderGapTensorT
            "x"
            (Term.var (freshName "gB" (0 + 3)))
            (((0 + 3) + 2) + 1))) := by
  simp [adjointHigherOrderGapFnT, adjointHigherOrderGapTensorT,
    adjointTypedFrom, splitCotangentSeedFrom]

private theorem adjointHigherOrderGap_ctx_no_tensor_gA
    {GammaPre GammaPost : LinearCtx} {ds : DimList}
    (h :
      ([("x", some adjointHigherOrderGapTensorT), ("gs", none),
        (freshName "gA" 0, some Typ.unit),
        (freshName "gB" 0, some adjointHigherOrderGapTensorT)] : LinearCtx) =
        GammaPre ++ [(freshName "gA" 0, some (Typ.tensor ds))] ++ GammaPost) :
    False := by
  cases GammaPre with
  | nil =>
      simp [adjointHigherOrderGapTensorT, freshName] at h
  | cons a GammaPre =>
      cases GammaPre with
      | nil =>
          simp [adjointHigherOrderGapTensorT, freshName] at h
      | cons b GammaPre =>
          cases GammaPre with
          | nil =>
              simp [adjointHigherOrderGapTensorT] at h
          | cons c GammaPre =>
              cases GammaPre with
              | nil =>
                  simp [adjointHigherOrderGapTensorT, freshName] at h
              | cons d GammaPre =>
                  simp [adjointHigherOrderGapTensorT, freshName] at h

private theorem adjointHigherOrderGap_copy_unit_absurd
    {Sigma : StoreTyp} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma
      ([( "x", some adjointHigherOrderGapTensorT), ("gs", none),
        (freshName "gA" 0, some Typ.unit),
        (freshName "gB" 0, some adjointHigherOrderGapTensorT)] : LinearCtx)
      (Term.copy (Term.var (freshName "gA" 0)))
      t
      eps
      GammaOut) :
    False := by
  obtain ⟨ds, _hteq, hVar⟩ := hasType_copy_inv h
  obtain ⟨GammaPre, GammaPost, hCtx, _hOut⟩ := hasType_var_ctx_inv hVar
  exact adjointHigherOrderGap_ctx_no_tensor_gA hCtx

private theorem hasType_unit_inv_local
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma GammaOut : LinearCtx}
    {t : Typ} {eps : EffectRow}
    (h : HasType Delta Sigma Gamma Term.unit t eps GammaOut) :
    t = Typ.unit ∧ GammaOut = Gamma := by
  generalize heq : Term.unit = e_in at h
  generalize hteq : t = t_in at h
  induction h using HasType.rec
    (motive_2 := fun _ _ _ _ _ _ _ _ => True) with
  | unit _ _ Gamma =>
      cases heq
      cases hteq
      exact ⟨rfl, rfl⟩
  | subEff _ _ _ _ _ _ _ _ _ _ ih =>
      exact ih heq hteq
  | _ =>
      (try cases heq) <;> (try cases hteq) <;>
        first | exact True.intro | (exfalso; contradiction)

private theorem adjointHigherOrderGap_seed_var_inv
    {Sigma : StoreTyp} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma
      ([("x", some adjointHigherOrderGapTensorT),
        ("gs", some adjointHigherOrderGapTensorT)] : LinearCtx)
      (Term.var "gs")
      t
      eps
      GammaOut) :
    t = adjointHigherOrderGapTensorT ∧
      GammaOut =
        ([("x", some adjointHigherOrderGapTensorT),
          ("gs", none)] : LinearCtx) := by
  obtain ⟨GammaPre, GammaPost, hCtx, hOut⟩ := hasType_var_ctx_inv h
  cases GammaPre with
  | nil =>
      simp [adjointHigherOrderGapTensorT] at hCtx
  | cons a GammaPre =>
      cases GammaPre with
      | nil =>
          cases GammaPost with
          | nil =>
              simp [adjointHigherOrderGapTensorT] at hCtx
              rcases hCtx with ⟨rfl, rfl⟩
              exact ⟨rfl, by simpa [adjointHigherOrderGapTensorT] using hOut⟩
          | cons b GammaPost =>
              simp [adjointHigherOrderGapTensorT] at hCtx
      | cons b GammaPre =>
          simp [adjointHigherOrderGapTensorT] at hCtx

private theorem adjointHigherOrderGap_seed_pair_inv
    {Sigma : StoreTyp} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma
      ([("x", some adjointHigherOrderGapTensorT),
        ("gs", some adjointHigherOrderGapTensorT)] : LinearCtx)
      (Term.pair (zeroCotangent adjointHigherOrderGapFnT) (Term.var "gs"))
      t
      eps
      GammaOut) :
    t = Typ.pair Typ.unit adjointHigherOrderGapTensorT ∧
      GammaOut =
        ([("x", some adjointHigherOrderGapTensorT),
          ("gs", none)] : LinearCtx) := by
  rcases hasType_pair_inv_local h with
    ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, hZero, hSeed, ht, hOut⟩
  have hUnitInv := hasType_unit_inv_local (by
    simpa [zeroCotangent, cotangentType, adjointHigherOrderGapFnT] using hZero)
  have hSeedInv := adjointHigherOrderGap_seed_var_inv (by simpa [hUnitInv.2] using hSeed)
  cases hUnitInv.1
  cases hSeedInv.1
  cases ht
  exact ⟨rfl, by simpa [hSeedInv.2] using hOut⟩

/-- Even after switching products/projections to typed cotangent seeds,
    the typed transform is still false on higher-order `grad` bodies:
    the current `abs` branch passes a non-tensor seed straight into the
    function body, so a later tensor primitive can still force an
    ill-typed `copy`. This witnesses that the next honest theorem
    surface must carry an explicit supported-fragment premise, not just
    a typed seed. -/
theorem adjointTyped_higherOrder_counterexample :
    (∀ Sigma,
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHigherOrderGapTensorT)] : LinearCtx)
        adjointHigherOrderGapBody
        adjointHigherOrderGapTensorT
        []
        ([("x", none)] : LinearCtx)) ∧
    (∀ Sigma, ¬ ∃ eps GammaOut,
      HasType [] Sigma
        ([("x", some adjointHigherOrderGapTensorT),
          ("gs", some adjointHigherOrderGapTensorT)] : LinearCtx)
        (adjointTypedFrom adjointHigherOrderGapBody
          adjointHigherOrderGapTensorT "x" (Term.var "gs") 0)
        Typ.unit
        eps
        GammaOut) := by
  refine ⟨?_, ?_⟩
  · intro Sigma
    exact adjointHigherOrderGapBody_typed (Sigma := Sigma)
  · intro Sigma
    intro h
    rcases h with ⟨eps, GammaOut, hAdj⟩
    rw [adjointHigherOrderGap_head] at hAdj
    rcases hasType_letpair_inv hAdj with
      ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, slotX, slotY,
        hSeedPair, hBody, _hOut⟩
    have hSeedInv := adjointHigherOrderGap_seed_pair_inv hSeedPair
    cases hSeedInv.1
    cases hSeedInv.2
    rcases hasType_letBind_inv_local hBody with
      ⟨GammaMid, GammaEnd, tAdj, epsAdj, epsRest, slotAdj,
        hAdjAbs, _hRest, _hOut⟩
    rw [adjointHigherOrderGap_abs_head] at hAdjAbs
    rcases hasType_letpair_inv hAdjAbs with
      ⟨GammaCopy, GammaCopyOut, tCopy1, tCopy2, epsCopy, epsBody,
        slotCopy1, slotCopy2, hCopy, _hBody, _hOut⟩
    exact adjointHigherOrderGap_copy_unit_absurd hCopy

private def adjointSndGapDim : Dim :=
  Dim.named "dSnd"

private def adjointSndGapBody : Term :=
  Term.snd (Typ.tensor (ins DimList.empty adjointSndGapDim))
    (Term.pair
      (Term.expand (Term.const 0 DimList.empty) adjointSndGapDim)
      (Term.const 0 DimList.empty))

private theorem adjointSndGapBody_typed
    {Sigma : StoreTyp} :
    HasType (Capability.diff :: []) Sigma
      ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
      adjointSndGapBody
      (Typ.tensor DimList.empty)
      []
      ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
  have hExpandArg :
      HasType (Capability.diff :: []) Sigma
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.const 0 DimList.empty)
        (Typ.tensor DimList.empty)
        []
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
    exact HasType.const (Capability.diff :: []) Sigma _ 0 DimList.empty
  have hExpand :
      HasType (Capability.diff :: []) Sigma
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.expand (Term.const 0 DimList.empty) adjointSndGapDim)
        (Typ.tensor (ins DimList.empty adjointSndGapDim))
        []
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
    exact HasType.texpand (Capability.diff :: []) Sigma
      _ _ _ DimList.empty adjointSndGapDim []
      hExpandArg
  have hConst :
      HasType (Capability.diff :: []) Sigma
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.const 0 DimList.empty)
        (Typ.tensor DimList.empty)
        []
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
    exact HasType.const (Capability.diff :: []) Sigma _ 0 DimList.empty
  have hPair :
      HasType (Capability.diff :: []) Sigma
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.pair
          (Term.expand (Term.const 0 DimList.empty) adjointSndGapDim)
          (Term.const 0 DimList.empty))
        (Typ.pair
          (Typ.tensor (ins DimList.empty adjointSndGapDim))
          (Typ.tensor DimList.empty))
        []
        ([("x", some (Typ.tensor DimList.empty))] : LinearCtx) := by
    simpa [EffectRow.union] using
      (HasType.tpair (Capability.diff :: []) Sigma
        ([( "x", some (Typ.tensor DimList.empty))] : LinearCtx)
        ([( "x", some (Typ.tensor DimList.empty))] : LinearCtx)
        ([( "x", some (Typ.tensor DimList.empty))] : LinearCtx)
        (Term.expand (Term.const 0 DimList.empty) adjointSndGapDim)
        (Term.const 0 DimList.empty)
        (Typ.tensor (ins DimList.empty adjointSndGapDim))
        (Typ.tensor DimList.empty)
        []
        []
        hExpand
        hConst)
  exact HasType.snd (Capability.diff :: []) Sigma
    ([( "x", some (Typ.tensor DimList.empty))] : LinearCtx)
    ([( "x", some (Typ.tensor DimList.empty))] : LinearCtx)
    (Term.pair
      (Term.expand (Term.const 0 DimList.empty) adjointSndGapDim)
      (Term.const 0 DimList.empty))
    (Typ.tensor (ins DimList.empty adjointSndGapDim))
    (Typ.tensor DimList.empty)
    []
    hPair

private def adjointHandleSeedGapT : Typ :=
  Typ.tensor DimList.empty

private def adjointHandleSeedGapClauseBody : Term :=
  Term.pair
    (Term.const 0 DimList.empty)
    (Term.const 0 DimList.empty)

private def adjointHandleSeedGapForwardBody : Term :=
  Term.letBind "u"
    (Term.perform EffectLabel.resource Term.unit)
    adjointHandleSeedGapClauseBody

private def adjointHandleSeedGapClauses :
    List (EffectLabel × String × String × Term) :=
  [(EffectLabel.resource, "p", "k", adjointHandleSeedGapClauseBody)]

private def adjointHandleSeedGapBody : Term :=
  Term.snd adjointHandleSeedGapT
    (Term.handle [EffectLabel.resource]
      adjointHandleSeedGapForwardBody
      adjointHandleSeedGapClauses)

private theorem adjointHandleSeedGapBody_typed
    {Sigma : StoreTyp} :
    HasType (Capability.diff :: []) Sigma
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      adjointHandleSeedGapBody
      adjointHandleSeedGapT
      []
      ([("x", some adjointHandleSeedGapT)] : LinearCtx) := by
  let tPair : Typ := Typ.pair adjointHandleSeedGapT adjointHandleSeedGapT
  let tK : Typ := Typ.arrow Typ.unit tPair []
  have hUnit :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        Term.unit Typ.unit []
        ([("x", some adjointHandleSeedGapT)] : LinearCtx) := by
    exact HasType.unit (Capability.diff :: []) Sigma _
  have hPerform :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        (Term.perform EffectLabel.resource Term.unit)
        Typ.unit
        [EffectLabel.resource]
        ([("x", some adjointHandleSeedGapT)] : LinearCtx) := by
    exact HasType.perform (Capability.diff :: []) Sigma _ _
      EffectLabel.resource Term.unit Typ.unit Typ.unit [] hUnit
      (by simp [OpSigMatch, opArgType, opRetType])
  have hConstX :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        (Term.const 0 DimList.empty)
        adjointHandleSeedGapT
        []
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx) := by
    simpa [adjointHandleSeedGapT] using
      (HasType.const (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        0 DimList.empty)
  have hPairBody :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        adjointHandleSeedGapClauseBody
        tPair
        []
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx) := by
    simpa [adjointHandleSeedGapClauseBody, tPair, EffectRow.union] using
      (HasType.tpair (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT), ("u", some Typ.unit)] : LinearCtx)
        (Term.const 0 DimList.empty)
        (Term.const 0 DimList.empty)
        adjointHandleSeedGapT
        adjointHandleSeedGapT
        []
        []
        hConstX
        hConstX)
  have hForwardBody :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        adjointHandleSeedGapForwardBody
        tPair
        [EffectLabel.resource]
        ([("x", some adjointHandleSeedGapT)] : LinearCtx) := by
    simpa [adjointHandleSeedGapForwardBody, tPair] using
      (HasType.letBind (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        "u"
        (Term.perform EffectLabel.resource Term.unit)
        adjointHandleSeedGapClauseBody
        Typ.unit
        tPair
        [EffectLabel.resource]
        []
        (some Typ.unit)
        hPerform
        hPairBody)
  have hConstP :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        (Term.const 0 DimList.empty)
        adjointHandleSeedGapT
        []
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx) := by
    simpa [adjointHandleSeedGapT] using
      (HasType.const (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        0 DimList.empty)
  have hClauseBody :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        adjointHandleSeedGapClauseBody
        tPair
        []
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx) := by
    simpa [adjointHandleSeedGapClauseBody, tPair, EffectRow.union] using
      (HasType.tpair (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT), ("p", some Typ.unit), ("k", some tK)] : LinearCtx)
        (Term.const 0 DimList.empty)
        (Term.const 0 DimList.empty)
        adjointHandleSeedGapT
        adjointHandleSeedGapT
        []
        []
        hConstP
        hConstP)
  have hClauses :
      ClausesTyped (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        tPair
        []
        adjointHandleSeedGapClauses := by
    exact ClausesTyped.cons (Capability.diff :: []) Sigma
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      tPair
      Typ.unit
      Typ.unit
      []
      EffectLabel.resource
      "p"
      "k"
      adjointHandleSeedGapClauseBody
      []
      (some Typ.unit)
      (some tK)
      (by simp [OpSigMatch, opArgType, opRetType])
      hClauseBody
      (ClausesTyped.nil (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        tPair
        [])
  have hHandle :
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        (Term.handle [EffectLabel.resource]
          adjointHandleSeedGapForwardBody
          adjointHandleSeedGapClauses)
        tPair
        []
        ([("x", some adjointHandleSeedGapT)] : LinearCtx) := by
    exact HasType.handle (Capability.diff :: []) Sigma
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      adjointHandleSeedGapForwardBody
      adjointHandleSeedGapClauses
      tPair
      [EffectLabel.resource]
      [EffectLabel.resource]
      hForwardBody
      (by intro op hop; simp at hop; rcases hop with rfl; simp)
      (by intro cl hmem; simp [adjointHandleSeedGapClauses] at hmem ⊢; rcases hmem with rfl; simp)
      (by
        intro op hop
        simp at hop
        rcases hop with rfl
        exact ⟨(EffectLabel.resource, "p", "k", adjointHandleSeedGapClauseBody),
          by simp [adjointHandleSeedGapClauses], rfl⟩)
      hClauses
  simpa [adjointHandleSeedGapBody, adjointHandleSeedGapT] using
    (HasType.snd (Capability.diff :: []) Sigma
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      ([("x", some adjointHandleSeedGapT)] : LinearCtx)
      (Term.handle [EffectLabel.resource]
        adjointHandleSeedGapForwardBody
        adjointHandleSeedGapClauses)
      adjointHandleSeedGapT
      adjointHandleSeedGapT
      []
      hHandle)

private theorem adjointHandleSeedGap_head :
    adjointFrom adjointHandleSeedGapBody "x" (Term.var "gs") 0 =
      Term.letpair (freshName "gA" 0) (freshName "gB" 0)
        (Term.copy (Term.pair (Term.const 0 DimList.empty) (Term.var "gs")))
        (Term.letBind (freshName "adjHb" 0)
          (adjointFrom adjointHandleSeedGapClauseBody "x"
            (Term.var (freshName "gA" 0)) (0 + 3))
          (adjointFrom adjointHandleSeedGapForwardBody "x"
            (Term.var (freshName "gB" 0)) (0 + 3))) := by
  simp [adjointHandleSeedGapBody, adjointHandleSeedGapT,
    adjointHandleSeedGapForwardBody, adjointHandleSeedGapClauses,
    adjointHandleSeedGapClauseBody, adjointFrom, adjointClausesFrom,
    zeroCotangent]

/-- The old product/projection false witness is gone from the transform
    itself, but the current public theorem surface is still false:
    `adjointFrom`'s legacy handler path splits every clause seed with
    tensor-only `copy`, so feeding a structured cotangent seed through
    `snd` into `handle` produces an untypable adjoint term before the
    `mul` case is even in play. The staged `adjointTypedFrom` /
    `adjointTypedClausesFrom` path in `AdjointTransform.lean` is the
    intended repair. -/
theorem adjoint_handle_product_seed_counterexample :
    (∀ Sigma,
      HasType (Capability.diff :: []) Sigma
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)
        adjointHandleSeedGapBody
        adjointHandleSeedGapT
        []
        ([("x", some adjointHandleSeedGapT)] : LinearCtx)) ∧
    (∀ Sigma, ¬ ∃ eps GammaOut,
      HasType [] Sigma
        ([("x", some adjointHandleSeedGapT), ("gs", some adjointHandleSeedGapT)] : LinearCtx)
        (adjointFrom adjointHandleSeedGapBody "x" (Term.var "gs") 0)
        Typ.unit
        eps
        GammaOut) := by
  refine ⟨?_, ?_⟩
  · intro Sigma
    exact adjointHandleSeedGapBody_typed (Sigma := Sigma)
  · intro Sigma
    intro h
    rcases h with ⟨eps, GammaOut, hAdj⟩
    rw [adjointHandleSeedGap_head] at hAdj
    rcases hasType_letpair_inv hAdj with
      ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, slotX, slotY,
        hCopy, _hBody, _hOut, _hSub⟩
    obtain ⟨ds, _hteq, hPairAsTensor⟩ := hasType_copy_inv hCopy
    exact hasType_pair_tensor_absurd hPairAsTensor

private def adjointMulCtxGapX : String :=
  freshName "x" 0

private def adjointMulCtxGapT : Typ :=
  Typ.tensor DimList.empty

private def adjointMulCtxGapBody : Term :=
  Term.mul (Term.var adjointMulCtxGapX) (Term.const 0 DimList.empty)

private theorem adjointMulCtxGapShape
    {Sigma : StoreTyp} :
    AdjointTypedShape (Capability.diff :: []) Sigma
      adjointMulCtxGapT
      adjointMulCtxGapBody := by
  refine .mul ?_ .var .const
  refine ⟨[(adjointMulCtxGapX, some adjointMulCtxGapT)],
    [(adjointMulCtxGapX, none)],
    [(adjointMulCtxGapX, none)],
    [], [], ?_, ?_⟩
  · simpa [adjointMulCtxGapX, adjointMulCtxGapT, List.append_assoc] using
      (HasType.var (Capability.diff :: []) Sigma
        ([] : LinearCtx)
        ([] : LinearCtx)
        adjointMulCtxGapX
        adjointMulCtxGapT)
  · simpa [adjointMulCtxGapT] using
      (HasType.const (Capability.diff :: []) Sigma
        ([(adjointMulCtxGapX, none)] : LinearCtx)
        0
        DimList.empty)

/-- The current private helper theorem shape for `adjointTypedFrom`
    cannot be right for `mul`: it quantifies over arbitrary seed
    contexts, but the generated term replays raw source operands via
    `copy e1` / `copy e2`. Even the tiny body `mul (var x) (const 0)`
    fails if the seed context omits `x`. -/
theorem adjointTyped_mul_ctx_counterexample :
    (∀ Sigma,
      AdjointTypedShape (Capability.diff :: []) Sigma
        adjointMulCtxGapT
        adjointMulCtxGapBody) ∧
    (∀ Sigma, ¬ HasType [] Sigma []
      (adjointTypedFrom adjointMulCtxGapBody
        adjointMulCtxGapT
        adjointMulCtxGapX
        (Term.const 0 DimList.empty)
        0)
      Typ.unit
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      []) := by
  refine ⟨?_, ?_⟩
  · intro Sigma
    exact adjointMulCtxGapShape (Sigma := Sigma)
  · intro Sigma
    intro hAdj
    simp [adjointMulCtxGapBody, adjointMulCtxGapT, adjointMulCtxGapX,
      adjointTypedFrom, splitCotangentSeedFrom] at hAdj
    rcases hasType_letpair_inv hAdj with
      ⟨Gamma2, Gamma3, t1, t2, _eps1, _eps2, slotGA, slotGB,
        hCopySeed, hAfterSeed, hOut⟩
    obtain ⟨dsSeed, _hSeedTy, hSeedConst⟩ := hasType_copy_inv hCopySeed
    have hGamma2Nil : Gamma2 = [] := has_type_closed_output_of_closed_input hSeedConst
    subst hGamma2Nil
    rcases hasType_letpair_inv hAfterSeed with
      ⟨Gamma2', Gamma3', tA, tATape, _epsA, _epsRest, slotA, slotATape,
        hCopyX, _hRest, hOut2⟩
    obtain ⟨dsX, _hCopyTy, hVarX⟩ := hasType_copy_inv hCopyX
    rcases hasType_var_ctx_inv hVarX with
      ⟨GammaPre, GammaPost, hIn, _hOutVar⟩
    have hxIn :
        adjointMulCtxGapX ∈
          linearCtxDom
            ([(freshName "gA" 0, some t1),
              (freshName "gB" 0, some t2)] : LinearCtx) := by
      have hIn' :
          ([(freshName "gA" 0, some t1),
            (freshName "gB" 0, some t2)] : LinearCtx) =
            GammaPre ++ [(adjointMulCtxGapX, some (Typ.tensor dsX))] ++ GammaPost := by
        simpa using hIn
      have :
          adjointMulCtxGapX ∈
            linearCtxDom
              (GammaPre ++ [(adjointMulCtxGapX, some (Typ.tensor dsX))] ++ GammaPost) := by
        simp [linearCtxDom]
      rw [hIn']
      exact this
    have hNoHashX : NoHash "x" := by
      show '#' ∉ ("x" : String).toList
      decide
    have hNoHashGA : NoHash "gA" := by
      show '#' ∉ ("gA" : String).toList
      decide
    have hNoHashGB : NoHash "gB" := by
      show '#' ∉ ("gB" : String).toList
      decide
    have hxNeGA :
        adjointMulCtxGapX ≠ freshName "gA" 0 := by
      simpa [adjointMulCtxGapX] using
        (freshName_ne_of_base_ne "x" "gA" 0 0 hNoHashX hNoHashGA (by decide))
    have hxNeGB :
        adjointMulCtxGapX ≠ freshName "gB" 0 := by
      simpa [adjointMulCtxGapX] using
        (freshName_ne_of_base_ne "x" "gB" 0 0 hNoHashX hNoHashGB (by decide))
    simp [linearCtxDom, hxNeGA, hxNeGB] at hxIn

private theorem adjointMulPublic_seed_var_inv
    {Sigma : StoreTyp} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    (h : HasType [] Sigma
      ([(adjointMulCtxGapX, some adjointMulCtxGapT),
        ("gs", some adjointMulCtxGapT)] : LinearCtx)
      (Term.var "gs")
      t
      eps
      GammaOut) :
    t = adjointMulCtxGapT ∧
      GammaOut =
        ([(adjointMulCtxGapX, some adjointMulCtxGapT),
          ("gs", none)] : LinearCtx) := by
  obtain ⟨GammaPre, GammaPost, hCtx, hOut⟩ := hasType_var_ctx_inv h
  cases GammaPre with
  | nil =>
      have : False := by
        simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hCtx
      exact this.elim
  | cons a GammaPre =>
      cases GammaPre with
      | nil =>
          cases GammaPost with
          | nil =>
              simp [adjointMulCtxGapX, adjointMulCtxGapT, freshName] at hCtx
              rcases hCtx with ⟨rfl, rfl⟩
              exact ⟨rfl, by
                simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hOut⟩
          | cons b GammaPost =>
              have : False := by
                simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hCtx
              exact this.elim
      | cons b GammaPre =>
          have : False := by
            simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hCtx
          exact this.elim

private theorem adjointMulPublic_copy_x_var_inv
    {Sigma : StoreTyp} {t : Typ} {eps : EffectRow} {GammaOut : LinearCtx}
    {t1 t2 : Typ}
    (h : HasType [] Sigma
      ([(adjointMulCtxGapX, some adjointMulCtxGapT),
        ("gs", none),
        (freshName "gA" 0, some t1),
        (freshName "gB" 0, some t2)] : LinearCtx)
      (Term.var adjointMulCtxGapX)
      t
      eps
      GammaOut) :
    t = adjointMulCtxGapT ∧
      GammaOut =
        ([(adjointMulCtxGapX, none),
          ("gs", none),
          (freshName "gA" 0, some t1),
          (freshName "gB" 0, some t2)] : LinearCtx) := by
  obtain ⟨GammaPre, GammaPost, hCtx, hOut⟩ := hasType_var_ctx_inv h
  cases GammaPre with
  | nil =>
      cases GammaPost with
      | nil =>
          have : False := by
            simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hCtx
          exact this.elim
      | cons a GammaPost =>
          cases GammaPost with
          | nil =>
              have : False := by
                simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hCtx
              exact this.elim
          | cons b GammaPost =>
              cases GammaPost with
              | nil =>
                  have : False := by
                    simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hCtx
                  exact this.elim
              | cons c GammaPost =>
                  cases GammaPost with
                  | nil =>
                      simp [adjointMulCtxGapX, adjointMulCtxGapT, freshName] at hCtx
                      rcases hCtx with ⟨rfl, rfl, rfl, rfl⟩
                      exact ⟨rfl, by
                        simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hOut⟩
                  | cons d GammaPost =>
                      have : False := by
                        simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hCtx
                      exact this.elim
  | cons a GammaPre =>
      cases GammaPre with
      | nil =>
          have : False := by
            simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hCtx
          exact this.elim
      | cons b GammaPre =>
          cases GammaPre with
          | nil =>
              have : False := by
                simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hCtx
              exact this.elim
          | cons c GammaPre =>
              cases GammaPre with
              | nil =>
                  have : False := by
                    simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hCtx
                  exact this.elim
              | cons d GammaPre =>
                  have : False := by
                    simpa [adjointMulCtxGapX, adjointMulCtxGapT, freshName] using hCtx
                  exact this.elim

/-- The current public typed theorem target for `adjointTypedFrom`
    cannot be right for `mul`: even on the real `grad`-style input
    context `x, gs`, the transformed term replays `copy x` and therefore
    cannot leave `x` live on output. -/
theorem adjointTyped_mul_output_counterexample :
    ∀ Sigma, ¬ HasType [] Sigma
      ([(adjointMulCtxGapX, some adjointMulCtxGapT),
        ("gs", some adjointMulCtxGapT)] : LinearCtx)
      (adjointTypedFrom adjointMulCtxGapBody
        adjointMulCtxGapT
        adjointMulCtxGapX
        (Term.var "gs")
        0)
      Typ.unit
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      ([(adjointMulCtxGapX, some adjointMulCtxGapT),
        ("gs", none)] : LinearCtx) := by
  intro Sigma
  intro hAdj
  simp [adjointMulCtxGapBody, adjointMulCtxGapT, adjointMulCtxGapX,
    adjointTypedFrom, splitCotangentSeedFrom] at hAdj
  rcases hasType_letpair_inv hAdj with
    ⟨Gamma2, Gamma3, t1, t2, _eps1, _eps2, slotGA, slotGB,
      hCopySeed, hAfterSeed, hOut⟩
  obtain ⟨_dsSeed, _hSeedTy, hVarSeed⟩ := hasType_copy_inv hCopySeed
  have hGamma2 :
      Gamma2 =
        ([(adjointMulCtxGapX, some adjointMulCtxGapT),
          ("gs", none)] : LinearCtx) := by
    obtain ⟨hTySeed, hOutSeed⟩ := adjointMulPublic_seed_var_inv (Sigma := Sigma) hVarSeed
    cases hTySeed
    exact hOutSeed
  subst hGamma2
  rcases hasType_letpair_inv hAfterSeed with
    ⟨Gamma2', Gamma3', tA, tATape, _epsA, _epsRest, slotA, slotATape,
      hCopyX, hRest, hOut2⟩
  obtain ⟨_dsX, _hCopyTy, hVarX⟩ := hasType_copy_inv hCopyX
  have hGamma2' :
      Gamma2' =
        ([(adjointMulCtxGapX, none),
          ("gs", none),
          (freshName "gA" 0, some t1),
          (freshName "gB" 0, some t2)] : LinearCtx) := by
    obtain ⟨hTyX, hOutX⟩ :=
      adjointMulPublic_copy_x_var_inv (Sigma := Sigma) (t1 := t1) (t2 := t2) hVarX
    cases hTyX
    exact hOutX
  subst hGamma2'
  have hLookupOut :
      lookupLinearCtx
        (Gamma3' ++
          [(freshName "a" 2, slotA), (freshName "aTape" 2, slotATape)])
        adjointMulCtxGapX = some adjointMulCtxGapT := by
    rw [← hOut2, ← hOut]
    simp [lookupLinearCtx, adjointMulCtxGapX, adjointMulCtxGapT, freshName]
  have hSlotSubRest := has_type_slotSub hRest
  obtain ⟨tIn, hLookupIn⟩ := slotSub_lookup_some hSlotSubRest hLookupOut
  simp [lookupLinearCtx, adjointMulCtxGapX, adjointMulCtxGapT, freshName] at hLookupIn

private def adjointMulShapeGapBody : Term :=
  Term.mul (Term.var "y") (Term.const 0 DimList.empty)

private theorem adjointMulShapeGapShape
    {Sigma : StoreTyp} :
    AdjointTypedShape (Capability.diff :: []) Sigma
      adjointMulCtxGapT
      adjointMulShapeGapBody := by
  refine .mul ?_ .var .const
  refine ⟨[("y", some adjointMulCtxGapT)],
    [("y", none)],
    [("y", none)],
    [], [], ?_, ?_⟩
  · simpa [adjointMulCtxGapT, List.append_assoc] using
      (HasType.var (Capability.diff :: []) Sigma
        ([] : LinearCtx)
        ([] : LinearCtx)
        "y"
        adjointMulCtxGapT)
  · simpa [adjointMulCtxGapT] using
      (HasType.const (Capability.diff :: []) Sigma
        ([("y", none)] : LinearCtx)
        0
        DimList.empty)

private theorem adjointMulShapeGap_head :
    adjointTypedFrom adjointMulShapeGapBody
      adjointMulCtxGapT
      adjointMulCtxGapX
      (Term.var "gs")
      0 =
      Term.letpair (freshName "gA" 0) (freshName "gB" 0)
        (Term.copy (Term.var "gs"))
        (Term.letpair (freshName "a" 2) (freshName "aTape" 2)
          (Term.copy (Term.var "y"))
          (Term.letpair (freshName "b" 3) (freshName "bTape" 3)
            (Term.copy (Term.const 0 DimList.empty))
            (Term.letBind (freshName "y" 4)
              (Term.mul (Term.var (freshName "a" 2)) (Term.var (freshName "b" 3)))
              (Term.letBind (freshName "adjA" 5)
                (adjointTypedFrom (Term.var "y")
                  adjointMulCtxGapT
                  adjointMulCtxGapX
                  (Term.mul (Term.var (freshName "gA" 0))
                    (Term.var (freshName "bTape" 3)))
                  6)
                (adjointTypedFrom (Term.const 0 DimList.empty)
                  adjointMulCtxGapT
                  adjointMulCtxGapX
                  (Term.mul (Term.var (freshName "gB" 0))
                    (Term.var (freshName "aTape" 2)))
                  6))))) := by
  simp [adjointMulShapeGapBody, adjointMulCtxGapT,
    adjointTypedFrom, splitCotangentSeedFrom]

private theorem adjointMulShapeGap_ctx_no_y
    {GammaPre GammaPost : LinearCtx} {ds : DimList}
    {t1 t2 : Typ}
    (h :
      ([(adjointMulCtxGapX, some adjointMulCtxGapT),
        ("gs", none),
        (freshName "gA" 0, some t1),
        (freshName "gB" 0, some t2)] : LinearCtx) =
        GammaPre ++ [("y", some (Typ.tensor ds))] ++ GammaPost) :
    False := by
  cases GammaPre with
  | nil =>
      simp [adjointMulCtxGapX, adjointMulCtxGapT, freshName] at h
  | cons a GammaPre =>
      cases GammaPre with
      | nil =>
          simp [adjointMulCtxGapX, adjointMulCtxGapT, freshName] at h
      | cons b GammaPre =>
          cases GammaPre with
          | nil =>
              simp [adjointMulCtxGapX, adjointMulCtxGapT, freshName] at h
          | cons c GammaPre =>
              cases GammaPre with
              | nil =>
                  simp [adjointMulCtxGapX, adjointMulCtxGapT, freshName] at h
              | cons d GammaPre =>
                  simp [adjointMulCtxGapX, adjointMulCtxGapT, freshName] at h

/-- The current shape-only private helper surface is still too strong for
    `mul`: even with the honest slot result, a shape witness alone does
    not constrain the current seed/source context enough to replay raw
    source operands like `var "y"` under `copy`. -/
theorem adjointTyped_mul_shape_counterexample :
    (∀ Sigma,
      AdjointTypedShape (Capability.diff :: []) Sigma
        adjointMulCtxGapT
        adjointMulShapeGapBody) ∧
    (∀ Sigma, ¬ ∃ eps GammaOut,
      HasType [] Sigma
        ([(adjointMulCtxGapX, some adjointMulCtxGapT),
          ("gs", some adjointMulCtxGapT)] : LinearCtx)
        (adjointTypedFrom adjointMulShapeGapBody
          adjointMulCtxGapT
          adjointMulCtxGapX
          (Term.var "gs")
          0)
        Typ.unit
        eps
        GammaOut) := by
  refine ⟨?_, ?_⟩
  · intro Sigma
    exact adjointMulShapeGapShape (Sigma := Sigma)
  · intro Sigma
    intro h
    rcases h with ⟨eps, GammaOut, hAdj⟩
    rw [adjointMulShapeGap_head] at hAdj
    rcases hasType_letpair_inv hAdj with
      ⟨Gamma2, Gamma3, t1, t2, eps1, eps2, slotGA, slotGB,
        hCopySeed, hAfterSeed, _hOut⟩
    obtain ⟨_dsSeed, _hCopyTy, hVarSeed⟩ := hasType_copy_inv hCopySeed
    obtain ⟨hTySeed, hOutSeed⟩ :=
      adjointMulPublic_seed_var_inv (Sigma := Sigma) hVarSeed
    cases hTySeed
    subst Gamma2
    rcases hasType_letpair_inv hAfterSeed with
      ⟨Gamma2', Gamma3', tA, tATape, _epsA, _epsRest, slotA, slotATape,
        hCopyY, _hRest, _hOut2⟩
    obtain ⟨dsY, _hCopyTy, hVarY⟩ := hasType_copy_inv hCopyY
    obtain ⟨GammaPre, GammaPost, hCtx, _hOutVar⟩ := hasType_var_ctx_inv hVarY
    exact adjointMulShapeGap_ctx_no_y (t1 := t1) (t2 := t2) hCtx

/- The legacy `adjointFrom` theorem family was removed from the live
   proof surface once handled product-seed counterexamples showed that
   its tensor-only clause threading is false. `Preservation` now uses
   the staged typed companion below. -/

private theorem hasType_var_with_suffix
    (Delta : CapCtx) (Sigma : StoreTyp)
    (GammaPre GammaPost : LinearCtx)
    (x : String) (t : Typ) :
    HasType Delta Sigma
      (GammaPre ++ [(x, some t)] ++ GammaPost)
      (Term.var x)
      t
      []
      (GammaPre ++ [(x, none)] ++ GammaPost) := by
  simpa [List.append_assoc] using
    (HasType.var Delta Sigma GammaPre GammaPost x t)

private theorem slot_choice_trans
    {slot0 slot1 slot2 : Option Typ}
    (h01 : slot1 = none ∨ slot1 = slot0)
    (h12 : slot2 = none ∨ slot2 = slot1) :
    slot2 = none ∨ slot2 = slot0 := by
  rcases h12 with h12 | h12
  · exact Or.inl h12
  · rcases h01 with h01 | h01
    · exact Or.inl (h12.trans h01)
    · exact Or.inr (h12.trans h01)

private theorem adjointTypedClauses_pair_finish_slot_typed
    (Delta : CapCtx) (Sigma : StoreTyp)
    {t1 t2 : Typ} {hb body : Term}
    {rest : List (EffectLabel × String × String × Term)}
    (ihHead :
      ∀ {Gamma suffixIn suffixOut : LinearCtx} {x : String} {gSeed : Term}
          {n : Nat} {slotIn : Option Typ},
        HasType Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          gSeed
          (cotangentType (Typ.pair t1 t2))
          []
          (Gamma ++ [(x, slotIn)] ++ suffixOut) →
        ∃ slotOut,
          (slotOut = none ∨ slotOut = slotIn) ∧
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (adjointTypedFrom hb (Typ.pair t1 t2) x gSeed n)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(x, slotOut)] ++ suffixOut))
    (ihRest :
      ∀ {Gamma suffixIn suffixOut : LinearCtx} {x : String} {body' : Term}
          {gSeed : Term} {n : Nat} {slotIn : Option Typ},
        (∀ {Gamma0 suffixIn0 suffixOut0 : LinearCtx} {x0 : String} {gSeed0 : Term}
            {n0 : Nat} {slotIn0 : Option Typ},
          HasType Delta Sigma
            (Gamma0 ++ [(x0, slotIn0)] ++ suffixIn0)
            gSeed0
            (cotangentType (Typ.pair t1 t2))
            []
            (Gamma0 ++ [(x0, slotIn0)] ++ suffixOut0) →
          ∃ slotOut0,
            (slotOut0 = none ∨ slotOut0 = slotIn0) ∧
            HasType Delta Sigma
              (Gamma0 ++ [(x0, slotIn0)] ++ suffixIn0)
              (adjointTypedFrom body' (Typ.pair t1 t2) x0 gSeed0 n0)
              Typ.unit
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
              (Gamma0 ++ [(x0, slotOut0)] ++ suffixOut0)) →
        HasType Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          gSeed
          (cotangentType (Typ.pair t1 t2))
          []
          (Gamma ++ [(x, slotIn)] ++ suffixOut) →
        ∃ slotOut,
          (slotOut = none ∨ slotOut = slotIn) ∧
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (adjointTypedClausesFrom rest x body' (Typ.pair t1 t2) gSeed n)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(x, slotOut)] ++ suffixOut))
    {Gamma suffixHead suffixRest suffixOut : LinearCtx}
    {param adjHb : String} {headSeed restSeed : Term}
    {n : Nat} {slotIn : Option Typ}
    (ihBody :
      ∀ {Gamma0 suffixIn0 suffixOut0 : LinearCtx} {x0 : String} {gSeed0 : Term}
          {n0 : Nat} {slotIn0 : Option Typ},
        HasType Delta Sigma
          (Gamma0 ++ [(x0, slotIn0)] ++ suffixIn0)
          gSeed0
          (cotangentType (Typ.pair t1 t2))
          []
          (Gamma0 ++ [(x0, slotIn0)] ++ suffixOut0) →
        ∃ slotOut0,
          (slotOut0 = none ∨ slotOut0 = slotIn0) ∧
          HasType Delta Sigma
            (Gamma0 ++ [(x0, slotIn0)] ++ suffixIn0)
            (adjointTypedFrom body (Typ.pair t1 t2) x0 gSeed0 n0)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma0 ++ [(x0, slotOut0)] ++ suffixOut0))
    (hHeadSeed :
      HasType Delta Sigma
        (Gamma ++ [(param, slotIn)] ++ suffixHead)
        headSeed
        (cotangentType (Typ.pair t1 t2))
        []
        (Gamma ++ [(param, slotIn)] ++ suffixRest))
    (hRestSeed :
      ∀ {slotMid : Option Typ},
        HasType Delta Sigma
          (Gamma ++ [(param, slotMid)] ++ suffixRest ++ [(adjHb, some Typ.unit)])
          restSeed
          (cotangentType (Typ.pair t1 t2))
          []
          (Gamma ++ [(param, slotMid)] ++ suffixOut ++ [(adjHb, some Typ.unit)])) :
    ∃ slotOut,
      (slotOut = none ∨ slotOut = slotIn) ∧
      HasType Delta Sigma
        (Gamma ++ [(param, slotIn)] ++ suffixHead)
        (Term.letBind adjHb
          (adjointTypedFrom hb (Typ.pair t1 t2) param headSeed n)
          (adjointTypedClausesFrom rest param body (Typ.pair t1 t2) restSeed n))
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma ++ [(param, slotOut)] ++ suffixOut) := by
  obtain ⟨slotMid, hslotMid, hHeadTyped⟩ :=
    ihHead
      (Gamma := Gamma)
      (suffixIn := suffixHead)
      (suffixOut := suffixRest)
      (x := param)
      (gSeed := headSeed)
      (n := n)
      (slotIn := slotIn)
      hHeadSeed
  obtain ⟨slotOut, hslotOut, hRestTyped⟩ :=
    ihRest
      (Gamma := Gamma)
      (suffixIn := suffixRest ++ [(adjHb, some Typ.unit)])
      (suffixOut := suffixOut ++ [(adjHb, some Typ.unit)])
      (x := param)
      (body' := body)
      (gSeed := restSeed)
      (n := n)
      (slotIn := slotMid)
      ihBody
      (by simpa [List.append_assoc] using hRestSeed (slotMid := slotMid))
  refine ⟨slotOut, slot_choice_trans hslotMid hslotOut, ?_⟩
  exact adjointTyped_seq_typed Delta Sigma
    (Gamma ++ [(param, slotIn)] ++ suffixHead)
    (Gamma ++ [(param, slotMid)] ++ suffixRest)
    (Gamma ++ [(param, slotOut)] ++ suffixOut)
    adjHb
    (adjointTypedFrom hb (Typ.pair t1 t2) param headSeed n)
    (adjointTypedClausesFrom rest param body (Typ.pair t1 t2) restSeed n)
    (some Typ.unit)
    hHeadTyped
    (by simpa [List.append_assoc] using hRestTyped)

private theorem adjointTypedClauses_cons_slot_typed
    (Delta : CapCtx) (Sigma : StoreTyp)
    {t : Typ} {op : EffectLabel} {xv kv param : String} {hb body : Term}
    {rest : List (EffectLabel × String × String × Term)}
    (ihHead :
      ∀ {Gamma suffixIn suffixOut : LinearCtx} {x : String} {gSeed : Term}
          {n : Nat} {slotIn : Option Typ},
        HasType Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          gSeed
          (cotangentType t)
          []
          (Gamma ++ [(x, slotIn)] ++ suffixOut) →
        ∃ slotOut,
          (slotOut = none ∨ slotOut = slotIn) ∧
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (adjointTypedFrom hb t x gSeed n)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(x, slotOut)] ++ suffixOut))
    (ihRest :
      ∀ {Gamma suffixIn suffixOut : LinearCtx} {x : String} {body' : Term}
          {gSeed : Term} {n : Nat} {slotIn : Option Typ},
        (∀ {Gamma0 suffixIn0 suffixOut0 : LinearCtx} {x0 : String} {gSeed0 : Term}
            {n0 : Nat} {slotIn0 : Option Typ},
          HasType Delta Sigma
            (Gamma0 ++ [(x0, slotIn0)] ++ suffixIn0)
            gSeed0
            (cotangentType t)
            []
            (Gamma0 ++ [(x0, slotIn0)] ++ suffixOut0) →
          ∃ slotOut0,
            (slotOut0 = none ∨ slotOut0 = slotIn0) ∧
            HasType Delta Sigma
              (Gamma0 ++ [(x0, slotIn0)] ++ suffixIn0)
              (adjointTypedFrom body' t x0 gSeed0 n0)
              Typ.unit
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
              (Gamma0 ++ [(x0, slotOut0)] ++ suffixOut0)) →
        HasType Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          gSeed
          (cotangentType t)
          []
          (Gamma ++ [(x, slotIn)] ++ suffixOut) →
        ∃ slotOut,
          (slotOut = none ∨ slotOut = slotIn) ∧
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (adjointTypedClausesFrom rest x body' t gSeed n)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(x, slotOut)] ++ suffixOut))
    {Gamma suffixIn suffixOut : LinearCtx} {gSeed : Term}
    {n : Nat} {slotIn : Option Typ}
    (ihBody :
      ∀ {Gamma0 suffixIn0 suffixOut0 : LinearCtx} {x0 : String} {gSeed0 : Term}
          {n0 : Nat} {slotIn0 : Option Typ},
        HasType Delta Sigma
          (Gamma0 ++ [(x0, slotIn0)] ++ suffixIn0)
          gSeed0
          (cotangentType t)
          []
          (Gamma0 ++ [(x0, slotIn0)] ++ suffixOut0) →
        ∃ slotOut0,
          (slotOut0 = none ∨ slotOut0 = slotIn0) ∧
          HasType Delta Sigma
            (Gamma0 ++ [(x0, slotIn0)] ++ suffixIn0)
            (adjointTypedFrom body t x0 gSeed0 n0)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma0 ++ [(x0, slotOut0)] ++ suffixOut0))
    (h_seed :
      HasType Delta Sigma
        (Gamma ++ [(param, slotIn)] ++ suffixIn)
        gSeed
        (cotangentType t)
        []
        (Gamma ++ [(param, slotIn)] ++ suffixOut)) :
    ∃ slotOut,
      (slotOut = none ∨ slotOut = slotIn) ∧
      HasType Delta Sigma
        (Gamma ++ [(param, slotIn)] ++ suffixIn)
        (adjointTypedClausesFrom ((op, xv, kv, hb) :: rest) param body t gSeed n)
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma ++ [(param, slotOut)] ++ suffixOut) := by
  cases t with
  | tensor ds =>
      let gA := freshName "gA" n
      let gB := freshName "gB" n
      let adjHb := freshName "adjHb" (n + 2)
      have hHeadSeed :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++
              [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))])
            (Term.var gA)
            (cotangentType (Typ.tensor ds))
            []
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++
              [(gA, none), (gB, some (Typ.tensor ds))]) := by
        simpa [gA, gB, cotangentType, List.append_assoc, linearCtx_append2_aux] using
          (HasType.var Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixOut)
            ([(gB, some (Typ.tensor ds))] : LinearCtx)
            gA
            (Typ.tensor ds))
      obtain ⟨slotMid, hslotMid, hHeadTyped⟩ :=
        ihHead
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))])
          (suffixOut := suffixOut ++ [(gA, none), (gB, some (Typ.tensor ds))])
          (x := param)
          (gSeed := Term.var gA)
          (n := n + 3)
          (slotIn := slotIn)
          (by simpa [List.append_assoc, gA, gB] using hHeadSeed)
      have hRestSeed :
          HasType Delta Sigma
            (Gamma ++ [(param, slotMid)] ++
              suffixOut ++ [(gA, none), (gB, some (Typ.tensor ds))] ++
              [(adjHb, some Typ.unit)])
            (Term.var gB)
            (cotangentType (Typ.tensor ds))
            []
            (Gamma ++ [(param, slotMid)] ++
              suffixOut ++ [(gA, none), (gB, none)] ++
              [(adjHb, some Typ.unit)]) := by
        refine Eq.mp ?_
          (HasType.var Delta Sigma
            (Gamma ++ [(param, slotMid)] ++ suffixOut ++ [(gA, none)])
            ([(adjHb, some Typ.unit)] : LinearCtx)
            gB
            (Typ.tensor ds))
        simp [gA, gB, adjHb, cotangentType, List.append_assoc,
          linearCtx_suffix_pair, linearCtx_append3_aux]
      obtain ⟨slotOut, hslotOut, hRestTyped⟩ :=
        ihRest
          (Gamma := Gamma)
          (suffixIn := suffixOut ++
            [(gA, none), (gB, some (Typ.tensor ds)), (adjHb, some Typ.unit)])
          (suffixOut := suffixOut ++
            [(gA, none), (gB, none), (adjHb, some Typ.unit)])
          (x := param)
          (body' := body)
          (gSeed := Term.var gB)
          (n := n + 3)
          (slotIn := slotMid)
          ihBody
          (by simpa [List.append_assoc, gA, gB, adjHb] using hRestSeed)
      have hSeq :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++
              suffixOut ++ [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))])
            (Term.letBind adjHb
              (adjointTypedFrom hb (Typ.tensor ds) param (Term.var gA) (n + 3))
              (adjointTypedClausesFrom rest param body (Typ.tensor ds) (Term.var gB)
                (n + 3)))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++
              suffixOut ++ [(gA, none), (gB, none)]) := by
        exact adjointTyped_seq_typed Delta Sigma
          (Gamma ++ [(param, slotIn)] ++
            suffixOut ++ [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))])
          (Gamma ++ [(param, slotMid)] ++
            suffixOut ++ [(gA, none), (gB, some (Typ.tensor ds))])
          (Gamma ++ [(param, slotOut)] ++
            suffixOut ++ [(gA, none), (gB, none)])
          adjHb
          (adjointTypedFrom hb (Typ.tensor ds) param (Term.var gA) (n + 3))
          (adjointTypedClausesFrom rest param body (Typ.tensor ds) (Term.var gB) (n + 3))
          (some Typ.unit)
          (by simpa [List.append_assoc] using hHeadTyped)
          (by simpa [List.append_assoc] using hRestTyped)
      have hSplit :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixIn)
            (adjointTypedClausesFrom ((op, xv, kv, hb) :: rest) param body
              (Typ.tensor ds) gSeed n)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++ suffixOut) := by
        simpa [adjointTypedClausesFrom] using
          (splitCotangentSeedFrom_typed Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixIn)
            (Gamma ++ [(param, slotIn)] ++ suffixOut)
            (Typ.tensor ds)
            gSeed
            n
            (fun n' gA' gB' =>
              let adjHb' := freshName "adjHb" n'
              Term.letBind adjHb'
                (adjointTypedFrom hb (Typ.tensor ds) param gA' (n' + 1))
                (adjointTypedClausesFrom rest param body (Typ.tensor ds) gB' (n' + 1)))
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++ suffixOut)
            h_seed
            ⟨none, none, by
              simpa [gA, gB, adjHb, List.append_assoc] using hSeq⟩)
      exact ⟨slotOut, slot_choice_trans hslotMid hslotOut, hSplit⟩
  | pair t1 t2 =>
      sorry
  | unit =>
      let adjA := freshName "adjA" n
      let adjHb := freshName "adjHb" (n + 1)
      have hHeadSeed :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)])
            Term.unit
            (cotangentType Typ.unit)
            []
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)]) := by
        simpa [adjA, cotangentType, List.append_assoc] using
          (HasType.unit Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)]))
      obtain ⟨slotMid, hslotMid, hHeadTyped⟩ :=
        ihHead
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(adjA, some Typ.unit)])
          (suffixOut := suffixOut ++ [(adjA, some Typ.unit)])
          (x := param)
          (gSeed := Term.unit)
          (n := n + 2)
          (slotIn := slotIn)
          (by simpa [List.append_assoc, adjA] using hHeadSeed)
      have hRestSeed :
          HasType Delta Sigma
            (Gamma ++ [(param, slotMid)] ++
              suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)])
            Term.unit
            (cotangentType Typ.unit)
            []
            (Gamma ++ [(param, slotMid)] ++
              suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)]) := by
        simpa [adjA, adjHb, cotangentType, List.append_assoc, linearCtx_append2_aux] using
          (HasType.unit Delta Sigma
            (Gamma ++ [(param, slotMid)] ++
              suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)]))
      obtain ⟨slotOut, hslotOut, hRestTyped⟩ :=
        ihRest
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)])
          (suffixOut := suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)])
          (x := param)
          (body' := body)
          (gSeed := Term.unit)
          (n := n + 2)
          (slotIn := slotMid)
          ihBody
          (by simpa [List.append_assoc, adjA, adjHb] using hRestSeed)
      have hSeq :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)])
            (Term.letBind adjHb
              (adjointTypedFrom hb Typ.unit param Term.unit (n + 2))
              (adjointTypedClausesFrom rest param body Typ.unit Term.unit (n + 2)))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++ suffixOut ++ [(adjA, some Typ.unit)]) := by
        exact adjointTyped_seq_typed Delta Sigma
          (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)])
          (Gamma ++ [(param, slotMid)] ++ suffixOut ++ [(adjA, some Typ.unit)])
          (Gamma ++ [(param, slotOut)] ++ suffixOut ++ [(adjA, some Typ.unit)])
          adjHb
          (adjointTypedFrom hb Typ.unit param Term.unit (n + 2))
          (adjointTypedClausesFrom rest param body Typ.unit Term.unit (n + 2))
          (some Typ.unit)
          (by simpa [List.append_assoc] using hHeadTyped)
          (by simpa [List.append_assoc] using hRestTyped)
      have hSplit :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixIn)
            (adjointTypedClausesFrom ((op, xv, kv, hb) :: rest) param body Typ.unit
              gSeed n)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++ suffixOut) := by
        simpa [adjointTypedClausesFrom] using
          (splitCotangentSeedFrom_typed Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixIn)
            (Gamma ++ [(param, slotIn)] ++ suffixOut)
            Typ.unit
            gSeed
            n
            (fun n' gA' gB' =>
              let adjHb' := freshName "adjHb" n'
              Term.letBind adjHb'
                (adjointTypedFrom hb Typ.unit param gA' (n' + 1))
                (adjointTypedClausesFrom rest param body Typ.unit gB' (n' + 1)))
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++ suffixOut)
            h_seed
            ⟨some Typ.unit, by
              simpa [adjA, adjHb, List.append_assoc] using hSeq⟩)
      exact ⟨slotOut, slot_choice_trans hslotMid hslotOut, hSplit⟩
  | arrow tIn tOut eps =>
      let adjA := freshName "adjA" n
      let adjHb := freshName "adjHb" (n + 1)
      have hHeadSeed :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)])
            Term.unit
            (cotangentType (Typ.arrow tIn tOut eps))
            []
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)]) := by
        simpa [adjA, cotangentType, List.append_assoc] using
          (HasType.unit Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)]))
      obtain ⟨slotMid, hslotMid, hHeadTyped⟩ :=
        ihHead
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(adjA, some Typ.unit)])
          (suffixOut := suffixOut ++ [(adjA, some Typ.unit)])
          (x := param)
          (gSeed := Term.unit)
          (n := n + 2)
          (slotIn := slotIn)
          (by simpa [List.append_assoc, adjA] using hHeadSeed)
      have hRestSeed :
          HasType Delta Sigma
            (Gamma ++ [(param, slotMid)] ++
              suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)])
            Term.unit
            (cotangentType (Typ.arrow tIn tOut eps))
            []
            (Gamma ++ [(param, slotMid)] ++
              suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)]) := by
        simpa [adjA, adjHb, cotangentType, List.append_assoc, linearCtx_append2_aux] using
          (HasType.unit Delta Sigma
            (Gamma ++ [(param, slotMid)] ++
              suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)]))
      obtain ⟨slotOut, hslotOut, hRestTyped⟩ :=
        ihRest
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)])
          (suffixOut := suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)])
          (x := param)
          (body' := body)
          (gSeed := Term.unit)
          (n := n + 2)
          (slotIn := slotMid)
          ihBody
          (by simpa [List.append_assoc, adjA, adjHb] using hRestSeed)
      have hSeq :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)])
            (Term.letBind adjHb
              (adjointTypedFrom hb (Typ.arrow tIn tOut eps) param Term.unit (n + 2))
              (adjointTypedClausesFrom rest param body (Typ.arrow tIn tOut eps)
                Term.unit (n + 2)))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++ suffixOut ++ [(adjA, some Typ.unit)]) := by
        exact adjointTyped_seq_typed Delta Sigma
          (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)])
          (Gamma ++ [(param, slotMid)] ++ suffixOut ++ [(adjA, some Typ.unit)])
          (Gamma ++ [(param, slotOut)] ++ suffixOut ++ [(adjA, some Typ.unit)])
          adjHb
          (adjointTypedFrom hb (Typ.arrow tIn tOut eps) param Term.unit (n + 2))
          (adjointTypedClausesFrom rest param body (Typ.arrow tIn tOut eps)
            Term.unit (n + 2))
          (some Typ.unit)
          (by simpa [List.append_assoc] using hHeadTyped)
          (by simpa [List.append_assoc] using hRestTyped)
      have hSplit :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixIn)
            (adjointTypedClausesFrom ((op, xv, kv, hb) :: rest) param body
              (Typ.arrow tIn tOut eps) gSeed n)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++ suffixOut) := by
        simpa [adjointTypedClausesFrom] using
          (splitCotangentSeedFrom_typed Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixIn)
            (Gamma ++ [(param, slotIn)] ++ suffixOut)
            (Typ.arrow tIn tOut eps)
            gSeed
            n
            (fun n' gA' gB' =>
              let adjHb' := freshName "adjHb" n'
              Term.letBind adjHb'
                (adjointTypedFrom hb (Typ.arrow tIn tOut eps) param gA' (n' + 1))
                (adjointTypedClausesFrom rest param body (Typ.arrow tIn tOut eps)
                  gB' (n' + 1)))
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++ suffixOut)
            h_seed
            ⟨some Typ.unit, by
              simpa [adjA, adjHb, List.append_assoc] using hSeq⟩)
      exact ⟨slotOut, slot_choice_trans hslotMid hslotOut, hSplit⟩
  | tyVar alpha =>
      let adjA := freshName "adjA" n
      let adjHb := freshName "adjHb" (n + 1)
      have hHeadSeed :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)])
            Term.unit
            (cotangentType (Typ.tyVar alpha))
            []
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)]) := by
        simpa [adjA, cotangentType, List.append_assoc] using
          (HasType.unit Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)]))
      obtain ⟨slotMid, hslotMid, hHeadTyped⟩ :=
        ihHead
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(adjA, some Typ.unit)])
          (suffixOut := suffixOut ++ [(adjA, some Typ.unit)])
          (x := param)
          (gSeed := Term.unit)
          (n := n + 2)
          (slotIn := slotIn)
          (by simpa [List.append_assoc, adjA] using hHeadSeed)
      have hRestSeed :
          HasType Delta Sigma
            (Gamma ++ [(param, slotMid)] ++
              suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)])
            Term.unit
            (cotangentType (Typ.tyVar alpha))
            []
            (Gamma ++ [(param, slotMid)] ++
              suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)]) := by
        simpa [adjA, adjHb, cotangentType, List.append_assoc, linearCtx_append2_aux] using
          (HasType.unit Delta Sigma
            (Gamma ++ [(param, slotMid)] ++
              suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)]))
      obtain ⟨slotOut, hslotOut, hRestTyped⟩ :=
        ihRest
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)])
          (suffixOut := suffixOut ++ [(adjA, some Typ.unit), (adjHb, some Typ.unit)])
          (x := param)
          (body' := body)
          (gSeed := Term.unit)
          (n := n + 2)
          (slotIn := slotMid)
          ihBody
          (by simpa [List.append_assoc, adjA, adjHb] using hRestSeed)
      have hSeq :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)])
            (Term.letBind adjHb
              (adjointTypedFrom hb (Typ.tyVar alpha) param Term.unit (n + 2))
              (adjointTypedClausesFrom rest param body (Typ.tyVar alpha) Term.unit
                (n + 2)))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++ suffixOut ++ [(adjA, some Typ.unit)]) := by
        exact adjointTyped_seq_typed Delta Sigma
          (Gamma ++ [(param, slotIn)] ++ suffixOut ++ [(adjA, some Typ.unit)])
          (Gamma ++ [(param, slotMid)] ++ suffixOut ++ [(adjA, some Typ.unit)])
          (Gamma ++ [(param, slotOut)] ++ suffixOut ++ [(adjA, some Typ.unit)])
          adjHb
          (adjointTypedFrom hb (Typ.tyVar alpha) param Term.unit (n + 2))
          (adjointTypedClausesFrom rest param body (Typ.tyVar alpha) Term.unit
            (n + 2))
          (some Typ.unit)
          (by simpa [List.append_assoc] using hHeadTyped)
          (by simpa [List.append_assoc] using hRestTyped)
      have hSplit :
          HasType Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixIn)
            (adjointTypedClausesFrom ((op, xv, kv, hb) :: rest) param body
              (Typ.tyVar alpha) gSeed n)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++ suffixOut) := by
        simpa [adjointTypedClausesFrom] using
          (splitCotangentSeedFrom_typed Delta Sigma
            (Gamma ++ [(param, slotIn)] ++ suffixIn)
            (Gamma ++ [(param, slotIn)] ++ suffixOut)
            (Typ.tyVar alpha)
            gSeed
            n
            (fun n' gA' gB' =>
              let adjHb' := freshName "adjHb" n'
              Term.letBind adjHb'
                (adjointTypedFrom hb (Typ.tyVar alpha) param gA' (n' + 1))
                (adjointTypedClausesFrom rest param body (Typ.tyVar alpha)
                  gB' (n' + 1)))
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(param, slotOut)] ++ suffixOut)
            h_seed
            ⟨some Typ.unit, by
              simpa [adjA, adjHb, List.append_assoc] using hSeq⟩)
      exact ⟨slotOut, slot_choice_trans hslotMid hslotOut, hSplit⟩

/-
private theorem adjointTypedClauses_cons_typed
    (Delta : CapCtx) (Sigma : StoreTyp)
    {t : Typ} {op : EffectLabel} {xv kv param : String} {hb body : Term}
    {rest : List (EffectLabel × String × String × Term)}
    (ihHead :
      ∀ {Gamma_s Gamma_s' suffix : LinearCtx} {x : String} {gSeed : Term} {n : Nat},
        HasType Delta Sigma
          (Gamma_s ++ suffix) gSeed (cotangentType t) [] (Gamma_s' ++ suffix) →
        HasType Delta Sigma
          (Gamma_s ++ suffix)
          (adjointTypedFrom hb t x gSeed n)
          Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix))
    (ihRest :
      ∀ {Gamma_s Gamma_s' suffix : LinearCtx} {x : String} {body' : Term} {gSeed : Term}
          {n : Nat},
        (∀ {Gamma_s0 Gamma_s0' suffix0 : LinearCtx} {x0 : String} {gSeed0 : Term}
            {n0 : Nat},
          HasType Delta Sigma
            (Gamma_s0 ++ suffix0) gSeed0 (cotangentType t) [] (Gamma_s0' ++ suffix0) →
          HasType Delta Sigma
            (Gamma_s0 ++ suffix0)
            (adjointTypedFrom body' t x0 gSeed0 n0)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s0' ++ suffix0)) →
        HasType Delta Sigma
          (Gamma_s ++ suffix) gSeed (cotangentType t) [] (Gamma_s' ++ suffix) →
        HasType Delta Sigma
          (Gamma_s ++ suffix)
          (adjointTypedClausesFrom rest x body' t gSeed n)
          Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix))
    {Gamma_s Gamma_s' suffix : LinearCtx} {gSeed : Term} {n : Nat}
    (ihBody :
      ∀ {Gamma_s0 Gamma_s0' suffix0 : LinearCtx} {x0 : String} {gSeed0 : Term}
          {n0 : Nat},
        HasType Delta Sigma
          (Gamma_s0 ++ suffix0) gSeed0 (cotangentType t) [] (Gamma_s0' ++ suffix0) →
        HasType Delta Sigma
          (Gamma_s0 ++ suffix0)
          (adjointTypedFrom body t x0 gSeed0 n0)
          Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s0' ++ suffix0))
    (h_seed :
      HasType Delta Sigma
        (Gamma_s ++ suffix) gSeed (cotangentType t) [] (Gamma_s' ++ suffix)) :
    HasType Delta Sigma
      (Gamma_s ++ suffix)
      (adjointTypedClausesFrom ((op, xv, kv, hb) :: rest) param body t gSeed n)
      Typ.unit
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      (Gamma_s' ++ suffix) := by
  cases t with
  | tensor ds =>
      let gA := freshName "gA" n
      let gB := freshName "gB" n
      let adjHb := freshName "adjHb" (n + 2)
      have hHeadSeed :
          HasType Delta Sigma
            (((Gamma_s' ++ suffix) ++ [(gA, some (Typ.tensor ds))]) ++
              [(gB, some (Typ.tensor ds))])
            (Term.var gA)
            (cotangentType (Typ.tensor ds))
            []
            (((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor ds))]) := by
        simpa [gA, gB, cotangentType, List.append_assoc] using
          (hasType_var_with_suffix Delta Sigma
            (Gamma_s' ++ suffix)
            ([(gB, some (Typ.tensor ds))] : LinearCtx)
            gA
            (Typ.tensor ds))
      have hHeadTyped :
          HasType Delta Sigma
            (((Gamma_s' ++ suffix) ++ [(gA, some (Typ.tensor ds))]) ++
              [(gB, some (Typ.tensor ds))])
            (adjointTypedFrom hb (Typ.tensor ds) param (Term.var gA) (n + 3))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor ds))]) := by
        simpa [gA, gB, List.append_assoc] using
          (ihHead
            (Gamma_s := (Gamma_s' ++ suffix) ++ [(gA, some (Typ.tensor ds))])
            (Gamma_s' := (Gamma_s' ++ suffix) ++ [(gA, none)])
            (suffix := ([(gB, some (Typ.tensor ds))] : LinearCtx))
            (x := param)
            (gSeed := Term.var gA)
            (n := n + 3)
            hHeadSeed)
      have hRestSeed :
          HasType Delta Sigma
            ((((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
                [(gB, some (Typ.tensor ds))]) ++
              [(adjHb, some Typ.unit)])
            (Term.var gB)
            (cotangentType (Typ.tensor ds))
            []
            ((((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
                [(gB, none)]) ++
              [(adjHb, some Typ.unit)]) := by
        refine Eq.mp ?_
          (hasType_var_with_suffix Delta Sigma
            ((Gamma_s' ++ suffix) ++ [(gA, none)])
            ([(adjHb, some Typ.unit)] : LinearCtx)
            gB
            (Typ.tensor ds))
        simp [gA, gB, adjHb, cotangentType, List.append_assoc, linearCtx_append3_aux]
      have hRestTyped :
          HasType Delta Sigma
            ((((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
                [(gB, some (Typ.tensor ds))]) ++
              [(adjHb, some Typ.unit)])
            (adjointTypedClausesFrom rest param body (Typ.tensor ds) (Term.var gB) (n + 3))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            ((((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
                [(gB, none)]) ++
              [(adjHb, some Typ.unit)]) := by
        simpa [gA, gB, adjHb, List.append_assoc] using
          (ihRest
            (Gamma_s := ((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor ds))])
            (Gamma_s' := ((Gamma_s' ++ suffix) ++ [(gA, none)]) ++ [(gB, none)])
            (suffix := ([(adjHb, some Typ.unit)] : LinearCtx))
            (x := param)
            (body' := body)
            (gSeed := Term.var gB)
            (n := n + 3)
            ihBody
            hRestSeed)
      have hSeq :
          HasType Delta Sigma
            (((Gamma_s' ++ suffix) ++ [(gA, some (Typ.tensor ds))]) ++
              [(gB, some (Typ.tensor ds))])
            (Term.letBind adjHb
              (adjointTypedFrom hb (Typ.tensor ds) param (Term.var gA) (n + 3))
              (adjointTypedClausesFrom rest param body (Typ.tensor ds) (Term.var gB)
                (n + 3)))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
              [(gB, none)]) := by
        exact adjointTyped_seq_typed Delta Sigma
          ((((Gamma_s' ++ suffix) ++ [(gA, some (Typ.tensor ds))]) ++
            [(gB, some (Typ.tensor ds))]))
          ((((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
            [(gB, some (Typ.tensor ds))]))
          ((((Gamma_s' ++ suffix) ++ [(gA, none)]) ++ [(gB, none)]))
          adjHb
          (adjointTypedFrom hb (Typ.tensor ds) param (Term.var gA) (n + 3))
          (adjointTypedClausesFrom rest param body (Typ.tensor ds) (Term.var gB) (n + 3))
          (some Typ.unit)
          hHeadTyped
          hRestTyped
      have hSplit :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (splitCotangentSeedFrom (Typ.tensor ds) gSeed n
              (fun n' gA' gB' =>
                let adjHb' := freshName "adjHb" n'
                Term.letBind adjHb'
                  (adjointTypedFrom hb (Typ.tensor ds) param gA' (n' + 1))
                  (adjointTypedClausesFrom rest param body (Typ.tensor ds) gB' (n' + 1))))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s' ++ suffix) := by
        exact splitCotangentSeedFrom_typed Delta Sigma
          (Gamma_s ++ suffix)
          (Gamma_s' ++ suffix)
          (Typ.tensor ds)
          gSeed
          n
          (fun n' gA' gB' =>
            let adjHb' := freshName "adjHb" n'
            Term.letBind adjHb'
              (adjointTypedFrom hb (Typ.tensor ds) param gA' (n' + 1))
              (adjointTypedClausesFrom rest param body (Typ.tensor ds) gB' (n' + 1)))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix)
          h_seed
          ⟨none, none, by simpa [gA, gB, adjHb, List.append_assoc] using hSeq⟩
      simpa [adjointTypedClausesFrom] using hSplit
  | pair t1 t2 =>
      sorry
  | unit =>
      let adjA := freshName "adjA" n
      let adjHb := freshName "adjHb" (n + 1)
      have hHeadSeed :
          HasType Delta Sigma
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            Term.unit
            (cotangentType Typ.unit)
            []
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) := by
        simpa [adjA, cotangentType, List.append_assoc] using
          (HasType.unit Delta Sigma (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])))
      have hHeadTyped :
          HasType Delta Sigma
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (adjointTypedFrom hb Typ.unit param Term.unit (n + 2))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) := by
        simpa [adjA, List.append_assoc] using
          (ihHead
            (Gamma_s := (Gamma_s' ++ suffix))
            (Gamma_s' := (Gamma_s' ++ suffix))
            (suffix := ([(adjA, some Typ.unit)] : LinearCtx))
            (x := param)
            (gSeed := Term.unit)
            (n := n + 2)
            hHeadSeed)
      have hRestSeed :
          HasType Delta Sigma
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)])
            Term.unit
            (cotangentType Typ.unit)
            []
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)]) := by
        simpa [adjA, adjHb, cotangentType, List.append_assoc] using
          (HasType.unit Delta Sigma
            ((((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)])))
      have hRestTyped :
          HasType Delta Sigma
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)])
            (adjointTypedClausesFrom rest param body Typ.unit Term.unit (n + 2))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)]) := by
        simpa [adjA, adjHb, List.append_assoc] using
          (ihRest
            (Gamma_s := (Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (Gamma_s' := (Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (suffix := ([(adjHb, some Typ.unit)] : LinearCtx))
            (x := param)
            (body' := body)
            (gSeed := Term.unit)
            (n := n + 2)
            ihBody
            hRestSeed)
      have hSeq :
          HasType Delta Sigma
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (Term.letBind adjHb
              (adjointTypedFrom hb Typ.unit param Term.unit (n + 2))
              (adjointTypedClausesFrom rest param body Typ.unit Term.unit (n + 2)))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) := by
        exact adjointTyped_seq_typed Delta Sigma
          (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]))
          (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]))
          (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]))
          adjHb
          (adjointTypedFrom hb Typ.unit param Term.unit (n + 2))
          (adjointTypedClausesFrom rest param body Typ.unit Term.unit (n + 2))
          (some Typ.unit)
          hHeadTyped
          hRestTyped
      have hSplit :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (splitCotangentSeedFrom Typ.unit gSeed n
              (fun n' gA' gB' =>
                let adjHb' := freshName "adjHb" n'
                Term.letBind adjHb'
                  (adjointTypedFrom hb Typ.unit param gA' (n' + 1))
                  (adjointTypedClausesFrom rest param body Typ.unit gB' (n' + 1))))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s' ++ suffix) := by
        exact splitCotangentSeedFrom_typed Delta Sigma
          (Gamma_s ++ suffix)
          (Gamma_s' ++ suffix)
          Typ.unit
          gSeed
          n
          (fun n' gA' gB' =>
            let adjHb' := freshName "adjHb" n'
            Term.letBind adjHb'
              (adjointTypedFrom hb Typ.unit param gA' (n' + 1))
              (adjointTypedClausesFrom rest param body Typ.unit gB' (n' + 1)))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix)
          h_seed
          ⟨some Typ.unit, by simpa [adjA, adjHb, List.append_assoc] using hSeq⟩
      simpa [adjointTypedClausesFrom] using hSplit
  | arrow tIn tOut eps =>
      let adjA := freshName "adjA" n
      let adjHb := freshName "adjHb" (n + 1)
      have hHeadSeed :
          HasType Delta Sigma
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            Term.unit
            (cotangentType (Typ.arrow tIn tOut eps))
            []
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) := by
        simpa [adjA, cotangentType, List.append_assoc] using
          (HasType.unit Delta Sigma (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])))
      have hHeadTyped :
          HasType Delta Sigma
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (adjointTypedFrom hb (Typ.arrow tIn tOut eps) param Term.unit (n + 2))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) := by
        simpa [adjA, List.append_assoc] using
          (ihHead
            (Gamma_s := (Gamma_s' ++ suffix))
            (Gamma_s' := (Gamma_s' ++ suffix))
            (suffix := ([(adjA, some Typ.unit)] : LinearCtx))
            (x := param)
            (gSeed := Term.unit)
            (n := n + 2)
            hHeadSeed)
      have hRestSeed :
          HasType Delta Sigma
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)])
            Term.unit
            (cotangentType (Typ.arrow tIn tOut eps))
            []
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)]) := by
        simpa [adjA, adjHb, cotangentType, List.append_assoc] using
          (HasType.unit Delta Sigma
            ((((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)])))
      have hRestTyped :
          HasType Delta Sigma
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)])
            (adjointTypedClausesFrom rest param body (Typ.arrow tIn tOut eps) Term.unit
              (n + 2))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)]) := by
        simpa [adjA, adjHb, List.append_assoc] using
          (ihRest
            (Gamma_s := (Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (Gamma_s' := (Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (suffix := ([(adjHb, some Typ.unit)] : LinearCtx))
            (x := param)
            (body' := body)
            (gSeed := Term.unit)
            (n := n + 2)
            ihBody
            hRestSeed)
      have hSeq :
          HasType Delta Sigma
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (Term.letBind adjHb
              (adjointTypedFrom hb (Typ.arrow tIn tOut eps) param Term.unit (n + 2))
              (adjointTypedClausesFrom rest param body (Typ.arrow tIn tOut eps) Term.unit
                (n + 2)))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) := by
        exact adjointTyped_seq_typed Delta Sigma
          (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]))
          (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]))
          (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]))
          adjHb
          (adjointTypedFrom hb (Typ.arrow tIn tOut eps) param Term.unit (n + 2))
          (adjointTypedClausesFrom rest param body (Typ.arrow tIn tOut eps) Term.unit
            (n + 2))
          (some Typ.unit)
          hHeadTyped
          hRestTyped
      have hSplit :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (splitCotangentSeedFrom (Typ.arrow tIn tOut eps) gSeed n
              (fun n' gA' gB' =>
                let adjHb' := freshName "adjHb" n'
                Term.letBind adjHb'
                  (adjointTypedFrom hb (Typ.arrow tIn tOut eps) param gA' (n' + 1))
                  (adjointTypedClausesFrom rest param body (Typ.arrow tIn tOut eps) gB'
                    (n' + 1))))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s' ++ suffix) := by
        exact splitCotangentSeedFrom_typed Delta Sigma
          (Gamma_s ++ suffix)
          (Gamma_s' ++ suffix)
          (Typ.arrow tIn tOut eps)
          gSeed
          n
          (fun n' gA' gB' =>
            let adjHb' := freshName "adjHb" n'
            Term.letBind adjHb'
              (adjointTypedFrom hb (Typ.arrow tIn tOut eps) param gA' (n' + 1))
              (adjointTypedClausesFrom rest param body (Typ.arrow tIn tOut eps) gB'
                (n' + 1)))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix)
          h_seed
          ⟨some Typ.unit, by simpa [adjA, adjHb, List.append_assoc] using hSeq⟩
      simpa [adjointTypedClausesFrom] using hSplit
  | tyVar alpha =>
      let adjA := freshName "adjA" n
      let adjHb := freshName "adjHb" (n + 1)
      have hHeadSeed :
          HasType Delta Sigma
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            Term.unit
            (cotangentType (Typ.tyVar alpha))
            []
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) := by
        simpa [adjA, cotangentType, List.append_assoc] using
          (HasType.unit Delta Sigma (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])))
      have hHeadTyped :
          HasType Delta Sigma
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (adjointTypedFrom hb (Typ.tyVar alpha) param Term.unit (n + 2))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) := by
        simpa [adjA, List.append_assoc] using
          (ihHead
            (Gamma_s := (Gamma_s' ++ suffix))
            (Gamma_s' := (Gamma_s' ++ suffix))
            (suffix := ([(adjA, some Typ.unit)] : LinearCtx))
            (x := param)
            (gSeed := Term.unit)
            (n := n + 2)
            hHeadSeed)
      have hRestSeed :
          HasType Delta Sigma
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)])
            Term.unit
            (cotangentType (Typ.tyVar alpha))
            []
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)]) := by
        simpa [adjA, adjHb, cotangentType, List.append_assoc] using
          (HasType.unit Delta Sigma
            ((((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)])))
      have hRestTyped :
          HasType Delta Sigma
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)])
            (adjointTypedClausesFrom rest param body (Typ.tyVar alpha) Term.unit (n + 2))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) ++
              [(adjHb, some Typ.unit)]) := by
        simpa [adjA, adjHb, List.append_assoc] using
          (ihRest
            (Gamma_s := (Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (Gamma_s' := (Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (suffix := ([(adjHb, some Typ.unit)] : LinearCtx))
            (x := param)
            (body' := body)
            (gSeed := Term.unit)
            (n := n + 2)
            ihBody
            hRestSeed)
      have hSeq :
          HasType Delta Sigma
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)])
            (Term.letBind adjHb
              (adjointTypedFrom hb (Typ.tyVar alpha) param Term.unit (n + 2))
              (adjointTypedClausesFrom rest param body (Typ.tyVar alpha) Term.unit (n + 2)))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            ((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]) := by
        exact adjointTyped_seq_typed Delta Sigma
          (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]))
          (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]))
          (((Gamma_s' ++ suffix) ++ [(adjA, some Typ.unit)]))
          adjHb
          (adjointTypedFrom hb (Typ.tyVar alpha) param Term.unit (n + 2))
          (adjointTypedClausesFrom rest param body (Typ.tyVar alpha) Term.unit (n + 2))
          (some Typ.unit)
          hHeadTyped
          hRestTyped
      have hSplit :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (splitCotangentSeedFrom (Typ.tyVar alpha) gSeed n
              (fun n' gA' gB' =>
                let adjHb' := freshName "adjHb" n'
                Term.letBind adjHb'
                  (adjointTypedFrom hb (Typ.tyVar alpha) param gA' (n' + 1))
                  (adjointTypedClausesFrom rest param body (Typ.tyVar alpha) gB'
                    (n' + 1))))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s' ++ suffix) := by
        exact splitCotangentSeedFrom_typed Delta Sigma
          (Gamma_s ++ suffix)
          (Gamma_s' ++ suffix)
          (Typ.tyVar alpha)
          gSeed
          n
          (fun n' gA' gB' =>
            let adjHb' := freshName "adjHb" n'
            Term.letBind adjHb'
              (adjointTypedFrom hb (Typ.tyVar alpha) param gA' (n' + 1))
              (adjointTypedClausesFrom rest param body (Typ.tyVar alpha) gB'
                (n' + 1)))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix)
          h_seed
          ⟨some Typ.unit, by simpa [adjA, adjHb, List.append_assoc] using hSeq⟩
      simpa [adjointTypedClausesFrom] using hSplit
-/

private theorem adjointTypedShape_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) :
    ∀ {t : Typ} {e : Term},
      AdjointTypedShape (Capability.diff :: Delta) Sigma t e →
      ∀ {Gamma suffixIn suffixOut : LinearCtx} {x : String} {gSeed : Term}
        {n : Nat} {slotIn : Option Typ},
        HasType Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          gSeed
          (cotangentType t)
          []
          (Gamma ++ [(x, slotIn)] ++ suffixOut) →
        ∃ slotOut,
          (slotOut = none ∨ slotOut = slotIn) ∧
          HasType Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (adjointTypedFrom e t x gSeed n)
          Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma ++ [(x, slotOut)] ++ suffixOut) := by
  intro t e hShape
  induction hShape using AdjointTypedShape.rec
    (motive_2 := fun t clauses _ =>
      ∀ {Gamma suffixIn suffixOut : LinearCtx} {x : String} {body gSeed : Term}
          {n : Nat} {slotIn : Option Typ},
        (∀ {Gamma0 suffixIn0 suffixOut0 : LinearCtx} {x0 : String} {gSeed0 : Term}
            {n0 : Nat} {slotIn0 : Option Typ},
          HasType Delta Sigma
            (Gamma0 ++ [(x0, slotIn0)] ++ suffixIn0)
            gSeed0
            (cotangentType t)
            []
            (Gamma0 ++ [(x0, slotIn0)] ++ suffixOut0) →
          ∃ slotOut0,
            (slotOut0 = none ∨ slotOut0 = slotIn0) ∧
            HasType Delta Sigma
              (Gamma0 ++ [(x0, slotIn0)] ++ suffixIn0)
              (adjointTypedFrom body t x0 gSeed0 n0)
              Typ.unit
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
              (Gamma0 ++ [(x0, slotOut0)] ++ suffixOut0)) →
        HasType Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          gSeed
          (cotangentType t)
          []
          (Gamma ++ [(x, slotIn)] ++ suffixOut) →
        ∃ slotOut,
          (slotOut = none ∨ slotOut = slotIn) ∧
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (adjointTypedClausesFrom clauses x body t gSeed n)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(x, slotOut)] ++ suffixOut))
  with
  | var =>
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      refine ⟨slotIn, Or.inr rfl, ?_⟩
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          _ [] n gSeed h_seed)
  | const =>
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      refine ⟨slotIn, Or.inr rfl, ?_⟩
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          _ [] n gSeed h_seed)
  | unit =>
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      refine ⟨slotIn, Or.inr rfl, ?_⟩
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          _ [] n gSeed h_seed)
  | loc =>
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      refine ⟨slotIn, Or.inr rfl, ?_⟩
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          _ [] n gSeed h_seed)
  | letBind hBody ih =>
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      simpa [adjointTypedFrom] using
        (ih
          (Gamma := Gamma)
          (suffixIn := suffixIn)
          (suffixOut := suffixOut)
          (x := x)
          (gSeed := gSeed)
          (n := n)
          (slotIn := slotIn)
          h_seed)
  | letpair hBody ih =>
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      simpa [adjointTypedFrom] using
        (ih
          (Gamma := Gamma)
          (suffixIn := suffixIn)
          (suffixOut := suffixOut)
          (x := x)
          (gSeed := gSeed)
          (n := n)
          (slotIn := slotIn)
          h_seed)
  | add =>
      rename_i ds e1 e2 h1 h2 ih1 ih2
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      let gA := freshName "gA" n
      let gB := freshName "gB" n
      let adjA := freshName "adjA" (n + 2)
      have hVarA :
          HasType Delta Sigma
            (((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, some (Typ.tensor ds))]) ++
              [(gB, some (Typ.tensor ds))])
            (Term.var gA)
            (Typ.tensor ds)
            []
            (((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor ds))]) := by
        simpa [gA, gB, List.append_assoc] using
          (hasType_var_with_suffix Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixOut)
            ([(gB, some (Typ.tensor ds))] : LinearCtx)
            gA
            (Typ.tensor ds))
      obtain ⟨slotMid, hslotMid, hAdj1⟩ :=
        ih1
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))])
          (suffixOut := suffixOut ++ [(gA, none), (gB, some (Typ.tensor ds))])
          (x := x)
          (gSeed := Term.var gA)
          (n := n + 3)
          (slotIn := slotIn)
          (by simpa [List.append_assoc, gA, gB] using hVarA)
      have hVarB :
          HasType Delta Sigma
            (Gamma ++ [(x, slotMid)] ++
              suffixOut ++ [(gA, none), (gB, some (Typ.tensor ds))] ++
              [(adjA, some Typ.unit)])
            (Term.var gB)
            (Typ.tensor ds)
            []
            (Gamma ++ [(x, slotMid)] ++
              suffixOut ++ [(gA, none), (gB, none)] ++
              [(adjA, some Typ.unit)]) := by
        refine Eq.mp ?_
          (HasType.var Delta Sigma
            (Gamma ++ [(x, slotMid)] ++ suffixOut ++ [(gA, none)])
            [(adjA, some Typ.unit)]
            gB
            (Typ.tensor ds))
        simp [gA, gB, adjA, List.append_assoc, linearCtx_suffix_pair, linearCtx_append3_aux]
      obtain ⟨slotOut, hslotOut, hAdj2⟩ :=
        ih2
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(gA, none), (gB, some (Typ.tensor ds)), (adjA, some Typ.unit)])
          (suffixOut := suffixOut ++ [(gA, none), (gB, none), (adjA, some Typ.unit)])
          (x := x)
          (gSeed := Term.var gB)
          (n := n + 3)
          (slotIn := slotMid)
          (by simpa [List.append_assoc, gA, gB, adjA] using hVarB)
      have hSeq :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++
              (suffixOut ++ [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))]))
            (Term.letBind adjA
              (adjointTypedFrom e1 (Typ.tensor ds) x (Term.var gA) (n + 3))
              (adjointTypedFrom e2 (Typ.tensor ds) x (Term.var gB) (n + 3)))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(x, slotOut)] ++
              (suffixOut ++ [(gA, none), (gB, none)])) := by
        exact
          adjointTyped_seq_typed Delta Sigma
            (Gamma ++ [(x, slotIn)] ++
              (suffixOut ++ [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))]))
            (Gamma ++ [(x, slotMid)] ++
              (suffixOut ++ [(gA, none), (gB, some (Typ.tensor ds))]))
            (Gamma ++ [(x, slotOut)] ++
              (suffixOut ++ [(gA, none), (gB, none)]))
            adjA
            (adjointTypedFrom e1 (Typ.tensor ds) x (Term.var gA) (n + 3))
            (adjointTypedFrom e2 (Typ.tensor ds) x (Term.var gB) (n + 3))
            (some Typ.unit)
            hAdj1
            (by simpa [List.append_assoc] using hAdj2)
      have hSplit :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (splitCotangentSeedFrom (Typ.tensor ds) gSeed n
              (fun n' gA' gB' =>
                let adjA' := freshName "adjA" n'
                Term.letBind adjA'
                  (adjointTypedFrom e1 (Typ.tensor ds) x gA' (n' + 1))
                  (adjointTypedFrom e2 (Typ.tensor ds) x gB' (n' + 1))))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(x, slotOut)] ++ suffixOut) := by
        exact splitCotangentSeedFrom_typed Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          (Typ.tensor ds)
          gSeed
          n
          (fun n' gA' gB' =>
            let adjA' := freshName "adjA" n'
            Term.letBind adjA'
              (adjointTypedFrom e1 (Typ.tensor ds) x gA' (n' + 1))
              (adjointTypedFrom e2 (Typ.tensor ds) x gB' (n' + 1)))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma ++ [(x, slotOut)] ++ suffixOut)
          h_seed
          ⟨none, none, by simpa [gA, gB, adjA, List.append_assoc] using hSeq⟩
      exact ⟨slotOut, slot_choice_trans hslotMid hslotOut, by
        simpa [adjointTypedFrom] using hSplit⟩
  | mul =>
      rename_i ds e1 e2 hMul h1 h2 ih1 ih2
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      sorry
  | sum =>
      rename_i ds d e hmem hBody ih
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      have hExpand :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (Term.expand gSeed d)
            (cotangentType (Typ.tensor ds))
            []
            (Gamma ++ [(x, slotIn)] ++ suffixOut) := by
        have := HasType.texpand Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          gSeed
          (rem ds d)
          d
          []
          h_seed
        simpa [cotangentType, ins_rem_eq_of_mem hmem] using this
      simpa [adjointTypedFrom, ins_rem_eq_of_mem hmem] using
        (ih
          (Gamma := Gamma)
          (suffixIn := suffixIn)
          (suffixOut := suffixOut)
          (x := x)
          (gSeed := Term.expand gSeed d)
          (n := n)
          (slotIn := slotIn)
          hExpand)
  | expand =>
      rename_i ds d e hBody ih
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      have hmem : d ∈ ins ds d := by
        refine Quotient.inductionOn ds ?_
        intro l
        change d ∈ (d :: l)
        simp
      have hSum :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (Term.sum gSeed d)
            (cotangentType (Typ.tensor ds))
            []
            (Gamma ++ [(x, slotIn)] ++ suffixOut) := by
        have := HasType.tsum Delta Sigma
          (Gamma ++ [(x, slotIn)] ++ suffixIn)
          (Gamma ++ [(x, slotIn)] ++ suffixOut)
          gSeed
          (ins ds d)
          d
          []
          h_seed
          hmem
        simpa [cotangentType, rem_ins_eq] using this
      simpa [adjointTypedFrom, rem_ins_eq] using
        (ih
          (Gamma := Gamma)
          (suffixIn := suffixIn)
          (suffixOut := suffixOut)
          (x := x)
          (gSeed := Term.sum gSeed d)
          (n := n)
          (slotIn := slotIn)
          hSum)
  | copy =>
      rename_i ds e hBody ih
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      let gA := freshName "gA" n
      let gB := freshName "gB" n
      have hVarA :
          HasType Delta Sigma
            (((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, some (Typ.tensor ds))]) ++
              [(gB, some (Typ.tensor ds))])
            (Term.var gA)
            (Typ.tensor ds)
            []
            (((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor ds))]) := by
        simpa [gA, gB, List.append_assoc] using
          (hasType_var_with_suffix Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixOut)
            ([(gB, some (Typ.tensor ds))] : LinearCtx)
            gA
            (Typ.tensor ds))
      have hVarB :
          HasType Delta Sigma
            (((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor ds))])
            (Term.var gB)
            (Typ.tensor ds)
            []
            (((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, none)]) ++
              [(gB, none)]) := by
        refine Eq.mp ?_
          (HasType.var Delta Sigma
            (((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, none)]))
            []
            gB
            (Typ.tensor ds))
        simp [gA, gB, List.append_assoc, linearCtx_suffix_pair]
      have hAddSeed :
          HasType Delta Sigma
            (((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, some (Typ.tensor ds))]) ++
              [(gB, some (Typ.tensor ds))])
            (Term.add (Term.var gA) (Term.var gB))
            (Typ.tensor ds)
            []
            (((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, none)]) ++
              [(gB, none)]) := by
        exact HasType.tadd Delta Sigma
          ((((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, some (Typ.tensor ds))]) ++
            [(gB, some (Typ.tensor ds))]))
          ((((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, none)]) ++
            [(gB, some (Typ.tensor ds))]))
          ((((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, none)]) ++
            [(gB, none)]))
          (Term.var gA) (Term.var gB) ds [] [] hVarA hVarB
      obtain ⟨slotOut, hslotOut, hAdj⟩ :=
        ih
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(gA, some (Typ.tensor ds)), (gB, some (Typ.tensor ds))])
          (suffixOut := suffixOut ++ [(gA, none), (gB, none)])
          (x := x)
          (gSeed := Term.add (Term.var gA) (Term.var gB))
          (n := n + 3)
          (slotIn := slotIn)
          (by simpa [List.append_assoc, gA, gB] using hAddSeed)
      have hPair :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (adjointTypedFrom (Term.copy e)
              (Typ.pair (Typ.tensor ds) (Typ.tensor ds)) x gSeed n)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(x, slotOut)] ++ suffixOut) := by
        simpa [adjointTypedFrom, gA, gB, List.append_assoc] using
          (HasType.letpair Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (Gamma ++ [(x, slotIn)] ++ suffixOut)
            (Gamma ++ [(x, slotOut)] ++ suffixOut)
            gA gB
            gSeed
            (adjointTypedFrom e (Typ.tensor ds) x
              (Term.add (Term.var gA) (Term.var gB)) (n + 3))
            (Typ.tensor ds) (Typ.tensor ds) Typ.unit
            []
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            none none
            h_seed
            (by simpa [List.append_assoc] using hAdj))
      exact ⟨slotOut, hslotOut, hPair⟩
  | pair =>
      rename_i t1 t2 e1 e2 h1 h2 ih1 ih2
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      let gA := freshName "gA" n
      let gB := freshName "gB" n
      let adjA := freshName "adjA" n
      have hVarA :
          HasType Delta Sigma
            (((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, some (cotangentType t1))]) ++
              [(gB, some (cotangentType t2))])
            (Term.var gA)
            (cotangentType t1)
            []
            (((Gamma ++ [(x, slotIn)] ++ suffixOut) ++ [(gA, none)]) ++
              [(gB, some (cotangentType t2))]) := by
        simpa [gA, gB, List.append_assoc] using
          (hasType_var_with_suffix Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixOut)
            ([(gB, some (cotangentType t2))] : LinearCtx)
            gA
            (cotangentType t1))
      obtain ⟨slotMid, hslotMid, hAdj1⟩ :=
        ih1
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(gA, some (cotangentType t1)), (gB, some (cotangentType t2))])
          (suffixOut := suffixOut ++ [(gA, none), (gB, some (cotangentType t2))])
          (x := x)
          (gSeed := Term.var gA)
          (n := n + 3)
          (slotIn := slotIn)
          (by simpa [List.append_assoc, gA, gB] using hVarA)
      have hVarB :
          HasType Delta Sigma
            (Gamma ++ [(x, slotMid)] ++
              suffixOut ++ [(gA, none), (gB, some (cotangentType t2))] ++
              [(adjA, some Typ.unit)])
            (Term.var gB)
            (cotangentType t2)
            []
            (Gamma ++ [(x, slotMid)] ++
              suffixOut ++ [(gA, none), (gB, none)] ++
              [(adjA, some Typ.unit)]) := by
        refine Eq.mp ?_
          (HasType.var Delta Sigma
            (Gamma ++ [(x, slotMid)] ++ suffixOut ++ [(gA, none)])
            [(adjA, some Typ.unit)]
            gB
            (cotangentType t2))
        simp [gA, gB, adjA, List.append_assoc, linearCtx_suffix_pair, linearCtx_append3_aux]
      obtain ⟨slotOut, hslotOut, hAdj2⟩ :=
        ih2
          (Gamma := Gamma)
          (suffixIn := suffixOut ++ [(gA, none), (gB, some (cotangentType t2)), (adjA, some Typ.unit)])
          (suffixOut := suffixOut ++ [(gA, none), (gB, none), (adjA, some Typ.unit)])
          (x := x)
          (gSeed := Term.var gB)
          (n := n + 3)
          (slotIn := slotMid)
          (by simpa [List.append_assoc, gA, gB, adjA] using hVarB)
      have hSeq :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++
              (suffixOut ++ [(gA, some (cotangentType t1)), (gB, some (cotangentType t2))]))
            (Term.letBind adjA
              (adjointTypedFrom e1 t1 x (Term.var gA) (n + 3))
              (adjointTypedFrom e2 t2 x (Term.var gB) (n + 3)))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(x, slotOut)] ++
              (suffixOut ++ [(gA, none), (gB, none)])) := by
        exact
          adjointTyped_seq_typed Delta Sigma
            (Gamma ++ [(x, slotIn)] ++
              (suffixOut ++ [(gA, some (cotangentType t1)), (gB, some (cotangentType t2))]))
            (Gamma ++ [(x, slotMid)] ++
              (suffixOut ++ [(gA, none), (gB, some (cotangentType t2))]))
            (Gamma ++ [(x, slotOut)] ++
              (suffixOut ++ [(gA, none), (gB, none)]))
            adjA
            (adjointTypedFrom e1 t1 x (Term.var gA) (n + 3))
            (adjointTypedFrom e2 t2 x (Term.var gB) (n + 3))
            (some Typ.unit)
            hAdj1
            (by simpa [List.append_assoc] using hAdj2)
      have hPair :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (adjointTypedFrom (Term.pair e1 e2) (Typ.pair t1 t2) x gSeed n)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma ++ [(x, slotOut)] ++ suffixOut) := by
        simpa [adjointTypedFrom, cotangentType, gA, gB, adjA, List.append_assoc] using
          (HasType.letpair Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (Gamma ++ [(x, slotIn)] ++ suffixOut)
            (Gamma ++ [(x, slotOut)] ++ suffixOut)
            gA gB
            gSeed
            (Term.letBind adjA
              (adjointTypedFrom e1 t1 x (Term.var gA) (n + 3))
              (adjointTypedFrom e2 t2 x (Term.var gB) (n + 3)))
            (cotangentType t1) (cotangentType t2) Typ.unit
            []
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            none none
            h_seed
            (by simpa [List.append_assoc] using hSeq))
      exact ⟨slotOut, slot_choice_trans hslotMid hslotOut, hPair⟩
  | fst =>
      rename_i t1 t2 e hBody ih
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      have hZero :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixOut)
            (zeroCotangent t2)
            (cotangentType t2)
            []
            (Gamma ++ [(x, slotIn)] ++ suffixOut) :=
        zeroCotangent_typed Delta Sigma (Gamma ++ [(x, slotIn)] ++ suffixOut) t2
      have hPairSeed :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (Term.pair gSeed (zeroCotangent t2))
            (cotangentType (Typ.pair t1 t2))
            []
            (Gamma ++ [(x, slotIn)] ++ suffixOut) := by
        simpa [cotangentType] using
          (HasType.tpair Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (Gamma ++ [(x, slotIn)] ++ suffixOut)
            (Gamma ++ [(x, slotIn)] ++ suffixOut)
            gSeed (zeroCotangent t2)
            (cotangentType t1) (cotangentType t2)
            [] [] h_seed hZero)
      simpa [adjointTypedFrom] using
        (ih
          (Gamma := Gamma)
          (suffixIn := suffixIn)
          (suffixOut := suffixOut)
          (x := x)
          (gSeed := Term.pair gSeed (zeroCotangent t2))
          (n := n)
          (slotIn := slotIn)
          hPairSeed)
  | snd =>
      rename_i t1 t2 e hBody ih
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      have hZero :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (zeroCotangent t1)
            (cotangentType t1)
            []
            (Gamma ++ [(x, slotIn)] ++ suffixIn) :=
        zeroCotangent_typed Delta Sigma (Gamma ++ [(x, slotIn)] ++ suffixIn) t1
      have hPairSeed :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (Term.pair (zeroCotangent t1) gSeed)
            (cotangentType (Typ.pair t1 t2))
            []
            (Gamma ++ [(x, slotIn)] ++ suffixOut) := by
        simpa [cotangentType] using
          (HasType.tpair Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            (Gamma ++ [(x, slotIn)] ++ suffixOut)
            (zeroCotangent t1) gSeed
            (cotangentType t1) (cotangentType t2)
            [] [] hZero h_seed)
      simpa [adjointTypedFrom] using
        (ih
          (Gamma := Gamma)
          (suffixIn := suffixIn)
          (suffixOut := suffixOut)
          (x := x)
          (gSeed := Term.pair (zeroCotangent t1) gSeed)
          (n := n)
          (slotIn := slotIn)
          hPairSeed)
  | handle =>
      rename_i t epsH body clauses hBody hClauses ihBody ihClauses
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      simpa [adjointTypedFrom] using
        (ihClauses
          (Gamma := Gamma)
          (suffixIn := suffixIn)
          (suffixOut := suffixOut)
          (x := x)
          (body := body)
          (gSeed := gSeed)
          (n := n)
          (slotIn := slotIn)
          (fun {Gamma0 suffixIn0 suffixOut0 x0 gSeed0 n0 slotIn0} h_seed0 =>
            ihBody
              (Gamma := Gamma0)
              (suffixIn := suffixIn0)
              (suffixOut := suffixOut0)
              (x := x0)
              (gSeed := gSeed0)
              (n := n0)
              (slotIn := slotIn0)
              h_seed0)
          h_seed)
  | perform =>
      rename_i op e hBody ih
      intro Gamma suffixIn suffixOut x gSeed n slotIn h_seed
      have hSeedArg :
          HasType Delta Sigma
            (Gamma ++ [(x, slotIn)] ++ suffixIn)
            gSeed
            (cotangentType (opArgType op))
            []
            (Gamma ++ [(x, slotIn)] ++ suffixOut) := by
        cases op <;> simpa [opArgType, cotangentType] using h_seed
      simpa [adjointTypedFrom] using
        (ih
          (Gamma := Gamma)
          (suffixIn := suffixIn)
          (suffixOut := suffixOut)
          (x := x)
          (gSeed := gSeed)
          (n := n)
          (slotIn := slotIn)
          hSeedArg)
  | nil =>
      rename_i t Gamma suffixIn suffixOut x body gSeed n slotIn ihBody h_seed
      simpa [adjointTypedClausesFrom] using
        (ihBody h_seed)
  | cons =>
      rename_i tShape op xv kv hb rest hHead hRest ihHead ihRest
        Gamma suffixIn suffixOut x body gSeed n slotIn ihBody h_seed
      exact adjointTypedClauses_cons_slot_typed Delta Sigma
        (t := tShape)
        (param := x)
        (body := body)
        (ihHead := ihHead)
        (ihRest := ihRest)
        (ihBody := ihBody)
        (h_seed := h_seed)

/-
  intro t e hShape
  induction hShape using AdjointTypedShape.rec
    (motive_2 := fun t clauses _ =>
      ∀ {Gamma_s Gamma_s' suffix : LinearCtx} {x : String} {body : Term} {gSeed : Term} {n : Nat},
        (∀ {Gamma_s0 Gamma_s0' suffix0 : LinearCtx} {x0 : String} {gSeed0 : Term} {n0 : Nat},
          HasType Delta Sigma
            (Gamma_s0 ++ suffix0) gSeed0 (cotangentType t) [] (Gamma_s0' ++ suffix0) →
          HasType Delta Sigma
            (Gamma_s0 ++ suffix0)
            (adjointTypedFrom body t x0 gSeed0 n0)
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s0' ++ suffix0)) →
        HasType Delta Sigma
          (Gamma_s ++ suffix) gSeed (cotangentType t) [] (Gamma_s' ++ suffix) →
        HasType Delta Sigma
          (Gamma_s ++ suffix)
          (adjointTypedClausesFrom clauses x body t gSeed n)
          Typ.unit
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix))
  with
  | var =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          _ [] n gSeed h_seed)
  | const =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          _ [] n gSeed h_seed)
  | unit =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          _ [] n gSeed h_seed)
  | loc =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (adjointLeaf_typed Delta Sigma (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          _ [] n gSeed h_seed)
  | letBind hBody ih =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (gSeed := gSeed) (n := n) h_seed)
  | letpair hBody ih =>
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (gSeed := gSeed) (n := n) h_seed)
  | add =>
      rename_i ds e1 e2 h1 h2 ih1 ih2
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      let gA := freshName "gA" n
      let gB := freshName "gB" n
      let adjA := freshName "adjA" (n + 2)
      have hVarA :
          HasType Delta Sigma
            (((Gamma_s' ++ suffix) ++ [(gA, some (Typ.tensor ds))]) ++
              [(gB, some (Typ.tensor ds))])
            (Term.var gA)
            (Typ.tensor ds)
            []
            (((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor ds))]) := by
        simpa [gA, gB, List.append_assoc] using
          (hasType_var_with_suffix Delta Sigma
            (Gamma_s' ++ suffix)
            [(gB, some (Typ.tensor ds))]
            gA
            (Typ.tensor ds))
      have hAdj1 :
          HasType Delta Sigma
            (((Gamma_s' ++ suffix) ++ [(gA, some (Typ.tensor ds))]) ++
              [(gB, some (Typ.tensor ds))])
            (adjointTypedFrom e1 (Typ.tensor ds) x (Term.var gA) (n + 3))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor ds))]) := by
        simpa [cotangentType, gA, gB, List.append_assoc] using
          (ih1
            (Gamma_s := (Gamma_s' ++ suffix) ++ [(gA, some (Typ.tensor ds))])
            (Gamma_s' := (Gamma_s' ++ suffix) ++ [(gA, none)])
            (suffix := [(gB, some (Typ.tensor ds))])
            (x := x)
            (gSeed := Term.var gA)
            (n := n + 3)
            hVarA)
      have hVarB :
          HasType Delta Sigma
            ((((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
              [(gB, some (Typ.tensor ds))]) ++
              [(adjA, some Typ.unit)])
            (Term.var gB)
            (Typ.tensor ds)
            []
            ((((Gamma_s' ++ suffix) ++ [(gA, none)]) ++
              [(gB, none)]) ++
              [(adjA, some Typ.unit)]) := by
        exact HasType.var Delta Sigma
          (((Gamma_s' ++ suffix) ++ [(gA, none)]))
          [(adjA, some Typ.unit)]
          gB
          (Typ.tensor ds)
      have hAdj2 :
          HasType Delta Sigma
            (Gamma_s' ++ suffix ++ [(gA, none)] ++
              [(gB, some (Typ.tensor ds))] ++
              [(adjA, some Typ.unit)])
            (adjointTypedFrom e2 (Typ.tensor ds) x (Term.var gB) (n + 3))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s' ++ suffix ++ [(gA, none)] ++
              [(gB, none)] ++
              [(adjA, some Typ.unit)]) := by
        simpa [cotangentType, gA, gB, adjA, List.append_assoc] using
          (ih2
            (Gamma_s := Gamma_s' ++ suffix ++ [(gA, none)] ++
              [(gB, some (Typ.tensor ds))])
            (Gamma_s' := Gamma_s' ++ suffix ++ [(gA, none)] ++
              [(gB, none)])
            (suffix := [(adjA, some Typ.unit)])
            (x := x)
            (gSeed := Term.var gB)
            (n := n + 3)
            hVarB)
      have hSeq :
          HasType Delta Sigma
            (Gamma_s' ++ suffix ++ [(gA, some (Typ.tensor ds))] ++
              [(gB, some (Typ.tensor ds))])
            (Term.letBind adjA
              (adjointTypedFrom e1 (Typ.tensor ds) x (Term.var gA) (n + 3))
              (adjointTypedFrom e2 (Typ.tensor ds) x (Term.var gB) (n + 3)))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s' ++ suffix ++ [(gA, none)] ++
              [(gB, none)]) := by
        exact adjointTyped_seq_typed Delta Sigma
          (Gamma_s' ++ suffix ++ [(gA, some (Typ.tensor ds))] ++
            [(gB, some (Typ.tensor ds))])
          (Gamma_s' ++ suffix ++ [(gA, none)] ++
            [(gB, some (Typ.tensor ds))])
          (Gamma_s' ++ suffix ++ [(gA, none)] ++
            [(gB, none)])
          adjA
          (adjointTypedFrom e1 (Typ.tensor ds) x (Term.var gA) (n + 3))
          (adjointTypedFrom e2 (Typ.tensor ds) x (Term.var gB) (n + 3))
          (some Typ.unit)
          hAdj1
          hAdj2
      have hSplit :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (splitCotangentSeedFrom (Typ.tensor ds) gSeed n
              (fun n' gA' gB' =>
                let adjA' := freshName "adjA" n'
                Term.letBind adjA'
                  (adjointTypedFrom e1 (Typ.tensor ds) x gA' (n' + 1))
                  (adjointTypedFrom e2 (Typ.tensor ds) x gB' (n' + 1))))
            Typ.unit
            (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
            (Gamma_s' ++ suffix) := by
        exact splitCotangentSeedFrom_typed Delta Sigma
          (Gamma_s ++ suffix)
          (Gamma_s' ++ suffix)
          (Typ.tensor ds)
          gSeed
          n
          (fun n' gA' gB' =>
            let adjA' := freshName "adjA" n'
            Term.letBind adjA'
              (adjointTypedFrom e1 (Typ.tensor ds) x gA' (n' + 1))
              (adjointTypedFrom e2 (Typ.tensor ds) x gB' (n' + 1)))
          (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
          (Gamma_s' ++ suffix)
          h_seed
          ⟨none, none, by simpa [gA, gB, adjA, List.append_assoc] using hSeq⟩
      simpa [adjointTypedFrom] using hSplit
  | mul =>
      rename_i ds e1 e2 hMul h1 h2 ih1 ih2
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      sorry
  | sum =>
      rename_i ds d e hmem hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      have hExpand :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (Term.expand gSeed d)
            (cotangentType (Typ.tensor ds))
            []
            (Gamma_s' ++ suffix) := by
        have := HasType.texpand Delta Sigma
          (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          gSeed
          (rem ds d)
          d
          []
          h_seed
        simpa [cotangentType, ins_rem_eq_of_mem hmem] using this
      simpa [adjointTypedFrom, ins_rem_eq_of_mem hmem] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (gSeed := Term.expand gSeed d) (n := n) hExpand)
  | expand =>
      rename_i ds d e hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      have hmem : d ∈ ins ds d := by
        refine Quotient.inductionOn ds ?_
        intro l
        change d ∈ (d :: l)
        simp
      have hSum :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (Term.sum gSeed d)
            (cotangentType (Typ.tensor ds))
            []
            (Gamma_s' ++ suffix) := by
        have := HasType.tsum Delta Sigma
          (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
          gSeed
          (ins ds d)
          d
          []
          h_seed
          hmem
        simpa [cotangentType, rem_ins_eq] using this
      simpa [adjointTypedFrom, rem_ins_eq] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (gSeed := Term.sum gSeed d) (n := n) hSum)
  | copy =>
      rename_i ds e hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa using
        (adjointTyped_copy_typed Delta Sigma
          (e := e)
          (ih := ih)
          (Gamma_s := Gamma_s)
          (Gamma_s' := Gamma_s')
          (suffix := suffix)
          (x := x)
          (gSeed := gSeed)
          (n := n)
          h_seed)
  | pair =>
      rename_i t1 t2 e1 e2 h1 h2 ih1 ih2
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa using
        (adjointTyped_pair_typed Delta Sigma
          (e1 := e1)
          (e2 := e2)
          (ih1 := ih1)
          (ih2 := ih2)
          (Gamma_s := Gamma_s)
          (Gamma_s' := Gamma_s')
          (suffix := suffix)
          (x := x)
          (gSeed := gSeed)
          (n := n)
          h_seed)
  | fst =>
      rename_i t1 t2 e hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      have hZero :
          HasType Delta Sigma
            (Gamma_s' ++ suffix)
            (zeroCotangent t2)
            (cotangentType t2)
            []
            (Gamma_s' ++ suffix) :=
        zeroCotangent_typed Delta Sigma (Gamma_s' ++ suffix) t2
      have hPairSeed :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (Term.pair gSeed (zeroCotangent t2))
            (cotangentType (Typ.pair t1 t2))
            []
            (Gamma_s' ++ suffix) := by
        simpa [cotangentType] using
          (HasType.tpair Delta Sigma
            (Gamma_s ++ suffix) (Gamma_s' ++ suffix) (Gamma_s' ++ suffix)
            gSeed (zeroCotangent t2)
            (cotangentType t1) (cotangentType t2)
            [] [] h_seed hZero)
      simpa [adjointTypedFrom] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x)
          (gSeed := Term.pair gSeed (zeroCotangent t2))
          (n := n) hPairSeed)
  | snd =>
      rename_i t1 t2 e hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      have hZero :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (zeroCotangent t1)
            (cotangentType t1)
            []
            (Gamma_s ++ suffix) :=
        zeroCotangent_typed Delta Sigma (Gamma_s ++ suffix) t1
      have hPairSeed :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            (Term.pair (zeroCotangent t1) gSeed)
            (cotangentType (Typ.pair t1 t2))
            []
            (Gamma_s' ++ suffix) := by
        simpa [cotangentType] using
          (HasType.tpair Delta Sigma
            (Gamma_s ++ suffix) (Gamma_s ++ suffix) (Gamma_s' ++ suffix)
            (zeroCotangent t1) gSeed
            (cotangentType t1) (cotangentType t2)
            [] [] hZero h_seed)
      simpa [adjointTypedFrom] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x)
          (gSeed := Term.pair (zeroCotangent t1) gSeed)
          (n := n) hPairSeed)
  | handle =>
      rename_i t epsH body clauses hBody hClauses ihBody ihClauses
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      simpa [adjointTypedFrom] using
        (ihClauses (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (body := body) (gSeed := gSeed) (n := n)
          (fun {Gamma_s0 Gamma_s0' suffix0 x0 gSeed0 n0} h_seed0 =>
            ihBody (Gamma_s := Gamma_s0) (Gamma_s' := Gamma_s0')
              (suffix := suffix0) (x := x0) (gSeed := gSeed0) (n := n0) h_seed0)
          h_seed)
  | perform =>
      rename_i op e hBody ih
      intro Gamma_s Gamma_s' suffix x gSeed n h_seed
      have hSeedArg :
          HasType Delta Sigma
            (Gamma_s ++ suffix)
            gSeed
            (cotangentType (opArgType op))
            []
            (Gamma_s' ++ suffix) := by
        cases op <;> simpa [opArgType, cotangentType] using h_seed
      simpa [adjointTypedFrom] using
        (ih (Gamma_s := Gamma_s) (Gamma_s' := Gamma_s') (suffix := suffix)
          (x := x) (gSeed := gSeed) (n := n) hSeedArg)
  | nil =>
      rename_i Gamma_s Gamma_s' suffix x body gSeed n ihBody h_seed
      simpa [adjointTypedClausesFrom] using
        (ihBody (Gamma_s0 := Gamma_s) (Gamma_s0' := Gamma_s')
          (suffix0 := suffix) (x0 := x) (gSeed0 := gSeed) (n0 := n) h_seed)
  | cons =>
      rename_i tShape op xv kv hb rest hHead hRest ihHead ihRest
        Gamma_s Gamma_s' suffix x body gSeed n ihBody h_seed
      have ihRest' :
          ∀ {Gamma_s0 Gamma_s0' suffix0 : LinearCtx} {x0 : String} {body' : Term}
              {gSeed0 : Term} {n0 : Nat},
            (∀ {Gamma_s1 Gamma_s1' suffix1 : LinearCtx} {x1 : String} {gSeed1 : Term}
                {n1 : Nat},
              HasType Delta Sigma
                (Gamma_s1 ++ suffix1) gSeed1 (cotangentType tShape) [] (Gamma_s1' ++ suffix1) →
              HasType Delta Sigma
                (Gamma_s1 ++ suffix1)
                (adjointTypedFrom body' tShape x1 gSeed1 n1)
                Typ.unit
                (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
                (Gamma_s1' ++ suffix1)) →
            HasType Delta Sigma
              (Gamma_s0 ++ suffix0) gSeed0 (cotangentType tShape) [] (Gamma_s0' ++ suffix0) →
            HasType Delta Sigma
              (Gamma_s0 ++ suffix0)
              (adjointTypedClausesFrom rest x0 body' tShape gSeed0 n0)
              Typ.unit
              (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
              (Gamma_s0' ++ suffix0) := by
        intro Gamma_s0 Gamma_s0' suffix0 x0 body' gSeed0 n0 ihBody0 hSeed0
        exact ihRest
          (Gamma_s := Gamma_s0)
          (Gamma_s' := Gamma_s0')
          (suffix := suffix0)
          (x := x0)
          (body := body')
          (gSeed := gSeed0)
          (n := n0)
          ihBody0
          hSeed0
      exact adjointTypedClauses_cons_typed Delta Sigma
        (t := tShape)
        (param := x)
        (body := body)
        (ihHead := ihHead)
        (ihRest := ihRest')
        (ihBody := ihBody)
        (h_seed := h_seed)
-/

/-- Typed public surface for the staged adjoint transform. This is the
    theorem `T-Grad` now needs because the operational reduct uses
    `adjointTypedFrom`, not the legacy `adjointFrom` shim. -/
theorem adjointTypedFrom_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (x gs : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
    (n : Nat) (slot : Option Typ)
    (_h_e : HasType (Capability.diff :: Delta) Sigma
                    (Gamma ++ [(x, some (Typ.tensor ds))])
                    e (Typ.tensor dsOut) eps
                    (Gamma ++ [(x, slot)]))
    (_h_compat : subsetEffRow eps DiffCompat = true)
    (_h_supp : AdjointSupported e)
    (_h_ctxSupp : AdjointFreeCtxSupported
      (Gamma ++ [(x, some (Typ.tensor ds))]) e)
    (_h_termFresh : AdjointTermFresh n e)
    (_h_fresh_full : AdjointNamesFresh n
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))]))
    (_h_fresh_small : AdjointNamesFresh n (Gamma ++ [(x, some (Typ.tensor ds))])) :
    ∃ slotAdj,
      (slotAdj = none ∨ slotAdj = some (Typ.tensor ds)) ∧
      HasType Delta Sigma
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
        (adjointTypedFrom e (Typ.tensor dsOut) x (Term.var gs) n)
        Typ.unit
        (EffectRow.union eps [EffectLabel.accum])
        (Gamma ++ [(x, slotAdj), (gs, none)]) := by
  have hShape : AdjointTypedShape (Capability.diff :: Delta) Sigma (Typ.tensor dsOut) e :=
    adjointTypedShape_of_typed _h_e _h_supp
  obtain ⟨slotAdj, hslotAdj, hAdjRaw⟩ :=
    adjointTypedShape_preserves_typing Delta Sigma hShape
      (Gamma := Gamma)
      (suffixIn := ([(gs, some (Typ.tensor dsOut))] : LinearCtx))
      (suffixOut := ([(gs, none)] : LinearCtx))
      (x := x)
      (gSeed := Term.var gs)
      (n := n)
      (slotIn := some (Typ.tensor ds))
      (by
        simpa [List.append_assoc] using
          (HasType.var Delta Sigma
            (Gamma ++ [(x, some (Typ.tensor ds))])
            ([] : LinearCtx)
            gs
            (Typ.tensor dsOut)))
  have hAdj :
      HasType Delta Sigma
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
        (adjointTypedFrom e (Typ.tensor dsOut) x (Term.var gs) n)
        Typ.unit
        (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
        (Gamma ++ [(x, slotAdj), (gs, none)]) := by
    simpa [List.append_assoc] using hAdjRaw
  exact ⟨slotAdj, hslotAdj,
    HasType.subEff Delta Sigma
      (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
      (Gamma ++ [(x, slotAdj), (gs, none)])
      (adjointTypedFrom e (Typ.tensor dsOut) x (Term.var gs) n)
      Typ.unit
      (EffectRow.union [EffectLabel.accum] ([] : EffectRow))
      (EffectRow.union eps [EffectLabel.accum])
      hAdj
      (subEff_accum_into_grad eps)⟩

theorem adjointTyped_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (x gs : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
    (slot : Option Typ)
    (h_e : HasType (Capability.diff :: Delta) Sigma
                   (Gamma ++ [(x, some (Typ.tensor ds))])
                   e (Typ.tensor dsOut) eps
                   (Gamma ++ [(x, slot)]))
    (h_compat : subsetEffRow eps DiffCompat = true)
    (h_supp : AdjointSupported e)
    (h_ctxSupp : AdjointFreeCtxSupported
      (Gamma ++ [(x, some (Typ.tensor ds))]) e)
    (h_termFresh : AdjointTermFresh 0 e)
    (h_fresh_full : AdjointNamesFresh 0
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))]))
    (h_fresh_small : AdjointNamesFresh 0 (Gamma ++ [(x, some (Typ.tensor ds))])) :
    ∃ slotAdj,
      (slotAdj = none ∨ slotAdj = some (Typ.tensor ds)) ∧
      HasType Delta Sigma
        (Gamma ++ [(x, some (Typ.tensor ds)), (gs, some (Typ.tensor dsOut))])
        (adjointTyped e (Typ.tensor dsOut) x (Term.var gs))
        Typ.unit
        (EffectRow.union eps [EffectLabel.accum])
        (Gamma ++ [(x, slotAdj), (gs, none)]) := by
  simpa [adjointTyped] using
    (adjointTypedFrom_preserves_typing Delta Sigma Gamma
      x gs ds dsOut e eps 0 slot h_e h_compat h_supp h_ctxSupp h_termFresh
      h_fresh_full h_fresh_small)

end LaCaDiLE
