-- LaCaDiLE/SubstitutionDB.lean — substitution metatheory on HasTypeDB.
--
-- Under Option C (see TypingDB.lean), the linear context is
-- `List (Option Typ)` whose length is preserved by every typing
-- rule. In this representation `shiftAt j` is uniformly correct and
-- the substitution lemmas admit clean uniform signatures.
--
-- Wave 5a closes `weakening_insert_db` (and the ClausesTypedDB
-- mutual partner) via a mutual block with structural termination.

import LaCaDiLE.SyntaxDB
import LaCaDiLE.TypingDB

namespace LaCaDiLE

/-! ## Positional insertion on DB linear contexts -/

def LinearCtxDB.insertAt : Nat → Option Typ → LinearCtxDB → LinearCtxDB
  | 0, s, Γ => s :: Γ
  | _ + 1, s, [] => [s]
  | j + 1, s, x :: xs => x :: LinearCtxDB.insertAt j s xs

@[simp] theorem LinearCtxDB.insertAt_zero (Γ : LinearCtxDB) (s : Option Typ) :
    LinearCtxDB.insertAt 0 s Γ = s :: Γ := rfl

@[simp] theorem LinearCtxDB.insertAt_cons_succ
    (x : Option Typ) (xs : LinearCtxDB) (j : Nat) (s : Option Typ) :
    LinearCtxDB.insertAt (j + 1) s (x :: xs) =
      x :: LinearCtxDB.insertAt j s xs := rfl

@[simp] theorem LinearCtxDB.insertAt_nil_succ
    (j : Nat) (s : Option Typ) :
    LinearCtxDB.insertAt (j + 1) s [] = [s] := rfl

theorem LinearCtxDB.length_insertAt
    (Γ : LinearCtxDB) (j : Nat) (s : Option Typ) :
    (LinearCtxDB.insertAt j s Γ).length = Γ.length + 1 := by
  induction j generalizing Γ with
  | zero => simp
  | succ k ih =>
    cases Γ with
    | nil => simp
    | cons x xs => simp [ih]

theorem LinearCtxDB.getElem?_insertAt_lt
    (Γ : LinearCtxDB) (j : Nat) (s : Option Typ) (i : Nat)
    (hij : i < j) (hiΓ : i < Γ.length) :
    (LinearCtxDB.insertAt j s Γ)[i]? = Γ[i]? := by
  induction j generalizing Γ i with
  | zero => exact (Nat.not_lt_zero _ hij).elim
  | succ k ih =>
    cases Γ with
    | nil => exact (Nat.not_lt_zero _ hiΓ).elim
    | cons x xs =>
      cases i with
      | zero => simp [LinearCtxDB.insertAt]
      | succ n =>
        have hn : n < k := Nat.lt_of_succ_lt_succ hij
        have hnΓ : n < xs.length := Nat.lt_of_succ_lt_succ hiΓ
        simp [LinearCtxDB.insertAt, ih xs n hn hnΓ]

theorem LinearCtxDB.getElem?_insertAt_gt
    (Γ : LinearCtxDB) (j : Nat) (s : Option Typ) (i : Nat) (h : j ≤ i) :
    (LinearCtxDB.insertAt j s Γ)[i + 1]? = Γ[i]? := by
  induction j generalizing Γ i with
  | zero =>
    cases Γ <;> simp [LinearCtxDB.insertAt]
  | succ k ih =>
    cases Γ with
    | nil =>
      cases i with
      | zero => exact absurd h (by simp)
      | succ n => simp [LinearCtxDB.insertAt]
    | cons x xs =>
      cases i with
      | zero => exact absurd h (by simp)
      | succ n =>
        have hn : k ≤ n := Nat.le_of_succ_le_succ h
        simp [LinearCtxDB.insertAt, ih xs n hn]

theorem LinearCtxDB.set_insertAt_lt
    (Γ : LinearCtxDB) (j : Nat) (s : Option Typ) (i : Nat) (v : Option Typ)
    (hij : i < j) (hiΓ : i < Γ.length) :
    (LinearCtxDB.insertAt j s Γ).set i v =
      LinearCtxDB.insertAt j s (Γ.set i v) := by
  induction j generalizing Γ i with
  | zero => exact (Nat.not_lt_zero _ hij).elim
  | succ k ih =>
    cases Γ with
    | nil => exact (Nat.not_lt_zero _ hiΓ).elim
    | cons x xs =>
      cases i with
      | zero => simp [LinearCtxDB.insertAt, List.set]
      | succ n =>
        have hn : n < k := Nat.lt_of_succ_lt_succ hij
        have hnΓ : n < xs.length := Nat.lt_of_succ_lt_succ hiΓ
        simp [LinearCtxDB.insertAt, List.set, ih xs n hn hnΓ]

theorem LinearCtxDB.set_insertAt_gt
    (Γ : LinearCtxDB) (j : Nat) (s : Option Typ) (i : Nat) (v : Option Typ)
    (h : j ≤ i) :
    (LinearCtxDB.insertAt j s Γ).set (i + 1) v =
      LinearCtxDB.insertAt j s (Γ.set i v) := by
  induction j generalizing Γ i with
  | zero =>
    cases Γ with
    | nil => simp [LinearCtxDB.insertAt, List.set]
    | cons x xs => simp [LinearCtxDB.insertAt, List.set]
  | succ k ih =>
    cases Γ with
    | nil =>
      cases i with
      | zero => exact absurd h (by simp)
      | succ n => simp [LinearCtxDB.insertAt, List.set]
    | cons x xs =>
      cases i with
      | zero => exact absurd h (by simp)
      | succ n =>
        have hn : k ≤ n := Nat.le_of_succ_le_succ h
        simp [LinearCtxDB.insertAt, List.set, ih xs n hn]

/-! ## Shifting and the var lemmas -/

abbrev shiftAt (j : Nat) (e : TermDB) : TermDB := liftAux j 1 e

theorem shiftAt_var_lt (j i : Nat) (h : i < j) :
    shiftAt j (TermDB.var i) = TermDB.var i := by
  simp [shiftAt, liftAux, h]

theorem shiftAt_var_ge (j i : Nat) (h : ¬ i < j) :
    shiftAt j (TermDB.var i) = TermDB.var (i + 1) := by
  simp [shiftAt, liftAux, h]

/-! ## Cutoff-equal helpers for substitution -/

theorem LinearCtxDB.getElem?_insertAt_eq
    (Γ : LinearCtxDB) (j : Nat) (s : Option Typ) (hj : j ≤ Γ.length) :
    (LinearCtxDB.insertAt j s Γ)[j]? = some s := by
  induction j generalizing Γ with
  | zero => simp [LinearCtxDB.insertAt]
  | succ k ih =>
    cases Γ with
    | nil => exact absurd hj (by simp)
    | cons x xs =>
      have hk : k ≤ xs.length := Nat.le_of_succ_le_succ hj
      simp [LinearCtxDB.insertAt, ih xs hk]

theorem LinearCtxDB.set_insertAt_eq
    (Γ : LinearCtxDB) (j : Nat) (s v : Option Typ) (hj : j ≤ Γ.length) :
    (LinearCtxDB.insertAt j s Γ).set j v = LinearCtxDB.insertAt j v Γ := by
  induction j generalizing Γ with
  | zero => cases Γ <;> simp [LinearCtxDB.insertAt, List.set]
  | succ k ih =>
    cases Γ with
    | nil => exact absurd hj (by simp)
    | cons x xs =>
      have hk : k ≤ xs.length := Nat.le_of_succ_le_succ hj
      simp [LinearCtxDB.insertAt, List.set, ih xs hk]

/-! ## `liftClausesAux` preserves clause operation names -/

/-- Given a member of the lifted clause list, find the corresponding
    original clause with the same operation name. -/
theorem exists_orig_of_mem_liftClausesAux
    (c d : Nat) (cls : List (EffectLabel × TermDB))
    (cl : EffectLabel × TermDB) (h : cl ∈ liftClausesAux c d cls) :
    ∃ hb, (cl.1, hb) ∈ cls := by
  induction cls with
  | nil => simp [liftClausesAux] at h
  | cons head rest ih =>
    cases head with
    | mk op hb =>
      simp only [liftClausesAux, List.mem_cons] at h
      rcases h with heq | hmem
      · exact ⟨hb, by cases cl with
          | mk op' hb' =>
            simp only [Prod.mk.injEq] at heq
            simp [heq.1]⟩
      · obtain ⟨hb', hmem'⟩ := ih hmem
        exact ⟨hb', List.mem_cons.mpr (Or.inr hmem')⟩

/-- Lifting a clause produces a list whose members all have
    operation names drawn from the original clause list. -/
theorem hClsH_liftClausesAux
    (c d : Nat) (cls : List (EffectLabel × TermDB)) (epsH : EffectRow)
    (h : ∀ cl ∈ cls, cl.1 ∈ epsH) :
    ∀ cl ∈ liftClausesAux c d cls, cl.1 ∈ epsH := by
  intro cl hmem
  obtain ⟨hb, hmem_orig⟩ := exists_orig_of_mem_liftClausesAux c d cls cl hmem
  exact h (cl.1, hb) hmem_orig

/-- For every operation in `epsH`, the lifted clause list still
    contains a clause with that operation. -/
theorem hCover_liftClausesAux
    (c d : Nat) (cls : List (EffectLabel × TermDB)) (epsH : EffectRow)
    (h : ∀ op ∈ epsH, ∃ cl ∈ cls, cl.1 = op) :
    ∀ op ∈ epsH, ∃ cl ∈ liftClausesAux c d cls, cl.1 = op := by
  intro op hop
  obtain ⟨cl, hcl_mem, hcl_eq⟩ := h op hop
  cases cl with
  | mk a b =>
    simp only at hcl_eq
    subst hcl_eq
    refine ⟨(a, liftAux (c + 2) d b), ?_, rfl⟩
    -- show (a, liftAux (c+2) d b) ∈ liftClausesAux c d cls
    clear h hop
    induction cls with
    | nil => exact absurd hcl_mem (by simp)
    | cons head rest ih =>
      cases head with
      | mk op' hb' =>
        simp only [List.mem_cons] at hcl_mem
        rcases hcl_mem with heq | hmem
        · simp only [Prod.mk.injEq] at heq
          simp [liftClausesAux, heq.1, heq.2]
        · simp only [liftClausesAux, List.mem_cons]
          exact Or.inr (ih hmem)

/-! ## Weakening via structural mutual recursion -/

mutual

theorem weakening_insert_db
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma Gamma' : LinearCtxDB}
    {e : TermDB} {t : Typ} {eps : EffectRow}
    (h : HasTypeDB Delta Sigma Gamma e t eps Gamma')
    (j : Nat) (t_new : Typ) :
    HasTypeDB Delta Sigma (Gamma.insertAt j (some t_new))
              (shiftAt j e) t eps
              (Gamma'.insertAt j (some t_new)) := by
  match h with
  | HasTypeDB.var Δ S Γ i t hlook =>
    by_cases hij : i < j
    · rw [shiftAt_var_lt j i hij]
      have hiΓ : i < Γ.length := by
        rcases hlt : Γ[i]? with _ | slot
        · rw [hlt] at hlook; cases hlook
        · exact (List.getElem?_eq_some_iff.mp hlt).1
      have hget : (LinearCtxDB.insertAt j (some t_new) Γ)[i]? = some (some t) := by
        rw [LinearCtxDB.getElem?_insertAt_lt Γ j (some t_new) i hij hiΓ]
        exact hlook
      have hout :
          LinearCtxDB.insertAt j (some t_new) (Γ.set i none) =
            (LinearCtxDB.insertAt j (some t_new) Γ).set i none :=
        (LinearCtxDB.set_insertAt_lt Γ j (some t_new) i none hij hiΓ).symm
      rw [hout]
      exact HasTypeDB.var Δ S (LinearCtxDB.insertAt j (some t_new) Γ) i t hget
    · rw [shiftAt_var_ge j i hij]
      have hji : j ≤ i := Nat.le_of_not_lt hij
      have hget : (LinearCtxDB.insertAt j (some t_new) Γ)[i + 1]? = some (some t) := by
        rw [LinearCtxDB.getElem?_insertAt_gt Γ j (some t_new) i hji]
        exact hlook
      have hout :
          LinearCtxDB.insertAt j (some t_new) (Γ.set i none) =
            (LinearCtxDB.insertAt j (some t_new) Γ).set (i + 1) none :=
        (LinearCtxDB.set_insertAt_gt Γ j (some t_new) i none hji).symm
      rw [hout]
      exact HasTypeDB.var Δ S (LinearCtxDB.insertAt j (some t_new) Γ) (i + 1) t hget
  | HasTypeDB.unit Δ S Γ =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.unit Δ S (LinearCtxDB.insertAt j (some t_new) Γ)
  | HasTypeDB.abs Δ S Γ1 Γ2 slot t1 t2 eps body hbody =>
    simp only [shiftAt, liftAux]
    have ihj := weakening_insert_db hbody (j + 1) t_new
    simp only [LinearCtxDB.insertAt_cons_succ] at ihj
    exact HasTypeDB.abs Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) slot t1 t2 eps _ ihj
  | HasTypeDB.app Δ S Γ1 Γ2 Γ3 e1 e2 t1 t2 eps eps1 eps2 h1 h2 =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.app Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) (LinearCtxDB.insertAt j (some t_new) Γ3)
            _ _ t1 t2 eps eps1 eps2
            (weakening_insert_db h1 j t_new) (weakening_insert_db h2 j t_new)
  | HasTypeDB.letBind Δ S Γ1 Γ2 Γ3 slot e1 e2 t1 t2 eps1 eps2 h1 h2 =>
    simp only [shiftAt, liftAux]
    have ih2 := weakening_insert_db h2 (j + 1) t_new
    simp only [LinearCtxDB.insertAt_cons_succ] at ih2
    exact HasTypeDB.letBind Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) (LinearCtxDB.insertAt j (some t_new) Γ3)
            slot _ _ t1 t2 eps1 eps2
            (weakening_insert_db h1 j t_new) ih2
  | HasTypeDB.copy Δ S Γ1 Γ2 e ds eps hbody =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.copy Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) _ ds eps
            (weakening_insert_db hbody j t_new)
  | HasTypeDB.letpair Δ S Γ1 Γ2 Γ3 slot1 slot2 e1 e2 t1 t2 t eps1 eps2 h1 h2 =>
    simp only [shiftAt, liftAux]
    have ih2 := weakening_insert_db h2 (j + 2) t_new
    have h2eq :
        LinearCtxDB.insertAt (j + 2) (some t_new) (some t2 :: some t1 :: Γ2) =
          some t2 :: some t1 :: LinearCtxDB.insertAt j (some t_new) Γ2 := rfl
    have h3eq :
        LinearCtxDB.insertAt (j + 2) (some t_new) (slot1 :: slot2 :: Γ3) =
          slot1 :: slot2 :: LinearCtxDB.insertAt j (some t_new) Γ3 := rfl
    rw [h2eq, h3eq] at ih2
    exact HasTypeDB.letpair Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) (LinearCtxDB.insertAt j (some t_new) Γ3)
            slot1 slot2 _ _ t1 t2 t eps1 eps2
            (weakening_insert_db h1 j t_new) ih2
  | HasTypeDB.tpair Δ S Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2 h1 h2 =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.tpair Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) (LinearCtxDB.insertAt j (some t_new) Γ3)
            _ _ t1 t2 eps1 eps2
            (weakening_insert_db h1 j t_new) (weakening_insert_db h2 j t_new)
  | HasTypeDB.fst Δ S Γ1 Γ2 e t1 t2 eps hbody =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.fst Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) _ t1 t2 eps
            (weakening_insert_db hbody j t_new)
  | HasTypeDB.snd Δ S Γ1 Γ2 e t1 t2 eps hbody =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.snd Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) _ t1 t2 eps
            (weakening_insert_db hbody j t_new)
  | HasTypeDB.const Δ S Γ v ds =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.const Δ S (LinearCtxDB.insertAt j (some t_new) Γ) v ds
  | HasTypeDB.tadd Δ S Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 h1 h2 =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.tadd Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) (LinearCtxDB.insertAt j (some t_new) Γ3)
            _ _ ds eps1 eps2
            (weakening_insert_db h1 j t_new) (weakening_insert_db h2 j t_new)
  | HasTypeDB.tmul Δ S Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 h1 h2 =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.tmul Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) (LinearCtxDB.insertAt j (some t_new) Γ3)
            _ _ ds eps1 eps2
            (weakening_insert_db h1 j t_new) (weakening_insert_db h2 j t_new)
  | HasTypeDB.tsum Δ S Γ1 Γ2 e ds i eps hbody ds' _ =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.tsum Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) _ ds i eps
            (weakening_insert_db hbody j t_new) ds' True.intro
  | HasTypeDB.texpand Δ S Γ1 Γ2 e ds i k eps hbody ds' _ =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.texpand Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) _ ds i k eps
            (weakening_insert_db hbody j t_new) ds' True.intro
  | HasTypeDB.uniformLike Δ S Γ1 Γ2 e ds lo hi eps hbody =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.uniformLike Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) _ ds lo hi eps
            (weakening_insert_db hbody j t_new)
  | HasTypeDB.perform Δ S Γ1 Γ2 op e tArg tRet eps hbody hM =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.perform Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) op _ tArg tRet eps
            (weakening_insert_db hbody j t_new) hM
  | HasTypeDB.handle Δ S Γ1 Γ2 Γ3 body clauses t epsH epsB hb hSubsH hClsH hCover hcls =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.handle Δ S (LinearCtxDB.insertAt j (some t_new) Γ1)
            (LinearCtxDB.insertAt j (some t_new) Γ2) (LinearCtxDB.insertAt j (some t_new) Γ3)
            _ _ t epsH epsB (weakening_insert_db hb j t_new)
            hSubsH (hClsH_liftClausesAux j 1 clauses epsH hClsH)
            (hCover_liftClausesAux j 1 clauses epsH hCover)
            (weakening_insert_clauses_db hcls j t_new)
  | HasTypeDB.tgrad Δ S Γ slot ds dsOut body eps hbody hsub =>
    simp only [shiftAt, liftAux]
    have ihj := weakening_insert_db hbody (j + 1) t_new
    simp only [LinearCtxDB.insertAt_cons_succ] at ihj
    exact HasTypeDB.tgrad Δ S (LinearCtxDB.insertAt j (some t_new) Γ) slot ds dsOut _ eps ihj hsub
  | HasTypeDB.tvmap Δ S Γ slot t1 t2 body eps d hbody =>
    simp only [shiftAt, liftAux]
    have ihj := weakening_insert_db hbody (j + 1) t_new
    simp only [LinearCtxDB.insertAt_cons_succ] at ihj
    exact HasTypeDB.tvmap Δ S (LinearCtxDB.insertAt j (some t_new) Γ) slot t1 t2 _ eps d ihj
  | HasTypeDB.loc Δ S Γ ell t hlook =>
    simp only [shiftAt, liftAux]
    exact HasTypeDB.loc Δ S (LinearCtxDB.insertAt j (some t_new) Γ) ell t hlook
  | HasTypeDB.subEff Δ S Γ Γ' e t eps eps' hbody hSub =>
    exact HasTypeDB.subEff Δ S (LinearCtxDB.insertAt j (some t_new) Γ)
            (LinearCtxDB.insertAt j (some t_new) Γ') _ t eps eps'
            (weakening_insert_db hbody j t_new) hSub

theorem weakening_insert_clauses_db
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma2 Gamma3 : LinearCtxDB}
    {t : Typ} {epsR : EffectRow} {cls : List (EffectLabel × TermDB)}
    (h : ClausesTypedDB Delta Sigma Gamma2 Gamma3 t epsR cls)
    (j : Nat) (t_new : Typ) :
    ClausesTypedDB Delta Sigma (Gamma2.insertAt j (some t_new))
                               (Gamma3.insertAt j (some t_new))
                               t epsR (liftClausesAux j 1 cls) := by
  match h with
  | ClausesTypedDB.nil Δ S Γ2 t epsR =>
    simp only [liftClausesAux]
    exact ClausesTypedDB.nil Δ S (LinearCtxDB.insertAt j (some t_new) Γ2) t epsR
  | ClausesTypedDB.cons Δ S Γ2 Γ3 slot1 slot2 t tArg tRet epsR op hb rest hbody hrest =>
    simp only [liftClausesAux]
    have ihb := weakening_insert_db hbody (j + 2) t_new
    have hctx_in :
        LinearCtxDB.insertAt (j + 2) (some t_new) (some (Typ.arrow tRet t epsR) :: some tArg :: Γ2) =
          some (Typ.arrow tRet t epsR) :: some tArg :: LinearCtxDB.insertAt j (some t_new) Γ2 := rfl
    have hctx_out :
        LinearCtxDB.insertAt (j + 2) (some t_new) (slot1 :: slot2 :: Γ3) =
          slot1 :: slot2 :: LinearCtxDB.insertAt j (some t_new) Γ3 := rfl
    rw [hctx_in, hctx_out] at ihb
    exact ClausesTypedDB.cons Δ S (LinearCtxDB.insertAt j (some t_new) Γ2)
            (LinearCtxDB.insertAt j (some t_new) Γ3) slot1 slot2 t tArg tRet epsR op _ _
            ihb (weakening_insert_clauses_db hrest j t_new)

end

/-! ## Substitution: `subst_preserves_typing_db`

Proved by induction on the term structure, using `weakening_insert_db`
above for the lifted substituted value in binder cases. Each case
pattern-matches on `h_e` to extract the structure of the typing
derivation for `e`. -/

/-- Build a derivation for `v` shifted up by 1, valid under a
    new head slot. Direct application of `weakening_insert_db` at
    cutoff 0. -/
private theorem weakening_head_db
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma : LinearCtxDB}
    {v : TermDB} {t_v t_new : Typ}
    (h : HasTypeDB Delta Sigma Gamma v t_v [] Gamma) :
    HasTypeDB Delta Sigma (some t_new :: Gamma) (lift v) t_v []
              (some t_new :: Gamma) := by
  have := weakening_insert_db h 0 t_new
  simpa [LinearCtxDB.insertAt, lift, shiftAt] using this

/-! ## Length preservation

Foundational invariant: every `HasTypeDB` derivation preserves the
length of the linear context from input to output. This is the
Option C consumption discipline made structural — marking a slot
`none` does not change list length. Used as a prerequisite for slot
persistence inversion and for the subst metatheory at large. -/

mutual

theorem hasTypeDB_length_preservation
    {Delta : CapCtx} {Sigma : StoreTyp} {Γ Γ' : LinearCtxDB}
    {e : TermDB} {t : Typ} {eps : EffectRow}
    (h : HasTypeDB Delta Sigma Γ e t eps Γ') : Γ.length = Γ'.length := by
  match h with
  | HasTypeDB.var _ _ _ _ _ _ => simp
  | HasTypeDB.unit _ _ _ => rfl
  | HasTypeDB.abs _ _ _ _ _ _ _ _ _ hbody =>
    exact Nat.succ.inj (hasTypeDB_length_preservation hbody)
  | HasTypeDB.app _ _ _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    exact (hasTypeDB_length_preservation h1).trans (hasTypeDB_length_preservation h2)
  | HasTypeDB.letBind _ _ _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    have l1 := hasTypeDB_length_preservation h1
    have l2 := hasTypeDB_length_preservation h2
    exact l1.trans (Nat.succ.inj l2)
  | HasTypeDB.copy _ _ _ _ _ _ _ hbody => exact hasTypeDB_length_preservation hbody
  | HasTypeDB.letpair _ _ _ _ _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    have l1 := hasTypeDB_length_preservation h1
    have l2 := hasTypeDB_length_preservation h2
    exact l1.trans (Nat.succ.inj (Nat.succ.inj l2))
  | HasTypeDB.tpair _ _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    exact (hasTypeDB_length_preservation h1).trans (hasTypeDB_length_preservation h2)
  | HasTypeDB.fst _ _ _ _ _ _ _ _ hbody => exact hasTypeDB_length_preservation hbody
  | HasTypeDB.snd _ _ _ _ _ _ _ _ hbody => exact hasTypeDB_length_preservation hbody
  | HasTypeDB.const _ _ _ _ _ => rfl
  | HasTypeDB.tadd _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    exact (hasTypeDB_length_preservation h1).trans (hasTypeDB_length_preservation h2)
  | HasTypeDB.tmul _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    exact (hasTypeDB_length_preservation h1).trans (hasTypeDB_length_preservation h2)
  | HasTypeDB.tsum _ _ _ _ _ _ _ _ hbody _ _ => exact hasTypeDB_length_preservation hbody
  | HasTypeDB.texpand _ _ _ _ _ _ _ _ _ hbody _ _ => exact hasTypeDB_length_preservation hbody
  | HasTypeDB.uniformLike _ _ _ _ _ _ _ _ _ hbody => exact hasTypeDB_length_preservation hbody
  | HasTypeDB.perform _ _ _ _ _ _ _ _ _ hbody _ => exact hasTypeDB_length_preservation hbody
  | HasTypeDB.handle _ _ _ _ _ _ _ _ _ _ hb _ _ _ hcls =>
    have l1 := hasTypeDB_length_preservation hb
    have l2 := hasTypeDB_length_preservation_clauses hcls
    exact l1.trans l2
  | HasTypeDB.tgrad _ _ _ _ _ _ _ _ hbody _ =>
    exact Nat.succ.inj (hasTypeDB_length_preservation hbody)
  | HasTypeDB.tvmap _ _ _ _ _ _ _ _ _ hbody =>
    exact Nat.succ.inj (hasTypeDB_length_preservation hbody)
  | HasTypeDB.loc _ _ _ _ _ _ => rfl
  | HasTypeDB.subEff _ _ _ _ _ _ _ _ hbody _ => exact hasTypeDB_length_preservation hbody

theorem hasTypeDB_length_preservation_clauses
    {Delta : CapCtx} {Sigma : StoreTyp} {Γ2 Γ3 : LinearCtxDB}
    {t : Typ} {epsR : EffectRow} {cls : List (EffectLabel × TermDB)}
    (h : ClausesTypedDB Delta Sigma Γ2 Γ3 t epsR cls) : Γ2.length = Γ3.length := by
  match h with
  | ClausesTypedDB.nil _ _ _ _ _ => rfl
  | ClausesTypedDB.cons _ _ _ _ _ _ _ _ _ _ _ _ _ _ hrest =>
    exact hasTypeDB_length_preservation_clauses hrest

end

/-! ## None-slot monotonicity

Once a slot in a DB linear context is `none` (consumed), no typing
rule can restore it to `some`. Formally: every rule either leaves a
given slot untouched or transitions it from `some` to `none`. This
monotonicity is one of two ingredients needed for the tail-rebase
machinery that `subst_preserves_typing_db` depends on (the other is
length preservation, landed in Wave 5b above).

The proof is structural on `HasTypeDB`, with a mutual partner for
`ClausesTypedDB`. -/

mutual

theorem hasTypeDB_none_monotone
    {Delta : CapCtx} {Sigma : StoreTyp} {Γ Γ' : LinearCtxDB}
    {e : TermDB} {t : Typ} {eps : EffectRow}
    (h : HasTypeDB Delta Sigma Γ e t eps Γ')
    (i : Nat) (hi : Γ[i]? = some none) : Γ'[i]? = some none := by
  match h with
  | HasTypeDB.var _ _ Γ k _ hlook =>
    -- var consumes position k (live slot). If i = k, `hi` would say
    -- Γ[k]? = some none, contradicting `hlook`. Otherwise set leaves
    -- position i untouched.
    by_cases hik : i = k
    · subst hik
      rw [hlook] at hi
      cases hi
    · have hne : k ≠ i := fun h => hik h.symm
      have h1 : (Γ.set k none)[i]? = Γ[i]? := List.getElem?_set_ne hne
      exact h1.trans hi
  | HasTypeDB.unit _ _ _ => exact hi
  | HasTypeDB.abs _ _ Γ1 _ _ t1 _ _ _ hbody =>
    have hi' : (some t1 :: Γ1)[i + 1]? = some none := by simp [hi]
    have hm := hasTypeDB_none_monotone hbody (i + 1) hi'
    simpa using hm
  | HasTypeDB.app _ _ _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    exact hasTypeDB_none_monotone h2 i (hasTypeDB_none_monotone h1 i hi)
  | HasTypeDB.letBind _ _ _ Γ2 _ _ _ _ t1 _ _ _ h1 h2 =>
    have m1 := hasTypeDB_none_monotone h1 i hi
    have m2 : (some t1 :: Γ2)[i + 1]? = some none := by simp [m1]
    have hm := hasTypeDB_none_monotone h2 (i + 1) m2
    simpa using hm
  | HasTypeDB.copy _ _ _ _ _ _ _ hbody => exact hasTypeDB_none_monotone hbody i hi
  | HasTypeDB.letpair _ _ _ Γ2 _ _ _ _ _ t1 t2 _ _ _ h1 h2 =>
    have m1 := hasTypeDB_none_monotone h1 i hi
    have m2 : (some t2 :: some t1 :: Γ2)[i + 2]? = some none := by simp [m1]
    have hm := hasTypeDB_none_monotone h2 (i + 2) m2
    simpa using hm
  | HasTypeDB.tpair _ _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    exact hasTypeDB_none_monotone h2 i (hasTypeDB_none_monotone h1 i hi)
  | HasTypeDB.fst _ _ _ _ _ _ _ _ hbody => exact hasTypeDB_none_monotone hbody i hi
  | HasTypeDB.snd _ _ _ _ _ _ _ _ hbody => exact hasTypeDB_none_monotone hbody i hi
  | HasTypeDB.const _ _ _ _ _ => exact hi
  | HasTypeDB.tadd _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    exact hasTypeDB_none_monotone h2 i (hasTypeDB_none_monotone h1 i hi)
  | HasTypeDB.tmul _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    exact hasTypeDB_none_monotone h2 i (hasTypeDB_none_monotone h1 i hi)
  | HasTypeDB.tsum _ _ _ _ _ _ _ _ hbody _ _ => exact hasTypeDB_none_monotone hbody i hi
  | HasTypeDB.texpand _ _ _ _ _ _ _ _ _ hbody _ _ => exact hasTypeDB_none_monotone hbody i hi
  | HasTypeDB.uniformLike _ _ _ _ _ _ _ _ _ hbody => exact hasTypeDB_none_monotone hbody i hi
  | HasTypeDB.perform _ _ _ _ _ _ _ _ _ hbody _ => exact hasTypeDB_none_monotone hbody i hi
  | HasTypeDB.handle _ _ _ _ _ _ _ _ _ _ hb _ _ _ hcls =>
    have m1 := hasTypeDB_none_monotone hb i hi
    exact hasTypeDB_none_monotone_clauses hcls i m1
  | HasTypeDB.tgrad _ _ Γ _ ds _ _ _ hbody _ =>
    have hi' : (some (Typ.tensor ds) :: Γ)[i + 1]? = some none := by simp [hi]
    have hm := hasTypeDB_none_monotone hbody (i + 1) hi'
    simpa using hm
  | HasTypeDB.tvmap _ _ Γ _ t1 _ _ _ _ hbody =>
    have hi' : (some t1 :: Γ)[i + 1]? = some none := by simp [hi]
    have hm := hasTypeDB_none_monotone hbody (i + 1) hi'
    simpa using hm
  | HasTypeDB.loc _ _ _ _ _ _ => exact hi
  | HasTypeDB.subEff _ _ _ _ _ _ _ _ hbody _ => exact hasTypeDB_none_monotone hbody i hi

theorem hasTypeDB_none_monotone_clauses
    {Delta : CapCtx} {Sigma : StoreTyp} {Γ2 Γ3 : LinearCtxDB}
    {t : Typ} {epsR : EffectRow} {cls : List (EffectLabel × TermDB)}
    (h : ClausesTypedDB Delta Sigma Γ2 Γ3 t epsR cls)
    (i : Nat) (hi : Γ2[i]? = some none) : Γ3[i]? = some none := by
  match h with
  | ClausesTypedDB.nil _ _ _ _ _ => exact hi
  | ClausesTypedDB.cons _ _ Γ2 _ _ _ t tArg tRet epsR _ _ _ hbody hrest =>
    have hi' :
        (some (Typ.arrow tRet t epsR) :: some tArg :: Γ2)[i + 2]? = some none := by
      simp [hi]
    have mb := hasTypeDB_none_monotone hbody (i + 2) hi'
    -- The body's output context is `slot1 :: slot2 :: Γ3`; extract Γ3[i]?.
    simpa using mb

end

/-! ## Live-slot preservation

The none-monotonicity lemma tracks only the `none` → `none` direction.
For the `tail_rebase_db` sandwich argument we also need that a
position which starts live with type `t` either stays live with
*the same* type `t` or gets consumed to `none`. No rule ever rewrites
a live slot to a differently-typed live slot: every typing rule
either preserves the slot verbatim or writes `none` via `.set i none`
(the var rule). -/

mutual

theorem hasTypeDB_live_slot_monotone
    {Delta : CapCtx} {Sigma : StoreTyp} {Γ Γ' : LinearCtxDB}
    {e : TermDB} {t : Typ} {eps : EffectRow}
    (h : HasTypeDB Delta Sigma Γ e t eps Γ')
    (i : Nat) (t_slot : Typ) (hi : Γ[i]? = some (some t_slot)) :
    Γ'[i]? = some (some t_slot) ∨ Γ'[i]? = some none := by
  match h with
  | HasTypeDB.var _ _ Γ k _ _ =>
    by_cases hik : i = k
    · subst hik
      right
      have hlt : i < Γ.length := by
        rcases hg : Γ[i]? with _ | _
        · rw [hg] at hi; cases hi
        · exact (List.getElem?_eq_some_iff.mp hg).1
      exact List.getElem?_set_self hlt
    · left
      have hne : k ≠ i := fun h => hik h.symm
      rw [List.getElem?_set_ne hne]; exact hi
  | HasTypeDB.unit _ _ _ => left; exact hi
  | HasTypeDB.abs _ _ Γ1 _ _ t1 _ _ _ hbody =>
    have hi' : (some t1 :: Γ1)[i + 1]? = some (some t_slot) := by simp [hi]
    have hm := hasTypeDB_live_slot_monotone hbody (i + 1) t_slot hi'
    simpa using hm
  | HasTypeDB.app _ _ _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    rcases hasTypeDB_live_slot_monotone h1 i t_slot hi with h1' | h1'
    · exact hasTypeDB_live_slot_monotone h2 i t_slot h1'
    · right; exact hasTypeDB_none_monotone h2 i h1'
  | HasTypeDB.letBind _ _ _ Γ2 _ _ _ _ t1 _ _ _ h1 h2 =>
    rcases hasTypeDB_live_slot_monotone h1 i t_slot hi with h1' | h1'
    · have m2 : (some t1 :: Γ2)[i + 1]? = some (some t_slot) := by simp [h1']
      have hm := hasTypeDB_live_slot_monotone h2 (i + 1) t_slot m2
      simpa using hm
    · have m2 : (some t1 :: Γ2)[i + 1]? = some none := by simp [h1']
      have hm := hasTypeDB_none_monotone h2 (i + 1) m2
      right; simpa using hm
  | HasTypeDB.copy _ _ _ _ _ _ _ hbody =>
    exact hasTypeDB_live_slot_monotone hbody i t_slot hi
  | HasTypeDB.letpair _ _ _ Γ2 _ _ _ _ _ t1 t2 _ _ _ h1 h2 =>
    rcases hasTypeDB_live_slot_monotone h1 i t_slot hi with h1' | h1'
    · have m2 : (some t2 :: some t1 :: Γ2)[i + 2]? = some (some t_slot) := by simp [h1']
      have hm := hasTypeDB_live_slot_monotone h2 (i + 2) t_slot m2
      simpa using hm
    · have m2 : (some t2 :: some t1 :: Γ2)[i + 2]? = some none := by simp [h1']
      have hm := hasTypeDB_none_monotone h2 (i + 2) m2
      right; simpa using hm
  | HasTypeDB.tpair _ _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    rcases hasTypeDB_live_slot_monotone h1 i t_slot hi with h1' | h1'
    · exact hasTypeDB_live_slot_monotone h2 i t_slot h1'
    · right; exact hasTypeDB_none_monotone h2 i h1'
  | HasTypeDB.fst _ _ _ _ _ _ _ _ hbody =>
    exact hasTypeDB_live_slot_monotone hbody i t_slot hi
  | HasTypeDB.snd _ _ _ _ _ _ _ _ hbody =>
    exact hasTypeDB_live_slot_monotone hbody i t_slot hi
  | HasTypeDB.const _ _ _ _ _ => left; exact hi
  | HasTypeDB.tadd _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    rcases hasTypeDB_live_slot_monotone h1 i t_slot hi with h1' | h1'
    · exact hasTypeDB_live_slot_monotone h2 i t_slot h1'
    · right; exact hasTypeDB_none_monotone h2 i h1'
  | HasTypeDB.tmul _ _ _ _ _ _ _ _ _ _ h1 h2 =>
    rcases hasTypeDB_live_slot_monotone h1 i t_slot hi with h1' | h1'
    · exact hasTypeDB_live_slot_monotone h2 i t_slot h1'
    · right; exact hasTypeDB_none_monotone h2 i h1'
  | HasTypeDB.tsum _ _ _ _ _ _ _ _ hbody _ _ =>
    exact hasTypeDB_live_slot_monotone hbody i t_slot hi
  | HasTypeDB.texpand _ _ _ _ _ _ _ _ _ hbody _ _ =>
    exact hasTypeDB_live_slot_monotone hbody i t_slot hi
  | HasTypeDB.uniformLike _ _ _ _ _ _ _ _ _ hbody =>
    exact hasTypeDB_live_slot_monotone hbody i t_slot hi
  | HasTypeDB.perform _ _ _ _ _ _ _ _ _ hbody _ =>
    exact hasTypeDB_live_slot_monotone hbody i t_slot hi
  | HasTypeDB.handle _ _ _ _ _ _ _ _ _ _ hb _ _ _ hcls =>
    rcases hasTypeDB_live_slot_monotone hb i t_slot hi with hb' | hb'
    · exact hasTypeDB_live_slot_monotone_clauses hcls i t_slot hb'
    · right; exact hasTypeDB_none_monotone_clauses hcls i hb'
  | HasTypeDB.tgrad _ _ Γ _ ds _ _ _ hbody _ =>
    have hi' : (some (Typ.tensor ds) :: Γ)[i + 1]? = some (some t_slot) := by simp [hi]
    have hm := hasTypeDB_live_slot_monotone hbody (i + 1) t_slot hi'
    simpa using hm
  | HasTypeDB.tvmap _ _ Γ _ t1 _ _ _ _ hbody =>
    have hi' : (some t1 :: Γ)[i + 1]? = some (some t_slot) := by simp [hi]
    have hm := hasTypeDB_live_slot_monotone hbody (i + 1) t_slot hi'
    simpa using hm
  | HasTypeDB.loc _ _ _ _ _ _ => left; exact hi
  | HasTypeDB.subEff _ _ _ _ _ _ _ _ hbody _ =>
    exact hasTypeDB_live_slot_monotone hbody i t_slot hi

theorem hasTypeDB_live_slot_monotone_clauses
    {Delta : CapCtx} {Sigma : StoreTyp} {Γ2 Γ3 : LinearCtxDB}
    {t : Typ} {epsR : EffectRow} {cls : List (EffectLabel × TermDB)}
    (h : ClausesTypedDB Delta Sigma Γ2 Γ3 t epsR cls)
    (i : Nat) (t_slot : Typ) (hi : Γ2[i]? = some (some t_slot)) :
    Γ3[i]? = some (some t_slot) ∨ Γ3[i]? = some none := by
  match h with
  | ClausesTypedDB.nil _ _ _ _ _ => left; exact hi
  | ClausesTypedDB.cons _ _ Γ2 _ _ _ _ _ _ _ _ _ _ _ hrest =>
    -- hrest : ClausesTypedDB Γ2 Γ3 ...; recurse on it.
    exact hasTypeDB_live_slot_monotone_clauses hrest i t_slot hi

end

/-! ## Append split helpers for Wave 5d

The tail-rebase lemma decomposes contexts as `pre ++ Γ_old`. Before
attempting the structural induction, we pre-land the pure list
manipulation glue so that Wave 5d proofs have a stable foundation
and each rule-case reduces to bookkeeping rather than new list
algebra. -/

/-- `getElem?` in the left half of an append. -/
theorem LinearCtxDB.getElem?_append_lt
    (pre tail : LinearCtxDB) (i : Nat) (h : i < pre.length) :
    (pre ++ tail)[i]? = pre[i]? :=
  List.getElem?_append_left h

/-- `getElem?` in the right half of an append. -/
theorem LinearCtxDB.getElem?_append_ge
    (pre tail : LinearCtxDB) (i : Nat) (h : pre.length ≤ i) :
    (pre ++ tail)[i]? = tail[i - pre.length]? :=
  List.getElem?_append_right h

/-- `set` into the left half of an append commutes with append. -/
theorem LinearCtxDB.set_append_lt
    (pre tail : LinearCtxDB) (i : Nat) (v : Option Typ)
    (h : i < pre.length) :
    (pre ++ tail).set i v = pre.set i v ++ tail := by
  rw [List.set_append]
  simp [h]

/-- `set` into the right half of an append commutes with append. -/
theorem LinearCtxDB.set_append_ge
    (pre tail : LinearCtxDB) (i : Nat) (v : Option Typ)
    (h : pre.length ≤ i) :
    (pre ++ tail).set i v = pre ++ tail.set (i - pre.length) v := by
  rw [List.set_append]
  simp [Nat.not_lt.mpr h]

/-- If a derivation has matching input/output endpoints modulo a
    common tail `Γ_old`, length preservation forces the prefixes to
    have equal length. -/
theorem hasTypeDB_tail_pre_length_eq
    {Delta : CapCtx} {Sigma : StoreTyp}
    {pre pre' Γ_old : LinearCtxDB}
    {e : TermDB} {t : Typ} {eps : EffectRow}
    (h : HasTypeDB Delta Sigma (pre ++ Γ_old) e t eps (pre' ++ Γ_old)) :
    pre.length = pre'.length := by
  have hl := hasTypeDB_length_preservation h
  simp [List.length_append] at hl
  exact hl

/-- Tail slots of the input are preserved: the `(pre.length + k)`-th
    slot of `pre ++ Γ_old` is exactly `Γ_old[k]?`. Pure append
    bookkeeping, stated in the form consumed by the Wave 5d
    monotonicity argument. -/
theorem LinearCtxDB.getElem?_append_tail
    (pre Γ_old : LinearCtxDB) (k : Nat) :
    (pre ++ Γ_old)[pre.length + k]? = Γ_old[k]? := by
  have hle : pre.length ≤ pre.length + k := Nat.le_add_right _ _
  rw [List.getElem?_append_right hle]
  simp

/-- Along any derivation whose endpoints both end in the common tail
    `Γ_old`, the intermediate context dropped past the prefix equals
    `Γ_old`. This is the live-slot + none-slot sandwich: if any tail
    slot were consumed midway, `hasTypeDB_none_monotone` through the
    second half would propagate `none` to the output, contradicting
    the output's intact tail. Type preservation of the unconsumed
    case comes from `hasTypeDB_live_slot_monotone`. -/
theorem hasTypeDB_tail_through_middle
    {Delta : CapCtx} {Sigma : StoreTyp}
    {pre pre' Γ_old Γ_mid : LinearCtxDB}
    {e1 e2 : TermDB} {t1 t2 : Typ} {eps1 eps2 : EffectRow}
    (h1 : HasTypeDB Delta Sigma (pre ++ Γ_old) e1 t1 eps1 Γ_mid)
    (h2 : HasTypeDB Delta Sigma Γ_mid e2 t2 eps2 (pre' ++ Γ_old)) :
    Γ_mid.drop pre.length = Γ_old := by
  have hl1 := hasTypeDB_length_preservation h1
  have hl2 := hasTypeDB_length_preservation h2
  have hlen_mid : Γ_mid.length = pre.length + Γ_old.length := by
    have h := hl1.symm
    simp [List.length_append] at h; exact h
  have hpre : pre.length = pre'.length := by
    have h : (pre ++ Γ_old).length = (pre' ++ Γ_old).length := hl1.trans hl2
    simp [List.length_append] at h; exact h
  have hdrop_len : (Γ_mid.drop pre.length).length = Γ_old.length := by
    rw [List.length_drop, hlen_mid]; omega
  apply List.ext_getElem? (l₁ := Γ_mid.drop pre.length) (l₂ := Γ_old)
  intro k
  by_cases hk : k < Γ_old.length
  · have hdrop_get : (Γ_mid.drop pre.length)[k]? = Γ_mid[pre.length + k]? := by
      rw [List.getElem?_drop]
    rw [hdrop_get]
    have hin : (pre ++ Γ_old)[pre.length + k]? = Γ_old[k]? :=
      LinearCtxDB.getElem?_append_tail pre Γ_old k
    have hout : (pre' ++ Γ_old)[pre.length + k]? = Γ_old[k]? := by
      have h := LinearCtxDB.getElem?_append_tail pre' Γ_old k
      rw [← hpre] at h; exact h
    -- Γ_old[k]? is some slot (since k < length).
    have hk_some : ∃ slot : Option Typ, Γ_old[k]? = some slot := by
      rcases hg : Γ_old[k]? with _ | s
      · exfalso; have := List.getElem?_eq_none_iff.mp hg; omega
      · exact ⟨s, rfl⟩
    obtain ⟨slot, hslot⟩ := hk_some
    rw [hslot]
    cases slot with
    | none =>
      -- Input is none at pre.length+k. By monotone h1, Γ_mid same.
      have hin_none : (pre ++ Γ_old)[pre.length + k]? = some none := by
        rw [hin, hslot]
      exact hasTypeDB_none_monotone h1 (pre.length + k) hin_none
    | some t =>
      -- Input is some (some t) at pre.length+k.
      have hin_live : (pre ++ Γ_old)[pre.length + k]? = some (some t) := by
        rw [hin, hslot]
      rcases hasTypeDB_live_slot_monotone h1 (pre.length + k) t hin_live with hmid_live | hmid_none
      · exact hmid_live
      · -- Γ_mid[p]? = some none; monotone h2 forces output = some none.
        exfalso
        have hout_none := hasTypeDB_none_monotone h2 (pre.length + k) hmid_none
        rw [hout, hslot] at hout_none
        cases hout_none
  · -- k ≥ Γ_old.length: both sides are none.
    have h1none : Γ_old[k]? = none := by
      rw [List.getElem?_eq_none_iff]; omega
    have h2none : (Γ_mid.drop pre.length)[k]? = none := by
      rw [List.getElem?_eq_none_iff, hdrop_len]; omega
    rw [h1none, h2none]



/-! ## Tail-rebase (Wave 5e)

`tail_rebase_db` rebases a derivation whose input and output contexts
share a common tail `Γ_old` onto a different tail `Γ_new` of equal
length. Proved by structural induction on the derivation via `match`,
with the shape constraints threaded as explicit equalities so that
each rule-case can `subst` them before doing local bookkeeping. -/

mutual

theorem tail_rebase_db
    {Δ : CapCtx} {S : StoreTyp}
    {Γ1 Γ2 : LinearCtxDB} {e : TermDB} {t : Typ} {eps : EffectRow}
    (h : HasTypeDB Δ S Γ1 e t eps Γ2)
    (pre pre' Γ_old Γ_new : LinearCtxDB)
    (hin : Γ1 = pre ++ Γ_old) (hout : Γ2 = pre' ++ Γ_old)
    (h_len : Γ_old.length = Γ_new.length) :
    HasTypeDB Δ S (pre ++ Γ_new) e t eps (pre' ++ Γ_new) := by
  match h with
  | HasTypeDB.var Δ_ S_ Γ i ti hlook =>
    subst hin
    -- Output of var rule: (pre ++ Γ_old).set i none = pre' ++ Γ_old.
    by_cases hip : i < pre.length
    · -- Var in the prefix: substitute via set_append_lt.
      have hiΓ : i < (pre ++ Γ_old).length := by
        rw [List.length_append]; exact Nat.lt_of_lt_of_le hip (Nat.le_add_right _ _)
      have hlook_pre : pre[i]? = some (some ti) := by
        rw [← LinearCtxDB.getElem?_append_lt pre Γ_old i hip]; exact hlook
      -- Cancel the tail in hout.
      have hout_pre : pre.set i none = pre' := by
        rw [LinearCtxDB.set_append_lt pre Γ_old i none hip] at hout
        exact List.append_cancel_right hout
      -- Target: HasTypeDB ... (pre ++ Γ_new) (var i) ti [] (pre' ++ Γ_new).
      have hget_new : (pre ++ Γ_new)[i]? = some (some ti) := by
        rw [LinearCtxDB.getElem?_append_lt pre Γ_new i hip]; exact hlook_pre
      have hset_new : (pre ++ Γ_new).set i none = pre' ++ Γ_new := by
        rw [LinearCtxDB.set_append_lt pre Γ_new i none hip, hout_pre]
      have := HasTypeDB.var Δ_ S_ (pre ++ Γ_new) i ti hget_new
      rw [hset_new] at this
      exact this
    · -- Var in the tail: contradiction.
      have hip : pre.length ≤ i := Nat.le_of_not_lt hip
      exfalso
      -- At position i, (pre ++ Γ_old)[i]? = some (some ti); the var rule
      -- sets it to none; but the output must also equal (pre' ++ Γ_old),
      -- which at position i is live (Γ_old[i - pre.length]? = some ti,
      -- using pre.length = pre'.length from length preservation).
      have hlen_pp : pre.length = pre'.length := by
        have h0 := hasTypeDB_length_preservation
          (HasTypeDB.var Δ_ S_ (pre ++ Γ_old) i ti hlook)
        rw [hout, List.length_append, List.length_append] at h0
        exact Nat.add_right_cancel h0
      -- LHS at i: ((pre ++ Γ_old).set i none)[i]? = some none.
      have hlt_len : i < (pre ++ Γ_old).length := by
        rcases hg : (pre ++ Γ_old)[i]? with _ | s
        · rw [hg] at hlook; cases hlook
        · exact (List.getElem?_eq_some_iff.mp hg).1
      have hlhs : ((pre ++ Γ_old).set i none)[i]? = some none :=
        List.getElem?_set_self hlt_len
      -- RHS at i: (pre' ++ Γ_old)[i]? = Γ_old[i - pre'.length]? = Γ_old[i - pre.length]?.
      have hip' : pre'.length ≤ i := hlen_pp ▸ hip
      have hrhs : (pre' ++ Γ_old)[i]? = Γ_old[i - pre.length]? := by
        rw [LinearCtxDB.getElem?_append_ge pre' Γ_old i hip', hlen_pp]
      -- Γ_old[i - pre.length]? = some (some ti) from hlook.
      have htail : Γ_old[i - pre.length]? = some (some ti) := by
        rw [← LinearCtxDB.getElem?_append_ge pre Γ_old i hip]; exact hlook
      -- Now ((pre ++ Γ_old).set i none)[i]? = (pre' ++ Γ_old)[i]? from hout.
      have heq : ((pre ++ Γ_old).set i none)[i]? = (pre' ++ Γ_old)[i]? := by
        rw [hout]
      rw [hlhs, hrhs, htail] at heq
      cases heq
  | HasTypeDB.unit Δ_ S_ Γ =>
    have hpp : pre = pre' := List.append_cancel_right (hin.symm.trans hout)
    rw [hpp]
    exact HasTypeDB.unit Δ_ S_ (pre' ++ Γ_new)
  | HasTypeDB.abs Δ_ S_ Γ1 Γ2 slot t1 t2 eps_ body hbody =>
    -- Γ1 = pre ++ Γ_old, Γ2 = pre' ++ Γ_old, from hin/hout.
    -- body: some t1 :: Γ1 → slot :: Γ2.
    have hin' : some t1 :: Γ1 = (some t1 :: pre) ++ Γ_old := by
      simp [hin, List.cons_append]
    have hout' : slot :: Γ2 = (slot :: pre') ++ Γ_old := by
      simp [hout, List.cons_append]
    have ihb := tail_rebase_db hbody (some t1 :: pre) (slot :: pre') Γ_old Γ_new hin' hout' h_len
    -- Peel: (slot :: pre') ++ Γ_new = slot :: (pre' ++ Γ_new); likewise input.
    simp [List.cons_append] at ihb
    exact HasTypeDB.abs Δ_ S_ (pre ++ Γ_new) (pre' ++ Γ_new) slot t1 t2 eps_ body ihb
  | HasTypeDB.app Δ_ S_ Γ1 Γ2 Γ3 e1 e2 t1 t2 epsA eps1 eps2 h1 h2 =>
    have h1_shape : HasTypeDB Δ_ S_ (pre ++ Γ_old) e1 (Typ.arrow t1 t2 epsA) eps1 Γ2 := hin ▸ h1
    have h2_shape : HasTypeDB Δ_ S_ Γ2 e2 t1 eps2 (pre' ++ Γ_old) := hout ▸ h2
    have hmid := hasTypeDB_tail_through_middle h1_shape h2_shape
    have hΓ2 : Γ2 = Γ2.take pre.length ++ Γ_old := by
      have := List.take_append_drop pre.length Γ2
      rw [hmid] at this; exact this.symm
    have h1' := tail_rebase_db h1 pre (Γ2.take pre.length) Γ_old Γ_new hin hΓ2 h_len
    have h2' := tail_rebase_db h2 (Γ2.take pre.length) pre' Γ_old Γ_new hΓ2 hout h_len
    exact HasTypeDB.app Δ_ S_ (pre ++ Γ_new) (Γ2.take pre.length ++ Γ_new)
            (pre' ++ Γ_new) e1 e2 t1 t2 epsA eps1 eps2 h1' h2'
  | HasTypeDB.letBind Δ_ S_ Γ1 Γ2 Γ3 slot e1 e2 t1 t2 eps1 eps2 h1 h2 =>
    have h1_shape : HasTypeDB Δ_ S_ (pre ++ Γ_old) e1 t1 eps1 Γ2 := hin ▸ h1
    have h2_shape : HasTypeDB Δ_ S_ (some t1 :: Γ2) e2 t2 eps2 ((slot :: pre') ++ Γ_old) := by
      have h2' : HasTypeDB Δ_ S_ (some t1 :: Γ2) e2 t2 eps2 (slot :: (pre' ++ Γ_old)) := hout ▸ h2
      simpa [List.cons_append] using h2' 
    -- h1 endpoints: (pre ++ Γ_old) → Γ2. h2_shape endpoints: (some t1 :: Γ2) → (slot :: pre') ++ Γ_old.
    -- tail_through_middle needs matching append shapes on both endpoints of a *single* derivation,
    -- but here h2 starts at `some t1 :: Γ2`, not at a shape involving Γ_old directly. Use none/live
    -- monotonicity directly on h2 to force Γ2.drop pre.length = Γ_old.
    -- Actually: consider the composite view. Build h2' whose INPUT is (some t1 :: pre) ++ Γ_old form
    -- requires showing Γ2 = pm ++ Γ_old. We use hasTypeDB_tail_through_middle on the pair
    -- (h1, h2_under_cons) where h2_under_cons starts at (some t1 :: Γ2). Re-shape:
    -- h1 : pre ++ Γ_old → Γ2 means with prefix `pre`. h2 : (some t1 :: Γ2) → (slot :: pre') ++ Γ_old
    -- means with prefix `slot :: pre'`. For tail_through_middle we need the middle
    -- context of h1 (=Γ2) to equal the input of h2 (=some t1 :: Γ2). They differ by one
    -- head slot. Workaround: we really need `Γ2.drop pre.length = Γ_old` directly.
    have hΓ2_drop : Γ2.drop pre.length = Γ_old := by
      -- Use the fact that h2 has input (some t1 :: Γ2) and output (slot :: pre' ++ Γ_old).
      -- Apply hasTypeDB_tail_through_middle where the first derivation is h1 with prefix pre
      -- and the second is h2 viewed with prefix (slot :: pre')... but the types don't align.
      -- Alternative: append `some t1 :: ` to everything and use a manual sandwich.
      -- We'll do the sandwich inline using monotone lemmas.
      apply List.ext_getElem?
      intro k
      by_cases hk : k < Γ_old.length
      · have hdrop_k : (Γ2.drop pre.length)[k]? = Γ2[pre.length + k]? := by
          rw [List.getElem?_drop]
        rw [hdrop_k]
        have hin_k : (pre ++ Γ_old)[pre.length + k]? = Γ_old[k]? :=
          LinearCtxDB.getElem?_append_tail pre Γ_old k
        have hout_k : ((slot :: pre') ++ Γ_old)[(slot :: pre').length + k]? = Γ_old[k]? :=
          LinearCtxDB.getElem?_append_tail (slot :: pre') Γ_old k
        have hlp1 := hasTypeDB_length_preservation h1_shape
        have hlp2 := hasTypeDB_length_preservation h2_shape
        have hlen_eq : pre.length = pre'.length := by
          simp [List.length_append] at hlp1 hlp2
          -- pre.len + old.len = Γ2.len; 1 + Γ2.len = 1 + pre'.len + old.len → Γ2.len = pre'.len + old.len.
          -- So pre.len + old.len = pre'.len + old.len → pre.len = pre'.len.
          omega
        rcases hg : Γ_old[k]? with _ | ⟨_ | ti⟩
        · exfalso; rw [List.getElem?_eq_none_iff] at hg; omega
        · -- Γ_old[k]? = some none
          have hin_none : (pre ++ Γ_old)[pre.length + k]? = some none := hin_k.trans hg
          exact hasTypeDB_none_monotone h1_shape (pre.length + k) hin_none
        · -- Γ_old[k]? = some (some ti)
          have hin_live : (pre ++ Γ_old)[pre.length + k]? = some (some ti) := hin_k.trans hg
          rcases hasTypeDB_live_slot_monotone h1_shape (pre.length + k) ti hin_live with hmid_live | hmid_none
          · exact hmid_live
          · exfalso
            have hcons : (some t1 :: Γ2)[(pre.length + k) + 1]? = some none := by
              simp [hmid_none]
            have hout_none := hasTypeDB_none_monotone h2_shape ((pre.length + k) + 1) hcons
            have hidx : (slot :: pre').length + k = (pre.length + k) + 1 := by
              simp [List.length_cons]; omega
            rw [← hidx] at hout_none
            rw [hout_k] at hout_none
            rw [hg] at hout_none
            cases hout_none
      · have hdrop_len : (Γ2.drop pre.length).length = Γ_old.length := by
          have hlp1 := hasTypeDB_length_preservation h1_shape
          rw [List.length_drop]
          simp [List.length_append] at hlp1; omega
        have h1none : Γ_old[k]? = none := by
          rw [List.getElem?_eq_none_iff]; omega
        have h2none : (Γ2.drop pre.length)[k]? = none := by
          rw [List.getElem?_eq_none_iff, hdrop_len]; omega
        rw [h1none, h2none]
    have hΓ2 : Γ2 = Γ2.take pre.length ++ Γ_old := by
      have := List.take_append_drop pre.length Γ2
      rw [hΓ2_drop] at this; exact this.symm
    have h1' := tail_rebase_db h1 pre (Γ2.take pre.length) Γ_old Γ_new hin hΓ2 h_len
    -- For h2, reshape via the hout-rewritten version.
    have hin2 : some t1 :: Γ2 = (some t1 :: Γ2.take pre.length) ++ Γ_old := by
      rw [List.cons_append, ← hΓ2]
    have hout2 : slot :: Γ3 = (slot :: pre') ++ Γ_old := by
      rw [hout]; simp [List.cons_append]
    have h2' := tail_rebase_db h2 (some t1 :: Γ2.take pre.length) (slot :: pre') Γ_old Γ_new hin2 hout2 h_len
    -- Reshape result of h2' back into head-cons form.
    simp [List.cons_append] at h2'
    exact HasTypeDB.letBind Δ_ S_ (pre ++ Γ_new) (Γ2.take pre.length ++ Γ_new)
            (pre' ++ Γ_new) slot e1 e2 t1 t2 eps1 eps2 h1' h2'
  | HasTypeDB.copy Δ_ S_ Γ1 Γ2 e_ ds eps_ hbody =>
    have ih := tail_rebase_db hbody pre pre' Γ_old Γ_new hin hout h_len
    exact HasTypeDB.copy Δ_ S_ (pre ++ Γ_new) (pre' ++ Γ_new) e_ ds eps_ ih
  | HasTypeDB.letpair Δ_ S_ Γ1 Γ2 Γ3 slot1 slot2 e1 e2 t1 t2 tR eps1 eps2 h1 h2 =>
    have h1_shape : HasTypeDB Δ_ S_ (pre ++ Γ_old) e1 (Typ.pair t1 t2) eps1 Γ2 := hin ▸ h1
    have h2_shape : HasTypeDB Δ_ S_ (some t2 :: some t1 :: Γ2) e2 tR eps2
                              ((slot1 :: slot2 :: pre') ++ Γ_old) := by
      have h2' : HasTypeDB Δ_ S_ (some t2 :: some t1 :: Γ2) e2 tR eps2
                             (slot1 :: slot2 :: (pre' ++ Γ_old)) := hout ▸ h2
      simpa [List.cons_append] using h2'
    have hΓ2_drop : Γ2.drop pre.length = Γ_old := by
      apply List.ext_getElem?
      intro k
      by_cases hk : k < Γ_old.length
      · have hdrop_k : (Γ2.drop pre.length)[k]? = Γ2[pre.length + k]? := by
          rw [List.getElem?_drop]
        rw [hdrop_k]
        have hin_k : (pre ++ Γ_old)[pre.length + k]? = Γ_old[k]? :=
          LinearCtxDB.getElem?_append_tail pre Γ_old k
        have hout_k : ((slot1 :: slot2 :: pre') ++ Γ_old)[(slot1 :: slot2 :: pre').length + k]? = Γ_old[k]? :=
          LinearCtxDB.getElem?_append_tail (slot1 :: slot2 :: pre') Γ_old k
        have hlp1 := hasTypeDB_length_preservation h1_shape
        have hlp2 := hasTypeDB_length_preservation h2_shape
        have hlen_eq : pre.length = pre'.length := by
          simp [List.length_append] at hlp1 hlp2; omega
        rcases hg : Γ_old[k]? with _ | ⟨_ | ti⟩
        · exfalso; rw [List.getElem?_eq_none_iff] at hg; omega
        · have hin_none : (pre ++ Γ_old)[pre.length + k]? = some none := hin_k.trans hg
          exact hasTypeDB_none_monotone h1_shape (pre.length + k) hin_none
        · have hin_live : (pre ++ Γ_old)[pre.length + k]? = some (some ti) := hin_k.trans hg
          rcases hasTypeDB_live_slot_monotone h1_shape (pre.length + k) ti hin_live with hmid_live | hmid_none
          · exact hmid_live
          · exfalso
            have hcons : (some t2 :: some t1 :: Γ2)[(pre.length + k) + 2]? = some none := by
              simp [hmid_none]
            have hout_none := hasTypeDB_none_monotone h2_shape ((pre.length + k) + 2) hcons
            have hidx : (slot1 :: slot2 :: pre').length + k = (pre.length + k) + 2 := by
              simp [List.length_cons]; omega
            rw [← hidx] at hout_none
            rw [hout_k] at hout_none
            rw [hg] at hout_none
            cases hout_none
      · have hdrop_len : (Γ2.drop pre.length).length = Γ_old.length := by
          have hlp1 := hasTypeDB_length_preservation h1_shape
          rw [List.length_drop]
          simp [List.length_append] at hlp1; omega
        have h1none : Γ_old[k]? = none := by
          rw [List.getElem?_eq_none_iff]; omega
        have h2none : (Γ2.drop pre.length)[k]? = none := by
          rw [List.getElem?_eq_none_iff, hdrop_len]; omega
        rw [h1none, h2none]
    have hΓ2 : Γ2 = Γ2.take pre.length ++ Γ_old := by
      have := List.take_append_drop pre.length Γ2
      rw [hΓ2_drop] at this; exact this.symm
    have h1' := tail_rebase_db h1 pre (Γ2.take pre.length) Γ_old Γ_new hin hΓ2 h_len
    have hin2 : some t2 :: some t1 :: Γ2 = (some t2 :: some t1 :: Γ2.take pre.length) ++ Γ_old := by
      rw [List.cons_append, List.cons_append, ← hΓ2]
    have hout2 : slot1 :: slot2 :: Γ3 = (slot1 :: slot2 :: pre') ++ Γ_old := by
      rw [hout]; simp [List.cons_append]
    have h2' := tail_rebase_db h2 (some t2 :: some t1 :: Γ2.take pre.length)
                  (slot1 :: slot2 :: pre') Γ_old Γ_new hin2 hout2 h_len
    simp [List.cons_append] at h2'
    exact HasTypeDB.letpair Δ_ S_ (pre ++ Γ_new) (Γ2.take pre.length ++ Γ_new)
            (pre' ++ Γ_new) slot1 slot2 e1 e2 t1 t2 tR eps1 eps2 h1' h2'
  | HasTypeDB.tpair Δ_ S_ Γ1 Γ2 Γ3 e1 e2 t1 t2 eps1 eps2 h1 h2 =>
    have h1_shape : HasTypeDB Δ_ S_ (pre ++ Γ_old) e1 t1 eps1 Γ2 := hin ▸ h1
    have h2_shape : HasTypeDB Δ_ S_ Γ2 e2 t2 eps2 (pre' ++ Γ_old) := hout ▸ h2
    have hmid := hasTypeDB_tail_through_middle h1_shape h2_shape
    have hΓ2 : Γ2 = Γ2.take pre.length ++ Γ_old := by
      have := List.take_append_drop pre.length Γ2
      rw [hmid] at this; exact this.symm
    have h1' := tail_rebase_db h1 pre (Γ2.take pre.length) Γ_old Γ_new hin hΓ2 h_len
    have h2' := tail_rebase_db h2 (Γ2.take pre.length) pre' Γ_old Γ_new hΓ2 hout h_len
    exact HasTypeDB.tpair Δ_ S_ (pre ++ Γ_new) (Γ2.take pre.length ++ Γ_new)
            (pre' ++ Γ_new) e1 e2 t1 t2 eps1 eps2 h1' h2'
  | HasTypeDB.fst Δ_ S_ Γ1 Γ2 e_ t1 t2 eps_ hbody =>
    have ih := tail_rebase_db hbody pre pre' Γ_old Γ_new hin hout h_len
    exact HasTypeDB.fst Δ_ S_ (pre ++ Γ_new) (pre' ++ Γ_new) e_ t1 t2 eps_ ih
  | HasTypeDB.snd Δ_ S_ Γ1 Γ2 e_ t1 t2 eps_ hbody =>
    have ih := tail_rebase_db hbody pre pre' Γ_old Γ_new hin hout h_len
    exact HasTypeDB.snd Δ_ S_ (pre ++ Γ_new) (pre' ++ Γ_new) e_ t1 t2 eps_ ih
  | HasTypeDB.const Δ_ S_ Γ v_ ds =>
    have hpp : pre = pre' := List.append_cancel_right (hin.symm.trans hout)
    rw [hpp]
    exact HasTypeDB.const Δ_ S_ (pre' ++ Γ_new) v_ ds
  | HasTypeDB.tadd Δ_ S_ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 h1 h2 =>
    have h1_shape : HasTypeDB Δ_ S_ (pre ++ Γ_old) e1 (Typ.tensor ds) eps1 Γ2 := hin ▸ h1
    have h2_shape : HasTypeDB Δ_ S_ Γ2 e2 (Typ.tensor ds) eps2 (pre' ++ Γ_old) := hout ▸ h2
    have hmid := hasTypeDB_tail_through_middle h1_shape h2_shape
    have hΓ2 : Γ2 = Γ2.take pre.length ++ Γ_old := by
      have := List.take_append_drop pre.length Γ2
      rw [hmid] at this; exact this.symm
    have h1' := tail_rebase_db h1 pre (Γ2.take pre.length) Γ_old Γ_new hin hΓ2 h_len
    have h2' := tail_rebase_db h2 (Γ2.take pre.length) pre' Γ_old Γ_new hΓ2 hout h_len
    exact HasTypeDB.tadd Δ_ S_ (pre ++ Γ_new) (Γ2.take pre.length ++ Γ_new)
            (pre' ++ Γ_new) e1 e2 ds eps1 eps2 h1' h2'
  | HasTypeDB.tmul Δ_ S_ Γ1 Γ2 Γ3 e1 e2 ds eps1 eps2 h1 h2 =>
    have h1_shape : HasTypeDB Δ_ S_ (pre ++ Γ_old) e1 (Typ.tensor ds) eps1 Γ2 := hin ▸ h1
    have h2_shape : HasTypeDB Δ_ S_ Γ2 e2 (Typ.tensor ds) eps2 (pre' ++ Γ_old) := hout ▸ h2
    have hmid := hasTypeDB_tail_through_middle h1_shape h2_shape
    have hΓ2 : Γ2 = Γ2.take pre.length ++ Γ_old := by
      have := List.take_append_drop pre.length Γ2
      rw [hmid] at this; exact this.symm
    have h1' := tail_rebase_db h1 pre (Γ2.take pre.length) Γ_old Γ_new hin hΓ2 h_len
    have h2' := tail_rebase_db h2 (Γ2.take pre.length) pre' Γ_old Γ_new hΓ2 hout h_len
    exact HasTypeDB.tmul Δ_ S_ (pre ++ Γ_new) (Γ2.take pre.length ++ Γ_new)
            (pre' ++ Γ_new) e1 e2 ds eps1 eps2 h1' h2'
  | HasTypeDB.tsum Δ_ S_ Γ1 Γ2 e_ ds i eps_ hbody ds' _ =>
    have ih := tail_rebase_db hbody pre pre' Γ_old Γ_new hin hout h_len
    exact HasTypeDB.tsum Δ_ S_ (pre ++ Γ_new) (pre' ++ Γ_new) e_ ds i eps_ ih ds' True.intro
  | HasTypeDB.texpand Δ_ S_ Γ1 Γ2 e_ ds i k eps_ hbody ds' _ =>
    have ih := tail_rebase_db hbody pre pre' Γ_old Γ_new hin hout h_len
    exact HasTypeDB.texpand Δ_ S_ (pre ++ Γ_new) (pre' ++ Γ_new) e_ ds i k eps_ ih ds' True.intro
  | HasTypeDB.uniformLike Δ_ S_ Γ1 Γ2 e_ ds lo hi eps_ hbody =>
    have ih := tail_rebase_db hbody pre pre' Γ_old Γ_new hin hout h_len
    exact HasTypeDB.uniformLike Δ_ S_ (pre ++ Γ_new) (pre' ++ Γ_new) e_ ds lo hi eps_ ih
  | HasTypeDB.perform Δ_ S_ Γ1 Γ2 op e_ tArg tRet eps_ hbody hM =>
    have ih := tail_rebase_db hbody pre pre' Γ_old Γ_new hin hout h_len
    exact HasTypeDB.perform Δ_ S_ (pre ++ Γ_new) (pre' ++ Γ_new) op e_ tArg tRet eps_ ih hM
  | HasTypeDB.handle Δ_ S_ Γ1 Γ2 Γ3 body clauses ty epsH epsB hb hSubsH hClsH hCover hcls =>
    have hb_shape : HasTypeDB Δ_ S_ (pre ++ Γ_old) body ty epsB Γ2 := hin ▸ hb
    have hcls_shape : ClausesTypedDB Δ_ S_ Γ2 (pre' ++ Γ_old) ty (EffectRow.removeOps epsB epsH) clauses := hout ▸ hcls
    -- Use tail_through_middle via an artificial continuation: we need
    -- Γ2.drop pre.length = Γ_old. We have h1 = hb and from hcls
    -- (a ClausesTypedDB) we get length and monotonicity. Use length +
    -- clauses monotone.
    have hΓ2_drop : Γ2.drop pre.length = Γ_old := by
      apply List.ext_getElem?
      intro k
      by_cases hk : k < Γ_old.length
      · have hdrop_k : (Γ2.drop pre.length)[k]? = Γ2[pre.length + k]? := by
          rw [List.getElem?_drop]
        rw [hdrop_k]
        have hin_k : (pre ++ Γ_old)[pre.length + k]? = Γ_old[k]? :=
          LinearCtxDB.getElem?_append_tail pre Γ_old k
        have hout_k : (pre' ++ Γ_old)[pre'.length + k]? = Γ_old[k]? :=
          LinearCtxDB.getElem?_append_tail pre' Γ_old k
        have hlp1 := hasTypeDB_length_preservation hb_shape
        have hlp2 := hasTypeDB_length_preservation_clauses hcls_shape
        have hlen_eq : pre.length = pre'.length := by
          simp [List.length_append] at hlp1 hlp2; omega
        rcases hg : Γ_old[k]? with _ | ⟨_ | ti⟩
        · exfalso; rw [List.getElem?_eq_none_iff] at hg; omega
        · have hin_none : (pre ++ Γ_old)[pre.length + k]? = some none := hin_k.trans hg
          exact hasTypeDB_none_monotone hb_shape (pre.length + k) hin_none
        · have hin_live : (pre ++ Γ_old)[pre.length + k]? = some (some ti) := hin_k.trans hg
          rcases hasTypeDB_live_slot_monotone hb_shape (pre.length + k) ti hin_live with hmid_live | hmid_none
          · exact hmid_live
          · exfalso
            have hout_none := hasTypeDB_none_monotone_clauses hcls_shape (pre.length + k) hmid_none
            have hidx : pre'.length + k = pre.length + k := by omega
            rw [← hidx] at hout_none
            rw [hout_k] at hout_none
            rw [hg] at hout_none
            cases hout_none
      · have hdrop_len : (Γ2.drop pre.length).length = Γ_old.length := by
          have hlp1 := hasTypeDB_length_preservation hb_shape
          rw [List.length_drop]
          simp [List.length_append] at hlp1; omega
        have h1none : Γ_old[k]? = none := by
          rw [List.getElem?_eq_none_iff]; omega
        have h2none : (Γ2.drop pre.length)[k]? = none := by
          rw [List.getElem?_eq_none_iff, hdrop_len]; omega
        rw [h1none, h2none]
    have hΓ2 : Γ2 = Γ2.take pre.length ++ Γ_old := by
      have := List.take_append_drop pre.length Γ2
      rw [hΓ2_drop] at this; exact this.symm
    have hb' := tail_rebase_db hb pre (Γ2.take pre.length) Γ_old Γ_new hin hΓ2 h_len
    have hcls' := tail_rebase_clauses_db hcls (Γ2.take pre.length) pre' Γ_old Γ_new hΓ2 hout h_len
    exact HasTypeDB.handle Δ_ S_ (pre ++ Γ_new) (Γ2.take pre.length ++ Γ_new)
            (pre' ++ Γ_new) body clauses ty epsH epsB hb' hSubsH hClsH hCover hcls'
  | HasTypeDB.tgrad Δ_ S_ Γ slot ds dsOut body eps_ hbody hsub =>
    have hpp : pre = pre' := List.append_cancel_right (hin.symm.trans hout)
    have hin_b : some (Typ.tensor ds) :: Γ =
                 (some (Typ.tensor ds) :: pre) ++ Γ_old := by
      rw [hin]; simp [List.cons_append]
    have hout_b : slot :: Γ = (slot :: pre) ++ Γ_old := by
      rw [hin]; simp [List.cons_append]
    have ihb := tail_rebase_db hbody (some (Typ.tensor ds) :: pre) (slot :: pre)
                  Γ_old Γ_new hin_b hout_b h_len
    simp [List.cons_append] at ihb
    subst hpp
    exact HasTypeDB.tgrad Δ_ S_ (pre ++ Γ_new) slot ds dsOut body eps_ ihb hsub
  | HasTypeDB.tvmap Δ_ S_ Γ slot t1 t2 body eps_ d hbody =>
    have hpp : pre = pre' := List.append_cancel_right (hin.symm.trans hout)
    have hin_b : some t1 :: Γ = (some t1 :: pre) ++ Γ_old := by
      rw [hin]; simp [List.cons_append]
    have hout_b : slot :: Γ = (slot :: pre) ++ Γ_old := by
      rw [hin]; simp [List.cons_append]
    have ihb := tail_rebase_db hbody (some t1 :: pre) (slot :: pre) Γ_old Γ_new hin_b hout_b h_len
    simp [List.cons_append] at ihb
    subst hpp
    exact HasTypeDB.tvmap Δ_ S_ (pre ++ Γ_new) slot t1 t2 body eps_ d ihb
  | HasTypeDB.loc Δ_ S_ Γ ell ty hlook =>
    have hpp : pre = pre' := List.append_cancel_right (hin.symm.trans hout)
    rw [hpp]
    exact HasTypeDB.loc Δ_ S_ (pre' ++ Γ_new) ell ty hlook
  | HasTypeDB.subEff Δ_ S_ Γ Γ' e_ ty eps_ eps'_ hbody hSub =>
    have ih := tail_rebase_db hbody pre pre' Γ_old Γ_new hin hout h_len
    exact HasTypeDB.subEff Δ_ S_ (pre ++ Γ_new) (pre' ++ Γ_new) e_ ty eps_ eps'_ ih hSub
termination_by structural h

theorem tail_rebase_clauses_db
    {Δ : CapCtx} {S : StoreTyp}
    {Γ2 Γ3 : LinearCtxDB} {ty : Typ} {epsR : EffectRow}
    {cls : List (EffectLabel × TermDB)}
    (h : ClausesTypedDB Δ S Γ2 Γ3 ty epsR cls)
    (pre pre' Γ_old Γ_new : LinearCtxDB)
    (hin : Γ2 = pre ++ Γ_old) (hout : Γ3 = pre' ++ Γ_old)
    (h_len : Γ_old.length = Γ_new.length) :
    ClausesTypedDB Δ S (pre ++ Γ_new) (pre' ++ Γ_new) ty epsR cls := by
  match h with
  | ClausesTypedDB.nil Δ_ S_ Γ_ ty_ epsR_ =>
    subst hin
    have : pre = pre' := List.append_cancel_right hout
    subst this
    exact ClausesTypedDB.nil Δ_ S_ (pre ++ Γ_new) ty_ epsR_
  | ClausesTypedDB.cons Δ_ S_ Γ2_ Γ3_ slot1 slot2 ty_ tArg tRet epsR_ op hb rest hbody hrest =>
    -- Γ2_ = pre ++ Γ_old, Γ3_ = pre' ++ Γ_old from hin/hout.
    have hin_b : some (Typ.arrow tRet ty_ epsR_) :: some tArg :: Γ2_ =
                 (some (Typ.arrow tRet ty_ epsR_) :: some tArg :: pre) ++ Γ_old := by
      rw [hin]; simp [List.cons_append]
    have hout_b : slot1 :: slot2 :: Γ3_ = (slot1 :: slot2 :: pre') ++ Γ_old := by
      rw [hout]; simp [List.cons_append]
    have ihb := tail_rebase_db hbody
      (some (Typ.arrow tRet ty_ epsR_) :: some tArg :: pre)
      (slot1 :: slot2 :: pre') Γ_old Γ_new hin_b hout_b h_len
    have ihr := tail_rebase_clauses_db hrest pre pre' Γ_old Γ_new hin hout h_len
    simp [List.cons_append] at ihb
    exact ClausesTypedDB.cons Δ_ S_ (pre ++ Γ_new) (pre' ++ Γ_new) slot1 slot2
            ty_ tArg tRet epsR_ op hb rest ihb ihr
termination_by structural h
end

/-- Corollary: if a pure derivation holds over `Γ_old`, it also holds
    over any equal-length `Γ_new`. This is the `pre = pre' = []`
    specialization of `tail_rebase_db`. Used by
    `subst_preserves_typing_db` to rehome value derivations across
    intermediate contexts. -/
theorem pure_context_rebase_db
    {Δ : CapCtx} {S : StoreTyp}
    {v : TermDB} {t_v : Typ}
    {Γ_old Γ_new : LinearCtxDB}
    (h_v : HasTypeDB Δ S Γ_old v t_v [] Γ_old)
    (h_len : Γ_old.length = Γ_new.length) :
    HasTypeDB Δ S Γ_new v t_v [] Γ_new := by
  have := tail_rebase_db h_v [] [] Γ_old Γ_new (by simp) (by simp) h_len
  simpa using this

/-! ## Substitution obligation — doc block for Wave 5c → 5d

Under Option C the target statement is:

```lean
theorem subst_preserves_typing_db
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma : LinearCtxDB}
    {e v : TermDB} {t t_v : Typ} {eps : EffectRow}
    (j : Nat)
    (h_e : HasTypeDB Delta Sigma (Gamma.insertAt j (some t_v)) e t eps
             (Gamma.insertAt j none))
    (h_v : HasTypeDB Delta Sigma Gamma v t_v [] Gamma) :
    HasTypeDB Delta Sigma Gamma (substDBAux j v e) t eps Gamma
```

**Status after Wave 5d prep.** The following structural
prerequisites are now landed in this file:

- `hasTypeDB_length_preservation` (+ clauses partner) — Wave 5b.
- `hasTypeDB_none_monotone` (+ clauses partner) — Wave 5c.
- `hasTypeDB_live_slot_monotone` (+ clauses partner) — Wave 5d prep.
  Strengthens none-monotonicity with the matching direction: if
  an input slot is `some (some t)`, the output slot is either
  `some (some t)` (unchanged) or `some none` (consumed). No rule
  rewrites a live slot to a differently-typed live slot.
- `LinearCtxDB.getElem?_append_lt/ge`, `set_append_lt/ge`,
  `getElem?_append_tail` — pure append bookkeeping.
- `hasTypeDB_tail_pre_length_eq` — pre lengths match across tails.
- `hasTypeDB_tail_through_middle` — the none/live-slot sandwich:
  along a derivation `(pre ++ Γ_old) → Γ_mid → (pre' ++ Γ_old)`,
  the intermediate's drop-past-prefix equals `Γ_old` verbatim.
  Consumption of a live tail slot anywhere in `Γ_mid` would
  propagate `none` to the output through `h2`'s monotonicity,
  contradicting the intact output tail.

**Refined obstruction analysis.** During Wave 5c the originally-
proposed `pure_context_rebase_db` lemma was shown to be unprovable
by direct structural induction on a pure `Γ → Γ` derivation.
Counter-example: the abs case has body derivation
`(some t1 :: Γ) → (slot :: Γ)` which is *not* itself of the
`Γ' → Γ'` pure shape, so the induction hypothesis does not apply to
the body. The same failure recurs in every binder case and in
every multi-context case where the intermediate context Γ_mid is
only known to length-match, not to equal, the endpoints.

The correct generalization is a **tail-rebase** lemma:

```lean
theorem tail_rebase_db :
    Γ_old.length = Γ_new.length →
    HasTypeDB Δ Σ (pre ++ Γ_old) e t eps (pre' ++ Γ_old) →
    HasTypeDB Δ Σ (pre ++ Γ_new) e t eps (pre' ++ Γ_new)
```

`pure_context_rebase_db` is the specialization at `pre = pre' = []`.
This reformulation is sound because, along any derivation whose
endpoints both end in `Γ_old`, no slot of `Γ_old` can have been
consumed anywhere in the middle: consumption at position
`pre.length + k` would push `(pre ++ Γ_old)[pre.length + k]?` from
`some (some t)` to `some none`, and `hasTypeDB_none_monotone` would
then force the output at that position to still be `some none`,
contradicting `(pre' ++ Γ_old)[pre.length + k]? = some (some t')`.

**Remaining Wave 5e work.** With the Wave 5d prep helpers landed,
the residual proof obligations reduce to three tactical theorems:

1. `tail_rebase_db` (mutual with `tail_rebase_clauses_db`) — proved
   by structural match on the derivation. The foundation is in
   place: `hasTypeDB_tail_through_middle` handles every multi-
   context rule (`app`, `letBind`, `letpair`, `tpair`, `tadd`,
   `tmul`, `handle`) by giving
   `Γ_mid = Γ_mid.take pre.length ++ Γ_old`, and binder rules
   (`abs`, `letBind`, `letpair`, `tgrad`, `tvmap`) extend the
   prefix via `(slot :: pre) ++ Γ_old = slot :: (pre ++ Γ_old)`
   re-association. The var case splits on `i < pre.length`
   (substitute in the prefix via `set_append_lt` +
   `getElem?_append_lt`) vs `i ≥ pre.length` (vacuous by
   `set` + `getElem?_append_ge` — the output at position `i`
   would be `some none`, but by the append-right equation it must
   equal the live `Γ_old` slot, contradiction).
2. `pure_context_rebase_db` — `pre = pre' = []` specialization of
   `tail_rebase_db`, trivial by `List.nil_append`.
3. `subst_preserves_typing_db` (+ clauses partner) — pattern-match
   on `h_e`:
   - `var`: three-way split on `i` vs `j`, using
     `getElem?_insertAt_eq` + `set_insertAt_eq` + `h_v`.
   - Leaf/binder cases: `unit`, `const`, `loc` trivial; `abs`,
     `letBind`, `letpair`, `tgrad`, `tvmap` recurse with
     `weakening_head_db` (and its two-slot variant for letpair /
     clause bodies).
   - Multi-context cases: use `tail_rebase_db` to re-home `h_v`
     over the intermediate base provided by
     `hasTypeDB_tail_through_middle`.

Wave 5c lands none-monotonicity. Wave 5d prep lands live-slot
monotonicity, append bookkeeping, and the tail-through-middle
sandwich. Wave 5e lands tail-rebase proper, the rebase corollary,
and subst itself, and bridges named ↔ DB so the two named sorries
in `Substitution.lean` can close against DB results. -/

end LaCaDiLE
