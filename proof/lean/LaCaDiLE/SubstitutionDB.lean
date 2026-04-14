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

/-! ## Substitution obligation — doc block for Wave 5b → 5c

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

**Wave 5b status.** Added `hasTypeDB_length_preservation` above,
plus its mutual partner `hasTypeDB_length_preservation_clauses`.
These are the structural prerequisite for every subsequent Option C
metatheoretic lemma over linear contexts, including the slot
persistence inversion that subst needs.

**Remaining obstruction for `subst_preserves_typing_db`.** The task
as originally scoped for Wave 5b requires, in addition to slot
persistence, a way to thread `h_v` through intermediate contexts of
multi-context constructors (`app Γ1 Γ2 Γ3`, `letBind`, `letpair`,
`handle`, binary tensor ops). Slot persistence alone is
insufficient: the IH call on `h2 : HasTypeDB .. Γ2 .. Γ3` needs a
hypothesis `h_v : HasTypeDB .. Γ2_base v t_v [] Γ2_base` under the
*intermediate* base, not the outer base `Γ`. Because `v` is typed at
pure effect with the trivial context transition `Γ → Γ`, it uses no
linear resources and so should transport to any length-matching
context, but that transport is itself an unstated lemma
(`pure_context_rebase`).

Concretely Wave 5c should add, in order:

1. `hasTypeDB_slot_persistence` — use `hasTypeDB_length_preservation`
   above to rule out empty-base edge cases when peeling
   binder-augmented outputs.
2. `pure_context_rebase` — a `Γ → Γ` pure derivation transports to
   any length-matching `Γ' → Γ'`. This is the "no free linear vars"
   observation made formal, and it relies on (1) applied at cutoff 0.
3. `subst_preserves_typing_db` as a mutual block with
   `subst_preserves_typing_clauses_db`. The var case is immediate
   from `getElem?_insertAt_eq` + `set_insertAt_eq` + `h_v`; binder
   cases use `weakening_head_db`; multi-context cases chain (1) and
   (2) to rebase `h_v` for each sub-derivation.

Wave 5b lands only the length preservation foundation. Slot
persistence + rebase + subst is Wave 5c alongside the named ↔ DB
translation so the two named theorems in `Substitution.lean` can
close against DB results at the same time. -/

end LaCaDiLE
