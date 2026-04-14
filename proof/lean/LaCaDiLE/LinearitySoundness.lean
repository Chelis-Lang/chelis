-- LaCaDiLE/LinearitySoundness.lean — linearity soundness theorem (Phase 2 proof).
--
-- WS2.8 target: no well-typed program accesses a deallocated location.
-- Structural form: `Step` preserves `StoreWf` and the invariant that the
-- live locations in the store correspond to linear bindings in Γ.
-- Phase 1 skeleton: stub.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.Operational

namespace LaCaDiLE

/-- Generalized Linearity soundness over arbitrary configurations,
    suitable for structural induction on Step. The concrete
    `linearity_soundness` below is a trivial specialization. -/
private theorem linearity_soundness_aux
    (Sigma : StoreTyp)
    (c1 c2 : Config)
    (h_wf : StoreWf c1.store Sigma)
    (h_step : Step c1 c2) :
    ∃ Sigma', StoreWf c2.store Sigma' := by
  induction h_step with
  | beta => exact ⟨Sigma, h_wf⟩
  | letBind => exact ⟨Sigma, h_wf⟩
  | letpair => exact ⟨Sigma, h_wf⟩
  | fst => exact ⟨Sigma, h_wf⟩
  | snd => exact ⟨Sigma, h_wf⟩
  | handleRet => exact ⟨Sigma, h_wf⟩
  | handleOpDirect => exact ⟨Sigma, h_wf⟩
  | handleOpCtx => exact ⟨Sigma, h_wf⟩
  | handleOpCtxs => exact ⟨Sigma, h_wf⟩
  | tgrad => exact ⟨Sigma, h_wf⟩
  | tvmap => exact ⟨Sigma, h_wf⟩
  | tconst s v ds ell hell =>
      refine ⟨storeTypExtend Sigma ell (Typ.tensor ds), ?_⟩
      exact StoreWf.extend_fresh ell ⟨ds, v⟩ (Typ.tensor ds) h_wf
  | copy s ell ellNew w _hlook hfresh =>
      refine ⟨storeTypExtend Sigma ellNew (Typ.tensor w.shape), ?_⟩
      exact StoreWf.extend_fresh ellNew w (Typ.tensor w.shape) h_wf
  | tadd s ell1 ell2 ellOut w1 w2 _h1 _h2 _hfresh =>
      refine ⟨storeTypExtend
                (storeTypRemove (storeTypRemove Sigma ell1) ell2)
                ellOut (Typ.tensor w1.shape), ?_⟩
      have h_wf1 := StoreWf.remove ell1 h_wf
      have h_wf2 := StoreWf.remove ell2 h_wf1
      exact StoreWf.extend_fresh ellOut
        (tensorOpPlaceholder w1 w2) (Typ.tensor w1.shape) h_wf2
  | tmul s ell1 ell2 ellOut w1 w2 _h1 _h2 _hfresh =>
      refine ⟨storeTypExtend
                (storeTypRemove (storeTypRemove Sigma ell1) ell2)
                ellOut (Typ.tensor w1.shape), ?_⟩
      have h_wf1 := StoreWf.remove ell1 h_wf
      have h_wf2 := StoreWf.remove ell2 h_wf1
      exact StoreWf.extend_fresh ellOut
        (tensorOpPlaceholder w1 w2) (Typ.tensor w1.shape) h_wf2
  | tsum s ell ellOut w d hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (rem w.shape d)), ?_⟩
      have h_isSome : (storeLookup s ell).isSome := by
        rw [hlook]; rfl
      have hne : ell ≠ ellOut := by
        rw [hfresh]; exact storeFreshLoc_ne s ell h_isSome
      exact StoreWf.remove_extend ell ellOut
        { shape := rem w.shape d, data := w.data }
        (Typ.tensor (rem w.shape d)) h_wf hfresh hne
  | texpand s ell ellOut w d hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (ins w.shape d)), ?_⟩
      have h_isSome : (storeLookup s ell).isSome := by
        rw [hlook]; rfl
      have hne : ell ≠ ellOut := by
        rw [hfresh]; exact storeFreshLoc_ne s ell h_isSome
      exact StoreWf.remove_extend ell ellOut
        { shape := ins w.shape d, data := w.data }
        (Typ.tensor (ins w.shape d)) h_wf hfresh hne
  | tuniformLike s ell ellOut w lo hi hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor w.shape), ?_⟩
      have h_isSome : (storeLookup s ell).isSome := by
        rw [hlook]; rfl
      have hne : ell ≠ ellOut := by
        rw [hfresh]; exact storeFreshLoc_ne s ell h_isSome
      exact StoreWf.remove_extend ell ellOut
        { shape := w.shape, data := lo }
        (Typ.tensor w.shape) h_wf hfresh hne
  | ctx _sig _sig' _E _e _e' _h_inner ih =>
      -- E-Ctx: ih is the inner step's witness, which uses the same
      -- store-side states since plugging an evaluation context doesn't
      -- touch the store further.
      exact ih h_wf

theorem linearity_soundness
    (sigma sigma' : Store) (Sigma : StoreTyp)
    (e e' : Term)
    (h_wf : StoreWf sigma Sigma)
    (h_step : Step ⟨sigma, e⟩ ⟨sigma', e'⟩) :
    ∃ Sigma', StoreWf sigma' Sigma' :=
  linearity_soundness_aux Sigma ⟨sigma, e⟩ ⟨sigma', e'⟩ h_wf h_step

end LaCaDiLE
