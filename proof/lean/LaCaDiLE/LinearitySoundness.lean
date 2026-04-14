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

/-- Linearity soundness: reduction preserves the live-location /
    linear-context correspondence. Structural form — shows that every
    Step produces a reachable store-typing witness. Stronger forms
    (exact Sigma monotonicity via StoreTypSub, correspondence with
    linear context domain) are tracked for Wave 5 once `StoreWf` is
    sharpened. -/
theorem linearity_soundness
    (sigma sigma' : Store) (Sigma : StoreTyp)
    (e e' : Term)
    (h_wf : StoreWf sigma Sigma)
    (h_step : Step ⟨sigma, e⟩ ⟨sigma', e'⟩) :
    ∃ Sigma', StoreWf sigma' Sigma' := by
  -- Case-analyze on the Step constructor. Most cases leave the store
  -- unchanged (beta, letBind, letpair, fst, snd, handleRet,
  -- handleOpDirect, tgrad, tvmap) — for these Sigma' = Sigma works.
  -- Store-mutating cases (tconst, copy, tadd, tmul, tsum, texpand,
  -- tuniformLike) need to exhibit a fresh Sigma' that tracks the
  -- new store shape. For Wave 1 we use a blunt witness — the image
  -- of the new store — and defer the sharpened correspondence to
  -- Wave 5 LinearityInvariant work.
  cases h_step with
  | beta => exact ⟨Sigma, h_wf⟩
  | letBind => exact ⟨Sigma, h_wf⟩
  | letpair => exact ⟨Sigma, h_wf⟩
  | fst => exact ⟨Sigma, h_wf⟩
  | snd => exact ⟨Sigma, h_wf⟩
  | handleRet => exact ⟨Sigma, h_wf⟩
  | handleOpDirect => exact ⟨Sigma, h_wf⟩
  | tgrad => exact ⟨Sigma, h_wf⟩
  | tvmap => exact ⟨Sigma, h_wf⟩
  | tconst s v ds ell hell =>
      refine ⟨storeTypExtend Sigma ell (Typ.tensor ds), ?_⟩
      exact StoreWf.extend_fresh ell ⟨ds, v⟩ (Typ.tensor ds) h_wf hell
  | copy s ell ellNew w _hlook hfresh =>
      refine ⟨storeTypExtend Sigma ellNew (Typ.tensor w.shape), ?_⟩
      exact StoreWf.extend_fresh ellNew w (Typ.tensor w.shape) h_wf hfresh
  | tadd s ell1 ell2 ellOut w1 w2 _ _ _ =>
      refine ⟨storeTypExtend Sigma ellOut (Typ.tensor w1.shape), ?_⟩
      sorry -- TODO Wave 5: StoreWf for remove-remove-extend
  | tmul s ell1 ell2 ellOut w1 w2 _ _ _ =>
      refine ⟨storeTypExtend Sigma ellOut (Typ.tensor w1.shape), ?_⟩
      sorry -- TODO Wave 5: StoreWf for remove-remove-extend
  | tsum s ell ellOut w i hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (rem w.shape i)), ?_⟩
      have h_isSome : (storeLookup sigma ell).isSome := by
        rw [hlook]; rfl
      have hne : ell ≠ ellOut := by
        rw [hfresh]; exact storeFreshLoc_ne sigma ell h_isSome
      exact StoreWf.remove_extend ell ellOut
        { shape := rem w.shape i, data := w.data }
        (Typ.tensor (rem w.shape i)) h_wf hfresh hne
  | texpand s ell ellOut w i k hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (ins w.shape i k)), ?_⟩
      have h_isSome : (storeLookup sigma ell).isSome := by
        rw [hlook]; rfl
      have hne : ell ≠ ellOut := by
        rw [hfresh]; exact storeFreshLoc_ne sigma ell h_isSome
      exact StoreWf.remove_extend ell ellOut
        { shape := ins w.shape i k, data := w.data }
        (Typ.tensor (ins w.shape i k)) h_wf hfresh hne
  | tuniformLike s ell ellOut w lo hi hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor w.shape), ?_⟩
      have h_isSome : (storeLookup sigma ell).isSome := by
        rw [hlook]; rfl
      have hne : ell ≠ ellOut := by
        rw [hfresh]; exact storeFreshLoc_ne sigma ell h_isSome
      exact StoreWf.remove_extend ell ellOut
        { shape := w.shape, data := lo }
        (Typ.tensor w.shape) h_wf hfresh hne
  | ctx _sig _sig' _E _e _e' _h_inner =>
      -- E-Ctx: cases can't directly dispatch because Step is indexed
      -- on Configs; we lose the induction hypothesis. Wave 5 will
      -- rewrite this using a helper that takes Step structurally.
      sorry -- TODO Wave 5: recurse via Step-structural helper

end LaCaDiLE
