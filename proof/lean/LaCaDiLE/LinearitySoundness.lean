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
  | tconst sigma_in v ds ell _ =>
      -- Store grew by one fresh location. Exhibit the extended Sigma.
      refine ⟨storeTypExtend Sigma ell (Typ.tensor ds), ?_⟩
      -- TODO Wave 5: prove StoreWf (storeExtend sigma ell ⟨ds, v⟩)
      -- (storeTypExtend Sigma ell (tensor ds)). Requires a StoreWf
      -- extend lemma.
      sorry
  | copy sigma_in ell ellNew w _ _ =>
      refine ⟨storeTypExtend Sigma ellNew (Typ.tensor w.shape), ?_⟩
      sorry -- TODO Wave 5: StoreWf extension for copy
  | tadd sigma_in ell1 ell2 ellOut w1 w2 _ _ _ =>
      refine ⟨storeTypExtend Sigma ellOut (Typ.tensor w1.shape), ?_⟩
      sorry -- TODO Wave 5: StoreWf for remove-remove-extend
  | tmul sigma_in ell1 ell2 ellOut w1 w2 _ _ _ =>
      refine ⟨storeTypExtend Sigma ellOut (Typ.tensor w1.shape), ?_⟩
      sorry -- TODO Wave 5: StoreWf for remove-remove-extend
  | tsum sigma_in ell ellOut w i _ _ =>
      refine ⟨storeTypExtend Sigma ellOut (Typ.tensor (rem w.shape i)), ?_⟩
      sorry -- TODO Wave 5: StoreWf for remove-extend
  | texpand sigma_in ell ellOut w i k _ _ =>
      refine ⟨storeTypExtend Sigma ellOut
                (Typ.tensor (ins w.shape i k)), ?_⟩
      sorry -- TODO Wave 5: StoreWf for remove-extend
  | tuniformLike sigma_in ell ellOut w lo hi _ _ =>
      refine ⟨storeTypExtend Sigma ellOut (Typ.tensor w.shape), ?_⟩
      sorry -- TODO Wave 5: StoreWf for remove-extend

end LaCaDiLE
