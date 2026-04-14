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
  | tconst sigma_in v ds ell hell =>
      refine ⟨storeTypExtend Sigma ell (Typ.tensor ds), ?_⟩
      exact StoreWf.extend_fresh ell ⟨ds, v⟩ (Typ.tensor ds) h_wf hell
  | copy sigma_in ell ellNew w _hlook hfresh =>
      refine ⟨storeTypExtend Sigma ellNew (Typ.tensor w.shape), ?_⟩
      exact StoreWf.extend_fresh ellNew w (Typ.tensor w.shape) h_wf hfresh
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
  | ctx _sig _sig' _E _e _e' _h_inner =>
      -- E-Ctx: cases can't directly dispatch because Step is indexed
      -- on Configs; we lose the induction hypothesis. Wave 5 will
      -- rewrite this using a helper that takes Step structurally.
      sorry -- TODO Wave 5: recurse via Step-structural helper

end LaCaDiLE
