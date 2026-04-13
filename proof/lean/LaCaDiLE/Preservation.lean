-- LaCaDiLE/Preservation.lean — preservation theorem (Phase 2 proof).
--
-- WS2.5 target: reduction preserves typing and store well-formedness.
-- Phase 1 skeleton: stub.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.Operational

namespace LaCaDiLE

/-- Preservation: if a configuration is well-typed and steps, the
    resulting configuration has the same type and preserves store
    well-formedness. -/
theorem preservation
    (sigma sigma' : Store) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (e e' : Term) (t : Typ) (eps : EffectRow)
    (_h_typ : HasType [] Gamma e t eps [])
    (_h_wf : StoreWf sigma Sigma)
    (_h_step : Step ⟨sigma, e⟩ ⟨sigma', e'⟩) :
    ∃ Sigma',
      HasType [] Gamma e' t eps [] ∧ StoreWf sigma' Sigma' := by
  sorry

end LaCaDiLE
