-- LaCaDiLE/Progress.lean — progress theorem (Phase 2 proof).
--
-- WS2.4 target: every well-typed closed term with an empty effect row
-- is either a value or can take a step. Phase 1 skeleton: stub.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.Operational

namespace LaCaDiLE

/-- Progress: a closed, effect-free, well-typed term under an arbitrary
    store typing is either a value or reducible. Wave 0 P3: the
    judgment now threads `Sigma` through `HasType`. -/
theorem progress
    (sigma : Store) (Sigma : StoreTyp) (e : Term) (t : Typ)
    (_h : HasType [] Sigma [] e t [] []) :
    IsValue e ∨ ∃ sigma' e', Step ⟨sigma, e⟩ ⟨sigma', e'⟩ := by
  sorry

end LaCaDiLE
