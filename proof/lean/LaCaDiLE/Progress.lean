-- LaCaDiLE/Progress.lean — progress theorem (Phase 2 proof).
--
-- WS2.4 target: every well-typed closed term with an empty effect row
-- is either a value or can take a step. Phase 1 skeleton: stub.

import LaCaDiLE.Syntax
import LaCaDiLE.Typing
import LaCaDiLE.Operational

namespace LaCaDiLE

/-- Progress: a closed, effect-free, well-typed term is either a value
    or reducible. -/
theorem progress
    (sigma : Store) (e : Term) (t : Typ)
    (_h : HasType [] [] e t [] []) :
    IsValue e ∨ ∃ sigma' e', Step ⟨sigma, e⟩ ⟨sigma', e'⟩ := by
  sorry

end LaCaDiLE
