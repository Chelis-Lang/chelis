-- LaCaDiLE/EffectCorrectness.lean — effect correctness theorem (Phase 2 proof).
--
-- WS2.7 target: a well-typed closed term with empty effect row can
-- reduce to a value without ever encountering an unhandled `perform`.
-- Current branch proves the current-frontier unhandled-perform
-- exclusion; the multi-step corollary still depends on the final
-- runtime/preservation package.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.Operational
import LaCaDiLE.Progress

namespace LaCaDiLE

/-- Effect correctness: a term with empty effect row does not perform
    an unhandled operation at its current evaluation frontier. The
    `StuckOnPerform` witness is exactly Progress's residual unhandled
    perform shape, and `stuck_bubbles` shows that such a witness would
    force the operation to occur in the term's effect row. -/
theorem effect_correctness
    (_sigma : Store) (Sigma : StoreTyp) (e : Term) (t : Typ)
    (h : HasType [] Sigma [] e t [] []) :
    ¬ ∃ op, StuckOnPerform op e := by
  intro hstuck
  rcases hstuck with ⟨op, hperform⟩
  cases hperform with
  | mk Es v hv hEs =>
      have hop : op ∈ ([] : EffectRow) := stuck_bubbles Es hEs h
      simp at hop

end LaCaDiLE
