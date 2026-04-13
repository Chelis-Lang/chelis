-- LaCaDiLE/EffectCorrectness.lean — effect correctness theorem (Phase 2 proof).
--
-- WS2.7 target: a well-typed closed term with empty effect row can
-- reduce to a value without ever encountering an unhandled `perform`.
-- Phase 1 skeleton: stub.

import LaCaDiLE.Syntax
import LaCaDiLE.Typing
import LaCaDiLE.Operational

namespace LaCaDiLE

/-- Effect correctness: a term with empty effect row does not perform
    an unhandled operation under reduction. -/
theorem effect_correctness
    (_sigma : Store) (_e : Term) (_t : Typ)
    (_h : HasType [] [] _e _t [] []) :
    True := trivial

end LaCaDiLE
