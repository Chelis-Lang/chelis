-- LaCaDiLE/Substitution.lean — substitution lemma (Phase 2 proof).
--
-- WS2.1 target: if a term is well-typed in a context that binds `x` to
-- `t1`, and a value `v` of type `t1` is available, then substituting
-- `v` for `x` preserves typing.
--
-- Phase 1 skeleton: signature-only stub with `sorry`. Phase 2 replaces
-- with a real proof that must handle the naive `subst` in Operational.lean
-- by either proving capture-avoidance or upgrading to a de-Bruijn
-- representation.

import LaCaDiLE.Syntax
import LaCaDiLE.Typing
import LaCaDiLE.Operational

namespace LaCaDiLE

/-- Substitution preserves typing. -/
theorem subst_preserves_typing
    (Delta : CapCtx) (Gamma1 Gamma2 : LinearCtx)
    (x : String) (t1 t2 : Typ) (eps : EffectRow)
    (e v : Term)
    (_h_e : HasType Delta (Gamma1 ++ [(x, t1)]) e t2 eps Gamma2)
    (_h_v : HasType Delta Gamma1 v t1 [] Gamma1) :
    HasType Delta Gamma1 (subst e v x) t2 eps
            (Gamma2.filter (fun p => p.1 ≠ x)) := by
  sorry

end LaCaDiLE
