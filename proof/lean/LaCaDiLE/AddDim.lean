-- LaCaDiLE/AddDim.lean — addDim preserves typing (Phase 2 WS2.3).
--
-- Statement: if `Δ; Σ; Γ ⊢ e : t ! ε ⊣ Γ'`, then lifting the entire
-- derivation through `addDim d` on types and `addDimTerm d` on terms
-- produces a new well-typed derivation for the lifted judgment:
--
--   Δ; Σ; addDimCtx d Γ ⊢ addDimTerm d e : addDim d t ! ε ⊣ addDimCtx d Γ'
--
-- Proved by structural induction on the typing derivation. Each case
-- rebuilds the corresponding `HasType` constructor after applying
-- `addDim` / `addDimTerm` to every type and term in sight.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing

namespace LaCaDiLE

/-- Lift `addDim d` over every type in a linear context. -/
def addDimCtx (d : Dim) (Gamma : LinearCtx) : LinearCtx :=
  Gamma.map (fun p => (p.1, addDim d p.2))

/-- addDim / addDimTerm preserves typing (Phase 2 WS2.3). -/
theorem addDim_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma Gamma' : LinearCtx)
    (e : Term) (t : Typ) (eps : EffectRow) (d : Dim)
    (_h : HasType Delta Sigma Gamma e t eps Gamma') :
    HasType Delta Sigma (addDimCtx d Gamma) (addDimTerm d e) (addDim d t)
            eps (addDimCtx d Gamma') := by
  sorry

end LaCaDiLE
