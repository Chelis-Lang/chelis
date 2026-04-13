-- LaCaDiLE/AdjointTyping.lean — adjoint typing lemma (Phase 2 proof).
--
-- WS2.2 target: the adjoint transformation preserves typing. Given a
-- well-typed body `e : tensor[dsOut] ! eps` in a context containing the
-- differentiated parameter `x : tensor[ds]`, and a seed gradient `gs`
-- of type `tensor[dsOut]`, the term `adjoint e x (var "gs")` type-checks
-- with effect row `eps ∪ {Accum}`.
--
-- Phase 1 skeleton: signature-only stub with `sorry`. This is the
-- linchpin of the entire metatheory and the hardest Phase 2 proof
-- besides AD correctness (WS2.9).

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.AdjointTransform

namespace LaCaDiLE

/-- The adjoint transformation preserves typing. Wave 0 P3: includes
    the store typing `Sigma` parameter, which passes through unchanged
    since the adjoint transformation operates on terms, not stores.
    Wave 0 P2: the output effect row uses `EffectRow.union` to avoid
    double-counting `accum` when the input already contains it. -/
theorem adjoint_preserves_typing
    (Delta : CapCtx) (Sigma : StoreTyp) (Gamma : LinearCtx)
    (x gs : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
    (_h_e : HasType (Capability.diff :: Delta) Sigma
                   (Gamma ++ [(x, Typ.tensor ds)])
                   e (Typ.tensor dsOut) eps Gamma)
    (_h_compat : subsetEffRow eps DiffCompat = true) :
    HasType Delta Sigma
            (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
            (adjoint e x (Term.var gs))
            (Typ.tensor ds)
            (EffectRow.union eps [EffectLabel.accum])
            Gamma := by
  sorry

end LaCaDiLE
