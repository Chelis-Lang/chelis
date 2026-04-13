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
import LaCaDiLE.Typing
import LaCaDiLE.AdjointTransform

namespace LaCaDiLE

/-- The adjoint transformation preserves typing. -/
theorem adjoint_preserves_typing
    (Delta : CapCtx) (Gamma : LinearCtx)
    (x gs : String) (ds dsOut : DimList) (e : Term) (eps : EffectRow)
    (_h_e : HasType (Capability.diff :: Delta)
                   (Gamma ++ [(x, Typ.tensor ds)])
                   e (Typ.tensor dsOut) eps Gamma)
    (_h_compat : subsetEffRow eps DiffCompat = true) :
    HasType Delta
            (Gamma ++ [(x, Typ.tensor ds), (gs, Typ.tensor dsOut)])
            (adjoint e x (Term.var gs))
            (Typ.tensor ds)
            (eps ++ [EffectLabel.accum])
            Gamma := by
  sorry

end LaCaDiLE
