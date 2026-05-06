-- LaCaDiLE/DimSafety.lean — dimension safety corollary (Phase 2 proof).
--
-- WS2.6 target: well-typed programs never hit a runtime dimension
-- mismatch (e.g., adding tensors of different shapes). Corollary of
-- preservation. Current branch proves the primitive-redex frontier
-- corollary under an explicit store-shape sidecar.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.Operational
import LaCaDiLE.Preservation

namespace LaCaDiLE

/-- Concrete runtime shape mismatch at the primitive redex frontier. -/
inductive PrimitiveShapeMismatch (sigma : Store) : Term → Prop where
  | add
      (ell1 ell2 : Loc) (w1 w2 : TensorVal) :
      storeLookup sigma ell1 = some w1 →
      storeLookup sigma ell2 = some w2 →
      w1.shape ≠ w2.shape →
      PrimitiveShapeMismatch sigma (Term.add (Term.loc ell1) (Term.loc ell2))
  | mul
      (ell1 ell2 : Loc) (w1 w2 : TensorVal) :
      storeLookup sigma ell1 = some w1 →
      storeLookup sigma ell2 = some w2 →
      w1.shape ≠ w2.shape →
      PrimitiveShapeMismatch sigma (Term.mul (Term.loc ell1) (Term.loc ell2))

/-- Dimension safety at the primitive redex frontier: a well-typed
    `add` or `mul` redex cannot apply the primitive to two tensor
    locations with mismatching runtime shapes, provided the store and
    store typing agree on tensor shapes. -/
theorem dimension_safety
    (sigma : Store) (Sigma : StoreTyp)
    (e : Term) (t : Typ) (eps : EffectRow)
    (h_shape : StoreShapeConsistent sigma Sigma)
    (h : HasType [] Sigma [] e t eps []) :
    ¬ PrimitiveShapeMismatch sigma e := by
  intro hbad
  cases hbad with
  | add ell1 ell2 w1 w2 hlook1 hlook2 hmismatch =>
      obtain ⟨ds, Gamma2, eps1, eps2, _ht, h1, h2⟩ := HasType.add_inv h
      obtain ⟨hTy1, _hCtx1⟩ := HasType.loc_inv h1
      obtain ⟨hTy2, _hCtx2⟩ := HasType.loc_inv h2
      have hshape1 : w1.shape = ds := h_shape ell1 ds w1 hTy1 hlook1
      have hshape2 : w2.shape = ds := h_shape ell2 ds w2 hTy2 hlook2
      exact hmismatch (hshape1.trans hshape2.symm)
  | mul ell1 ell2 w1 w2 hlook1 hlook2 hmismatch =>
      obtain ⟨ds, Gamma2, eps1, eps2, _ht, h1, h2⟩ := HasType.mul_inv h
      obtain ⟨hTy1, _hCtx1⟩ := HasType.loc_inv h1
      obtain ⟨hTy2, _hCtx2⟩ := HasType.loc_inv h2
      have hshape1 : w1.shape = ds := h_shape ell1 ds w1 hTy1 hlook1
      have hshape2 : w2.shape = ds := h_shape ell2 ds w2 hTy2 hlook2
      exact hmismatch (hshape1.trans hshape2.symm)

end LaCaDiLE
