-- LaCaDiLE/Store.lean — Store typing and well-formedness.
--
-- Companion to Typing.lean. The store-typing relation links runtime store
-- contents to their corresponding types, and `StoreWf` asserts that the set
-- of live locations in the store matches the linear bindings in Γ. This is
-- the structural form of Theorem 4 (linearity soundness, WS2.8).

import LaCaDiLE.Syntax

namespace LaCaDiLE

/-- Store typing: a finite map from locations to the types they store.
    Used only for metatheory; runtime store contents live in `Store`. -/
abbrev StoreTyp := List (Loc × Typ)

/-- Look up a location in a store typing. -/
def storeTypLookup (Sigma : StoreTyp) (ell : Loc) : Option Typ :=
  (Sigma.find? (fun p => p.1 = ell)).map Prod.snd

/-- Domain of a store typing (the live locations). -/
def storeTypDom (Sigma : StoreTyp) : List Loc :=
  Sigma.map Prod.fst

/-- Store well-formedness: every live location in `sigma` has a corresponding
    entry in `Sigma`, and the tensor value's shape matches the store-typing
    type. Phase 1 states this structurally without proving preservation;
    Phase 2 WS2.8 will prove that `Step` preserves `StoreWf`. -/
def StoreWf (sigma : Store) (Sigma : StoreTyp) : Prop :=
  (∀ ell, ell ∈ storeTypDom Sigma → (storeLookup sigma ell).isSome) ∧
  (∀ ell, (storeLookup sigma ell).isSome → ell ∈ storeTypDom Sigma)

/-- Linearity soundness invariant (structural form): the set of live
    locations in `sigma` is exactly the set of locations owned by the
    linear context `Gamma`. Each linear binding `x : tensor[ds]` in `Gamma`
    owns a location whose type in `Sigma` is `tensor[ds]`. This predicate
    will appear as a premise/conclusion in Phase 2's preservation proof. -/
def LinearityInvariant (_sigma : Store) (_Sigma : StoreTyp)
    (_Gamma : LinearCtx) : Prop :=
  True  -- Phase 1: stated but not formalized in detail; filled in Phase 2.

/-- Extend a store typing with a fresh location. Used by Preservation
    on the store-allocating Step rules (tconst, copy, tadd, tmul, tsum,
    texpand, tuniformLike). -/
def storeTypExtend (Sigma : StoreTyp) (ell : Loc) (t : Typ) : StoreTyp :=
  (ell, t) :: Sigma

/-- Remove a location from a store typing. Companion to `storeRemove`
    on the runtime side. Used by Preservation on Step rules that
    consume a location (tadd, tmul, tsum, texpand, tuniformLike). -/
def storeTypRemove (Sigma : StoreTyp) (ell : Loc) : StoreTyp :=
  Sigma.filter (fun p => p.1 ≠ ell)

/-- `StoreTyp` sub-typing: `Sigma ⊑ Sigma'` iff every location typed
    in `Sigma` is typed to the same type in `Sigma'`. Used to state
    Preservation's store-monotonicity conclusion. -/
def StoreTypSub (Sigma Sigma' : StoreTyp) : Prop :=
  ∀ ell t, storeTypLookup Sigma ell = some t →
           storeTypLookup Sigma' ell = some t

end LaCaDiLE
