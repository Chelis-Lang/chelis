-- LaCaDiLE/ADCorrectness.lean — AD correctness theorem surface.
--
-- This file deliberately keeps the denotational semantics abstract:
-- a concrete mathematical model supplies a `PrimitiveADSpec` with
-- primitive derivative laws and structural closure rules for the
-- first-order fragment implemented by `adjointTyped`.

import LaCaDiLE.AdjointTransform
import LaCaDiLE.Typing

namespace LaCaDiLE

/-! ## Abstract denotational AD model

`PrimitiveADSpec` is the semantic contract used by the proof package.
It does not add axioms: users instantiate the record with a concrete
interpretation of tensors, environments, and primitive derivative laws.
The theorem below proves that the syntactic supported-fragment side
condition is enough to assemble the model's term-correctness evidence
and then exposes the final `grad` correctness claim for the generated
typed adjoint builder.
-/

/-- A semantic model for the supported first-order AD fragment.

`TermADCorrect x e` should be read as: under this model, the AD rule for
`e` is denotationally correct for the differentiated variable `x`.
`GradADCorrect ... adj` is the public mathematical statement for
`grad`: applying the generated adjoint builder `adj` to an output
cotangent denotes the pullback/gradient of `body`.

The fields are split between primitive laws (`const_law`, `add_law`,
`mul_law`, `sum_law`, `expand_law`, `copy_law`,
`perform_diffcompat_law`) and structural closure rules for the current
first-order syntax (`let`, products, projections, handlers). Unsupported
higher-order forms are intentionally absent. -/
structure PrimitiveADSpec where
  TermADCorrect : String → Term → Prop
  ClausesADCorrect :
    String → Term → List (EffectLabel × String × String × Term) → Prop
  GradADCorrect :
    CapCtx → StoreTyp → LinearCtx → String → DimList → DimList →
      Term → EffectRow → (Term → Term) → Prop

  var_law :
    ∀ x y, TermADCorrect x (Term.var y)
  const_law :
    ∀ x v ds, TermADCorrect x (Term.const v ds)
  unit_law :
    ∀ x, TermADCorrect x Term.unit
  loc_law :
    ∀ x ell, TermADCorrect x (Term.loc ell)
  add_law :
    ∀ x e1 e2,
      TermADCorrect x e1 →
      TermADCorrect x e2 →
      TermADCorrect x (Term.add e1 e2)
  mul_law :
    ∀ x e1 e2,
      TermADCorrect x e1 →
      TermADCorrect x e2 →
      TermADCorrect x (Term.mul e1 e2)
  sum_law :
    ∀ x e d,
      TermADCorrect x e →
      TermADCorrect x (Term.sum e d)
  expand_law :
    ∀ x e d,
      TermADCorrect x e →
      TermADCorrect x (Term.expand e d)
  copy_law :
    ∀ x e,
      TermADCorrect x e →
      TermADCorrect x (Term.copy e)
  let_law :
    ∀ x y e1 e2,
      TermADCorrect x e1 →
      TermADCorrect x e2 →
      TermADCorrect x (Term.letBind y e1 e2)
  letpair_law :
    ∀ x y z e1 e2,
      TermADCorrect x e1 →
      TermADCorrect x e2 →
      TermADCorrect x (Term.letpair y z e1 e2)
  pair_law :
    ∀ x e1 e2,
      TermADCorrect x e1 →
      TermADCorrect x e2 →
      TermADCorrect x (Term.pair e1 e2)
  fst_law :
    ∀ x tRight e,
      TermADCorrect x e →
      TermADCorrect x (Term.fst tRight e)
  snd_law :
    ∀ x tLeft e,
      TermADCorrect x e →
      TermADCorrect x (Term.snd tLeft e)
  perform_diffcompat_law :
    ∀ x op e,
      op ∈ DiffCompat →
      TermADCorrect x e →
      TermADCorrect x (Term.perform op e)
  handle_law :
    ∀ x epsH body clauses,
      TermADCorrect x body →
      ClausesADCorrect x body clauses →
      TermADCorrect x (Term.handle epsH body clauses)

  clauses_nil_law :
    ∀ x body, ClausesADCorrect x body []
  clauses_cons_law :
    ∀ x body op y k hb rest,
      TermADCorrect x hb →
      ClausesADCorrect x body rest →
      ClausesADCorrect x body ((op, y, k, hb) :: rest)

  grad_law :
    ∀ {Delta Sigma Gamma x ds dsOut body eps slot},
      HasType (Capability.diff :: Delta) Sigma
        (Gamma ++ [(x, some (Typ.tensor ds))])
        body (Typ.tensor dsOut) eps
        (Gamma ++ [(x, slot)]) →
      subsetEffRow eps DiffCompat = true →
      AdjointSupported body →
      AdjointFreeCtxSupported
        (Gamma ++ [(x, some (Typ.tensor ds))]) body →
      TermADCorrect x body →
      GradADCorrect Delta Sigma Gamma x ds dsOut body eps
        (fun gSeed => adjointTyped body (Typ.tensor dsOut) x gSeed)

mutual

/-- Supported terms assemble semantic AD correctness from the model's
    primitive and structural laws. -/
theorem termADCorrect_of_supported
    (M : PrimitiveADSpec) (x : String) :
    ∀ e, AdjointSupported e → M.TermADCorrect x e
  | Term.var y, _ =>
      M.var_law x y
  | Term.const v ds, _ =>
      M.const_law x v ds
  | Term.unit, _ =>
      M.unit_law x
  | Term.loc ell, _ =>
      M.loc_law x ell
  | Term.add e1 e2, h => by
      exact M.add_law x e1 e2
        (termADCorrect_of_supported M x e1 h.1)
        (termADCorrect_of_supported M x e2 h.2)
  | Term.mul e1 e2, h => by
      exact M.mul_law x e1 e2
        (termADCorrect_of_supported M x e1 h.1)
        (termADCorrect_of_supported M x e2 h.2)
  | Term.sum e d, h => by
      exact M.sum_law x e d
        (termADCorrect_of_supported M x e h)
  | Term.expand e d, h => by
      exact M.expand_law x e d
        (termADCorrect_of_supported M x e h)
  | Term.copy e, h => by
      exact M.copy_law x e
        (termADCorrect_of_supported M x e h)
  | Term.letBind y e1 e2, h => by
      exact M.let_law x y e1 e2
        (termADCorrect_of_supported M x e1 h.1)
        (termADCorrect_of_supported M x e2 h.2)
  | Term.letpair y z e1 e2, h => by
      exact M.letpair_law x y z e1 e2
        (termADCorrect_of_supported M x e1 h.1)
        (termADCorrect_of_supported M x e2 h.2)
  | Term.pair e1 e2, h => by
      exact M.pair_law x e1 e2
        (termADCorrect_of_supported M x e1 h.1)
        (termADCorrect_of_supported M x e2 h.2)
  | Term.fst tRight e, h => by
      exact M.fst_law x tRight e
        (termADCorrect_of_supported M x e h)
  | Term.snd tLeft e, h => by
      exact M.snd_law x tLeft e
        (termADCorrect_of_supported M x e h)
  | Term.perform op e, h => by
      exact M.perform_diffcompat_law x op e h.1
        (termADCorrect_of_supported M x e h.2)
  | Term.handle epsH body clauses, h => by
      exact M.handle_law x epsH body clauses
        (termADCorrect_of_supported M x body h.1)
        (clausesADCorrect_of_supported M x body clauses h.2)
  | Term.abs _ _ _, h => by
      cases h
  | Term.app _ _, h => by
      cases h
  | Term.uniformLike _ _ _, h => by
      cases h
  | Term.grad _ _ _ _, h => by
      cases h
  | Term.vmap _ _ _ _, h => by
      cases h

/-- Clause-list companion to `termADCorrect_of_supported`. -/
theorem clausesADCorrect_of_supported
    (M : PrimitiveADSpec) (x : String) (body : Term) :
    ∀ clauses,
      AdjointSupportedClauses clauses →
      M.ClausesADCorrect x body clauses
  | [], _ =>
      M.clauses_nil_law x body
  | (op, y, k, hb) :: rest, h => by
      exact M.clauses_cons_law x body op y k hb rest
        (termADCorrect_of_supported M x hb h.1)
        (clausesADCorrect_of_supported M x body rest h.2)

end

/-- AD correctness for the current typed first-order `grad` fragment.

The theorem is intentionally parameterized by `PrimitiveADSpec`: this
file proves that a typed `grad` body lying in `AdjointSupported` is
covered by the model's primitive/structural laws and therefore the
generated `adjointTyped` builder satisfies the model's public gradient
correctness relation. -/
theorem ad_correctness
    (M : PrimitiveADSpec)
    {Delta : CapCtx} {Sigma : StoreTyp} {Gamma : LinearCtx}
    {x : String} {ds dsOut : DimList} {body : Term}
    {eps : EffectRow} {slot : Option Typ}
    (hBody : HasType (Capability.diff :: Delta) Sigma
      (Gamma ++ [(x, some (Typ.tensor ds))])
      body (Typ.tensor dsOut) eps
      (Gamma ++ [(x, slot)]))
    (hEff : subsetEffRow eps DiffCompat = true)
    (hSupp : AdjointSupported body)
    (hFree : AdjointFreeCtxSupported
      (Gamma ++ [(x, some (Typ.tensor ds))]) body) :
    M.GradADCorrect Delta Sigma Gamma x ds dsOut body eps
      (fun gSeed => adjointTyped body (Typ.tensor dsOut) x gSeed) := by
  exact M.grad_law hBody hEff hSupp hFree
    (termADCorrect_of_supported M x body hSupp)

end LaCaDiLE
