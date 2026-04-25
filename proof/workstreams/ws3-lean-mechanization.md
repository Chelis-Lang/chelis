# WS3: Lean 4 Formalization

- **Size:** Large. 3000–5000 lines of Lean 4. The substitution lemma (WS3.6), preservation (WS3.10), and AD correctness (WS3.14) dominate.
- **Dependencies:** [WS1](ws1-core-calculus.md) (stable rules), [WS2](ws2-paper-proofs.md) (proof strategy). WS2 and WS3 proceed in parallel per-theorem: paper proof first, then encode.
- **Output:** Anonymized `chelis-metatheory/` tarball. README documenting file structure, theorems proved, admitted lemmas, build instructions, Lean version, non-standard axioms (per CFP requirement). Lean tree lives under [`../lean/`](../lean/).
- **See also:** [decisions.md](../decisions.md), [risks.md](../risks.md).

---

Mechanize all five theorems. Full mechanization is the main contribution.

## WS3.1 — `Syntax.lean`

Inductive types for: types (with tensor, function, linear pair, unit), dimensions, dimension lists, effect rows, capability sets, terms, values, evaluation contexts, store.

## WS3.2 — `Typing.lean`

Typing judgment `Δ; Γ ⊢ e : τ ! ε ⊣ Γ'` as an inductive relation. Each typing rule is a constructor. The capability context `Δ` threads through all rules. This is where errors in the paper rules surface — Lean forces totality.

## WS3.3 — `Store.lean`

Store typing relation. Location allocation, deallocation, and access. The heap `σ : Loc → TensorVal` as a Lean `Map`. Store well-formedness invariant linking `σ` to the linear context `Γ`.

## WS3.4 — `Operational.lean`

Small-step semantics `⟨σ, e⟩ ↦ ⟨σ', e'⟩` as an inductive relation. Includes: RISC primitive evaluation (with store effects), effect handling, `grad` as adjoint transformation, `vmap` as dimension-adding transformation.

## WS3.5 — `AdjointTransform.lean`

The adjoint transformation as a recursive function on terms. Defined per-primitive. Must be a total function for Lean to accept it. Produces a term that: (a) borrows all intermediate values into a tape during the forward pass, (b) wraps the backward pass in an explicit `Accum` handler, (c) references taped borrows in adjoint computations. Separate file because it's a complex recursive definition.

## WS3.6 — `Substitution.lean`

The substitution lemma with linear context splitting. Always the most painful part. The capability context `Δ` is non-linear (passes through unchanged), which helps — only `Γ` requires splitting.

## WS3.7 — `AdjointTyping.lean`

The adjoint typing lemma (WS2.2) mechanized. The hardest standalone lemma in the formalization. Must show: if `e` is well-typed with `Diff` and linear use, then `adjoint(e)` is well-typed with `Accum` in the effect row. Case analysis on the six primitives. The `mul` case is the crux — must show that taped borrows from the forward pass are in scope for the backward pass and that the borrow/consume interaction with the linear context is sound. Depends on WS3.5 (adjoint transformation) and WS3.2 (typing relation).

## WS3.8 — `AddDim.lean`

The `addDim` meta-function and the lemma that it preserves typing. Structural induction on types. Required for the `vmap` case of preservation.

## WS3.9 — `Progress.lean`

Case analysis on the typing derivation. Depends on WS3.1–WS3.4.

## WS3.10 — `Preservation.lean`

Case analysis on the reduction relation + typing derivation. Depends on WS3.6 (substitution), WS3.7 (adjoint typing), WS3.8 (addDim). The mechanized theorem has now been strengthened to the honest runtime statement:

- if `HasType [] Sigma [] e t eps []`
- and `e` is `WellScoped`
- and `e` is `RuntimeLinear`
- and `⟨sigma, e⟩ ↦ ⟨sigma', e'⟩`

then typing is preserved for some post-step store typing `Sigma'`.

This extra `RuntimeLinear` premise is necessary: the Lean tree contains a concrete closed, well-typed counterexample showing that unconditional runtime preservation is false once explicit store locations can be duplicated in sibling subterms. The `ctx` case was closed by weakening the store-side transport to frame-local agreement on the untouched context locations rather than global store monotonicity.

That premise is still not the final answer. The current branch now also shows:

- `ActiveRuntimeLinear` repairs the captured-handler context counterexamples
- but `ActiveRuntimeLinear` is still too weak for direct handled operations, because a dormant clause body can become active in one step and expose duplicated locations
- and the recursive `DeepActiveRuntimeLinear` repair is still not compositional: a typed beta step can expose a dormant clause beside an active sibling with the same location, and one contextual `handleOpDirect` step later the invariant fails

So the immediate remaining preservation work is:

- settle the stronger cross-boundary handler-aware runtime invariant
- `tgrad`, still upstream of `AdjointTyping.lean`
- `tvmap` is now closed on the explicit batch-dimension surface

## WS3.11 — `DimSafety.lean`

Corollary of preservation + dimension-specific lemmas. Smaller now that the `tvmap` rule surface and preservation case are both settled, but it still depends on re-establishing the stronger preservation premise across steps so preservation can be iterated over multi-step execution.

## WS3.12 — `EffectCorrectness.lean`

Corollary of preservation + effect monotonicity lemma. Relatively small.

## WS3.13 — `LinearitySoundness.lean`

Store invariant maintenance proof. Case analysis on reduction rules showing the live-location invariant is preserved. Depends on WS3.3 (store model).

This workstream now also carries the runtime-side invariant needed to make preservation compositional. The earlier “preserve `RuntimeLinear`” theorem shape is now known to be false, and the first “preserve `ActiveRuntimeLinear`” repair is also insufficient. The recursive `DeepActiveRuntimeLinear` repair is still too weak because it does not constrain dormant clause bodies relative to the surrounding active frame. The actual remaining task is to define and prove the stronger handler-aware invariant that:

- ignores dormant clause bodies for captured-continuation duplication, but
- still constrains dormant clause bodies strongly enough for `handleOpDirect`
- and preserves the necessary separation between dormant clause bodies and surrounding active siblings across intermediate steps such as beta

## WS3.14 — `ADCorrectness.lean`

The hardest file. Requires a denotational semantics mapping terms to mathematical functions, plus a proof that the adjoint transformation computes the derivative. If this proves intractable in Lean, fall back to stating the theorem and providing a paper proof in the appendix — but attempt the full mechanization first.
