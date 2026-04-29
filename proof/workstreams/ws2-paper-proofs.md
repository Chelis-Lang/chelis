# WS2: Metatheory (Paper Proofs)

- **Size:** Large. WS2.1 (substitution), WS2.2 (adjoint typing), WS2.5 (preservation), and WS2.9 (AD correctness) are the bulk.
- **Dependencies:** [WS1](ws1-core-calculus.md) (stable rules).
- **Output:** Full proofs as supplementary appendix (PDF). Proof sketches for §5 of the paper body.
- **See also:** [decisions.md](../decisions.md), [risks.md](../risks.md), [ws3-lean-mechanization.md](ws3-lean-mechanization.md).

---

Prove all five theorems on paper. Proof sketches for the paper body, full proofs for the supplementary appendix. Paper proofs first, then encode in Lean — you want to know the proof works before committing to mechanization.

## WS2.1 — Substitution lemma

Standard but requires care with linear contexts and the capability context. If `Δ; Γ, x:τ₁ ⊢ e : τ₂ ! ε ⊣ Γ'` and `Δ; Γ₀ ⊢ v : τ₁ ⊣ Γ₀'`, then `Δ; Γ₀ + (Γ \ x) ⊢ e[v/x] : τ₂ ! ε ⊣ ...`. Define context splitting (`Γ = Γ₁ + Γ₂`) and prove it's well-behaved. The capability context `Δ` passes through unchanged (it's not linear).

## WS2.2 — Adjoint typing lemma

The linchpin of the entire metatheory — prove as a standalone lemma before anything else in WS2. If `Δ, Diff; Γ ⊢ e : τ ! ε ⊣ Γ'` and `e` uses its free variables linearly, then `Δ; Γ_adj ⊢ adjoint(e) : τ_adj ! ε, Accum ⊣ Γ_adj'`. Requires defining the adjoint transformation formally for each of the six primitives and showing each case preserves typing. The current Lean branch has already ruled out more than one over-strong theorem shape here: the old seed-polymorphic helper for `expand` is false without an explicit source-typing/dimension premise tying the seed dimensions to the expanded axis; the transform has now been upgraded to typed cotangent seeds for products/projections; the legacy public `adjointFrom` surface is still false on handled product seeds; unrestricted `adjointTypedFrom` is also false on raw higher-order terms; the current private typed helper is false on arbitrary seed-only contexts because `mul` replays raw source operands; and now the current public `adjointTypedFrom_preserves_typing` statement is also false on the real `grad`-style `mul` output shape because `copy x` tombstones `x` before the claimed output. The branch now carries an explicit first-order supported fragment plus substitution/value closure lemmas for it, and the public typed theorem surface is already stated on `adjointTypedFrom`. So the remaining proof work is no longer “invent the right product routing” or even only “restate the helper”; it is to redesign the `mul` replay/tape boundary and public output-context claim so the theorem surface is honest, then restate the admitted private helper on that source-typed domain, then close the remaining `mul` and clause-list `cons` cases.

The critical subtlety: the tape. The adjoint of `mul(a, b)` produces `(g*b, g*a)`, which references `b` in `a`'s adjoint and `a` in `b`'s adjoint. But `a` and `b` were consumed by `mul` in the forward pass and are no longer in the linear context. The adjoint transformation must be defined as: (1) evaluate the forward pass, borrowing all intermediate values into a tape before consuming them, (2) evaluate the backward pass using tape borrows + output gradient. Formally, the adjoint transformation produces a term where every consumed intermediate has a corresponding borrow in scope. The tape borrows are `&` references — they don't consume from `Γ`, so the linear context accounting works.

The Dex comparison is directly relevant here: Dex doesn't need a tape mechanism because it has no linearity — values can be freely referenced in both forward and backward passes. LaCaDiLE's linearity forces the tape to be explicit, which is actually an advantage (makes AD memory behavior visible and typed — this is something practitioners care about, since PyTorch's autograd tape is a major source of memory leaks) but it's harder to formalize.

**If this lemma fails, the calculus needs revision before anything else proceeds.** The tape/borrow interaction with the linear context is where unsoundness is most likely to hide.

## WS2.3 — `addDim` preserves typing

If `Δ; Γ ⊢ e : τ ! ε ⊣ Γ'`, then `Δ; addDimCtx(d, Γ) ⊢ addDimTerm(d, e) : addDim(d, τ) ! ε ⊣ addDimCtx(d, Γ')`. Straightforward by structural induction. Required for the `vmap` case of preservation.

## WS2.4 — Progress (Theorem 1a)

For each typing rule, show a well-typed non-value can step. Interesting cases:

- `grad(v)` where `v` is a function value: must show the adjoint transformation is defined.
- `handle` where the body performs an effect: must show the handler has a matching clause.
- RISC primitives at tensor values: must show dimension compatibility guarantees the operation is defined (this is where dimension safety connects to progress).
- `perform op(v)` where no handler is in scope: this is the case that Theorem 3 (effect correctness) eliminates — if the effect row is empty, no `perform` can occur.

## WS2.5 — Preservation (Theorem 1b)

For each reduction rule, show the result is well-typed. The current mechanized statement is the honest one:

- if `Δ; ∅ ⊢ e : τ ! ε` and the runtime configuration is `RuntimeLinear`, then a reduction step preserves typing

This extra premise is not cosmetic. The Lean development now contains a concrete closed, well-typed counterexample showing that unconditional preservation is false for runtime terms with duplicated explicit locations in sibling subterms. The paper proof therefore has to split the argument into two pieces:

1. preservation under the stronger cross-boundary handler-aware runtime boundary now named `HandlerAwareRuntimeLinear` in the Lean tree
2. a separate invariant showing evaluation from checked source programs stays inside that invariant

The hard cases:

- `grad` reduction: the reduced term is `λx. handle[Accum] adjoint(e, x) with {...}`. The old concrete `letBind` / `letpair` routing bug is now repaired in Lean, and the transform now routes typed cotangent seeds through `pair` / `fst` / `snd` / `copy`. But the live typed theorem surface is still not honest yet: legacy `adjointFrom` is false for handled product seeds, unrestricted `adjointTypedFrom` is false on higher-order bodies, the current private typed helper is too weak for `mul` because it forgets the source context needed to type replayed operands, and the current public `adjointTypedFrom_preserves_typing` output claim is itself false on a real `mul`-based `grad` input because `copy x` tombstones `x`. So the paper proof still has to redesign the `mul` replay/tape boundary and public output-context claim first, then restate the helper on an explicit supported, source-typed domain, then finish the separate `mul` and clause-list threading cases before rerunning the standard “adjoint body is typed with `Accum`, handler removes `Accum`” argument.
- `vmap` reduction: typing and stepping now share one explicit batch-dimension choice through the `vmap` term itself, and that preservation case is now closed on the executable Lean branch.
- Effect handling with one-shot continuation: the continuation captures linear context. One-shot consumption (continuation is linear) preserves linearity. Uses the substitution lemma (WS2.1).
- Store operations: allocation extends `σ` consistently; deallocation removes entries consumed linearly. `copy` allocates a fresh location with duplicated data. The store typing relation must be maintained.
- Congruence (`ctx`): the proof needs frame-local store agreement, not global store monotonicity. The preserved invariant is agreement only on locations still mentioned by untouched sibling subterms.

## WS2.6 — Dimension safety (Theorem 2)

Corollary of preservation. Key lemma: dimensions are preserved or transformed only as specified by typing rules. `sum` removes a dimension, `expand` adds one, elementwise operations preserve — and each of these is reflected in the type. No runtime dimension mismatch can occur because the operational semantics for RISC primitives is defined only on dimension-compatible inputs, and progress guarantees that well-typed terms have compatible inputs.

## WS2.7 — Effect correctness (Theorem 3)

If `Δ; ⊢ e : τ ! ∅` then evaluation never encounters an unhandled `perform`. Key lemma: effect rows only shrink under reduction (handlers remove effects), never grow. Only `perform` introduces effects, and its typing rule requires the effect to be in the row. Empty effect row + no `perform` in typing = no unhandled effects at runtime.

## WS2.8 — Linearity soundness (Theorem 4)

Using the heap store semantics. If a binding at location `ℓ` is consumed (not borrowed, not copied), then `σ(ℓ)` is accessed exactly once and `ℓ` is deallocated. The theorem: no well-typed program accesses a deallocated location. Proof: maintain an invariant that the set of live locations in `σ` corresponds exactly to the live bindings in `Γ`. Linear consumption removes from both `Γ` and `σ`. Borrowing reads without removing from either. The invariant is preserved by each reduction step (by case analysis on the reduction rules).

This workstream now also needs the runtime-location side invariant that closes the preservation loop. The earlier candidate theorem

- if a checked runtime configuration is `RuntimeLinear`, one reduction step preserves `RuntimeLinear`

is false, and the naive handler-specific repair is not strong enough either. The branch now names the stronger cross-boundary predicate `HandlerAwareRuntimeLinear`, and `LinearitySoundness.lean` proves a partial one-step boundary that either preserves it or exposes explicit debt. A newer config-level `RuntimeSafeConfig` wrapper is also not yet final: there is now a concrete `handleOpCtx` counterexample against generic closure for that surface. What remains is not inventing the invariant from scratch, but closing or refining the explicit debt cases and deciding which extra typing and store-freshness side conditions belong in the paper theorem. The blocking raw-runtime witnesses are:

- beta substitution can duplicate a bound location that was invisible before substitution
- direct-handler substitution needs the one-shot / linear-argument typing discipline
- context steps can allocate a fresh live location that collides with a stale sibling mention outside the redex

## WS2.9 — AD correctness (Theorem 5)

If `Δ, Diff; ⊢ grad(f) : (tensor[d̄] → tensor[d̄] ! ε)` and `f` computes a mathematically differentiable function, then `grad(f)(x)` computes the correct mathematical gradient. Requires formalizing each RISC primitive as a mathematical function and showing adjoint rules produce correct partial derivatives. The Accum mechanism must correctly sum contributions from multiple uses (via explicit `copy`). This is the hardest theorem — requires a denotational semantics layer mapping tensor operations to real-valued functions and connecting the syntactic adjoint transformation to the mathematical derivative.
