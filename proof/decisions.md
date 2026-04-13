# Locked Design Decisions

This file is the canonical "why is it this way?" reference for Chelis Proof. Every workstream cites it.

## Prose: Locked Decisions

**Full Lean mechanization (Option A).** The paper claims "first mechanized soundness proof for a tensor calculus with effects, linearity, and dimension indexing." All five theorems proved in Lean 4. Proof scripts ship as anonymized supplementary material with the submission. This is the strongest paper — Option C ("we mechanized some but not all") invites "why didn't you finish?" and Option B (paper proofs only) is weaker in a venue that increasingly expects mechanization.

**Diff as capability context.** A separate capability context `Δ` threads alongside the type context `Γ` and effect row `ε`. Typing judgment: `Δ; Γ ⊢ e : τ ! ε ⊣ Γ'`. The structural predicate alternative (`diff(e)` holds iff all primitives are differentiable) breaks at abstraction boundaries — can't check differentiability of an opaque function reference without inlining. The capability context is compositional, which is what POPL reviewers care about. It also mirrors differential linear logic: in DiLL the exponential `!A` mediates between linear and unrestricted use and the differential operator `∂` is typed relative to the linear/exponential structure. Having `Diff` as a first-class tracked capability is the same modal distinction.

**Fail effect, not Option.** `Fail` replaces `Option[τ]`. Removes `Some`, `None`, `match` from the calculus — three fewer term formers, fewer cases in every proof. `perform Fail(msg)` is handled like any other effect. Demonstrates that the effect system is general enough to encode sum-type control flow. Smaller, cleaner calculus.

**Drop precision from the core calculus.** Precision equality is syntactic equality — same proof structure as dimensions but metatheoretically uninteresting. Zero theorems mention it. Mention in §6 (Implementation) that the compiler tracks precision; the core calculus abstracts it away.

**One-shot handlers with continuation as linear binding.** Multi-shot breaks linearity without a much more complex type system. One-shot matches Chelis's implementation. The key design insight: enforce the one-shot restriction by making the continuation `k` itself a linear binding. Calling `k` consumes it. The one-shot restriction falls out of the linearity system rather than being a separate side condition. This is an elegant interaction where effects and linearity reinforce each other — highlight in the paper.

**Accum as internal effect for gradient accumulation, with explicit handler in reduced term.** `grad(f)` transforms `f`'s body by introducing `Accum` effect annotations where gradient contributions accumulate. The reduced term of `grad(λx.e)` is `λx. handle[Accum] adjoint(e, x) with { accum(loc, val, k) → k(update_grad(loc, val)) }` — the `Accum` handler is explicit in the reduced term, not magic. This is a standard effect handler, so the preservation proof for `grad` reuses existing `handle` machinery rather than requiring a bespoke reduction rule. This resolves the linearity tension in AD: the adjoint of `add(a, b)` is `(g, g)` — a copy of `g` — which violates linearity. But the duplication is typed as `Accum` accumulation, a controlled form of non-linear use. Linearity, effects, and AD all interact through `Accum`. This is precisely the Dex insight, but combined with linearity — Dex uses `Accum` without linearity, so it can't guarantee buffer safety. Spell this out in the paper.

**Heap-based store semantics for Theorem 4, with physical copy.** Extend the operational semantics with `σ : Loc → TensorVal`. Tensor creation (`const`, `uniform_like`) allocates a fresh location. Linear consumption deallocates. `copy(ℓ)` allocates a fresh location `ℓ'` with a byte-identical copy of the data at `ℓ`, and returns the **linear pair** `(ℓ, ℓ')` — the original `ℓ` is retained and re-bound by the destructuring `let (x, x') = copy(e) in ...` in the caller, while `ℓ'` owns the new location. This is the canonical interpretation pinned down in Phase 1 T0 §1.0: `copy : τ → τ ⊗ τ`. Every live location still has exactly one owner. This keeps the store invariant simple (no reference counting, no aliasing) without forcing the caller to lose the original value. The implementation uses `Arc` (reference counting) as an optimization, which is semantically equivalent for immutable tensor data — noted in §6. Theorem 4: no well-typed program accesses a deallocated location.

**`Resource` simplified to unparameterized effect.** Drop `Resource(Device)` parameterization. `Resource` as a single effect demonstrates that the type system tracks device allocation. Parameterized effects require subeffecting or effect row matching on the parameter, adding complexity to the effect row algebra for no additional metatheoretic insight. Every theorem says the same thing whether `Resource` is parameterized or not. Mention parameterization in §6 as an implementation extension.

**Explicit tape for AD forward/backward pass.** The adjoint transformation produces a term with two phases: (1) forward pass evaluates normally but borrows (`&`) all intermediate values into a tape before consuming them, (2) backward pass uses taped borrows plus the output gradient. This resolves the linearity tension in multi-input primitives: `mul(a, b)`'s adjoint needs `b` for `a`'s gradient and `a` for `b`'s gradient, but `a` and `b` were consumed by `mul`. The tape borrows keep them in scope as read-only references without violating linear consumption. This is the single most technically subtle point in the calculus — the adjoint typing lemma (WS2.2) depends on this mechanism being correct. The Dex comparison: Dex doesn't need a tape mechanism because it has no linearity. LaCaDiLE's linearity forces the tape to be explicit, making AD memory behavior visible and typed — a practical advantage since PyTorch's implicit autograd tape is a major source of memory leaks.

**`addDim` meta-function for vmap.** Defined recursively on type structure:

```
addDim(d, tensor[d̄]) = tensor[d, d̄]
addDim(d, τ₁ → τ₂ ! ε) = addDim(d, τ₁) → addDim(d, τ₂) ! ε
addDim(d, τ₁ ⊗ τ₂) = addDim(d, τ₁) ⊗ addDim(d, τ₂)
addDim(d, unit) = unit
```

The `vmap` typing rule becomes: if `Γ ⊢ f : τ₁ → τ₂ ! ε` and `d` is fresh, then `Γ ⊢ vmap(f) : addDim(d, τ₁) → addDim(d, τ₂) ! ε`. Preservation requires a lemma that `addDim` preserves typing — straightforward by structural induction.

**Six RISC primitives in the core calculus:**

| Primitive | Role in the formalization |
|---|---|
| `const(v, shape)` | Tensor introduction. Without it, no way to create a tensor literal. |
| `add(a, b)` | Elementwise binary. Dimension equality requirement. Adjoint is `(g, g)` — triggers the Accum interaction. |
| `mul(a, b)` | Elementwise binary. Adjoint is `(g*b, g*a)` — requires the other operand, tests multi-input adjoint. |
| `sum(x, axis)` | Reduction. Dimension removal. Adjoint is `expand` — so `expand` must be in the calculus for `grad(sum)` to close. |
| `expand(x, axis, k)` | Dimension addition (inserts a new dimension of extent `k` at position `axis`). Adjoint of `sum`. Required for the metatheory to close under `grad` reduction. Replaces `reshape` (which doesn't interact with effects, linearity, or Diff). |
| `uniform_like(x, lo, hi)` | The `Random` effect primitive. Required to demonstrate effect tracking. |

Every theorem has something to say about this set. `add`/`mul` exercise dimension equality + AD. `sum`/`expand` exercise dimension manipulation + AD closure. `uniform_like` exercises effects. `const` is structural necessity.

**Ehrhard & Regnier / differential linear logic as the intellectual spine.** This goes in the introduction, not just related work. The paper's framing: DiLL establishes that differentiation and linearity are fundamentally connected (the differential operator is the dual of the exponential modality). LaCaDiLE is the first computational type system that realizes this connection for tensor programs, with algebraic effects providing the additional structure for stochasticity and device management. This reframes the paper from "we made a type system for our language" to "we realized a well-studied theoretical principle as a practical type system for tensor computation." The former is engineering. The latter is science. The specific connection: in DiLL, the typing rule for the differential combinator requires its argument to be in the linear zone (not under `!`). In LaCaDiLE, `grad` requires its argument to use the differentiated parameter linearly. Same constraint, different formalisms. Side-by-side comparison of the typing rules in the paper.

## Summary Table

| Decision | Choice | Rationale |
|---|---|---|
| Mechanization scope | Full (all 5 theorems in Lean) | Strongest contribution claim |
| `Diff` mechanism | Capability context `Δ` | Compositional across abstraction boundaries; mirrors DiLL |
| Fallibility | `Fail` effect | Smaller calculus, demonstrates effect system generality |
| Precision | Dropped from core calculus | Zero metatheoretic content |
| `Resource` | Unparameterized (no `Device` argument) | No metatheoretic payoff; parameterization noted in §6 |
| Handler continuations | One-shot, `k` is linear binding | Linearity enforces one-shot; elegant interaction |
| `grad` and `vmap` scope | Both restricted to literal abstractions `λx:τ.e` in Phase 1 | Enables syntactic pattern matching in the operational rules and enforces `grad`'s linear-use side condition via `T-Var` failure in the premise. Eta-expand variable function references. |
| Gradient accumulation | `Accum` effect, explicit handler in reduced term | Reuses standard `handle` machinery for preservation proof |
| AD tape | Explicit borrows saved during forward pass | Resolves linearity/adjoint tension for multi-input primitives |
| `copy` semantics | Physical copy (fresh location) | Simple store invariant (one owner per location); `Arc` is §6 optimization |
| Linearity semantics | Heap store `σ : Loc → TensorVal` | Standard, Lean-friendly |
| `vmap` typing | `addDim` meta-function on types | Avoids variable capture; structural induction |
| `grad` effect constraint | `DiffCompat = {Resource, Accum}` | Resource is compatible; Random/IO/Fail are not |
| RISC primitives | `const`, `add`, `mul`, `sum`, `expand`, `uniform_like` | Minimal set where every theorem has something to say |
| Intellectual framing | Ehrhard & Regnier / DiLL | "Realized a theoretical principle" not "built a type system" |
