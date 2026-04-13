# WS1: Core Calculus on Paper

- **Size:** Medium
- **Dependencies:** None. Can start immediately.
- **Output:** Standalone LaTeX document containing the complete LaCaDiLE definition.
- **See also:** [decisions.md](../decisions.md), [risks.md](../risks.md), [ws2-paper-proofs.md](ws2-paper-proofs.md).

---

Write LaCaDiLE as a standalone formal definition. This is the foundation everything else builds on.

## WS1.1 — Syntax definition

All decisions in [decisions.md](../decisions.md) are locked. The syntax is:

Types:

```
τ ::= tensor[d̄]           -- tensor with named dimension list (precision abstracted away)
    | τ₁ → τ₂ ! ε          -- function with effect row
    | τ₁ ⊗ τ₂              -- linear pair
    | unit
    | α                     -- type variable
```

Dimensions:

```
d ::= n                    -- named dimension
    | δ                    -- dimension variable
    | k                    -- literal size
d̄ ::= ε | d, d̄
```

Effects:

```
ε ::= ∅ | Random, ε | Resource, ε | IO, ε | Fail, ε | Accum, ε | ρ
```

Capabilities:

```
κ ::= Diff
```

Terms:

```
e ::= x | λx:τ. e | e₁ e₂ | let x = e₁ in e₂
    | op(e₁, ..., eₙ)      -- RISC primitives (const, add, mul, sum, expand, uniform_like)
    | grad(e)               -- AD transform (introduces + handles Accum internally)
    | vmap(e)               -- vectorization transform
    | handle[ε] e with h   -- effect handler (one-shot continuations)
    | perform op(e)         -- effect operation (including Fail)
    | copy(e) | &e          -- linearity operations
    | (e₁, e₂) | fst(e) | snd(e)  -- pairs
```

## WS1.2 — Typing rules

Every rule in inference-rule notation. The critical rules, with locked designs:

*Linear context threading:* Input/output contexts `Δ; Γ ⊢ e : τ ! ε ⊣ Γ'` where `Δ` is the capability context and `Γ` is the linear type context.

*The `handle` rule with continuation as linear binding:*

```
Δ; Γ₁ ⊢ body : τ ! {ε_handled, ε_rest} ⊣ Γ₂
For each operation op of ε_handled:
  Δ; Γ₂, x:τ_op, k:(τ_ret → τ ! ε_rest) ⊢ h_op : τ ! ε_rest ⊣ Γ₃
  (k is linear — calling it consumes it, enforcing one-shot)
──────────────────────────────────────────────────────
Δ; Γ₁ ⊢ handle body with {op(x, k) → h_op, ...} : τ ! ε_rest ⊣ Γ₃
```

*The `grad` rule with Accum:*

```
Δ, Diff; Γ ⊢ f : (tensor[d̄] → tensor[d̄'] ! ε)
f uses its argument linearly
ε ⊆ DiffCompat                    -- differentiation-compatible effects
──────────────────────────────────────────────────────
Δ; Γ ⊢ grad(f) : (tensor[d̄] → tensor[d̄] ! ε')
```

Where `DiffCompat = {Resource, Accum}` — effects that don't affect the mathematical computation. `Resource` is permitted because device allocation doesn't change the function being differentiated. `Random`, `IO`, and `Fail` are excluded because stochasticity, side effects, and partiality are incompatible with deterministic gradient computation. `grad` internally introduces and handles `Accum` for gradient accumulation — the reduced term wraps the adjoint body in an explicit `Accum` handler (see WS1.3). The return type has the same dimension list as the input (gradient shape = parameter shape). `ε'` is the residual effect row after `Accum` is handled.

*The `vmap` rule with `addDim`:*

```
Δ; Γ ⊢ f : τ₁ → τ₂ ! ε
d fresh
─────────────────────────────────
Δ; Γ ⊢ vmap(f) : addDim(d, τ₁) → addDim(d, τ₂) ! ε
```

*Effect row operations:* Union (sequencing), subtraction (handling), row variable instantiation (polymorphism).

## WS1.3 — Operational semantics

Small-step reduction with heap store.

- Call-by-value.
- Configuration: `⟨σ, e⟩` where `σ : Loc → TensorVal` is the store.
- `const(v, shape)` allocates: `⟨σ, const(v, shape)⟩ ↦ ⟨σ[ℓ ↦ tensor_val(d̄, data)], ℓ⟩` for fresh `ℓ`.
- `uniform_like(ℓ, lo, hi)` allocates a new location with random data (under `Random` effect).
- Linear consumption deallocates: when a RISC primitive consumes a tensor at location `ℓ`, `σ` is updated to remove `ℓ`.
- Borrowing reads `σ(ℓ)` without removing.
- `copy(ℓ)` allocates a fresh location `ℓ'` with the same data: `⟨σ, copy(ℓ)⟩ ↦ ⟨σ[ℓ' ↦ σ(ℓ)], (ℓ, ℓ')⟩`. This is physical copy — no aliasing, every location has exactly one owner. The store invariant remains simple: each live location has exactly one owner in `Γ`. (The implementation uses `Arc` / reference counting as an optimization; this is semantically equivalent for immutable tensor data and is noted in §6.)
- `grad(λx.e)` reduces to a term with an explicit `Accum` handler wrapping the adjoint body:

  ```
  grad(λx.e)  ↦  λx. handle[Accum] adjoint(e, x) with { accum(loc, val, k) → k(update_grad(loc, val)) }
  ```

  The handler is a standard effect handler — not special-cased. Preservation for the `grad` reduction case follows from the standard `handle` typing rule applied to the adjoint-transformed body. This reuses existing machinery rather than requiring a bespoke reduction rule.
- **The tape mechanism for adjoint typing:** The adjoint of `mul(a, b)` produces `(g*b, g*a)` — referencing `b` in `a`'s adjoint and vice versa. But `a` and `b` were consumed by `mul` in the forward pass. The adjoint transformation must produce a *tape* — a set of read-only borrows saved during the forward pass. The structure: (1) the forward pass evaluates normally but borrows all intermediate values into a tape before consuming them, (2) the backward pass uses the taped borrows plus the output gradient. The tape values are borrows (`&`), not linear bindings — they don't affect the linear context. This is how real AD implementations work, and the formalization must reflect it. Without the tape, `adjoint(mul(a, b))` references consumed bindings and the adjoint typing lemma fails. **This is the single most technically subtle point in the calculus — get it right before anything else in WS1 proceeds.**
- Effect handling: `⟨σ, handle[ε] (perform op(v)) with h⟩ ↦ ⟨σ, h_op[v/x, k/k]⟩` where `k` is the captured one-shot continuation.

## WS1.4 — Review and iterate

Specifically test:

- Can a multi-use of a linear tensor sneak through the handler rule? (Should be impossible — `k` is linear, invoking it twice is a type error.)
- Does `vmap(grad(f))` compose correctly? (The batch dimension from `vmap` should thread through the adjoint transformation without interference.)
- Can effect row polymorphism introduce an unsound instantiation where `Diff` capability is assumed but not satisfied? (The capability context `Δ` is separate from the effect row `ε`, so row variable instantiation can't affect `Δ`.)
- Does the `Accum` effect correctly mediate between linearity and gradient accumulation? (The key case: `grad(λx. add(x, x))` should be ill-typed because `x` is used non-linearly. But `grad(λx. let y = copy(x) in add(x, y))` is well-typed — the explicit copy produces two linear bindings, and the adjoint accumulates both gradients via `Accum`.)
