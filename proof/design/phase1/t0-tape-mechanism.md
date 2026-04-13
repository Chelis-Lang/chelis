# T0 — Tape Mechanism for LaCaDiLE Adjoint Transformation

**Phase 1, Task T0.** Highest-risk design task in the project. Settles how the adjoint
transformation in `grad` interacts with linear contexts so the adjoint typing lemma
(WS2.2) closes. Nothing downstream of T0 proceeds until this document is reviewed by
red-team round 1.

**Status:** Draft for Wave 1 red team. Not yet rule-encoded in LaTeX (T2) or Lean (T7).

---

## 0. Non-goals and scope

The T0 design covers **first-order** `grad`: `grad(f)` where `f`'s body is composed
of RISC primitives, `let` bindings, `copy`, `handle[ε]` with `ε ⊆ DiffCompat`, and
`perform` calls on effects in `DiffCompat`. `f`'s body may be wrapped *from outside*
in additional effect handlers (e.g. `handle[Random] ... grad(f) ...`), but the body
itself must type-check under the `DiffCompat` restriction.

> **Nested `grad` applied to another `grad` (e.g. `grad(grad(f))`) is out of scope for
> Phase 1.** It requires the adjoint transformation to be recursive through itself — a
> separate design task. For Phase 1, `grad` only takes a first-order function whose
> body does not contain another `grad`. A `grad` *nested inside an effect handler*
> (e.g. `handle[Random] ... grad(f) ...`) IS in scope and is exercised by T4's third
> example.

> **`vmap` inside the body of a `grad` is also out of scope for Phase 1.** That is,
> `grad(λx. ... vmap(g)(xs) ...)` — where `vmap` appears in the differentiated
> function's body — requires a `vmap` clause in the adjoint transformation, which
> is a non-trivial design task: one must decide whether to push `grad` inside `vmap`
> (commuting with `jacrev`/`jacfwd` semantics, as JAX does) or treat `vmap(g)` as
> opaque and apply the adjoint to its lifted form. Both options involve metatheory
> this design does not settle. For Phase 1, `grad`'s body does not contain `vmap`
> calls. The **outer-form `vmap(grad(f))`** — where `vmap` wraps a complete `grad`
> call — IS in scope and is the form used in T4's third example: the `grad(f)` is
> handled by this document's adjoint rules, and the `vmap` wrapper just lifts the
> whole gradient function point-wise over a new batch dimension.

Also out of scope: the definition of `adjoint` as a Lean function (that's T8), the
linearity soundness lemma (that's Phase 2 WS2.8), and the adjoint typing lemma itself
(that's Phase 2 WS2.2). T0 is the *informal* design that those deliverables will then
encode and prove.

## 1. Notation and conventions

Terms use the syntax fragment locked in `decisions.md` and `proof/workstreams/ws1-core-calculus.md`:

```
e ::= x | λx:τ. e | e₁ e₂ | let x = e₁ in e₂
    | copy(e)                      -- physical copy: produces a linear pair (original, fresh copy)
    | op(e₁, ..., eₙ)              -- RISC primitives: const, add, mul, sum, expand, uniform_like
    | grad(e) | vmap(e)
    | handle[ε] e with h | perform op(e)
    | (e₁, e₂) | fst(e) | snd(e)
```

Contexts and judgments:

- `Δ; Γ ⊢ e : τ ! ε ⊣ Γ'` — input capability context `Δ`, input linear context `Γ`,
  output linear context `Γ'`.
- `Γ = Γ₁ + Γ₂` — context split (each linear binding goes to exactly one side).
- Store `σ : Loc → TensorVal`. A linear binding `x : τ` in `Γ` owns a location
  `ℓ_x ∈ σ` unless stated otherwise.

### 1.0 `copy` is a pair-producing primitive

**This is the single load-bearing semantic point in the T0 design.** `copy` is typed:

```
copy : τ → τ ⊗ τ
```

That is: `copy(e)` takes a linear binding of type `τ` and produces a linear **pair**
whose first component is the original value (re-bound after the consuming call) and
whose second component is a fresh copy on a new store location. The canonical use
form is pattern destructuring:

```
let (x, x') = copy(e) in <rest>
```

where `x` is the original value (same data, same interpretation, now bound to a new
variable because `e` was consumed as the `copy` argument) and `x'` is the fresh
copy on a fresh location.

Operationally, `copy` reduces via:

```
⟨σ, copy(e)⟩  ↦*  ⟨σ, v⟩  (evaluate e to a location value ℓ)
⟨σ, copy(ℓ)⟩  ↦  ⟨σ ⊎ {ℓ' ↦ σ(ℓ)}, (ℓ, ℓ')⟩   for fresh ℓ'
```

The original location `ℓ` is **not freed** by `copy`; it is retained, owned by the
first component of the returned pair. A fresh location `ℓ'` is allocated with a
byte-identical copy of the data and owned by the second component.

**Single-owner invariant still holds** (`decisions.md` §"Heap-based store semantics"):
before `copy`, `e` owned `ℓ`. After `copy`, the first component of the returned
pair owns `ℓ` and the second component owns `ℓ'`. Each live location has exactly
one owner. The difference from a naïve "unary `copy` that consumes" reading is
that `copy` does not *free* its input — it *relocates ownership* from the argument
binding to the first component of the pair. This is consistent with
`decisions.md`'s "allocates a fresh location `ℓ'` with the same data — physical
copy, not aliasing" (which specifies what happens to `ℓ'` but is silent on the
fate of `ℓ`; we pin it here).

**Shorthand.** When the original is not needed again, the shorthand
`let x' = copy(e)` is equivalent to `let (_, x') = copy(e) in let _ = drop(_) in ...`
— but LaCaDiLE does not have `drop`, so this shorthand is only meaningful when the
first component is consumed by some later operation in the same `let` chain. For
clarity, every derivation in this document uses the explicit pair-destructuring form
`let (x, x') = copy(e)` and threads both components through the subsequent terms.

### 1.2 What we mean by "tape"

In PyTorch / JAX autograd, the tape is an implicit data structure the runtime keeps
alive for the duration of backward. In LaCaDiLE, the tape is **nothing more than a set
of extra linear bindings introduced during the forward pass** and left in scope until
the backward pass consumes them. There is no separate runtime data structure; the
linear type system tracks the tape via ordinary Γ entries.

The adjoint transformation uses `copy` (§1.0) to build tape entries: before each
primitive that needs to remember an operand's value for the backward pass, the
transformation inserts `let (a, a_tape) = copy(a)` — preserving `a` for the
forward-pass primitive and producing `a_tape` as the tape entry for the backward
pass. Both `a` and `a_tape` are linear bindings on distinct locations; the primitive
consumes `a` normally, and `a_tape` stays in scope until the backward pass consumes
it. Every tape entry is consumed **exactly once** by the backward pass.

The `&` notation in the master plan's decision text ("borrows into a tape") is
informal. In the formal calculus, tape entries are **`copy`-produced linear
bindings**, never raw borrows. There is no dangling-borrow problem because `copy`
retains the original (see §1.0) and the fresh copy has its own lifetime.

The implementation can and will use `Arc`-based reference counting for immutable
tensor data (see `decisions.md` and `proof/workstreams/ws4-paper-draft.md` §6) so
the "physical copy" is effectively a bumped refcount, not a memcpy. This is an
optimization and does not affect the metatheory.

### 1.3 Gradient accumulation via `Accum`

The `Accum` effect provides a controlled mechanism for gradient accumulation. The
reduced term of `grad(λx. e)` (from `decisions.md`) is:

```
λx. handle[Accum] (adjoint(e, x)) with {
  accum(loc, val, k) → k(update_grad(loc, val))
}
```

Where:

- `adjoint(e, x)` is the transformed body: forward pass followed by backward pass,
  with gradient contributions for each linear binding emitted via
  `perform accum(loc, grad_contribution)`.
- `update_grad(loc, val)` adds `val` to the running gradient buffer indexed by `loc`.
  The continuation `k` receives the updated buffer and resumes.
- At the end of `adjoint(e, x)`, the buffer entry for `x`'s location is extracted
  and returned as the gradient.
- `k` is a **linear binding** (one-shot by construction, per `decisions.md`).

**Key point for T0:** `Accum` handles gradient accumulation *across primitives*
(when the same source-level binding appears as an operand to multiple primitives
whose adjoints each contribute to its gradient). It does **not** handle
*within-primitive* gradient duplication (like `add`'s adjoint `(g, g)`, which needs
the same `g` in two places within a single primitive's adjoint rule). Within-primitive
duplication must use explicit `copy(g)`. See §2.2 below.

## 2. Atomic primitives

For each primitive, we give the adjoint transformation rule: given a term of the
form `let y = op(...operands...) in ...` (which consumes the operands and produces
`y`), we define how to insert tape-building copies in the forward pass and how to
produce backward-pass contributions.

### 2.1 `mul(a, b)` — the canonical hard case

Forward term (the programmer writes this):

```
let y = mul(a, b) in <rest>
```

Input context before: `Γ = Γ_pre + {a : tensor[d̄], b : tensor[d̄]}`.

After `mul`, in the non-grad case: `Γ = Γ_pre + {y : tensor[d̄]}`. `a` and `b` are
consumed.

**Under `grad`, the transformation inserts tape copies using the pair-destructuring
form from §1.0:**

```
let (a, a_tape) = copy(a) in
let (b, b_tape) = copy(b) in
let y = mul(a, b) in
<rest-transformed>
```

Linear context threading:

| Point | Γ |
|---|---|
| Before `copy(a)` | `Γ_pre, a:τ, b:τ` |
| After `let (a, a_tape) = copy(a)` | `Γ_pre, a:τ, b:τ, a_tape:τ` |
| After `let (b, b_tape) = copy(b)` | `Γ_pre, a:τ, b:τ, a_tape:τ, b_tape:τ` |
| After `let y = mul(a, b)` | `Γ_pre, a_tape:τ, b_tape:τ, y:τ` |

Each `copy` call takes a linear binding and re-binds the original plus introduces a
fresh tape copy on a new location (§1.0). After the destructure, both `a` and
`a_tape` are in scope with distinct locations — `a` is the same data the caller
passed in, now rebound; `a_tape` is the fresh physical copy. The forward `mul(a, b)`
consumes `a` and `b` normally, freeing their (already-rebound) locations. `a_tape`
and `b_tape` remain in `Γ` as tape entries until the backward pass consumes them.

**Backward pass** for this `mul`, given incoming output gradient `g : tensor[d̄]` (g
is either the seed gradient at the top of `adjoint` or a contribution from a
downstream primitive's adjoint):

```
let (g, g_for_a) = copy(g) in            -- duplicate g explicitly; needed for two uses
let grad_a = mul(g_for_a, b_tape) in      -- ∂L/∂a = g ⊙ b
perform accum(loc(a), grad_a);             -- contribute ∂L/∂a to a's gradient buffer
let grad_b = mul(g, a_tape) in             -- ∂L/∂b = g ⊙ a (g is last-used here)
perform accum(loc(b), grad_b);
```

Where `loc(a)` and `loc(b)` are the location identifiers — in the encoded calculus,
these are symbolic references to the original linear bindings `a` and `b` in the
enclosing grad scope, resolved through a compile-time environment. (The implementation
uses a numerical location index; the metatheory treats them as symbolic.)

Linear context threading on the backward (Γ_pre elided):

| Point | Γ |
|---|---|
| Before backward | `a_tape:τ, b_tape:τ, g:τ` |
| After `let (g, g_for_a) = copy(g)` | `a_tape:τ, b_tape:τ, g:τ, g_for_a:τ` |
| After `let grad_a = mul(g_for_a, b_tape)` | `a_tape:τ, g:τ, grad_a:τ` (g_for_a and b_tape consumed) |
| After `perform accum(loc(a), grad_a)` | `a_tape:τ, g:τ` (grad_a consumed by the effect op) |
| After `let grad_b = mul(g, a_tape)` | `grad_b:τ` (g and a_tape consumed) |
| After `perform accum(loc(b), grad_b)` | `Γ_pre` (grad_b consumed) |

All tape entries are consumed exactly once. Both components of each `copy`-produced
pair are consumed exactly once. Linear context accounting is balanced — the final Γ
is exactly `Γ_pre`, meaning this `mul`'s adjoint introduced and then destroyed
exactly the bindings it needed.

**Why the explicit `copy(g)`?** The `mul` adjoint needs `g` in two places: once
multiplied by `b_tape` to produce `grad_a`, and once multiplied by `a_tape` to produce
`grad_b`. Under LaCaDiLE's linearity rules, `g` can only be consumed once. To use it
twice, we need two linear bindings with the same value — which is exactly what
`copy` produces. This is not an optimization we avoid; it's a semantic requirement.
At the runtime level, `Arc` makes the duplication free for immutable tensor data, but
at the type level it's mandatory.

**Relation to `Accum`:** The two `perform accum` calls do NOT duplicate `grad_a` or
`grad_b` — each produces a single contribution for a single location. `Accum`'s role
here is to *collect* these contributions into the per-location gradient buffer, where
they may be added to contributions from other primitives whose adjoints also
reference `a` or `b`. The within-primitive duplication of `g` is handled by `copy`;
the across-primitive accumulation is handled by `Accum`.

### 2.2 `add(a, b)` — within-primitive duplication

Forward:

```
let y = add(a, b) in <rest>
```

Adjoint rule:

```
-- forward (add's adjoint does not depend on a or b, so no tape needed)
let y = add(a, b) in <rest-transformed>

-- backward, with incoming gradient g : tensor[d̄]
let (g, g_for_b) = copy(g) in   -- destructure: g is the original rebinding, g_for_b is the fresh copy
perform accum(loc(a), g);        -- contribute ∂L/∂a = g_in to a's gradient buffer
perform accum(loc(b), g_for_b);  -- contribute ∂L/∂b = g_in to b's gradient buffer
```

No tape. The forward pass is unchanged — `add`'s adjoint doesn't depend on the
operand values, only on the output gradient. The backward pass uses `copy(g)` to
duplicate the gradient (§1.0: pair-producing copy), then routes each component to
one operand's `accum`.

Linear threading (Γ_pre elided):

| Point | Γ |
|---|---|
| Before backward | `g:τ` |
| After `let (g, g_for_b) = copy(g)` | `g:τ, g_for_b:τ` |
| After `perform accum(loc(a), g)` | `g_for_b:τ` |
| After `perform accum(loc(b), g_for_b)` | `Γ_pre` |

**Γ accounting is balanced.** Two linear values consumed by two accum calls.

**Decisions.md position vindicated.** The phrase "the adjoint of `add(a, b)` is
`(g, g)` — a copy of `g` — which violates linearity. But the duplication is typed as
`Accum` accumulation" should be read as: the literal `(g, g)` would violate linearity,
which is why the adjoint explicitly does `copy(g)`. The `Accum` piece is what lets
the two separate contributions be added together into the shared gradient buffer —
making the effect of `(g, g)` visible to the caller *as if* it were a single
gradient duplication. The linearity system requires the `copy` to be explicit; the
effect system provides the accumulation. Both are necessary.

### 2.3 `sum(x, axis)` — dimension-list adjoint

Forward:

```
let y = sum(x, axis) in <rest>
```

Where `x : tensor[d̄]` and `y : tensor[d̄ \ axis]` (the dimension at `axis` is removed
from the shape).

`sum`'s adjoint depends on `x`'s *shape* (specifically the extent of the removed
dimension), not its values. But since dimensions are part of the type, the shape is
statically known; we don't need to tape the full `x`. We do need the dimension list
`d̄` — which the type system tracks symbolically — to construct the `expand` call
in the backward pass.

Adjoint rule:

```
-- forward: no tape needed (shape is static, values not needed)
let y = sum(x, axis) in <rest-transformed>

-- backward, with incoming g : tensor[d̄ \ axis]
let grad_x = expand(g, axis, d̄) in    -- e.g. expand(g, 0, [batch, seq, d_model])
perform accum(loc(x), grad_x);
```

The `expand` reinstates the dimension that `sum` removed by replicating the gradient
along that axis. The shape argument `d̄` to `expand` is syntactically derivable from
`x`'s type at transformation time.

Linear threading (Γ_pre elided):

| Point | Γ |
|---|---|
| Before backward | `g:tensor[d̄\axis]` |
| After `let grad_x = expand(g, axis, d̄)` | `grad_x:tensor[d̄]` (g consumed) |
| After `perform accum(loc(x), grad_x)` | `Γ_pre` (grad_x consumed) |

Balanced. No copies needed; `g` is used exactly once.

### 2.4 `expand(x, shape)` — complementary to `sum`

Forward:

```
let y = expand(x, axis, k) in <rest>
```

Where `x : tensor[d̄]` and `y : tensor[axis_k, d̄]` (a new dimension of extent `k`
at position `axis`).

Adjoint:

```
-- forward: no tape needed
let y = expand(x, axis, k) in <rest-transformed>

-- backward, with incoming g : tensor[axis_k, d̄]
let grad_x = sum(g, axis) in
perform accum(loc(x), grad_x);
```

`expand` adds a dimension; its adjoint removes that dimension by summing. Dual of
§2.3 in every respect. Linear threading identical in shape.

**Metatheoretic importance.** `sum` and `expand` are *mutually adjoint*. `grad(sum)`'s
body contains `expand`, and `grad(expand)`'s body contains `sum`. This is why both
must be in the core calculus: if we dropped `expand`, `grad(sum)` would not reduce
to a well-typed term (it would need an operator that isn't in the calculus). The
six-primitive set in `decisions.md` is minimal for AD closure over the dimension
algebra.

### 2.5 `const(v, shape)` — no inputs, adjoint routes through handler filter

Forward:

```
let y = const(v, shape) in <rest>
```

`const` has no tensor operands; it just allocates a fresh location with the given
data. A literal value has no "upstream" source-level binding whose gradient could
be affected, so the gradient contribution for `const` has nowhere to go.

**Design decision (uniform transformation, handler-filtered).** The adjoint rule for
`const` is **uniform with all other primitives**: when the backward pass reaches a
`const` primitive, it emits `perform accum(loc(y), g)` (where `g` is the incoming
gradient for the `const`'s output and `loc(y)` is the `const`'s output location).
The `Accum` handler then **filters** this contribution by detecting that `loc(y)`'s
origin is a `const` literal (not a parameter of the enclosing `grad`) and dropping
the contribution — i.e., `update_grad` is a no-op for `const`-origin locations.

```
-- forward
let y = const(v, shape) in <rest-transformed>

-- backward (incoming gradient g : tensor[shape])
perform accum(loc(y), g);
```

Linear threading:

| Point | Γ |
|---|---|
| Before backward | `g:tensor[shape]` |
| After `perform accum(loc(y), g)` | `Γ_pre` (g consumed by the effect op) |

**Why uniform instead of pass-through?** The alternative — make `const`'s adjoint
the identity function `g → g` and rely on the parent primitive to consume the
passed-through gradient — creates non-uniformity in the adjoint transformation: each
primitive must know whether its operand was a `const` and behave differently. The
uniform accum-and-filter version keeps the transformation primitive-local and pushes
the special-casing into the handler's `update_grad`, which is a single compile-time
function. Simpler to encode in Lean (T7/T8) and simpler for the metatheory: every
gradient value flows through exactly one `accum` call, with no exceptions.

**Why not just delete the accum call?** Linearity. The incoming `g` must be consumed
somewhere; LaCaDiLE does not silently drop linear values. `perform accum(loc, g)` is
how the backward pass consumes `g`. The handler receives it, inspects the location,
and decides whether to update the gradient buffer — but the `perform` call itself
is mandatory to preserve linear accounting.

**How does the handler know `loc(y)` is a `const` origin?** The `grad` reduction
threads a compile-time location-to-origin map (see §5). `const` primitives register
their output location as origin-kind = `const-literal`, and `update_grad` checks the
origin-kind before mutating the buffer. See §5 for the full handler shape and §8 for
open questions about how origin-kind is formally tracked in the Lean encoding.

### 2.6 `uniform_like(template, lo, hi)` — rejected by `grad`

`uniform_like` has effect `Random`. From `decisions.md`:
`DiffCompat = {Resource, Accum}`. Since `Random ∉ DiffCompat`, any body containing
`uniform_like` makes `grad` reject the term at the type-checking phase. The adjoint
rule for `uniform_like` is therefore vacuous — it never runs. The check happens in
`T-Grad`: the premise `ε ⊆ DiffCompat` fails because `Random ∈ ε`.

**Worked end-to-end rejection.** Consider:

```
grad(λx. let r = uniform_like(x, 0, 1) in mul(x, r))
```

Proposed typing derivation, bottom-up:

1. `x : tensor[d̄]` in scope.
2. `uniform_like(x, 0, 1)` — need to type-check. The T-UniformLike rule produces a
   tensor of the same shape as its template and introduces effect `Random` into the
   effect row. Result type: `tensor[d̄]`, effect row contribution: `{Random}`.
3. `let r = uniform_like(x, 0, 1) in ...` — binds `r : tensor[d̄]`, the body's
   effect row starts with `{Random}`.
4. `mul(x, r)` — need `x` linear and `r` linear; but `x` was already consumed by
   the `uniform_like` call in step 2 (since `uniform_like` takes `template`
   linearly). **First problem**: `x` is not in Γ anymore when `mul` tries to use it.
   The fix in the source is to `copy` `x` before `uniform_like`: rewrite as
   `let (x, x') = copy(x) in let r = uniform_like(x', 0, 1) in mul(x, r)`. Assume
   the programmer did this.
5. Now the body of the lambda has shape `let (x, x') = copy(x) in let r =
   uniform_like(x', 0, 1) in mul(x, r)`. The body's effect row, computed
   bottom-up: `copy` contributes `∅`, `uniform_like` contributes `{Random}`,
   `mul` contributes `∅`. Total effect row: `{Random}`.
6. The lambda is typed `tensor[d̄] → tensor[d̄] ! {Random}`.
7. Now apply `T-Grad`:
   ```
   Δ, Diff; Γ ⊢ f : (tensor[d̄] → tensor[d̄'] ! ε)
   f uses its argument linearly         -- OK, we copied x explicitly
   ε ⊆ DiffCompat                        -- FAILS: {Random} ⊄ {Resource, Accum}
   ──────────────────────────────────────────
   Δ; Γ ⊢ grad(f) : ...
   ```
   The premise `ε ⊆ DiffCompat` fails because `Random ∈ ε` and `Random ∉
   {Resource, Accum}`. The rule does not apply. **`grad(f)` does not type-check.**

The adjoint transformation is therefore never invoked — T-Grad rejects the term
before `adjoint(e, x)` can reduce. The rejection is at the **type-checking phase**,
which is exactly what the locked decision in `decisions.md` requires.

If a programmer *wants* to differentiate through a stochastic function (e.g. for a
reparameterization-trick gradient estimator), they would need to sample the noise
outside the `grad` scope and pass it in as a non-random tensor:

```
let noise = uniform_like(template, 0, 1) in
grad(λx. mul(x, noise))(params)
```

Here the `uniform_like` is outside `grad`, so the `grad`'s body no longer has
`Random` in its effect row. This is the standard pattern and is cleanly supported.

## 3. Composition

Atomic primitives are the easy case. Composition — where primitives feed each other
— is where tape management gets subtle. This section works two non-trivial examples.

### 3.1 Composition case 1: `grad(λx. sum(mul(x, x), 0))`

This exercises: a self-multiplication (same operand used twice, requiring `copy`),
feeding into a reduction, differentiated w.r.t. the input.

Source term:

```
grad(λx. sum(mul(x, x), 0))
```

Issue: `mul(x, x)` uses `x` twice, which violates linearity. We must insert an
explicit `copy` in the source before the adjoint transformation is even legal. The
well-typed source is:

```
grad(λx.
  let (x, x1) = copy(x) in        -- canonical pair-destructure form (§1.0)
  let x2 = x in                    -- rename: the original rebinding is now x2
  let y = mul(x1, x2) in
  sum(y, 0)
)
```

Under the `grad(λx.e)` reduction, the body `e` becomes the argument to `adjoint`.
Applying the transformation primitive-by-primitive from §2:

**Forward pass with tape insertion:**

```
let (x, x1) = copy(x) in              -- source-level linearity copy (programmer-supplied)
let x2 = x in                          -- rename
let (x1, x1_tape) = copy(x1) in        -- adjoint: tape for mul's a operand
let (x2, x2_tape) = copy(x2) in        -- adjoint: tape for mul's b operand
let y = mul(x1, x2) in                 -- forward mul (consumes x1, x2)
let s = sum(y, 0) in                    -- forward sum (consumes y)
-- s is the output; s : tensor[d̄ \ 0]
```

`y_tape` is NOT created because `sum`'s adjoint doesn't need `y`'s value (§2.3).
The tape only captures operand values that the backward pass will actually
reference. The adjoint transformation is a static analysis: for each primitive, it
knows from the primitive's adjoint rule whether a tape entry is needed.

Linear context at the forward/backward boundary:

`Γ_fwd_end = { x1_tape:τ, x2_tape:τ, s:τ\0 }`

The outer `x` is out of scope — it was consumed by the source-level `copy(x)` at
the very start (with both components immediately rebound: `x` itself as the rename
target of `let x2 = x` which follows, and `x1` as the fresh copy). `x1` and `x2`
were consumed by `mul`. `y` was consumed by `sum`. Only the tape entries and the
forward output remain.

**Seed gradient** at the start of the backward pass: `g_s : tensor[d̄ \ 0]` — the
seed for `grad(f)(x₀) = ∂f/∂x|_{x=x₀}` is an abstract external argument passed to
the outer `grad` reduction. For a scalar-output `f`, the user supplies `g_s = 1`
(a scalar tensor of ones, constructible as `const(1, d̄\0)`), but we treat it
symbolically here.

**Backward pass**, unwinding the forward in reverse order:

```
-- start: Γ = { x1_tape:τ, x2_tape:τ, s:τ\0, g_s:τ\0 }
-- (note: s still exists but is the handler's overall return; for the adjoint
-- derivation we treat it as consumed at the handler boundary.)

-- step 1: invert sum (§2.3 rule with g = g_s, axis = 0, shape = d̄):
let grad_y = expand(g_s, 0, d̄) in    -- grad_y : τ (same shape as y / x1 / x2)
-- Γ = { x1_tape:τ, x2_tape:τ, grad_y:τ }

-- step 2: invert mul(x1, x2) (§2.1 rule with a=x1, b=x2, tape=(x1_tape, x2_tape), g=grad_y):
let (grad_y, g_for_x1) = copy(grad_y) in   -- canonical pair-destructure; grad_y is rebound, g_for_x1 is the fresh copy
let grad_x1 = mul(g_for_x1, x2_tape) in     -- ∂(x1*x2)/∂x1 = x2, so grad_x1 = g * x2_tape
perform accum(loc(x1), grad_x1);             -- contribute to x1's gradient buffer
let grad_x2 = mul(grad_y, x1_tape) in        -- ∂(x1*x2)/∂x2 = x1 (grad_y is last-used here)
perform accum(loc(x2), grad_x2);
-- Γ = Γ_pre after all tape entries and gradients consumed
```

**Balanced linear accounting on the backward pass:**

- `grad_y`: produced once (by `expand`); destructured via `copy` into the
  rebinding `grad_y` and the fresh `g_for_x1`, each consumed exactly once (by the
  two `mul` calls).
- `x1_tape` and `x2_tape`: each produced once (by the forward `copy`), consumed
  once (by the two backward `mul` calls).
- `grad_x1`, `grad_x2`: produced once each, consumed by the `accum` effect op.

**All balanced.** The linear context at the end of the backward pass is exactly
`Γ_pre`. ✓

**Origin tracking for the final gradient.** `loc(x1)` and `loc(x2)` are the
locations of the source-level bindings `x1` and `x2`. Both trace back to `x`
via the source-level `copy(x)`: the second component of `let (x, x1) = copy(x)`
is `x1` and the first is later renamed to `x2`. Both locations therefore have
the same **origin** (the outer parameter `x`), and the `Accum` handler's
`update_origin_buffer` adds both contributions to `x`'s buffer entry (§5).
When the `grad` scope closes, the handler reads that entry and returns it as
the gradient of the body with respect to `x`.

**Full reduced-term sketch** (putting it together inside the `grad` handler):

```
grad(λx. sum(mul(x, x), 0))  ↦

λx. λg_s. handle[Accum] (
  -- source-level linearity setup:
  let (x, x1) = copy(x) in
  let x2 = x in
  -- adjoint of mul(x1, x2):
  let (x1, x1_tape) = copy(x1) in
  let (x2, x2_tape) = copy(x2) in
  let y = mul(x1, x2) in
  -- adjoint of sum(y, 0):
  let s = sum(y, 0) in
  -- backward: sum
  let grad_y = expand(g_s, 0, d̄) in
  -- backward: mul
  let (grad_y, g_for_x1) = copy(grad_y) in
  let grad_x1 = mul(g_for_x1, x2_tape) in
  perform accum(loc(x1), grad_x1);
  let grad_x2 = mul(grad_y, x1_tape) in
  perform accum(loc(x2), grad_x2);
  -- the handler's buffer has accumulated both x1 and x2 contributions
  -- under the same origin (x); return that entry.
  extract_grad(loc(x))
) with {
  accum(loc, val, k) →
    k(update_origin_buffer(origin_of(loc), val))
}
```

Note the extra `λg_s.` outer abstraction: per §8's "seed gradient" discussion,
the `grad` reduction produces a function that takes two arguments — the parameter
value and the output-gradient seed. This matches the
`grad(f)(x₀, cotangent)` shape used in academic presentations.

This term typechecks under the T2 rules (to be written in Wave 2). All linear
bindings are consumed exactly once. All tape entries are produced by `copy` and
consumed by `mul`. All gradient contributions flow through `perform accum` to the
handler. The handler's per-origin buffer is threaded through the continuation
`k`, which is linear (one-shot by construction).

### 3.2 Composition case 2: `grad(λx. let y = mul(x, const(2, [])) in sum(y, 0))`

This exercises: a primitive whose operand is a `const` literal (zero gradient),
combined with a source-level `let` binding, feeding a reduction.

Source:

```
grad(λx. let y = mul(x, const(2, [])) in sum(y, 0))
```

No source-level linearity issues — `x` is used once and `const(2, [])` produces a
fresh value. No source-level `copy` needed.

**Forward with tape** (using the canonical pair-destructure form from §1.0):

```
let c = const(2, []) in              -- fresh constant
let (x, x_tape) = copy(x) in          -- tape for mul's a operand
let (c, c_tape) = copy(c) in          -- tape for mul's b operand
let y = mul(x, c) in                  -- forward mul (x and c consumed)
let s = sum(y, 0) in                  -- forward sum (y consumed)
```

Γ at the forward/backward boundary: `{ x_tape:τ, c_tape:τ, s:τ\0 }` (x, c, y all
consumed).

**Backward:**

```
-- incoming seed g_s : τ\0
-- invert sum:
let grad_y = expand(g_s, 0, d̄) in
-- invert mul(x, c): need copy(grad_y) for the two adjoint uses
let (grad_y, g_for_x) = copy(grad_y) in
let grad_x = mul(g_for_x, c_tape) in           -- ∂(x*c)/∂x = c, so grad_x = g * c_tape
perform accum(loc(x), grad_x);
let grad_c = mul(grad_y, x_tape) in            -- ∂(x*c)/∂c = x, so grad_c = g * x_tape
perform accum(loc(c), grad_c);                  -- const-origin filter fires in the handler (§2.5, §5)
```

The `perform accum(loc(c), grad_c)` is required for linearity (grad_c must be
consumed). The handler's `update_origin_buffer` detects that `loc(c)`'s origin is
`const-literal` and returns the buffer unchanged (§2.5, §5). This is the uniform
accum-and-filter pattern: every gradient flows through exactly one `accum`, and the
handler decides routing. No special cases in the adjoint transformation itself.

Γ accounting: all tape entries and gradients consumed by mul/accum/sum/expand
calls exactly once. Balanced. ✓

**Key point made by this example:** source-level `let` bindings do NOT break the
tape. Each `let` introduces one linear binding; the adjoint transformation inserts
tape copies before each primitive as a local operation. The backward pass unwinds in
reverse order of the forward, and the linear context shrinks monotonically from the
forward/backward boundary to zero.

## 4. The general recursive pattern

From §2 and §3 we can extract the general shape of the adjoint transformation.
Informally:

```
adjoint(e, x) := forward_pass(e) ; backward_pass(e, x)
```

where:

- `forward_pass(e)` walks `e` in source order. For each primitive `op(args...)` whose
  adjoint rule needs operand values (§2.1: `mul`; not §2.3: `sum`), insert
  `let (arg, arg_tape) = copy(arg)` immediately before the primitive. Let-bindings
  and application nodes in `e` are unchanged structurally — only the primitive call
  sites get tape-copy wrappers.

- `backward_pass(e, x)` walks `e` in *reverse* order. Starting from a seed gradient
  at the output of `e`, each primitive's adjoint rule:
  1. Consumes the incoming gradient (with `copy` if it needs to be used in multiple
     places).
  2. Consumes the relevant tape entries.
  3. Produces gradient contributions for each operand.
  4. Emits `perform accum(loc(operand), contribution)` for each.

- The backward pass ends when every operand of every primitive has had its gradient
  accumulated, at which point the linear context is empty (modulo `Γ_pre` and the
  final gradient value for `x`).

The formal definition is structural recursion on `e`:

```
adjoint(let y = op(a₁, ..., aₙ) in rest, x) :=
  -- forward tape insertion (only for operands op's adjoint rule needs):
  let (a₁, a₁_tape) = copy(a₁) in        -- if op needs a₁_tape
  ...
  let (aₖ, aₖ_tape) = copy(aₖ) in        -- if op needs aₖ_tape
  -- forward primitive:
  let y = op(a₁, ..., aₙ) in
  -- recurse on the rest of the forward:
  adjoint(rest, x)
```

and the backward is implicit in the handler wrapping: when `adjoint(rest, x)`
returns, the backward of the current primitive runs, using the tape entries in
scope. Concretely, we factor the transformation into two passes:

```
adjoint(e, x) := adjoint_fwd(e) ; adjoint_bwd(e, x, seed)
```

with `seed` threaded through as the incoming gradient at each backward step. The
formal Lean encoding of this (T8) will make the recursion explicit; for T0 it's
enough that the term-level pattern is clear from the examples.

**Invariants** that any implementation of `adjoint` must preserve:

1. **Tape entries are `copy`-produced linear bindings, never raw borrows.** This is
   what makes the linear context accounting consistent; tape entries are consumed
   exactly once by the backward pass and freed when that consumption happens.

2. **Gradient accumulation within a primitive uses explicit `copy(g)`**, not
   `Accum`. The `Accum` effect is only for gradient accumulation *across* primitives
   (when the same source-level binding appears in multiple primitive calls). This
   keeps the `Accum` handler simple (it just sums contributions per location) and
   keeps within-primitive linearity checks local.

3. **The `Accum` handler wraps the entire body of `adjoint(e, x)`** and is introduced
   by the `grad` reduction, not by the adjoint transformation. The transformation
   produces a term with `perform accum` calls; the `handle` wrapper is the outer
   shell.

4. **Every linear binding in the forward pass is either consumed by the forward
   primitive, consumed by a backward primitive, or consumed by an `accum` effect
   op.** No silent drops; no dangling.

5. **Backward pass reverses forward order.** For `let y = op1(a) in let z = op2(b) in
   ...`, the backward is `...backward of op2 then backward of op1`. This ensures
   tape entries for `op1` remain in scope until `op1`'s adjoint runs.

## 5. Accum handler shape and location tracking

For reference, the final reduced term of `grad(λx. e)` has this shape (formally,
with the gradient buffer threaded as a linear argument of the continuation `k` —
no mutable state):

```
λx. λg_s.
handle[Accum] (adjoint(e, x, g_s)) with {
  accum(loc, val, k) →
    k(update_origin_buffer(origin_of(loc), val))
}
-- and the final return value is extract_grad(buffer, loc(x))
-- (the gradient for the parameter x, read out of the buffer that flows out of the handler)
```

The two-argument shape (`λx. λg_s.`) follows from the explicit-seed design pinned
down in §8 and exercised in §3.1's reduced-term sketch: the user supplies the
parameter `x` and the output-gradient seed `g_s : τ_out` (typically a scalar
tensor of ones for a scalar-output loss). This matches the academic
`grad(f)(x, cotangent)` shape. The `adjoint` transformation threads `g_s` as
the starting gradient for the backward pass; every call to `perform accum`
eventually causes a contribution to land in the handler's buffer.

Where:

- The gradient **buffer** is a map from *origin* locations to tensor values, and
  is threaded through the continuation `k` as its linear argument. Each
  `perform accum(loc, val)` call invokes the handler, which computes an updated
  buffer and passes it to `k`. The continuation `k` is linear (one-shot by
  construction, per `decisions.md`), so each `accum` call produces exactly one new
  buffer and each buffer has exactly one consumer.

- **Locations that share an origin** are grouped. Locations that came from a chain
  of `copy` calls from the same source binding all have the same origin; the
  buffer is keyed by origin, not by raw location. This is what allows a primitive's
  adjoint to emit `perform accum(loc(a), ...)` for a `copy`-produced tape entry
  `a` and have the contribution land in the same buffer slot as a contribution
  from a sibling `copy` of the same source.

- `update_origin_buffer(o, v)` is a pure function: it takes an origin identifier
  and a tensor value, looks up the origin's current entry in the buffer, adds `v`
  to it (or inserts if absent), and returns the new buffer.

- `origin_of(loc)` is a **compile-time function** that walks the `copy` chain back
  to the source location. It is not a runtime lookup; `grad`'s reduction produces a
  term where every `loc(...)` argument is already resolved to a static symbolic
  origin, and the handler's pattern-match is on the origin directly.

- `extract_grad(buffer, loc(x))` reads the final gradient for the parameter `x`
  out of the buffer after the handler closes. The handler's return shape is
  `handle ... with { ... } : τ_param` where `τ_param` is the type of the
  differentiated parameter.

- **Filtering for `const`-origin locations** (§2.5) happens inside
  `update_origin_buffer`: when the origin identifier is tagged as `const-literal`
  (a compile-time attribute set when `const` primitives register their output
  location), the function returns the buffer unchanged — the contribution is
  discarded. For all other origin kinds (parameter, intermediate), the buffer is
  updated.

The precise formal shape of the buffer threading and the handler's return type is
pinned down in T2 (typing rules) and T3 (operational semantics) during Wave 2. §8
flags this and `origin_of` as open questions for the Lean encoding (T7).

## 6. Interaction with `grad` inside an effect handler

T4's third example (per the plan): `grad(f)` appears inside a `handle[Random]` scope.
This is in scope for Phase 1 (it's the "grad inside handler" case mentioned in the
non-goals scope note).

The key question: does the tape/Accum mechanism interfere with the outer handler?

**Answer: no.** The `grad` reduction introduces its own `handle[Accum]` wrapper,
which captures only the `Accum` effect. The outer `handle[Random]` is unaffected by
`Accum` (different effect labels, and the row polymorphism in the effect system
ensures they compose cleanly). `Accum` is never part of a user-facing effect row;
it is introduced and immediately handled within the `grad` reduction.

The inner body `adjoint(e, x)` may itself perform `Random` operations (if `e` does),
and those propagate up through the `handle[Accum]` wrapper (since `Accum`'s handler
doesn't handle `Random`) to the outer `handle[Random]`. The two handlers are
orthogonal — `Accum` handles `Accum` effects, `Random` handles `Random` effects,
both walk the same continuation.

**However**, `T-Grad`'s premise `ε ⊆ DiffCompat` means `e`'s effect row must be a
subset of `{Resource, Accum}`. If `e` performs `Random` (e.g. dropout), the `grad`
is rejected — even if the `Random` would be handled by an outer handler. This is
the design: `grad` can only differentiate through deterministic functions.

This is stricter than what you might naïvely want (a stochastic gradient estimator
like REINFORCE would want `grad` over a `Random`-using body), but it matches the
locked decision in `decisions.md`. T4 example 3 therefore uses a body with only
`Resource` effects, not `Random`. A future extension could relax `DiffCompat` to
allow `Random` via reparameterization tricks; that's out of Phase 1 scope.

## 7. Summary of decisions made by T0

| Question | Decision |
|---|---|
| What is `copy`, formally? | `copy : τ → τ ⊗ τ`. Takes a linear input and returns a linear pair: first component is the original (rebound), second component is a fresh physical copy on a new location. The original is not freed. Canonical use: `let (x, x') = copy(x) in ...`. See §1.0. |
| What is a tape entry? | The second component of a `copy`-produced pair, introduced during the forward pass and consumed exactly once by the backward pass. Tape entries are linear bindings in Γ, not borrows. |
| How is within-primitive gradient duplication handled? | Explicit `copy(g)` in the backward adjoint rule, with the destructuring form `let (g, g') = copy(g)`. Both components are linear bindings, each consumed exactly once. |
| How is across-primitive gradient accumulation handled? | The `Accum` effect. `perform accum(loc, contribution)` emits one contribution per primitive per operand; the handler collects them by origin through a buffer threaded through the continuation. |
| Which primitives need tape entries? | `mul` needs both operands. `add` needs neither. `sum`/`expand` need neither (shape is static). `const`/`uniform_like` have no linear operands. |
| What happens to `const`'s gradient? | Uniform: the backward emits `perform accum(loc(const_output), g)`; the handler's `update_origin_buffer` detects the `const-literal` origin kind and returns the buffer unchanged. Every linear value flows through one `accum` call; the filter is in the handler, not the transformation. |
| What happens for `uniform_like`? | `T-Grad`'s `ε ⊆ DiffCompat` premise rejects any body whose effect row contains `Random`. The primitive's adjoint rule never runs. Worked rejection derivation in §2.6. |
| Is `grad(grad(f))` supported? | **No.** Out of scope for Phase 1. First-order `grad` only. |
| Is `vmap` inside a `grad` body supported? | **No.** Out of scope for Phase 1. §4's adjoint definition has no clause for `vmap`. The outer-form `vmap(grad(f))` IS supported (the `grad(f)` is an opaque gradient function; `vmap` just lifts it point-wise). |
| Is `grad(f)` inside a handler (e.g. `handle[Random] ... grad(f) ...`) supported? | **Yes**, provided `f`'s body itself has effects in `DiffCompat`. The outer handler is orthogonal to the `Accum` handler introduced by the `grad` reduction. |

## 8. Open questions flagged for Wave 2

These are not blockers for T0 — they are questions the LaTeX rules in Wave 2 must
answer:

- **Origin tracking in the typing rules.** §5 describes the gradient buffer as
  keyed by "origin" locations, with `copy` chains tracked back to source
  bindings. T0's story is that `origin_of(loc)` is a compile-time function on
  the syntax (not a runtime lookup). Wave 2 must make this precise in T2: either
  (a) the typing rule for `copy` records an origin tag in the output type, or
  (b) the `grad` reduction's handler pattern-matches on origin tags that are
  static in the reduced term. Leaning (b). T7 Lean encoding will pick an
  implementation once T2 settles.

- **`copy`'s type in the paper rules.** §1.0 pins `copy : τ → τ ⊗ τ` (pair-producing).
  T2 must add an explicit T-Copy rule matching this shape. The existing
  `decisions.md` text ("allocates a fresh location `ℓ'` ... every location has
  exactly one owner") is consistent with this reading; T0 treats it as the
  canonical interpretation. If Wave 2 discovers a conflict with another decision,
  raise it as a spec update, not a T0 fix.

- **Seed gradient for top-level `grad`.** §3.1 and §5 commit to the
  **explicit-seed form**: `grad(λx. e)` reduces to a *two-argument* function
  `λx. λg_s. handle[Accum] ...`, where `g_s : τ_out` is the output-gradient
  seed supplied by the caller. This matches the academic
  `grad(f)(x₀, cotangent)` shape used in JAX and the PL literature. For
  scalar-output losses, callers typically pass `g_s = const(1, shape_of(s))`.
  T0 does not use an implicit `ones_like` primitive — that would require
  extending the RISC primitive set. Wave 2 must pin the `T-Grad` rule to
  produce a function type of arity 2 (parameter + seed).

- **`expand` operand count** — **RESOLVED.** Wave 2 T2 adopted the three-argument form
  `expand(e, i, k)`, and `decisions.md` was updated in round 2 to match.
  `T-Expand` and `E-Expand` both take three arguments, and `Syntax.ins` in Lean
  implements the insertion semantics. This question is closed; T0 §2.3's use of
  `expand(g, axis, d̄)` should be read as the three-argument form with the dim
  list serving as the extent source — consistent with the locked decision.

- **Handler buffer threading in `T-Handle`.** §5 describes the gradient buffer
  as threaded through the continuation `k`. `decisions.md`'s handler-continuation
  shape doesn't explicitly show a buffer argument. Wave 2's T-Handle rule will
  need to either accept that the continuation's type carries a buffer-shaped
  argument, or formalize the `Accum` handler as a special case with state
  threading. Leaning toward the general "continuation has a type that can carry
  state" interpretation, which keeps `T-Handle` uniform.

None of the above affect the linearity accounting or the tape correctness
established in §2 and §3. They are downstream refinements needed by the LaTeX and
Lean encodings but are orthogonal to T0's core claim: that the tape mechanism keeps
linearity balanced across forward and backward passes for all six RISC primitives
and for nested `let`-composed terms.

---

## Red team checklist for round 1

Round 1 red team, please verify:

1. **Linearity balance.** For every worked example in §2 and §3, every linear
   binding introduced is consumed exactly once. Flag any double-consumption,
   never-consumed, or "consumed-while-borrowed" case.

2. **Tape entry semantics.** §1.1 claims tape entries are `copy`-produced linear
   bindings, not borrows. Does this claim hold consistently across all examples?

3. **`copy(g)` coverage.** For every primitive in §2 whose adjoint uses `g` more
   than once (`mul` §2.1, `add` §2.2), is there an explicit `copy(g)` in the
   derivation? Flag any missed cases.

4. **Accum filtering for `const`.** §2.5 and §3.2 rely on the handler discarding
   contributions whose origin is a `const`-produced location. Is this defensible
   without introducing non-uniformity in the handler? Probe by constructing a
   `grad(λx. let c = const(...) in ...use c...)` example and walking it through.

5. **`Random`-rejection for `uniform_like`.** Construct
   `grad(λx. let r = uniform_like(x, 0, 1) in mul(x, r))` and verify that it is
   rejected by the `T-Grad` `ε ⊆ DiffCompat` premise. This should be CRITICAL if
   the derivation goes through.

6. **`grad` inside `handle[Random]`.** Construct
   `handle[Random] ... grad(λx. mul(x, x)) ... with ...` and confirm that the
   outer `handle[Random]` is unaffected by the `Accum` handler introduced by the
   `grad` reduction. The derivation should complete.

7. **`grad(grad(f))` is out of scope.** Do not flag its absence as a finding. Do
   flag any claim in this document that implies `grad(grad(f))` works.

8. **Counterexample probe.** Construct one program designed to break the tape
   mechanism — e.g. a `grad` over a partially applied function, or a `grad` whose
   body uses `vmap` — and walk it through. If it breaks, that's a finding; if it
   works, that's vindication.

A report with zero findings is suspicious. Look for silent fabrication in the
primitive adjoint rules, for linearity holes in the composition cases, and for any
hand-waving about origin tracking (§5) or `const` gradient routing.
