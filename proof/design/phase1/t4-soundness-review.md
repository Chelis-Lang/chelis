# T4 — Soundness Review: Four Worked Examples

**Phase 1, Task T4.** Walks four canonical example programs through the typing
rules (`proof/paper/figures/typing.tex`, T2) and the operational semantics
(`proof/paper/figures/opsem.tex`, T3) to confirm the rules are internally
consistent and that the tape mechanism from T0 threads through cleanly.

**Status:** Draft for Wave 2 red team review.

T4 is an informal soundness review. Full progress/preservation proofs happen in
Phase 2 (WS2.4, WS2.5) and are mechanized in Lean in Phase 2 T9/T10. This
document's job is to **catch bugs in T2/T3 before they are encoded in Lean**,
by walking concrete programs and checking that the rules apply cleanly.

---

## Convention

- `Γ ⊢ e : τ ! ε ⊣ Γ'` shorthand elides `Δ` when the capability context is
  the same throughout (it is unless `grad` intervenes, in which case we note
  the `Diff` toggle explicitly).
- `Γ_0` is the external context that every derivation threads unchanged as
  the outer `Γ_pre`. `Γ_0` contains any closure captures or top-level bindings.
- Dimension notation: `tensor[d]` means a tensor with a single named dimension
  `d`, `tensor[]` is a scalar tensor, `tensor[d₁, d₂]` is a matrix.
- Tape entries from §1.0 of T0 are introduced via the canonical destructure
  form `let (a, a_tape) = copy(a)` as standard in the adjoint transformation.

---

## Example 1 — tape + linearity: `grad(λx. let y = mul(x, const(2, [])) in sum(y, 0))`

This is the canonical tape/linearity interaction. `mul` has a non-trivial
adjoint that needs both operand values on the tape; `const` has no input and
exercises the const-origin filter; `sum` has a shape-dependent adjoint; `grad`
at the outer level triggers the whole chain.

### Type of the body (without `grad`)

Context: `Δ = ∅`, `Γ = Γ_0, x : tensor[d₁]` (assume `d₁` is a single dimension
for concreteness).

```
let (y_final_type): tensor[]                     -- tensor[] is tensor[d \ 0] where d ≡ [d₁]
```

Derivation bottom-up:

1. `Γ_0, x : tensor[d₁] ⊢ x : tensor[d₁] ! ∅ ⊣ Γ_0` by **T-Var**.
2. `Γ_0 ⊢ const(2, []) : tensor[] ! ∅ ⊣ Γ_0` by **T-Const** (2 is a scalar).
   Wait — `const(2, [])` has shape `[]` (empty dimension list), so its type is
   `tensor[]`. But `mul` requires both operands to have equal dimension lists
   (shared `bar{d}` metavariable in **T-Mul**). `x : tensor[d₁]` and
   `const : tensor[]` do NOT have equal dimension lists. **The naïve
   derivation is ill-typed at `mul`.**

   This is a real finding about the example as originally stated in the plan.
   The example only type-checks if `const`'s shape matches `x`'s — so the
   honest version is `let y = mul(x, const(2, [d₁])) in sum(y, 0)`, where
   `const(2, [d₁])` is a tensor of shape `[d₁]` with every element equal to 2.

   Rewriting to the honest version and re-deriving:

3. `Γ_0, x : tensor[d₁] ⊢ x : tensor[d₁] ! ∅ ⊣ Γ_0` by **T-Var**.
4. `Γ_0 ⊢ const(2, [d₁]) : tensor[d₁] ! ∅ ⊣ Γ_0` by **T-Const**.
5. `Γ_0, x : tensor[d₁] ⊢ mul(x, const(2, [d₁])) : tensor[d₁] ! ∅ ⊣ Γ_0` by
   **T-Mul**, splitting the premises across `Γ_0, x : tensor[d₁]` (which
   delivers `x` to the first sub-derivation) and `Γ_0` (which delivers nothing
   linear to the second, just the const constant). Output context is `Γ_0`
   because both operands are consumed.
6. `Γ_0, x : tensor[d₁] ⊢ let y = mul(x, const(2, [d₁])) in sum(y, 0) : tensor[] ! ∅ ⊣ Γ_0`
   by **T-Let** threading `y : tensor[d₁]` into the inner derivation, then
   **T-Sum** reducing `tensor[d₁]` to `tensor[]` by removing axis 0.

So the body has type `tensor[d₁] → tensor[] ! ∅` as a function of `x`.

### Applying `grad`

Now wrap it in `grad`:

`Γ_0 ⊢ grad(λx : tensor[d₁]. <body>) : ? ! ∅ ⊣ Γ_0`

Apply **T-Grad**:

- Premise: `Δ ∪ {Diff}; Γ_0 ⊢ λx.<body> : tensor[d₁] → tensor[] ! ∅ ⊣ Γ_0`.
  This holds by **T-Abs** + the body derivation above (the empty effect row
  `∅` is trivially a subset of `DiffCompat`).
- Premise: the body uses `x` linearly. Inspect the body: `x` appears once, as
  the first operand to `mul`. Linearity holds.
- Premise: `∅ ⊆ DiffCompat`. Trivial.

Conclusion: `Γ_0 ⊢ grad(λx.<body>) : tensor[d₁] → tensor[] → tensor[d₁] ! ∅ ⊣ Γ_0`.

The result is a **two-argument function**: take the parameter `x : tensor[d₁]`
and the output-gradient seed `g_s : tensor[]` (a scalar cotangent), return the
parameter-gradient `tensor[d₁]`. This matches **T-Grad**'s two-argument form
and **E-Grad**'s `λx. λg_s. handle[Accum] ...` reduction.

### Reduction (operational)

By **E-Grad**:

```
grad(λx.<body>)  ↦  λx. λg_s. handle[{Accum}] (adjoint(<body>, x, g_s)) with h_accum
```

Where `h_accum = {accum(p, k) → k(update_origin_buffer(p))}`.

The reduced term is a two-argument function whose body is an `Accum`-handling
block around the adjoint transformation. `adjoint(<body>, x, g_s)` (per T0 §4
and §3.2) elaborates to:

```
-- forward pass with tape:
let c = const(2, [d₁]) in
let (x, x_tape) = copy(x) in
let (c, c_tape) = copy(c) in
let y = mul(x, c) in
let s = sum(y, 0) in
-- backward pass:
let grad_y = expand(g_s, 0, [d₁]) in
let (grad_y, g_for_x) = copy(grad_y) in
let grad_x = mul(g_for_x, c_tape) in
perform accum(loc(x), grad_x);
let grad_c = mul(grad_y, x_tape) in
perform accum(loc(c), grad_c);
s   -- final result forwarded, but the handler uses extract_grad(loc(x))
```

Every linear binding is consumed exactly once (verified in T0 §3.2 by the
Γ-walk). The `const`-origin filter in `h_accum` drops `grad_c` since
`loc(c)`'s origin is a `const` literal. The extracted gradient is the buffer
entry for `loc(x)`, which received one contribution (`grad_x`).

**Verdict:** well-typed, reduces cleanly, linear balance holds end-to-end.
The original plan phrasing `const(2, [])` needs to be corrected to
`const(2, [d₁])` for the dimension check in **T-Mul** to pass; flag this as
an update to the plan's example. ✓ (with correction)

---

## Example 2 — handler + linearity: `handle[Random] (let x = uniform_like(t, 0, 1) in add(x, x))`

This exercises two things: (a) that `add(x, x)` is ill-typed because `x` is
used twice, and (b) that the `copy`-based fix threads through cleanly.

### Naïve (ill-typed) version

Context: `Γ = Γ_0, t : tensor[d₁]`.

Attempt to derive `Γ ⊢ handle[{Random}] (let x = uniform_like(t, 0, 1) in add(x, x)) with h : ? ! ∅`
for some handler `h` that handles `Random`.

Walk the body `let x = uniform_like(t, 0, 1) in add(x, x)`:

1. `Γ_0, t : tensor[d₁] ⊢ t : tensor[d₁] ! ∅ ⊣ Γ_0` by **T-Var**.
2. `Γ_0, t : tensor[d₁] ⊢ uniform_like(t, 0, 1) : tensor[d₁] ! {Random} ⊣ Γ_0`
   by **T-UniformLike** (consuming `t`, producing a fresh tensor, adding
   `Random` to the row).

   (Note: **T-UniformLike** in my T2 does not consume the template — wait,
   let me re-read. The rule is:

   ```
   Δ; Γ₁ ⊢ e : tensor[d̄] ! ε ⊣ Γ₂
   ──────────────────────────────────────────────────
   Δ; Γ₁ ⊢ uniform_like(e, lo, hi) : tensor[d̄] ! ε ∪ {Random} ⊣ Γ₂
   ```

   The output context `Γ₂` is the same one returned by the sub-derivation
   for `e`. Since `t` is a `T-Var` invocation, the sub-derivation returns
   `Γ₀` (consuming `t`). So `uniform_like` implicitly consumes the template.
   This is consistent with T0 §2.6's worked derivation: `uniform_like(x, 0, 1)`
   consumes `x` and the rejection example requires an explicit `copy(x)`
   beforehand to preserve `x` for subsequent use.)

3. `Γ_0, x : tensor[d₁] ⊢ x : tensor[d₁] ! ∅ ⊣ Γ_0` by **T-Var** — but
   this consumes `x` from Γ.
4. Second operand of `add`: again `x`, but `x` has already been consumed.
   **T-Var** fails: `x` is not in the context.

The attempt fails at step 4. The body is ill-typed. ✓

### Copy-fixed version

Rewrite:

```
handle[{Random}] (
  let x = uniform_like(t, 0, 1) in
  let (x, x2) = copy(x) in
  add(x, x2)
) with h
```

Walk:

1. `Γ_0, t ⊢ uniform_like(t, 0, 1) : tensor[d₁] ! {Random} ⊣ Γ_0` (consumes `t`).
2. `Γ_0, x : tensor[d₁] ⊢ copy(x) : tensor[d₁] ⊗ tensor[d₁] ! ∅ ⊣ Γ_0` by
   **T-Copy** (consuming `x`).
3. Destructure: `Γ_0, x : tensor[d₁], x2 : tensor[d₁] ⊢ add(x, x2) : tensor[d₁] ! ∅ ⊣ Γ_0`
   by **T-Add**, which consumes both operands.
4. **T-Let** + **T-LetPair** threading: the whole body is
   `tensor[d₁] ! {Random}` with output context `Γ_0`.
5. **T-Handle**: the handler handles `{Random}`, so the residual row is
   `∅`. The result type is `tensor[d₁]`, output context `Γ_0`.

All linear bindings consumed exactly once. ✓

**Finding:** the `handle`-then-`let` nesting in the naïve version has a
subtle point: the `uniform_like` call sits inside the handler body and
inside a `let`, so the **T-Handle** rule sees the whole `let` expression
as the body `e` in its premise. This is fine because **T-Handle**'s effect
row tracking is lexical (it sees every effect produced by any subterm of
`e`). No issue with the rule.

---

## Example 3 — vmap outside grad: `vmap(grad(λx. f x))` for `f : tensor[d] → tensor[] ! {Resource}`

This exercises the interaction between `grad` (AD) and `vmap` (vectorization)
in the **outer form**: `vmap` wraps a complete `grad` call. The inner form
(`grad` over a body containing `vmap`) is out of scope per T0 §0.

Assume `Γ_0 ⊢ f : tensor[d] → tensor[] ! {Resource} ⊣ Γ_0` as a top-level
binding. The updated **T-Grad** rule requires its argument to be a literal
abstraction $\lambda x.\, e$, so we cannot write `grad(f)` directly — we
must eta-expand to `grad(λx : tensor[d]. f x)`. The master plan's original
wording `vmap(grad(f))` is therefore rewritten here as
`vmap(grad(λx : tensor[d]. f x))`, which is the well-typed form under Phase 1
rules.

### Step 1: type of `grad(λx : tensor[d]. f x)`

Apply **T-Grad**:

- Premise: `Δ ∪ {Diff}; Γ_0, x : tensor[d] ⊢ f x : tensor[] ! {Resource} ⊣ Γ_0`.
  Derivation: **T-Var** on `f` returns `Γ_0, x : tensor[d] ⊢ f : tensor[d] → tensor[] ! {Resource} ⊣ Γ_0, x : tensor[d]` (`f` is non-linear, or bound at top-level and not consumed from `Γ_0`; assume the standard treatment). Then **T-App** on `f` applied to `x`: the first premise delivers `f` leaving `x` in context, the second premise delivers `x` consuming it. Result: `tensor[] ! {Resource}` with output context `Γ_0`.
- Premise: `{Resource} ⊆ DiffCompat = {Resource, Accum}`. Holds.
- `x` is consumed exactly once (by the application to `f`), so the linearity
  side-condition (enforced by the `⊣ Γ_0` output context) holds.

Conclusion: `Γ_0 ⊢ grad(λx : tensor[d]. f x) : tensor[d] → tensor[] → tensor[d] ! {Resource} ⊣ Γ_0`.

### Step 2: type of `vmap(grad(λx. f x))`

Apply **T-Vmap**:

- Premise: `grad(λx. f x)` has type
  `tensor[d] → (tensor[] → tensor[d] ! {Resource})`. This is a
  right-associative function type: the outer arrow is a pure function that
  returns another function. The outer arrow carries effect row `∅` because
  **T-Abs** gives `τ₁ → τ₂ ! ε` where `ε` is the body's effect row, and the
  body here is itself an abstraction (introduced by the reduced `grad` form
  `λx. λg_s. ...`). Nested abstractions have `∅` at every outer arrow; the
  `{Resource}` effect lives at the innermost arrow.

  For **T-Vmap**'s premise shape `f : τ₁ → τ₂ ! ε`, we take
  `τ₁ = tensor[d]`, `τ₂ = tensor[] → tensor[d] ! {Resource}`, and `ε = ∅`.

- Premise: `batch` fresh.

Apply **T-Vmap**:

```
addDim(batch, tensor[d]) = tensor[batch, d]
addDim(batch, tensor[] → tensor[d] ! {Resource}) =
    addDim(batch, tensor[]) → addDim(batch, tensor[d]) ! {Resource}
  = tensor[batch] → tensor[batch, d] ! {Resource}
```

So: `Γ_0 ⊢ vmap(grad(λx. f x)) : tensor[batch, d] → tensor[batch] → tensor[batch, d] ! {Resource} ⊣ Γ_0`.

**Reading:** takes a batched parameter `tensor[batch, d]` and a batched seed
`tensor[batch]` (one scalar cotangent per batch element), and returns a
batched gradient `tensor[batch, d]` (per-example gradients). This is the
classic JAX `vmap(grad(f))` pattern for per-example gradients, and Phase 1's
T-Grad restriction forces the explicit eta-expansion.

**Verdict:** well-typed. Dimensions thread through `addDim` correctly at
every layer of the function type. Effects (`Resource`) are preserved through
both `grad` and `vmap` applications. The eta-expansion is a Phase 1 wart
from restricting `grad` to literal abstractions; a future phase could lift
the restriction by proving the linear-use property syntactically for
variables bound to known linear functions. ✓

---

## Example 4 — effect row polymorphism: `vmap(dropout)` over a batch

The master plan's original version uses `map(dropout, xs)` for `map : (α → β ! ε) → List[α] → List[β] ! ε`. LaCaDiLE's Phase 1 core calculus does not have `List` or parametric polymorphism, so the literal `map` example does not type-check. We substitute `vmap(dropout)` as the closest in-scope analog: `vmap` is the core calculus's equivalent of mapping-with-effects over a fresh dimension, and it preserves the effect row by construction (**T-Vmap**'s conclusion copies `ε` unchanged).

Assume `Γ_0 ⊢ dropout : tensor[d] → tensor[d] ! {Random} ⊣ Γ_0` (a top-level
library function that zeros out elements with some probability, using the
`Random` effect to sample the mask).

Apply **T-Vmap**:

- Premise: `Γ_0 ⊢ dropout : tensor[d] → tensor[d] ! {Random} ⊣ Γ_0`. Given.
- Premise: `batch` fresh.

Conclusion:

```
addDim(batch, tensor[d]) = tensor[batch, d]
Γ_0 ⊢ vmap(dropout) : tensor[batch, d] → tensor[batch, d] ! {Random} ⊣ Γ_0.
```

The `Random` effect propagates unchanged from the inner function type to the
outer lifted type. This is the "effect row polymorphism" behavior the master
plan wanted to demonstrate, restricted to `vmap`'s monomorphic effect-row
signature. Full higher-rank row polymorphism (with a row variable `ρ` in the
signature of `map`) requires polymorphic type variables in the calculus,
which Phase 1 intentionally does not introduce — the master plan notes this
as a Phase 2+ extension under "unit tests for row polymorphism."

**Finding:** the example demonstrates effect propagation through a lifted
function, but does **not** demonstrate full row polymorphism (row variables
in user-defined functions). Flag that Phase 1 T2 does not yet support row
variables in function signatures, and T4's example 4 is a weakened version
of the master plan's intended check. Row-polymorphic `map` should be
reintroduced when Phase 2 adds polymorphism.

---

## Findings summary

| # | Example | Verdict | Issue |
|---|---|---|---|
| 1 | `grad(λx. let y = mul(x, const(2, [])) in sum(y, 0))` | well-typed with correction | The `const(2, [])` needs to be `const(2, [d₁])` for dimension compatibility in `T-Mul`. Also uses `expand` in its reduced form, relying on **insertion** semantics for the dimension list (not substitution); T-Expand's notation `ins(\bar d, i, k)` makes this explicit. |
| 2 | `handle[Random] (let x = uniform_like(t, 0, 1) in add(x, x))` | ill-typed (as expected) | Naïve version fails `T-Var` on second use of `x`. Copy-fixed version type-checks cleanly. |
| 3 | `vmap(grad(λx. f x))` for `f : tensor[d] → tensor[] ! {Resource}` | well-typed (after eta-expansion) | Phase 1's T-Grad restriction to literal abstractions forces eta-expansion. Dimensions thread through `addDim` at every arrow; `Resource` effect preserved. |
| 4 | `vmap(dropout)` (substituted for `map(dropout, xs)`) | well-typed, but weakened | Phase 1 core calculus lacks row variables in function signatures. Full `map` + row polymorphism is Phase 2+. |

**Wave 2 red-team found two rule bugs** that led to corrections during Wave 2:
T-Expand's dimension-list notation was corrected to `\mathsf{ins}(\bar d, i, k)`
(insertion, required for `sum`/`expand` to be mutual adjoints), and T-Grad was
restricted to literal abstractions so the linear-use side-condition has real
force. Both corrections are reflected in the final T2 figures. One plan
example (#1) needs a dimension fix; example (#3) needs eta-expansion; example
(#4) needs to be explicitly weakened in the paper narrative to match Phase
1's monomorphic calculus.

## Open questions surfaced by the walks

1. **Does `T-UniformLike` consume the template?** My walk of Example 2
   assumed yes (consistent with T0 §2.6). The rule in `typing.tex` threads
   `Γ₁ → Γ₂` from the sub-derivation of `e`, which for `T-Var(t)` returns
   `Γ₀` (consuming `t`). So yes, implicit consumption. The rule is correct
   but the behavior isn't obvious from the rule alone — Wave 2 red team
   should flag if this is confusing. Consider adding an explicit comment.

2. **`grad(f)` where `f` is a top-level name rather than a literal lambda.**
   Example 3 uses `f` as a top-level binding. My derivation assumes
   `grad(f)` type-checks by taking `f`'s declared type as given. But **T-Grad**'s
   premise is `Δ ∪ {Diff}; Γ ⊢ f : ... ! ε`, which requires the typing
   derivation for `f` to succeed with `Diff` in the capability context.
   For a top-level binding with no `grad` in its definition, adding `Diff`
   is a no-op — nothing uses it. This works but deserves a Phase 2 lemma:
   "capability weakening is admissible" (adding a capability to `Δ` does
   not invalidate a derivation).

3. **`mul`'s dimension-equality premise** (shared `bar{d}` metavariable) is
   enforced purely by metavariable unification, not by an explicit side
   condition. This is standard in inference-rule notation but should be
   explicitly mechanized in Lean (T7) as a premise `d̄₁ = d̄₂`. The paper
   rule can stay as written.

4. **Effect row union vs set union.** In my rules I wrote `ε₁ ∪ ε₂` for
   combining effect rows. Effect rows with row variables are not quite sets —
   they are (finite multiset, row variable) pairs. For the closed-row cases
   in these examples, set union is correct. The open-row case (row variable
   `ρ`) is deferred to Phase 2.
