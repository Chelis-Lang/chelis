# spec/04-type-system.md — Chelis Type System

**Status:** v0.3
**Scope:** Phase 0 type system plus the shipped Phase 2a effect subset: ADTs,
Hindley-Milner inference, numeric precision types, named tensor dimensions, fitness
scoring, annotated checked Deep, and bounded `Random` / `Resource(Device)` effect
checking.

**Executable completeness note:** this document includes type-level shapes that are
larger than the currently practical tensor-only execution story. In particular, the
remaining Phase `3c` / `3d` / `3g` work is what turns scalar/string/collection ideas
into a real end-to-end AI-programming surface rather than only syntax or type examples.

---

## 0. Checked Deep Contract

The type checker is the first pass that upgrades raw Deep into the downstream
compiler-facing representation.

- `check_phase0e_program(...)` returns a `CheckedProgram`, not just a success/failure bit
- a `CheckedProgram` carries annotated Deep, with `type` metadata written onto the
  returned tree
- lowering, evaluation, effect checking, and CLI build/eval paths consume that
  annotated tree rather than the original raw Deep

This contract matters for Phase 2a because effect inference/checking runs after HM type
inference on the same annotated tree.

---

## 1. Type Representation

Types are represented as Deep AST nodes using the `t-*` tag family.

### 1.1 Primitive Types

```scheme
(t-prim {} f32)       ;; 32-bit float
(t-prim {} f64)       ;; 64-bit float
(t-prim {} f16)       ;; 16-bit float
(t-prim {} bf16)      ;; bfloat16
(t-prim {} f8e4m3)    ;; 8-bit float (E4M3 format)
(t-prim {} int8)      ;; 8-bit integer
(t-prim {} int32)     ;; 32-bit integer
(t-prim {} int64)     ;; 64-bit integer
(t-prim {} bool)      ;; boolean
(t-prim {} string)    ;; string type exists in the grammar/type layer; practical first-class runtime support is a remaining Phase 3 item
```

### 1.2 Function Types

```scheme
(t-fn {} arg₁ arg₂ ... ret)
```

Last child is the return type. All preceding children are argument types. Multi-argument functions are flat (not curried):

```scheme
;; f : (f32, f32) -> f32
(t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32))

;; g : tensor[batch, hidden, f32] -> tensor[batch, hidden, f32]
(t-fn {}
  (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))
  (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32)))
```

### 1.3 Tensor Types

```scheme
(t-tensor {} dim₁ dim₂ ... precision)
```

Last child is the precision type (must be a numeric `t-prim`). All preceding children are dimension expressions (`d-name`, `d-var`, or `d-lit`).

```scheme
;; A 2D tensor: batch × hidden, f32
(t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))

;; A polymorphic tensor: any dims a × b, bf16
(t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} bf16))

;; A tensor with known literal size: 512 × 768, f32
(t-tensor {} (d-lit {} 512) (d-lit {} 768) (t-prim {} f32))

;; Rank-0 tensor (scalar on device): just precision
(t-tensor {} (t-prim {} f32))
```

### 1.4 ADT Types

```scheme
;; Option f32
(t-adt {} Option (t-prim {} f32))

;; List (tensor[batch, f32]) -- illustrative future/planned collection typing surface
(t-adt {} List (t-tensor {} (d-name {} batch) (t-prim {} f32)))

;; No type arguments
(t-adt {} Activation)
```

### 1.5 Other Types

```scheme
(t-var {} a)                                  ;; type variable (for inference)
(t-var {} _)                                  ;; explicit "infer this"
(t-unit {})                                   ;; unit type (empty tuple)
(t-tuple {} (t-prim {} f32) (t-prim {} f32))  ;; tuple type
```

---

## 2. Algebraic Data Types

### 2.1 Declaration

```scheme
(deftype {} Option (a)
  (variant {} Some (t-var {} a))
  (variant {} None))
```

Type parameters are listed after the name. Variants may have positional type arguments or named fields.

### 2.2 Record Variants

```scheme
(deftype {} Optimizer ()
  (variant {} Adam
    (field {} lr (t-prim {} f32))
    (field {} betas (t-tuple {} (t-prim {} f32) (t-prim {} f32)))
    (field {} eps (t-prim {} f32)))
  (variant {} SGD
    (field {} lr (t-prim {} f32))
    (field {} momentum (t-prim {} f32))))
```

### 2.3 Recursive Types

Recursive references are by name. No explicit `mu` type needed:

```scheme
(deftype {} List (a)
  (variant {} Cons
    (field {} head (t-var {} a))
    (field {} tail (t-adt {} List (t-var {} a))))
  (variant {} Nil))
```

The recursive `List` example shows the intended type-level shape of collection support.
It should not be read as a claim that the full practical collection runtime and
iteration surface are already shipped today.

### 2.4 Exhaustive Pattern Matching

The type checker verifies that `match` expressions cover all variants. Missing variants are a type error, not a warning.

---

## 3. Hindley-Milner Inference

### 2.5 Planned Remaining Phase 3 Type-Surface Extensions

The remaining language-completeness work is expected to make the following type-level
surfaces practical and executable:

- first-class non-tensor `Int`, `Float`, and `Bool` values
- first-class `string` values with ordinary operations
- `Option[T]` as the ergonomic failure-returning surface for parse/lookups
- `List[T]` and `Dict[K, V]` as practical collection types
- explicit bridges between collection values and tensor values

Those items should be treated as Phase 3 implementation targets, not as a claim that the
current evaluator/backends already provide the full runtime behavior implied by the type
examples above.

### 3.1 Algorithm

Standard Algorithm W with extensions for tensor types. The flow:

1. **Constraint generation:** Walk the typed Deep AST. At each node, generate type equations between the expected type and the actual type.
2. **Unification:** Solve the constraint set. Unification handles type variables, function types, tensor types (with dimension unification), and ADT types.
3. **Generalization:** At `let` boundaries, generalize unconstrained type variables to produce polymorphic types.
4. **Annotation checking:** Where the programmer/agent provided `type` metadata, check that the inferred type is compatible with the annotation.

### 3.2 Inference Rules

Standard notation: Γ ⊢ e : τ means "in environment Γ, expression e has type τ."

**Variable:**
```
    x : σ ∈ Γ      τ = instantiate(σ)
    ─────────────────────────────────
           Γ ⊢ (var {} x) : τ
```

**Literal:**
```
    v has primitive type τ
    ──────────────────────────────
    Γ ⊢ (lit {type: τ} v) : τ
```

**Application:**
```
    Γ ⊢ f : (τ₁, τ₂, ..., τₙ) → τᵣ
    Γ ⊢ aᵢ : τᵢ   for each i ∈ 1..n
    ─────────────────────────────────────
    Γ ⊢ (app {} f a₁ a₂ ... aₙ) : τᵣ
```

**Lambda:**
```
    Γ, x₁:τ₁, ..., xₙ:τₙ ⊢ body : τᵣ
    ──────────────────────────────────────────────
    Γ ⊢ (fn {} (params x₁ ... xₙ) body) : (τ₁, ..., τₙ) → τᵣ
```

**Let (with polymorphism):**
```
    Γ ⊢ e₁ : τ₁
    σ₁ = generalize(Γ, τ₁)
    Γ, x₁:σ₁ ⊢ ... (remaining binds and body) ... : τ
    ───────────────────────────────────────────────────
    Γ ⊢ (let {} (bind x₁ e₁ ...) body) : τ
```

**If:**
```
    Γ ⊢ c : bool     Γ ⊢ a : τ     Γ ⊢ b : τ
    ────────────────────────────────────────────
           Γ ⊢ (if {} c a b) : τ
```

**Match:**
```
    Γ ⊢ e : τₛ
    For each (arm {} pᵢ () bᵢ):
        Γ, bindings(pᵢ, τₛ) ⊢ bᵢ : τᵣ
    patterns {pᵢ} are exhaustive over τₛ
    ──────────────────────────────────────
    Γ ⊢ (match {} e arm₁ ... armₙ) : τᵣ
```

**Pipe:**
```
    Γ ⊢ e₁ : τ₁
    Γ ⊢ e₂ : τ₁ → τ₂
    Γ ⊢ e₃ : τ₂ → τ₃
    ──────────────────────────────────────
    Γ ⊢ (pipe {} e₁ e₂ e₃) : τ₃
```

**Tuple:**
```
    Γ ⊢ eᵢ : τᵢ   for each i
    ─────────────────────────────────────
    Γ ⊢ (tuple {} e₁ ... eₙ) : (τ₁, ..., τₙ)
```

**Tuple-get:**
```
    Γ ⊢ e : (τ₁, ..., τₙ)     0 ≤ k < n
    ──────────────────────────────────────
    Γ ⊢ (tuple-get {} e k) : τₖ
```

**grad (shipped 2db result shape):**
If `f` has scalar floating output, `grad(f)` returns gradients only.
The gradient payload shape is:

- one differentiable target => that gradient type directly
- multiple differentiable targets => flat `t-tuple` in target order
- explicit `wrt` on a non-differentiable parameter => type error

The forward value is not bundled into the `grad(...)` result in the shipped language
surface.

---

## 4. Tensor Type Algebra

### 4.1 Dimension Matching

Tensor operations require strict dimension matching. Two dimension lists are compatible if they unify element-wise:

- `(d-name {} batch)` unifies with `(d-name {} batch)` — same name.
- `(d-name {} batch)` does NOT unify with `(d-name {} seq)` — different names → **type error**.
- `(d-var {} a)` unifies with any dimension — binds `a` to that dimension.
- `(d-lit {} 512)` unifies with `(d-lit {} 512)` — same literal.
- `(d-lit {} 512)` does NOT unify with `(d-lit {} 768)` — different sizes → **type error**.
- `(d-name {} batch)` unifies with `(d-var {} a)` — binds `a = batch`.

### 4.2 No Broadcasting

Chelis does NOT support implicit broadcasting. All rank and dimension manipulation must be explicit via `expand`, `reshape`, `permute`.

```scheme
;; WRONG: dimensions don't match
;; tensor[batch, hidden, f32] + tensor[hidden, f32]  →  TYPE ERROR

;; CORRECT: explicit expand
;; tensor[batch, hidden, f32] + expand(tensor[hidden, f32], [batch, hidden])
```

Rationale: Broadcasting masks fatal dimension errors in AI-generated code. Named dimensions + no broadcasting means the type checker catches transposition bugs, broadcasting bugs, and shape mismatches at compile time.

### 4.3 How RISC Primitives Transform Tensor Types

| Operation | Input type(s) | Output type | Dimension rule |
|---|---|---|---|
| `add`, `mul`, `max_elem` | `tensor[D, p]`, `tensor[D, p]` | `tensor[D, p]` | Dimensions must match exactly |
| `neg`, `exp`, `log`, `sin`, `sqrt` | `tensor[D, p]` | `tensor[D, p]` | Dimensions preserved |
| `sum(x, axis=k)` | `tensor[d₁,...,dₙ, p]` | `tensor[d₁,...,d_{k-1},d_{k+1},...,dₙ, p]` | Remove dimension at axis k |
| `max_reduce(x, axis=k)` | same as sum | same as sum | same as sum |
| `reshape(x, shape)` | `tensor[D_old, p]` | `tensor[D_new, p]` | Product of dims must match. New dims are `d-lit` or `d-name` (user-specified) |
| `permute(x, axes)` | `tensor[d₁,...,dₙ, p]` | `tensor[d_{axes[0]},...,d_{axes[n-1]}, p]` | Reorder dimensions |
| `expand(x, shape)` | `tensor[D_small, p]` | `tensor[D_large, p]` | Add dimensions. Each new dim is explicit. |
| `pad(x, ...)` | `tensor[D, p]` | `tensor[D', p]` | Padded dimensions get new sizes (d-lit) |
| `cast(x, new_p)` | `tensor[D, p]` | `tensor[D, new_p]` | Dimensions preserved, precision changes |

### 4.4 Dimension Polymorphism

Functions can be generic over dimensions using dimension variables:

```scheme
;; In Surf:
;; def transpose[a, b](x: tensor[a, b, f32]): tensor[b, a, f32]

(defsig {} transpose
  (t-fn {}
    (t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} f32))
    (t-tensor {} (d-var {} b) (d-var {} a) (t-prim {} f32))))
```

At call sites, dimension variables are instantiated by unification — no explicit dimension arguments needed:

```scheme
;; transpose(my_matrix)
;; If my_matrix : tensor[batch, seq, f32]
;; Then a = batch, b = seq
;; Result : tensor[seq, batch, f32]
```

### 4.5 The Wildcard Dimension

`(d-name {} *)` represents an unknown/dynamic dimension. Produced by operations where the compiler cannot statically determine the dimension:

- Concatenation along an axis: `concat(tensor[batch, seq1, f32], tensor[batch, seq2, f32], axis=1)` → `tensor[batch, *, f32]`
- Data loading with dynamic shapes
- Results of control flow where branches have different known dimensions

A wildcard dimension unifies with any other dimension (like a variable) but is NOT generalized — it's a permanent "I don't know." To restore named-dimension checking after a wildcard, use an explicit annotation.

---

## 5. Precision Type Rules

### 5.1 No Implicit Promotion

All operands of an arithmetic operation must have the same precision. Mixed precision is a type error:

```
add(tensor[D, f32], tensor[D, bf16])  →  TYPE ERROR
    Expected: tensor[D, f32]
    Got:      tensor[D, bf16]
    Fix:      cast(y, f32) or cast(x, bf16)
```

### 5.2 Cast

`cast` changes precision. Dimensions are preserved:

```
cast(x: tensor[D, f32], bf16) : tensor[D, bf16]
cast(x: tensor[D, int32], f32) : tensor[D, f32]
```

Cast is always explicit. The compiler never inserts implicit casts.

### 5.3 Literal Types

Integer literals default to `int32`. Float literals default to `f32`. These defaults can be overridden by context (annotation on the `lit` node) or by explicit `cast`.

### 5.4 Precision Compatibility Table

Operations accept same-precision operands only. The table of valid combinations:

| Operation type | Valid precisions |
|---|---|
| Arithmetic (add, mul, sub, div) | f32, f64, f16, bf16, f8e4m3, int32, int64 (all same) |
| Comparison (cmplt, eq) | any numeric (same precision) → bool |
| Logical (and, or, not) | bool only |
| Transcendental (exp, log, sin, sqrt) | f32, f64, f16, bf16 only (not integer) |

---

## 6. Fitness Scoring

The compiler produces a fitness report for every compilation attempt. This is the training signal for AI agents.

### 6.1 Score Components

| Component | Weight | Measurement |
|---|---|---|
| Parse success | 0.1 | 1.0 if parses, 0.0 if not |
| Structural validity | 0.1 | Fraction of nodes with valid tags and arity |
| Name resolution | 0.2 | Fraction of `(var {} name)` references that resolve |
| Type check | 0.6 | Fraction of sub-expressions that unify successfully |

**Total score:** weighted sum, 0.0 to 1.0.

### 6.2 Partial Type Inference

When full type checking fails, the compiler still infers types for as many sub-expressions as possible. Failed nodes get error metadata:

```scheme
(app {type: error, error: "precision_mismatch", expected: (t-prim {} f32), got: (t-prim {} bf16)}
  (var {type: (t-fn {} (t-tensor {} ...) (t-tensor {} ...) (t-tensor {} ...))} add)
  (var {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} x)
  (var {type: (t-tensor {} (d-name {} batch) (t-prim {} bf16))} y))
```

The agent can read the annotated AST and see exactly which nodes type-checked and which didn't.

### 6.3 Repair Suggestions

For common error patterns, the compiler produces structured repair suggestions:

| Error | Suggestion |
|---|---|
| Precision mismatch | `"Insert cast: (cast {} y (t-prim {} f32))"` |
| Dimension mismatch | `"Dimensions [batch, seq] vs [batch, hidden]. Transpose? Reshape?"` |
| Missing match arm | `"Missing variant: None. Add (arm {} (pat-ctor {} None) () ...)"` |
| Unknown variable | `"Did you mean: [relu, reshape, reduce]"` (Levenshtein) |
| Wrong arity | `"f expects 3 args, got 2"` |
| Non-exhaustive let | `"Unbound variable x in body. Missing bind?"` |

Suggestions are structured data in the fitness report JSON, not just strings.

### 6.4 Fitness Report Format

```json
{
  "score": 0.73,
  "components": {
    "parse": 1.0,
    "structure": 1.0,
    "names": 0.85,
    "types": 0.62
  },
  "errors": [
    {
      "kind": "precision_mismatch",
      "loc": {"line": 12, "col": 5},
      "expected": "tensor[batch, hidden, f32]",
      "got": "tensor[batch, hidden, bf16]",
      "suggestions": ["Insert cast(y, f32)"],
      "severity": 0.8
    }
  ],
  "typed_ast": "... (annotated Deep with types on every node) ...",
  "unresolved_names": ["typo_var"],
  "untyped_nodes": 4,
  "total_nodes": 31
}
```

---

## 7. Effects (Phase 2a Shipped Subset)

### 7.1 Effect Model

The long-term effect design still points toward row-polymorphic effect inference, but
the shipped Phase 2a subset is narrower and explicit about its boundaries.

Built-in effect vocabulary in the type layer:

- `Random` -- stochasticity introduced by compiler-known operations such as `dropout`
- `Accum` -- internal-only hook for associative gradient accumulation
- `IO` -- host-side debugging/logging effects such as `print` and `debug`
- `Resource(Device)` -- allocation / placement region on a concrete device

Settled Phase 2a design decisions:

- `Diff` is a compiler capability, not a user-visible boundary effect
- `Accum` is internal-only in v1; users do not handle it directly
- `Random` and `Resource(Device)` are the real Phase 2a boundary effects
- `IO` is a shipped Phase 3 host-side effect rather than a Phase 2a handler boundary

Current shipped inference/checking behavior:

- effect inference runs after HM type inference on the annotated Deep returned by the
  type checker
- a function's inferred effect set is the union of the effects of compiler-known
  operations in its body
- `dropout(x, rate)` is the concrete shipped `Random` source
- `print(x)` and `debug(x)` are the concrete shipped `IO` sources
- `with seed(seed) { ... }` handles `Random`
- `with device(device) { ... }` marks a resource region that is validated against the
  chosen build target
- declared `Resource("...")` annotations are accepted on `t-fn` type expressions, but
  the current checker does not yet synthesize `Resource(Device)` onto checked `fn`
  metadata the way it does for inferred `Random`
- unhandled top-level `Random` is a check error with repair guidance
- top-level `IO` is currently permitted for debugging/logging programs

The current shipped checker does **not** yet claim the full Phase 2 design:

- no full user-visible `Diff` effect checking
- no user-visible `Accum` inference/handling
- no general row-polymorphic higher-order effect surface promised as shipped behavior

### 7.2 Type Representation

Declared function types with effects use `eff` metadata on `t-fn`:

```scheme
;; f : tensor[D, f32] -> tensor[D, f32] ! {Random, Resource("gpu:0")}
(t-fn {eff: (effects {} random (resource {} "gpu:0"))}
  (t-tensor {} (d-var {} d) (t-prim {} f32))
  (t-tensor {} (d-var {} d) (t-prim {} f32)))
```

Checked function bodies may also carry inferred effect metadata:

```scheme
(fn {type: (t-fn {} ...), effects: (effects {} random)} (params {} x) body)
```

In the shipped subset, this inferred `effects` metadata is used for effect information
the checker actually synthesizes today, notably `Random`. Resource regions are enforced
at the handler/build boundary, but are not yet written back onto checked `fn` metadata.

Effect annotations remain optional in Surf and Deep. They are accepted as part of the
surface syntax even where the current checker only implements a bounded subset of the
eventual design.

### 7.3 Phase 0 Extension Point

The `eff` meta key on `t-fn` nodes and the `effects` meta key on checked `fn` nodes are
the active Phase 2a extension points.

---

## 8. Linearity (Phase 2b)

### 8.1 Model

Phase 2b uses lightweight uniqueness, not a Rust-style ownership-and-lifetimes
system. Tensor values are consume-by-default. A consuming use makes the binding dead
unless the program inserted `copy(...)` before that use. Borrowing with `&x` provides
temporary read-only access without consumption.

### 8.2 Type Representation

```scheme
;; Linear tensor (default)
(t-tensor {lin: once} (d-name {} batch) (t-prim {} f32))

;; Borrowed tensor (read-only reference)
(t-tensor {lin: borrow} (d-name {} batch) (t-prim {} f32))
```

The `lin` metadata key remains a representation hook, but the shipped Phase 2b user
surface is expression-based:

- `copy(x)` is the only explicit duplication form
- `&x` is only valid as a direct call argument
- passing a tensor to a normal call consumes it unless the caller wrote `&x`
- borrows cannot be stored, returned, rebound for later use, or captured by closures

Linearity is checked after effect inference, before lowering:

`parse -> desugar -> type infer/check -> effect infer/check -> linearity check -> lower`

Linearity is expected to compose with effects, especially `Resource(Device)`: the
effect system tracks where a tensor lives, while linearity tracks when it is consumed.
That gives the compiler a stronger basis for safe in-place buffer reuse.

### 8.3 Static Rules

- A bare linear binding is consumed on use.
- `copy(x)` reads `x` without consuming it and yields a fresh tensor value.
- Pattern matching on a tuple or other value carrying tensor payloads consumes the
  scrutinee; any tensor payloads bound by the pattern become the new live bindings.
- Creating a closure that captures a tensor consumes that outer binding at closure
  creation time.
- Diagnostics report the consume site and suggest inserting `copy(...)` when reuse was
  intended.
