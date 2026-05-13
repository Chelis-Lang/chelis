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

- `check_ir_program(...)` returns a `CheckedProgram`, not just a success/failure bit
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

The active numeric primitive set is exactly nine names:

```scheme
(t-prim {} f32)       ;; 32-bit float
(t-prim {} f64)       ;; 64-bit float
(t-prim {} bf16)      ;; bfloat16
(t-prim {} f16)       ;; 16-bit float (IEEE 754 binary16)
(t-prim {} int8)      ;; 8-bit signed integer
(t-prim {} int16)     ;; 16-bit signed integer
(t-prim {} int32)     ;; 32-bit signed integer
(t-prim {} int64)     ;; 64-bit signed integer
(t-prim {} bool)      ;; boolean
(t-prim {} string)    ;; string type exists in the grammar/type layer; practical first-class runtime support is a remaining Phase 3 item
```

This is the **active** numeric dtype list. Code, tests, examples, and stdlib
signatures referenced from any active spec section must resolve to one of these
names (or to a documented deferred name in §1.1.1).

#### 1.1.1 Deferred Numeric Primitives

The following primitive name is reserved in the spec but **not active** in the
current dtype build-out cycle. It is documented here rather than silently
dropped so producers do not assume it has gone away.

- `f8e4m3` — 8-bit float (E4M3 format). Deferred. **Rationale:** no current
  Chelis backend implements f8e4m3; revisit when a concrete backend (HIP, C, or
  Metal) gains native E4M3 support and a corresponding evaluator round-trip
  representation. Until then, `(t-prim {} f8e4m3)` is rejected by the type
  checker with a diagnostic pointing at this section. `cast(x, f8e4m3)` is also
  rejected.

#### 1.1.2 Out-of-Scope: Unsigned Integer Types

Unsigned integer types (`u8`, `u16`, `u32`, `u64`, or any `uint*` spelling) are
**out of scope for this cycle** and are not part of the active or deferred
numeric primitive set. **Rationale:** no current customer use case justifies
the implementation surface (separate signed/unsigned arithmetic, comparison,
overflow, AD adjoints, and backend dispatch). The documented workaround is
to cast to a signed integer type (typically `int32` or `int64`) at the
boundary where unsigned data enters the program. If a future cycle adds
unsigned types, this section must be revised at the same time as the active
list above.

#### 1.1.3 Per-Backend Dtype Support Matrix

The active primitive set in §1.1 is the **language-level** dtype contract: a
program that mentions one of the nine active primitives is well-typed in
every Chelis pass that does not select a backend (parser, type checker, IR
evaluator). Backend code generation is a separate surface; not every backend
admits every active dtype. This sub-section is the authoritative per-backend
matrix. Any "Metal supports X" or "C backend supports Y" claim elsewhere in
the spec or in user-facing docs must resolve to a cell in this table.

| dtype  | C backend                               | HIP backend                | Metal backend                                | Evaluator |
|--------|-----------------------------------------|----------------------------|----------------------------------------------|-----------|
| f32    | admitted                                | admitted                   | admitted                                     | admitted  |
| f64    | admitted                                | admitted                   | **rejected (hardware)**                      | admitted  |
| bf16   | rejected (deferred)                     | admitted (matmul + load/store via `hipblasGemmEx`) | admitted on Apple7+ (M3 or later) | admitted |
| f16    | rejected (deferred)                     | admitted (matmul + load/store via `hipblasGemmEx`) | admitted                          | admitted |
| int8   | admitted                                | admitted                   | admitted                                     | admitted  |
| int16  | admitted                                | admitted                   | admitted                                     | admitted  |
| int32  | admitted                                | admitted                   | admitted                                     | admitted  |
| int64  | admitted                                | admitted                   | admitted                                     | admitted  |
| bool   | admitted                                | admitted                   | admitted                                     | admitted  |

Cell semantics:

- **admitted** — the backend codegen accepts the dtype on every shipped op
  surface that admits any dtype, and emits correct code. Per-op restrictions
  (e.g. matmul accumulator dispatch per §5.7.1, transcendental ops are
  float-only per §5.4) apply uniformly across backends and are not encoded
  in this matrix.
- **rejected (deferred)** — the backend rejects the dtype at codegen with a
  diagnostic naming the deferral. The dtype is admitted at the language
  level and on at least one other backend; the gap is implementation-side
  and the spec will lift the rejection in a future cycle.
- **rejected (hardware)** — the backend rejects the dtype at codegen with a
  diagnostic naming the hardware constraint. There is no planned lift,
  because no lift is achievable on the target hardware without software
  emulation, which is explicitly out of scope (see the f64-on-Metal entry
  below).

##### f64 on Metal: hard-rejected (hardware rationale)

f64 on the Metal backend is **hard-rejected**, not deferred. Apple Silicon
GPUs (M1, M2, M3, M4, and every announced successor in the Apple GPU family)
have no double-precision floating-point ALUs in their GPU compute units;
this is a hardware constraint, not a Chelis design choice. Software
emulation (e.g., double-double arithmetic over two f32s) is **explicitly
out of scope** for this cycle and any foreseeable cycle: the precision,
performance, and AD-adjoint stories for emulated f64 do not match the
language-level f64 contract, and silently substituting an emulated value
would violate the no-implicit-precision-promotion rule in §5.

The required diagnostic when a program uses f64 with `--target metal` is:

> "Apple Silicon GPUs lack FP64 ALUs; use `--target c` or `--target hip` for
> f64 workloads."

This rejection is enforced at the CLI gate, at the IR validation pass, and
defensively at the codegen entry point. All three surfaces must surface the
same diagnostic. f64 must never reach the kernel emission path on Metal.

##### bf16 on Metal: requires Apple7+ GPU family

bf16 on the Metal backend is admitted only on the Apple7 GPU family (M3) or
later. There are two enforcement surfaces, and the spec pins both because
they fail in different places at different times:

- **Compile-time (kernel template).** MSL exposes the `bfloat` scalar type
  only when the target language version is 3.2 or higher (which corresponds
  to Apple7+ GPU family targeting). The Metal kernel template emits any
  `bfloat`-typed kernel inside `#if __METAL_VERSION__ >= 320 ... #endif`. A
  CLI build that emits bf16 kernels still produces a syntactically valid
  Metal source artifact: the artifact contains the `#if`-guarded kernel
  text and is acceptable to `metal` / `clang++ -framework Metal` on every
  toolchain version, regardless of the target GPU family.
- **Runtime (pipeline creation).** Even with the kernel artifact present,
  pipeline state creation on a pre-Apple7 device (M1, M2) will fail because
  the device's GPU does not support `bfloat` operations. The Metal runtime
  surfaces this as a clean diagnostic at pipeline creation time, not as a
  silent kernel-load failure.

The required diagnostic when bf16 pipeline creation fails on a pre-Apple7
device is:

> "bf16 requires Apple7+ GPU family (M3 or later); detected device family is
> Apple{N}."

Programs that target bf16 on Metal must be tested on Apple7+ hardware.
Test runners on M1/M2 hardware should mark bf16 Metal correctness tests as
ignored with the manual gate documented in the owning phase plan.

##### Metal runtime header: ARC vs MRC and MPS wrapper ownership model

The Metal backend's Objective-C++ runtime header
(`crates/chelis-backend-metal/runtime/chelis_metal_runtime.h`) is **compiled
under ARC** (Automatic Reference Counting). The user-facing build
invocation is documented as `clang++ -fobjc-arc -framework Metal -framework
Foundation` against the emitted `.mm`; the `-fobjc-arc` flag is part of
the user-facing contract for the Metal backend. The runtime header itself
documents and relies on this model: `chelis_metal_alloc` returns an
ARC-owned `id<MTLBuffer>` whose strong reference is released automatically
when the local goes out of scope, and there is intentionally no
`chelis_metal_free`.

Future helpers that the runtime header exposes — including the planned
`MetalPerformanceShaders.MPSMatrixMultiplication` wrapper helpers
(`chelis_metal_mps_gemm_f32`, `chelis_metal_mps_gemm_f16`, and any
sibling) — must follow the **same ARC model**. The four invariants below
are pinned for every present and future Metal runtime helper:

1. **ARC vs MRC.** The runtime header is compiled under ARC. Every helper
   added to `chelis_metal_runtime.h` MUST follow the same model.
   **Mixing ARC and MRC in the same translation unit is forbidden**; an
   MRC helper that needs to coexist with the ARC runtime must live in a
   separate translation unit with explicit boundary documentation.
2. **Lifetime boundary.** MPS wrapper objects (`MPSMatrixDescriptor`,
   `MPSMatrix`, and any analogous `MPS*` helper object) MUST NOT outlive
   the `MTLBuffer`s they wrap. The existing `chelis_metal_alloc` plus
   `chelis_tensor` ownership manages `MTLBuffer` lifetimes; the wrapper
   helpers must construct their `MPS*` objects inside the
   `chelis_metal_mps_gemm_*` call, use them for one dispatch, and let
   them be released before the call returns. There is **no caller-side
   ownership of the MPS objects**: callers see only the `MTLBuffer`
   inputs and outputs they already owned. The `MTLBuffer` references the
   wrappers hold MUST be live for the duration of the call (the caller
   keeps strong references through the dispatch).
3. **Autorelease pattern.** Because the runtime is ARC, each MPS wrapper
   helper function MUST wrap its dispatch in `@autoreleasepool { ... }`
   so that any MPS-internal autoreleased objects (descriptors created by
   convenience constructors, transient `NSError` chains, etc.) do not
   leak across host calls. The autorelease pool is per-helper-call, not
   per-program.
4. **No-mixed-model invariant.** Future additions to
   `chelis_metal_runtime.h` MUST follow the same ARC model. Any new
   helper that would mix models must be in a separate translation unit
   with explicit boundary documentation; it must not be added to
   `chelis_metal_runtime.h`.

These four invariants exist so that the implementation phase that adds the
MPS wrappers does not re-decide the ownership model, and so that future
Metal runtime work cannot drift the contract.

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
- later Phase 3 core numeric surfaces such as `einsum`, `gather`, `where`, `sort`,
  `diagonal` / `trace`, and `clamp`
- later standard-library host types such as `Std.Time` and `Std.Decimal`

The currently shipped executable `3d` slice covers compiled collection foundations:
`List[T]`, list literals, `len`, `index`, `append`, `concat`, `take`, `drop`,
`chunk`, `flatten`, `range`, `zip`, `enumerate`, numeric `to_tensor`, practical rank-1
`to_list`, `pad_sequences`, and the first immutable `Dict[K, V]` surface (`dict_of`,
`dict_get`, `dict_contains`, `dict_remove`, `dict_insert`, `dict_merge`, `dict_keys`,
`dict_values`, `dict_entries`) plus higher-order iteration (`map`, `filter`, `fold`,
`scan`, `partition`, `flat_map`) compile through the host-value lane, and callback
effects propagate through iteration in the shipped checker. This compiled helper set is
now the practical collection foundation for `3g`: nested token lists can be flattened
or chunked on the host side, dictionaries can be updated/trimmed immutably, and
`pad_sequences` still marks the explicit bridge into fixed-shape tensors.

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

### 4.6 Property Definitions

Surf `@property` declarations type-check as ordinary functions whose result
type is `bool`. Every binder must have an explicit type in v1. The `where`
preconditions must type-check as `bool` expressions in the binder scope; the
predicate body must type-check as `bool`.

Desugaring emits a `defsig` with the binder types and `bool` result, plus an
ordinary `def` carrying the property metadata specified in
`spec/design/chelis_property_spec.md`. Type checking trusts neither the metadata
nor the property annotation; it checks the resulting Deep function against the
signature like any other definition.

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

Integer literals default to `int32`. Float literals default to `f32`. These
defaults can be overridden in three ways:

1. an explicit literal suffix (§5.5) attached to the literal token
2. a known element type in the surrounding position (§5.6)
3. an explicit `cast` around the literal expression

There is **no implicit precision promotion** from these defaults to any other
type. A bare `[1, 2, 3]` in an unannotated position is `tensor[3, int32]`, not
`tensor[3, int64]`. A bare `[1.0, 2.0, 3.0]` in an unannotated position is
`tensor[3, f32]`, not `tensor[3, f64]`. Programs that need a wider literal
type must say so via suffix, declared element type, or `cast`.

The default is the **user-facing contract**. The lexer parses an unsuffixed
integer or float literal token at i64/f64 precision so that out-of-range
literals can be diagnosed before defaulting; the desugarer/type-check
narrows the literal to `int32` (for integer tokens) or `f32` (for float
tokens) before Deep is materialized. The narrowing is mechanical and
non-overridable except by the three mechanisms above. Implementation
references for verification: `crates/chelis-surf/src/desugar.rs` (literal
desugaring emits `(lit {type: (t-prim {} int32)} N)`), `crates/chelis-types/src/infer.rs`
(literal inference rule maps `Atom::Int → Prim::Int32`, `Atom::Float → Prim::F32`).

### 5.4 Precision Compatibility Table

Operations accept same-precision operands only. The table of valid combinations:

| Operation type | Valid precisions |
|---|---|
| Arithmetic (add, mul, sub, div) | f32, f64, bf16, f16, int8, int16, int32, int64 (all same) |
| Comparison (cmplt, eq) | any numeric (same precision) → bool |
| Logical (and, or, not) | bool only |
| Transcendental (exp, log, sin, sqrt) | f32, f64, bf16, f16 only (not integer) |

`f8e4m3` is deferred (§1.1.1) and is not a valid arithmetic precision in any
row. Unsigned integer types are out of scope (§1.1.2) and never appear in any
row.

### 5.5 Literal Suffix Grammar

Numeric literal tokens may carry an explicit precision suffix. A suffixed
literal binds at exactly that precision; no inference, no widening, no
narrowing. The closed suffix set is:

| Suffix | Bound type | Example |
|---|---|---|
| `f32` | `(t-prim {} f32)` | `1.0f32`, `3.14e-2f32` |
| `f64` | `(t-prim {} f64)` | `1.0f64` |
| `bf16` | `(t-prim {} bf16)` | `1.0bf16` |
| `f16` | `(t-prim {} f16)` | `1.0f16` |
| `i8` | `(t-prim {} int8)` | `42i8` |
| `i16` | `(t-prim {} int16)` | `42i16` |
| `i32` | `(t-prim {} int32)` | `42i32` |
| `i64` | `(t-prim {} int64)` | `42i64` |

Float-typed suffixes (`f32`, `f64`, `bf16`, `f16`) attach to either an integer
or a float literal token (`42f32` and `1.0f32` are both well-formed and bind
at f32). Integer-typed suffixes (`i8`, `i16`, `i32`, `i64`) attach to integer
literal tokens only; `1.0i8` is a parse error.

Suffix lexing rule: a suffix is part of the literal token only if it
**immediately** follows the digit sequence with no intervening whitespace,
comment, or other character. `1.0 f32` (with whitespace) is two tokens (a
float followed by an identifier) and binds at the literal default per §5.3,
which is then subject to the surrounding-position rules in the type checker.

Hex-literal interaction (parser-implementation note): the lexer's hex-literal
rule consumes `[0-9a-fA-F_]*` after `0x`. Because `f` is a hex digit, a hex
integer literal cannot directly carry a float-typed suffix (`0xFFf32` is not
"hex 0xFF then suffix f32"; it is "hex 0xFFf then int 32" under maximal-munch
hex lexing, which is rejected as malformed). Hex integer literals MAY carry
integer-typed suffixes only (`0xFFi8`, `0xFFi32`, etc.); float-typed suffixes
on hex literals are a parse error with a diagnostic suggesting an explicit
`cast`. Decimal float literals carry float suffixes without ambiguity
(`1.0f32`, `1.0e3f32`).

Deferred / out-of-scope suffixes:

- `f8e4m3` is deferred (§1.1.1); the suffix `f8e4m3` is rejected at lex time
  with a diagnostic pointing at §1.1.1.
- Unsigned suffixes (`u8`, `u16`, `u32`, `u64`) are out of scope (§1.1.2) and
  are rejected at lex time with a diagnostic pointing at §1.1.2.
- Any other unrecognized identifier sequence directly adjacent to a numeric
  literal (e.g. `1.0xyz`) is a parse error rather than a silently-split
  literal-then-identifier pair.

The suffix grammar is identical in Surf (`spec/02-surf-syntax.md` §3, P10)
and Deep (`spec/03-deep-syntax.md` §6.4).

### 5.6 Contextual Tensor-Literal Inference

When a tensor literal appears in a position with a **known element type**, the
numeric literals in the tensor body adopt that element type instead of the
literal default in §5.3. The closed set of "known-element-type" positions is
exactly:

1. the right-hand side of a binding whose declared type is a tensor type, e.g.
   `let xs: tensor[3, f64] = [1.0, 2.0, 3.0]`
2. the corresponding argument position of a call whose callee has a declared
   signature whose parameter at that position is a tensor type, e.g.
   `f(xs)` where `f : tensor[3, f64] -> ...`
3. the body expression of a function with a declared return type that is a
   tensor type, when the body is a tensor literal
4. the first argument of a `cast(literal, p)` expression, where `p` is a
   precision type literal — the literal body adopts `p`

Outside this closed set, numeric literals in a tensor body fall back to the
§5.3 literal defaults: integer literals to `int32`, float literals to `f32`.

A tensor literal with mixed-suffix entries is well-formed only if every
suffix matches the inferred element type. `[1.0, 2.0f64, 3.0]` in an
`f32`-context is a type error: the f64-suffixed literal at index 1 has an
explicit dtype that disagrees with the surrounding `f32` element type.

A bare tensor literal `[1, 2, 3]` in an unannotated position evaluates to
`tensor[3, int32]`, not `tensor[3, int64]`. The fallback to the §5.3
default is the spec contract; the implementation must not silently widen.

### 5.7 Mixed-Precision Accumulator Parameter

Two reduction-shaped operations carry an **optional** accumulator-precision
parameter that controls the precision used for the inner sum:

- `matmul(A, B, accumulator=p)` — the precision of the inner-product
  accumulator before the result is downcast to the operand precision (when
  the accumulator is wider than the operands)
- `reduce_sum(x, axis=k, accumulator=p)` (also known under the spec name
  `sum`; see `spec/05-risc-primitives.md` §2.3) — the precision of the
  running sum

The accumulator parameter is the **only** mechanism for mixed precision in
these ops. There is no implicit precision promotion: passing two `bf16`
operands to `matmul` does not implicitly widen them. The accumulator
parameter is what tells the backend to compute the inner sum at a wider
precision and (where the result is the operand precision) downcast at the
end.

When the accumulator parameter is **omitted**, the compiler resolves it to
the documented default for the operand precision, before any backend is
invoked. The IR matmul/reduce_sum nodes always carry a populated
accumulator-precision field; "no accumulator parameter" is a Surf/Deep
ergonomic shorthand, not an IR state.

#### 5.7.1 Default Accumulator Precision

For operands of precision `p`, the default accumulator precision is:

| Operand precision `p` | Default accumulator (matmul) | Default accumulator (reduce_sum) | Result precision |
|---|---|---|---|
| `bf16` | `f32` | `f32` | operand precision (`bf16`) |
| `f16` | `f32` | `f32` | operand precision (`f16`) |
| `f32` | `f32` | `f32` | operand precision (`f32`) |
| `f64` | `f64` | `f64` | operand precision (`f64`) |
| `int8` | (matmul not defined for int8 — see §5.7.2) | `int32` | `int32` |
| `int16` | (matmul not defined for int16 — see §5.7.2) | `int32` | `int32` |
| `int32` | (matmul not defined for int32 — see §5.7.2) | `int32` | `int32` |
| `int64` | (matmul not defined for int64 — see §5.7.2) | `int64` | `int64` |

Rationale for the bf16/f16 → f32 default: numerical stability of long
inner-product reductions in low-precision arithmetic. PyTorch and JAX use the
same wider-accumulator default for bf16/f16 matmul.

Rationale for the i8/i16 → i32 default: overflow safety. Summing 200
non-trivial `int8` values overflows `int8` but fits comfortably in `int32`.
This matches PyTorch's `torch.sum` accumulator-promotion rule for narrow
integer inputs.

The accumulator parameter is permitted only when it is at least as wide as
the operand precision and is not narrower than the documented default. A
program that explicitly requests a narrower accumulator (e.g.
`reduce_sum(x: tensor[N, int8], accumulator=int8)`) is a type error with a
diagnostic suggesting either omitting the parameter (which yields the i32
default) or accepting the wider default explicitly.

The user-facing result precision of `reduce_sum` is given by the
"Result precision" column of the §5.7.1 table above. For `int8` and
`int16` operands the result widens to the accumulator (`int32`) to
prevent silent overflow; for `int32`, `int64`, `f32`, and `f64`
operands the result equals both operand and accumulator. For `bf16`
and `f16` operands the result returns to the operand precision (the
`f32` accumulator is consumed inside the op and downcast on output) so
the caller sees a uniform-precision result tensor.

At the IR level, the `Sum` node's output precision is always the
accumulator precision; lowering inserts an explicit `Cast` for the
`bf16`/`f16` row to recover the operand-precision result documented in
the table.

The result precision of `matmul` matches the operand precision (the
wider accumulator is consumed inside the op and downcast on output) so
that the caller sees a uniform-precision result tensor.

#### 5.7.2 Integer matmul

The active matmul signature does not admit integer operand precisions
(`int8`, `int16`, `int32`, `int64`). The spec deliberately does not pin an
integer-matmul accumulator rule in this cycle: there is no current backend
that supports integer BLAS, and an integer-matmul surface raises questions
(saturating vs wrapping accumulator, signed-vs-unsigned interaction with
§1.1.2) that are out of scope here. Integer `reduce_sum` is supported per
§5.7.1.

### 5.8 Stdlib Generalization Shape

Every public tensor-op signature exported from `packages/chelis-std/` is
generalized over a precision type variable. Type-check enforces that each
instantiation works against the active dtype set (§1.1) and the precision
compatibility table (§5.4).

A typical generalized signature has the shape:

```scheme
;; Std.Tensor.add : forall p. tensor[D, p] -> tensor[D, p] -> tensor[D, p]
(defsig {} add
  (t-fn {}
    (t-tensor {} (d-var {} d) (t-var {} p))
    (t-tensor {} (d-var {} d) (t-var {} p))
    (t-tensor {} (d-var {} d) (t-var {} p))))
```

Float-only operations (those whose §5.4 row reads "f32, f64, bf16, f16 only")
carry an explicit kind restriction limiting `p` to the float subset of the
active dtype set. Examples include `exp`, `log`, `sin`, `sqrt`, and the
transcendental row in §5.4. Calling a float-only op on an integer tensor is a
type error reported at the call site, not deep inside the implementation.

The implementation is permitted to specialize each instantiation (e.g. via
monomorphization) so that backends never see a polymorphic stdlib body. The
spec contract is that the **public signature** is generalized; per-backend
specialization is an implementation detail.

The active dtype set is the source of truth for stdlib generalization
coverage: every public tensor op must be usable at every dtype in §1.1 that
its §5.4 row admits. A stdlib op that fails for a §5.4-admissible dtype is a
spec compliance bug, not a documentation bug.

#### 5.8.1 Contextual Precision Desugar (WS-A5)

The Surf surface admits precision polymorphism in user-written sigs by
treating identifiers in the precision slot of a `tensor[...]` type
contextually:

> In a sig with quantified type variables, names appearing in the
> precision slot of a `tensor[...]` type that match the sig's quantifier
> list become `(t-var {} <name>)`, not `(t-prim {} <name>)`. Names
> matching a primitive (`f32`, `f64`, `bf16`, `f16`, `i8`, `i16`,
> `i32`, `i64`, `bool`) stay as `(t-prim {} <name>)`. Outside a sig
> (e.g., in a value-position type annotation), no quantifier exists,
> so the existing rule applies.

Quantifiers in a sig are **implicit**: any lowercase, non-primitive,
non-`spec/04-type-system.md` §1.1.2-unsigned identifier that appears in
the sig's type expression is treated as a `forall`-quantified type
variable. The §1.1.2 unsigned aliases (`u8`, `u16`, `u32`, `u64`,
`uint8`, `uint16`, `uint32`, `uint64`) are explicitly excluded so they
reach the type-checker's §1.1.2 rejection path with a precise
diagnostic, not silently absorbed as quantifiers.

The internal type representation carries this through `TensorPrec`:

```rust
pub enum TensorPrec {
    Concrete(Prim), // resolved or user-written concrete primitive
    Var(TypeVar),   // sig-quantified precision polymorphism
}
```

Unification of two tensor types unifies their precision slots:
`Concrete(p) ~ Var(v)` binds `v ↦ p`; `Var(v1) ~ Var(v2)` links the
two vars; `Concrete(p1) ~ Concrete(p2)` succeeds only when `p1 == p2`
(the existing `PrecisionMismatch` rule). Generalization
(`Env::generalize`) collects free precision-slot vars so each call
site instantiates the sig with a fresh precision variable.

After monomorphization, every reachable tensor type at lowering time
must carry `TensorPrec::Concrete(_)`. Backends assert this invariant
at the lowering match arm; a `TensorPrec::Var(_)` reaching a backend
is a monomorphization bug, not user error.

Monomorphization is implemented (WS-A8) by threading a precision
substitution map (`prec_substitutions`) through `LowerCtx`, populated
at every call site of a polymorphic-precision sig from the formal vs
actual parameter types. Inlining at a concrete call site supplies the
concrete precision before the backend boundary; standalone emission
of a polymorphic-precision sig is intentionally elided (no caller can
use the symbol without a concrete instantiation). Cross-row spec
rules (§5.4 transcendental float-only, §5.7.2 integer matmul not
admitted) are enforced both at direct primitive call sites and at
polymorphic-call-site instantiation by the
`validate_polymorphic_op_constraints` pass; an integer instantiation
of a polymorphic body that uses `matmul` (or float-only operand of a
transcendental row op) surfaces with the spec's existing
`PrecisionMismatch` diagnostic kind and a §5.7.2 / §5.4 citation.

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
| Malformed binding form | `"Unbound variable x in body. Missing bind?"` |

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
- `dropout(x, rate)` and tensor RNG operations such as `uniform_like` are the
  concrete shipped `Random` sources; stdlib random helpers such as
  `normal_like` and Kaiming/Xavier initializers inherit that effect through
  calls
- `print(x)` and `debug(x)` are the concrete shipped `IO` sources
- `with seed(seed) { ... }` handles `Random` across direct operations and calls made
  inside the handled region; the C host backend preserves this with generated
  handler-scoped RNG state for nested stdlib/user functions
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

## 8. Linearity (Phase 2b, Superseded By Implicit Linearity)

### 8.1 Model

Phase 2b uses lightweight uniqueness, not a Rust-style ownership-and-lifetimes
system. Tensor values are owned by default, but read-only calls borrow their tensor
arguments. A consuming use makes the binding dead; a borrow leaves the owned binding
live. The `copy-drop` model in `spec/design/implicit_linearity.md` supersedes the old
requirement that every local owner write an explicit final consume.

### 8.2 Type Representation

```scheme
;; Linear tensor (default)
(t-tensor {lin: once} (d-name {} batch) (t-prim {} f32))

;; Borrowed tensor (read-only reference)
(t-ref {} (t-tensor {} (d-name {} batch) (t-prim {} f32)))
```

The shipped Phase 2b user surface is type- and expression-based:

- `&T` is the read-only borrow type, represented in Deep as `t-ref`
- `&x` is an optional explicit borrow expression, represented as `(borrow {} x)`
- passing owned `T` where `&T` is expected auto-borrows; this rule also applies to pipe stages
- passing `&T` where owned `T` is expected is a type error unless the program writes `copy(x)`
- `copy(x)` accepts either owned `T` or borrowed `&T` and yields a fresh owned value;
  explicit and compiler-inserted copies lower to `RiscOp::Copy`
- borrows cannot be stored in aggregates, returned, or captured by closures
- borrow types are erased before IR and backend lowering; implicit linearity then
  inserts explicit `RiscOp::Copy` and `RiscOp::Drop` nodes

Linearity is checked after effect inference, before lowering:

`parse -> desugar -> type infer/check -> effect infer/check -> linearity check -> lower`

Linearity is expected to compose with effects, especially `Resource(Device)`: the
effect system tracks where a tensor lives, while linearity tracks when it is consumed.
That gives the compiler a stronger basis for safe in-place buffer reuse.

### 8.3 Static Rules

- A local owned linear binding must have exactly one terminal path in lowered IR:
  either a consuming use or an inserted `Drop`. Borrow sites do not count as consumes.
- Function parameters are ownership-transfer boundaries: an owned parameter may be
  borrowed throughout the function body and then leave the function scope without an
  implicit local `drop` expression.
- A local value that was borrowed but never consumed receives an inserted end-of-scope
  `Drop` in lowered IR. This is not a user-facing type error.
- `copy(x)` reads `x` without consuming it and yields a fresh tensor value.
- `drop(x)` is an explicit consume. The compiler also inserts implicit end-of-scope
  drops for locals that are not otherwise consumed.
- Pattern matching on a tuple or other value carrying tensor payloads consumes the
  scrutinee; any tensor payloads bound by the pattern become the new live bindings.
- Creating a closure that captures a tensor consumes that outer binding at closure
  creation time.
- Ordinary consuming fan-out is handled by inserted copies. Diagnostics remain for
  invalid borrows, borrow escapes, impossible branch/loop ownership, and recursive or
  cyclic consume cases outside the v1 inference scope.
