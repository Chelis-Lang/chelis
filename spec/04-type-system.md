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

| dtype  | C backend                                                                                                         | HIP backend                                          | Metal backend                                | Evaluator |
|--------|-------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|----------------------------------------------|-----------|
| f32    | admitted                                                                                                          | admitted                                             | admitted                                     | admitted  |
| f64    | admitted                                                                                                          | admitted                                             | **rejected (hardware)**                      | admitted  |
| bf16   | admitted (storage as `uint16_t`; arithmetic via convert-to-f32; matmul via convert-then-`cblas_sgemm` per §5.7.1) | admitted (matmul + load/store via `hipblasGemmEx`)   | admitted on Apple7+ (M3 or later)            | admitted  |
| f16    | admitted (storage as `uint16_t`; arithmetic via convert-to-f32; matmul via convert-then-`cblas_sgemm` per §5.7.1) | admitted (matmul + load/store via `hipblasGemmEx`)   | admitted                                     | admitted  |
| int8   | admitted                                                                                                          | admitted                                             | admitted                                     | admitted  |
| int16  | admitted                                                                                                          | admitted                                             | admitted                                     | admitted  |
| int32  | admitted                                                                                                          | admitted                                             | admitted                                     | admitted  |
| int64  | admitted                                                                                                          | admitted                                             | admitted                                     | admitted  |
| bool   | admitted                                                                                                          | admitted                                             | admitted                                     | admitted  |

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
- `(d-name {} batch)` unifies with `(d-lit {} N)` — a concrete literal satisfies a
  concrete-but-named slot at the call site. Names are preserved in diagnostics; they do
  not impose a distinct-from-literal constraint. This is Option A from
  `Chelis-Lang/chelis#219`, which removes the prior `Name <-> Lit` asymmetry that
  rejected `f(to_tensor([[1, 2, 3]]))` against `def f(x: tensor[batch, hidden, f32])`
  even though the corresponding `Var <-> Lit` shape was accepted.

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

Declared dimension parameters are **rigid within the def body**. When a
definition is checked against its declared signature, each declared dim
parameter is universally quantified: the body must type-check for *all*
instantiations of that dim. Consequently two **distinct** declared dim
parameters do not unify with each other during body validation — a body
that requires `n` and `m` to be the same dimension does not satisfy a
signature that declares them separately. This is a body-validation rule,
distinct from call-site instantiation: it is only at the call site that a
dim variable is genuinely bound to a concrete dimension.

```scheme
;; def f[n, m](x: tensor[n, f32], y: tensor[m, f32]): tensor[n, f32] = y
;; TYPE ERROR: the body returns tensor[m, f32] but the declared return is
;; tensor[n, f32]; n and m are distinct rigid dim parameters.
```

#### 4.4.1 Return-Only Dim Parameters (chelis#273)

A declared dim parameter that appears **only in the return type** has a
different relationship to the body than a param-position one. Chelis has
no explicit dimension application: callers instantiate dim parameters by
unification against *arguments*, and an argument never mentions a
return-only dim. The body is therefore the only place the output
dimension can come from, and a return-only dim parameter is
**output-inferred** rather than fully rigid. Two body behaviors are
legitimate:

- the body leaves the dim var unbound — a clean, generalizable
  dimension (e.g. a variable-fed `to_tensor` whose shape is genuinely
  unknown), or
- the body resolves it to a **body-internal** concrete dimension; the
  registered scheme then resolves to the produced dim. This is the
  `examples/hello_tensor.ch` shape: `def main() -> tensor[n, f32]`
  whose body builds a `tensor[3, f32]`.

What the body must **not** do is couple the promised-independent output
dimension to the caller-visible input world. Both of the following are
`DimensionMismatch` type errors:

- **input-coupled pin**: the return-only dim parameter resolves to a
  concrete literal that occurs (after unification) in a declared
  parameter position;
- **input-coupled collapse**: the return-only dim parameter unifies with
  a distinct param-position declared dim parameter.

```scheme
;; def f[k](a: tensor[2, f32]): tensor[k, f32] = a
;; TYPE ERROR: the body pins the return-only dim parameter k to the
;; parameter's concrete Lit(2); the signature promised an output
;; dimension the body does not derive from the inputs.

;; def f[n, m](x: tensor[n, f32]): tensor[m, f32] = x
;; TYPE ERROR: the return-only dim parameter m collapses with the
;; param-position dim parameter n.

;; def make(): tensor[n, f32] = to_tensor([1.0, 2.0, 3.0])
;; OK: output-inferred. The body produces a body-internal tensor[3, f32]
;; and the scheme resolves n := 3; no input dimension is involved.
```

Three deliberate boundaries of this rule:

- a body-internal concrete pin whose literal does *not* occur in any
  declared parameter position is tolerated even when the def has
  parameters (`def f(x: tensor[2, f32]) -> tensor[k, f32] =
  to_tensor([1.0, 2.0, 3.0])` is accepted with `k := 3`) — the guard
  compares resolved dimensions, not provenance, so a body-internal
  literal that happens to *equal* a parameter dim is conservatively
  rejected, and one that differs is conservatively accepted;
- coupling through a *named* symbolic dim
  (`def f(x: tensor[batch, f32]) -> tensor[m, f32] = x`, which binds
  `m` to `batch`) is not flagged: `Dim::Name` unifies permissively by
  design (chelis#219) and no declared dim parameter participates.
  (When a param-position declared dim parameter *also* resolves to the
  same name — e.g. `def f[n, m](x: tensor[n, f32],
  y: tensor[batch, f32]) -> tensor[m, f32] = add(x, y)` binds both `n`
  and `m` to `batch` — the collapse rule above does fire, because the
  two declared dim parameters now share a resolution.);
- coupling through a *wildcard* param dim is invisible
  (`def f[k](b: tensor[*, f32]) -> tensor[k, f32] = b` is accepted):
  the wildcard unifies permissively without binding (§4.5), so `k`
  stays unbound and generalizes even though the returned value's
  runtime dimension is the input's. This is the pre-existing §4.5
  wildcard permissiveness, not a new tolerance of this rule.

Enforcement is `check_return_only_dvars_rigid` in
`crates/chelis-types/src/infer.rs`, run at the same def-vs-signature
reconciliation point as the §4.4 param-position guard; the acceptance
oracle is `crates/chelis-cli/tests/issue_273_return_dvar_rigidity.rs`.

### 4.5 The Wildcard Dimension

`(d-name {} *)` represents an unknown/dynamic dimension. Produced by operations where the compiler cannot statically determine the dimension:

- Concatenation along an axis: `concat(tensor[batch, seq1, f32], tensor[batch, seq2, f32], axis=1)` → `tensor[batch, *, f32]`
- Data loading with dynamic shapes
- Results of control flow where branches have different known dimensions

A wildcard dimension unifies with any other dimension (like a variable) but is NOT generalized — it's a permanent "I don't know." To restore named-dimension checking after a wildcard, use an explicit annotation.

#### 4.5.1 Rank-Uniform `List[tensor[...]]` Elements

A `List[T]` is statically homogeneous in `T`, and a tensor's rank is part
of its type. An annotation like `List[tensor[k, f32]]` therefore fixes a
single rank for every element — the dim slot `k` is a dimension variable,
not a shape-vector variable. A list literal `[a, b]` whose elements have
different ranks is a type error, surfaced as `DimensionMismatch` at the
list literal expression with a message of the form:

```
list element rank mismatch: 1 dims vs 2 dims;
List[tensor[...]] requires rank-uniform elements (the dim slot is a
dimension variable, not a shape-vector variable). Reshape or flatten
elements to a common rank before listing (spec/04-type-system.md §4.5.1).
```

The message is emitted on a single line; the wrapping above is for
readability only. The trailing `(spec/04-type-system.md §4.5.1)`
back-reference is part of the diagnostic so a reader or agent can
locate this rule from the error text alone.

Rationale: Chelis dimension variables (§4.4) range over individual
dimensions, not over shape vectors. Permitting `[rank-1, rank-2]` to
unify by erasing the rank would mask the kinds of transposition and
reshape bugs that named dimensions exist to catch (§4.2 rationale). A
rank-erased element type is not provided in the shipped surface; users
who genuinely need to carry mixed-rank tensors through a list must
reshape elements to a common rank before listing, or use a sum type
that names each rank as a separate variant. A rank-polymorphic
`List[tensor[k, f32]]` (letting `k` range over shape vectors per call
site) was considered and deferred: the named-dim safety guarantee in
§4.2 is preferred over the additional flexibility, and the
reshape-at-the-boundary idiom is cheap enough that current consumers
absorb it without losing per-tensor named dimensions. A *constrained,
name-preserving* shape spread that does **not** sacrifice that guarantee
is admitted for top-level tensor signatures by Tier-3 rank polymorphism
(§4.5.3) — but it remains barred from `List`/ADT element position, so the
rank-uniform-list guarantee above is unaffected.

```chelis
;; WRONG: rank-1 and rank-2 elements in the same List[tensor[k, f32]]
;; def make_mixed() -> List[tensor[k, f32]] = {
;;   a = to_tensor([cast(1.0, f32), cast(2.0, f32)])
;;   b = to_tensor([[cast(1.0, f32), cast(2.0, f32)],
;;                  [cast(3.0, f32), cast(4.0, f32)]])
;;   [a, b]  ;; DimensionMismatch: list element rank mismatch
;; }

;; CORRECT: flatten the rank-2 element to rank-1 first
;; def make_uniform() -> List[tensor[k, f32]] = {
;;   a = to_tensor([cast(1.0, f32), cast(2.0, f32)])
;;   b_flat = reshape(
;;     to_tensor([[cast(1.0, f32), cast(2.0, f32)],
;;                [cast(3.0, f32), cast(4.0, f32)]]),
;;     [cast(4, int64)])
;;   [a, b_flat]
;; }
```

#### 4.5.2 List-literal dimension joining

This rule governs the per-axis dimensions of a list literal *once its
elements are rank-uniform* (§4.5.1): rank uniformity is checked first,
and only matching-rank elements reach the per-axis join below.

A list literal of tensors, `[a, b, ...]`, desugars to a `Cons`/`Nil`
chain. Each `Cons` step computes a **per-axis join** of the new element
against the running list-element type. The join is intentionally
permissive about *concrete* shape so that the common
`concat([a, b], axis)` pattern accepts elements whose concrete axes
differ (chelis#218):

- equal concrete dims (two equal literals, or two equal names) are
  preserved;
- two **genuinely-mismatched concrete** axes (e.g. `(d-lit {} 2)` vs
  `(d-lit {} 3)`, or two distinct names) widen to `(d-name {} *)`
  along that axis. The resulting `List[tensor[..., *, ...]]` is the
  defensible "the lengths differ along this axis" type that lets
  `concat` consume a ragged list;
- a pair where at least one side is a **dimension variable** (a declared
  rigid dim parameter such as `k`) is **unified**, not widened. Two
  distinct rigid dims unify with each other; a `(rigid, concrete)` pair
  pins the rigid dim to the concrete value.

This is the chelis#272 tightening. The earlier behavior widened *every*
non-equal pair — including pairs naming rigid dim parameters — to a
wildcard, which then satisfied an explicit `List[tensor[k, f32]]`
return annotation and silently defeated the §4.4 rigid-distinct-dim
guarantee. Two consequences of the tightened rule:

1. A body whose list mixes two **distinct rigid dims** (e.g.
   `def make[k, m](a: tensor[k, f32], b: tensor[m, f32])
   -> List[tensor[k, f32]] = [a, b]`) is rejected: the join unifies
   `k` and `m`, and the §4.4 rigid-dim guard reports the collapse — the
   same diagnostic class as the non-list `def f[n, m](...) = y` case.

2. A body whose list has **heterogeneous concrete** element lengths
   (e.g. `def make[k](a: tensor[2, f32], b: tensor[3, f32])
   -> List[tensor[k, f32]] = [a, b]`) is rejected: the `2`/`3` join
   widens to `(d-name {} *)`, and a join-origin wildcard list element
   may **not** satisfy a declared element type that names a rigid or
   named dimension. `List[tensor[k, f32]]` promises every element shares
   the length `k`; a heterogeneous list does not.

The wildcard list element remains acceptable when the surrounding
binding makes **no** uniformity promise — a bare
`out = concat([...], axis)` with no return annotation and no declared
dim parameters, or an explicit `List[tensor[*, f32]]` annotation, both
type-check.

Two boundary properties of the current rule are intentional but narrow,
and are locked by dedicated tests so a future change is a conscious one:

- **The `(concrete, wildcard)` join is head-biased.** A `Cons` step
  resolves the joined axis to whatever the *head* (the element being
  prepended, i.e. the earlier list position) resolves to. So
  `[tensor[2, f32], tensor[*, f32]]` joins to element `tensor[2, f32]`
  (the concrete head absorbs the wildcard tail), whereas the reordered
  `[tensor[*, f32], tensor[2, f32]]` joins to `tensor[*, f32]` (the
  wildcard head erases the concrete tail). Under a `def f[k](a:
  tensor[2, f32], b: tensor[*, f32]) -> List[tensor[k, f32]]`
  annotation **both orderings are now rejected**, but through different
  guards: the concrete-element ordering pins the return-only `k` to the
  parameter's `Lit(2)` and trips the §4.4.1 return-position rigidity
  check (chelis#273 — parity with the non-list
  `def f[k](a: tensor[2, f32]) -> tensor[k, f32] = a`), while the
  wildcard-element ordering trips the join-origin-wildcard uniformity
  check below. The head bias itself is still observable in *which*
  diagnostic fires. (Before chelis#273 the concrete-element ordering
  type-checked, because the return-only `k` was outside the rigidity
  guard's param-position scope.) Genuinely-mismatched *concrete*
  heads/tails still widen to `*` regardless of order (the ragged-axis
  arm).
- **The uniformity check is single-level.** It compares the declared and
  body element axes of one `List[tensor[..]]`; it does **not** recurse
  into a nested element. A wildcard tensor under
  `List[List[tensor[k, f32]]]` is *not* checked against the inner `k` and
  currently type-checks.

The enforcement is in `crates/chelis-types/src/infer.rs`
(`infer_app`'s `Cons` join and `check_list_elem_rigid_dim_vs_wildcard`,
alongside `check_declared_dvars_rigid`); the acceptance oracle is
`crates/chelis-cli/tests/issue_272_list_dim_rigidity.rs` with the
chelis#218 ergonomics locked by
`crates/chelis-cli/tests/issue_218_to_tensor_in_grad_body.rs`.

#### 4.5.3 Name-Preserving Rank Polymorphism (Tier-3)

§4.5.1 deferred a shape-vector variable because *rank erasure masks
transposition bugs*. Tier-3 admits a **constrained, name-preserving** shape
spread that does not reopen that hole: a rank variable `..r` (the `Dim::Rank`
spread) binds to the *actual named dims it covers*, so per-axis names are
retained, not erased, and the surviving axes carry their identity through.

A tensor shape may interleave spreads with concrete **named anchors** — the
admitted form is `Rank? (Name Rank?)*` (a given spread name appears at most once
per shape). The headline use is a **named-axis reduction**: a single `def`
reduces a named axis at any rank, and the checker computes the output shape
symbolically.

```chelis
;; reduce the named `seq` axis, keep everything else by name:
;; def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)
;;   tensor[batch, seq, hidden] -> tensor[batch, hidden]
;;   tensor[a, b, seq, c]       -> tensor[a, b, c]
```

Multiple axes are reduced by composing single-axis reductions
(`sum(sum(x, head), seq)`).

**Unification (unitary).** A row shape unifies with a ground shape by locating
each named anchor uniquely in the ground and binding the spreads to the runs
between. Because each interior split is fixed by a name, there is one
most-general unifier:

- A named anchor must occur in the operand **exactly once**; absent, ambiguous,
  or non-named (a fully-literal operand under the §4.1 Name↔Lit rule) is a
  **hard error**, never a guessed split.
- Two spreads with no anchor between them (`tensor[..a, ..b]`) is an
  *undetermined* split and is rejected at unification — except in an output
  position (e.g. a reduction's `tensor[..pre, ..post]` result), which is only
  ever matched against an identical row or expanded after its spreads are bound.

**Soundness (§4.2).** Order is preserved (shapes stay ordered positional
sequences — never unordered "rows"); the reduced axis is a retained name; and a
rank-poly def body is restricted by the §4.2 Body-Discipline check to
*name-trackable* operations only — shape-identity (elementwise) ops and
named-axis reductions. A *positional* shape-rewriter (`permute`, `reshape`,
`matmul`, positional `gather`) is rejected inside a `..r` body: its output shape
is not name-trackable at symbolic rank, so it could hide an untracked
transposition. This is what keeps the §4.5.1 transposition-safety guarantee
intact while admitting the deferred flexibility for the reduction case.

Enforcement: the unification arm is `crates/chelis-types/src/unify.rs`
(`unify_row_against_ground` / `unify_row_against_row`); the named-axis reduction
arm is `check_reduction_signature` and the discipline check is
`check_rank_body_discipline` in `crates/chelis-types/src/infer.rs`. Call-site
**rank monomorphization** (`tensor_rank_substitutions` /
`extract_rank_var_bindings` in `crates/chelis-ir/src/lower.rs`) substitutes each
spread's concrete run and resolves the named axis to a positional index at
lowering, so a rank-poly reduce **builds and runs** on the C backend. The
acceptance oracle is `crates/chelis-cli/tests/rank_poly_tier3.rs`.

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

### 4.7 Runtime Shape Semantics

Some Chelis programs need to pass a runtime axis size as an argument to a
movement primitive, typically when the input tensor has a symbolic batch
dimension that is only known at run time. The relevant built-ins are:

- `shape(x, axis)`: returns the size of `x`'s `axis`-th dimension as an
  `int32` value. `axis` must be a concrete non-negative integer literal
  (either a bare `int32` literal or a `cast(N, int32)` form; both reach
  the axis-bounds check). The result is a runtime scalar, not a symbolic
  dim reference.
- `expand(x, axis, size)`: insert or set a dimension at position `axis`
  with width `size`.
- `reshape(x, shape_list)`: reinterpret the memory of `x` against
  `shape_list`, a `List<Int64>`.

This section pins which call shapes preserve symbolic dims in the type
checker's output and which fall back to `(d-name {} *)` (see §4.5). The
canonical examples live in
[`examples/illustrative/runtime_shape_semantics.ch`](../examples/illustrative/runtime_shape_semantics.ch).

#### 4.7.1 `shape` axis form

Both forms below produce an `int32` value and both pass the infer-time
axis-bounds check (`cast(N, int32)` is unwrapped by the cast-aware
extractor described in `crates/chelis-types/src/infer.rs`):

```text
shape(x, 0)
shape(x, cast(0, int32))
```

Internal callers (stdlib) use the `cast(0, int32)` form for stability
against future literal-default changes. User code MAY use either.

A negative axis or an axis greater than or equal to the input rank is a
type error (`DimensionMismatch`).

#### 4.7.2 `expand` with a runtime size

`expand(x, axis, size)` accepts three forms for its `size` argument:

1. an integer literal (or `cast(N, int32)` form): produces an output
   dim of `(d-lit {} N)`.
2. a symbolic dim name in scope (a bare `var` reference such as a
   declared `[batch]` dim parameter): produces an output dim of
   `(d-name {} batch)`.
3. any other `int32` expression, including a runtime `shape(...)`
   call: the typer defers the output rank slot to whatever the
   declared signature's return-type or the surrounding call context
   imposes via standard unification.

Form (3) is how the `bias_broadcast` pattern from
`examples/illustrative/runtime_shape_semantics.ch` preserves the
symbolic batch dim `n`:

```chelis
sig bias_broadcast: &tensor[n, 4, f32] -> &tensor[4, f32] -> tensor[n, 4, f32]
def bias_broadcast(x, b) -> tensor[n, 4, f32] = expand(b, 0, shape(x, cast(0, int32)))
```

The declared return type `tensor[n, 4, f32]` is unified with the
fresh result tvar produced for the `expand` call; the typer does not
need to derive the symbolic dim from the runtime `shape(...)`
expression itself. If the call site does not impose a known dim at
the expanded axis (for example, an unannotated `let` binding), the
output dim at that axis falls back to `(d-name {} *)`.

#### 4.7.3 `reshape` with runtime sizes from `shape(x, ...)`

`reshape` recognizes one specific syntactic source for each element
of its shape list and propagates the corresponding input axis into
the result type. The recognized form for a shape-list element is:

```text
cast(shape(<reshape-input-var>, <literal-axis>), int64)
```

All four conditions are required:

1. The outer `cast`'s target type is `int64` (matching the
   `List<Int64>` element type that `reshape` expects).
2. The cast's inner expression is a `shape(...)` call.
3. The `shape` call's first argument is a bare `var` whose bound name
   is the same as the reshape's input tensor argument.
4. The `shape` call's axis argument extracts to a concrete
   non-negative integer (literal or `cast(N, int32)` form).

When all four conditions hold, the typer propagates the input's dim
at the named axis into the corresponding output dim. The
`flatten_batch` and `flatten_two` patterns from
`examples/illustrative/runtime_shape_semantics.ch` are the canonical
examples:

```chelis
sig flatten_batch: &tensor[n, 4, f32] -> tensor[n, 4, f32]
def flatten_batch(x) -> tensor[n, 4, f32] = reshape(x, [cast(shape(x, cast(0, int32)), int64), cast(4, int64)])
```

A plain integer literal element (`cast(4, int64)`) still produces
`(d-lit {} 4)`. Any element shape that is not literal and does not
match the four conditions above falls back to `(d-name {} *)`.

#### 4.7.4 Patterns that fall back to wildcard

The following shape-list forms in `reshape` fall back to
`(d-name {} *)` for that element. The typer does NOT manufacture a
symbolic dim for them; the contract is "recognize the exact
syntactic pattern from §4.7.3 or fall back."

- **Cross-tensor shape source**: `cast(shape(y, axis), int64)` where
  `y` is a different tensor than the one being reshaped. The
  recognizer's same-input-var check requires bound-name equality.
  Inventing a propagated dim here would silently unify two distinct
  declared dim parameters and trip the rigidity check in §4.4.
- **Arithmetic or other expression wrappers**: `cast(add(shape(x,
  0), 1), int64)`, `cast(mul(shape(x, 0), 2), int64)`, etc. The
  outer cast peel does not find a direct `shape(input, lit_axis)`
  call, so the recognizer falls back. Propagating `x`'s dim through
  an arithmetic wrapper would be unsound.
- **Non-`var` reshape input**: `reshape(reshape(x, ...), [...])`
  where the inner reshape is the input. The reshape input is not a
  bare `var` Deep node, so there is no input-var name to match
  against the inner `shape(...)`'s tensor arg. Inner reshape's body
  still type-checks via the polymorphic fresh result tvar unifying
  with the declared signature.

#### 4.7.5 Precision rule for `reshape`'s shape list

`reshape`'s shape list must be homogeneous. The bare `shape(x, k)`
returns `int32` and integer literals default to `int32`, but
`reshape`'s shape list is conventionally `List<Int64>`, so every
element must be cast to `int64` before it can appear in the list:

```text
reshape(x, [shape(x, 0), 4])                          ;; TYPE ERROR: precision mismatch: expected int32, got int64
reshape(x, [cast(shape(x, 0), int64), cast(4, int64)]) ;; OK
```

The diagnostic direction reflects the unification order in the
type-checker: the first list element fixes the "expected"
precision, and later entries that disagree trip the mismatch on the
"got" side. The bare `shape(x, 0)` pins `int32` as the expected
precision; the integer literal `4` (also `int32`) is fine, but the
moment any `int64` enters the list the mismatch fires. In practice
the fix is the same in both directions — cast every element to
`int64` — so the unification-order artifact does not affect
remediation guidance.

#### 4.7.6 Out-of-scope: arbitrary runtime shape expressions

Chelis does NOT currently propagate symbolic dims through arbitrary
runtime shape expressions. If a downstream tool needs to emit code
that derives a reshape dim from an arithmetic combination of input
axes (`shape(x, 0) * 2`, `add(shape(x, 0), shape(y, 0))`, etc.), the
result dim is `(d-name {} *)` and the surrounding code must accept
that wildcard. The recommended downstream pattern is to emit only
the recognized forms in §4.7.3 and reject other shapes at the tool
boundary with a clear diagnostic, so the user is not surprised by a
silent `Wildcard` cascade.

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

Position 4 applies to a **bare scalar numeric literal** as well as to a
tensor-literal body (issue #308). `cast(1.1, f64)` binds the decimal `1.1`
at `f64` — exactly `0x3ff199999999999a` — it does NOT narrow to the §5.3
`f32` default and then widen (which would yield the f32-truncation value
`1.100000023841858`). Likewise `cast(3000000000, int64)` binds the literal
at `int64`, which is what makes the §5.3 out-of-int32-range escape hatch
work. The adoption re-binds the literal at `p` and the §5.6 range checks
apply at `p`: `cast(2147483648, int32)` is still a range error. Adoption
is limited to unsuffixed numeric literals with a numeric `p` of matching
kind: a suffixed literal binds at its suffix (§5.5; `cast(1.1f32, f64)`
widens the f32 value), and a float literal under an integer `p` keeps the
default-then-truncate cast semantics because a decimal cannot bind at an
integer type.

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
- `IO` -- host-side effects such as `print` and `debug` (debugging/logging),
  the file builtins (`read_file`, `write_file`, ...), and subprocess exec via
  `process_run`. The effect enum itself is unchanged; `IO` is the single
  effect that covers all host-side observable interaction, now including
  spawning external processes.
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
- `print(x)` and `debug(x)` are the concrete shipped `IO` sources, alongside
  the file builtins (`read_file`, `write_file`, `read_lines`, `read_bytes`,
  `file_exists`, `list_dir`, `mmap_file`) and `process_run` (subprocess exec;
  eval/test-only, rejected by the build backends per spec/05-risc-primitives.md
  §2.6)
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

#### Borrow target classification (the `&x` inner type)

The inner of a `&x` borrow expression must be — or must ultimately resolve
to — a tensor or a tensor-carrying value (a tensor-carrying ADT per §8.4 or
a tuple containing one). Borrowing a concretely non-tensor value (a scalar
`t-prim`, `()`, a function, a non-tensor-carrying ADT, or a tuple of
scalars) is a type error.

The borrow inner's type is not always concrete at the borrow site. When the
inner is the result of a polymorphic-return expression — for example
`relu(prev_out)` or `mean(...)` whose dimension variables are pinned only
by a later `&tensor[..]` parameter in the surrounding call — the inner is
still an unresolved type variable when the borrow is first checked. In that
case classification is **deferred**: the borrow is provisionally accepted
and the surrounding flow's expected argument type pins the variable through
unification. A previously-required workaround was to round-trip the value
through a monomorphic identity (`def id4[a,c,h,w](x: tensor[a,c,h,w,f32]) ->
tensor[a,c,h,w,f32] = x`) to re-bind the dimension variables before the
borrow; that workaround is no longer necessary.

The deferral is sound only when the variable is *eventually* pinned to a
tensor or tensor carrier. If the consumer is itself fully polymorphic
(e.g. `def consume_any[a](t: a) -> bool`), the variable is never pinned to a
tensor, and a genuinely non-tensor value — including one a caller
instantiates at a scalar type — would otherwise be borrowed. After a
function body's inference completes, every deferred borrow is re-checked
against the final substitution; a variable that did not resolve to a tensor
or tensor carrier is rejected with the same diagnostic as a concretely
non-tensor borrow inner. There is no terminating program that can borrow a
non-tensor value through the deferred path.

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

### 8.4 Tensor-carrying ADTs

An ADT `T` is **tensor-carrying** iff at least one of `T`'s variant fields has a
type that contains a tensor, where "contains a tensor" is the least relation
satisfying:

- `tensor[...]` contains a tensor.
- `(t1, t2, ...)` (tuple) contains a tensor iff some `ti` does.
- `&U` contains a tensor iff `U` does.
- `U[arg1, arg2, ...]` (ADT instantiation) contains a tensor iff `U` is itself
  tensor-carrying, **or** some `argi` contains a tensor.
- Function types `t-fn` are not treated as containing a tensor for this rule,
  even when their parameters or return contain one. Closures that capture
  tensors are handled by the §8.3 capture rule, not the carrier set.

Tensor-carrying ADTs participate in linearity exactly like bare tensors:

- An owned `T` value is linear; it has exactly one terminal path (a consuming
  use or an inserted `Drop`).
- `&T` is a valid borrow expression and a valid borrow type, and follows the
  same auto-borrow rules as `&tensor[...]`.
- A `match` that scrutinizes an owned tensor-carrying `T` consumes the
  scrutinee per §8.3; field bindings on the matched variant become the new
  live owners of any tensor payloads they expose.

The carrier rule is transitive across `deftype` declarations: if `Outer` has a
field of type `Inner[k]` and `Inner` is tensor-carrying, then `Outer` is
tensor-carrying. Resolution is a least-fixed-point over all `deftype` decls
visible at check time. When the linearity checker runs against composed
contexts (library + new code), both halves are resolved in one pass so a
new-code `Outer` whose carrier status depends on a library `Inner` is
recognized correctly.

### 8.5 Type-Name Uniqueness

Within a single checked program, every `deftype` and `typealias` name must be
unique. The type-name namespace is flat: a `deftype Foo` cannot coexist with
another `deftype Foo` nor with a `typealias Foo = ...`, and user code cannot
re-declare a prelude type name (e.g. `Option`, `List`). Collisions are
rejected at declaration time as `DuplicateDefinition`. This rule is what
makes the §8.4 carrier set well-defined when keyed on the bare ADT name.

### 8.6 Builtin-Name Shadowing

A top-level `def` or `sig` whose name appears in the closed builtin function
vocabulary (`BUILTIN_NAMES` in `crates/chelis-types/src/builtins.rs`) is
rejected at declaration time as `BuiltinShadowing`, before inference runs.

Rationale: call sites are dispatched builtin-first by name in both the host
evaluator and IR lowering, so a user definition that shadows a builtin name
can never be reached by name. Pre-rule, the checker resolved such calls to
the user signature while eval and the backends resolved them to the builtin —
three lanes, three different answers (chelis#353: `def sum` checked clean,
failed with the builtin's arity error under eval, and segfaulted on the C
backend). The rejection reads the same `BUILTIN_NAMES` table the evaluator
dispatch and IR lowering import, so the rejected set and the dispatched set
cannot drift.

Scope:

- The rule binds to top-level `def` and `defsig` declarations after module
  flattening, including load-style top-level bindings (`sum = ...` desugars
  to a `def`), in every lane that runs the type checker (`check`, `eval`,
  `build`, `test`, `cost`). `chelis validate` is a syntax-grammar lane that
  does not run the type checker and therefore does not surface this (or any
  other) semantic rejection. It is a semantic rejection, not a style-gate
  rule: `--allow-style-violations` and `CHELIS_STYLE_GATE_DISABLE=1` do not
  bypass it.
- Reef package modules are exempt by construction: package declarations are
  internal-name-rewritten (`pkg__<package>__<module>__<name>`) before the
  checker runs and their call sites are rewritten with them, so a
  package-scoped `def sum` neither collides with the builtin table nor
  mis-dispatches — inside a package the user def genuinely wins (the
  stdlib's `Std.Decimal.normalize` and `Std.Test.fail` rely on this).
- Function parameters and block-local bindings may reuse builtin names: they
  bind values, not call-site dispatch, and shadow harmlessly on every lane.
  Known residual asymmetry: *calling* a function-typed parameter or local
  named like a builtin still dispatches builtin-first under eval (a loud
  eval-time error) while the C backend compiles the call correctly; that
  gap is documented here rather than rejected, because rejecting it would
  break programs the backend lane compiles and runs correctly today.
- Builtin-adjacent reserved keywords (`cast`, `grad`, `vmap`, ...) are not
  part of `BUILTIN_NAMES`; a `def cast` is already a parse error.
