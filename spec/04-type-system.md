# spec/04-type-system.md — Chelis Type System

**Scope:** ADTs, Hindley-Milner inference, numeric precision types, named tensor
dimensions, fitness scoring, annotated checked Deep, effects, and linearity.

**Terminology:** a *lane* is one execution path, such as the evaluator or a compiled
target, as defined in [spec/01 §1.8](01-nomenclature.md#18-lanes).

---

## 0. Checked Deep Contract

The type checker is the first pass that upgrades raw Deep into the downstream
compiler-facing representation.

- `check_ir_program(...)` returns a `CheckedProgram`, not just a success/failure bit
- a `CheckedProgram` carries annotated Deep, with `type` metadata written onto the
  returned tree
- lowering, evaluation, effect checking, and CLI build/eval paths consume that
  annotated tree rather than the original raw Deep

Effect inference and checking run after HM type inference on the same annotated tree.

---

## 1. Type Representation

Types are represented as Deep AST nodes using the `t-*` tag family.

### 1.1 Primitive Types

The active primitive set is exactly eleven names:

```scheme
(t-prim {} f32)       ;; 32-bit float
(t-prim {} f64)       ;; 64-bit float
(t-prim {} bf16)      ;; bfloat16
(t-prim {} f16)       ;; 16-bit float (IEEE 754 binary16)
(t-prim {} i8)        ;; 8-bit signed integer
(t-prim {} i16)       ;; 16-bit signed integer
(t-prim {} i32)       ;; 32-bit signed integer
(t-prim {} i64)       ;; 64-bit signed integer
(t-prim {} bool)      ;; boolean
(t-prim {} string)    ;; UTF-8 string
(t-prim {} key)       ;; random key
```

These closed subsets are used throughout the specs:

- the **eight active numeric dtypes** are the four floats (`f16`, `bf16`,
  `f32`, `f64`) and four signed integers (`i8`, `i16`, `i32`, `i64`);
- the **ten active tensor element dtypes** are those eight numeric dtypes,
  `bool`, and `key`;
- the **nine active data element dtypes** are the active tensor element dtypes
  other than `key`: the eight numeric dtypes plus `bool`; and
- `string` is an active host primitive, but is neither a numeric dtype nor a
  tensor element dtype.

A `key` is a 64-bit random key ([05-RNG-2]). It is not numeric, and it has
no arithmetic, comparison, cast, literal, or default value. [05-OP-69] creates
keys, [05-OP-70] through [05-OP-72] derive them, and a draw keyed by a key
reads it under [05-RNG-2]. An operation admits `key` elements only where its
own atom names `key`: a domain written as every active tensor element dtype
admits exactly the nine active data element dtypes. Keys are affine
([04-LIN-9]), and no function's type parameter stands for a key-carrying
type ([04-LIN-10]).

Code, tests, examples, and stdlib signatures referenced from any active spec
section must use these set names with exactly those meanings. The reserved
spellings in §1.1.1 are rejected.

The retired v0.18 spellings `int8`, `int16`, `int32`, and `int64` are not
primitive names. Normal Surf and Deep ingress reject them and direct the
author to the corresponding versioned migration command. They remain the
stable ecosystem spellings of existing JSON, WireDag, cache, and ABI-facing
interchange contracts; that external vocabulary does not make them source or
Deep type names. Retired spellings are not available to implicit or explicit
type-variable binding. This exclusion does not depend on classifying them as
future-reserved primitive names.

#### 1.1.1 Reserved And Rejected Numeric Names

> **[04-DTYPE-1]** A primitive type position SHALL name one of the active
> primitives in §1.1. Every name in this section is reserved but rejected by
> the type checker, including as a `cast` target. Any other primitive spelling is an
> unknown type and is likewise rejected; it SHALL NOT acquire the operand's
> type or a default dtype.

The following primitive names are reserved but **not members of the language's
primitive set**. Every name below is rejected by the type checker with a
diagnostic pointing at this section, and `cast(x, <name>)` is rejected with it.

Each reserved name records the arithmetic family that its spelling denotes;
because the name is rejected, that record does not admit values or operations.
The spellings and format identities follow numpy, PyTorch, JAX, Apache Arrow,
and the OCP 8-bit and Microscaling specifications rather than introducing
Chelis-specific variants.

**Reduced-precision floats.**

- `f8e4m3` — 8-bit float (E4M3 per the OCP 8-bit Floating Point
  Specification). Arithmetic width: **f32**. OCP E4M3 is not IEEE-shaped: it
  has no infinities and reuses that encoding to extend range. Chelis defines
  no finalization rule for this reserved spelling, so the checker rejects it.
- `f8e5m2` — 8-bit float (E5M2 per the same OCP specification). Arithmetic
  width: **f32**. **Rationale:** E4M3 and E5M2 are one format pair, not two
  independent dtypes: FP8 training uses E4M3 for forward values and E5M2 for
  gradients. Reserving only E4M3 would reserve half a format.

**Unsigned integers.**

- `uint8`, `uint16`, `uint32`, `uint64` — unsigned integers at width.
  Arithmetic width: **exact at their own width**, unsigned. Overflow traps per
  [04-NUM-3]. No arithmetic operation, including [04-NUM-7]'s modular
  operations, admits these reserved spellings. **Rationale:**
  universal in the ecosystem (numpy, PyTorch, JAX, and Arrow all carry the
  full set) and unavoidable at real ingress boundaries - `uint8` is the image
  dtype, and hashing, checksums, and PRNG state are natively unsigned.
- `int4`, `uint4` — 4-bit integers. Arithmetic width: **exact at their own
  width**. **Rationale:** the quantized-inference frontier (JAX carries both).
  A 4-bit element has no addressable byte, and Chelis defines no packing or
  element-order rule for these reserved spellings.

**Complex.**

- `complex64`, `complex128` — complex numbers with `f32` and `f64` components
  respectively. Arithmetic width: **f32 and f64 components**. **Rationale:**
  present in numpy, PyTorch, and JAX under exactly these spellings; the naming
  convention is TOTAL bits, so `complex64` is a pair of f32 - Chelis follows
  the ecosystem spelling rather than inventing `complex32x2`. Chelis defines
  no complex arithmetic operations for these reserved spellings.

**Decimal (interchange only).**

- `decimal128`, `decimal256` — exact base-10 values at a declared
  (precision, scale), per Apache Arrow. Arithmetic width: **exact base-10 at
  the declared scale**; this is a third arithmetic family alongside float and
  integer, not a parameterization of either. **Rationale and scope limit:** no
  array-computation library carries decimal - numpy, PyTorch, and JAX all
  decline it - because it is a database and dataframe type. Chelis already
  serves that need twice: `Std.Decimal` is the standard-library exact decimal
  scalar (`spec/05-risc-primitives.md` [05-OP-76]: at most 38 significant
  digits, scale 0..38), and `Std.Io.Parquet` is the interop surface. These
  names are therefore reserved for the **Arrow and parquet interchange
  boundary only** and are explicitly NOT tensor element types. They are also
  the only reserved names that are parameterized, which is itself a reason to
  keep them off that path: every other `Prim` is a bare name. At that
  boundary a `decimal128` or `decimal256` value whose exact value lies in
  [05-OP-76]'s value set converts to that exact `Std.Decimal`; any other
  value is a `domain` failure and is never rounded; and a `Std.Decimal`
  value reaches a declared (precision, scale) only through an explicit
  [05-OP-74] rounding mode.

**Not reserved, and deliberately so: scaled and block-scaled formats.**
`qint8`/`quint8` (PyTorch), the MX formats of the OCP Microscaling
specification (MXFP8, MXFP6, MXFP4), and FP8 training's amax-scaled tensors
are NOT additional dtypes. Each is a storage dtype PLUS scale metadata - per
tensor, per channel, or per block of 32. Modelling scale once, as an axis over
a storage dtype, admits all of them; adding them as primitive names would
recreate per-format dtype cells. Scaled storage is represented by a storage
dtype plus an explicit scale axis.

**Saturating arithmetic is not implicit.** Ordinary arithmetic traps rather
than saturating under [04-NUM-3]. Saturating behavior exists only where a
separate named operation defines it; it is never a mode or default.

#### 1.1.2 Unsigned Integer Spellings

Unsigned integer types are reserved under §1.1.1 as `uint8`, `uint16`,
`uint32`, and `uint64`, with `int4`/`uint4` alongside them. The short spellings
`u8`/`u16`/`u32`/`u64` are NOT reserved; the `uint*` spellings are canonical,
matching numpy and Arrow. All of these spellings are rejected under
[04-DTYPE-1].

#### 1.1.3 Per-Backend Dtype Support Matrix

The active primitive set in §1.1 is the **language-level** dtype contract: a
program that mentions one of the eleven active primitives is well-typed in
every Chelis pass that does not select a backend (parser, type checker, IR
evaluator). Backend code generation is a separate surface; not every backend
admits every active dtype. This sub-section is the authoritative per-backend
matrix. Any "Metal supports X" or "C backend supports Y" claim elsewhere in
the spec or in user-facing docs must resolve to a cell in this table.

| dtype  | C backend                                                                                                         | HIP backend                                                                                                                                                                                              | Metal backend                                | Evaluator |
|--------|-------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|-----------|
| f32    | admitted                                                                                                          | admitted                                                                                                                                                                                                 | admitted                                     | admitted  |
| f64    | admitted                                                                                                          | admitted                                                                                                                                                                                                 | **rejected (hardware)**                      | admitted  |
| bf16   | admitted (storage as `uint16_t`; arithmetic via convert-to-f32; matmul via convert-then-`cblas_sgemm` per §5.7.1) | **operation-limited**: tensor load/store, exact-bit `Realize`, [05-OP-40] `max_elem`/`min_elem` and their adjoints, [05-OP-41] `sub`, dedicated [05-OP-43] `Relu`/`ReluAdjoint`, and `BlasMatmul` via `hipblasGemmEx` subject to each operation's own restrictions | admitted on Apple7+ (M3 or later)            | admitted  |
| f16    | admitted (storage as `uint16_t`; arithmetic via convert-to-f32; matmul via convert-then-`cblas_sgemm` per §5.7.1) | **operation-limited**: tensor load/store, exact-bit `Realize`, [05-OP-40] `max_elem`/`min_elem` and their adjoints, [05-OP-41] `sub`, dedicated [05-OP-43] `Relu`/`ReluAdjoint`, and `BlasMatmul` via `hipblasGemmEx` subject to each operation's own restrictions | admitted                                     | admitted  |
| i8   | admitted                                                                                                          | admitted                                                                                                                                                                                                 | admitted                                     | admitted  |
| i16  | admitted                                                                                                          | admitted                                                                                                                                                                                                 | admitted                                     | admitted  |
| i32  | admitted                                                                                                          | admitted                                                                                                                                                                                                 | admitted                                     | admitted  |
| i64  | admitted                                                                                                          | admitted                                                                                                                                                                                                 | admitted                                     | admitted  |
| bool   | admitted                                                                                                          | admitted                                                                                                                                                                                                 | admitted                                     | admitted  |
| key    | admitted                                                                                                          | **operation-limited**: no operation                                                                                                                                                                      | **operation-limited**: no operation          | admitted  |

**Arithmetic width is not a cell of this table.** It is a target-independent
property of the dtype, declared once by [04-NUM-8] and owned by the semantic
side of the capability split (`spec/design/capability_table.md`'s Table A,
whose rejections the CHECKER reports because they hold whatever the target
is). This table records target capability, which build and lowering decide
after the target is known. Reproducing the widths here would put an A-fact in a
B-table and create a second copy to keep in sync by hand.

The constraint this table DOES carry is the consequence: a backend that
admits a dtype computes it at the width [04-NUM-8] declares, and a backend
that cannot is a `rejected` cell - never an `admitted` cell that silently
widens, narrows, or substitutes. The bf16/f16 C-backend notes below are that
rule already applied, not an exception to it.

Cell semantics:

- **admitted** — the backend codegen accepts the dtype on every operation
  surface whose own dtype rule admits it, and emits code satisfying that rule. Per-op restrictions
  (e.g. matmul accumulator dispatch per §5.7.1, transcendental ops are
  float-only per §5.4) apply uniformly across backends and are not encoded
  in this matrix.
- **operation-limited** — the backend accepts the dtype only on the operation
  identities named in that cell, subject to each named operation's own
  restrictions. Every unlisted operation that is otherwise valid for the dtype
  rejects before emission. The backend SHALL NOT route an unlisted operation
  through another dtype, operation identity, fallback kernel, or software
  substitution.
- **rejected (hardware)** — the backend rejects the dtype at codegen with a
  diagnostic naming the hardware constraint and does not substitute a dtype
  or software emulation (see the f64-on-Metal entry below).

##### f64 on Metal: hard-rejected (hardware rationale)

> **[04-TGT-1]** The Metal target SHALL reject every f64 value before kernel
> emission and SHALL direct the user to the C or HIP target. It SHALL NOT
> substitute f32 or software-emulated arithmetic for the language's f64
> semantics.

f64 on the Metal backend is **hard-rejected**. Apple Silicon GPUs have no
double-precision floating-point ALUs in their GPU compute units;
this is a hardware constraint, not a Chelis design choice. Software
emulation (e.g., double-double arithmetic over two f32s) does not satisfy the
language-level f64 contract and is not a Metal-target substitute.

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

Any helper that the runtime header exposes — including
`MetalPerformanceShaders.MPSMatrixMultiplication` wrapper helpers — must
follow the **same ARC model**. The four invariants below govern every Metal
runtime helper:

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
4. **No-mixed-model invariant.** Additions to
   `chelis_metal_runtime.h` MUST follow the same ARC model. Any new
   helper that would mix models must be in a separate translation unit
   with explicit boundary documentation; it must not be added to
   `chelis_metal_runtime.h`.

These four invariants keep one ownership model across the Metal runtime.

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

;; List (tensor[batch, f32])
(t-adt {} List (t-tensor {} (d-name {} batch) (t-prim {} f32)))

;; Column with a concrete extent argument
(t-adt {} Column (d-lit {} 3))

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

### 2.3.1 Generic ADT representation classification

> **[04-ADT-1]** Backend layout and generic-function specialization SHALL use
> the checked program's alias-resolved ADT registry and checked authored
> signatures. They SHALL NOT reconstruct parameter ownership, constructor
> fields, or polymorphism from authored `deftype` / `defsig` syntax.

A parameter used only as a tensor dimension is representation-erased; a
parameter used as a tensor precision or another value-represented field is
stored. The distinction is recursive through nested ADTs. Thus
`Frame[n,a] -> Hamt[Column[n,a]] -> tensor[n,a]` erases `n` but must retain
and specialize `a`. The checked registry and authored
signature marker survive serialization, context composition, effects
reannotation, and linearity annotation.

> **[04-ADT-2]** Every use of a generic function signature SHALL instantiate
> a fresh substitution. When computing the free variables of an environment
> scheme, a substitution SHALL NOT be applied through that scheme's quantified
> type, dimension, or rank variables, including through an alias chain. Thus
> one concrete tensor extent, rank, precision, or stored ADT argument cannot
> constrain a later use in the same consuming module.

### 2.3.2 Nominal parameter kinds and applications

> **[04-ADT-3]** Every parameter of a `deftype` or `typealias` SHALL have one
> checker-owned kind, `Type` or `Dimension`, fixed for the whole check unit
> before any declaration body is resolved. A parameter has `Dimension` kind
> exactly when it has at least one use in a dimension slot and no use in a
> type slot. A tensor axis is a dimension slot; other type-expression
> positions are type slots. A nominal argument position inherits the target
> header's corresponding kind. This inheritance is the least fixed point over
> all source headers, including forward, recursive, mutually recursive, and
> alias-mediated references. An unused parameter defaults to `Type`. A
> parameter used in both kinds is a type error; an implementation SHALL NOT
> choose one occurrence, create independent variables, or reinterpret either
> occurrence as an inference wildcard.

> **[04-ADT-4]** A nominal application SHALL have the target header's exact
> arity and SHALL match every parameter kind positionally. A `Type` parameter
> accepts only a type argument. A `Dimension` parameter accepts one dimension
> (`d-name`, `d-var`, or `d-lit`) and does not accept a type or rank spread.
> Consequently, a Surf integer argument such as `Column[3]` denotes the exact
> literal dimension `(d-lit {} 3)`, while `Option[3]` is a type error. The
> dimension argument participates in ordinary dimension substitution and
> unification through constructors, aliases, signatures, records, matches,
> evaluation, and lowering. Unequal literal extents are a
> `DimensionMismatch`; an implementation SHALL NOT erase, freshen, or
> wildcard the argument before checking. Type aliases substitute type and
> dimension parameters transparently and preserve the same rule.

### 2.4 Exhaustive Pattern Matching

The type checker verifies that `match` expressions cover all variants. Missing variants are a type error, not a warning.

A top-level irrefutable arm without a guard covers the match: a bare
variable pattern (`| x =>`) or an as-pattern whose inner pattern is
irrefutable (`| q @ x =>`, `| q @ _ =>`). The coverage applies at the arm
level only; a variable pattern NESTED inside a constructor or record pattern
does not cover the other variants. A guarded arm covers nothing
([04-PAT-2]).

Coverage is a separate question from whether a pattern is admissible at the
scrutinee type at all.

> **[04-PAT-1]** A literal pattern is a typing constraint on the scrutinee, at
> every depth a pattern may occur. Its value atom SHALL agree with the
> scrutinee's primitive type under [04-LIT-1]'s closed pairing: an integer atom
> matches only an integer primitive, a float atom only a float primitive, a
> boolean atom only `bool`, and a string atom only `string`. [04-LIT-1]'s one
> cross-family form, an Int atom marked `literal_source: integer` under a float
> primitive, requires `lit` metadata that a `pat-lit` cannot carry and therefore
> does not arise in pattern position. A literal pattern SHALL NOT be admitted
> against a non-primitive scrutinee: a tensor, nominal, tuple, record, or
> function scrutinee admits no literal pattern. A numeric literal pattern binds
> at the scrutinee's primitive type under [04-LIT-2] and SHALL be rejected when
> its value there is out of range or non-finite, as a literal bound at that
> type in expression position is. Each violation SHALL be a
> `TypeMismatch` located at the offending pattern, at every checker ingress and
> before any evaluation or lowering lane runs; an implementation SHALL NOT admit
> the arm as merely unreachable, drop it, or defer the diagnostic to a lane. A
> literal pattern selects no width: a Deep `pat-lit` carries only its raw value,
> with no precision slot and no admissible suffix
> (`spec/03-deep-syntax.md` §6.4, `spec/02-surf-syntax.md` §P10a), so an
> unsuffixed integer pattern is admissible against every integer primitive and
> an unsuffixed float pattern against every float primitive, and §5.3's literal
> default does not apply in pattern position.

> **[04-PAT-2]** A `match` evaluates its scrutinee once and tries its arms in
> declaration order. An arm is selected when its pattern matches the scrutinee
> and its guard, if it has one, evaluates to `true`; the `match` evaluates to
> the selected arm's body, evaluated in that arm's scope. A guard is evaluated
> only after its arm's pattern has matched, in the arm's scope: every name the
> pattern binds is visible to it, bound to the part of the scrutinee it matched,
> as is every name visible at the `match`. A guard has type `bool`, under the
> same obligation as an `if` condition (§3.2). When a guard evaluates to
> `false`, its arm is not selected, the bindings its pattern introduced are
> discarded, and matching continues with the next arm. No guard is evaluated for
> an arm whose pattern did not match or for any arm after the selected one, and
> a guard whose evaluation traps makes the `match` trap rather than reading as
> `false`. Because a guard can be `false`, a guarded arm contributes nothing to
> the coverage §2.4 requires: a top-level irrefutable arm covers the match only
> when it has no guard, and a variant is covered only by an arm without a guard.
> An arm without a guard carries `()` in the Deep guard slot
> (`spec/03-deep-syntax.md` §2.3) and is selected whenever its pattern matches.
> Every execution lane SHALL implement this rule: a lane that cannot lower a
> guarded arm SHALL reject the program with a diagnostic, and SHALL NOT drop the
> guard, drop the arm, or select an arm whose guard is `false`.

### 2.5 Opaque Types

A `deftype` carrying `opaque: true` metadata (Surf: the `@opaque`
annotation, spec/02 §5.2) declares an opaque type: constructible and
inspectable only inside its defining module, enforced by the type
checker during inference. The authoritative design record is
`spec/design/opaque_invariants_rfc.md` (D-CHECK); this section is the
normative summary.

**Named-module requirement.** `@opaque` requires a named enclosing
module. A top-level opaque declaration has no module identity (all
top-level code in a check unit shares one anonymous key), so the
declaration is rejected: `@opaque type X requires a named enclosing
module`.

**One wrapper per module.** A named module may be opened by at most
one `(module ...)` wrapper per check unit. Module identity is
otherwise a forgeable string: a second `(module Stats.Prob ...)`
wrapper would let its declarations construct and inspect
`Stats.Prob`'s opaque types as if they were inside the defining
module. Re-opening a module name is a `DuplicateModule` checker error
(same ambiguity rationale as the named-module requirement). Surf emits
one module per file and the reef package linker strips wrappers before
inference, so this rule only constrains hand-written Deep. Distinct
module wrappers in one check unit (the ordinary out-of-module setup)
are unaffected.

**Module identity.** Each top-level item keys to a module:

- Lexical encoding: the enclosing `(module ...)` wrapper names, with
  nested wrappers joined by `.` (Surf `module Stats.Prob` desugars to
  the key `stats.prob`).
- Package-linker encoding: reef rewrites top-level names to
  `Pkg__<pkg>__<Module>__<Name>` (or the lowercase `pkg__` twin); the
  stem between the marker and the trailing name is the module key,
  rendered with `.` separators (`Pkg__opq__Demo__Types__Probability`
  keys to `opq.Demo.Types`). The lexical wrapper wins when both are
  present.

The defining module is recorded on the registry entry at `deftype`
registration and persists through the compiled-context caches.

**Reserved linker name format.** The package-linker encoding
(`Pkg__<pkg>__<Module>__<Name>` / lowercase twin) is the reef linker's
PRIVATE output. Because module identity derives from it, a
hand-authored program of mangled names self-keys to a module and would
construct and inspect its opaque types as if in-module. So a top-level
`deftype`/`def`/`defsig`/`typealias`/`defmacro` whose binding name
matches that format is a `ReservedLinkerName` declaration error in any
program NOT produced by the linker. Provenance is an in-process flag
set at the reef link boundary (`prepare_program_for_file` and the
reef-aware check/eval/build entry points), never filesystem or name
heuristics; raw `.ch`/`.dp` ingestion keeps it FALSE. The linker feeds
linked Deep to the checker in-process, and re-mangles every user source
name, so its own output checks clean while a user cannot smuggle a
clean mangled name into linked output. As belt-and-suspenders, a
stem-derived module identity colliding with a lexical wrapper key in
one check unit is also a `DuplicateModule` error (genuine linker output
has no lexical wrappers, so this never fires on it). `validate --deep`
applies the reserved-name rejection unconditionally, since the linker
never writes `.dp` for re-ingestion.

The provenance flag answers "did the LINKER produce this decl," not
"is there linked content in this check unit." The linker re-mangles
the names of decls it admits at each boundary: the package/source
rewrite (`rewrite_module_decls`, used by `check`/`build`/the library
context) rewrites every binding name through `internal_name`. The
eval/test entry rewrite (`rewrite_eval_module_decls`, used by `chelis
test`, `chelis eval`'s in-context path, module-init, and the batch
batch path) keeps user-authored binding names (so eval can report
roots and the test runner can select synthetic roots by name) and
therefore rejects the reserved format at that boundary directly: a
user entry/test decl whose name matches the reserved format is a
declaration error regardless of the surrounding linked flag, because
user entry decls are never linker output. This pairing — every
flag-TRUE boundary either re-mangles the names it admits or rejects
the reserved format on the user decls it admits — is the structural
invariant that prevents a hand-authored mangled name from self-keying
to a victim module on any surface.

**The rejection set.** Outside the defining module, each of the
following is a `CheckErrorKind::OpaqueTypeViolation`. Every rejection
returns the expression's TRUE type, so a violation never cascades into
secondary type errors:

1. Record-literal construction (`Probability { value: x }`).
2. Positional constructor application (`Meters(x)`).
3. Bare constructor reference (`grab = Probability`): the constructor
   binding itself is hidden, including nullary constructors.
4. Pattern inspection: `pat-record` and `pat-ctor` patterns naming an
   opaque type's constructor.
5. Field access and Deep `record-update`. Targets whose type is still
   an unresolved variable when the site is inferred are recorded in a
   deferred ledger and re-checked after def-level resolution; a target
   that is NEVER pinned (e.g. an unannotated accessor lambda that
   let-generalization makes polymorphic) is rejected fail-closed when
   the check unit declares any opaque type.
6. Forging: Deep `cast` into the type (both `t-prim` and `t-adt`
   target shapes), `cast` out of an opaque value, and
   `{type: (t-adt ...)}` literal metadata -- which is reachable from
   BOTH surfaces, because Surf expression ascription
   (`0.5 : Probability`) and block-binding ascription desugar to
   exactly that metadata.

**The sixth rejection (unexported references).** An out-of-module
reference to an *unexported* binding of the defining module whose
signature mentions the opaque type is rejected -- in the result, in
any parameter, in function-typed parameter domains, or as a
non-function binding's type. "Mentions" is containment chased through
named type definitions (an unexported `helper -> WrapRec` where the
non-opaque `WrapRec` carries a `Probability` field mentions
`Probability`). A defining module with no `export` decl is fully
sealed for its T-mentioning bindings. Names the checker cannot
attribute to a module with a known export set are never flagged
(fail-open), so pipelines that do not carry export information cannot
reject legitimately exported producers.

**Alias transparency.** Opacity is keyed to the nominal registry
entry. A transparent alias (`type P2 = Probability`) resolves to the
nominal type at construction heads, cast targets, and literal
ascriptions, so aliases cannot launder any rejection.

**Macro attribution.** Macro expansion is in-place: an expansion lands
in the CALLER's module subtree and is checked under the caller's
module key. A macro defined in the defining module but expanded
outside it is rejected at the call site (fail-closed).

**Duplicate declarations.** A same-name `deftype` in another module is
already a `DuplicateDefinition` error; that rejection is part of the
opacity invariant set (nominal keying would otherwise be
last-write-wins).

**Exhaustiveness over opaque scrutinees.** Outside code may match an
opaque scrutinee only with irrefutable patterns; with §2.4's
irrefutable-arm rule, `| x =>`, `| q @ x =>`, and `| _ =>` all cover
such a match without naming constructors.

**Equality over opaque values.** Structural `eq` and `neq` ([05-OP-36]) are
admitted on opaque values outside the defining module, since equality
inspects no field through the surface.

**Error contract.** The violation message names the type, the defining
module, and the exported producers of that module with signatures;
location context is the enclosing def name embedded in the message:

```
in def `bad`: record construction of opaque type `Probability`
outside its defining module `stats.prob`; exported producers of
`stats.prob`: probability: (f32) -> Probability
```

On the reef package surface the message renders user-facing
(de-mangled) names, not the package linker's internal
`Pkg__<pkg>__<Module>__<Name>` forms: type, def, binding, and producer
identifiers show their trailing user-written segment, and the defining
module shows its source module path (`Demo.Types`) with the package
prefix stripped. The same out-of-module construction in a package
named `opq` reads `in def `bad`: ... opaque type `Probability` ...
defining module `Demo.Types`; exported producers of `Demo.Types`:
probability: (f32) -> Probability`.

**Solver-free.** Opacity is a module-identity check inside ordinary
inference. `chelis check` stays solver-free: the optional declared
invariant (RFC D-WF and later workstreams) is never evaluated by the
checker.

**Gating.** `chelis check` is a scorer-with-exit-code: for a file target it
always reports a fitness score and the full error list, and its exit code
mirrors that list (`0` iff empty, non-zero otherwise). A directory
target reports one score per checked file inside the envelope of §6.4
§ Directory mode, and [04-FIT-25] states its exit rule in the same terms.
A declaration error such
as `OpaqueTypeViolation` is therefore visible on `check` (a non-zero exit
with the error listed), never a silent score-1 pass. The front-end
surfaces that consume a program -- `chelis build`, `chelis eval --file`,
and the in-process check entry the prove pipeline calls -- gate on a
non-empty error list and refuse to proceed. `chelis validate` is a
structural Deep/Surf well-formedness validator and does not run the type
or opacity checker, so it does not gate on these semantic declaration
errors; use `check`/`build`/`eval` for that.

Before build-specific eval-only removal or reachability pruning, `chelis build`
SHALL finish type, effect, and linearity checks for every definition in its
selected source or complete linked Reef package target. A semantic error in
an unreachable selected definition SHALL fail the build. Files outside the
selected source or linked target SHALL remain outside this semantic gate.

#### 2.5.1 Invariant Declaration Well-Formedness

An opaque type may carry one declared invariant (Surf:
`@invariant(<binder>) <expr>`, spec/02 §P16a; Deep: the `invariant` and
`invariant_amenability` metadata keys, spec/03 §2.2). The checker runs a
declaration-time **well-formedness** pass over Deep (covering both `.ch`
post-desugar and raw `.dp`). The authoritative design record is
`spec/design/opaque_invariants_rfc.md` (D-WF). Each failure is an
`OpaqueTypeViolation` declaration error:

- **Opaque required.** An `invariant` key requires `opaque: true`
  (assumption injection is unsound for a forgeable type).
- **Value class.** The representation must be exactly one record-shaped
  variant; every field must be in the invariant value class: a **numeric or
  boolean** scalar primitive (`f32`, `f64`, the signed integer widths,
  `bool`), a fixed-shape `f32`/`f64` tensor (every dimension literal), or
  a nested single-variant record whose fields are themselves value class.
  `List`, function types, parameterized records, symbolic tensor
  dimensions, multi-variant ADTs, and **non-numeric scalar primitives
  (`string`, `f8e4m3`) or non-`f32`/`f64`-element tensors** are rejected,
  naming the field. The value class is exactly the set of representations
  the prover can mechanically verify an invariant over: `chelis check` and
  `chelis prove` share one definition
  (`chelis_types::invariants::invariant_value_class_prim`), so a
  representation the checker admits always has its producer obligations
  collected -- it is never silently dropped (covered-or-rejected).
- **Predicate grammar.** The predicate admits: literals; the binder and
  its field projections; arithmetic (`+ - * /`); comparisons;
  `and`/`or`/`not`; `if`; the whitelisted intrinsics
  `abs`/`min`/`max`/`sqrt`/`exp`/`log`/`sin`/`cos`; `sum` over a binder
  field projection; and references to in-module zero-argument constant
  defs. Anything else (general calls, `match`, lambdas, other tensor
  ops, effects) is a declaration error. The grammar admits partial
  functions (`/`, `log`, `sqrt`); totality is not guaranteed here (the
  partial-eval semantics are pinned downstream, RFC D-WF).
- **Free variables.** Every free reference must be the binder or an
  in-module zero-argument constant def.
- **Boolean-shaped.** The predicate must be boolean at the top.
- **Amenability recording.** `invariant_amenability` is recorded at
  desugar (`chelis_pred::classify_predicate`) and the checker
  **recomputes** it, erroring on a mismatch or a missing key. This
  protects hand-written `.dp`. The language rejects nothing on
  amenability grounds; it records.
- **Exact float equality** in a predicate (`==` over a representation
  field) draws an advisory lint (`invariant-float-equality`), not an
  error: exact float equality starves Tier C generation by design (RFC
  D-STARVE). The advisory points at the tolerance-band idiom.

**Invariant invisible to type checking.** This is not refinement
typing. The checker records the invariant and never evaluates it: there
are no predicates in the typing judgment, no verification conditions at
use sites, and no solver in the check loop. A program whose invariant
is **violated** by an in-module constructor (e.g. constructing a
`Probability { value: 5.0 }` under `value <= 1.0`) still type-checks.
The invariant is consumed by `chelis prove` for derived producer obligations
and assumption injection, not by `chelis check`.

### 2.6 Collection Observation

The read-only container queries `len` and `index` are observational on a
tensor-carrying `List` or `Dict`: they auto-borrow their container argument
and do not consume it, so reading a container's length or an element does not
forbid a later reuse of the container. See `spec/05-risc-primitives.md`
§1.3.1.

---

## 3. Hindley-Milner Inference

### 3.1 Algorithm

Standard Algorithm W with extensions for tensor types. The flow:

1. **Constraint generation:** Walk the typed Deep AST. At each node, generate type equations between the expected type and the actual type.
2. **Unification:** Solve the constraint set. Unification handles type variables, function types, tensor types (with dimension unification), and ADT types.
3. **Generalization:** At `let` boundaries, generalize unconstrained type variables to produce polymorphic types.
4. **Annotation checking:** Where the programmer/agent provided `type` metadata, check that the inferred type is compatible with the annotation.

> **[04-INF-1]** A lambda whose body reaches a semantic typing rule while an
> operand's outer type constructor is still unknown SHALL retain that rule as
> an obligation and SHALL remain monomorphic until an application binds the
> operand. The trigger is semantic: an absent annotation, a synthesized
> wildcard signature slot, an authored bare type variable, and a variable
> derived from such a parameter by projection all remain unknown constructors.
> The first application within the enclosing declaration binds the lambda and
> replays the same semantic rule; every later use has that same instantiation.
> If no application or outer-constructor parameter annotation resolves the
> obligation by that declaration's own boundary, the declaration is a type
> error; later top-level declarations are not binding sites for it, and a
> result annotation alone does not resolve an unknown parameter constructor.
> Ordinary lambdas with no deferred semantic obligation generalize normally.
> A symbolic tensor is not an unknown constructor: declared dimension and
> precision variables remain polymorphic, independently declared rigid
> dimensions remain distinct, and they unify only when an ordinary body
> constraint requires equality.

For a literal pattern against an unresolved flexible scrutinee, [04-PAT-1]
is such an obligation. The lambda stays monomorphic until its first application
determines the scrutinee type, and the pattern is then checked at that type.
If no application determines it within the declaration, the declaration is a
type error rather than a generalized literal-pattern function. An authored
type binder is governed by [04-INF-6] and is checked at every admissible
instantiation.

An operation restriction on an inferred scalar-or-tensor operand constrains
its numeric dtype; it does not turn a dtype bound into a family of
tensor types. An authored `p: Float`, for example, still admits only float
primitive types, never `tensor[..., f32]`. Restrictions transported by
function values retain that distinction. A failing transported restriction
is a `PrecisionMismatch` naming the required family and offending type under
[04-DTYPE-2], rather than requiring the original operation's body or name.

The replay requirement applies to every operation whose result or admission
depends on the resolved operand shape, not to a hand-maintained exception for
one builtin. In particular, a `matmul`, reduction, `expand`, `insert`,
`layer_norm`, `conv`, or `scatter_elements` reached through a bare lambda
parameter is
checked again after the parameter binds. The check used on replay is the
operation's ordinary typing rule, so immediate and deferred applications
cannot acquire different semantics.

#### 3.1.1 Uniform Recursive Instantiation

A recursive binding group is a strongly connected component of the top-level
`def` call graph. A top-level generic function — a `def` whose
checker-recorded signature carries a type variable — may be a member of a
recursive binding group, whether the recursion is direct or mutual.

> **[04-INF-2]** Every recursive call inside a recursive binding group SHALL
> be typed at the caller's own instantiation of the group's type parameters.
> A type argument of the recursive call satisfies this requirement when it
> resolves to the caller's own type parameter, or remains unconstrained and
> is thereby chosen as it. For a group member whose signature type variables
> are introduced by inference rather than authored binders, a fully concrete
> type argument — one containing no type variable — is also admitted: a
> variable-free argument cannot grow the instantiation set. An authored type
> binder admits no such substitute; its recursive arguments are the caller's
> own parameters. The instantiation set of every accepted program is
> therefore finite: every in-group edge maps the group's type parameters
> into the caller's own parameters or into a fixed set of concrete types, so
> an application of a group member at a concrete type application reaches
> only finitely many instantiations.

> **[04-INF-3]** A recursive call whose type application is not admitted by
> [04-INF-2] — in particular one that embeds a type variable inside a larger
> constructed type (polymorphic recursion) — SHALL be a type error. The
> checker is the earliest competent stage for this question per [05-UNS-2]
> (`spec/05-risc-primitives.md` §7); the diagnostic SHALL name the function,
> the caller's instantiation, and the differing recursive instantiation, and
> SHALL carry this atom as its deciding authority per [05-UNS-5]. The rule
> is lane-uniform: because the rejection happens at check time, the eval, C,
> HIP, and Metal paths reject an offending program identically, before any
> lane-specific stage runs.

#### 3.1.2 Top-level value scope

> **[04-INF-4]** A top-level non-function `def` SHALL become visible only after
> its declaration in its exact declaration namespace. A reference to that
> value from an earlier declaration is a type error reported as
> `UnboundVariable`, whichever checker entry receives the program; body-type
> metadata SHALL NOT make that later value visible. A reference to an
> earlier value SHALL resolve, whether it occurs in a value's initializer or
> in a function body. An explicitly typed self-reference, spelled either
> `x: T = x` or `x = (x : T)`, declares an external input rather than reading
> an eager value; its type is available only while checking its own
> declaration, and the binding becomes visible to later declarations
> afterward. The untyped spelling `x = x` is an ordinary eager self-reference
> and is not an external input declaration. Local `let` bindings are
> sequential per `spec/03-deep-syntax.md` §6.2. Function recursion and
> inference groups remain governed by [04-INF-2] and [04-INF-3], and an eager
> value cycle remains a type error reported as `CycleDetected`.

#### 3.1.3 Signature holes and authored binders

A signature's type expression can carry two kinds of variable. A wildcard,
`(t-var {} _)` or its dimension and rank spellings, is an inference hole
(`spec/03-deep-syntax.md` §2.5); the desugarer synthesizes one for every
omitted parameter or result annotation of a `def` that carries at least one
annotation (`spec/02-surf-syntax.md` §5.2). A named type, dimension, or
rank variable is a binder only when listed in the declaration's explicit
binder list. The two are different objects and the checker treats them
differently.

> **[04-INF-5]** A wildcard slot in a declaration's signature, whether
> written or synthesized for an omitted annotation, is an inference hole and
> not a type binder. It SHALL NOT be quantified, and it confers no type on
> the declaration: the declaration's type at that slot is the type its body
> determines, generalized together with the body under §3.1. Every reference
> to the declaration SHALL be typed at that body-determined signature,
> whichever checker entry receives the program and wherever the reference
> sits relative to the declaration, so no reference can observe the hole
> before the body has filled it. A reference whose use disagrees with the
> body-determined slot is a type error at the reference; a body is never
> narrowed to satisfy a reference. Inside a group of declarations inferred
> as one unit, whether a recursive binding group of §3.1.1 or a component
> that the reference graph closes through a top-level value, an in-group
> reference is typed at the member's provisional monomorphic type, as
> [04-INF-2] provides for a recursive call.

> **[04-INF-6]** An authored type, dimension, or rank variable listed in a
> declaration's binder list has one scope throughout that declaration's body.
> Every type-position occurrence of that name within the body SHALL resolve to
> the same binder, including a tensor precision slot in a lambda parameter,
> expression ascription, block binding, nested ADT argument, property
> quantifier type, precondition, predicate body, or expression-valued option.
> A name not declared by an enclosing explicit binder remains subject to
> §5.8.1's closed primitive and undeclared-name rules; no occurrence introduces
> a binder.
>
> A name-resolution rejection in those declaration-owned type positions has
> one diagnostic owner per lexical module, declaration, offending spelling,
> and diagnostic class. A standalone `sig` and its matching definition's
> inline annotations and body type positions share that owner. Repeating the
> same rejected primitive or undeclared type, dimension, or rank spelling
> within that owner SHALL reuse the first diagnostic witness rather than emit
> one diagnostic per occurrence. Different declarations, lexical modules,
> spellings, or diagnostic classes retain different diagnostics in
> deterministic declaration order. This ownership does not extend to an
> independent expression or runtime/type-use failure merely because it names
> the same spelling.
>
> Except for the sole role-sensitive exception below, each such binder is
> universally quantified and rigid: the body SHALL type-check for every
> admissible instantiation. A body constraint that identifies an authored
> binder with a concrete type or shape, with another authored binder of the
> same signature, or with a type or shape containing either is a type error
> reported at the declaration, and the declaration's scheme is its declared
> signature, never a narrowing of it. A wildcard slot that the body resolves
> to an authored binder takes that binder's type. A dtype bound
> ([04-DTYPE-2]) restricts the admissible instantiations without making the
> binder concrete.
>
> The sole role-sensitive exception is §4.4.1: a dimension binder appearing
> only in a function result remains output-inferred. The body may leave it
> unbound or resolve it to a body-internal output, but SHALL NOT pin it to a
> dimension from a declared input or collapse it with an input-position
> dimension binder. A non-function declaration has no function-result position
> and receives no such exception.

An unsuffixed literal binds at its default primitive type
(`spec/02-surf-syntax.md` §P10), so `lt(x, 0.0)` with `x: p` identifies the
binder `p` with `f32` and is rejected under [04-INF-6]; the polymorphic
spelling is the `cast(0.0, p)` override that §P10 names.

#### 3.1.4 Eager initialization references

> **[04-INF-7]** The eager reference set of a top-level eager value is the
> least set that contains every free reference in the value's initializer,
> including a reference inside a lambda body nested anywhere in the
> initializer, and, for every top-level `def` whose name is in the set,
> every free reference in that `def` (a function's body or a value's
> initializer), likewise including references inside nested lambda bodies.
> For this rule and for [04-INF-4], a top-level `def` whose initializer is
> a lambda expression is a function declaration, not an eager value. A
> reference is free when
> it names a top-level declaration rather than a parameter, `let` binding,
> or pattern binder in scope at the reference. The eager value cycle that
> [04-INF-4] reports as `CycleDetected` exists exactly when the value's own
> name occurs in its eager reference set, as a read or as an application,
> other than as the value's own explicitly typed self-reference. The set is
> a syntactic over-approximation: a lambda or a function value that reaches
> an initializer is treated as applied during that initialization whether
> or not the receiving callee applies it. The rule depends only on the
> program's declarations, never on the checker entry that receives the
> program or on the order in which declaration bodies are inferred.

A program rejected only by the over-approximation, one whose initializer
stores a function value that reads the initialized binding and that no
initializer ever applies, is written so the function receives that value as
an argument instead of reading the top-level binding.

> **[04-INF-8]** For a top-level eager value `V`, every non-function
> top-level value in `V`'s eager reference set ([04-INF-7]) SHALL be available
> before `V`'s declaration. If `V` itself does not occur in the set, a member
> declared later than `V` is a type error reported as `UnboundVariable`, naming
> both `V` and the later value, whichever checker entry receives the program
> and before evaluation or lowering begins. The comparison is against `V`'s
> source position, not the position of a function or nested lambda that
> contains the textual reference. If `V` itself occurs in the set,
> [04-INF-7]'s `CycleDetected` verdict takes precedence. Function declarations
> and recursive function groups remain
> governed by [04-INF-2] and [04-INF-3]; a forward function reference is legal
> when every non-function value in its eager closure is already available to
> the initiating eager value. Values imported from an already-checked library
> are available before the current unit and do not participate in this
> source-position comparison.

#### 3.1.5 Explicit generic operation contracts

An operation's static admission requirements determine which operand types it
accepts and any required relationship between its operand and result types.
They include dtype-bound restrictions and collection-constructor requirements.
They are distinct from value-dependent conditions for which the operation's
specification requires a runtime check.

> **[04-INF-9]** A newly authored generic function SHALL declare a type
> contract sufficient for every operation in its body. The body SHALL be
> checked for every instantiation admitted by that contract, using each
> operation's ordinary typing rule. A requirement not entailed by the
> declared contract is a declaration error, not an inferred restriction that
> silently strengthens the contract. Omitting a signature or using signature
> holes SHALL NOT authorize generalizing unresolved operation-admission
> requirements into an implicit constrained generic contract. This rule
> applies to named functions and anonymous function abstractions, including
> local functions and functions that escape through results or higher-order
> values. Ordinary local inference and monomorphic obligation replay remain
> governed by [04-INF-1] and [04-INF-5]; unconstrained polymorphism remains
> admissible. An authored named binder remains rigid under [04-INF-6], even
> when a local call supplies a concrete argument.
>
> An already-checked function value retains its contract when aliased,
> instantiated, generalized, passed or returned, placed in an aggregate,
> imported, or serialized and restored as checked metadata. These operations
> SHALL NOT erase its restrictions or require its body to be available for
> validation. A newly authored wrapper SHALL itself meet this rule; merely
> aliasing an existing function does not require redeclaring its contract.
> A builtin function value carries the contract of its governing operation
> specification. Calls SHALL satisfy the callee's checked contract, including
> in recursive and indirect calls, without reconstructing admission
> requirements by inspecting the callee's body.

For example, a generic function applying `mean` to `tensor[3, p]` requires
`p:Float`; a bare `p` or `p:Numeric` admits types that the operation rejects.
A generic length function accepting `List[a]` can leave `a` unconstrained:
the `List` constructor supplies the admission information that `len` needs.
In contrast, `def size(x) = len(x)` cannot become an implicitly constrained
generic function; a later top-level call with a list does not repair the
declaration. Neither does a signature promising `a -> i64` for arbitrary
`a`. An unannotated local lambda applying `len` may still bind monomorphically
to a list at its first application within the enclosing declaration under
[04-INF-1]. The identity function needs no operation-admission restriction
and may generalize without an annotation.

Insufficient dtype-bound admission is a `PrecisionMismatch`. Other
insufficient static operation-admission contracts retain the diagnostic kind
required by the operation's specification, or use `TypeMismatch` when that
specification assigns no more specific kind.
The declaration diagnostic identifies the function or anonymous abstraction,
the insufficient operand type or binder, and the required restriction.
These declaration checks do not replace the runtime checks required by an
operation's value-dependent extent or value semantics.


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

A dimension variable minted by an application's instantiation that unifies
with a runtime extent `*` and that no argument of that application binds to a
literal or named dimension denotes that runtime extent and is `*` in the
application's result; a literal or name another argument of the same
application binds to it is a claim on the runtime extent, checked by a §4.7
guard.

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
    For each (arm {} pᵢ gᵢ bᵢ):
        Γ, bindings(pᵢ, τₛ) ⊢ gᵢ : bool      when gᵢ is not ()
        Γ, bindings(pᵢ, τₛ) ⊢ bᵢ : τᵣ
    patterns {pᵢ | gᵢ is ()} are exhaustive over τₛ
    ──────────────────────────────────────
    Γ ⊢ (match {} e arm₁ ... armₙ) : τᵣ
```

`bindings(p, τₛ)` is defined only when `p` is admissible at the scrutinee type
`τₛ`. An inadmissible pattern is a type error at its own arm, not an arm that
contributes no bindings. A `pat-lit` binds nothing and contributes exactly one
constraint on `τₛ`, which [04-PAT-1] states. A guard `gᵢ` carries the
obligation an `if` condition does, discharged by unifying its type with
`bool`; [04-PAT-2] states what it does at run time.

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

**grad:**
If `f` has scalar floating output, `grad(f)` returns gradients only.
The gradient payload shape is:

- one differentiable target => that gradient type directly
- multiple differentiable targets => flat `t-tuple` in target order
- explicit `wrt` on a non-differentiable parameter => type error

The forward value is not bundled into the `grad(...)` result.

#### Operand decisions on an unresolved operand

An operation's operand type is not always known where the operation is
checked. When the operand is still an unresolved type variable — a lambda
parameter awaiting its argument, a signature hole, or a binding that takes its
type from one of those — the operation has not been shown to be ill-typed,
only to be undecided. In that case the decision is **deferred**: it is settled
when the variable is bound, and reaches the verdict the operation would have
reached had the operand carried that type when the operation was first
checked. An operand whose outer type constructor is already known is not such
a case, and is decided immediately whatever remains unresolved inside it; a
tensor with unresolved dimension variables is decided at once.

The deferral is sound only because a variable that is never bound is still
rejected. After a function body's inference completes, every deferred operand
decision that has not settled is rejected with the same diagnostic as an
operand of a concretely inadmissible type. Deferral never admits an operand
the operation would otherwise reject, and the verdict never depends on the
order in which inference reaches the operand.

(A small number of operations do not implement this yet; they are tracked on
[#1489](https://github.com/Chelis-Lang/chelis/issues/1489).)

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
  not impose a distinct-from-literal constraint.

### 4.2 No Broadcasting

Chelis does NOT support implicit broadcasting. All rank and dimension manipulation must be explicit via `insert`, `expand`, `reshape`, `permute`.

```scheme
;; WRONG: dimensions don't match
;; tensor[batch, hidden, f32] + tensor[hidden, f32]  →  TYPE ERROR

;; CORRECT: explicit insert
;; tensor[batch, hidden, f32] + insert(tensor[hidden, f32], 0, batch)
```

Rationale: Broadcasting masks fatal dimension errors in AI-generated code. Named dimensions + no broadcasting means the type checker catches transposition bugs, broadcasting bugs, and shape mismatches at compile time.

### 4.3 How RISC Primitives Transform Tensor Types

| Operation | Input type(s) | Output type | Dimension rule |
|---|---|---|---|
| `add`, `mul`, `div`, `floor_div`, `trunc_div`, `max_elem`, `wrap_add`, `wrap_sub`, `wrap_mul` | `tensor[D, p]`, `tensor[D, p]` | `tensor[D, p]` | Dimensions must match exactly; each operation's dtype domain remains as specified in spec/05 |
| `cmplt`, `lt`, `gt`, `gte`, `lte` | `tensor[D, p_numeric]`, `tensor[D, p_numeric]` | `tensor[D, bool]` | Dimensions and numeric dtypes must match exactly |
| `eq`, `neq` | `tensor[D, p]`, `tensor[D, p]` | `tensor[D, bool]` | Dimensions and active tensor element dtypes must match exactly; the scalar and recursive host-value forms are governed by [05-OP-36] |
| `neg`, `recip`, `exp`, `log`, `sin`, `cos`, `tan`, `atan`, `sqrt`, `abs`, `floor`, `ceil`, `round` | `tensor[D, p]` | `tensor[D, p]` | Dimensions preserved; each operation's dtype domain remains as specified in spec/05 |
| `is_nan`, `is_finite`, `is_infinite` | `tensor[D, p_float]` | `tensor[D, bool]` | Dimensions preserved |
| `sum(x, axis=k, accumulator=a)` | `tensor[d₁,...,dₙ, p]` | `tensor[d₁,...,d_{k-1},d_{k+1},...,dₙ, sum_result(p,a)]` | Remove dimension at axis k; `sum_result` is §5.7.1's result-precision rule |
| `count(x, axis=k)` | `tensor[d₁,...,dₙ, bool]` | `tensor[d₁,...,d_{k-1},d_{k+1},...,dₙ, i64]` | Remove dimension at axis k; the named multi-axis form removes every selected axis |
| `mean`, `max_reduce`, `min_reduce`, `prod_reduce` `(x, axis=k)` | `tensor[d₁,...,dₙ, p]` | `tensor[d₁,...,d_{k-1},d_{k+1},...,dₙ, p]` | Remove dimension at axis k; `mean` additionally requires float `p` |
| `argmax_reduce`, `argmin_reduce` `(x, axis=k)` | `tensor[d₁,...,dₙ, p]` | `tensor[d₁,...,d_{k-1},d_{k+1},...,dₙ, i64]` | Remove dimension at axis k |
| `cumsum(x, axis=k)` | `tensor[D, p]` | `tensor[D, sum_result(p,default(p))]` | Dimensions preserved; every prefix uses §5.7.1's default sum accumulator |
| `sort(x, axis=k)` | `tensor[D, p]` | `(tensor[D, p], tensor[D, i64])` | Values and stable source indices preserve the input dimensions |
| `where(cond, yes, no)` | `tensor[D, bool]`, `tensor[D, p]`, `tensor[D, p]` | `tensor[D, p]` | All three dimension lists match exactly |
| `clamp(x, lower, upper)` | `tensor[D, p]` plus rank-zero or `tensor[D, p]` bounds | `tensor[D, p]` | Bounds are explicitly rank-zero or shape-equal; no broadcasting rule is inferred |
| `diagonal(x, axis1, axis2)` | `tensor[D, p]` | `tensor[D_diagonal, p]` | Remove axis2 and replace axis1 by the smaller selected extent as [05-OP-33] specifies |
| `trace(x, axis1, axis2)` | `tensor[D, p]` | `tensor[D_without_axes, sum_result(p,default(p))]` | Remove both axes; sum the selected diagonal with the default accumulator |
| `einsum(equation, left, right, accumulator=a)` | two `tensor[..., p]` operands | `tensor[D_output, sum_result(p,a)]` | Explicit output labels define `D_output`; [05-OP-33] owns label/extent legality |
| `reshape(x, shape)` | `tensor[D_old, p]` | `tensor[D_new, p]` | Product of dims must match. New dims are `d-lit` or `d-name` (user-specified) |
| `permute(x, axes)` | `tensor[d₁,...,dₙ, p]` | `tensor[d_{axes[0]},...,d_{axes[n-1]}, p]` | Reorder dimensions |
| `expand(x, axis, size)` | `tensor[D, p]` | `tensor[D', p]` | Set the size-1 dimension at `axis` to `size`; rank unchanged |
| `insert(x, axis, size)` | `tensor[D, p]` | `tensor[D_plus, p]` | Add one dimension of extent `size` at `axis`; rank increases by one |
| `pad(x, ...)` | `tensor[D, p]` | `tensor[D', p]` | Padded dimensions get new sizes (d-lit) |
| `cast(x, new_p)` | `tensor[D, p]` | `tensor[D, new_p]` | Dimensions preserved, precision changes |

### 4.4 Dimension Polymorphism

Functions can be generic over dimensions using dimension variables:

```scheme
;; In Surf:
;; def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32]

(defsig {} transpose (a b)
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
;; def f[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = y
;; TYPE ERROR: the body returns tensor[m, f32] but the declared return is
;; tensor[n, f32]; n and m are distinct rigid dim parameters.
```

#### 4.4.1 Return-Only Dim Parameters

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
  `examples/hello_tensor.ch` shape: `def main[n]() -> tensor[n, f32]`
  whose body builds a `tensor[3, f32]`.

At run time a return-only dim parameter takes its value from the first site,
in evaluation order, that produces an extent for it, such as a block binding
whose ascription names it. Every later site that names it, the declared
result included, is a claim under §4.7: execution checks equality and traps
`Domain` on mismatch.

What the body must **not** do is couple the promised-independent output
dimension to the caller-visible input world. Both of the following are
`DimensionMismatch` type errors:

- **input-coupled pin**: the return-only dim parameter resolves to a
  concrete literal that occurs (after unification) in a declared
  parameter position;
- **input-coupled collapse**: the return-only dim parameter unifies with
  a distinct param-position declared dim parameter.

```scheme
;; def f[k](a: tensor[2, f32]) -> tensor[k, f32] = a
;; TYPE ERROR: the body pins the return-only dim parameter k to the
;; parameter's concrete Lit(2); the signature promised an output
;; dimension the body does not derive from the inputs.

;; def f[n, m](x: tensor[n, f32]) -> tensor[m, f32] = x
;; TYPE ERROR: the return-only dim parameter m collapses with the
;; param-position dim parameter n.

;; def make[n]() -> tensor[n, f32] = to_tensor([1.0, 2.0, 3.0])
;; OK: output-inferred. The body produces a body-internal tensor[3, f32]
;; and the scheme resolves n := 3; no input dimension is involved.
```

Three deliberate boundaries of this rule:

- a body-internal concrete pin whose literal does *not* occur in any
  declared parameter position is tolerated even when the def has
  parameters (`def f[k](x: tensor[2, f32]) -> tensor[k, f32] =
  to_tensor([1.0, 2.0, 3.0])` is accepted with `k := 3`) — the guard
  compares resolved dimensions, not provenance, so a body-internal
  literal that happens to *equal* a parameter dim is conservatively
  rejected, and one that differs is conservatively accepted;
- coupling through a *named* symbolic dim
  (`def f[m](x: tensor[batch, f32]) -> tensor[m, f32] = x`, which binds
  `m` to `batch`) is not flagged: `Dim::Name` unifies permissively by
  design and no declared dim parameter participates.
  (When a param-position declared dim parameter *also* resolves to the
  same name — e.g. `def f[n, m](x: tensor[n, f32],
  y: tensor[batch, f32]) -> tensor[m, f32] = add(x, y)` binds both `n`
  and `m` to `batch` — the collapse rule above does fire, because the
  two declared dim parameters now share a resolution.);
- coupling through a *wildcard* param dim is invisible
  (`def f[k](b: tensor[*, f32]) -> tensor[k, f32] = b` is accepted):
  the wildcard unifies permissively without binding (§4.5), so `k`
  stays unbound and generalizes even though the returned value's
  runtime dimension is the input's. This follows §4.5's wildcard
  permissiveness.

### 4.5 The Wildcard Dimension

`(d-name {} *)` represents an unknown/dynamic dimension. Produced by operations where the compiler cannot statically determine the dimension:

- Concatenation along an axis whose operand extents are not statically
  summable: `concat(tensor[batch, seq1, f32], tensor[batch, seq2, f32],
  axis=1)` → `tensor[batch, *, f32]` (named `seq1`/`seq2` cannot be
  summed; see §4.5.4 for when concat DOES produce a concrete extent)
- Data loading with dynamic shapes
- Results of control flow where branches have different known dimensions

A wildcard dimension unifies with any other dimension (like a variable) but is NOT generalized — it's a permanent "I don't know." To restore named-dimension checking after a wildcard, use an explicit annotation.

The spelling `*` does not declare a dimension binder. Distinct wildcard
occurrences impose no equality on their runtime extents, including across
axes or parameters of one function; genuine repeated names still impose
their declared equality obligations (§4.7).

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
rank-erased element type is not part of the language; users
who genuinely need to carry mixed-rank tensors through a list must
reshape elements to a common rank before listing, or use a sum type
that names each rank as a separate variant. A rank-polymorphic
`List[tensor[k, f32]]` does not let `k` range over shape vectors; the
named-dim safety guarantee in §4.2 takes precedence over that additional
flexibility. A *constrained,
name-preserving* shape spread that does **not** sacrifice that guarantee
is admitted for top-level tensor signatures by Tier-3 rank polymorphism
(§4.5.3) — but it remains barred from `List`/ADT element position, so the
rank-uniform-list guarantee above is unaffected.

```chelis
;; WRONG: rank-1 and rank-2 elements in the same List[tensor[k, f32]]
;; def make_mixed[k]() -> List[tensor[k, f32]] = {
;;   a = to_tensor([1.0f32, 2.0f32])
;;   b = to_tensor([[1.0f32, 2.0f32],
;;                  [3.0f32, 4.0f32]])
;;   [a, b]  ;; DimensionMismatch: list element rank mismatch
;; }

;; CORRECT: flatten the rank-2 element to rank-1 first
;; def make_uniform[k]() -> List[tensor[k, f32]] = {
;;   a = to_tensor([1.0f32, 2.0f32])
;;   b_flat = reshape(
;;     to_tensor([[1.0f32, 2.0f32],
;;                [3.0f32, 4.0f32]]),
;;     [4i64])
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
`concat([a, b], axis)` pattern accepts elements whose concrete axes differ:

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

Two consequences of the rule are:

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

Two boundary properties complete the rule:

- **The `(concrete, wildcard)` join is head-biased.** A `Cons` step
  resolves the joined axis to whatever the *head* (the element being
  prepended, i.e. the earlier list position) resolves to. So
  `[tensor[2, f32], tensor[*, f32]]` joins to element `tensor[2, f32]`
  (the concrete head absorbs the wildcard tail), whereas the reordered
  `[tensor[*, f32], tensor[2, f32]]` joins to `tensor[*, f32]` (the
  wildcard head erases the concrete tail). Under a `def f[k](a:
  tensor[2, f32], b: tensor[*, f32]) -> List[tensor[k, f32]]`
  annotation **both orderings are rejected**, but through different
  guards: the concrete-element ordering pins the return-only `k` to the
  parameter's `Lit(2)` and trips the §4.4.1 return-position rigidity
  check (the same rule as the non-list
  `def f[k](a: tensor[2, f32]) -> tensor[k, f32] = a`), while the
  wildcard-element ordering trips the join-origin-wildcard uniformity
  check below. The head bias itself is still observable in *which*
  diagnostic fires. Genuinely-mismatched *concrete*
  heads/tails still widen to `*` regardless of order (the ragged-axis
  arm).
- **The uniformity check recurses through nested `List` wrappers.** It compares the declared and body element axes of a
  `List[tensor[..]]`, and when the element is itself a `List` it descends
  into it, so an inner rigid/named dim under `List[List[tensor[k, f32]]]`
  — at any nesting depth — is checked too. A wildcard tensor nested under
  `List[List[tensor[k, f32]]]` is therefore rejected, the same as the
  single-level `List[tensor[k, f32]]` case: `List[List[tensor[k]]]` still
  promises every innermost element shares length `k`.

#### 4.5.3 Name-Preserving Rank Polymorphism

Rank-erasing shape variables are forbidden because *rank erasure masks
transposition bugs*. Chelis instead admits a **constrained, name-preserving** shape
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
;; def reduce_seq[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)
;;   tensor[batch, seq, hidden] -> tensor[batch, hidden]
;;   tensor[a, b, seq, c]       -> tensor[a, b, c]
```

Multiple axes may be reduced in one call. A rank-polymorphic operand uses
unique named axes — `sum(x, seq, head)` or `count(mask, seq, head)`. A
concrete-rank operand may instead use one or more compile-time i32 positional
axes — `sum(x, 1i32, 3i32)` or `count(mask, -1i32, 0i32)`. Each positional
axis applies §4.7's one-step negative normalization against the original rank;
every normalized axis must be in range and unique. Named and positional axes
cannot be mixed in one call. For value reductions, the result is obtained by
composing single-axis reductions in the
canonical order below. The variadic form is order-insensitive:
`sum(x, head, seq)` and `sum(x, seq, head)` produce the same result because
axis arguments are sorted by their positions in the original operand, from
highest position to lowest, before lowering. The resulting single-axis
composition owns the exact value, trap, NaN-selection, and adjoint behavior;
source spelling order never does. A dedicated multi-axis node such as
`Count` stores this same strictly descending normalized original-position
vector rather than the source order. The variadic form is defined for the value
reductions `sum`, `mean`, `max_reduce`, `min_reduce`, and `prod_reduce`.
`count` instead resolves the complete unique axis set and executes the
single dedicated multi-axis reduction of [05-OP-29]; it cannot compose
single-axis `count` operations because the first result has dtype `i64`.
The variadic form is **not** defined for the
index-returning reductions `argmax_reduce`/`argmin_reduce`: an index along one
axis is not composable with a second reduction, so a variadic call on those is
a hard error. Every named axis must resolve per the rules above; duplicate
names, duplicate normalized positional axes, a dynamic positional value, or a
mixed named/positional list is a hard error. Inside a `..r` body, the
Body-Discipline admits exactly the named-axis reduction operations whose
signatures preserve symbolic-rank trackability, including `sum`, `count`,
`mean`, `max_reduce`, `min_reduce`, and `prod_reduce`, with their ordinary
operand and result dtype rules. At lowering, value reductions
desugar to the canonical highest-original-position-first composition, while
`count` lowers once with the complete resolved axis vector.

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

**Named-axis insert (`R+1`).** The inverse arithmetic direction: `insert`
adds a *named* axis in a rank-polymorphic way when its axis argument is a
dimension name rather than an integer. Two call forms are admitted:

```chelis
;; insert a trailing named axis (the new axis goes after every existing axis):
;; def add_axis[rest](x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = insert(x, one, 1)
;; insert immediately BEFORE an existing named anchor (4-arg form):
;; def widen[pre, post, c](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, c, seq, ..post, f32]
;;   = insert(x, c, 5, seq)
```

- `insert(x, new, size)` — `new` is a bare dimension name: add a new
  **trailing** axis named `new` with extent `size`. The symbolic output is the
  operand's row form with `new` appended.
- `insert(x, new, size, anchor)` — additionally name an **anchor**, an existing
  named axis of the operand; the new axis is inserted immediately *before* the
  anchor. Leading-end insertion is expressible exactly when the row begins with
  a named anchor (`tensor[first, ..rest]` + `insert(x, c, k, first)`); a row
  that begins with a spread has no leading anchor and admits trailing or
  anchored insertion only.

Both forms keep unification unitary: the insertion point is either an end of
the row or a position fixed by a named anchor located uniquely in the operand.
Hard errors (`DimensionMismatch`, never a guessed placement):

- the inserted name already names an axis of the operand (a duplicate dim name
  would make every later by-name lookup ambiguous). This holds through
  call-site rank monomorphization too: when the inserted name survives into
  the result row, a caller whose spread-covered axes include that name is
  rejected at the earliest fully instantiated typed boundary. This includes
  a name consumed later in the body by insert-then-reduce and a collision
  originating from a single-letter dimension variable; neither may be
  deferred into lowering or resolved by guessing;
- the anchor is absent from the operand's row, or ambiguous (appears more than
  once);
- a *positional* (integer) insert axis on a rank-spread operand — an index is
  meaningless at symbolic rank, so the positional form is concrete-rank only;
- insertion *strictly inside* an opaque spread has no anchor and is not
  expressible: the computed output row places the new axis only at an end or
  at an anchor, so a declared result such as `tensor[..lo, c, ..hi, f32]` from
  an operand `tensor[..rest, f32]` fails row unification (differing anchor
  structure) and is rejected.

The inserted axis is a *named* dim: declared result types refer to it by name
(`tensor[..rest, one, f32]`). A bare identifier in the axis slot is read as a
dimension name only when it is **not bound in the value environment**: a bound
`i32` variable is a runtime value and keeps the static-axis rule:
`insert(x, ax, 4i64)` with `ax: i32` is an error, never a trailing insert
of an axis named `ax`. The `size` argument is any expression of exactly type
`i64`. A static negative value is a type error; a runtime negative value
traps `Domain`. The inserted named dimension carries the executed extent. A
literal or named extent claimed by a surrounding result type is either proven
equal statically or protected by an execution-time equality guard under
§4.7.2; no backend may require a literal size. At lowering, the named insertion point
is resolved against the monomorphized operand dims (trailing → operand rank;
anchored → the anchor's index), mirroring named-axis reduction.

**Soundness (§4.2).** Order is preserved (shapes stay ordered positional
sequences — never unordered "rows"); the reduced axis is a retained name; and a
rank-poly def body is restricted by the §4.2 Body-Discipline check to
*name-trackable* operations only — shape-identity (elementwise) ops,
named-axis reductions, named-axis insert, and the key derivations
[05-OP-69]–[05-OP-72]. Each tensor result of a key derivation preserves
the complete operand shape as an ordered prefix; only `split_keys` appends
an axis, at the trailing end, with the count extent. No derivation reorders
or removes an axis, including inside an opaque spread. A *positional* shape-rewriter
(`permute`, `reshape`, `matmul`, positional `gather`) is rejected inside a
`..r` body: its output shape is not name-trackable at symbolic rank, so it
could hide an untracked transposition. For the name-tracked ops the procedural
inference arm is the real gate: it rejects a positional index at symbolic rank
and a non-existent/ambiguous/duplicate axis name, so no transposition can slip
past. This is what keeps the §4.5.1 transposition-safety guarantee intact while
admitting reductions and named-axis expansion at symbolic rank.

#### 4.5.4 Concat Result Typing

`concat(list, axis)` over a `List[tensor[...]]` types its result from the
joined element type (§4.5.2), the **concat axis value**, and the
**statically-visible elements**. A list's length is not part of its
type, so what survives depends on the expression: a literal `Cons`/`Nil`
chain at the concat site exposes every element, while a variable bound
to a list literal in an enclosing `let` (or top-level bind) carries only
its literal **length** to the concat site. The rules, in order:

1. **Direct per-element sum.** Axis is a static integer
   literal (negative axes normalize against the element rank) and the
   list is a literal chain at the concat site whose every element
   carries a **literal** extent on the concat axis: the concat axis
   types `Lit(sum of the extents)` — ragged lists included
   (`[tensor[2], tensor[3]]` on axis 0 types `tensor[5]`; an axis-0
   concat of two `[1, m]` rows types `[2, m]`). Any non-literal element
   extent (a name, a variable, a wildcard) makes the sum unknown and
   the axis falls to rule 3. Every other axis is the joined element
   type's axis unchanged.
2. **Binding-carried count.** Axis is a static literal and the list is
   a variable carrying a literal length `n`: the concat axis types
   `Lit(k * n)` when the **joined** element's concat-axis dim is a
   literal `k` — uniform extents only, since the §4.5.2 join widens
   mismatched literals to `*` before the concat rule sees them.
   Soundness note: the head-biased `(concrete, wildcard)` join boundary
   (§4.5.2) is inherited on this path (a wildcard tail behind a
   concrete head still counts as `k`), and the runtime dim guards keep
   any violation loud, never mis-sized. On the DIRECT path of rule 1
   the head bias is gone: a wildcard element makes the sum unknown.
3. **Honest wildcard.** Axis is a static literal but neither rule above
   produces a literal (non-literal element extents, unknown count — a
   function result, a parameter, a `split` output): the **concat
   axis** — not the last axis — types `(d-name {} *)`; every other axis
   is the element type's axis unchanged.
4. **Out-of-bounds axis.** A static literal axis outside the element
   rank (after negative-axis normalization) is a check-time
   `DimensionMismatch`.
5. **Dynamic axis.** A non-static axis expression is legal (the host
   runtime concatenates along a computed axis): every axis of the result
   types `*` at the element rank — rank is still statically known
   (§4.5.1 rank uniformity), per-axis extents are not.

### 4.6 Property Definitions

Surf `@property` declarations type-check as ordinary functions whose result
type is `bool`. Every binder must have an explicit type. The `where`
preconditions must type-check as `bool` expressions in the binder scope; the
predicate body must type-check as `bool`.

An optional `[..]` list after the property name is the declaration's complete
explicit type, dimension, and rank binder list under [04-INF-6] and §5.8.1.
Those binders are rigid and scope the property quantifier types, preconditions,
predicate body, and expression-valued options. Duplicate, forbidden, unlisted,
and dtype-bounded names follow the same rules as a function
declaration. That same binder scope applies to tensor precision slots
throughout those property positions.

Desugaring emits a `defsig` with that explicit binder list, the quantifier
types, and `bool` result, plus an ordinary `def` carrying the property metadata specified in
`spec/design/chelis_property_spec.md`. Type checking trusts neither the metadata
nor the property annotation; it checks the resulting Deep function against the
signature like any other definition.

### 4.7 Runtime Shape Semantics

Runtime shape values are ordinary typed integer values. Their legality never
depends on whether an expression is literal, inline, let-bound, top-level,
returned by a function, or derived by integer arithmetic. The relevant
built-ins are:

- `shape(x, axis)`: returns the size of `x`'s `axis`-th dimension as an
  `i64` value (`spec/05-risc-primitives.md` [05-DIM-2]). `axis` is any
  expression of exactly type `i32`. The result is a runtime scalar, not a
  symbolic dim reference.
- `expand(x, axis, size)`: set the size-1 dimension at position `axis` to
  width `size`, where `size` is any expression of exactly type `i64`. The
  rank is unchanged.
- `insert(x, axis, size)`: add a new dimension of width `size` at position
  `axis`, where `size` is any expression of exactly type `i64`. When `axis`
  is a dimension *name* instead of an integer, the call is the named-axis
  insert form (§4.5.3): it adds a new named axis at the trailing end, or —
  with a fourth `anchor` argument — immediately before an existing named axis.
- `reshape(x, shape_list)`: reinterpret the memory of `x` against
  `shape_list`, a `List<i64>`; every element may be computed at runtime.
- `reduce_window_*`: consume runtime `List[i64]` window and stride values
  under [05-RWIN-1..2]. Their values and list lengths are validated at
  execution when not statically known.

Static knowledge improves diagnostics and symbolic dimension propagation; it
does not define a smaller executable language. A violation proven from
literals is a type error. Literals that become visible only when a call is
inlined prove it just the same when the lowered graph fixes the claimed axis
to a different extent: the program is rejected before any execution, on every
lane, and the equality guard §4.7.3 imposes on that axis is decided then rather
than emitted. An axis the lowered graph does not fix keeps its run-time guard.
A constraint that depends on runtime values is checked before allocation or
element access and traps `Domain` or `Overflow` under the owning operation.
Every execution mode observes the same values and traps.

A runtime extent guard is the check that a declared, named, or otherwise
claimed extent agrees with the value actually observed, or that a runtime
extent is non-negative. Each guard is evaluated exactly once, after every
value it compares is available and before the first allocation or element
access whose shape depends on the guarded extent. An interface guard enforces
a declared signature claim on an input tensor's axis or scalar parameter and
compares only interface values (including literals). It is evaluated at
function entry, in declared signature order, before
any other operation of the function runs. An entry that declares no signature
orders those guards by its ABI input-slot order instead: the order in which
the caller supplies that entry's inputs. Whatever rule assigns the slots, the
guard order follows the assigned slots, and never a separate traversal by
binding name, hash iteration, or node identity. A guard that compares a locally
computed value (checked integer arithmetic, a user-function result, or an
extent an operation computes) is evaluated after its producers and takes the
source position of the operation that introduces the guarded extent: an
independent effect or trap that precedes that operation in source order is
observed first, and one that follows it is observed only if the guard passes.
A declared-result guard on an axis produced by a body operation belongs to
that operation, even when the axis's size can be calculated entirely from
interface values before the operation executes. Entry placement applies to
claims on the interface witnesses themselves; the ability to calculate a
later result's extent early does not turn its result claim into an entry
guard. For `insert(x, axis, shape(y, k))` returned under an independent
declared extent, an earlier independent body trap occurs before the result
guard, and a failing result guard names `insert`.
A `cast` takes the placement of the value it casts. Guards ready at the same
source position are evaluated in declaration order. These constraints are the
complete observable contract; a guard and an operation related by neither data
dependence nor source order may be evaluated in either order.
Every execution mode places guards by this rule.

A runtime extent guard is a typed operation-precondition guard under
[04-NUM-9] and is therefore itself the trap-producing primitive. A failing
equality guard and a failing non-negativity guard both raise a `Domain` trap.
Its `<op>` slot is the canonical name of the operation that introduces the
guarded extent: for an interface guard, the
`load` primitive of the later witness in signature order
(spec/05-risc-primitives.md §2.5); for a non-negativity guard, the owning
movement operation; for a declared-result guard, the primitive that produced
the returned value, whether the return expression names it directly or reaches
it through a block tail, a binding, or a callee. A tensor that a builtin
returns, directly or held at any depth in the aggregate it returns, has that
builtin as the producing primitive named by [04-NUM-9]'s `<op>` slot,
whichever operand, element, callback result, seed or iteration supplied it.
The exceptions are [05-OP-54]'s List selections `index`, `take` and `skip`,
which return one contiguous part of their List input selected by an `i64`
index or count: like a pattern, they project, and a tensor they return or hold
keeps its producer. A runtime extent guard's `<prim>` slot is `i64`, because
the result this
guard finalizes is an extent ([05-DIM-1]) and not a tensor element. The
complete user-facing line is therefore
`numeric trap: domain in <op> at i64`, and [04-NUM-9] permits it no prefix
and no suffix. Every lane SHALL also convey, on separate lines accompanying
that trap, the names of the disagreeing sources, the axis, and the value
observed for each; that requirement binds the information conveyed and not
the bytes rendered, and the context is not part of the trap line. The failure
is a value inside its lane under [04-NUM-10] and becomes a process failure
only at that lane's boundary.

For the movement primitives, symbolic-dim pass-through is
**identity-only**: a `stride` axis with literal
step 1 and a `pad` axis with zero padding keep the input's symbolic dim;
every other movement axis — a non-identity literal step or padding, or
any runtime bound — types a fresh `(d-name {} *)` whose extent the
owning op declares and guards at run time (spec/05-risc-primitives.md
§2.4.1). `shrink` has no checker-detectable identity form for a
symbolic axis (a full-axis slice of a symbolic dim necessarily spells
its end as a runtime value), so its symbolic axes always mint fresh
extents.

#### 4.7.1 `shape` axis form

Every form below produces an `i64` value ([05-DIM-2]):

```text
shape(x, 0)
shape(x, cast(0, i32))
shape(x, -1)
shape(x, cast(-1, i32))
shape(x, axis_parameter)
shape(x, computed_int32_axis)
```

The bare and cast literal spellings are equivalent. Every literal or computed
axis first adds the input rank exactly once when negative, so `-1` names the
last axis. A statically known normalized value outside `0..rank` is a type
error (`DimensionMismatch`). A computed normalized value outside that range
traps `Domain` as operation `shape` before reading metadata. An axis whose
type is not exactly `i32` is a type error; no width is inferred or coerced.
The zero-cotangent and target-independent execution rules are [05-OP-7] and
[05-SHAPE-1].

#### 4.7.2 `expand` and `insert` with a runtime size

`expand(x, axis, size)` and `insert(x, axis, size)` each accept any `i64`
`size`. A literal produces a literal result extent; an in-scope symbolic
dimension may preserve its name; and every other expression produces a fresh
runtime extent. A name occurring anywhere in `size` that denotes both a value
binding (a parameter, a local or a top-level binding) and an in-scope
dimension (a dimension binder of the enclosing definition, or a dimension a
tensor type in scope carries) is a type error, because the two readings can
denote different extents. A static negative size is a type error. A runtime negative
size traps `Domain` before allocation or access.

Each operation has exactly one result shape. `expand` sets the extent at
`axis` and leaves the rank unchanged; `insert` adds an axis of extent `size`
at `axis` and produces rank `rank(x) + 1`. No result is deferred, no consumer
selects between shapes, and no context supplies a default. `expand` requires
`axis` within `rank(x)` and an operand extent at `axis` of 1, which
`spec/05-risc-primitives.md` §2.4.1 states as a claim: a literal operand
extent other than 1 is a type error, and a symbolic or runtime one is checked
by a §4.7 runtime extent guard. `insert` admits `axis` in `0..=rank(x)`, so
`axis == rank(x)` appends a trailing axis. An axis outside its operation's
range is a type error.

When a declared or inferred result dimension claims a literal or named extent
that is not statically proven equal to `size`, execution checks equality and
traps `Domain` on mismatch. An unconstrained computed extent is represented as
`(d-name {} *)`. The runtime value itself is always retained. A function
parameter, top-level or local binding, user-function result, cast, or checked
integer-arithmetic expression is equally admissible; no stage may reject an
extent because of its provenance or default it to one.

#### 4.7.3 `reshape` with runtime sizes

`reshape` accepts a `List[i64]` whose arity is statically known, because
tensor rank is part of the static type. Every element may be an arbitrary
runtime i64 expression. A literal extent becomes a literal result dim. A
direct shape read may preserve an input dimension identity only when ordinary
type reasoning proves that identity; an arithmetic expression, a cross-tensor
read without a proof of equality, or any other computed value produces a
fresh `(d-name {} *)` extent. A surrounding signature may give that extent a
literal or name only by imposing an execution-time equality guard.

Every target evaluates the complete shape list in source order. A negative
extent, checked product overflow, or product different from the input element
count traps before allocation or access. No syntactic form, binding scope,
function boundary, or source-tensor identity changes acceptance.

#### 4.7.4 Runtime shape expression execution

Runtime extent expressions are lowered as ordinary typed integer dataflow.
This includes function parameters, top-level and local bindings, user-function
results, casts, and arbitrary checked integer arithmetic over literal,
symbolic, or `shape`-derived values. Eval, C, HIP, and Metal execute the same
graph and checks. A backend may optimize a proven constant but may not require
one, infer provenance to narrow the language, or substitute a guessed extent.

> **[04-SHAPE-1]** Any equality or ordering over tensor element counts or
> storage capacities that an implementation uses to select, alias, or reuse a
> buffer SHALL be sound over the exact mathematical values of the complete
> typed extent expressions. An implementation MAY conservatively decline to
> prove two capacities equal, but it SHALL NOT wrap, saturate, truncate, or
> substitute an overflow sentinel that can make unequal mathematical counts
> equal. Projection from the exact count into `i64`, `usize`, or a target
> allocation-size domain SHALL be checked and SHALL fail before planning,
> allocation, or element access when the value is outside that domain. This
> rule applies to compiler analyses as well as runtime allocation paths; a
> reuse decision is not exempt because no bytes have yet been touched.

#### 4.7.5 Precision rule for `reshape`'s shape list

`reshape`'s shape list is `List<i64>`: its elements are extent-domain
quantities under `spec/05-risc-primitives.md` [05-DIM-1]. The list must be
homogeneous, and its element precision must be `i64`. Those are two
separate requirements and either can fail alone.

No §5.6 position reaches a list literal, so a literal element states
`i64` itself, with a suffix or an explicit `cast`; the §5.3 `i32`
default never satisfies this slot. A non-literal element needs no
annotation when its producer is already `i64`, which [05-DIM-2] makes
true of `shape()`:

```text
reshape(x, [2i64, 2i64])                               ;; OK - explicit suffix
reshape(x, [shape(x, 0), 4i64])                        ;; OK - shape() is i64; the literal states it
reshape(x, [cast(shape(x, 0), i64), cast(4, i64)]) ;; OK - explicit cast
reshape(x, [2, 2])                                     ;; TYPE ERROR: i32 literals in a List<i64> slot
reshape(x, [2i64, 2i32])                               ;; TYPE ERROR: mixed element precision
```

The suffix requirement is deliberate, not a §5.6 gap; §5.6 records why
list literals do not adopt. The two error lines want different
diagnostics. The all-`i32` list is a slot mismatch, and its message
should name the fix (write `2i64`) rather than only the mismatch. The
mixed list is an intra-list disagreement that no defaulting rule can
resolve, and its message should name the disagreeing element rather than
render an `expected`/`got` pair, whose direction reflects unification
order rather than which element the user got wrong.

#### 4.7.6 Arbitrary Runtime Shape Expressions

Chelis executes arbitrary runtime shape expressions. When ordinary type
reasoning cannot prove a symbolic identity for an expression such as
`shape(x, 0) * 2` or `add(shape(x, 0), shape(y, 0))`, the result dim is
`(d-name {} *)`. A surrounding literal or named-dimension claim adds the
runtime equality guard from §4.7.2/§4.7.3; it does not make the expression
illegal. Producers may emit any well-typed expression and need not rewrite it
into a privileged syntactic shape.

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
cast(x: tensor[D, i32], f32) : tensor[D, f32]
```

Cast is always explicit. The compiler never inserts implicit casts.

**Value semantics ([04-NUM-14]).** A cast is a numeric operation under §9: its result
is finalized into the target dtype or traps ([04-NUM-1..4]), identically
on scalar and tensor surfaces. Per direction:

- any source -> float target: IEEE RNE finalize at the target width
  ([04-NUM-2]; overflow is the correctly signed infinity, never a trap).
- integer/bool source -> integer target: exact value; out of the target
  range traps `overflow` (no wrap).
- float source -> integer target: the value must be finite, integral, and
  in range. A fractional value or NaN/±inf traps `domain`; an integral
  value outside the target width traps `overflow`. The default cast never
  chooses a rounding rule; write `cast(floor(x), i32)` or
  `cast(round(x), i32)` to state one explicitly.
- any source -> bool target: strict {0, 1} membership; exactly 0/1
  encodes false/true, anything else traps `domain` ([04-NUM-4]).

Integer-to-float cast is total IEEE RNE and can lose integer exactness:
`cast(9007199254740993i64, f64) = 9007199254740992.0` and
`cast(16777217i32, f32) = 16777216.0`. This target-width rounding is the
float finalization rule, not an implicit source-language cast. Conversion
that discards a fractional part, wraps, or saturates requires an explicit
rounding operation or a distinct named conversion; it is never the default
float-to-integer behavior of `cast`.

**Named conversions ([04-NUM-16]).** `cast_trunc`, `cast_saturate`, and
`cast_wrap` are separate operations with the source/target domains and exact
value rules in `spec/05-risc-primitives.md` §3.8. There is no `cast_round`;
write `cast(round(x), target)` when that composition is intended.

### 5.3 Literal Types

Integer literals default to `i32`. Float literals default to `f32`. These
defaults can be overridden in three ways:

1. an explicit literal suffix (§5.5) attached to the literal token
2. a known element type in the surrounding position (§5.6)
3. an explicit `cast` around the literal expression

There is **no implicit precision promotion** from these defaults to any other
type. A bare `[1, 2, 3]` in an unannotated position is `tensor[3, i32]`, not
`tensor[3, i64]`. A bare `[1.0, 2.0, 3.0]` in an unannotated position is
`tensor[3, f32]`, not `tensor[3, f64]`. Programs that need a wider literal
type must say so via suffix, declared element type, or `cast`.

The default is the **user-facing contract** and is non-overridable except by
the three mechanisms above. Whichever dtype they select, the literal binds
there under [04-LIT-2].

> **[04-LIT-1]** A primitive literal's Deep value atom SHALL agree with its
> declared primitive family: integer atoms denote only integer primitives,
> float atoms denote only float primitives, boolean atoms denote only `bool`,
> and string atoms denote only `string`. Primitive type metadata never casts
> or reinterprets an atom. The sole cross-family representation is an exact
> Int atom explicitly marked `literal_source: integer` and bound directly to
> a float primitive by a suffix or surrounding literal context. It SHALL be
> finalized once at the declared float width; it SHALL NOT pass through f64
> first. Every producer SHALL emit one of these canonical forms and every
> consumer SHALL reject an unmarked contradiction or a malformed marker.
> `spec/03-deep-syntax.md` §6.4 defines the canonical Deep forms.

> **[04-LIT-2]** A numeric literal SHALL bind at its dtype by one
> finalization of its value there. Its dtype is its suffix (§5.5), the §5.3
> default, the element type of a §5.6 adopting position, each admissible
> instantiation of a dtype binder it adopts under §5.6, or, for a literal
> pattern, the scrutinee's primitive ([04-PAT-1]). An integer dtype admits
> the value exactly or rejects the literal, and a float dtype rounds it per
> [04-NUM-2]. A literal whose value at any of its dtypes is out of range or
> non-finite SHALL be rejected at every ingress before any evaluation or
> lowering lane runs. Rounding a nonzero value to zero or to a subnormal is
> ordinary rounding, not a rejection. An explicit `cast` of an already-bound
> value is an operation under [04-NUM-14], not a literal, and may produce an
> infinity.

### 5.4 Precision Compatibility Table

Operations accept same-precision operands only. The table of valid combinations:

| Operation type | Valid precisions |
|---|---|
| Arithmetic (add, mul, sub) | f32, f64, bf16, f16, i8, i16, i32, i64 (all same) |
| Float division (div) | f32, f64, bf16, f16 only (not integer; integer operands cite `spec/05-risc-primitives.md` §2.1 and point at `floor_div` / `trunc_div`) |
| Floor division (floor_div) | f32, f64, bf16, f16, i8, i16, i32, i64 (all same) |
| Truncating division (trunc_div) | i8, i16, i32, i64 only (integer-only; float operands are a type error) |
| Ordered comparison (`cmplt`, `lt`, `gt`, `gte`, `lte`) | any active numeric dtype (both operands same dtype) → bool |
| Equality (`eq`, `neq`) | any active numeric dtype or bool (both operands same dtype), plus the recursively comparable host-value domain in [05-OP-36] → bool |
| Logical (and, or, not) | bool only |
| Transcendental (exp, log, sin, cos, tan, atan, sqrt) | f32, f64, bf16, f16 only (not integer) |

Every reserved name of §1.1.1 - `f8e4m3`, `f8e5m2`, the `uint*` family,
`int4`/`uint4`, `complex64`/`complex128`, and `decimal128`/`decimal256` - is
rejected and is not a valid arithmetic precision in any row.
None of these reserved spellings has an arithmetic row.

Every tensor comparison preserves the operand surface: two same-dtype tensors
with identical dimensions return a bool tensor with those dimensions. Numeric
or bool scalar equality returns a bool scalar; ordered scalar comparison is
numeric only. The recursive host-value equality forms return one bool scalar.
Mixed tensor/scalar surfaces or tensor dimensions are type errors.
[05-OP-36] owns the exact values and the complete equality domain.

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
| `i8` | `(t-prim {} i8)` | `42i8` |
| `i16` | `(t-prim {} i16)` | `42i16` |
| `i32` | `(t-prim {} i32)` | `42i32` |
| `i64` | `(t-prim {} i64)` | `42i64` |

Float-typed suffixes (`f32`, `f64`, `bf16`, `f16`) accept a float body or a
decimal integer body, and the two bind differently. `42.0f32` binds the decoded
float at that type. `42f32` binds the exact integer directly at that width and
is [04-LIT-1]'s suffix-bound cross-family form, an exact Int atom marked
`literal_source: integer`; it SHALL NOT be finalized through `f64`. Both are
canonical Surf, and `chelis fmt` preserves the body it was given rather than
converting between them; Deep-to-Surf output follows
`spec/03-deep-syntax.md` §6.3.2 instead. A body, integer or decimal, whose
value rounds to infinity at the declared width is not a literal of that type
([04-LIT-2]).
Integer-typed suffixes (`i8`, `i16`, `i32`, `i64`) attach to integer literal
tokens only; `1.0i8` is a parse error.

Suffix lexing rule: a suffix is part of the literal token only if it
**immediately** follows the digit sequence with no intervening whitespace,
comment, or other character. `1.0 f32` (with whitespace) is two tokens (a
float followed by an identifier) and binds at the literal default per §5.3,
which is then subject to the surrounding-position rules in the type checker.

The normal Surf parser accepts value-preserving hexadecimal/binary integer
spellings, digit separators strictly between digits, and every finite decimal
float body that decodes to the literal's value, including equivalent exponent
spellings and bodies with digits past the shortest round-trippable spelling.
Integer radix forms may carry an integer suffix; they may not carry a float
suffix. These lexical choices do not change the exact suffix binding rule, and
the canonical printer emits the decoded token. Canonical decimal float literals
carry float suffixes without ambiguity (`1.0f32`, `1000.0f32`).

Rejected suffixes:

- No suffix exists for any reserved name of §1.1.1 (`f8e4m3`, `f8e5m2`, the
  `uint*` family, `int4`/`uint4`, `complex64`/`complex128`,
  `decimal128`/`decimal256`). Each is rejected at lex time with a diagnostic
  pointing at §1.1.1.
- The short unsigned spellings (`u8`, `u16`, `u32`, `u64`) are not reserved in
  any form; `uint8`/`uint16`/`uint32`/`uint64` are the canonical names per
  §1.1.2, matching numpy and Arrow.
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
   precision type literal or a dtype-bounded type binder
   ([04-DTYPE-2]) — the literal body adopts `p`

Position 4 applies to a **bare scalar numeric literal** as well as to a
tensor-literal body. `cast(1.1, f64)` binds the decimal `1.1`
at `f64` — exactly `0x3ff199999999999a` — it does NOT narrow to the §5.3
`f32` default and then widen (which would yield the f32-truncation value
`1.100000023841858`). Likewise `cast(3000000000, i64)` binds the literal
at `i64`, which is what makes the §5.3 out-of-i32-range escape hatch
work. The adoption re-binds the literal at `p`, and [04-LIT-2]'s range and
finiteness checks apply at `p`: `cast(2147483648, i32)` is still a range
error, and `cast(70000.0, f16)` is rejected because 70000 rounds to infinity
at `f16`, whereas `cast(70000.0f32, f16)` casts a finite `f32` value and
yields infinity under [04-NUM-14]. For a binder `p`, the literal binds at
each admissible instantiation ([04-INF-6]), so those checks apply at every
member of `p`'s family: `cast(300, p)` under `p: Int` and
`cast(70000.0, p)` under `p: Float` are rejected at the declaration. Adoption
is limited to unsuffixed numeric literals with a numeric `p` of matching
kind: a suffixed literal binds at its suffix (§5.5; `cast(1.1f32, f64)`
widens the f32 value), and a float literal under an integer `p` keeps the
float source type because a decimal cannot bind at an integer type; the
explicit cast then applies [04-NUM-14], accepting only a finite integral
value in range and trapping `Domain` on a fractional value.

The set is closed on purpose. Positions 1–3 adopt **tensor-literal
bodies** only: a tensor's element type is a single property of the value,
stated once in its type, and the body is bulk data — a per-element suffix
on a thousand-element weights literal is noise that buries the one
element that differs. Position 4 adopts a **bare scalar**, but its target
dtype is spelled at the site and the position exists for
**expressibility**, not convenience: without it `cast(3000000000, i64)`
cannot be written at all, and `cast(1.1, f64)` would round through the
`f32` default. What no position does is adopt a **list literal, or a
scalar against a remote callee signature** — position 2 reaches through a
signature, but only into a tensor body — and none should be added for
ergonomics alone. A structural argument — a shape list, a bounds pair, a
stride step — is program rather than payload, and under
`spec/05-risc-primitives.md` [05-DIM-1] its dtype states which KIND of
quantity it is: an extent is `i64` and an axis is `i32`, so a
context-inferred `[2, 2]` would hide exactly the distinction the dtype
exists to carry. Chelis programs are written and, more often, audited by
agents; a suffix states the kind at the site, the write-side cost is one
edit under a diagnostic that names the fix, and the read-side cost of
context-dependent literals is paid on every audit.

Outside this closed set, numeric literals in a tensor body fall back to the
§5.3 literal defaults: integer literals to `i32`, float literals to `f32`.

A tensor literal with mixed-suffix entries is well-formed only if every
suffix matches the inferred element type. `[1.0, 2.0f64, 3.0]` in an
`f32`-context is a type error: the f64-suffixed literal at index 1 has an
explicit dtype that disagrees with the surrounding `f32` element type.

A bare tensor literal `[1, 2, 3]` in an unannotated position evaluates to
`tensor[3, i32]`, not `tensor[3, i64]`. The fallback to the §5.3
default is the spec contract; no stage may silently widen it.

### 5.7 Mixed-Precision Accumulator Parameter

Exactly three reduction-shaped operations carry an **optional** accumulator-precision
parameter that controls the precision used for the inner sum:

- `matmul(A, B, accumulator=p)` — the precision of the inner-product
  accumulator before the result is downcast to the operand precision (when
  the accumulator is wider than the operands)
- `reduce_sum(x, axis=k, accumulator=p)` (also known under the spec name
  `sum`; see `spec/05-risc-primitives.md` §2.3) — the precision of the
  running sum
- `einsum(equation, left, right, accumulator=p)` — the precision of every
  contracted-label sum

The accumulator parameter is the **only** mechanism for mixed precision in
these ops. There is no implicit precision promotion: passing two `bf16`
operands to `matmul` does not implicitly widen them. The accumulator
parameter is what tells the backend to compute the inner sum at a wider
precision and (where the result is the operand precision) downcast at the
end.

The accumulator is optional only on the user-facing Surf and Deep call
surfaces. When it is omitted, the compiler resolves it to the documented
default for the operand precision before any backend is invoked. The IR
`Matmul`, `Sum`, and `Einsum` nodes always carry a populated
accumulator-precision field; "no accumulator parameter" is not an IR state.

#### 5.7.1 Default Accumulator Precision

For operands of precision `p`, the default accumulator precision is:

| Operand precision `p` | Default accumulator (matmul) | Default accumulator (sum and einsum) | `sum_result(p, default(p))` |
|---|---|---|---|
| `bf16` | `f32` | `f32` | operand precision (`bf16`) |
| `f16` | `f32` | `f32` | operand precision (`f16`) |
| `f32` | `f32` | `f32` | operand precision (`f32`) |
| `f64` | `f64` | `f64` | operand precision (`f64`) |
| `i8` | (matmul not defined for i8 — see §5.7.2) | `i32` | `i32` |
| `i16` | (matmul not defined for i16 — see §5.7.2) | `i32` | `i32` |
| `i32` | (matmul not defined for i32 — see §5.7.2) | `i32` | `i32` |
| `i64` | (matmul not defined for i64 — see §5.7.2) | `i64` | `i64` |

Rationale for the bf16/f16 → f32 default: numerical stability of long
inner-product reductions in low-precision arithmetic. PyTorch and JAX use the
same wider-accumulator default for bf16/f16 matmul.

Rationale for the i8/i16 → i32 default: overflow safety. Summing 200
non-trivial `i8` values overflows `i8` but fits comfortably in `i32`.
The same instinct exists ecosystem-wide, but the details differ: PyTorch's
`torch.sum` promotes ALL integral inputs to `int64`, and NumPy accumulates at
the platform default integer. Chelis deliberately widens one step instead of
jumping to `i64`; the §5.7 accumulator parameter is the authored route to a
wider accumulator when a reduction genuinely needs one.

An accumulator has the same numeric kind as its operands: a signed-integer
operand selects a signed-integer accumulator, and a float operand selects a
float accumulator. Cross-kind accumulation is a type error. Within that kind,
the accumulator parameter is permitted only when it is at least as wide as
the operand precision and is not narrower than the documented default. A
program that explicitly requests a narrower accumulator (e.g.
`reduce_sum(x: tensor[N, i8], accumulator=i8)`) is a type error with a
diagnostic suggesting either omitting the parameter (which yields the i32
default) or accepting the wider default explicitly.

The permitted accumulator and result pairs are total and exact:

| Operand dtype `p` | Permitted accumulator dtype `a` | `sum_result(p, a)` |
|---|---|---|
| `bf16` | `f32`, `f64` | `bf16` |
| `f16` | `f32`, `f64` | `f16` |
| `f32` | `f32`, `f64` | accumulator dtype `a` |
| `f64` | `f64` | `f64` |
| `i8` | `i32`, `i64` | accumulator dtype `a` |
| `i16` | `i32`, `i64` | accumulator dtype `a` |
| `i32` | `i32`, `i64` | accumulator dtype `a` |
| `i64` | `i64` | `i64` |

Equivalently, `sum_result(p, a) = p` exactly when `p` is `bf16` or `f16`;
otherwise `sum_result(p, a) = a`. A pair absent from this table is a type
error. Thus an explicitly wider accumulator controls the public result dtype
for signed integers and for `f32`, while the reduced-precision float rows
always restore their operand storage dtype after accumulating.

At the IR level, the `Sum` node's output precision is always the
accumulator precision; lowering inserts an explicit `Cast` for the
`bf16`/`f16` rows to recover the operand-precision result documented in
the table. No other row inserts that result cast.

The result precision of `matmul` matches the operand precision (the wider
accumulator is consumed inside the op and downcast on output). The result
precision of `einsum` is `sum_result(p, a)`, exactly like `sum`, where `a` is
the explicit accumulator or `default(p)` when the argument is omitted.

#### 5.7.2 Integer matmul

The matmul signature rejects integer operand precisions (`i8`, `i16`,
`i32`, `i64`). Integer `sum` is admitted per §5.7.1.

### 5.8 Stdlib Generalization Shape

Every public tensor-op signature exported from `packages/chelis-std/` is
generalized over a precision type variable. Type-check enforces that each
instantiation works against the active dtype set (§1.1) and the precision
compatibility table (§5.4).

A typical generalized signature has the shape:

```scheme
;; Std.Tensor.add : forall p. tensor[D, p] -> tensor[D, p] -> tensor[D, p]
(defsig {} add (d p)
  (t-fn {}
    (t-tensor {} (d-var {} d) (t-var {} p))
    (t-tensor {} (d-var {} d) (t-var {} p))
    (t-tensor {} (d-var {} d) (t-var {} p))))
```

Float-only operations (those whose §5.4 row reads "f32, f64, bf16, f16 only")
carry an explicit kind restriction limiting `p` to the float subset of the
active dtype set. Examples include `exp`, `log`, `sin`, `sqrt`, the
transcendental row in §5.4, and `div` (float division — its integer-operand
diagnostic points at `floor_div` / `trunc_div`). Calling a float-only op on an
integer tensor is a type error reported at the call site. `trunc_div` is the mirror case (integer-only): applying it to a
float tensor is a type error.

Backends may specialize each instantiation, but the **public signature** is
generalized and specialization cannot change its admitted dtypes or results.

The active dtype set is the source of truth for stdlib generalization
coverage: every public tensor op must be usable at every dtype in §1.1 that
its §5.4 row admits. A stdlib op that fails for a §5.4-admissible dtype is a
spec compliance bug, not a documentation bug.

#### 5.8.1 Explicit Signature Binders

The Surf surface admits precision polymorphism in user-written signatures
through the declaration's explicit `[..]` binder list:

> In a sig with quantified type variables, names appearing in the
> precision slot of a `tensor[...]` type that match the sig's quantifier
> list become `(t-var {} <name>)`, not `(t-prim {} <name>)`. Names
> matching a primitive (`f32`, `f64`, `bf16`, `f16`, `i8`, `i16`,
> `i32`, `i64`, `bool`) stay as `(t-prim {} <name>)`, so canonical
> Surf and Deep carry one spelling per primitive. The retired spellings
> `int8`, `int16`, `int32`, and `int64` are rejected and offered only to
> the explicit v0.18 migrations; they never become `t-var`. Outside a sig
> (e.g., in a value-position type annotation), no quantifier exists,
> so the existing rule applies.

An enclosing explicit declaration binder has one scope throughout that
declaration's body. Every type-position occurrence of its name resolves to the
same binder, including tensor precision slots in lambda parameters, expression
ascriptions, block bindings, nested ADT arguments, property quantifier types,
preconditions, predicate bodies, and expression-valued options. With no
enclosing explicit binder for the name, a body-local occurrence remains a
value-position request and follows the closed primitive or undeclared-name
rule below; it does not introduce a binder.

No occurrence introduces a binder. Every `t-var`, `d-var`, and `d-rank`
name in a signature appears in its explicit list. An unlisted lowercase
name in a scalar or precision type position remains a primitive-name
request and is rejected as unknown, with a nearest-active-dtype
suggestion; an unlisted dimension or rank variable is rejected as
undeclared. This is a structural rule, not a dtype-shaped-name heuristic.
The §1.1.2 unsigned aliases (`u8`, `u16`, `u32`, `u64`, `uint8`,
`uint16`, `uint32`, `uint64`) and every other spelling [04-DTYPE-1]
rejects name no type variable in any type position: a binder list does
not rebind one, and the same rejection applies on Surf and Deep.
An unbounded name may be listed without occurring in the signature; it is a
vacuous universal quantifier and canonical round-tripping preserves it. A
dtype-bounded binder must occur in the declared type (§5.9), so bounds
cannot be used as inert metadata.

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

Every polymorphic call supplies a concrete precision before the backend
boundary. The restrictions in §5.4 and §5.7.2 apply both to direct primitive
calls and to generic definitions under [04-INF-9]. A polymorphic call's
instantiation satisfies the checked contract, including its dtype
bounds under [04-DTYPE-2]. An inadmissible instantiation is a
`PrecisionMismatch` at the call site with a citation to the governing section.

### 5.9 Dtype Bounds

A declaration's type-binder list may constrain a binder to a **dtype bound**.
A bound is written in one of two forms: a **family name**, or an **explicit
dtype set**.

A **dtype family** is a named subset of the active primitive set of §1.1.
There are three families.

| family | members |
|---|---|
| `Float` | every active float dtype of §1.1 |
| `Int` | every active signed integer dtype of §1.1 |
| `Numeric` | the union of `Float` and `Int` |

Membership follows §1.1's active set rather than an enumeration repeated here,
so a dtype §1.1 admits into a family is admitted by every bound naming that
family. The reserved spellings of §1.1.1 belong to no family. `bool` and
`string` belong to no family.

An **explicit dtype set** is written `{d1, d2, ...}` and admits exactly the
dtypes it lists, in any order.

```
def widen[p: {f32, f64}](x: p) -> p = ...
def approx[p: {f32, f64, bf16}](x: p) -> p = ...
def scale[p: Float](x: p) -> p = ...
```

The two forms differ in how membership is decided, and that difference is
normative. A family denotes whatever §1.1 admits into it, so activating a
dtype widens every family that admits it without amending this section. An
explicit set denotes its listed members and nothing else, so activating a
dtype never widens one. A declaration that must exclude a dtype states the
set it admits; a declaration that tracks a family names the family. Neither
form is an abbreviation of the other, and a set that happens to enumerate a
family's current members is not that family.

> **[04-DTYPE-2]** A type binder that declares a dtype bound SHALL occupy type
> positions only and SHALL be instantiated only at an active primitive of §1.1
> the bound admits. The bound is part of the
> declaration's scheme rather than a property of one call: instantiation
> installs it on each fresh variable, unification propagates it through every
> variable the bounded variable is identified with, and generalization
> re-quantifies it, so the bound survives aliases, wrappers, higher-order
> values, imports, and recursive calls. Unifying two bounded variables SHALL
> yield the intersection of their families. Where either bound is an explicit
> set, that intersection SHALL instead be the dtypes both bounds admit and is
> itself an explicit set: a family and a set intersect as the set's members the
> family admits, and two sets intersect as their common members.
> An instantiation outside the bound
> SHALL be a `PrecisionMismatch` naming the required bound and the offending
> type; an empty intersection SHALL be a `PrecisionMismatch` naming both
> families. Where either bound is an explicit set, that diagnostic SHALL name
> both bounds. A binder that declares no bound
> remains an unconstrained type variable admitting every type, not only a
> dtype. A bound that is neither a family of this section nor an explicit set,
> an explicit set that is empty, repeats a dtype, or names anything but an
> active primitive of §1.1, a bounded
> binder used in a dimension slot or as a rank spread, and a bounded binder
> that does not occur in the type it is declared for are each declaration
> errors.

A declaration's bounds live in exactly one binder list. When a standalone
signature declares a name, that signature's binder list carries the bounds for
that declaration, and a bound written in the same declaration's `def` binder
list is a declaration error. A `def` with no standalone signature carries its
bounds in its own binder list.

[04-INF-9] requires a generic body's necessary dtype restriction to
follow from this declared contract. Applying a float-only operation does not
infer a `Float` bound on an otherwise unbounded or more broadly bounded
authored binder, nor publish such a bound from an unannotated abstraction.

A public stdlib signature whose `[05-OP-35]` registry domain is exactly one of
these families declares that family as a bound. That registry writes its domains
as the signature-table metavariables `p_float`, `p_int`, and `p_numeric`, and
each denotes the family of the same name; a signature whose domain is some other
set, such as every active tensor element dtype, declares no bound.

The surface spelling of a binder list is `spec/02-surf-syntax.md` §P4b and of
a bound is its §P4c; the Deep encoding is `spec/03-deep-syntax.md` §1.1.

---

## 6. Fitness Scoring

The compiler produces a fitness report for every compilation attempt. This is the feedback an agent acts on.

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

The agent can use the fitness counters and diagnostics to locate failed checks.
The compiler's checked-program product carries annotated Deep after successful
type checking; the fitness report is a diagnostic document.

> **[04-FIT-1]** `typed_nodes` and `total_nodes` SHALL report the type
> inference product's checked-node counters, not a fabricated structural AST
> count. A cached or layered check SHALL preserve those counters per checked
> unit and add them across the program partition, so its fitness report is
> byte-identical to an equivalent monolithic check.
> The `structure` component remains a separate structural-AST measurement.

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

`chelis check` emits one JSON document per invocation: the report below
for a file target, or the directory envelope (§ Directory mode) for a
directory target. It is a
machine-facing contract rather than an illustration: reward surfaces,
conformance corpora, and downstream tooling consume it, so a change to its
shape is a change to a published interface.

> **[04-FIT-11]** The report SHALL be produced by serializing one typed
> value. A hand-assembled document -- string concatenation, format
> templating, or any second producer of the same shape -- is not a
> conforming implementation, because it admits drift between the emitted
> document and the type that describes it.

> **[04-FIT-12]** Every `chelis check` failure SHALL be reported through
> that same typed value, including Surf and Deep parse failures and any
> preparation failure occurring before type checking begins. A failure
> path that bypasses the report and emits a display string is not
> conforming: a consumer cannot distinguish "no diagnostics" from
> "the diagnostics were not transported".

#### Document fields

| field | type | presence |
|---|---|---|
| `score` | number | always |
| `components` | object of `parse`, `structure`, `names`, `types` | always |
| `typed_nodes`, `untyped_nodes`, `total_nodes` | integer | always |
| `unresolved_names` | array of string | always, possibly empty |
| `errors` | array of diagnostic | always, possibly empty |
| `inferred_signatures` | structured signature tree | only when the caller requests inferred signatures |

> **[04-FIT-18]** `score`, `components.parse`, `components.structure`,
> `components.names`, `components.types`, and diagnostic `severity` SHALL be
> finite `f64` values in `[0, 1]`. Signed zero is admitted and preserved.
> `severity` orders the producer's assessment from least to most severe;
> it neither changes diagnostic presence nor replaces the error-list exit rule.
> Score components and the weighted total retain §6.1's measurements and
> weights. Checked-node counters SHALL be exact nonnegative `i64`, with
> `typed_nodes <= total_nodes` and
> `untyped_nodes = total_nodes - typed_nodes`. Their provenance remains
> [04-FIT-1]. Counter overflow or an inconsistent report is an error, never
> a wrapped counter or a repaired total. The fixed-dtype number codecs in
> spec/10 §3.5 preserve the JSON number/integer field shapes above through
> canonical sealed numeric carriers. NaN, infinity, an out-of-domain value,
> or a missing required field is rejected, never rounded, clamped or defaulted.

(The fixed-dtype report carrier requirement is not fully implemented; see chelis#1288.)

> **[04-FIT-13]** When requested, `inferred_signatures` SHALL be carried
> in the same typed value as the rest of the report, absent by omission when
> not requested. The report SHALL NOT carry a `typed_ast` member. Annotated
> Deep belongs to the compiler's checked-program product, which represents
> successful type checking; the fitness report carries the diagnostics and
> measurements of a compilation attempt. A consumer SHALL NOT infer the
> existence of a checked-program product from the presence of a report.

#### Diagnostic fields

Each element of `errors` carries:

| field | type | presence |
|---|---|---|
| `kind` | closed vocabulary member | always |
| `message` | string | always |
| `severity` | number | always |
| `expected`, `got` | string | when the producing check determined them |
| `suggestions` | array of string | when non-empty |
| `span` | coordinate or range | when the producing node carried one |
| `span_id` | producer-supplied identity | when the producing node carried one |
| `deep_path` | Deep address | when the producing check determined it |

> **[04-FIT-14]** `kind` SHALL be a member of a closed, validated
> diagnostic-kind vocabulary. A debug rendering of a producer-internal
> enumeration is not a conforming source for this field: it makes an
> internal variant rename an unannounced change to a published interface.
> Every diagnostic-producing stage SHALL map into that one vocabulary, and
> a consumer SHALL be able to reject a member outside it.

> **[04-FIT-15]** `expected`, `got`, `suggestions`, `severity`, and
> `deep_path` SHALL be retained uniformly for checker, effect, and
> linearity diagnostics alike. A field the producing stage populated SHALL
> reach the document, and which fields the document carries SHALL NOT vary
> by which stage produced the diagnostic.

#### Location

> **[04-FIT-16]** The document SHALL carry the source location as the
> producing node held it: a local coordinate or range into the checked
> source, and the producer-supplied identity when the node carried one. The
> two are independently optional, and the document SHALL be able to carry
> either without the other. An opaque or source-qualified identity SHALL be
> transported rather than discarded, and SHALL NOT be reconstructed from a
> coordinate: an identity minted by an external producer is not recoverable
> from an offset and a length.
>
> Whether the coordinate and the identity occupy one member or two is a
> shape choice this atom does not decide; independent optionality is the
> requirement.


> **[04-FIT-17]** The serializer SHALL NOT fabricate an extent it did not
> measure. Where a producer supplied only a point, or only an opaque
> identity, the document SHALL carry neither an invented length nor a
> zero-width range: the extent SHALL be ABSENT. Where a producer supplied
> no location at all, `span` SHALL be absent. A zero length and a
> silently-defaulted length are both indistinguishable from a measured
> extent, so a consumer that reasons about ranges cannot tell whether the
> compiler measured one.

> **[04-FIT-27]** For a checker rejection of `to_list`,
> `tensor_to_scalar`, `copy`, `cast`, `split_keys` (count), `expand`
> (axis or size), `shrink` (arity, tensor, or bounds), `reshape`
> (shape type, tensor input, or element count), a reduction's tensor input
> or selected axis, or a function
> application's argument count or tensor-dimension match, the diagnostic SHALL
> retain the available location of the rejecting call or offending operand,
> not substitute a containing declaration or unrelated argument. Its message
> SHALL name the callee and rejected argument position; a per-axis mismatch
> SHALL name the axis as well.
> When that check knows a directional requirement and the value that failed
> it, the diagnostic SHALL carry both in `expected` and `got`, and the human
> message SHALL use those same values. The requirement is the operation's
> admitted input, the callee parameter, or a declared type; `got` is the
> actual operand or inferred body. Symmetric unification alone establishes
> neither direction and SHALL NOT assign one: a caller that cannot establish
> the direction leaves the pair absent instead of reversing or inventing it.
> This diagnostic rule does not decide whether an unresolved type variable
> should be rejected; the check's existing admission rule still decides that.


#### Example

Illustrative of the shape only; the atoms above are normative.

```json
{
  "score": 0.73,
  "components": {"parse": 1.0, "structure": 1.0, "names": 0.85, "types": 0.62},
  "typed_nodes": 27,
  "untyped_nodes": 4,
  "total_nodes": 31,
  "unresolved_names": ["typo_var"],
  "errors": [
    {
      "kind": "PrecisionMismatch",
      "message": "precision mismatch: expected f32, got bf16",
      "severity": 0.8,
      "expected": "tensor[batch, hidden, f32]",
      "got": "tensor[batch, hidden, bf16]",
      "suggestions": ["Insert explicit cast"],
      "span": {"span": "range", "offset": 786, "len": 20},
      "span_id": "surf:786..806"
    }
  ]
}
```

> **[04-FIT-2]** Every `UnboundVariable` and `UnknownConstructor`
> diagnostic SHALL contribute to the `names` component and SHALL add the
> offending identifier to `unresolved_names`. The list contains one entry per
> such diagnostic in checker diagnostic order; repeated diagnostics are not
> deduplicated. A report containing either kind SHALL have `names < 1`, a
> non-empty `errors` list, and `score < 1`. The fitness report adds no separate
> name-resolution wire field: these invariants govern the existing
> `components.names`, `errors`, and `unresolved_names` fields.

#### Directory mode

`chelis check <dir>` checks every source file under a directory and emits
one document for the whole invocation, the directory envelope. The envelope
carries the per-file reports defined above; it is not a second report
format.

| field | type | presence |
|---|---|---|
| `files` | array of entry | always, possibly empty |
| `errors` | array of diagnostic (§ Diagnostic fields) | always, possibly empty |

Each entry carries:

| field | type | presence |
|---|---|---|
| `file` | string | always |
| `report` | the report above | always |

> **[04-FIT-19]** The directory envelope SHALL be produced by serializing
> one typed value that holds every per-file report as a member, under
> [04-FIT-11]. An envelope assembled from separately rendered report
> documents is not conforming. `files` and `errors` are both present in
> every envelope, including an empty one.

> **[04-FIT-20]** The corpus of a directory target is every checkable file
> the walk reaches from it. The walk follows symbolic links wherever they
> resolve, including to directories and to locations outside the target.
> Below the target, it excludes every entry whose name begins with `.` and
> every directory named `target`; the target itself is never excluded. The
> dot-prefix test applies to the entry's own name before anything about the
> entry is resolved, so a dot-prefixed link contributes nothing, whatever it
> resolves to or fails to resolve to. The `target` exclusion is a rule about
> directories, so it applies once the entry resolves to one. A checkable
> file is a regular file, reached directly or through links, whose name ends
> in `.ch` or `.dp`.
>
> The walk visits each file and each directory at most once, identified by
> its canonical path, and the first path in walk order ([04-FIT-22]) names
> it. A file reached by several paths is therefore one entry, and a link
> cycle or a link back to a visited directory adds nothing.
>
> An entry the walk cannot resolve is a checkable file when its own name is
> one: it is an entry, and its report carries the failure that naming it
> directly would. Otherwise it contributes nothing when its referent does
> not exist, and is a walk failure under [04-FIT-23] when it cannot be
> resolved for any other reason.

> **[04-FIT-21]** Each entry's `file` is the file's path relative to the
> target, its components joined by `/` on every host, and its `report` is
> that file's report. Every entry SHALL carry a report: a file that cannot be
> read, formatted, parsed, or checked is an entry whose report carries the
> failure under [04-FIT-12], never an entry carrying a message in place of a
> report. A relative path that is not valid UTF-8 cannot be written as
> `file`, and no replacement character or other substitution may stand in
> for it: such a file is not an entry, and the walk reports it under
> [04-FIT-23].

> **[04-FIT-22]** `files` is in walk order: a depth-first pre-order
> traversal that visits the entries of each directory in ascending order of
> the byte sequence of the name the host reports, the order [05-HOST-4]
> fixes for `list_dir`. That order is not a sort of the relative paths: a
> directory `a` and everything under it precede a sibling file `a.ch`,
> although the path `a.ch` sorts before `a/b.ch`.

> **[04-FIT-23]** A failure that belongs to no single file SHALL be reported
> as a diagnostic in the envelope's `errors`. In directory mode, [04-FIT-12]
> is met by an entry's report for a failure of that file, and by the
> envelope's `errors` for a failure of the walk; neither is replaced by an
> empty or truncated document or by a display string. A directory the walk
> cannot read, whether the target or a directory below it, reached directly
> or through links, contributes one `directory_walk_error` diagnostic naming
> that directory. So does each entry [04-FIT-20] makes a walk failure, and
> each path [04-FIT-21] cannot represent. The walk continues past every one
> of them: each checkable file it can still reach is an entry. An unreadable
> target therefore yields an envelope with no entries and one diagnostic.
> `errors` is in the walk order of [04-FIT-22], as `files` is, so a second
> run over an unchanged tree emits the same document. A path inside one of
> these messages is rendered as [05-HOST-4] renders one: reversible escaped
> host bytes. A message that names a path SHALL NOT substitute for bytes it
> cannot represent, or two paths the walk rejected become one message a
> reader cannot tell apart. The reports of the readable files are additional
> information, not a partial success; the envelope still fails under
> [04-FIT-25].

> **[04-FIT-24]** An empty corpus is a failure. When the walk completes
> without a [04-FIT-23] diagnostic and the corpus is empty, `errors` SHALL
> carry one `empty_corpus` diagnostic, whose message names the target and
> reports the checkable files that [04-FIT-20]'s exclusions removed.
> Emptiness is judged once, over the whole walk: an empty directory below a
> non-empty corpus contributes nothing. When a [04-FIT-23] diagnostic is
> present, that diagnostic accounts for the result and no `empty_corpus`
> diagnostic is added, because the walk has not established that the corpus
> is empty.

> **[04-FIT-25]** Directory mode SHALL exit `0` if and only if the
> envelope's `errors` and every entry's report `errors` are all empty, and
> otherwise `2`. Directory mode has no other exit status.

### 6.5 Source Identity In Diagnostics

A diagnostic names the entities the user wrote. The checker's internal
identities for those entities -- `DimVar` and `TypeVar` ordinals, rendered
`dN` and `?N` -- are inference bookkeeping, and a reader has no way to map
one back to their source.

> **[04-FIT-9]** Where source provenance exists for a binding, type variable,
> or dimension variable, a user-facing diagnostic SHALL identify it by its
> source spelling. A compiler-generated inference identity SHALL NOT be its
> sole user-facing identity.

Provenance survives resolution, generalization, and instantiation. A checker
that reports on an instantiated signature therefore recovers the declared
spelling rather than treating instantiation as the point where the name is
lost; the fresh variables an instantiation mints are the same entities the
source declared.

> **[04-FIT-10]** Where provenance genuinely does not exist, a diagnostic
> SHALL render the inference identity as synthesized, distinguishably from a
> spelling the user wrote, and SHALL NOT invent a source name for it.

Absent provenance is the only admissible route to [04-FIT-10]. A producer
that holds a spelling and declines to thread it through is not covered by it.

(The borrow and cast diagnostics do not yet satisfy [04-FIT-9]; chelis#260
owns that gap.)

### 6.6 Textual Rendering Of Check Diagnostics

A command that stops on a check rejection and reports it as text rather
than as the §6.4 document -- `chelis build`, for example -- still reports
the same diagnostics.

> **[04-FIT-26]** A textual rendering of a check rejection -- type, effect,
> or linearity diagnostics alike, the stages [04-FIT-15] names -- SHALL render
> each diagnostic from the projection §6.4's `errors` elements carry: its
> `kind` vocabulary member, its `message`, and its location as
> [04-FIT-16] and [04-FIT-17] admit it. Each diagnostic SHALL occupy its
> own line, in the order the checker reported it, so a rejection carrying
> `N` diagnostics renders `N` lines. A debug rendering of a
> producer-internal value is not a conforming rendering: it publishes
> field names, absent-value markers, and variant spellings that
> [04-FIT-14] keeps off the published interface. A diagnostic that cannot
> be projected is a failure reported as such, never a debug rendering in
> its place.

(The effect and linearity renderings and `chelis reef build` do not yet
satisfy [04-FIT-26]; chelis#2130 owns that gap.)

---

## 7. Effects

### 7.1 Effect Model

Built-in effect vocabulary in the type layer:

- `Accum` -- internal-only hook for associative gradient accumulation
- `IO` -- host-side effects such as `print` and `debug`,
  the file builtins (`read_file`, `write_file`, ...), subprocess exec via
  `process_run`, and the clock reads `clock_wall_read` and
  `clock_monotonic_read`; `IO` is the single effect for host-side observable
  interaction
- `Test` -- assertions whose failure is observed by the Chelis test runner
- `Resource(Device)` -- allocation / placement region on a concrete device

`Diff` is a compiler capability marker, not a user-handled boundary effect.
`Accum` is internal-only and users do not handle it directly.
`Resource(Device)` is the one user-handler boundary. `IO` may remain
unhandled at the program boundary. `Test` is consumed by `chelis test`; other
execution boundaries reject an unhandled `Test` effect.

Randomness is not an effect. A random primitive ([05-OP-8], [05-OP-37]) is a
pure function of the explicit key it is given ([05-RNG-1]), and a function
that draws takes a `key` parameter and needs no effect annotation.

> **[04-EFF-1]** A `handle-effect` form SHALL name the one user handler
> boundary, `resource`. Any other handler kind is a type error; lowering
> SHALL NOT erase its handler or execute the body as if no handler were
> present.

> **[04-EFF-2]** Before emitting a host-C artifact, the build boundary SHALL
> reject every reachable `resource` handler whose device designator is not
> exactly `cpu`. The rejection SHALL be a `BuildTargetMismatch` diagnostic and
> SHALL occur before any artifact or emission observation is produced. This
> rule applies independently to every nested resource region. It defines the
> host-C admission boundary only; it does not define device-label vocabulary,
> placement, transfer, or accelerator-target semantics.

Inference and checking obey these rules:

- effect inference runs after HM type inference on the annotated Deep returned by the
  type checker
- a function's inferred effect set is the union of the effects of compiler-known
  operations in its body
- random primitives and the key operations [05-OP-69] through [05-OP-72]
  contribute no effect; [04-LIN-9] makes each key single-use
- `print(x)` and `debug(x)` are `IO` sources, alongside
  the file builtins (`read_file`, `write_file`, `read_lines`, `read_bytes`,
  `file_exists`, `list_dir`, `mmap_file`), `process_run` (subprocess exec),
  and the clock reads `clock_wall_read` and `clock_monotonic_read` ([05-OP-75]).
  Compiled host execution preserves these effects and their order under
  spec/05-risc-primitives.md [05-HOST-1..2]
- `with device(device) { ... }` marks a resource region that is validated against the
  chosen build target; [04-EFF-2] defines the host-C admission boundary
- declared `Resource("...")` annotations on `t-fn` expressions constrain the
  inferred resource set, and checked `fn` metadata records every unhandled
  `Resource(Device)` effect
- top-level `IO` is permitted
- assertion operations introduce `Test`; only the test runner handles it

### 7.2 Native Test Selection

`chelis test` discovers ordinary tests as nullary definitions named
`test_*` whose result type is `unit`. A path may name one `.ch` file or a
directory tree; a substring filter narrows the discovered definitions by
their `<file>::<name>` identity.

> **[04-TEST-1]** A completed ordinary `chelis test` invocation SHALL run at
> least one selected test. Zero selected tests is a runner error in each of
> these cases: the completed directory walk found no `.ch` files; `.ch` files
> were found but none declared a runnable `test_*`; or `--filter` excluded
> every runnable test. The condition is judged once over the complete target,
> so an empty subdirectory inside a target with a selected test contributes
> nothing. The runner SHALL exit `2` for this error and SHALL NOT emit a
> passing `0 passed, 0 failed` summary.

> **[04-TEST-2]** The zero-selection diagnostic SHALL name the target and
> distinguish the three cases in [04-TEST-1]. When no `.ch` file was found,
> it SHALL also report the `.ch` files excluded under dot-prefixed entries or
> directories named `target`, using an exact count when that excluded walk
> completes and a lower bound otherwise. Text mode SHALL render the diagnostic
> as an error. JSON mode SHALL emit one record with a top-level `errors` array
> containing exactly one diagnostic with kind `empty_test_selection`, its
> message, and severity `1.0`; it SHALL emit no test row and no summary record.

> **[04-TEST-3]** Only completed discovery establishes zero selection. A
> failure while walking the target SHALL retain its walk-error result and SHALL
> NOT also emit `empty_test_selection`. Likewise, a file that cannot be read or
> parsed continues through the ordinary file-failure path rather than being
> counted as evidence that the selected set is empty.

`--expect neg|blocked` is a file-probe mode rather than an ordinary test
selection. Each discovered `.ch` file remains one probe even when it declares
no `test_*`, because its compile or check diagnostic may be the expected
outcome. Its existing non-empty-suite requirement is therefore based on
discovered probe files, not on [04-TEST-1]'s runnable-test count.

### 7.3 Type Representation

Declared function types with effects use `eff` metadata on `t-fn`:

```scheme
;; f : tensor[D, f32] -> tensor[D, f32] ! {IO, Resource("gpu:0")}
(t-fn {eff: (effects {} io (resource {} "gpu:0"))}
  (t-tensor {} (d-var {} d) (t-prim {} f32))
  (t-tensor {} (d-var {} d) (t-prim {} f32)))
```

Checked function bodies may also carry inferred effect metadata:

```scheme
(fn {type: (t-fn {} ...), effects: (effects {} io)} (params {} x) body)
```

The `effects` metadata records the complete inferred, unhandled effect set.
Resource regions are also enforced at the handler and build boundary. Effect
annotations are optional in Surf and Deep; when present, the inferred set must
fit the declared upper bound.

Build APIs enforce Resource compatibility before exposing an emitted artifact.
An entry-scoped build includes the entry's reachable source helpers, including
imported helpers; unused library definitions do not constrain that entry's
target. A whole-program build checks the whole emitted program. Target checking
does not change the authored input or output ABI.

### 7.4 Metadata Contract

The `eff` metadata key on `t-fn` nodes carries a declared effect upper bound;
the `effects` key on checked `fn` nodes carries the inferred unhandled set.

---

## 8. Linearity

### 8.1 Model

Chelis uses lightweight uniqueness, not a Rust-style ownership-and-lifetimes
system. Tensor values are owned by default, but read-only calls borrow their tensor
arguments. A consuming use makes the binding dead; a borrow leaves the owned binding
live. For an unconsumed local owner, the compiler inserts `Drop` at the earliest
post-dominating point after its last use, as [04-LIN-8] requires. Lexical scope
exit is the fallback only when no earlier valid terminal point can be proved.

Random keys are the exception to copying. A key-carrying value (§8.4.1) is
affine: it is used at most once, never copied or borrowed, and may be dropped
unused ([04-LIN-9]). A second key comes from deriving one, never from a copy.

### 8.2 Type Representation

```scheme
;; Linear tensor (default)
(t-tensor {lin: once} (d-name {} batch) (t-prim {} f32))

;; Borrowed tensor (read-only reference)
(t-ref {} (t-tensor {} (d-name {} batch) (t-prim {} f32)))
```

The user surface is type- and expression-based:

- `&T` is the read-only borrow type, represented in Deep as `t-ref`
- `&x` is an optional explicit borrow expression, represented as `(borrow {} x)`
- passing owned `T` where `&T` is expected auto-borrows; this rule also applies to pipe stages
- passing `&T` where owned `T` is expected is a type error unless the program writes `copy(x)`
- `copy(x)` accepts either owned `T` or borrowed `&T` and yields a fresh owned value;
  explicit and compiler-inserted copies lower to `RiscOp::Copy`
- borrows cannot be stored in aggregates, returned, or captured by closures
- source borrow syntax is erased before backend lowering, but the resolved ownership
  disposition of every use remains explicit in ownership-lowered IR; implicit
  linearity inserts explicit clone/move/borrow dispositions and terminal `Drop`
  operations before any backend sees the program

#### Borrow target classification (the `&x` inner type)

The inner of a `&x` borrow expression must be — or must ultimately resolve
to — a tensor or a tensor-carrying value (a tensor-carrying ADT per §8.4 or
a tuple containing one). Borrowing a concretely non-tensor value (a scalar
`t-prim`, `()`, a function, a non-tensor-carrying ADT, or a tuple of
scalars) is a type error.

The borrow inner's type is not always known at the borrow site. When the
inner's outer type constructor is still unknown — a parameter whose
annotation is an inference hole, a bare authored binder, or a `let` binding
or lambda parameter that takes its type from one of those — the inner is
still an unresolved type variable when the borrow is first checked. In that
case classification is **deferred**: the borrow is provisionally accepted
and the surrounding flow's expected argument type pins the variable through
unification. A tensor whose dimension variables are unresolved is not such
a case: its outer constructor is already a tensor, and it is classified
immediately.

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

Linearity composes with effects, especially `Resource(Device)`: the
effect system tracks where a tensor lives, while linearity tracks when it is consumed.
That gives the compiler a stronger basis for safe in-place buffer reuse.

### 8.3 Static Rules

- A local owned linear binding must have exactly one terminal path in lowered IR:
  either a consuming use or an inserted `Drop`. Borrow sites do not count as consumes.
- Function parameters are ownership-transfer boundaries: an owned parameter becomes
  the callee's owner and must be moved into a result or another consuming destination,
  or receive a `Drop`, on every return path.
- A local value that was borrowed but never consumed receives an inserted `Drop` at
  the earliest point after its last use that post-dominates that use on the applicable
  control-flow path. This is not a user-facing type error.
- `copy(x)` reads `x` without consuming it and yields a fresh owned value.
- `drop(x)` is an explicit consume. The compiler also inserts implicit last-use
  drops for locals that are not otherwise consumed.
- Pattern matching on a tuple or other value carrying tensor payloads consumes the
  scrutinee; any tensor payloads bound by the pattern become the new live bindings.
- Creating a closure whose body consumes a captured tensor consumes that outer binding
  at closure creation time; a capture whose body uses are all borrow-reads borrows the
  outer binding instead. Which binding a consuming capture lands on is [04-LIN-2]'s
  subject below.
- A top-level function declaration ([04-INF-7]) is not a closure creation, and checking
  it never changes top-level ownership state. Because a declaration may be called after
  every top-level initializer, including from another module, its body is checked
  against the ownership state that holds once every top-level initializer has run: a
  free reference to a top-level value that no initializer consumes does not consume it,
  and one to a value that an initializer consumes is rejected whatever the textual
  order. A consuming use of such a reference yields each call's owned result through a
  copy, per [04-LIN-4], so a declaration cannot use a key-carrying top-level value,
  whose copy [04-LIN-9] refuses.
- Ordinary consuming fan-out is handled by inserted copies, except on a
  key-carrying value, which [04-LIN-9] makes affine. Diagnostics remain for
  invalid borrows, borrow escapes, impossible branch/loop ownership, and recursive or
  cyclic consume cases for which a unique terminal path cannot be proven.
- **Destructured components are excepted from copy insertion.** A binding introduced by
  a destructuring `let` — any `let` whose pattern is not a single name — is a
  *component*. A component projects a fresh owned value out of the destructured value
  rather than aliasing it, so consuming fan-out on a component is not copyable and is a
  hard error. Write `copy(x)` on the earlier consuming use to fan out a component.
  The exception is scoped to the component itself: it does not extend to the value that
  was destructured, to bindings that merely follow the destructuring `let`, or to
  a component's name after an ordinary `let` re-binds it.
- A binding introduced inside a new declaration region is an ordinary binding, whatever
  the name denoted outside. Closure captures, `match` arm binders, and `if` / `match`
  branch bodies each open such a region, so a component's fan-out inside one is copied
  like any other value. A destructuring `let` written inside the region introduces
  components of that region and is excepted there as above.

Two requirements pin the binding-identity semantics the rules above rest on:

> **[04-LIN-1]** Every binding introduction — a `let` bind (destructuring
> or not), a function parameter, a closure capture, a `match` binder, a
> top-level `def` — creates a binding distinct from every other binding,
> including earlier and later bindings of the same name. A name at a use
> site denotes the innermost such binding whose scope encloses the site.
> Every ownership fact — an alias relationship, a consumption, a borrow,
> destructured-component identity — SHALL attach to the binding it was
> resolved against where it was recorded, never to the name. In
> particular, an alias denotes the binding its source name denoted where
> the alias was introduced: a later re-binding of that name SHALL
> neither re-point the alias nor confer the new binding's properties
> (such as a component restriction) on it, and consuming through the
> alias SHALL affect the aliased binding, not whichever binding owns the
> name at the consuming use.

> **[04-LIN-2]** Creating a closure whose body consumes a captured value
> consumes the binding the capture names at closure creation time; a
> capture whose uses in the body are all borrow-reads borrows that
> binding instead. Distinct user-visible bindings of one underlying
> value — an ordinary alias `y = x` beside its source — are distinct for
> capture: one closure consuming through `y` and another through `x`
> each consume their own binding, and both closure creations are
> accepted. The sole forwarding is a consuming capture of a destructured
> component (or of an alias of one), which consumes the component's
> carrier binding.

These binding identities also govern runtime lookup: a named declaration's
free references are not rebound by a caller's same-named local or parameter.
An anonymous closure retains its actual lexical captures even when a declaration
has the same name. Reusing an initialized declaration does not introduce a new
lexical binding, change which callable is selected, or memoize function results.

> **[04-LIN-3]** Evaluating an expression of an owned linear type SHALL
> produce exactly one logical owner. Binding another name to that value does
> not create a second owner. A second terminal use is legal only when an
> explicit or compiler-inserted `copy` creates another owned value. Each
> logical owner SHALL reach exactly one terminal consuming use or `Drop` on
> every control-flow path; a borrow neither creates nor terminates an owner.

> **[04-LIN-4]** An owned function parameter is a consuming call edge and a
> borrowed parameter is a non-consuming call edge. Every function result of
> an owned linear type is a new logical owner for the caller on every return
> path, including a path whose result has the same value or physical storage
> as an argument or capture. The result owner may be the owner transferred
> through an owned parameter. A borrowed argument or still-live capture may
> become an owned result only after the ordinary copy operation creates an
> independent owner. A backend SHALL NOT infer a returned owner from pointer
> equality, a source name, or a selected return arm.

> **[04-LIN-5]** An `if`, `match`, loop, or fold that produces an owned linear
> result SHALL join with exactly one owned incoming value from every
> predecessor path. A path that forwards an existing owner transfers it into
> the join; a path that preserves another live use first creates a copy.
> Fold and loop-carried owners are block parameters: each iteration consumes
> the previous owner exactly once and produces the next owner exactly once.
> Alias provenance SHALL NOT be overwritten when the loop target is rebound.

> **[04-LIN-6]** Each entry in a top-level root manifest is, in manifest
> order, an implicit terminal consuming use of the binding it observes.
> Root observation participates in the same copy insertion as any authored
> consuming fan-out. Therefore two roots denoting one value receive two
> independently owned results, while the final root may consume the original.
> Every owned top-level value that is not consumed by a manifested root or an
> authored use receives a `Drop`.

> **[04-LIN-7]** A compiled artifact's externally supplied entry arguments
> are borrowed for the complete invocation, irrespective of the owned
> parameter modes used by calls inside the artifact. The artifact SHALL
> neither mutate nor release their storage. Every owned result returned across
> that boundary is an independent owner for the caller, even when its value is
> equal to an entry argument. Before an entry value crosses an internal
> owned-parameter edge, the compiler SHALL create an ordinary copy; an internal
> borrowed-parameter edge may borrow the entry value directly. Passing the
> result back transfers or borrows it
> only according to the next invocation's boundary; ownership is never
> inferred from address equality.

> **[04-LIN-8]** After an owner's terminal use, that owner is no longer live;
> a move may transfer the value to one explicit successor owner. A `Drop` or
> consuming use with no successor owner SHALL make its storage reclaimable
> before a following tail call or loop back-edge whose live set excludes it.
> In-place reuse is permitted only for program-owned storage
> proved unique at that point. Entry arguments, borrowed values, storage with
> another live owner, and storage retained by a view never satisfy that proof.
> An implementation may reclaim later only when an explicitly live owner or
> view requires the storage; recursion depth alone is not such a reason.

> **[04-LIN-9]** A key-carrying value (§8.4.1) is affine: it SHALL have
> at most one consuming use on every control-flow path, and it needs none,
> so an unused one is dropped. No explicit or compiler-inserted `copy`
> applies to it, so the copies that [04-LIN-3], [04-LIN-5] and [04-LIN-6]
> allow for another owned value are refused for it. It SHALL NOT be
> borrowed, copied, captured by a closure, or read by any operation that
> leaves it live, and no signature SHALL declare a borrowed key-carrying
> parameter. Binding it to another name moves it. A key-carrying value
> reaches only the operations this rule admits: a key derivation
> ([05-OP-70] to [05-OP-72]) or a draw ([05-OP-8], [05-OP-37]) at its key
> operand; `drop` ([05-OP-67]); a runtime branch's join; the construction
> and destructuring of tuples, records and data values; a binding, or a
> block's, a handler region's or a function's result; a builtin operation
> whose atom routes each value of a type parameter to exactly one consumer;
> and a call through a
> parameter whose declared type carries a key, whether direct or through
> `grad`, `vmap` or `jit`. Every other operation refuses it, `realize`,
> `cast` and `copy` included, and a builtin passed as a function value
> admits it only where a call to that builtin would. Every call consumes a
> key-carrying argument, including the otherwise observational arguments of
> a `grad(f)(...)` or `vmap(f)(...)` call. A key inside a key-carrying
> value is reached only by consuming that value: a destructuring `let` or
> `match` pattern; a tuple projection, which takes each key-carrying
> component at most once and leaves the tuple unusable as a whole; a field
> access, which consumes the whole value; or `vmap` over a key axis, which
> gives each row to one application. Reading a consumed key's bits again in
> a backward pass or a checkpoint recomputation is not a use. A key tensor's
> extent is not key material, so `shape`, `numel` and a call's extent check
> admit a key tensor and read its extent without a use, leaving the key live,
> and like any read such a read precedes the key's consuming use. A violation
> is a type error whose suggested repair derives fresh keys with
> `split_key` or `split_keys` ([05-OP-70], [05-OP-71]), never `copy`.

> **[04-LIN-10]** A function's type parameter SHALL NOT be instantiated at a
> key-carrying type (§8.4.1), because a generic body may use a value of its
> parameter type more than once. Every type variable that a definition's or
> a generalized `let` binding's type scheme quantifies is such a parameter
> unless it belongs solely to an already-checked key-derivation builtin's
> closed operation relation ([04-INF-9], [05-OP-69]..[05-OP-72]). This
> includes an authored type binder, even one
> that names a tensor element dtype with or without a dtype bound, and
> a variable that inference leaves free in the binding's type, whatever the
> bound value is, since a tuple or a data value can hold a closure over it.
> A closed relation variable is selected by that operation's contract, with
> operation identity preserved through aliases and aggregates.
> A variable exposed outside that exact callable or shared with another
> constraint remains a generic parameter and SHALL stay key-free. A `let`
> binding without an ascription is generalized: `e = Nil` followed by
> `Cons(k, e)` instantiates `e`'s type parameter at `key` and is refused,
> while the ascribed `e: List[key] = Nil` and the value written where it is
> used, `Cons(k, Nil)`, quantify nothing. The rule holds
> however the generic is reached: called directly, bound to another name,
> stored in a tuple or a data value, returned from a function, or passed as
> an argument to another function, a builtin operation included. A key
> reaches a function only through a parameter whose declared type carries a
> key without a type parameter, such as `key`, `tensor[n, key]`, or a data
> type with a key field. A data type's own type parameters are not function
> type parameters: `List[key]`, `Option[key]`, and a data type instantiated
> at a key are key-carrying types that construction, matching, and the
> builtin operations handle under [04-LIN-9]. A violation is a type error
> that names the generic, and the type parameter when the program spells
> one. Its suggested repair passes the key through such a parameter or, when
> the generic is a value binding rather than a function, ascribes the
> binding a type with no type parameter or writes its value where it is
> used.

Diagnostics for violations of these rules SHALL name a binding the
program's source spells — the alias or component name written at the
faulting use — never a compiler-synthesized intermediate.

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
new-code `Outer` whose carrier classification depends on a library `Inner` is
recognized correctly.

#### 8.4.1 Key-carrying types

A type **carries a key** iff it is in the least relation satisfying:

- `key` carries a key, and so does `tensor[..., key]`.
- `(t1, t2, ...)` carries a key iff some `ti` does.
- `&U` carries a key iff `U` does.
- `U[arg1, arg2, ...]` carries a key iff `U` is key-carrying, **or** some
  `argi` carries a key.
- Function types `t-fn` carry no key, even when their parameters or return
  do. A closure never captures a key ([04-LIN-9]).

An ADT is **key-carrying** iff one of its variant fields has a type that
carries a key. The relation is resolved as the §8.4 carrier set is: a least
fixed point over every `deftype` visible at check time, library and new code
together. A value that carries both a tensor and a key follows §8.4 and
[04-LIN-9] at once, and [04-LIN-9] refuses the borrows and copies that §8.3
would otherwise allow it.

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

Rationale: top-level builtin names identify the language's intrinsic call
surface. Rejecting a same-name top-level declaration prevents a declaration
from presenting a signature that differs from the intrinsic operation.

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
  stdlib's `Std.Test.fail` relies on this).
- Function parameters and block-local bindings may reuse builtin names: they
  shadow the builtin under ordinary lexical scoping in every lane. A call to
  that unqualified name invokes the innermost local binding, including when
  the binding is function-typed. The shadowed builtin is inaccessible by that
  unqualified name until the local scope ends.
- Builtin-adjacent reserved keywords (`cast`, `grad`, `vmap`, ...) are not
  part of `BUILTIN_NAMES`; a `def cast` is already a parse error.

---

## 9. Numeric Value Semantics

> **[04-NUM-1]** Every numeric op result SHALL be finalized into its
> declared dtype - rounding for floats, width and domain checks for
> integers and bool - before it becomes observable to any subsequent op,
> comparison, fold, or output, in every lane and on every surface
> (scalar and tensor alike).

> **[04-NUM-2]** Float finalization SHALL be IEEE-754 round-to-nearest,
> ties-to-even, at the dtype's own STORAGE width (f64 identity; f32
> 24-bit, f16 11-bit including subnormals, bf16 8-bit mantissa), with
> overflow to the correctly signed infinity and signed zero and infinities
> preserved. Any floating arithmetic or numeric conversion that produces a
> NaN, including one given a signaling-NaN input, SHALL finalize to the
> target dtype's canonical quiet NaN: f16 `0x7e00`, bf16 `0x7fc0`, f32
> `0x7fc00000`, or f64 `0x7ff8000000000000`. Arithmetic preserves the NaN
> class, not an input payload or sign. A pure bit-moving or selection
> operation preserves NaN payload bits only when its governing operation atom
> explicitly says it is bit-preserving. The width at which the op is COMPUTED
> before finalization is fixed by [04-NUM-8], not by this atom.

> **[04-NUM-3]** Integer op results that are not exactly representable
> in the declared width SHALL trap with the branded overflow diagnostic;
> no lane and no surface SHALL wrap (except via the named modular
> operations of [04-NUM-7] and the named `cast_wrap` of [04-NUM-16]),
> saturate (except via the named `cast_saturate` of [04-NUM-16]), or
> silently widen.
> In-range integer arithmetic SHALL be exact at every width.

> **[04-NUM-4]** A `bool` value SHALL be exactly 0 or 1. Arithmetic operators
> and arithmetic reductions SHALL reject a `bool` operand at the checker;
> they SHALL NOT promote, wrap, saturate, reinterpret, or otherwise admit
> bool-arithmetic behavior. `and`, `or`, and `not` are the logical operations,
> and [05-OP-29] `count` is the bool-tensor counting operation. If a corrupted
> internal or external carrier presents any other bool representation, the
> first typed boundary SHALL trap `Domain` before the value enters an
> operation.

> **[04-NUM-5]** Comparisons SHALL compare finalized values: a cast's
> rounding applies before any comparison reads it, including in
> compile-time condition folds, which SHALL either fold with exact
> per-dtype semantics or decline to fold. A fold SHALL never remove a
> branch that exact semantics would take, and SHALL never fold away or
> introduce a trap.

> **[04-NUM-6]** `f64 add(2^53, 1) == 2^53` and every other correctly
> rounded float result at the dtype's own mantissa boundary is CORRECT
> and SHALL NOT be "fixed"; identical printed numbers at an integer
> dtype are a defect. Same inputs, opposite verdicts, by design.

> **[04-NUM-7]** A named modular-arithmetic operation - exactly `wrap_add`,
> `wrap_sub`, or `wrap_mul` - on a signed-integer dtype SHALL produce the unique
> value
> in that dtype's range congruent to the exact mathematical result
> modulo 2^width, and SHALL NOT trap. These operations are the
> explicit, user-visible escape hatch for modular arithmetic (hashing,
> RNGs, checksums); the wrap prohibited by [04-NUM-3] is the *implicit*
> overflow behavior of the ordinary arithmetic ops, not these named
> ops, whose result is in-range by construction. Named modular
> operations are defined ONLY on the integer dtypes (`i8`, `i16`,
> `i32`, `i64`); they SHALL NOT be defined on `bool` or any float
> dtype (floats overflow to infinity per [04-NUM-2] and have no modular
> escape hatch by construction), and requesting one on a non-integer
> dtype is a checker-level type error.

> **[04-NUM-8]** Every dtype declares a STORED REPRESENTATION and an
> ARITHMETIC WIDTH in addition to its storage width. Every op SHALL be
> performed at its operands' arithmetic width and finalized to the storage
> width once per op, in every lane and on every surface. No lane SHALL compute
> at any other width. The representations and arithmetic widths are:
>
> | dtype | stored representation | storage width | arithmetic width |
> |---|---|---|---|
> | `f64` | IEEE-754 binary64 | 64 | f64 |
> | `f32` | IEEE-754 binary32 | 32 | f32 |
> | `f16` | IEEE-754 binary16 | 16 | f32 |
> | `bf16` | bfloat16 | 16 | f32 |
> | `i64` | signed two's-complement 64-bit integer | 64 | exact i64 |
> | `i32` | signed two's-complement 32-bit integer | 32 | exact i32 |
> | `i16` | signed two's-complement 16-bit integer | 16 | exact i16 |
> | `i8` | signed two's-complement 8-bit integer | 8 | exact i8 |
> | `bool` | canonical Bool8 (`0x00` false, `0x01` true) | 8 | not an arithmetic dtype ([04-NUM-4]) |
> | `key` | opaque 64-bit word, every bit pattern a key ([05-RNG-2]) | 64 | not an arithmetic dtype (§1.1) |
>
> Stored representation, storage width, and arithmetic width are separate
> facts. Equal storage widths do not make two representations interchangeable:
> for example, `f32` and `i32` are both 32 bits, and `bool` and `i8` are
> both 8 bits, but neither pair may share a typed load, store, carrier, or
> kernel element spelling. Every boundary and lane SHALL match the exact
> representation identity, not only its byte width. No implementation may
> infer arithmetic width from storage width or storage width from arithmetic
> width.
>
> The `matmul`, `sum`, and `einsum` accumulator parameter of §5.7 is the ONLY
> user-selectable widening; it is explicit, typed, defaulted per §5.7.1,
> and it does not license any other widening. An implementation MAY
> compute a BASIC f16 or bf16 operation (add, sub, mul, div, sqrt)
> natively where the target provides that instruction: for exactly those
> operations the f32 intermediate and the native narrow op produce
> identical bits (f32 carries >= 2p+2 bits for p <= 11), so the native
> form is an implementation of the declared width, not a departure from
> it. The permission does not extend to transcendentals or any other op
> where the bits could differ - those compute at f32 per the table. No
> lane SHALL compute at any width wider than the one declared above.
>
> REDUCED-precision computation - performing an op at an arithmetic width
> narrower than the one declared above SUCH THAT THE RESULT BITS CAN
> DIFFER, such as a 19-bit tensor-core mode for f32 matmul (the
> bit-identical native-narrow case above is not reduced precision) -
> SHALL be available only through a named, explicit
> opt-in at the call site, SHALL never be a default, and SHALL never be
> selected by a backend, a build flag, or a global mode. This specification
> defines no reduced-precision opt-in; narrower-than-declared computation is
> therefore non-conforming in every lane.
>
> Arithmetic a backend SYNTHESIZES that is not an op on program values -
> the loop counters and addressing expressions of generated code - carries
> no declared dtype and is outside the table above. It MAY use a machine
> width whose range provably contains every value it carries, and it SHALL
> NOT be the carrier of a declared-dtype value crossing a boundary
> ([04-NUM-11] governs those crossings).

**Rationale for the f16/bf16 rows.** These are not exclusions carved out
of a general rule; the arithmetic width is part of what the format is.
BF16 instruction families such as AVX512-BF16 `vdpbf16ps` and ARM
BFDOT/BFMMLA accumulate into f32 rather than providing bf16 addition. f16
arithmetic does exist (ARMv8.2-A,
AVX512-FP16, NVIDIA `__hadd`), which is why the atom permits it, but even
there the transcendental path converts to f32 because the special-function
units are f32. The same property holds one rung down and is why §1.1.1
reserves `f8e4m3`: FP8 is a matmul-input format whose tensor-core ops
accumulate in f32 and which requires an out-of-band scale factor. The
reserved spelling therefore has no self-contained value or finalization
contract.

**Why f32 is not widened.** f32 has native arithmetic on every CPU and GPU
in the supported set, so computing it at f64 invents a width the format
does not have. For the basic operations that substitution is bit-identical
and therefore only forfeits the compute (half the SIMD lanes, two
conversions per op, no tensor-core path); for transcendentals and for
multi-step reductions it also changes the answer, which is what would
otherwise force a cross-lane tolerance table between two lanes that
should agree exactly. The same argument applies to routing exact integer
arithmetic through f64, which additionally destroys i64 exactness above
2^53.

**Why reduced precision is opt-in only.** The prohibition on narrowing is
not symmetric with the prohibition on widening by accident. Widening is
invisible in the result for the basic operations and merely wasteful;
narrowing silently changes results everywhere. The ecosystem's own
experience is the argument: NVIDIA's 19-bit TF32 mode is enabled by
default for f32 matmul on Ampere and later, and PyTorch's
`torch.backends.cuda.matmul.allow_tf32` changing its default silently
altered users' numerics. A program that asked for f32 got something else
because a backend decided. Under §5 that is exactly the class of decision
that must be spelled at the call site, so Chelis admits the capability and
refuses the default.

> **[04-NUM-9]** A numeric trap has a CLOSED set of kinds: `Overflow` (an
> integer result outside the declared dtype's range), `Domain` (an input or
> result violates the operation's defined mathematical domain or the target
> dtype's value set - including a fractional or non-finite value at an
> integer dtype, a value other than 0 or 1 at `bool`, or an empty reduction
> whose operation has no identity), and `DivZero`
> (integer division or remainder by a zero divisor). Every trap SHALL name
> its kind, the operation that raised it, and the dtype it was finalizing
> to, and SHALL render byte-identically in every lane and on every surface.
> A lane that raises a numeric failure outside this set, or renders one of
> these differently from another lane, is a defect. The exact renderings are
> `numeric trap: overflow in <op> at <prim>`,
> `numeric trap: domain in <op> at <prim>`, and
> `numeric trap: division by zero in <op> at <prim>`. The `<op>` slot SHALL
> be the canonical name of the primitive whose numeric kernel produced the
> trap after lowering. When a composed source operation lowers to that
> primitive, the trap SHALL retain the lowered primitive name; it SHALL NOT
> be renamed to the composed operation or wrapped in lane, lowering, or
> evaluator plumbing. No additional prefix or suffix is permitted on any
> user-facing numeric-trap line. A typed operation-precondition guard is
> itself the trap-producing primitive for this rule and carries the guarded
> builtin's canonical name and the dtype of the quantity that guard
> finalizes. A failure raised by such a guard therefore names the guarded
> builtin; it is not a renamed trap from a later primitive in the successful
> lowering.

> **[04-NUM-10]** A numeric trap SHALL be a VALUE inside a lane and SHALL
> become a process failure only at that lane's boundary: `chelis eval`
> raises a diagnostic and exits nonzero; a compiled binary writes to stderr
> and exits nonzero. No lane SHALL panic, abort mid-computation, or
> terminate the compiler for a trap caused by user input. On a device lane,
> where a kernel cannot raise mid-flight, the trap is carried as a
> device-side error flag - at minimum a set/unset flag and the first failing
> element index - read by the host after dispatch completion, which then
> raises with the identical branded message. Dispatch completion IS the
> device lane's boundary, so this rule is satisfied there by construction
> rather than excepted from.

**The availability trade, stated.** Trapping converts silent data
corruption into loud termination, by design: a long-running job that
overflows an `i64` counter DIES where wrapping arithmetic would have
carried a silently wrong value to completion. That operational cost is
deliberate, and this record carries it alongside the benefit: the
alternative outcome is not a successful run but a plausible wrong result,
which is the strictly worse failure mode. Code that WANTS mod-2^width semantics states it with
[04-NUM-7]'s named `wrap_*` operations. Saturating conversion is the named
`cast_saturate` of [04-NUM-16]; saturating arithmetic would require separate
named operations if introduced. Behaviors are named operations, never modes.

> **[04-NUM-11]** A value SHALL survive storage, transport, and every
> boundary crossing at its declared dtype without collapse. An `i64`
> value above 2^53 that is exact when produced SHALL still be exact after
> being stored in a tensor, serialized onto the execution wire, returned
> through a language binding, and read back. A representation that cannot
> carry a dtype's full value set is not a conforming representation for
> that dtype, and no stage SHALL substitute a wider or narrower one to
> compensate. A language binding or device descriptor SHALL preserve rank as i32
> and each extent, stride, element count, and byte capacity as i64, matching
> the domains of [05-DIM-1], [05-DIM-2], and [05-OP-31]. It SHALL carry the
> exact dtype tag and dynamic rank; a fixed-rank carrier, a narrower metadata
> field, or an element pointer not coupled to the exact tag in the same
> validated descriptor is not a conforming substitute. A
> boundary MAY reject a value outside the declared domain before crossing, but
> it SHALL NOT narrow, clamp, wrap, or fabricate metadata to make it fit.
> A compiled public entry SHALL compare the dtype tag of every tensor supplied
> to it, including a tensor nested in a supplied value, with the declared dtype
> before it reads that tensor's elements. A mismatch is a `Domain` trap in
> `load` at the declared dtype under [04-NUM-9], accompanied by the input's
> name and both dtypes; the supplied storage is never read at the declared
> dtype. (Not fully implemented for the HIP and Metal entries, chelis#2510.)

> **[04-NUM-12]** A numeric trap's OCCURRENCE is deterministic within a
> lane and is defined by that lane's documented evaluation order. For a
> multi-step operation (a reduction or scan), whether an intermediate
> result leaves the accumulator dtype's range - and therefore whether
> the operation traps - is evaluated against the lane's documented
> accumulation order in `spec/05-risc-primitives.md`. Same program, same
> inputs, same lane SHALL always produce
> the same trap or the same completion. Where two lanes document
> different accumulation orders, an integer accumulation whose
> intermediate results approach the accumulator's range MAY trap in one
> lane and complete in another; a lane that completes SHALL produce the
> exact result. Trap-versus-exact at accumulator range edges is the ONLY
> permitted cross-lane divergence in trap behavior: trap versus a wrong
> value is never permitted, the tolerance table never applies to traps,
> and when two lanes both trap, [04-NUM-9]'s rendering identity applies
> in full.

> **[04-NUM-13]** `shl` and `shr` on a signed integer dtype SHALL operate
> on that dtype's fixed-width two's-complement bit pattern. `shl` discards
> bits beyond the declared width and `shr` is arithmetic (sign-extending).
> A non-negative count at least the declared width produces zero for
> `shl` and for `shr` of a non-negative value, and `-1` for `shr` of a
> negative value. A negative count SHALL trap with
> `shift amount must be non-negative, got N`. These semantics are
> independent of host-language signed-shift behavior; a compiled backend
> SHALL NOT invoke undefined or implementation-defined signed shifts.

> **[04-NUM-14]** `cast(source, target)` SHALL be explicit and SHALL apply
> the target dtype's finalization rule identically on scalar and tensor
> surfaces. An integer or bool source cast to an integer target preserves
> the exact value or traps `Overflow` when it is out of range. A float source
> cast to an integer target SHALL trap `Domain` unless it is finite and
> integral, and SHALL trap `Overflow` when that integral value is outside
> the target range; the default cast SHALL NOT truncate, round, saturate, or
> wrap. A source cast to a float target is total IEEE-754 round-to-nearest,
> ties-to-even at the target width per [04-NUM-2], even when that loses
> integer exactness. A source cast to `bool` accepts exactly 0 or 1 per
> [04-NUM-4]. For automatic differentiation, a float-to-float cast has
> adjoint `cast(g, source_dtype)` on the same scalar or tensor surface; the
> backward cast obeys this atom's exact target-width finalization and traps.
> An exact same-float-dtype cast is the identity in both directions. A float
> source cast to an integer or bool target is piecewise constant and
> structurally rejects `grad` with
> `AdRejectionReason::PiecewiseConstant`; it never contributes a silent zero.
> A bool or integer source is a discrete forward-only value and carries no
> cotangent, irrespective of target. These rules apply equally to explicit
> default casts introduced inside another authored operation.

> **[04-NUM-15]** For an elementwise operation that maps a tensor to a
> tensor and can trap per-element, the trap the operation raises SHALL be
> the one belonging to the offending element with the LOWEST row-major flat
> index. This fixes the "documented evaluation order" [04-NUM-12] requires
> for the elementwise case: an elementwise trapping map's documented order
> is row-major flat index order, in every lane. When a tensor carries
> several offending elements whose kinds differ, the lowest-indexed
> offender therefore determines the kind, and [04-NUM-9]'s rendering
> identity applies to that trap unchanged. A lane SHALL NOT select the
> offender by evaluation happenstance - thread scheduling, a whole-buffer
> pre-pass biased toward one kind, or vectorization order - and SHALL NOT
> report a different kind from another lane for the same input. This rule
> does not constrain HOW a lane finds that element: a parallel
> implementation that reduces per-thread candidates to the global minimum
> index is conforming, and serialization is not required.

> **[04-NUM-16]** The named lossy casts are separate operations, never a
> mode of the checked `cast` in [04-NUM-14]. `cast_saturate(source, target)`
> admits an active signed-integer or float source and a signed-integer target.
> It reads the source value exactly at its stored dtype, truncates a finite float
> toward zero, and clamps the resulting mathematical integer to the target's
> inclusive range; `-inf` clamps to the target minimum, `+inf` to its maximum,
> and `NaN` traps `Domain`. `cast_wrap(source, target)` admits active
> signed-integer source and target dtypes and returns the unique signed
> target-width representative congruent to the exact source modulo
> `2^target_width`. Both operations apply
> elementwise with identical scalar and tensor semantics. Neither admits
> `bool`, `string`, or a reserved dtype. A rounding cast is not a separate
> operation: a program spells `round(source)` followed by checked `cast`.
> [04-NUM-15] governs the selected failure when a tensor `cast_saturate`
> contains more than one `NaN`.

---

### 9.1 Per-Dtype Value Semantics (Consolidated Normative Table)

The atoms above decide these cells one rule at a time; this table is the
consolidated view of the same decisions, per dtype, and is normative. Where a
cell and an atom disagree the atom wins and this table has a bug. "Wide
intermediate" below means the value an op kernel produced at the dtype's
ARITHMETIC WIDTH ([04-NUM-8]) before finalize - NOT an unconditional f64 or
i64.

| dtype | value set | arithmetic width | finalize(wide) | overflow / out of range | special values |
|---|---|---|---|---|---|
| `f64` | IEEE binary64 | f64 | identity except canonical-NaN finalization | n/a (IEEE handles it) | arithmetic/conversion NaN -> `0x7ff8000000000000`; ±inf and -0.0 preserved |
| `f32` | IEEE binary32 | f32 | RNE to 24-bit mantissa | rounds to ±inf per IEEE | arithmetic/conversion NaN -> `0x7fc00000`; ±inf and -0.0 preserved |
| `f16` | IEEE binary16 | f32 | RNE to 11-bit mantissa, incl. subnormals | overflow -> ±inf (`mul(65504f16, 2f16) = inf`) | arithmetic/conversion NaN -> `0x7e00`; ±inf and -0.0 preserved |
| `bf16` | bfloat16 | f32 | RNE to 8-bit mantissa | overflow -> ±inf | arithmetic/conversion NaN -> `0x7fc0`; ±inf and -0.0 preserved |
| `i64` | integers in [-2^63, 2^63-1] | exact i64 | must be integral and in range, else trap | trap: `Overflow` out of range, `Domain` non-integral ([04-NUM-9]) | none |
| `i32` / `i16` / `i8` | integers at width | exact at width | same rule at width | trap: `Overflow` / `Domain` at width | none |
| `bool` | {0, 1} | n/a (not an arithmetic dtype) | must be exactly 0 or 1, else trap | trap, kind `Domain` | none |
| reserved names (§1.1.1) | rejected by the checker | - | unreachable: rejection is compile-time-visible, never a runtime arm | - | - |

Reading notes:

- The **arithmetic width** column is [04-NUM-8]'s table restated per row. It
  is deliberately NOT reproduced into §1.1.3's per-backend matrix: it is a
  target-independent language fact rather than a target capability cell.
- The **trap cells** name [04-NUM-9]'s kinds; for multi-step operations,
  trap OCCURRENCE is governed by [04-NUM-12] (each lane's documented
  accumulation order).
- The **f64 row's identity finalize** is why [04-NUM-6] holds:
  `f64 add(2^53, 1) == 2^53` is the correctly rounded answer and stays. The
  same two numbers at `i64` are exact or trap, never silently collapsed -
  same inputs, opposite verdicts, by design.
- The **bool row has no arithmetic width** because arithmetic on `bool` is
  rejected rather than performed ([04-NUM-4]). `and` / `or` / `not` are the
  logical operations and `count` is the bool-tensor counting operation.
- The **reserved row** covers every name in §1.1.1. Those names are rejected
  before evaluation and therefore have no runtime finalization semantics.

## 10. Checker Totality

> **[04-TOT-1]** Every Deep tag in the closed vocabulary
> (spec/03-deep-syntax.md) SHALL have an explicit checker disposition: a
> real inference case, or a rejection with a pushed diagnostic. A
> construct the checker does not recognize SHALL produce a diagnostic,
> never a silent exemption of its subtree.

> **[04-TOT-2]** If a check completes with an empty error vector, the
> typed result SHALL contain no error-typed expression: `Type::Error`
> without a corresponding reported diagnostic SHALL be unconstructible.

> **[04-TOT-3]** A structurally malformed Deep form that reaches the
> checker SHALL be rejected with a diagnostic naming the tag and the
> expected shape; deferring the failure to a later stage is not a
> disposition.

> **[04-TOT-4]** Every child of a Deep form whose content that form's
> semantics reads SHALL be consumed by that form's checker disposition or
> rejected with a pushed diagnostic. Where a form reads such a child
> through a partial extraction - an integer axis, a symbol constructor
> head, a function-typed operand - a failed extraction SHALL push a
> diagnostic naming the form and the shape it expected. An omitted
> optional child and a present child the form cannot read are distinct
> inputs: only the omission MAY take the form's declared default.
> Coverage is a property of the submitted program rather than of the
> checked result, so a node inference never visited SHALL NOT be reported
> as successfully checked on the ground that the result it is absent from
> contains no error.

> **[04-TOT-5]** A Deep program's checker verdict SHALL NOT depend on which
> checker entry receives it, nor on which admitted representation carries it.
> For one program, every entry SHALL accept or reject alike and SHALL report
> the same defects; a check that one admitted representation receives SHALL be
> applied to every other admitted representation of the same program. A
> representation the checker admits but a check cannot read is a silent
> exemption under [04-TOT-1] and SHALL be diagnosed rather than skipped. A
> check applied at one entry and not another SHALL be resolved by deciding the
> check, never by narrowing the entry that applies it: either every entry
> applies it, or no entry does and the rejection it performed moves to the
> stage whose capability it describes.
