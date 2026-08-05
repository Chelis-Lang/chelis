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

> **[04-DTYPE-1]** A primitive type position SHALL name one of the active
> primitives in §1.1. Every deferred name in this section is reserved but
> rejected by the type checker, including as a `cast` target, until a spec
> revision moves it into the active set. Any other primitive spelling is an
> unknown type and is likewise rejected; it SHALL NOT acquire the operand's
> type or a default dtype.

The following primitive names are reserved in the spec but **not active** in
the current dtype build-out cycle. They are documented here rather than
silently dropped so producers do not assume they have gone away. Every name
below is rejected by the type checker with a diagnostic pointing at this
section, and `cast(x, <name>)` is rejected with it.

Each reserved name declares its ARITHMETIC WIDTH ([04-NUM-8]) at reservation
time, so the width question is settled before anyone implements the dtype
rather than being decided per-lane during implementation - the failure mode
chelis#727 documents. The names and widths track the ecosystem's own
vocabulary (numpy, PyTorch, JAX, Apache Arrow, and the OCP 8-bit and
Microscaling specifications); Chelis reserves the standard spelling and the
standard semantics, never a bespoke variant.

**Reduced-precision floats.**

- `f8e4m3` — 8-bit float (E4M3 per the OCP 8-bit Floating Point
  Specification). Arithmetic width: **f32**. **Rationale:** no current Chelis
  backend implements f8e4m3; revisit when a concrete backend (HIP, C, or
  Metal) gains native E4M3 support and a corresponding evaluator round-trip
  representation. Note OCP E4M3 is not IEEE-shaped: it has no infinities and
  reuses that encoding to extend range, so its finalize row is not a
  parameterization of [04-NUM-2] and must be authored when it activates.
- `f8e5m2` — 8-bit float (E5M2 per the same OCP specification). Arithmetic
  width: **f32**. **Rationale:** E4M3 and E5M2 are one format pair, not two
  independent dtypes - FP8 training uses E4M3 for forward values and E5M2 for
  gradients, and every implementation that ships one ships both. Reserving
  only E4M3 would reserve half a format.

**Unsigned integers.**

- `uint8`, `uint16`, `uint32`, `uint64` — unsigned integers at width.
  Arithmetic width: **exact at their own width**, unsigned. Overflow traps per
  [04-NUM-3]; the named modular operations of [04-NUM-7] are the wrapping
  escape hatch and extend to these dtypes when they activate. **Rationale:**
  universal in the ecosystem (numpy, PyTorch, JAX, and Arrow all carry the
  full set) and unavoidable at real ingress boundaries - `uint8` is the image
  dtype, and hashing, checksums, and PRNG state are natively unsigned.
  Deferred rather than active because the implementation surface is real:
  separate unsigned comparison and division, unsigned AD adjoints, and
  per-backend dispatch. Until activation, cast to a signed type (typically
  `int32` or `int64`) at the boundary where unsigned data enters.
- `int4`, `uint4` — 4-bit integers. Arithmetic width: **exact at their own
  width**. **Rationale:** the quantized-inference frontier (JAX carries both).
  Deferred additionally on a storage question the other widths do not raise:
  a 4-bit element has no addressable byte, so activation requires a packing
  decision (two elements per byte, and which nibble is element zero) that
  belongs with the representation work, not with this list.

**Complex.**

- `complex64`, `complex128` — complex numbers with `f32` and `f64` components
  respectively. Arithmetic width: **f32 and f64 components**. **Rationale:**
  present in numpy, PyTorch, and JAX under exactly these spellings; the naming
  convention is TOTAL bits, so `complex64` is a pair of f32 - Chelis follows
  the ecosystem spelling rather than inventing `complex32x2`. Deferred because
  complex multiply and divide have their own accuracy and overflow story
  (naive division overflows for representable inputs; the ecosystem uses
  Smith's algorithm or a variant) which must be authored, not inherited.

**Decimal (interchange only).**

- `decimal128`, `decimal256` — exact base-10 values at a declared
  (precision, scale), per Apache Arrow. Arithmetic width: **exact base-10 at
  the declared scale**; this is a third arithmetic family alongside float and
  integer, not a parameterization of either. **Rationale and scope limit:** no
  array-computation library carries decimal - numpy, PyTorch, and JAX all
  decline it - because it is a database and dataframe type. Chelis already
  serves that need twice: `Std.Decimal` is the standard-library scalar type
  (built on `trunc_div` scale shifts, see `spec/05-risc-primitives.md` §2.1),
  and `Std.Io.Parquet` is the interop surface. These names are therefore
  reserved for the **Arrow and parquet interchange boundary only** and are
  explicitly NOT reserved as tensor element types; activating them does not
  put decimal into the tensor arithmetic path. They are also the only reserved
  names that are parameterized, which is itself a reason to keep them off that
  path: every other `Prim` is a bare name.

**Not reserved, and deliberately so: scaled and block-scaled formats.**
`qint8`/`quint8` (PyTorch), the MX formats of the OCP Microscaling
specification (MXFP8, MXFP6, MXFP4), and FP8 training's amax-scaled tensors
are NOT additional dtypes. Each is a storage dtype PLUS scale metadata - per
tensor, per channel, or per block of 32. Modelling scale once, as an axis over
a storage dtype, admits all of them; adding them as primitive names would
recreate exactly the per-cell scatter chelis#727 exists to end. When scaled
storage is authored it is authored as that axis, and this paragraph is the
record of the decision.

**Not reserved: saturating arithmetic.** Image pipelines saturate rather than
trap or wrap (OpenCV and PIL both, and SIMD provides native saturating adds
such as `paddusb`). If `uint8` image work activates, `sat_add` and siblings
are the named ops to author, on the [04-NUM-7] pattern - a named op, never a
mode. Recorded here so the third behavior is not rediscovered as a surprise.
The pressure will arrive as "wrapping is wrong AND trapping is tiresome" the
day `uint8` image work is real; the answer that preserves [04-NUM-3]'s
doctrine is the named op, authored then - not a mode, and not a silent
default.

#### 1.1.2 Unsigned Integer Types: Deferred, Not Out Of Scope

Unsigned integer types are reserved under §1.1.1 as `uint8`, `uint16`,
`uint32`, and `uint64`, with `int4`/`uint4` alongside them. This section
formerly declared them "out of scope for this cycle" on the rationale that no
customer use case justified the implementation surface. That stance was
revised 2026-07-28: the implementation-surface argument still holds and keeps
them deferred, but declaring a type the entire ecosystem carries to be out of
scope left Chelis without a reserved spelling for data it must eventually
ingest, and the deferral list is the honest home for that. The short
spellings `u8`/`u16`/`u32`/`u64` are NOT reserved; the `uint*` spellings are
canonical, matching numpy and Arrow.

The workaround while deferred is unchanged: cast to a signed integer type
(typically `int32` or `int64`) at the boundary where unsigned data enters the
program.

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

**Arithmetic width is not a cell of this table.** It is a target-independent
property of the dtype, declared once by [04-NUM-8] and owned by the semantic
side of the capability split (`spec/design/capability_table.md`'s Table A,
whose rejections the CHECKER reports because they hold whatever the target
is). This table is Table B's ancestor: it records per-backend implementation
status, which is a different decision reported by build and lowering where
the target is known. Reproducing the widths here would put an A-fact in a
B-table and create a second copy to keep in sync by hand.

The constraint this table DOES carry is the consequence: a backend that
admits a dtype computes it at the width [04-NUM-8] declares, and a backend
that cannot is a `rejected` cell - never an `admitted` cell that silently
widens, narrows, or substitutes. The bf16/f16 C-backend notes below are that
rule already applied, not an exception to it.

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

> **[04-TGT-1]** The Metal target SHALL reject every f64 value before kernel
> emission and SHALL direct the user to the C or HIP target. It SHALL NOT
> substitute f32 or software-emulated arithmetic for the language's f64
> semantics.

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

### 2.3.1 Generic ADT representation classification

> **[04-ADT-1]** Backend layout and generic-function specialization SHALL use
> the checked program's alias-resolved ADT registry and checked authored
> signatures. They SHALL NOT reconstruct parameter ownership, constructor
> fields, or polymorphism from authored `deftype` / `defsig` syntax.

A parameter used only as a tensor dimension is representation-erased; a
parameter used as a tensor precision or another value-represented field is
stored. The distinction is recursive through nested ADTs. Thus
`Frame[n,a] -> Hamt[Column[n,a]] -> tensor[n,a]` erases `n` but must retain
and specialize `a` (chelis#940, chelis#948). The checked registry and authored
signature marker survive serialization, context composition, effects
reannotation, and linearity annotation.

> **[04-ADT-2]** Every use of a generic function signature SHALL instantiate
> a fresh substitution. When computing the free variables of an environment
> scheme, a substitution SHALL NOT be applied through that scheme's quantified
> type, dimension, or rank variables, including through an alias chain. Thus
> one concrete tensor extent, rank, precision, or stored ADT argument cannot
> constrain a later use in the same consuming module (chelis#968).

### 2.4 Exhaustive Pattern Matching

The type checker verifies that `match` expressions cover all variants. Missing variants are a type error, not a warning.

A top-level irrefutable arm covers the match: a bare variable pattern
(`| x =>`) or an as-pattern whose inner pattern is irrefutable
(`| q @ x =>`, `| q @ _ =>`). The coverage applies at the arm level
only; a variable pattern NESTED inside a constructor or record pattern
does not cover the other variants.

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
test`, `chelis eval`'s in-context path, module-init, and the legacy
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

**Gating.** `chelis check` is a scorer-with-exit-code: it always reports a
fitness score and the full error list, and its exit code mirrors that list
(`0` iff empty, non-zero otherwise; Issue #207). A declaration error such
as `OpaqueTypeViolation` is therefore visible on `check` (a non-zero exit
with the error listed), never a silent score-1 pass. The front-end
surfaces that consume a program -- `chelis build`, `chelis eval --file`,
and the in-process check entry the prove pipeline calls -- gate on a
non-empty error list and refuse to proceed. `chelis validate` is a
structural Deep/Surf well-formedness validator and does not run the type
or opacity checker, so it does not gate on these semantic declaration
errors; use `check`/`build`/`eval` for that.

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
  variant; every field must be in the V1 value class: a **numeric or
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
The invariant is consumed by `chelis prove` (derived producer
obligations and assumption injection) in later workstreams, not by
`chelis check`.

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

The read-only container queries `len` and `index` are observational on a
tensor-carrying `List` / `Dict`: they auto-borrow their container argument and do not
consume it, so reading a list's length or an element does not forbid a later reuse of the
list (chelis#527). See `spec/05-risc-primitives.md` §1.3.1.

### 3.1 Algorithm

Standard Algorithm W with extensions for tensor types. The flow:

1. **Constraint generation:** Walk the typed Deep AST. At each node, generate type equations between the expected type and the actual type.
2. **Unification:** Solve the constraint set. Unification handles type variables, function types, tensor types (with dimension unification), and ADT types.
3. **Generalization:** At `let` boundaries, generalize unconstrained type variables to produce polymorphic types.
4. **Annotation checking:** Where the programmer/agent provided `type` metadata, check that the inferred type is compatible with the annotation.

> **[04-INF-1]** An unannotated lambda whose body reaches a semantic typing
> rule while an operand's outer type constructor is still unknown SHALL retain
> that rule as an obligation and SHALL remain monomorphic until an application
> binds the operand. The first such application binds the lambda and replays
> the same semantic rule; every later use has that same instantiation. If no
> application or annotation resolves the obligation by the enclosing
> declaration boundary, the declaration is a type error. Ordinary lambdas
> with no deferred semantic obligation generalize normally. A symbolic tensor
> is not an unknown constructor: declared dimension and precision variables
> remain polymorphic, independently declared rigid dimensions remain distinct,
> and they unify only when an ordinary body constraint requires equality.

The replay requirement applies to every operation whose result or admission
depends on the resolved operand shape, not to a hand-maintained exception for
one builtin. In particular, a `matmul`, reduction, `expand`, `layer_norm`,
`conv2d`, or `scatter_elements` reached through a bare lambda parameter is
checked again after the parameter binds. The check used on replay is the
operation's ordinary typing rule, so immediate and deferred applications
cannot acquire different semantics.

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
| `neg`, `exp`, `log`, `sin`, `cos`, `tan`, `atan`, `sqrt`, `abs`, `floor`, `ceil` | `tensor[D, p]` | `tensor[D, p]` | Dimensions preserved |
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
;; def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32]

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
;; def f[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = y
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
;; def f[k](a: tensor[2, f32]) -> tensor[k, f32] = a
;; TYPE ERROR: the body pins the return-only dim parameter k to the
;; parameter's concrete Lit(2); the signature promised an output
;; dimension the body does not derive from the inputs.

;; def f[n, m](x: tensor[n, f32]) -> tensor[m, f32] = x
;; TYPE ERROR: the return-only dim parameter m collapses with the
;; param-position dim parameter n.

;; def make() -> tensor[n, f32] = to_tensor([1.0, 2.0, 3.0])
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

- Concatenation along an axis whose operand extents are not statically
  summable: `concat(tensor[batch, seq1, f32], tensor[batch, seq2, f32],
  axis=1)` → `tensor[batch, *, f32]` (named `seq1`/`seq2` cannot be
  summed; see §4.5.4 for when concat DOES produce a concrete extent)
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
- **The uniformity check recurses through nested `List` wrappers**
  (chelis#276). It compares the declared and body element axes of a
  `List[tensor[..]]`, and when the element is itself a `List` it descends
  into it, so an inner rigid/named dim under `List[List[tensor[k, f32]]]`
  — at any nesting depth — is checked too. A wildcard tensor nested under
  `List[List[tensor[k, f32]]]` is therefore rejected, the same as the
  single-level `List[tensor[k, f32]]` case: `List[List[tensor[k]]]` still
  promises every innermost element shares length `k`. (The earlier
  single-level check left this same #272 soundness gap one `List`
  deeper.)

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

Multiple **named** axes may be reduced in one call — the variadic form
`sum(x, seq, head)` (chelis#339) — or by composing single-axis reductions
(`sum(sum(x, head), seq)`); the two are equivalent, and the variadic form is
order-insensitive (`sum(x, head, seq)` produces the same result). The variadic
form is defined for the value reductions `sum`, `mean`, `max_reduce`,
`min_reduce`, and `prod_reduce` (for `mean`, reducing axes one at a time with
uniform weights equals the joint mean). It is **not** defined for the
index-returning reductions `argmax_reduce`/`argmin_reduce`: an index along one
axis is not composable with a second reduction, so a variadic call on those is
a hard error. Every axis in a variadic call must be a *named* axis (a
positional integer is only valid as the single axis of a concrete-rank
operand), each name must resolve per the rules above, and a **duplicate** axis
name in the list is a hard error. Inside a `..r` body the Body-Discipline
admission is unchanged (`sum`/`mean` only, chelis#340). At lowering the
variadic call desugars to the composition, innermost stage reducing the last
listed axis.

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

**Named-axis expand (`R+1`).** The inverse arithmetic direction: `expand`
inserts a *named* axis in a rank-polymorphic way when its axis argument is a
dimension name rather than an integer. Two call forms are admitted
(chelis#339):

```chelis
;; insert a trailing named axis (the new axis goes after every existing axis):
;; def add_axis(x: &tensor[..rest, f32]) -> tensor[..rest, one, f32] = expand(x, one, 1)
;; insert immediately BEFORE an existing named anchor (4-arg form):
;; def widen(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, c, seq, ..post, f32]
;;   = expand(x, c, 5, seq)
```

- `expand(x, new, size)` — `new` is a bare dimension name: insert a new
  **trailing** axis named `new` with extent `size`. The symbolic output is the
  operand's row form with `new` appended.
- `expand(x, new, size, anchor)` — additionally name an **anchor**, an existing
  named axis of the operand; the new axis is inserted immediately *before* the
  anchor. Leading-end insertion is expressible exactly when the row begins with
  a named anchor (`tensor[first, ..rest]` + `expand(x, c, k, first)`); a row
  that begins with a spread has no leading anchor and admits trailing or
  anchored insertion only.

Both forms keep unification unitary: the insertion point is either an end of
the row or a position fixed by a named anchor located uniquely in the operand.
Hard errors (`DimensionMismatch`, never a guessed placement):

- the inserted name already names an axis of the operand (a duplicate dim name
  would make every later by-name lookup ambiguous). This holds through
  call-site rank monomorphization too: when the inserted name survives into
  the result row, a caller whose spread-covered axes include that name is
  rejected at check time (the introduced-name rule in `unify.rs`); when the
  inserted name is consumed inside the body (insert + reduce), or the
  collision comes from a single-letter dim *var* whose source letter matches,
  the check cannot see it and the collision surfaces as a **fatal lowering
  error** at build/eval — loud, never a silent wrong-axis resolution;
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
`int32` variable is a runtime value and keeps the compile-time-constant
rejection (issue #259) — `expand(x, ax, 4)` with `ax: int32` is still an
error, never a trailing insert of an axis named `ax`. The `size` argument
must be a **positive compile-time literal** (an `int64` literal `Ni64`, or a
`cast(N, int64)` form):
the inserted axis's extent is stamped onto the new named dim at lowering, and
a symbolic-dim or runtime `int64` size has no stampable extent (the eval lane
cannot stage it and the C backend would reference an undeclared dim symbol),
so those forms are rejected at check time. The positional concrete-rank
`expand` forms (§4.7.2, insert-or-set) are unchanged and keep the three
§4.7.2 size forms. At lowering, the named insertion point
is resolved against the monomorphized operand dims (trailing → operand rank;
anchored → the anchor's index), mirroring named-axis reduction.

**Soundness (§4.2).** Order is preserved (shapes stay ordered positional
sequences — never unordered "rows"); the reduced axis is a retained name; and a
rank-poly def body is restricted by the §4.2 Body-Discipline check to
*name-trackable* operations only — shape-identity (elementwise) ops,
named-axis reductions, and named-axis expand. A *positional* shape-rewriter
(`permute`, `reshape`, `matmul`, positional `gather`) is rejected inside a
`..r` body: its output shape is not name-trackable at symbolic rank, so it
could hide an untracked transposition. For the name-tracked ops the procedural
inference arm is the real gate: it rejects a positional index at symbolic rank
and a non-existent/ambiguous/duplicate axis name, so no transposition can slip
past. This is what keeps the §4.5.1 transposition-safety guarantee intact while
admitting the deferred flexibility for the reduction and expand cases.

Enforcement: the unification arm is `crates/chelis-types/src/unify.rs`
(`unify_row_against_ground` / `unify_row_against_row`); the named-axis reduction
arm is `check_reduction_signature`, the named-axis expand arm is
`check_expand_signature`, and the discipline check is
`check_rank_body_discipline` in `crates/chelis-types/src/infer.rs`. Call-site
**rank monomorphization** (`tensor_rank_substitutions` /
`extract_rank_var_bindings` in `crates/chelis-ir/src/lower.rs`) substitutes each
spread's concrete run and resolves the named axis to a positional index at
lowering, so a rank-poly reduce **builds and runs** on the C backend. The
acceptance oracle is `crates/chelis-cli/tests/rank_poly_tier3.rs`.

#### 4.5.4 Concat Result Typing (chelis#631, chelis#594)

`concat(list, axis)` over a `List[tensor[...]]` types its result from the
joined element type (§4.5.2), the **concat axis value**, and the
**statically-visible elements**. A list's length is not part of its
type, so what survives depends on the expression: a literal `Cons`/`Nil`
chain at the concat site exposes every element, while a variable bound
to a list literal in an enclosing `let` (or top-level bind) carries only
its literal **length** to the concat site. The rules, in order:

1. **Direct per-element sum (chelis#594).** Axis is a static integer
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

History: before chelis#631 the checker typed the result from the element
type alone and wildcarded the **last** axis unconditionally, so an
axis-0 concat of two `[1, m]` rows was typed `[1, m]` — a wrong concrete
extent that the host-program C lane baked into tensor-helper signatures,
turning valid guarded (`if`/`fail`) forward programs into runtime
aborts. That misplaced-wildcard shape is also what chelis#594 reported.
Enforcement is `tensor_concat_result_type` in
`crates/chelis-types/src/infer.rs`; the acceptance oracle is
`crates/chelis-cli/tests/issue_631_guarded_forward_concat_c_parity.rs`.

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
  `int64` value (`spec/05-risc-primitives.md` [05-DIM-2]). `axis` must be
  a concrete non-negative integer literal (either a bare `int32` literal
  or a `cast(N, int32)` form; both reach the axis-bounds check). The
  result is a runtime scalar, not a symbolic dim reference.
- `expand(x, axis, size)`: insert or set a dimension at position `axis`
  with width `size`. When `axis` is a dimension *name* instead of an
  integer, the call is the named-axis expand form (§4.5.3): it inserts a
  new named axis at the trailing end, or — with a fourth `anchor`
  argument — immediately before an existing named axis.
- `reshape(x, shape_list)`: reinterpret the memory of `x` against
  `shape_list`, a `List<int64>`.

This section pins which call shapes preserve symbolic dims in the type
checker's output and which fall back to `(d-name {} *)` (see §4.5). The
canonical examples live in
[`examples/illustrative/runtime_shape_semantics.ch`](../examples/illustrative/runtime_shape_semantics.ch).

For the movement primitives, symbolic-dim pass-through is
**identity-only** (chelis#632, mirroring the IR-side rule in
`chelis_ir::dag::shape_source_for_axis`): a `stride` axis with literal
step 1 and a `pad` axis with zero padding keep the input's symbolic dim;
every other movement axis — a non-identity literal step or padding, or
any runtime bound — types a fresh `(d-name {} *)` whose extent the
owning op declares and guards at run time (spec/05-risc-primitives.md
§2.4.1). `shrink` has no checker-detectable identity form for a
symbolic axis (a full-axis slice of a symbolic dim necessarily spells
its end as a runtime value), so its symbolic axes always mint fresh
extents.

#### 4.7.1 `shape` axis form

Both forms below produce an `int64` value ([05-DIM-2]) and both pass the
infer-time axis-bounds check (`cast(N, int32)` is unwrapped by the
cast-aware extractor described in `crates/chelis-types/src/infer.rs`):

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

1. an `int64` integer literal (`Ni64`, or a `cast(N, int64)` form):
   produces an output dim of `(d-lit {} N)`.
2. a symbolic dim name in scope (a bare `var` reference such as a
   declared `[batch]` dim parameter): produces an output dim of
   `(d-name {} batch)`.
3. any other `int64` expression, including a runtime `shape(...)`
   call: the typer defers the output rank slot to whatever the
   declared signature's return-type or the surrounding call context
   imposes via standard unification.

For a positional three-argument call, a declared result tensor or the first
shape-bearing consumer fixes which of the two shapes applies: a same-rank
result replaces the extent at `axis`, while a result of rank `rank(x) + 1`
inserts the new extent at `axis`. The result remains one monomorphic value
while that choice is deferred; separate uses cannot choose different shapes
for the same binding. If a shape-neutral consumer such as `cast` requires the
tensor type before any shape-bearing context fixes it, an axis within the
input rank selects the established same-rank replacement form. `axis ==
rank(x)` has no replacement form and therefore selects trailing insertion.
The same default is materialized when no consumer in the complete program
fixes the shape. A reusable library context carries the unresolved choice to
its downstream program rather than deciding it early. An axis greater than
`rank(x)` is a type error.

**Sourceless-size rejection (source-tracking).** A runtime `size`
that is neither form (1) nor a form-(2)/form-(3) shape source has no
extent the backend can materialize, so the checker rejects it at check
time with the §4.7.2 sourceless-size diagnostic. The discriminator is
**provenance, not surface spelling**: a runtime size is accepted iff
its value provably

- **folds to a compile-time constant** — a literal, a `cast(N, _)`,
  or integer arithmetic (`add`/`sub`/`mul`/`div`/`mod`/`neg`) over
  such values — or
- **derives from an in-scope tensor's shape** — a `shape(t, axis)`
  read, or a bare `var` naming an in-scope tensor dimension (form 2)
  — followed transitively through `let` bindings, `cast` wrappers,
  and integer arithmetic.

Everything else is sourceless: a bare runtime scalar parameter (e.g.
`a_dim: int64`), a `cast`-wrapped one (`cast(a_dim, int64)`), a `let`
bound to one (`d = a_dim`), arithmetic that *touches* one
(`add(a_dim, 1)` — sourceless is absorbing), or **any other
function-call value** (`ident(a_dim)`, a user `def` — an inline
non-arithmetic / non-`shape` / non-`cast` `app`-rooted size produces a
runtime value with no shape source the check layer can see). All of
these are rejected **uniformly**, whatever the spelling. The provenance
analysis is per-scope-correct: a name that re-binds to a sourceless RHS
(`len = shape(x, 0)` then `len = k`) loses its earlier shape provenance,
and a value parameter that shadows an outer shape-sourced name (a `d:
int32` parameter shadowing an outer `d = shape(&xs, 0)`) does not
inherit it. This keeps the "a check-clean program must build" invariant:
a sourceless runtime expand size (whether a single `expand(g, 0, a_dim)`
or a chained rank-1 → rank-N broadcast such as `broadcast_to_achw`) is
rejected identically at `check`, `build`, and `eval`. Source the extent
from a tensor in scope via the form-(3) `shape(x, cast(axis, int32))`
read instead — directly or bound to a `let`. Tracked by
Chelis-Lang/chelis#397 and #469.

The check-time accept set matches what the evaluator and host runtime
materialize. The C backend's IR lowering now materializes the runtime
extent for each of these forms (Chelis-Lang/chelis#469):

- an **inline `shape(x, axis)` read** and a **`let`-bound shape read**
  (`a = shape(x, 0)` then `expand(b, 0, a)`, followed through `cast`
  wrappers AND `let`-to-`let` aliases — `c = a; expand(b, 0, c)`) bind
  the extent to the shape-source operand `x` via a `shape_dep` liveness
  edge that keeps `x`'s `Load` alive through DCE; alias recovery records
  the underlying `shape(x, axis)` app, so the extent binds to the actual
  source tensor and axis (the `let`-bound recovery is
  Chelis-Lang/chelis#369/#495, alias threading is #469 RT-3);
- a **`literal`/`cast(N, _)`** size, or **static integer arithmetic**
  over such values (`add`/`sub`/`mul`/`mod`/`neg`, followed through
  `cast` and `let`-bound static names), is const-folded to a concrete
  extent (Chelis-Lang/chelis#528). `div` is float-only (§ divergence
  below / spec/05 §2.1) and never reaches this fold as an integer size.

Two residual materializable-at-check forms are **rejected loudly at C
`build`** rather than resolved — a build-side capability limit, never a
silent miscompile:

- **integer arithmetic that combines a `shape(x, axis)` read (or a
  symbolic dim) with another term** (`mul(shape(x, 0), 2)`,
  `add(shape(x, 0), 1)`): the extent is a runtime `shape * k` / `shape
  + k` the backend's `DimExpr` has no representation for, so it rejects
  with the #469 diagnostic;
- a **top-level-`def`-bound** shape read or static value referenced as
  an expand size (`a = shape(&x, 0)` at module scope then
  `expand(b, 0, a)`): top-level binding provenance is not threaded into
  IR lowering (only `let`/block-scoped bindings are), so it rejects at
  build.

Both stay fail-closed: rejected identically enough that no `build`
emits a wrong extent. (The old spelling-based predicate allowed a
silent miscompile for the `cast`-wrapped and the function-call
sourceless spellings; closing those is what makes the uniform-rejection
claim above true at check.)

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

The above is the **typer's** behavior. IR lowering separately
recovers the concrete broadcast *extent* for codegen and autodiff:
when the `size` argument reads `shape(operand, axis)` — whether the
`shape(...)` call sits directly in the `size` slot or is bound to a
`let` name and referenced as `cast(len, int64)` (an identity cast under
[05-DIM-2]) — lowering reads the
extent from `operand`'s already-resolved axis dim rather than
defaulting to size 1. The `let`-indirection case is the canonical
`tensor_full_like` / scalar-broadcast helper idiom (`len = shape(x,
0)` then `expand(s, 0, cast(len, int64))`); without the recovery the
expand lowers to a size-1 axis, the size-1 broadcast that the typer
accepts is rejected by the no-implicit-broadcasting IR, and
`grad` fails to construct the backward DAG (Chelis-Lang/chelis#318,
#369). Lowering additionally const-folds a fully-static size (a
`literal`/`cast(N, _)`, integer arithmetic over such, or a `let`-bound
static name) to a concrete extent. A `size` that is neither a recognized
static value nor a resolvable single-tensor shape source — a bare
runtime scalar, or arithmetic that *combines* a `shape(...)` read with
another term (`mul(shape(x, 0), 2)`) — has no extent the backend can
materialize and is rejected loudly at lowering, never silently defaulted
to size 1 (§4.7.2 Form-3, Chelis-Lang/chelis#469).

#### 4.7.3 `reshape` with runtime sizes from `shape(x, ...)`

`reshape` recognizes one specific syntactic source for each element
of its shape list and propagates the corresponding input axis into
the result type. The recognized forms for a shape-list element are:

```text
shape(<reshape-input-var>, <literal-axis>)
cast(shape(<reshape-input-var>, <literal-axis>), int64)
```

The second form's outer cast is an identity cast under [05-DIM-2],
accepted so the pre-[05-DIM-2] spelling keeps its propagation. All
conditions are required:

1. The element is a `shape(...)` call, optionally wrapped in a `cast`
   whose target type is `int64` (matching the `List<int64>` element
   type that `reshape` expects).
2. The `shape` call's first argument is a bare `var` whose bound name
   is the same as the reshape's input tensor argument.
3. The `shape` call's axis argument extracts to a concrete
   non-negative integer (literal or `cast(N, int32)` form).

When all conditions hold, the typer propagates the input's dim
at the named axis into the corresponding output dim. The
`flatten_batch` and `flatten_two` patterns from
`examples/illustrative/runtime_shape_semantics.ch` are the canonical
examples:

```chelis
sig flatten_batch: &tensor[n, 4, f32] -> tensor[n, 4, f32]
def flatten_batch(x) -> tensor[n, 4, f32] = reshape(x, [cast(shape(x, cast(0, int32)), int64), cast(4, int64)])
```

A plain integer literal element (`4i64`, or `cast(4, int64)`) still
produces `(d-lit {} 4)`. Any element shape that is not literal and does
not match the conditions above falls back to `(d-name {} *)`.

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

`reshape`'s shape list is `List<int64>`: its elements are extent-domain
quantities under `spec/05-risc-primitives.md` [05-DIM-1]. The list must be
homogeneous, and its element precision must be `int64`. Those are two
separate requirements and either can fail alone.

No §5.6 position reaches a list literal, so a literal element states
`int64` itself, with a suffix or an explicit `cast`; the §5.3 `int32`
default never satisfies this slot. A non-literal element needs no
annotation when its producer is already `int64`, which [05-DIM-2] makes
true of `shape()`:

```text
reshape(x, [2i64, 2i64])                               ;; OK - explicit suffix
reshape(x, [shape(x, 0), 4i64])                        ;; OK - shape() is int64; the literal states it
reshape(x, [cast(shape(x, 0), int64), cast(4, int64)]) ;; OK - explicit cast
reshape(x, [2, 2])                                     ;; TYPE ERROR: int32 literals in a List<int64> slot
reshape(x, [2i64, 2i32])                               ;; TYPE ERROR: mixed element precision
```

The suffix requirement is deliberate, not a §5.6 gap; §5.6 records why
list literals do not adopt. The two error lines want different
diagnostics. The all-`int32` list is a slot mismatch, and its message
should name the fix (write `2i64`) rather than only the mismatch. The
mixed list is an intra-list disagreement that no defaulting rule can
resolve, and its message should name the disagreeing element rather than
render an `expected`/`got` pair, whose direction reflects unification
order rather than which element the user got wrong.

*(Not fully implemented; chelis#1112 owns the producer-side gap and
chelis#916 the diagnostic form.)*

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
  chooses a rounding rule; write `cast(floor(x), int32)` or
  `cast(round(x), int32)` to state one explicitly.
- any source -> bool target: strict {0, 1} membership; exactly 0/1
  encodes false/true, anything else traps `domain` ([04-NUM-4]).

Integer-to-float cast is total IEEE RNE and can lose integer exactness:
`cast(9007199254740993i64, f64) = 9007199254740992.0` and
`cast(16777217i32, f32) = 16777216.0`. This target-width rounding is the
float finalization rule, not an implicit source-language cast. Conversion
that discards a fractional part, wraps, or saturates requires an explicit
rounding operation or a distinct named conversion; it is never the default
float-to-integer behavior of `cast`.

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

### 5.4 Precision Compatibility Table

Operations accept same-precision operands only. The table of valid combinations:

| Operation type | Valid precisions |
|---|---|
| Arithmetic (add, mul, sub) | f32, f64, bf16, f16, int8, int16, int32, int64 (all same) |
| Float division (div) | f32, f64, bf16, f16 only (not integer; integer operands cite `spec/05-risc-primitives.md` §2.1 and point at `floor_div` / `trunc_div`) |
| Floor division (floor_div) | f32, f64, bf16, f16, int8, int16, int32, int64 (all same) |
| Truncating division (trunc_div) | int8, int16, int32, int64 only (integer-only; float operands are a type error) |
| Comparison (cmplt, eq) | any numeric (same precision) → bool |
| Logical (and, or, not) | bool only |
| Transcendental (exp, log, sin, cos, tan, atan, sqrt) | f32, f64, bf16, f16 only (not integer) |

Every deferred name of §1.1.1 - `f8e4m3`, `f8e5m2`, the `uint*` family,
`int4`/`uint4`, `complex64`/`complex128`, and `decimal128`/`decimal256` - is
reserved but not active, and is not a valid arithmetic precision in any row.
Each carries a declared arithmetic width at §1.1.1 so that activating it adds
rows here without reopening the width question.

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

Float-typed suffixes (`f32`, `f64`, `bf16`, `f16`) bind the decoded float at
that type. Canonical Surf requires the canonical float body (`42.0f32`, not
the v0.18 alias `42f32`). Integer-typed suffixes (`i8`, `i16`, `i32`, `i64`)
attach to integer literal tokens only; `1.0i8` is a parse error.

Suffix lexing rule: a suffix is part of the literal token only if it
**immediately** follows the digit sequence with no intervening whitespace,
comment, or other character. `1.0 f32` (with whitespace) is two tokens (a
float followed by an identifier) and binds at the literal default per §5.3,
which is then subject to the surrounding-position rules in the type checker.

The normal Surf parser accepts value-preserving hexadecimal/binary integer
spellings, digit separators strictly between digits, and equivalent finite
exponent spellings. Integer radix forms may carry an integer suffix; they may
not carry a float suffix. These lexical choices do not change the exact suffix
binding rule, and the canonical printer emits the decoded decimal token.
Canonical decimal float literals carry float suffixes without ambiguity
(`1.0f32`, `1000.0f32`).

Deferred suffixes:

- No suffix exists for any deferred name of §1.1.1 (`f8e4m3`, `f8e5m2`, the
  `uint*` family, `int4`/`uint4`, `complex64`/`complex128`,
  `decimal128`/`decimal256`). Each is rejected at lex time with a diagnostic
  pointing at §1.1.1. A suffix is authored only when its dtype activates.
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
   precision type literal — the literal body adopts `p`

Position 4 applies to a **bare scalar numeric literal** as well as to a
tensor-literal body (issue #308; the end-to-end parse→eval consequence is
issue #394). `cast(1.1, f64)` binds the decimal `1.1`
at `f64` — exactly `0x3ff199999999999a` — it does NOT narrow to the §5.3
`f32` default and then widen (which would yield the f32-truncation value
`1.100000023841858`). Likewise `cast(3000000000, int64)` binds the literal
at `int64`, which is what makes the §5.3 out-of-int32-range escape hatch
work. The adoption re-binds the literal at `p` and the §5.6 range checks
apply at `p`: `cast(2147483648, int32)` is still a range error. Adoption
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
**expressibility**, not convenience: without it `cast(3000000000, int64)`
cannot be written at all, and `cast(1.1, f64)` would round through the
`f32` default. What no position does is adopt a **list literal, or a
scalar against a remote callee signature** — position 2 reaches through a
signature, but only into a tensor body — and none should be added for
ergonomics alone. A structural argument — a shape list, a bounds pair, a
stride step — is program rather than payload, and under
`spec/05-risc-primitives.md` [05-DIM-1] its dtype states which KIND of
quantity it is: an extent is `int64` and an axis is `int32`, so a
context-inferred `[2, 2]` would hide exactly the distinction the dtype
exists to carry. Chelis programs are written and, more often, audited by
agents; a suffix states the kind at the site, the write-side cost is one
edit under a diagnostic that names the fix, and the read-side cost of
context-dependent literals is paid on every audit. This is the trade
`with seed(...)` already records (§7.1): its seed demands `int64` and the
literal states it (`with seed(42i64)`), with no adoption carve-out.

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
The same instinct exists ecosystem-wide, but the details differ: PyTorch's
`torch.sum` promotes ALL integral inputs to `int64`, and NumPy accumulates at
the platform default integer. Chelis deliberately widens one step instead of
jumping to `int64`; the §5.7 accumulator parameter is the authored route to a
wider accumulator when a reduction genuinely needs one. (Corrected
2026-07-30: this paragraph formerly claimed the `int32` default "matches
PyTorch's `torch.sum` accumulator-promotion rule"; PyTorch's rule is
`int64`.)

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
active dtype set. Examples include `exp`, `log`, `sin`, `sqrt`, the
transcendental row in §5.4, and `div` (float division — its integer-operand
diagnostic points at `floor_div` / `trunc_div`). Calling a float-only op on an
integer tensor is a type error reported at the call site, not deep inside the
implementation. `trunc_div` is the mirror case (integer-only): applying it to a
float tensor is a type error.

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

> **[04-FIT-1]** `typed_nodes` and `total_nodes` SHALL report the type
> inference product's checked-node counters, not a fabricated structural AST
> count. A cached or layered check SHALL preserve those counters per checked
> unit and add them across the program partition, so its fitness report is
> byte-identical to an equivalent monolithic check (chelis#858, chelis#973).
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

> **[04-EFF-1]** A `handle-effect` form SHALL name one of the two user
> handler boundaries, `random` or `resource`. Any other handler kind is a
> type error; lowering SHALL NOT erase its handler or execute the body as if
> no handler were present.

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
  handler-scoped RNG state for nested stdlib/user functions. The seed is
  semantically int64, and a seed written as an integer literal SHALL carry the
  `i64` suffix (`with seed(42i64) { ... }`, spec/02-surf-syntax.md §P10a); an
  unsuffixed literal is a type error naming the required suffix (chelis#731 Phase 1
  / §C1.5, the reject-diagnostic half chelis#771 left to this phase). The body is
  checked in the enclosing context and its type is returned, so the enclosing
  signature is enforced. Per-lane determinism of a seeded region is
  `spec/05-risc-primitives.md` [05-RNG-1]; cross-lane stream identity (eval and the
  compiled-C host lane producing the same draw sequence) is tracked at chelis#735,
  now unblocked by Phase 1's seed-literal diagnostic
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

---

## 9. Numeric Value Semantics (Decided 2026-07; Implementation Tracked As chelis#729)

**Status banner - read before citing.** The atoms below are DECIDED
normative semantics, authored 2026-07-16 out of the numeric audit
(chelis#680-#734; metas #695/#727). They are NOT yet implemented: today's
behavior diverges per the issue references in each atom's note, and the
divergences are locked as issue-linked `#[ignore]`d tests. Those tests are
visible known-failure records only: under `spec/design/spec_provenance.md`
§C3-§C4, registration, freshness, execution, debt, and waiver remain separate,
and an ignored test does not satisfy coverage. The delivery plan and the full
elaboration (finalize semantics, kernel signatures, storage) is
`spec/design/dtype_semantics.md`. Atom IDs are stable, and the current
blockquote authorities remain normative until selected for fixture-proven
migration in chelis#733 Phase 1. The pinned Buoy shell-side integration—not a
`chelis-lint` rule—attaches and checks full semantic revisions.

**Amendment 2026-07-28 - lifted contracts.** [04-NUM-9], [04-NUM-10], and
[04-NUM-11] were added on the same date, moving three decisions out of
`spec/design/dtype_semantics.md` and into this section: the closed trap-kind
set with cross-lane rendering identity (its §C2), traps-as-values-until-the-
lane-boundary including the device-lane error-flag shape (also §C2), and the
guarantee that a value survives storage and transport at its declared dtype
(the observable half of its §C3 storage decision). None of the three is a new
decision; each was already decided and each lived only in a design document,
which is a working artifact that stops being read once its phases ship. The
numbered spec is where a decision has to live to outlast the work that made
it - the precedent is [05-OBS-1..5], lifted the same way out of
`faithful_observation.md`. The design documents keep the elaboration, the
mechanism, and the evidence, and now point here for the rule.

**Amendment 2026-07-28 - arithmetic width.** [04-NUM-2] formerly closed with
"Computing a single op in f64 and rounding once is a conforming implementation
for f32/f16/bf16." That clause was REMOVED and replaced by [04-NUM-8], which
declares an arithmetic width per dtype. It is recorded here so the deletion is
not re-derived as an oversight. The clause was permissive and single-op scoped,
but `spec/design/dtype_semantics.md` cited it to mandate f64 computation for
all float ops, and the IR evaluator extended that to multi-step float
reductions and to `argmax`/`argmin` operand comparison - three levels of drift
from one sentence. [04-NUM-8] states the width positively so there is nothing
left to widen from.

**Amendment 2026-07-30 - trap occurrence, and the record carries its
cost.** A review pass on the same change set added [04-NUM-12]: trap
OCCURRENCE for multi-step operations is defined relative to each lane's
documented accumulation order, and trap-versus-exact divergence at
accumulator range edges is the one permitted cross-lane trap divergence.
[04-NUM-9] required identical RENDERING but was silent on occurrence,
which order-dependence makes a real question for int64 reductions - and
it had to be answered before chelis#729 Phase 2 freezes the trap
contract. The same pass stated the availability trade in the rationale
(trapping converts silent corruption into loud termination,
deliberately), scoped [04-NUM-8]'s native-narrow permission to the basic
operations so it no longer conflicts with the reduced-precision opt-in
rule, and re-pointed the deferred-name rejection diagnostics at §1.1.1
in the same change set (chelis#944: both lexers, the checker's
tensor-element and cast-target arms, and their locking tests - the
`uint*` messages formerly cited §1.1.2's superseded "out of scope"
stance, and the newly reserved names fell to the generic unknown-name
rejections with no citation).

> **[04-NUM-1]** Every numeric op result SHALL be finalized into its
> declared dtype - rounding for floats, width and domain checks for
> integers and bool - before it becomes observable to any subsequent op,
> comparison, fold, or output, in every lane and on every surface
> (scalar and tensor alike).

*(Not honored today: chelis#689, #693, #699, #714.)*

> **[04-NUM-2]** Float finalization SHALL be IEEE-754 round-to-nearest,
> ties-to-even, at the dtype's own STORAGE width (f64 identity; f32
> 24-bit, f16 11-bit including subnormals, bf16 8-bit mantissa), with
> overflow to the correctly signed infinity, and NaN, signed zero, and
> infinities preserved. The width at which the op is COMPUTED before
> finalization is fixed by [04-NUM-8], not by this atom.

*(Not honored today: chelis#714.)*

> **[04-NUM-3]** Integer op results that are not exactly representable
> in the declared width SHALL trap with the branded overflow diagnostic;
> no lane and no surface SHALL wrap (except via the named modular
> operations of [04-NUM-7]), saturate, or silently widen.
> In-range integer arithmetic SHALL be exact at every width.

*(Not honored today: chelis#689.)*

> **[04-NUM-4]** A `bool` value SHALL be exactly 0 or 1; arithmetic
> that would produce any other value in a bool-typed position SHALL be
> rejected by the checker or trap.

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

*(Honored and locked: `precision_matrix.rs` ByDesign rows.)*

> **[04-NUM-7]** A named modular-arithmetic operation (initially
> `wrap_add`, `wrap_sub`, `wrap_mul`; the roster is owned by the
> capability table) on an integer dtype SHALL produce the unique value
> in that dtype's range congruent to the exact mathematical result
> modulo 2^width, and SHALL NOT trap. These operations are the
> explicit, user-visible escape hatch for modular arithmetic (hashing,
> RNGs, checksums); the wrap prohibited by [04-NUM-3] is the *implicit*
> overflow behavior of the ordinary arithmetic ops, not these named
> ops, whose result is in-range by construction. Named modular
> operations are defined ONLY on the integer dtypes (`int8`, `int16`,
> `int32`, `int64`); they SHALL NOT be defined on `bool` or any float
> dtype (floats overflow to infinity per [04-NUM-2] and have no modular
> escape hatch by construction), and requesting one on a non-integer
> dtype is a checker-level type error.

*(Not expressible today: no `wrap_*` builtins exist; the eval RNG's
splitmix hash (`dropout_sample`, `chelis-ir/src/eval.rs`) is the
in-tree witness of the need. Tracked by chelis#753.)*

> **[04-NUM-8]** Every dtype declares an ARITHMETIC WIDTH in addition to
> its storage width. Every op SHALL be performed at its operands'
> arithmetic width and finalized to the storage width once per op, in
> every lane and on every surface. No lane SHALL compute at any other
> width. The arithmetic widths are:
>
> | dtype | storage width | arithmetic width |
> |---|---|---|
> | `f64` | 64 | f64 |
> | `f32` | 32 | f32 |
> | `f16` | 16 | f32 |
> | `bf16` | 16 | f32 |
> | `int64` | 64 | exact int64 |
> | `int32` | 32 | exact int32 |
> | `int16` | 16 | exact int16 |
> | `int8` | 8 | exact int8 |
> | `bool` | 8 | not an arithmetic dtype ([04-NUM-4]) |
>
> The reduction and matmul accumulator parameter of §5.7 is the ONLY
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
> selected by a backend, a build flag, or a global mode. No such opt-in is
> authored yet; until one is, narrower-than-declared computation is
> non-conforming in every lane.
>
> Arithmetic a backend SYNTHESIZES that is not an op on program values -
> the loop counters and addressing expressions of generated code - carries
> no declared dtype and is outside the table above. It MAY use a machine
> width whose range provably contains every value it carries, and it SHALL
> NOT be the carrier of a declared-dtype value crossing a boundary
> ([04-NUM-11] governs those crossings).

*(Not honored today: the IR evaluator computes every float op and every
float reduction in f64 because its tensor store itself is f64-backed -
`TensorValue { data: Vec<f64> }` in `chelis-ir/src/eval.rs` - so
`binary_map`/`unary_map`'s closures and the `reduce` fold (`init: f64`)
run at f64 whatever the dtype, and `reduce_argcmp` compares
`argmax`/`argmin` operands as f64, returning the wrong index for
adjacent int64 values above 2^53. The storage-width column's `bool` row
is also ahead of the shipped ABI: the C runtime still stores bool at 4
bytes through the f32 encoding (`RuntimeDType::byte_width`), and the
native byte lands with the §C3 storage decision at 0.19 (chelis#892,
chelis#894). Tracked by chelis#729 Phase 2's kernel split and Phase 1's
storage decision.)*

**Rationale for the f16/bf16 rows.** These are not exclusions carved out
of a general rule; the arithmetic width is part of what the format is.
No shipped hardware provides a bf16 arithmetic instruction: AVX512-BF16's
`vdpbf16ps` and ARM's BFDOT/BFMMLA both accumulate into f32, and there is
no `vaddbf16` on any target. f16 arithmetic does exist (ARMv8.2-A,
AVX512-FP16, NVIDIA `__hadd`), which is why the atom permits it, but even
there the transcendental path converts to f32 because the special-function
units are f32. The same property holds one rung down and is why §1.1.1
defers `f8e4m3`: FP8 is a matmul-input format whose tensor-core ops
accumulate in f32 and which requires an out-of-band scale factor, so it
has no self-contained arithmetic width to declare yet.

**Why f32 is not widened.** f32 has native arithmetic on every CPU and GPU
in the supported set, so computing it at f64 invents a width the format
does not have. For the basic operations that substitution is bit-identical
and therefore only forfeits the compute (half the SIMD lanes, two
conversions per op, no tensor-core path); for transcendentals and for
multi-step reductions it also changes the answer, which is what would
otherwise force a cross-lane tolerance table between two lanes that
should agree exactly. The same argument applies to routing exact integer
arithmetic through f64, which additionally destroys int64 exactness above
2^53 (chelis#684, chelis#680).

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
> integer result outside the declared dtype's range), `Domain` (a value
> outside the declared dtype's set - a fractional or non-finite value at an
> integer dtype, a value other than 0 or 1 at `bool`), and `DivZero`
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
> user-facing numeric-trap line.

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

*(The device-lane shape is evidence-backed and provisional pending the HIP
half, chelis#736. The Metal spike (chelis#737) measured flag detection as
effectively free for the memory-bound elementwise shape the backend emits -
worst case +0.7% median, int64 multiply included - and verified MSL int64
bit-exactness. Implementation rider from the same spike, recorded here
because it is a correctness constraint rather than a preference: the clang
overflow builtins are BANNED in emitted MSL. `__builtin_mul_overflow(long)`
crashes the backend compiler reproducibly, and at-scale vectorization
miscompiles the add and sub forms into false positives; the division-based
form crashes the backend as well. The hand-written checks - widening for
int32, sign-bit XOR for add and sub, `mulhi` for int64 multiply - are the
implementation.)*

**The availability trade, stated.** Trapping converts silent data
corruption into loud termination, by design: a long-running job that
overflows an `int64` counter DIES where wrapping arithmetic would have
carried a silently wrong value to completion. That operational cost is
deliberate, and this record carries it alongside the benefit: the
alternative outcome is not a successful run but a plausible wrong result,
which the 2026-07 audit measured as the strictly worse failure mode
(chelis#703). Code that WANTS mod-2^width semantics states it with
[04-NUM-7]'s named `wrap_*` operations; saturation - the third behavior,
which image pipelines want - is authored as named `sat_*` ops if `uint8`
activates (§1.1.1's non-reservation note). Behaviors are named ops, never
modes.

> **[04-NUM-11]** A value SHALL survive storage, transport, and every
> boundary crossing at its declared dtype without collapse. An `int64`
> value above 2^53 that is exact when produced SHALL still be exact after
> being stored in a tensor, serialized onto the execution wire, returned
> through a language binding, and read back. A representation that cannot
> carry a dtype's full value set is not a conforming representation for
> that dtype, and no stage SHALL substitute a wider or narrower one to
> compensate.

*(Not honored before chelis#729 Phase 1: tensor storage, the wire schema,
and the Python binding each flattened numeric payloads to f64, so an int64
above 2^53 collapsed at every boundary no matter how exactly it had been
computed - chelis#684, chelis#685, chelis#686. This atom is the
user-visible statement of what `spec/design/dtype_semantics.md` §C3's
storage decision delivers: that document owns the mechanism, this atom owns
the guarantee. Also not honored on the Metal host/device bool boundary,
where a value crosses at a quarter of its storage width: chelis#892.)*

> **[04-NUM-12]** A numeric trap's OCCURRENCE is deterministic within a
> lane and is defined by that lane's documented evaluation order. For a
> multi-step operation (a reduction or scan), whether an intermediate
> result leaves the accumulator dtype's range - and therefore whether
> the operation traps - is evaluated against the lane's documented
> accumulation order: the host lanes' stride-4 cascade
> (`spec/05-risc-primitives.md` §2.3) or a device lane's documented tree
> reduction. Same program, same inputs, same lane SHALL always produce
> the same trap or the same completion. Where two lanes document
> different accumulation orders, an integer accumulation whose
> intermediate sums approach the accumulator's range MAY trap in one
> lane and complete in another; a lane that completes SHALL produce the
> exact result. Trap-versus-exact at accumulator range edges is the ONLY
> permitted cross-lane divergence in trap behavior: trap versus a wrong
> value is never permitted, the tolerance table never applies to traps,
> and when two lanes both trap, [04-NUM-9]'s rendering identity applies
> in full.

*(Authored 2026-07-30, before chelis#729 Phase 2 freezes the trap
strings, because [04-NUM-9]'s identity requirement was silent on exactly
this interaction: trapping addition is associative in VALUE but not in
trap occurrence, so pinned-order host lanes and tree-order device lanes
can legitimately differ at range edges. The alternative - pinning one
accumulation order across every lane - was considered and rejected: §2.3
already documents per-lane order divergence for float values, and
serializing device reductions to the host cascade would forfeit the
parallel reduction for a case reachable only at the extreme edge of the
range. Practical reach: §5.7.1's widened default accumulators make
occurrence order-independent for the narrow integer dtypes whenever
n·max|element| fits the accumulator - no ordering of partial sums each
bounded by n·max|element| can overflow int32 before roughly 2^24 int8
elements - so the divergence window is real only for int64 (and
extreme-length int32) accumulations near the range edge. No lane traps
integer overflow today ([04-NUM-3]'s divergence note), so this atom
rides chelis#729 Phase 2 with the trap contract itself; the chelis#687
and chelis#754 cross-lane oracles treat a trap-versus-complete
divergence as conforming only under this atom's conditions.)*
 > **[04-NUM-13]** `shl` and `shr` on a signed integer dtype SHALL operate
> on that dtype's fixed-width two's-complement bit pattern. `shl` discards
> bits beyond the declared width and `shr` is arithmetic (sign-extending).
> A non-negative count at least the declared width produces zero for
> `shl` and for `shr` of a non-negative value, and `-1` for `shr` of a
> negative value. A negative count SHALL trap with
> `shift amount must be non-negative, got N`. These semantics are
> independent of host-language signed-shift behavior; a compiled backend
> SHALL NOT invoke undefined or implementation-defined signed shifts.

 *(Implemented and UBSan-locked in eval and the C host lane for chelis#682.)*

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
> [04-NUM-4].

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

*(Authored 2026-08-04. [04-NUM-12] defined occurrence by "that lane's
documented evaluation order", but no lane documented an order for an
elementwise trapping map, so the multi-offender case was unauthored while
both [04-NUM-9]'s cross-lane rendering identity and [04-NUM-12]'s
within-lane determinism formally applied to it. Index order is the choice
consistent with the two existing precedents: [04-NUM-10] already requires a
device lane to carry "the first failing element index", and
`spec/05-risc-primitives.md`'s `Scatter` deterministic-order rule already
uses updates-tensor row-major flat order. The alternative - leaving the
offender unspecified while guaranteeing the trap - was rejected because the
kind is part of the rendered line, so an unspecified offender would have
required weakening [04-NUM-9]'s byte-identity requirement to accommodate an
implementation. Stated for elementwise trapping maps generally rather than
for `cast`, so a later trapping map does not reopen the same question. Not
fully honored on the checked `cast` today: chelis#1152.)*

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
| `f64` | IEEE binary64 | f64 | identity | n/a (IEEE handles it) | NaN, ±inf, -0.0 preserved |
| `f32` | IEEE binary32 | f32 | RNE to 24-bit mantissa | rounds to ±inf per IEEE | NaN preserved (quiet), ±inf, -0.0 preserved |
| `f16` | IEEE binary16 | f32 | RNE to 11-bit mantissa, incl. subnormals | overflow -> ±inf (`mul(65504f16, 2f16) = inf`) | as f32 |
| `bf16` | bfloat16 | f32 | RNE to 8-bit mantissa | overflow -> ±inf | as f32 |
| `int64` | integers in [-2^63, 2^63-1] | exact int64 | must be integral and in range, else trap | trap: `Overflow` out of range, `Domain` non-integral ([04-NUM-9]) | none |
| `int32` / `int16` / `int8` | integers at width | exact at width | same rule at width | trap: `Overflow` / `Domain` at width | none |
| `bool` | {0, 1} | n/a (not an arithmetic dtype) | must be exactly 0 or 1, else trap | trap, kind `Domain` | none |
| deferred names (§1.1.1) | rejected by the checker | - | unreachable: rejection is compile-time-visible, never a runtime arm | - | - |

Reading notes:

- The **arithmetic width** column is [04-NUM-8]'s table restated per row. It
  is deliberately NOT reproduced into §1.1.3's per-backend matrix: it is a
  target-independent fact and §1.1.3 records per-target implementation status.
- The **trap cells** name [04-NUM-9]'s kinds; for multi-step operations,
  trap OCCURRENCE is governed by [04-NUM-12] (each lane's documented
  accumulation order).
- The **f64 row's identity finalize** is why [04-NUM-6] holds:
  `f64 add(2^53, 1) == 2^53` is the correctly rounded answer and stays. The
  same two numbers at `int64` are exact or trap, never silently collapsed -
  same inputs, opposite verdicts, by design.
- The **bool row has no arithmetic width** because arithmetic on `bool` is
  rejected rather than performed ([04-NUM-4]). `and` / `or` / `not` are the
  logical operations; counting is the explicit-cast idiom.
- The **deferred row** covers every name reserved in §1.1.1. Each has its
  arithmetic width declared at reservation, so activating one adds a row here
  without reopening the width question; `f8e4m3` additionally needs its
  finalize cell authored on activation, since OCP E4M3 has no infinities and
  is therefore not a parameterization of [04-NUM-2].

## 10. Checker Totality (Decided 2026-07; Ratified At chelis#731 Phase 1)

**Status banner:** same transitional-blockquote and honesty rules as §9. The
delivery plan is `spec/design/checker_totality.md`. chelis#731 Phase 1
ratifies this section's semantics; the atom IDs remain the citation grammar and
the blockquotes remain normative until chelis#733 Phase 1 migrates them through
the pinned Buoy shell-side integration and attaches full revisions.

> **[04-TOT-1]** Every Deep tag in the closed vocabulary
> (spec/03-deep-syntax.md) SHALL have an explicit checker disposition: a
> real inference case, or a rejection with a pushed diagnostic. A
> construct the checker does not recognize SHALL produce a diagnostic,
> never a silent exemption of its subtree.

*(Honored at the construct level as of chelis#731 Phase 1: the
`handle-effect` case landed - the seed/device handler is checked per
effect kind and the body's type is returned so the enclosing signature is
enforced (chelis#709) - and `infer_expr`'s unknown-tag wildcard now pushes
`UnknownForm` instead of a silent `Type::Error`.)*

*(Compile-time exhaustiveness landed at chelis#731 Phase 3 and was
strengthened to decode-once at the 2026-07-24 rework: the closed
vocabulary is the `chelis_deep::DeepTag` enum, the parser stamps it as
`Atom::Tag` so the tag string does not exist in the parsed tree, and
`infer_expr`, `lower_expr`'s tag dispatch, the `.dp` structural
validators, and every migrated consumer match it exhaustively with no
wildcard arm, so a new tag fails the build at every chokepoint that has
not chosen a disposition. In-vocabulary tags with no expression-position
inference case are an explicit loud `UnknownForm` disposition naming the
tag (`block` has a real sequencing case per chelis#859), and the
unknown-string wildcard survives only at raw-string boundaries - input
that never crossed the parser's vocabulary screen. A top-level list the
checker cannot decode is a loud diagnostic, never a silent skip
(chelis#858). Delivery record: `spec/design/checker_totality.md`
Phase 3.)*

> **[04-TOT-2]** If a check completes with an empty error vector, the
> typed result SHALL contain no error-typed expression: `Type::Error`
> without a corresponding reported diagnostic SHALL be unconstructible.

*(Structurally honored as of chelis#731 Phase 2: `Type::Error` carries a
private `ErrorWitness`. A fresh witness is minted only by
`chelis_types::errors::report`, which appends the owning diagnostic through
the checker's private `DiagnosticSink`; `propagate` copies an existing witness
for cascade suppression. The sink is append-only, and the shared
`session::run_result` boundary vetoes `Ok` whenever that authoritative vector
is non-empty. A planted bare `Type::Error` fails to compile outside the
diagnostics module.)*

*(Fresh inference results converge on `finalize_checked_program`, which
validates the annotated runtime tree's required function, pattern, and
expression stamps, with incoming and inferred signature metadata as
backstops. Annotation consumes canonical stamps from the owning
`InferenceProduct` epoch; it does not semantically re-infer a node under a
fresh substitution. The exhaustive child-role classification includes an
effects-owned `EffectHandler` payload and a type-owned handled body. Public raw
`from_parts`/`try_from_parts` reconstruction does not exist. The effects-owned
tree transformation is fallible
`CheckedProgram::try_with_effect_annotations`: it requires identical roots,
spans, atoms, children, and all metadata except the effects-owned `effects`
entry; preserves the checked type environment, signature inference, and
linearity; and reruns totality validation. The
effects pass maps a violation to one `TypeTotality` error. Other operations
have disjoint ownership: `CheckedProgram::with_linearity` changes only
linearity metadata, and `CheckedProgram::compose` combines two
already-successful, context-stacked checked halves. Neither rewrites
type-owned tree structure.)*

*(The Surf-reachable chelis#755 field-access and chelis#756 deep-type sites are
closed here. Deep type and dimension resolution has one centralized, located,
witnessed boundary: `DeepTypeResolver` returns a resolved type or an
`ErrorWitness` minted while appending the exact owning diagnostic. Recursive
parents propagate without re-reporting. Binder visibility is an explicit,
serde-skipped `TypeResolutionScope` field on the cloned lexical `Env`, never
ambient process/thread state. Binder modes distinguish closed input, explicit
`deftype`/`typealias` binders, implicit-generic `defsig` binders, and trusted
compiler metadata. Nominal headers and arities are precollected for the full
check unit, preserving legal self/forward references while preventing unknown
names, wrong arities, malformed nested nodes, and bare declaration-field types
from entering a successful/cacheable context. Bare and canonical cast targets
use the same boundary and exact-arity checks. Recursive functions are
prebound only within genuine SCCs, inferred/generalized as a unit, with no
post-report diagnostic deletion. The Phase 0 invariant mirror, fitness-honesty
corpus, cascade-count corpus, and checker-totality Phase 2 oracle in
`spec/design/checker_totality.md` enforce these claims. See spec/03
§2.5.1/§2.6.)*

> **[04-TOT-3]** A structurally malformed Deep form that reaches the
> checker SHALL be rejected with a diagnostic naming the tag and the
> expected shape; deferring the failure to a later stage is not a
> disposition.

*(Honored as of chelis#731 Phase 1: the malformed arity/shape guards push
`MalformedForm` - the two Phase 0 holes (`(def)` with no body, `(cast)`
with no target) plus the census-extension family (jit/realize/copy/borrow/
var/lit/match/pipe/tuple-get/record/access/record-update/grad/vmap).)*
