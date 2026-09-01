# Runtime Representation and Tensor-Access Safety

**Status:** ACTIVE. Phase 0 is implemented and continuously enforced by its
authoritative oracle; Phases 1–5 remain planned. Tracking issue: [#893]. Code
evidence was rechecked on `main` at `8190b6d8` unless a later receipt is named.
**Owning specs:** `spec/04-type-system.md` [04-NUM-4], [04-NUM-8],
[04-NUM-10], [04-NUM-11], and [04-SHAPE-1], plus
`spec/05-risc-primitives.md` [05-DIM-1], [05-DIM-2], [05-OP-31], and
[05-OP-33]. Those atoms decide language and ABI behavior. This document owns
only the implementation structure and delivery order. Where they disagree, the
numbered specs win and this document has a bug.
**Class fixed:** a tensor's dtype, stored representation, shape, capacity, and
lane element type can be stated independently, so an unchecked cast or stale
mirror can make one buffer mean two incompatible things. The end state makes an
in-repository mismatch unavailable through ordinary APIs and makes a malformed
foreign carrier fail before data access.

## Summary

Chelis already has most of the right facts. `RuntimeDType::repr()` names the
stored representation and `Repr::byte_width()` derives its width. The public C
carrier is exact and tagged. Host allocation validates rank, extents, element
count, byte count, and views. HIP sizing now delegates to the same width source.

Those facts are still optional at their highest-risk consumers:

- `chelis_tensor.data` remains a public `*mut u8`, and the public unsafe
  `TensorElement` trait returns pointers after a runtime tag check that a caller
  can bypass;
- equal-width representations remain interchangeable to a cast even though
  [04-NUM-8] says their bits have different meanings;
- `DimExpr::normalized_key` uses saturating products, so two unequal symbolic
  capacities can compare equal before allocation;
- Python and the HIP support header independently mirror a fixed-rank,
  int32-sized device tensor whose element pointer is `float *`; and
- backend element spellings and stores are text selected at call sites, so a
  width repair can leave a kernel writing the old representation.

This plan removes those choices. A closed representation registration supplies
the storage marker, arithmetic representation, width, and lane identities. A
validated tensor view is the only in-repository route from a raw carrier to
elements. Exact capacity identities never saturate. Host and device descriptors
are generated from one schema. Kernel loads and stores are typed separately at
their source and destination representations. The final seal is module privacy,
backed by compile-fail and source-completeness tests, not a helper that callers
are expected to remember.

The public C layout in [05-OP-31] remains source- and ABI-compatible. C can always
forge bytes, so "unrepresentable" has a precise boundary here:

1. repository-owned Rust, generated C, HIP, Metal, and Python code cannot obtain
   an element view without a matching representation proof; and
2. an arbitrary foreign `chelis_tensor` is treated as untrusted: every
   observable descriptor invariant is checked before access, while physical
   allocation size, lifetime, overlap, and cross-call synchronization remain
   the foreign caller preconditions stated by [05-OP-31]. No Rust reference or
   slice is formed from that carrier.

## Why a structural plan and not another patch

The class has already survived local repairs:

1. Typed accessors existed while direct field casts decoded native int32 storage
   as f32. A correct optional accessor did not stop the wrong path.
2. [#1289] replaced the public numeric C boundary with exact tagged carriers and
   [#1347] fixed zero-extent accounting, but an independent Python/HIP device
   mirror still carries rank-eight int32 metadata ([#1345]).
3. [#1360] made HIP allocation and element spelling agree for Bool8 and made the
   unsupported kernels fail loudly. The required Bool8 operation family remains
   [#1364]; a width assertion alone cannot prove a kernel writes canonical 0/1.
4. Host runtime allocation now uses checked metadata, while compiler capacity
   equivalence still saturates before the runtime sees a value ([#888]).
5. `Repr` derives byte width but not [04-NUM-8]'s arithmetic representation
   ([#899]). The missing fact is restated or inferred wherever a lane needs it.

Each repair made one path correct. None removed the ability to construct the
next inconsistent path. This plan lands its irreversible steps last: first make
every consumer use the typed replacement, then remove the raw route and require
the tree to compile.

## Non-goals

- No public C layout change. `chelis_tensor`, `chelis_scalar`, the dtype tags,
  and the callables governed by [05-OP-31]/[05-OP-33] retain their exact fields,
  widths, order, and meanings.
- No new dtype, arithmetic rule, implicit cast, compatibility fallback, or
  versionless carrier. [04-NUM-8] remains the sole semantic table.
- No transfer of [#892]. Metal Bool8 storage remains owned by [#729]; this plan
  consumes its representation and probes it but does not change its parent or
  closure condition.
- No claim that a hostile C caller can be made statically type-safe. Its
  contract is validation before access and a typed failure, not Rust privacy.
- No general GPU-runtime rewrite. Device descriptor and element access change
  only as required to remove fixed-rank, narrow, or untyped representation
  seams.
- No new numeric-surface exception. Every generated or changed public callable
  still needs its exact [05-OP-N] authority and capacity registration.

## Vocabulary

- **Representation** — the exact stored bit interpretation named by `Repr`, not
  merely its byte width.
- **Arithmetic representation** — the width/kind in which [04-NUM-8] performs
  an operation before finalization. Bool has none.
- **Element marker** — a sealed Rust type whose size and bit interpretation are
  one `Repr`, such as `Bool8`, `F16Bits`, or `Bf16Bits`.
- **Raw carrier** — a `#[repr(C)]` descriptor or foreign pointer whose fields
  have not yet been validated together.
- **Validated metadata** — rank, shape, strides, count, capacity, ownership,
  dtype, and representation proven mutually consistent.
- **Typed view** — a `TensorRef<T>` or `TensorMut<T>` constructed only from
  validated metadata whose dtype is `T::DTYPE`.
- **Lane binding** — the closed association among an element marker, exact
  `Repr`, backend spelling, load form, store form, and probe codec.
- **Receipt** — already-landed behavior that the final oracle preserves. A
  receipt is not permission to close an issue before the structural path lands.

---

# Part I — implementation contracts

## C1. One closed representation registration

`chelis-vocab` remains the dependency-bottom owner of representation identity.
It gains a closed arithmetic representation enum and one exhaustive contract
projection:

```rust
pub enum ArithmeticRepr {
    Ieee754Binary32,
    Ieee754Binary64,
    ExactTwosComplement8,
    ExactTwosComplement16,
    ExactTwosComplement32,
    ExactTwosComplement64,
}

pub struct DTypeContract {
    dtype: RuntimeDType,
    repr: Repr,
    arithmetic: Option<ArithmeticRepr>,
}

impl RuntimeDType {
    pub const fn contract(self) -> DTypeContract;
}

impl DTypeContract {
    pub const fn dtype(self) -> RuntimeDType;
    pub const fn repr(self) -> Repr;
    pub const fn arithmetic(self) -> Option<ArithmeticRepr>;
    pub const fn byte_width(self) -> usize;
}
```

`RuntimeDType::repr()` and `byte_width()` become projections of `contract()`;
they do not retain separate matches. `DTypeContract::byte_width()` delegates to
`Repr::byte_width()`. `bool` has `arithmetic: None`; every other active runtime
dtype has the exact row from [04-NUM-8]. There is no numeric or stringly width
field that a consumer can supply.

The runtime owns a sealed element trait. Only its storage module can
implement it:

```rust
pub trait TensorElement: private::Sealed + Copy {
    const DTYPE: RuntimeDType;
    const REPR: Repr;
    type Arithmetic;
}
```

The implementations are `f64`, `f32`, `F16Bits`, `Bf16Bits`, `i64`, `i32`,
`i16`, `i8`, and `Bool8`. `bool` and bare `u8`/`u16`/`u32` do not implement the
trait. Narrow-float and Bool8 markers are `#[repr(transparent)]`; every bit
pattern is valid for the float storage markers, while Bool8 construction checks
for exactly 0 or 1. A compile-time assertion binds `size_of::<T>()` to
`T::REPR.byte_width()` for every implementation.

The following are forbidden when C1 closes in Phase 4. Phase 1 freezes the
vocabulary and migrates runtime/ABI consumers; its exact debt manifest keeps
the remaining backend duplicates visible until Phase 4 removes them:

- a second width or arithmetic-width match over `RuntimeDType`/`Prim`;
- a `TensorElement` implementation outside the owner module;
- treating `size_of::<T>()` as representation identity;
- using `u16` as an unlabelled f16/bf16 element or `u8` as bool; and
- choosing a lane type from a byte count.

Adding a `RuntimeDType`, `Repr`, arithmetic representation, or element marker
must break an exhaustive compile target until the complete registration exists.
That mutation is a Phase 1 acceptance test, not a review convention.

## C2. Exact capacity identity and checked finite counts

Compiler equality and runtime allocation are related but distinct domains.
Neither may use a saturating or wrapping integer.

### C2.1 Compiler capacity keys

`CapacityKey` is a canonical tree over the complete multiplication/division
vocabulary:

```rust
pub enum CapacityKey {
    Literal(BigUint),
    Symbol(DimSymbol),
    Product(Vec<CapacityKey>),
    ExactQuotient {
        dividend: Box<CapacityKey>,
        divisor: Box<CapacityKey>,
    },
}
```

`CapacityKey` is deliberately carrier-independent: it is built from the
complete semantically typed extent expression, not from the IR enum that
happens to carry that expression. `DimExpr`, `RtDim`, and `InputAxis` are
carrier spellings, not key variants. Phase 1 consumes whichever carrier
[#1277] has landed; it must not translate an `RtDim`/`InputAxis` edge back into
`DimExpr` or recover it by name.

Product-only regions flatten, sort, fold arbitrary-precision literals, and
remove multiplicative identities. They may collapse a zero product only when
every factor whose key would be discarded is statically total or its validity
predicates have been discharged. Thus `0 * symbol` may become zero, but
`0 * (1 / n)` retains the partial quotient and remains distinct from zero
unless `n != 0` and `n` divides 1 have both been proved.
Division is a partial exact-integer operation, not rational arithmetic, so an
`ExactQuotient` remains an ordered tree and is a barrier to product flattening.
The constructor rejects a statically zero divisor and folds a literal quotient
only when the divisor is nonzero and divides evenly. It does not cancel a
symbolic factor, reassociate multiplication across a quotient, collapse
`0 / symbol`, or reduce `symbol / symbol` without a separately checked proof
that preserves the original nonzero and divisibility domain. In particular,
`(a * 3) / 2` and `a * (3 / 2)`, `n / n` and `1`, and `0 / n` and `0` have
distinct keys absent such a proof. This deliberately retires the current
rational-equivalence tests: they preserve the defect, not a language rule.

An implementation may attach a closed `CapacityPredicate` proof set
(`NonZero` and `DividesEvenly`) to a key and perform a stronger reduction only
after those predicates have been discharged for the exact typed expression at
the reuse site. Unproved predicates return `NotProvenEqual`; they are never
assumed from a successful value seen on another execution. The key and its
proof arithmetic are never projected to `u64`, `i64`, or `usize`. Equality is
therefore exact over both value and validity domain.

The set is deliberately closed against equality learned only by passing a
[#1277] runtime guard. A passed guard establishes a fact for that execution
but supplies no proof object with identity, dominance, and lifetime at the
reuse site. Such
equality therefore returns `NotProvenEqual`. A later optimization may add a
proof-carrying predicate only if it structurally carries guard identity,
dominance, and scope; observing equal runtime values is insufficient.

The red controls include zero multiplied by a partial quotient, nested partial
quotients under otherwise total products, and both operand orders. Evaluation
must visit the same validity obligations as the original typed expression; a
zero value does not short-circuit an invalid divisor in the key.

A later dimension-expression operation outside this closed vocabulary must add
an exact key rule or return `NotProvenEqual`; it may not add a lossy fallback.
Memory planners may reuse storage only when both the exact capacity key and the
exact `Repr` agree. A failure to prove equality loses reuse, never correctness.

### C2.2 Runtime metadata types

All host and device allocation/view paths consume privately constructed values:

```rust
pub struct ShapeMetadata { /* rank, extents, strides, count */ }
pub struct ElementCount(i64);
pub struct ByteCount(i64);
pub struct AllocationBytes(usize);
```

`ShapeMetadata` checks rank/domain agreement, non-negative extents, and checked
product. Owned contiguous allocation derives checked row-major strides; a view
retains supplied strides only after proving its reachable maximum byte offset is
within `byte_capacity`. Rank zero has one element. Any zero extent has zero
elements. `ByteCount` is checked multiplication of an `ElementCount` and a
`Repr` width. `AllocationBytes` is a checked target-sized projection performed
only at allocation/copy submission.

No consumer recomputes a product from raw shape fields. Allocation, views,
copies, fills, reshape, DLPack export, host/device transfer, and memory planning
accept these types or a typed tensor view that already contains them. The
existing checked host allocation is the starting receipt for [#889], not a
parallel helper that may remain optional.

Failures occur before allocation or access and retain the owning operation's
typed `Domain`/`Overflow` behavior. Release and debug builds execute the same
checked path.

## C3. One generated host/device descriptor schema

A new leaf crate, `chelis-abi`, depends only on `chelis-vocab` outside the
standard library. It owns a declarative field schema and renderers; it does not
allocate, free, or interpret tensor values. The schema generates:

1. the Rust raw host descriptor used by the runtime owner module;
2. the existing public C `chelis_tensor` declaration, with [05-OP-31]'s exact
   layout unchanged;
3. a Rust raw device descriptor used by Python's private device-entry module;
   and
4. the corresponding internal C/HIP device declaration.

Delivery ownership is intentionally asymmetric. Phase 3 creates the shared
schema and installs the generated host and public-C artifacts. That host slice
depends only on Phase 1 and defines every shared field class needed by a later
device renderer. Phase 2 requires Phase 3 and consumes that landed schema for
the generated device descriptor and the Python/device/DLPack migration. It
does not stage or install a private host descriptor ahead of the Phase 3 host
consumer migration.

The host and device descriptors share these schema field classes: opaque data
pointer, dynamic shape/stride pointers, `int64` element count and byte capacity,
`int32` rank, exact dtype tag, ownership, and reserved bytes. Device ownership
and address-space behavior remain distinct, so the two descriptors may be
different named types; their numeric metadata cannot diverge.

The schema macro/renderers expand inside each owning private module. Generated
Rust fields are private to that module; ordinary consumers receive opaque
handles or typed views. The checked-in C fragments are generated artifacts with
byte-for-byte freshness tests. Every published header remains reachable from
`chelis_runtime.h`; the public-header census sees the same canonical declarations
in every preprocessing context.

Python deletes `CHELIS_MAX_DIM`, `[i32; 8]`, int32 `size`/`storage_size`, and
`*mut f32` from its device carrier. The HIP support header deletes its matching
fixed arrays, `int` products, scalar special case that rewrites zero count to
one, and `float *` view parameter. Rank, shape, count, and capacity cross both
boundaries at the numbered-spec domains.

This consolidation creates no new public numeric channel. If implementation
requires a new public callable or field, that is a scope change: author its
[05-OP-N] rule and capacity registration before modifying this plan.

## C4. Validated typed tensor access

The runtime moves the raw descriptor into a `tensor_storage` module. Its fields,
including `data`, are private to that module even within `chelis-runtime`.
`#[repr(C)]` fixes layout and does not require Rust field visibility.

There are exactly three ingress classes:

1. an owned allocation built from `ShapeMetadata` and `DTypeContract`;
2. an in-repository borrowed view built from validated metadata and an explicit
   ownership disposition; and
3. a foreign carrier validated field-by-field before use.

Successful repository-owned or repository-borrowed ingress yields an
exhaustive erased view:

```rust
pub enum AnyTensorRef<'a> {
    F64(TensorRef<'a, f64>),
    F32(TensorRef<'a, f32>),
    F16(TensorRef<'a, F16Bits>),
    Bf16(TensorRef<'a, Bf16Bits>),
    I64(TensorRef<'a, i64>),
    I32(TensorRef<'a, i32>),
    I16(TensorRef<'a, i16>),
    I8(TensorRef<'a, i8>),
    Bool(TensorRef<'a, Bool8>),
}
```

The mutable form is separate and can expose a Rust slice only when its
repository-owned allocation or in-repository borrow proves lifetime, range,
and exclusivity. It cannot coexist with another live mutable or immutable view
of that range. `TensorRef<T>` and `TensorMut<T>` expose slices, indexed
reads/writes, shape metadata, and narrow exact-bit operations. They do not
expose an arbitrary typed pointer cast. Constructors are private; the single
dtype dispatch in `tensor_storage` is the only erased-to-typed conversion.

A foreign carrier never constructs `TensorRef<T>`, `TensorMut<T>`, `&[T]`, or
`&mut [T]`. After validating every descriptor property the runtime can observe,
the C entry point dispatches to a branded `ForeignTensorAccess<T>` held inside
the owner module. That wrapper retains a typed `NonNull<T>`, checked metadata,
and the foreign disposition, but exposes only narrow indexed raw-pointer
operations inside the audited unsafe core. It does not claim to prove physical
allocation size, lifetime, overlap with another descriptor, or synchronization
between foreign calls; [05-OP-31]'s caller preconditions continue to own those
facts. This preserves the public boundary without manufacturing a Rust
exclusivity guarantee it cannot establish.

Raw operations remain only where the operation is semantically byte-based:
allocation/free, exact-bit copy, zero initialization, foreign ownership
transfer, and device submission. Each takes validated `ByteCount` and a branded
source/destination representation. There is no general `as_bytes_mut` escape
that lets a caller reconstruct the old cast surface.

Every public C entry validates every observable property before access:

- non-null descriptor and metadata pointers where required;
- known dtype tag and its exact `Repr`;
- rank and dynamic shape/stride length;
- non-negative extents and exact count/byte products;
- stride reachability within capacity;
- data pointer presence and alignment for non-empty tensors;
- ownership and reserved-field invariants; and
- canonical Bool8 bytes at the first typed boundary when values originated
  outside the runtime.

Python host and DLPack paths use a private validated wrapper. DLPack export may
release an opaque pointer only together with the exact dtype, shape, stride,
byte offset, device, and deleter metadata derived from that wrapper.

Delivery is split without weakening this contract. Phase 2 owns the
Python/device/DLPack wrappers because they depend on the generated device
descriptor. Phase 3 owns the repository runtime and public C entries and
requires only Phase 1; it seals the host `data` field independently of excluded
binding/device work. Phase 5 requires both paths before class closure.

The field seal lands last in Phase 3. At that commit, a clean compile plus the
source-completeness oracle proves that no in-repository direct field path was
missed.

## C5. Typed lane loads, stores, and Bool8

Each backend has a private renderer over the same closed lane identity:

```rust
pub struct LaneElement<T: TensorElement> { /* sealed */ }
pub struct Loaded<T: TensorElement> { /* expression + representation */ }
pub struct Store<'a, T: TensorElement> { /* destination + index */ }

fn load<T>(buffer: BufferRef<T>, index: Index) -> Loaded<T>;
fn store<T>(buffer: BufferMut<T>, index: Index, value: Loaded<T>);
fn cast<S, D>(value: Loaded<S>) -> Result<Loaded<D>, NumericTrap>;
```

The renderer chooses C/HIP/Metal spelling from `T::REPR`; call sites cannot
pass a spelling string. Source and destination type parameters are distinct, so
`cmplt` reads `T` and stores `Bool8`, while `cast` reads `S` and stores `D`.
Copy/realize retain one representation and exact byte count. Buffer construction
checks that the descriptor tag and lane marker agree.

Backend-private raw template functions may emit text only after receiving these
typed values. The structural oracle rejects direct pointer casts, hand-authored
element spellings, or raw store templates outside the renderer modules. This is
the smallest unsafe/textual core; tests mutate that core because a type system
cannot prove that an unsafe implementation emits the intended token.

HIP Bool8 completion under [#1364] includes:

- `cmplt` and every bool-result comparison writing exactly `0` or `1`;
- checked casts to bool, with non-0/1 inputs producing [04-NUM-10]'s device
  error flag and first failing index;
- casts from Bool8 reading one byte and producing the exact target value;
- copy and realize using Bool8 element types and one-byte counts; and
- guard-byte hardware tests proving no write before/after the allocation.

The lane probe compares exact representation, never only width. For every
admitted dtype it round-trips sentinel bytes through source generation,
compile/load, device transfer, one identity kernel, and return. Sentinels cover
signed zero, canonical and noncanonical NaN payloads where bit-preserving ops
require them, float extrema, signed-integer extrema, f16/bf16 bit patterns, and
Bool8. A width-equal f32/int32 swap must fail the probe.

## C6. Permanent guards

Phase 0 creates `scripts/runtime_representation_oracle.py`; every later phase
extends the same script. Its structural inventory is derived from tracked
source and classifies every hit into one of these final forms:

- the private raw owner module;
- a generated descriptor artifact with a freshness proof;
- a typed view or typed lane operation; or
- a foreign-boundary validator that yields only branded indexed foreign access
  before the first operation.

Until Phase 5, a hit may instead match one exact Phase-0 transition-debt
identity with one owning deletion phase. That frozen manifest is generated
from the reviewed Phase-0 tree, integrity-digested, and shrink-only. The digest
binds one canonical object containing both the immutable foundation rows and
the executable coverage manifest (enumerator, command, success condition, and
mutation set); the separately stored active-debt identity list is deliberately
outside that digest so it can only shrink. Co-editing the stored and live
coverage manifest without moving the reviewed digest therefore fails. The
oracle may delete active rows, but regeneration cannot bless an addition,
rename, signature change, relocation, reclassification, or weakened coverage
contract. Such a change removes the old identity and introduces a new
unclassified hit, which fails. At Phase 5 the debt set must be empty. This is
an explicit migration ledger, not an allow-list or a final authority class.

Anything neither final nor an unchanged frozen debt identity fails. The
inventory includes descriptor fields, `data` access, pointer casts,
width/arithmetic matches, `normalized_key` arithmetic, fixed-rank arrays,
narrow metadata fields, backend element spellings, and load/store templates.
A final-form exception may name a private owner function and reason, but a
stale or unmatched entry fails and an issue citation does not authorize a raw
path.

The C-family portion is parsed by the shared `chelis-c-surface` crate through
libclang's compiler AST and preprocessor for the source's declared dialect,
rather than matched as declaration text. Every carrier row is owned by its
enclosing declaration. Its identity includes the authored declaration and
libclang's complete canonical type: modifiers and address spaces, every
pointer/reference layer, function signature, and every array extent.
Relocating an unchanged carrier between functions, changing one extent of a
multidimensional array, or changing `float *` to `_Atomic(float) *` therefore
changes identity. Width-equal spellings such as `int` and `unsigned int`
cannot collapse to one identity. The parser evaluates production
preprocessor configurations explicitly; syntax rejected in every declared
dialect fails closed rather than disappearing from the inventory.

Typedefs (including aggregate definitions) and object-like or function-like
type macros are collected from every tracked C-family source as a conservative
include prelude. Every definition of a name contributes to a monotone union:
a later macro redefinition or block-local typedef can add meaning but cannot
erase an earlier numeric meaning. This intentionally over-approximates
preprocessor and lexical scopes; a source-specific definition may expose a
carrier for review, but it can never launder one out of the inventory.

Rust emitters are parsed with `syn`. Ordinary/raw string literals,
`stringify!` inputs, and `macro_rules!` token trees enter the same C-family
parser with their complete enclosing Rust item path. Named or positional
format holes may stand only for a type or declarator name inside an otherwise
complete declaration; a wholly dynamic C-family declaration fails closed.
Only a module whose attribute is exactly `cfg(test)` is excluded;
`cfg(not(test))`, production items after a test module, and production files
whose names end in `_tests.rs` remain in the source universe. The tracked
suffix set covers C, C++, Objective-C, Objective-C++, CUDA (including `.cuh`
headers), HIP, Metal, and Rust source/header forms. A complete source
containing carrier syntax must have closed lexical constructs and balanced
delimiters. An emitted Rust fragment may leave only its outer C block open;
each carrier candidate must remain locally complete, so splitting the type
from a pointer or array declarator fails instead of disappearing.

The parser and the public-header capacity census share lexical normalization,
the closed arithmetic/non-arithmetic type vocabulary, alias expansion, and
numeric classification. Known SDK handles and control enums occupy an exact
external-nonnumeric set. An unknown arithmetic-shaped spelling, unresolved
alias chain, malformed candidate, or recognized incomplete carrier fragment is
a build failure, not an unclassified spelling that a later regular expression
may or may not learn.

The oracle self-validates with temporary mutations that are restored before it
returns:

- add a dtype without a complete contract;
- pair a 32-bit float marker with the int32 representation;
- add a direct `chelis_tensor.data` access at an identity absent from the
  frozen debt manifest;
- add a public raw byte accessor;
- replace exact product arithmetic with saturation;
- add a fixed-rank device field or narrow one metadata field;
- handwrite a second ABI field list;
- vary C qualifier placement, use an array declarator or pointer cast, hide a
  carrier behind typedef/macro aliases, or add qualified `sizeof` arithmetic;
- redefine a numeric alias as nonnumeric, add C++ reference/template carriers,
  or introduce an unknown C arithmetic spelling;
- use positional Rust format holes or a `macro_rules!` literal, or split one
  emitted declarator across Rust string fragments;
- change a Bool8 lane spelling to `float`; and
- make a Bool8 kernel store `1.0f` or omit the device failure flag.

Every mutation must make the relevant phase command fail for the intended
reason. The same command executes the parser's positive and negative contract
suite. A source scan is the completeness guard; compile-fail and execution tests
prove its sanctioned replacements work.

---

# Part II — process rules

## B1. Freeze points

- Phase 0 freezes the derived inventory, mutation set, current accepted/rejected
  behavior, and the exact issue-to-phase map. Later phases may reduce raw hits
  but may not add an exception. The Phase-0 implementation moves this freeze
  to schema 3 because schema 2 still used an incomplete token scanner and bound
  prose mutation labels instead of executable witness semantics. Schema 3 uses
  the compiler-backed and Rust-AST identities defined in C6. For every mutation
  it binds a stable witness ID, source path, exact mutation implementation
  digest, expected failure code and reason prefix, and required phase command.
  Changing a mutation body or expected disposition therefore moves the freeze
  digest. The manifest-integrity and adversarial parser controls above are the
  required negative evidence for this freeze move.
- Phase 1 freezes `DTypeContract`, sealed element markers, exact capacity keys,
  and checked finite-count types. Later phases consume them without parallel
  tables.
- Phase 2 freezes the device renderer, generated device descriptor, and
  binding-layout parity against the schema Phase 3 landed.
- Phase 3 freezes the shared descriptor schema, generated host/public-C layout,
  typed runtime ingress, and the private data field. Later public field changes
  amend [05-OP-31] first and regenerate every consumer in one change. Reopening
  a raw accessor is a design change, not a local optimization.
- Phase 4 freezes the typed lane renderer and all-lanes representation probes.
  A new dtype or lane cannot land without extending both.
- Phase 5 freezes the composite oracle and closure receipts. No individual
  child test substitutes for it.

Moving a freeze requires changing this document, the owning numbered spec when
semantics move, the oracle's integrity digest, and a mutation that would have
accepted the forbidden behavior.

## B2. Invariants at every phase boundary

1. The public C ABI remains [05-OP-31]-exact and configuration-invariant.
2. A correct landed receipt does not regress while its structural replacement
   is being built.
3. No phase introduces a second width, arithmetic, dtype-tag, or ABI field
   authority, even temporarily.
4. Debug/release and source/device paths have the same validation disposition.
5. Zero extents mean zero elements; rank zero means one element.
6. Equal capacity never authorizes reuse across unequal `Repr` values.
7. Observable foreign metadata validation happens before the first read,
   write, copy, render, or ownership action; no Rust reference is formed from
   a foreign carrier, and [05-OP-31]'s allocation/lifetime/synchronization
   preconditions are not reclassified as validated facts.
8. Unsupported typed cells remain loud under [#730]; this plan never replaces
   an unsupported operation with a default value.
9. Each phase's one command ends in its exact PASS line or the phase is not
   complete.

## B3. How to pick up a phase

Before implementation, re-read [#893] and every live child in full, confirm the
controlling numbered atoms, and run the preceding phase oracle when one exists.
Write the positive and negative tests named by the phase before implementation.
Work in a dedicated branch/worktree and claim only the issues whose
implementation the slice actually owns. A phase PR uses `Part of #893`; it uses
`Closes #N` only when that issue's complete acceptance row goes green in the
same PR.

Every PR, including a documentation-only amendment, receives a fresh-context
red-team review at its exact head. A changed head after an in-scope P0/P1 repair
requires another fresh review. Hardware-required evidence is not replaced by a
code-generation text test.

---

# Part III — phases

## Phase 0 — executable inventory and red controls

**Delivers:** the derived inventory and exact shrink-only transition-debt
manifest in C6; release-profile reproducers for exact capacity collision,
count/byte overflow, zero extents, and malformed foreign metadata; one
detection mutation for every source classifier, including a new direct field
access and incomplete dtype registration; a fail-closed structural parser for
C/HIP/Metal sources and Rust-emitted C-family fragments, including monotone
cross-source alias resolution and injective declarator identities; source-only
and hardware probe harnesses; all landed receipts as positive controls.

The inventory records identities, not mutable line numbers. Each enumerator has
a mutation that plants a new hit in a different file/configuration. HIP and
Metal ignored tests are listed with their hardware command and cannot count as
executed merely because the default suite skipped them.

**Does not:** change representation, allocation, descriptors, field privacy, or
kernel behavior. Direct field access becomes a compile-fail fixture in Phase 3,
after the field seal exists; Phase 0 proves only that a new unlisted access is
detected and that the existing debt cannot grow or move.

**Oracle:**

```sh
uv run --managed-python --python 3.11 --no-project python \
  scripts/runtime_representation_oracle.py --phase 0
```

Final line: `RUNTIME REPRESENTATION PHASE 0: PASS`.

## Phase 1 — closed representation and capacity vocabulary

**Requires:** Phase 0.

**Delivers:** C1's closed vocabulary and runtime/ABI projection plus C2
completely: `DTypeContract`, arithmetic representation, sealed markers, exact
`CapacityKey`, `ShapeMetadata`, `ElementCount`, `ByteCount`, and checked target
projection. All existing host allocation/view entry points use the checked
types. All planners compare exact capacity plus exact `Repr`. Backend lane
consumers complete C1's consumer-exclusivity condition in Phase 4; their exact
frozen debt cannot grow before then.

**Issue exits:** [#888] closes when its collision witnesses remain distinct,
product-only equivalent factorizations remain equal, division/domain witnesses
remain distinct unless their exact predicates are proved, and every reuse
consumer uses the exact key.
[#889] closes when every allocation/view/copy path consumes the checked types in
release and debug, including negative, zero, product-overflow, byte-overflow, and
capacity-offset controls.

**Oracle:**

```sh
uv run --managed-python --python 3.11 --no-project python \
  scripts/runtime_representation_oracle.py --phase 1
```

Final line: `RUNTIME REPRESENTATION PHASE 1: PASS`.

## Phase 2 — canonical ABI descriptors

**Requires:** Phase 3, which includes Phase 1 and lands the shared schema plus
the generated host/public-C descriptor.

**Delivers:** C3's device half and the binding/device portion of C4: the device
renderer over `chelis-abi`'s landed shared schema, the generated device
descriptor, freshness and layout probes, dynamic-rank exact metadata plus
private validated Python/DLPack wrappers, and the deletion of every handwritten
device or binding mirror. Phase 2 requires Phase 3 and consumes that landed
schema; it neither installs nor seals the raw host descriptor. The current
[#1289] public ABI and [#1347] zero-extent behavior are positive receipts.

**Issue exit:** [#1345] closes after Python host-to-device, device entry,
device-to-host, and DLPack paths pass rank 0, 1, 8, and greater-than-8 cases,
preserve `int64` values above `i32::MAX`, reject out-of-domain values before
copy, and have no fixed array or narrow mirror left.

**Oracle:**

```sh
uv run --managed-python --python 3.11 --no-project python \
  scripts/runtime_representation_oracle.py --phase 2
```

Final line: `RUNTIME REPRESENTATION PHASE 2: PASS`.

## Phase 3 — typed host runtime access and the field seal

**Requires:** Phase 1. Phase 2 does not gate this phase. Phase 3 creates the
shared `chelis-abi` field schema, host renderer, generated raw host descriptor,
and generated public-C descriptor without consuming the Python or device
descriptor.

**Delivers:** C3's shared-schema and host/public-C portion plus the host-runtime
and public-C portion of C4. Migrate
repository-owned allocation, views, elementwise kernels, reductions,
formatting, collection ingress/egress, and ownership paths to validated typed
views; foreign C entries use only the branded indexed access core. Delete
public unchecked pointer helpers and broad raw-byte access. Install the
generated raw host descriptor only with that migrated consumer set, then make
every raw host descriptor field private in the owner module as the last diff.
Phase 2 later owns the binding/device portion, and C3/C4 are complete only when
both phases are green.

The exact-head compile after the seal must exercise every runtime target and
test target. A compile-fail fixture outside the owner module attempts direct
access and wrong-typed view construction. Foreign C probes mutate each field
independently and prove failure precedes access with guard pages around data.

**Oracle:**

```sh
uv run --managed-python --python 3.11 --no-project python \
  scripts/runtime_representation_oracle.py --phase 3 --host
```

Final line: `RUNTIME REPRESENTATION PHASE 3: PASS`.

## Phase 4 — typed lanes and Bool8 execution

**Requires:** Phase 1 for the renderer and Phase 2 for device descriptors;
Phase 2 already includes the Phase 3 host/schema prerequisite.

**Delivers:** C5 completely across C, HIP, and Metal. Delete duplicate width and
element-spelling tables, route loads/stores/casts through typed builders, and run
exact representation probes. Implement the [#1364] Bool8 operation family and
device-side checked cast. Preserve [#1360]'s allocation/spelling agreement as a
receipt and make its old `float` mutation fail at compile/structural and hardware
layers.

**Issue exit:** [#1364] closes after `cmplt`, casts both ways, copy, and realize
execute on HIP hardware for empty, one-element, boundary-size, and multi-block
inputs; non-0/1 casts report the first failing index; guard bytes remain intact;
and exact returned bytes match the host reference.

[#899] closes in this phase, not Phase 1: every runtime and backend consumer
must derive stored and arithmetic representation from `DTypeContract`, every
duplicate width/element-spelling table must be gone, and the new-dtype mutation
must fail the complete consumer set.

Each hardware host first runs its exact lane and emits the receipt consumed by
the coordinator:

```sh
uv run --managed-python --python 3.11 --no-project python \
  scripts/runtime_representation_oracle.py --phase 4 --lane hip \
  --emit-receipt runtime-representation-receipts/hip.json
uv run --managed-python --python 3.11 --no-project python \
  scripts/runtime_representation_oracle.py --phase 4 --lane metal \
  --emit-receipt runtime-representation-receipts/metal.json
```

**Oracle:**

```sh
uv run --managed-python --python 3.11 --no-project python \
  scripts/runtime_representation_oracle.py --phase 4 --all-lanes \
  --receipt-dir runtime-representation-receipts
```

Final line: `RUNTIME REPRESENTATION PHASE 4: PASS`. A missing compiler,
device, or runnable hardware test reports `BLOCKED`, never PASS.

## Phase 5 — class elimination

**Requires:** Phases 0–4.

Delete transitional adapters, stale comments, old proposal artifacts, and
temporary inventory classifications. Re-run the capacity census, public-header
closure, Python binding census, all runtime tests in release and debug, and the
source/device sentinel matrix. Update current-state docs and the tracker only
from this evidence.

HIP and Metal evidence may be produced on different machines. Each lane runner
emits a content-addressed receipt containing the exact git commit, dirty-state
verdict, oracle/version digest, test-binary and generated-artifact digests,
toolchain and device identity, command, and result. A receipt is accepted only
for the exact candidate head and only after the coordinator reruns the
source-only mutation/freshness legs at that head. Replaying a receipt for an
older commit, a different oracle digest, a dirty tree, or incomplete hardware
matrix fails. The coordinator may execute a locally available lane directly or
verify its receipt; absence of either is `BLOCKED`.

The one authoritative class oracle is:

```sh
uv run --managed-python --python 3.11 --no-project python \
  scripts/runtime_representation_oracle.py --all-lanes \
  --receipt-dir runtime-representation-receipts
```

Its final line is exactly `RUNTIME REPRESENTATION ORACLE: PASS`. It invokes the
frozen child phases, runs locally available HIP/Metal hardware and verifies the
other exact-head lane receipts, verifies restoration after every self-mutation,
and fails if the worktree is dirty afterward.

[#893] closes only after that command passes at the exact implementation head
and every open child has its own closure receipt. A green crate-local suite,
source-only codegen, or individual child PR is supporting evidence, not the
class oracle.

---

# Part IV — issue map and interlocks

## Issue map

| issue | role | closure evidence |
|---|---|---|
| [#893] | class tracker and sub-issue parent | Phase 5 all-lanes oracle and every child receipt |
| [#899] | arithmetic representation missing from `Repr`'s closed contract | Phase 4 complete-consumer and new-dtype mutations |
| [#889] | unchecked or optional count/byte arithmetic | Phase 1 checked-type exclusivity in debug and release |
| [#1289] | exact public tagged C carrier | landed receipt; Phase 2 layout/freshness probes prevent a successor mirror |
| [#1345] | fixed-rank int32 Python/HIP device carrier | Phase 2 dynamic-rank exact metadata execution |
| [#1360] | HIP Bool8 allocation/spelling mismatch | landed receipt; Phase 4 old-spelling mutation and guard-byte hardware test |
| [#1364] | missing HIP Bool8 kernel family | Phase 4 exact hardware execution and device-side failure flag |
| [#888] | saturating symbolic capacity equality | interlocked Phase 1 obligation; keeps its current issue ownership |
| [#1347] | zero-extent allocation accounting | landed receipt; Phase 1 and Phase 5 preserve zero-element semantics |

Closed receipts remain closed and are not made children retroactively. [#888]
remains outside the GitHub sub-issue tree and is linked as `Also part of #888`;
its oracle turns green in Phase 1, so it is part of the complete blocker
delivery without changing parentage.

## Launch-blocker alignment

The current [#1362] text places the host runtime/ABI slice in Tier 1 section B
and names [#889], [#888], [#1347], and [#1289]. Its scope note excludes HIP,
Metal, and Python from the launch gate. The live [#893] child graph additionally
carries [#899], [#1345], [#1360], and [#1364]. This plan covers the union without
silently widening the launch gate: the section-B blocker exit uses the Phase 1
host-capacity evidence, the Phase 3 `--host` field-seal evidence, and the landed
[#1289]/[#1347] receipts. Phase 3 deliberately has no Phase 2 prerequisite, so
that exit does not pull Python, HIP, or Metal into the gate. Full [#893] class
closure continues through the non-launch-gating device/binding Phases 2, 4,
and 5. Parentage and landed receipts do not change. Any decision to make those
later phases release blockers belongs in [#1362], not this document.

## Interlock with runtime extents ([#1277])

[#1277] and [`runtime_extents.md`](runtime_extents.md) own which extent a
program has, the `RtDim`/`InputAxis` carrier, `output_axis_sources`, and
`spec/04` §4.7's guarded-claim rule. A declared extent claim that is not
statically proven adds an execution-time guard that traps `Domain`; it is never
rejected merely because the proof is unavailable.

This plan owns only whether two capacities may be proven equal for allocation
or reuse. [04-SHAPE-1]'s conservative `NotProvenEqual` loses reuse, never
correctness; it neither substitutes for nor discharges a §4.7 guard. Phase 1
builds `CapacityKey` over the semantically typed extent expression regardless
of which carrier [#1277] has landed.

## Interlock with dtype semantics ([#729])

`dtype_semantics.md` owns what every dtype means and the permanent numeric
surface ratchet. This document consumes [04-NUM-8]'s stored/arithmetic
representation and the capability tables; it does not author a new dtype cell.
[#892] stays under [#729]. A [#729] storage change cannot land unless this
plan's closed representation registration and lane probes either already admit
it or fail the change.

## Interlock with loud unsupported ([#730])

[#730] owns typed unsupported/capability failures. This plan uses that channel
when a target lacks a registered typed kernel. It owns the representation proof
that makes a supported path safe and [04-NUM-10]'s device flag for a supported
operation's data-dependent trap. It may not turn a missing kernel into a value,
and [#730] may not classify a representation mismatch as an ordinary unsupported
case after a buffer has been accessed.

## Interlock with the capacity ratchet ([#1288])

The existing primary, C-header, wire, and Python enumerators remain the public
surface completeness boundary. This plan adds no exception class. `chelis-abi`
and generated header fragments must be reached by the existing surface
enumerators in the same phase that creates them; a renamed or generated field is
a new exact identity until classified by final authority.

## Decision record

| question | decision | reason |
|---|---|---|
| public C carrier | preserve [05-OP-31], validate before access | external C is inherently forgeable; an ABI break does not make it typed |
| Rust raw field visibility | private to one owner module | `pub(crate)` leaves every runtime site able to repeat the defect |
| raw byte accessor | no general accessor | it recreates arbitrary typed casts; exact-bit operations are named narrowly |
| capacity canonicalization | arbitrary-precision reduced product key | saturation is the #888 mechanism; conservative non-reuse remains available |
| device metadata | generated dynamic-rank int64 descriptor | exact host domains already exist; fixed rank/int32 has no normative authority |
| backend type choice | sealed marker plus typed load/store builders | width-only checks miss equal-width representation swaps |
| foreign malformed carrier | checked failure before access | static prevention stops at the FFI boundary |
| final completion | one all-lanes oracle | source text cannot validate device writes or failure flags |

[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#888]: https://github.com/Chelis-Lang/chelis/issues/888
[#889]: https://github.com/Chelis-Lang/chelis/issues/889
[#892]: https://github.com/Chelis-Lang/chelis/issues/892
[#893]: https://github.com/Chelis-Lang/chelis/issues/893
[#899]: https://github.com/Chelis-Lang/chelis/issues/899
[#1277]: https://github.com/Chelis-Lang/chelis/issues/1277
[#1288]: https://github.com/Chelis-Lang/chelis/issues/1288
[#1289]: https://github.com/Chelis-Lang/chelis/issues/1289
[#1345]: https://github.com/Chelis-Lang/chelis/issues/1345
[#1347]: https://github.com/Chelis-Lang/chelis/issues/1347
[#1360]: https://github.com/Chelis-Lang/chelis/issues/1360
[#1362]: https://github.com/Chelis-Lang/chelis/issues/1362
[#1364]: https://github.com/Chelis-Lang/chelis/issues/1364
