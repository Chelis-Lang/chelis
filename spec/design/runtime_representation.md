# Runtime Representation and Tensor-Access Safety

**Status:** ACTIVE. Phases 0 and 1 are implemented and enforced by the Phase 1
composite oracle; Phases 2-5 remain planned. Tracking issue: [#893]. Acceptance
evidence is commit-bound by the oracle's execution receipts.
**Owning specs:** `spec/04-type-system.md` [04-NUM-4], [04-NUM-8],
[04-NUM-10], [04-NUM-11], and [04-SHAPE-1], plus
`spec/05-risc-primitives.md` [05-DIM-1], [05-DIM-2], [05-OP-31], [05-OP-33],
and [05-OP-44]. Those atoms decide language and ABI behavior. This document owns
only the implementation structure and delivery order. Where they disagree, the
numbered specs win and this document has a bug.
**Class fixed:** a tensor's dtype, stored representation, shape, capacity, and
lane element type can be stated independently, so an unchecked cast or stale
mirror can make one buffer mean two incompatible things. The end state makes an
in-repository mismatch unavailable through ordinary APIs and makes a malformed
foreign carrier fail before data access.

The Phase 1 composite runs in the dedicated
`runtime-representation-phase0-oracle` job in `heavy-e2e.yml`, daily at 03:17
UTC and on manual dispatch, with a 145-minute timeout: the 120-minute oracle
budget plus 25 minutes for cold environment setup. The stable job identity
predates Phase 1; its display name and command identify the current inherited
phase. Ordinary PR and main-push CI do not run this full oracle, so a completion
claim requires a candidate-head dispatch or equivalent clean execution receipt.

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

The public C carrier is [05-OP-31]'s tagged value and view layouts over
[05-OP-44]'s opaque handles: the layout-visible `chelis_tensor` that PR #1400
froze was superseded by [#1286]'s Phase 1 entry amendment, and the seal now
rides the opaque handle rather than a public field set. C can always
forge bytes, so "unrepresentable" has a precise boundary here:

1. repository-owned Rust, generated C, HIP, Metal, and Python code cannot obtain
   an element view without a matching representation proof; and
2. an arbitrary foreign entry borrow is treated as untrusted: every
   observable descriptor invariant is checked before access, while physical
   allocation size, lifetime, overlap, and cross-call synchronization remain
   the foreign caller preconditions stated by [05-OP-44]. No Rust reference or
   slice is formed from that storage.

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
4. Host runtime allocation gained checked metadata while compiler capacity
   equivalence still saturated before the runtime saw a value ([#888]); the
   exact capacity authority and legacy retirement below remove that path.
5. `Repr` derives byte width but not [04-NUM-8]'s arithmetic representation
   ([#899]). The missing fact is restated or inferred wherever a lane needs it.

Each repair made one path correct. None removed the ability to construct the
next inconsistent path. This plan lands its irreversible steps last: first make
every consumer use the typed replacement, then remove the raw route and require
the tree to compile.

## Non-goals

- No second public C carrier. `chelis_scalar`, the dtype tags, the read and
  write views, and the callables governed by [05-OP-31]/[05-OP-33]/[05-OP-44]
  keep their exact declarations, widths, order, and meanings; this plan adds no
  field, flag, or layout beside the opaque handle.
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

### C1 vocabulary delivery boundary

The dependency-bottom vocabulary can ship independently of the runtime element
seal and checked metadata: existing `RuntimeDType::repr()` and `byte_width()`
callers immediately consume `contract()`, without changing storage, ABI IDs,
kernel behavior, or arithmetic policy. This slice implements `DTypeContract`
and `ArithmeticRepr`, not the whole Phase 1 or the #899 Phase 4 exit.

Its executable acceptance surface is:

```sh
cargo nextest run -p chelis-vocab --test dtype_contract --test dtype_contract_compile
```

Every selected test must execute and pass. The table test covers all nine
[04-NUM-8] rows and negative equal-width/bool cases. Compile controls admit a
real external consumer, reject forged or rewritten private contracts, and
mutate each vocabulary in isolated source copies. New dtype controls complete
the other naming/decoding projections so the missing contract itself must
fail compilation. No tracked source is mutated by these tests.

The inventory gains `ArithmeticRepr` variant enumeration. Its six exact
registered variants and `DTypeContract::byte_width` in the vocabulary owner
are final forms, justified by this executed contract suite; another variant,
path, or width-helper owner is not. Existing foundation rows remain unchanged,
and no new transition debt is authorized. The code-derived Phase 0 coverage
manifest names these final forms and the contract suite. The runner verifies
that manifest against the configuration it executes; this acceptance addition
does not move the foundation digest. The Phase 1 composite below includes this
vocabulary contract.

### C1 runtime element delivery boundary

The inventory enumerates every trait implementation in the private `element`
owner, not only implementations whose written trait name is `ElementStorage`.
Only the exact storage, private-seal, arithmetic-identity, and blanket-projection
rows are admitted. Import aliases, including aliases defined in another file,
cannot hide an extra binding; an additional seal implementation is itself a
new row. Qualified trait spelling and enclosing module ownership remain part
of the row identity. The added-binding mutation exercises an ordinary alias.

The next independently shippable slice seals the runtime's existing element
implementations and binds their storage and arithmetic types to `DTypeContract`.
It leaves the existing unsafe pointer methods at their frozen identities until
Phase 3 replaces them with validated views; sealing implementations is not a
claim that arbitrary pointer access is gone.

`element.rs` owns `ElementStorage`, with a private sealing supertrait. The
public `TensorElement` requires that sealed storage registration and has one
blanket implementation projecting its dtype, `Repr`, and arithmetic type.
Neither downstream crates nor sibling runtime modules can register another
storage type. An exhaustive dtype match instantiates compile-time assertions
for every registration: exact dtype and representation identity, storage size,
and arithmetic representation. An equal-width float/integer pairing must fail,
not merely an unequal-size pairing.

`F16Bits` and `Bf16Bits` name the existing distinct `half::f16` and `half::bf16`
transparent storage types, not bare `u16` aliases. Reusing those types preserves
all existing kernel implementations and exact bit conversions. `Bool8` retains
its checked constructors and uses `()` to denote no arithmetic type; `()` is
not a tensor element. This slice adds no numerical operation or C ABI.

Acceptance is `cargo nextest run -p chelis-runtime --test element_contract`:
all nine storage/arithmetic rows, all narrow-float bit patterns, Bool8's complete
byte domain, equal-width mismatch rejection, and executed compile controls for
external and sibling-module sealing and incomplete/wrong registrations. Compile
mutations use isolated copies of the real registration and trait declaration,
with stubs only for the unchanged raw-access environment; positive controls
also compile against the real runtime crate. The Phase 0 oracle executes this
suite in release mode and inventories every trait implementation in the element
owner. Its exact final set admits the nine storage registrations and private
seals, the seven arithmetic identities, the derived blanket projection, and
the compile-time assertion owner. The old nine unsealed bindings leave
active debt; foundation rows remain immutable. No raw-access debt is relocated
or reauthorized. Phase 1 still awaits checked metadata and its complete oracle.

## C2. Exact capacity identity and checked finite counts

Compiler equality and runtime allocation are related but distinct domains.
Neither may use a saturating or wrapping integer.

### C2.1 Compiler capacity keys

`CapacityKey` is an opaque proof carrier backed by a canonical tree over the
complete multiplication/division vocabulary:

```rust
#[derive(Clone, Debug)]
pub struct CapacityKey { /* private canonical tree and validity domain */ }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NotProvenEqual { /* private reason */ }

impl CapacityKey {
    pub fn prove_equal(&self, other: &Self) -> Result<(), NotProvenEqual>;
}
```

`CapacityKey` is deliberately carrier-independent: it is built from the
complete semantically typed extent expression, not from the IR enum that
happens to carry that expression. `DimExpr`, `RtDim`, and `InputAxis` are
carrier spellings, not key variants. Phase 1 consumes whichever carrier
[#1277] has landed; it must not translate an `RtDim`/`InputAxis` edge back into
`DimExpr` or recover it by name.

The canonical tree, its source-identity atoms, its validity domain, and every
constructor are private to `chelis-ir`. Backends may receive and compare the
opaque proof carrier only through `prove_equal`; they cannot forge a literal or
symbol, invoke normalization, recover a source by string spelling, or project
the key into `u64`, `i64`, or `usize`. A crate-private literal accessor may
borrow the exact `BigUint` only when the complete key is a total literal.
`CapacityKey` implements neither raw equality nor hashing: those traits would
bypass the validity-domain check and let a pending quotient compare equal to
itself without a proof. `prove_equal` is the only equality surface.
Every dynamic source atom includes an opaque identity for the verified program
that owns it. DAG-local node and axis identifiers are meaningful only within
that scope, so independently verified programs cannot prove equal merely
because their local identifiers coincide.

Product-only regions flatten, sort, fold arbitrary-precision literals, and
remove multiplicative identities. They may collapse a zero product only when
every factor whose key would be discarded is statically total or its validity
predicates have been discharged. Thus `0 * symbol` may become zero, but
`0 * (1 / n)` retains the partial quotient and remains distinct from zero
unless `n != 0` and `n` divides 1 have both been proved.
The literal fold is owned by one private exact-product carrier whose state and
input are both `BigUint`; no free primitive accumulator or alternate final-form
arithmetic owner is admitted.
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

Phase 1's first implementation slice does not add such a discharge surface.
It records the intrinsic validity obligations of an exact quotient and returns
`NotProvenEqual` while any remain. Only static literal evaluation can discharge
them in that slice. This is a conservative loss of reuse, never permission to
drop or assume an obligation.

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

#### Legacy capacity authority retirement (#888)

The closeout removes `DimExprKey`, `DimExpr::normalized_key`, and their rational
normalizer; there is no compatibility alias. The only remaining production
caller was `specialize::dims_equivalent`, which compared single `DimInfo`
atoms. It compares resolved literal values or unresolved symbol spellings
directly through `DimExpr::from`, preserving that local axis-pattern decision.
It neither proves storage capacity nor manufactures a scoped source identity.

`DimExpr` remains the renderable/evaluable extent carrier. Its finite
`evaluate` and `as_concrete` projections use checked multiplication, returning
their existing error and `None` channels on overflow. Exact mathematical
identity remains exclusively `CapacityKey::prove_equal`.

The old rational-equivalence corpus migrates to the private exact-key tests:
product-only equivalences remain positive; symbolic cancellation, quotient
reassociation, and zero/partial-domain erasure become negative controls.
The original large-product key witness is retained in
`capacity_key::tests::capacity_key_products_use_arbitrary_precision_without_collision`;
the C/HIP `issue_888_capacity_collision` suites continue to prove placement.
This closeout does not complete Phase 1: #889's mandatory checked runtime
metadata and the composite Phase 1 oracle still have to land.

The code-derived Phase 0 coverage manifest names the inverted shared-plan
witness, executes finite-projection overflow controls in release, and adds a
mutation restoring `DimExpr::normalized_key`. That restored owner must be
rejected as unclassified even though it existed in the immutable foundation.
Paired compile-fail/compiling API probes independently prevent the retired key
and method from returning. The foundation identities remain byte-identical;
only deleted active debt is removed and current samples are refreshed.

### C2.2 Runtime metadata types

`chelis-abi` is the shared owner of checked descriptor metadata. The runtime's
existing descriptor/count/byte/stride validation moves into that owner; it is
not copied into a second implementation. Runtime and binding consumers use the
same checked types. All host and device allocation/view paths consume values
whose fields are private to this owner and whose public constructors validate:

```rust
struct CheckedDomain { /* rank, extents, dtype, count, logical bytes */ }
pub struct ShapeMetadata { /* one domain, canonical strides */ }
pub struct StridedMetadata { /* one domain, supplied strides, reachable span */ }
pub struct ElementCount(i64);
pub struct ByteCount(i64);
pub struct AllocationBytes(usize);
```

The crate depends only on the standard library and `chelis-vocab`. Metadata
construction may own shape and stride vectors, but it never owns or allocates
tensor storage, frees tensor bytes, or performs arithmetic on tensor payloads.
Runtime-specific execution and iteration consumers may remain in the runtime;
they consume the shared checked authority instead of deriving another product,
byte count, stride, or descriptor validity decision. Python does not link the C
runtime merely to obtain this validation.

One private `CheckedDomain` constructor checks rank/domain agreement,
non-negative extents, dtype, element product and logical bytes. Existing
`ShapeMetadata` owns one such domain and derives canonical row-major strides;
its contiguous runtime indexing and iteration contracts remain unchanged.
`StridedMetadata` owns one domain, immutable supplied strides and a checked
reachable span. It does not contain or reconstruct a second `ShapeMetadata`.
Both layouts reuse `ElementCount` and `ByteCount`; no consumer duplicates their
product or representation-width validation.

The zero-offset device view API admits nonnegative int64 strides, including
broadcast stride zero. Negative strides need an explicit base/offset and minimum
bound model and are rejected by this API; this is not a language restriction.
All supplied domains are checked before an empty shortcut. Any zero extent has
zero reachable bytes without computing unused canonical suffix products; rank
zero has exactly one element. A nonempty view checks the largest reachable
element offset, its representation bytes and target projection against the
storage owner's actual byte capacity. Logical count/bytes remain distinct from
reachable span, so a broadcast view need not own logical-count-many elements.
`ByteCount` is checked multiplication of an `ElementCount` and a
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

#### Host checked-metadata delivery (#889)

The host slice moves the private metadata authority into `chelis-runtime`'s
`metadata` module. The opaque tensor stores one `ShapeMetadata` rather than
independently assignable shape, strides, count, rank, and dtype. Metadata is
immutable after checked construction; repurpose replaces it atomically after
the existing uniqueness, provenance, and exact-storage-capacity checks.
Storage capacity is a validated `ByteCount`. This is metadata privacy, not the
later descriptor/element-pointer ownership seal.

That module is the source of the Phase 2 extraction into `chelis-abi`, not a
permanent second owner. The extraction preserves the checked types' behavior
and error classes. It does not flatten the host tensor's owner graph or change
its public C ABI.

`ElementCount` owns zero-aware extent products. `ShapeMetadata` additionally
derives every canonical suffix stride using checked arithmetic: an empty
`[MAX, MAX, 0]` is valid, while `[0, MAX, MAX]` still has an unrepresentable
stride. No operation derives its own product from raw tensor extents. Indexed
movement uses checked metadata indexing and byte-range projection; axis loops
receive checked decomposition from metadata. Empty operations return before
requesting irrelevant nonempty iteration spaces. `IterationSpace` owns contraction
loop extents without inventing storage strides for a domain that has no tensor.
Scratch entry counts use the same checked count/byte/target projection, but their
width is the physical Rust entry layout (which may include an accumulator or
index), not a second interpretation of a Chelis stored representation.

Public host views remain contiguous under [05-OP-31]. This slice introduces no
new strided-view ABI or device descriptor. Entry borrows validate the declared
capacity against the checked contiguous range; the physical foreign allocation
and lifetime remain caller obligations. Allocation and byte-copy submission
receive `AllocationBytes`, never recompute count times width. Public count and
shape observation project the validated int64 values without re-evaluation.

Padding preserves its int64 width until checked shape construction, and uses
metadata-derived offsets. Nested tensor ingress and padding write scalar bits
through a tensor-and-index boundary that checks representation and byte range,
not a raw data pointer plus an independently calculated offset.

The host slice's acceptance surface runs `checked_metadata`,
`checked_metadata_padding`, `metadata_compile`, and the existing
`exact_tagged_c_abi`, `op33_empty_tensor_axis_decomposition`, and
`op33_tensor_validation` integration suites in both debug and release, together
with the dtype-domain matrix, int64 carrier, repurpose, and write-guard controls.
The int64 carrier's existing greater-than-8-GiB allocation test remains an
explicitly ignored manual gate owned by #1112; it is not an executed receipt.
The final Phase 1 command still additionally
requires generated-C adoption and execution-receipt/mutation integration;
host-only green does not close #889 or #893.

This delivery extends the code-derived Phase 0 configuration with the private
`metadata.rs` source and the exact width owners `ElementCount::bytes` (the
closed representation width) and `ElementCount::scratch_len` (physical scratch
layout).
Neither admits another owner, pointer cast, or dtype authority. Both feed checked
byte construction and allocation projection. Optimized executable mutations must
reject weakened extent, count, byte, stride, target, capacity, and scratch checks;
paired compiling/noncompiling callers prove the private construction boundary and
the field-exposure mutation proves that boundary's test sensitivity. An added
unregistered width owner in this same module must fail the structural inventory.
The 358 immutable foundation rows remain byte-identical. Two retired raw-index
helper rows leave the active debt (345 to 343); no new foundation debt is added.
The host contract suites run in both profiles through the existing Phase 0
command and hosted job. Their addition is supporting evidence for this host
slice, not the final Phase 1 completion oracle.

#### Generated C snapshot and reshape delivery (#889)

Generated DAG snapshots consume `chelis_tensor_stride` and
`chelis_tensor_byte_count`, projections of the host's private checked metadata.
The byte count describes the logical copy range, not spare storage capacity.
The host stride adapter delegates to the same observation. Neither emitter
reconstructs those values from raw shape and dtype observations.

DAG reshape calls `chelis_tensor_check_reshape` before destination allocation or
repurpose, transporting the same target shape as that submission through exact
tagged int64 rank and extent scalars, like the existing repurpose boundary.
Host reshape delegates
to `chelis_tensor_reshape`, whose runtime owner validates a flat exact int64 list,
constructs checked target metadata, checks equal counts, and copies through
`AllocationBytes` into an independent result. Empty copies submit no null-pointer
memory operation. All four declarations have exact [05-OP-33] registrations.
This is a separable adoption of existing checked metadata; it creates no new
descriptor or Python interface.

The slice's acceptance surface combines `checked_c_metadata` in debug and release,
optimized generated host/DAG reshape execution with undefined-behavior
sanitization, and bounded emitter delegation controls. Positive cases include all
nine representations, exact stored bits, scalar and zero-extent shapes, int64
metadata above int32, and metadata observation during a write guard. Negative
cases cover malformed carriers, invalid axes, unequal counts, overflow of count,
stride or byte size, and data access during a write guard. A mutation that restores
raw snapshot arithmetic must fail the bounded delegation control. The
code-derived Phase 0 manifest includes these supporting tests and mutation; its
358-row immutable foundation inventory is unchanged. The two retired emitter
spelling owners leave active debt (343 to 341).

#### Generated C shared indexing delivery (#889)

The shared elementwise cohort saves runtime-checked scalar/identity projections
before allocating or repurposing output storage. `ShapeMetadata` validates a
tensor-domain projection; `IterationSpace` validates an unmaterialized domain
without imposing storage bytes or suffix strides. [05-OP-33] owns the two C
signatures and their exact tagged-int64 boundary. No Python interface changes.

The bounded DAG cohort is `emit_binary`, `emit_floor_div`,
`emit_floor_div_reduced_f`, `emit_binary_reduced_f`, `emit_binary_func`,
`emit_cmplt`, `emit_unary`, `emit_integer_abs`, `emit_recip`, `emit_unary_func`,
`emit_unary_reduced_f`, `emit_recip_reduced_f`, `emit_binary_func_reduced_f`,
`emit_extrema_adjoint`, `emit_unary_func_reduced_f`, `emit_fused_elem`,
`emit_realize`, and `emit_cast`. The host cohort is checked cast plus the four
binary/unary operator/function dispatch arms. Their loops use the checked output
count and `i * step`; direct-index fast paths additionally require identity
projections or at most one output element. Cast failure selection retains the
smallest failing domain index and applies the saved projection when reclassifying
that input. The host's two coordinate helpers are deleted; its BLAS stride
observer and the public raw movement/reduction helpers remain.

The supporting acceptance surface is the existing Phase 0 command, extended with
the runtime `checked_c_indexing` suite in debug and release, the backend's exact
18+5 cohort control, optimized generated-C sanitizer execution, and the existing
dtype, cast, and storage-reuse suites. Runtime mutations execute removed shape
checking and incorrect scalar/identity steps against real owner tests. Emitter
controls reject restored raw indexing, late validation, unchecked loop bounds,
and scalar admission to fast paths. The registered `checked_reshape.ch` example
also exercises ordinary elementwise composition and a checked cast.

This is one shippable slice because the new index projections and their emitter
consumers establish one shared iteration contract. The code-derived Phase 0
manifest names these commands and controls; its 358 immutable foundation rows
remain unchanged. Fifteen retired load/store-template owners leave active debt
(341 to 326); reduced-float and other surviving obligations keep their rows.
The Phase 1 composite below includes this supporting execution surface.

The two new emitter projection helpers are exact final metadata owners in the
inventory, alongside the existing checked runtime owners. Their int64 declarations
are projections of the registered [05-OP-33] operations. The executable recorder
compiles those production methods and rejects restored raw calculations, constant
steps, and weakened identity conditions. This final classification is limited to
these two methods and `backend-element-spelling`; another numeric or width owner
still fails the inventory. It adds no foundation or active debt.

#### Generated C permutation and expansion delivery (#889)

The DAG `emit_permute` and `emit_expand` paths validate the exact target metadata
and movement relationship before their existing storage-plan allocation or
repurpose. The runtime requires a complete normalized axis bijection for
permutation, and a unit replacement axis or one inserted axis for expansion;
every unchanged axis must agree, including on empty tensors. Existing extent
claim guards and source selection retain their runtime-extents authority.

Their loops transport coordinates as canonical int64 `chelis_scalar` values.
The runtime's private checked metadata owns unraveling and flattening; the C
consumer only reorders coordinates or supplies the explicit zero coordinate on
a replaced unit axis. No per-element heap allocation is needed. These are
metadata-only operations, so generated producer write guards remain live.
Element payload access and storage ownership keep their existing mechanisms.
The four exact [05-OP-33] operations and their census registrations ship with
these two consumers as one checked coordinate-mapping slice.

Supporting Phase 0 acceptance includes `checked_c_movement` runtime contracts in
debug and release, the two-method emitter adoption control, and generated C
execution under optimized ASan/UBSan. Positive and negative cases cover all nine
representations, scalar/empty/dynamic ranks, exact large metadata, permutation
bijections, expansion axis relationships, canonical scalar coordinates, and
linear/coordinate range rejection. Executable mutations must detect unchecked
coordinates and erased target validation. The immutable 358-row foundation is
preserved; the two retired movement-template rows reduce active debt from 326
to 324 without adding an owner exception. The Phase 1 composite below includes these supporting controls.

#### Generated C padding, shrinking, and striding (#889)

This slice routes these three DAG emitters through the private checked shape
owner. Three metadata-only C operations derive and validate the complete target
shape before exposing its extents; one affine-coordinate operation checks the
offset/step map through that same owner. Generated C preserves extent-claim order,
checks the submitted shape against the computed shape before allocation or reuse,
and uses checked unraveling and affine indices without per-element allocation.
Payload copying preserves stored bits; padding uses the existing tagged fill.
Runtime-bound shrinking retains its existing empty-range rejection, shared with
Eval; this slice does not change that operation-level admission rule. The metadata
operations and statically empty C paths retain zero-element shapes.

The bounded acceptance surface is `checked_c_affine` in debug and release,
the backend movement adoption controls, and optimized generated-C sanitizer
execution. Controls cover all nine dtypes, scalar/empty/high ranks, exact large
metadata, invalid bounds and target shapes, arithmetic overflow, and erased/late
validation. The code-derived Phase 0 coverage manifest adds the runtime suite in
both profiles without moving the foundation digest. The immutable 358-row
foundation is preserved, and retiring the three raw movement templates reduces
active debt from 324 to 321. Checked-add/multiply and bounds mutations execute
against the private metadata owner. No new inventory identity or owner exception
is admitted. The Phase 1 composite below includes these controls.

The consumer deliveries below feed the Phase 1 execution receipt/mutation oracle.
Host-only results do not establish device execution or close #893. Generated host/device descriptors and
validated Python/DLPack wrappers remain under #893/#1345; #1288 consumes those
interfaces and owns their exact discovery and authority registrations.

#### Generated C reductions and Count (#889)

Seven reduction emitter paths obtain a checked `ReductionMetadata` plan before
allocation or repurpose: materialized Sum, Count, Max (including its reduced-float
arm), Min/Prod, Argmax/Argmin, and fused Sum/Max. The opaque C plan snapshots the
input domain, selected axes, checked result metadata, and row-major leaf count.
A fused domain owns no tensor payload or unused storage strides. Empty results
have no reachable groups; empty selected domains retain their operation's identity.
Exact result shapes are checked after the existing ordered extent claims and before
submission. The plan survives input repurpose or release without retaining storage.

Sum and Count allocate tree scratch through checked runtime tensors, at the actual
accumulator representation. Scratch bytes and target projection are checked before
result allocation; each worker owns its scratch tensor and write guard. Sum keeps
#1299's adjacent-pair tree and integer finalization; Count keeps its original
row-major leaves, checked int64 pairs, and odd tails. Existing dtype rejection,
non-Sum arithmetic, and vendor-selection obligations remain separately owned.
Kernel scratch remains outside the shared planner's distinct DAG-slot bound;
using checked runtime allocation does not make scratch a planned tensor slot.

This is one shippable adoption slice because result validation, loop bounds, source
indices, and scratch capacity must agree for the same grouping. Its oracle combines
`checked_c_reduction` and the private `checked_metadata`/`metadata_compile` controls
in debug and release, generated native and sanitizer executions, existing fused,
integer-promotion and exact-tree regressions, and `count_bool_axes.ch` parity.
The code-derived Phase 0 manifest adds these commands and the grouping/index
bypass mutations. The 358 foundation identities remain unchanged; no new owner
exception is admitted. The retired Sum raw-index template leaves active debt
(314 to 313). This supports the reduction slice only: sparse/BLAS/window
consumers and the complete Phase 1 execution-receipt oracle remain open. It does
not close #889 or #893.


#### Generated C sparse loops (#889)

Gather, ScatterAdd, ScatterReplace, and ScatterElements share a checked
`SparseMetadata` domain. Hyperplane operations bind the base prefix/suffix to the
complete index shape; elementwise replacement binds each index coordinate to its
base axis. The runtime checks exact updates/result shapes and dtype, and projects
index slots and selected base indices from this domain. Snapshots retain no payload
and require no per-index allocation. Generated DAG loops and all three host sparse
summary paths check the target before allocation or reuse. Scatter copies consume
checked byte counts, and ascending update positions preserve duplicate-write order.

The seven registered C functions and their opaque plan form one adoption slice with
the sparse consumers: exact shapes, loop domains, and index maps must agree before
submission. Arithmetic algorithms and existing backend dtype admission are unchanged.
The bounded oracle combines runtime `checked_c_sparse`, private metadata and
construction/mutation controls in debug/release, backend `checked_c_sparse`, native
`checked_c_sparse_` sanitizer tests, host-summary execution/rejection tests, and
`checked_sparse_axes.ch` evaluator/C parity. Native controls cover all nine stored
payload representations on Gather/Replace/Elements, duplicate f32 ScatterAdd,
nontrailing domains, empty results, and invalid indices; the host Add summary is
executed directly from IR because it is produced by AD rather than a Surf builtin.

The Phase 0 foundation keeps all 358 identities. Replacing thirteen raw sparse
consumer owners reduces active debt from 313 to 300 without a new exception. The
code-derived coverage manifest adds these executable suites to the Phase 1
composite below.


#### Generated C BLAS submission metadata (#889)

Already selected internal BLAS nodes and host summaries obtain a `MatmulMetadata`
plan after explicit batch alignment. Its operand/result snapshots own complete
shapes, batch counts, per-matrix counts, checked matrix indices, and allocation
bounds. Required declarations and native link flags follow the selected nodes, including
when no further specialization was requested. Exact result-shape and vendor-dimension checks precede output allocation
or reuse. Reduced-float conversion scratch uses checked f32 capacities and runtime
tensor owners/write guards; one batch loop consumes checked source/result indices.
Empty results make no vendor call, and zero contraction writes dtype-zero output.

The generated translation unit binds its dimension type to the actual sgemm and
dgemm function prototypes using C11 type assertions. Accelerate's `__LAPACK_int`,
OpenBLAS's `blasint`, and Netlib's `CBLAS_INT` declarations select a signed 32- or
64-bit domain; a missing, unsigned, unsupported-width, or inconsistent declaration
fails compilation. These declarations are compiler-private, so the published
Chelis ABI stays configuration invariant. Tagged int64 dimensions are checked
against that domain before casts at the call. Exact per-matrix f32 scratch bytes
are checked even when source storage uses f16/bf16.

This is metadata adoption, not algorithm selection. [05-OP-30]'s canonical
contraction rule still prevents shape-only BLAS specialization; production host
preparation continues clearing those summaries. Direct internal node and summary
fixtures exercise the submission boundary without re-enabling a vendor shortcut.
Operand/accumulator/destination dtype choices and conversion arithmetic stay pinned
by the existing IR. The bounded oracle combines private metadata/projection and
construction controls, runtime C plan tests in debug/release, generated native
sanitizer tests, and actual/fake vendor-header width/prototype controls. It does
not prove vendor arithmetic equivalent for unrestricted inputs or complete Phase 1.
The 358-entry foundation is unchanged; five retired BLAS consumer identities reduce
active debt from 300 to 295 without an owner exception. Coverage adds the matrix
runtime suite in both profiles, delegation bypass controls, and native submission
and vendor-prototype execution. Production matrix-index and vendor-range mutations
must fail with overflow checks disabled; virtual f16 operands demonstrate a fitting
source allocation whose f32 scratch capacity overflows.


#### Generated C window geometry (#889)

`WindowMetadata` snapshots complete source/result shapes and positive trailing
window/stride lists, checks valid-padding extents, and maps a result position and
row-major leaf to a checked source index. Seven OP33 entry points expose that
immutable plan without retaining payload or tensor ownership. Shape and dtype
validation precede generated forward/gradient allocation, including same-capacity
wrong shapes. Leading empty axes have no reachable index and no unused window
product. The generated loops carry int64 positions and use no rank-sized coordinate
arrays or compiler-computed window-volume product. Forward leaves and serial
cotangent/leaf updates retain their prior order.

The bounded acceptance surface combines `checked_metadata`, `metadata_compile`,
`checked_c_window`, native `exec_compile::checked_windows_*`, existing window
emission/numerical tests, and `parity_checked_window_geometry`. Production
stride/valid-padding mutations must fail with overflow checks disabled; private
fields reject outside mutation. Runtime tests cover all nine storage dtypes and
five diagnostic selectors, invalid tags/axes/indices/targets, source write guards,
release and repurpose. Native f32 fixtures exercise four forward reducers and
four gradients, both nonempty and leading-empty, plus incorrect result/cotangent
shapes under optimized ASan/UBSan. The executable example runs Eval/C parity.

This delivery preserves the existing f32/static-window-output admission and the
existing accumulation/NaN/tie behavior. #1298 owns window arithmetic remediation;
these geometry tests do not establish its full normative contract. The 358-row
foundation stays identical, and the removed gradient coordinate template reduces
active debt from 295 to 294 without adding an owner exception. The Phase 1 composite below includes this geometry surface; Phase 2
ABI/Python/DLPack work remains separate.


#### Generated C JSON ordering scratch (#889)

Canonical JSON object ordering allocates its int64 index scratch through the
runtime tensor owner, retains one write guard during ordering, and ends/releases
it after building the ordered list. The ordering algorithm is unchanged. The
scratch count comes from the validated list length, and allocation checks its
int64 product, byte count and target capacity before the first scratch access.
Empty objects allocate a zero-count owner without entering the ordering loops.

The bounded acceptance surface is `checked_c_json_scratch` for delegation and
cleanup-tail spelling, plus `issue_1314_json_bigint`'s recursive canonical Unicode,
reordered and empty object case through Eval/C. The `issue_1314_json_bigint_ledger`
target's `json_scratch_execution_detects_skipped_cleanup` test requires the CLI's
`ownership-ledger` feature, so the runtime `chelis build` stages records the private
ownership ledger. The test links that runtime and executes the emitted ordering
helper with an explicitly released caller. Empty and two-entry objects leave zero
live owners/bytes. Output-preserving last-iteration return and omitted-owner-release
mutations must leave live ownership; omitted guard exit must fail with the active
write-guard error. The source check alone does not prove control-flow cleanup.
These execution registrations extend the code-derived Phase 0 manifest without
moving its immutable foundation. The Phase 1 composite below includes this
ownership contract; Phase 2 remains separate.

#### Generated C literal ingress (#889)

Tensor literals carry finalized tagged scalar images instead of a raw payload
array and an unchecked byte-count copy. Two exact OP33 entry points validate the
complete result rank/shape, zero dtype exemplar and element count before storage
allocation or reuse, then validate every source carrier before the first guarded
write. The write preserves each declared storage width and its exact bits; it
neither converts values nor accumulates. Mismatched IR storage/result dtypes are
rejected during emission. Empty literals submit a null array with exact zero count.

The C dimension renderer accepts only `DimInfo` leaves; its unused arithmetic
`DimExpr` renderer is removed, so those callers cannot emit unchecked products or
divisions through that path.

The bounded execution surface is `checked_c_literal` in debug/release,
`checked_c_host_metadata` delegation/bypass controls, native
`exec_compile::checked_literals_*` under optimized ASan/UBSan.
Literal tests exercise all nine storage representations, rank zero and empty
domains, exact int64 values above 2^53, floating bit patterns, invalid counts,
tags/payloads and overflow. A malformed last carrier must trap with canonical
`const`/int64 identity while every destination element remains unchanged. These
commands extend the code-derived Phase 0 manifest; removing or reordering
ingress calls is rejected by the bounded source controls.

The immutable 358-row foundation is unchanged. Removing the literal emitter's raw
storage/element-spelling templates reduces active debt from 294 to 292 without a
new identity or owner exception. The Phase 1 composite below includes this
literal contract; Phase 2 descriptors/bindings remain separate.

#### Immutable movement geometry (#889)

The C permutation, expansion/insertion, padding, shrinking and striding consumers
snapshot checked input/result metadata into one private movement plan before output
allocation or reuse. The plan projects each checked row-major index directly;
rank-sized coordinate arrays and per-element scratch are removed. Its projection
allocation is checked and cannot grow during construction. Padding iterates the
source domain and maps into the filled result; the other forms iterate the result
and map into the source. Empty domains perform no projection or payload access.

The exact eight OP33 APIs bind this metadata contract and preserve every payload
representation. Closed operation and side enums select the shape policy and
canonical diagnostic, without carrying numeric data. Plans retain no payload and
remain valid during source writes and after release/repurpose. The shared IR
`ExpansionKind` derivation distinguishes rank-preserving expansion from insertion
on verified static types under spec/10 §3.4; both the C plan selector and local dimension guard use
that form. No serialized IR or Python binding signature changes.

The bounded acceptance surface combines runtime `checked_c_movement_plans` and
private `checked_metadata` in debug/release, `metadata_compile`'s field-privacy and
production step/offset/bijection mutations, the two backend movement source suites,
`exec_compile::checked_c_movement_*` with optimized ASan/UBSan and exact stored-bit
maps, `movement_expansion_kind`, the runtime movement CLI suite, and
`parity_checked_reshape`. The CLI positive/negative pair uses an extent read from a
locally shortened tensor and checks exact `expand`/`insert` trap identities. Bare
scalar expansion extents remain outside the admitted CLI surface under #469.
The code-derived Phase 0 manifest requires these executions; its foundation
stays fixed, with no new owner exception. The explicit movement receipt
registrations and their negative controls also enter the Phase 1 composite below.
Phase 2 descriptors and bindings remain separate. This is one geometry slice because the shared plan, all
five consumers, semantic authority and executable controls must ship together.

#### Checked C shape observation and allocation (#889)

DAG metadata observation uses the existing checked extent API directly, removing
per-tensor runtime-rank shape and unused stride arrays from the C stack. Shape
operations capture their scalar extent before destination submission, so a storage
transition cannot change the logical metadata being observed. Other extent
consumers already capture their bounds before allocating or repurposing output.
The host allocate-like adapter delegates to `chelis_tensor_alloc_like`: checked
input shape and an explicit zero tagged exemplar determine independently owned,
zero-filled output at the selected representation. It allocates no C shape scratch.
No descriptor layout or Python binding changes in this slice.

The bounded oracle includes all 81 input/output dtype pairs at ranks 0, 1, 8 and 9,
empty shapes with extents above int32, active input guards, released-input
independence, invalid exemplars and output-byte overflow, in debug and release.
C execution covers an ordinary shape read and an explicit legal same-capacity
resubmission fixture; moving the observation after repurpose must fail. The fixture
does not claim that the current planner reuses a last-use input. Source controls
reject restored metadata VLAs, raw stride construction and byte multiplication.
The vmap CLI control preserves the mapped axis shift. Existing cast, movement,
elementwise and physical-slot lifetime regressions remain required supporting
coverage. The code-derived Phase 0 manifest adds these named executions and
negative controls while preserving the immutable foundation and admitting no
new inventory owner. The Phase 1 composite below includes this shape observation
contract.

## C3. One generated host/device descriptor schema

A leaf crate, `chelis-abi`, depends only on `chelis-vocab` outside the
standard library. It owns C2.2's checked descriptor metadata, a declarative
field schema, and renderers. It never owns tensor storage, frees tensor bytes,
or interprets tensor payloads. The schema generates:

1. [05-OP-31]'s exact Rust and C read/write view layouts;
2. a Rust raw device descriptor used by Python's private device-entry module;
   and
3. the corresponding internal C/HIP device declaration.

The host tensor retains its private `HeapHeader`, `TensorStorage`, one
`ShapeMetadata`, access state, and embedded write-guard composition. There is
no generated flat host record containing independently writable copies of its
rank, shape, strides, count, or capacity. The opaque [05-OP-44] handle exposes
no layout, so a C mirror of this private owner graph is neither needed nor
permitted. Public read/write views are validated projections with the exact
layout and lifetime from [05-OP-31], not another metadata authority.

Host metadata and the raw device descriptor use the same checked field
domains: opaque data pointer, dynamic shape/stride pointers, `int64` element
count and byte capacity, `int32` rank, exact dtype tag, ownership, and reserved
bytes. A raw device packet is a projection of validated metadata with a live
owner; it cannot create checked authority. Device ownership and address-space
behavior remain distinct from the host owner graph. Sharing field domains
does not require identical internal objects.

The schema macro/renderers expand inside each owning private module. Generated
raw device descriptor fields and checked metadata/storage-owner fields are
private to their owning modules; ordinary consumers receive opaque handles or
typed views. The exact public OP31 read/write transport views retain their
existing public Rust fields; a view is not a validated owner and cannot mint
descriptor authority. The checked-in C fragments are generated artifacts with
byte-for-byte freshness tests. Every published host runtime header remains
reachable from `chelis_runtime.h`. The HIP support root is staged as a complete
published closure and preprocessed only under the committed Phase-0 SDK stubs.
Its recursively discovered source-header set selects attributed files before
row extraction, so nested reached support declarations cannot be removed by a
basename filter and SDK declarations remain non-published inputs.
The recursively discovered published Metal header set currently has no
attributable ABI row and is guarded by an explicit enrollment tripwire instead
of a vacuous census lane. That tripwire uses the shared C-family lexer, so braces
inside comments, strings, characters, and raw literals cannot swallow a later
public declaration. The public-header census sees the same canonical declarations
in every declared preprocessing context.

`cargo run -p chelis-abi --example generate_headers -- --write` regenerates the
fragments; without `--write` it checks freshness. The host-view fragment lives
under `crates/chelis-abi/generated/` and is embedded into the marked generated
region of `chelis_runtime.h`, so existing standalone header staging stays valid.
The private device declaration is generated beside the HIP support header.
Neither artifact may contain a second handwritten field list.

The HIP support root requires the official `hipblas/hipblas.h` from the same
supported SDK as the linked hipBLAS library. It does not redeclare SDK types or
functions when that header is missing. Generated helpers use that SDK's actual
API: a library symbol's spelling does not establish its argument types. The
header census preprocesses the complete support root with clang on the fixed
`x86_64-unknown-linux-gnu` target, `-ffreestanding -nostdlibinc`, and the
committed Phase-0 SDK stubs. Every linemarker path is canonicalized and must
remain inside the staged published closure, a declared stub root, or clang's
resource headers. Hardware acceptance separately records the installed header,
library, and executed numeric behavior. The environment contract and its
outstanding evidence live in `docs/local_hip_environment.md`.

The opaque [05-OP-33] `chelis_metadata_plan` C adapter owns a closed contiguous
or strided metadata variant from `chelis-abi`. Tagged rank/extent/stride and
exemplar inputs follow the existing shape-reduction-plan ingress convention.
Its immutable projections supply the generated device packet; packet helpers
never compute a second product or repair strides after construction. The device
owner retains the plan and proves the supplied allocation capacity. Its caller
retains the library until that library's finalizer has released the metadata and
owned device storage.
The metadata plan allocates no tensor payload and cannot prove a foreign
allocation's physical bounds merely from its declared capacity.

The device owner is defined only in the separately compiled
`crates/chelis-backend-hip/runtime/chelis_device_owner.cpp`; the published
`chelis_device_owner.h` declares its opaque handle and [05-OP-33]'s eight exact
operations. The support root includes that header and the generated packet
fragment. No implementation source is included into a published header, and the
header census receives no C++ privacy exemption. The opaque handle directly
owns its plan and contains the packet it observes; a packet pointer is never
cast back to an owner. Python retains the loaded library through owner release. The private owner also
records actual HIP device identity; its exact observation is the ninth device
operation. Nonempty pointer attributes and current context must agree, and the
finalizer selects/restores the owner device. Python never assigns output device
identity from input zero. Explicit completion of device copies protects source
owners and host guards even when an SDK transfer can complete asynchronously.

Storage-slot lifetime remains the proof for temporary views: each view owns its
metadata while borrowing an input or slot retained until its last use. Cleanup
releases views before slots. Every escaping output calls the independent clone
operation, which creates a contiguous plan and materializes logical order before
any source release. This design adds no shared-storage refcount. The checked
`byte_offset` projection keeps coordinate/stride/width arithmetic in the shared
metadata authority; companion copying derives a nonempty element width from
checked logical bytes/count and never introduces a dtype-width table. Empty
transfers do not divide by count or access data.

Both HIP artifact paths stage `chelis_device_owner.cpp`, its public header and
the generated descriptor alongside existing runtime headers. The companion is a
separate compiler input, linked with the same artifact's metadata-plan runtime
archive. Python's all-C/C++ artifact build includes it exactly once. CLI builds,
test staging, installed packages, source closure, runtime representation inventory
and compiler input/cache identity include its bytes and generated dependencies.
Missing companion/header/runtime symbols are prerequisite failures. ABI2 admission
rejects ABI1 before these files or a library are consumed; DLPack and unrelated
wire/cache versions keep their own format contracts. Tests separately exercise
CPU SDK-fixture copy/lifetime behavior, whole-root HIP and Metal header discovery,
actual materialization/compile/link, and the unresolved real HIP hardware gate.
Metal retains its host tensor ABI and private Objective-C buffer ownership rather
than adopting a ROCm pointer packet. Its support root and complete local include
closure remain mandatory census inputs, with the generated OP31 host views
coming from the same runtime header and authority as HIP host transfers.

Python deletes `CHELIS_MAX_DIM`, `[i32; 8]`, int32 `size`/`storage_size`, and
`*mut f32` from its device carrier. The HIP support header deletes its matching
fixed arrays, `int` products, scalar special case that rewrites zero count to
one, and `float *` view parameter. Rank, shape, count, and capacity cross both
boundaries at the numbered-spec domains.

The public metadata-callable additions are exactly the eleven
`chelis_metadata_plan` identities in [05-OP-33]'s normative registry. Their
executable capacity registrations and positive/negative controls are required
in the same cutover; neither the generated packet nor this plan grants numeric
authority. Any further public callable or field requires its owning [05-OP-N]
rule and exact registration before implementation.

This delivery extends the code-derived Phase 0 source universe without changing
its immutable 358-row foundation. The universe grows from 74 to 80 files: the
generated device packet, opaque owner header and C++ companion, Python DLPack
and native owner modules, and the standalone generated host-view header. The
C/C++ scanner uses one fixed C++17 lane with the committed HIP/hipBLAS and
standard-library fixtures; adding a `.cpp` under a backend runtime root is
covered by the existing unregistered-source mutation.

Exactly 35 new scanner rows are final forms rather than transition debt: the 13
field/carrier observations of the generated packet, nine private opaque-owner
operations, four immutable metadata-plan shape/stride projections, six validated
Python ingress/owner observations, and three HIP emitter projections that consume
the closed dtype contract. The freeze names every path, kind, and owner; a rename,
new owner, or additional row remains unclassified. A typed cast of an existing
descriptor `data` access keeps that access's frozen identity and records the cast
in its sample instead of manufacturing a second debt class. Forty-six retired
fixed-rank, narrow, handwritten descriptor and raw-owner identities are deleted
from the active ledger, leaving 244 active Phase 3/4 rows. Generated-layout
freshness, metadata/device execution, binding execution, and the backend-header
capacity census are the executable authority for these final forms.

Typed comparison, bool-only logical operations, and stored-bit `where` add six
closed backend scanner owners as final typed-lane forms rather than transition
debt: the element-spelling owners
`CEmitter::{emit_compare,emit_logical,emit_where}`,
the `emit_where` load/store template, `HipEmitter::comparison_c_type`,
and `REDUCED_FLOAT_COMPARISON_HELPERS`. Comparison element spelling and reduced
float decoding are governed by [05-OP-36], bool-only logical spelling by
[05-OP-26..28], and stored-bit selection by [05-OP-33]. No owner accepts a bare
runtime dtype id or creates a public numeric carrier. The Phase 0 manifest binds
these exact owner identities to the IR semantic/AD suite, compiled C all-dtype
execution, HIP structural admission, and the ignored real-HIP exact-bit matrix.
A renamed or additional owner remains unclassified until it independently
supplies the same final-form authority and execution contract.

The #1281 exact C reduction cutover adds two more closed
`backend-element-spelling` owners as final forms:
`CEmitter::emit_mean_nonempty_guard` and
`CEmitter::emit_reduce_extreme`. The former observes the lowered divisor in
its declared storage/arithmetic width and enforces [05-OP-11]'s runtime-empty
Domain trap before division. The latter implements [05-OP-12..13] selection at
the declared arithmetic width while copying the selected source storage bits,
including first-NaN and equal-value behavior. Neither owner accepts a raw dtype
identifier or creates a public numeric carrier. The Phase 0 manifest binds both
exact identities to the complete compiled-C
`issue_1281_exact_reductions` suite; Phase 1 freezes that suite's current ten
test identities, including every admitted storage width and runtime-empty
negative controls. Renaming or splitting either owner requires a new
final-authority classification and execution contract rather than transition
debt.

Integer-weight gradients register four exact `backend-element-spelling`
final forms: `integer_float::integer_to_float_bits`, HIP
`cast_integer_to_float`, and Metal `Emitter::{emit_integer_float_cast,emit_expand}`.
The shared helper constructs target IEEE bits by rounding once in integer space
under [04-NUM-14]; the device cast owners select exact signed source storage and
target width from `Prim`. No float intermediary or raw dtype identifier carries
the operand. Expand checks axis insertion and copies
stored elements of the unchanged declared dtype under [05-OP-49]. The Phase 0
manifest binds these identities to the shared generator's native C/UBSan
exact-bit and invalid-target tests, and HIP/Metal integer-Abs lowering and
rejection controls. Its manual Metal `integer_abs_guard` command executes exact
casts, Grad, expansion, empty shapes, and nonempty/MIN trap twins; the registered
HIP `integer_abs` device command remains a hardware gate, distinct from executed
CPU kernel shims. Constant-fill load/store retirement follows `ScalarValue` /
`ElementRef` dispatch; binary/reduction launch-template retirements follow the
shared buffer-binding emitter, which transports opaque buffers rather than
reading tensor elements. These registrations and shrink-only retirements do not
change the frozen foundation or mutation contract.

`NUMERIC_DEVICE_HELPERS` is a separate numeric final-form owner for the
`uniform_like` sampler governed by [05-OP-8]. Splitting it from the common HIP
device helpers does not transfer it into the typed-nonnumeric cohort. Its exact
affine and per-dtype rounding remain bound to the existing ignored real-HIP
`gpu_correctness` execution lane.

`HostEmitter::assign_uniform_like` joins that cohort with two rows, a
`backend-element-spelling` and a `load-store-template`, under the same
[05-OP-8] authority (chelis#2120). It is the C host lane's draw, reached when a
`uniform_like` template is not constant-foldable and the binding therefore does
not route to the tensor-DAG lane. Its execution contract is a different lane
from the device sampler's: the compiled-C parity tests in
`crates/chelis-cli/tests/issue_2120_uniform_like_runtime_template.rs`, which
build, link, and run each program and compare it against `chelis eval` on f32
bits. Those tests cover every active float dtype, because [05-OP-8] admits all
four and a runtime-derived template is precisely the shape the DAG lane does
not serve. The two rows are the seams the Phase 0 scanner observes in that
owner; neither is gratuitous, and dropping either reproduces an unclassified
inventory hit.

Runtime List-map capture and ordered cotangent summation add two exact C
numeric-operation final forms under [05-OP-55] and spec/06 section 2.4:
`CEmitter::emit_list_map_capture` has a `load-store-template`, and
`CEmitter::emit_ordered_adjoint_sum` has a `backend-element-spelling`. The
capture copies the source scalar's declared element bits into the actual
invocation rows; the sum loads those rows in invocation and consumer order,
then combines them at the declared float width. Both operate on private
typed tensor storage and introduce no public bare-number channel. The Phase 0
execution legs bind these exact owners to native C length-mismatch rejection
and compiled Eval/C parity for the ordered tree, each float-width rounding,
inactive and empty rows, and a false forward range claim. Renaming an owner
or introducing another spelling remains an unclassified scanner hit until it
has its own semantic authority and execution contract.

HIP exact direct arithmetic adds four closed typed-lane final forms:
`binary_elementwise_typed` and `fused_reduced_step_lines` each supply one
`backend-element-spelling`, while `binary_extrema_reduced` and
`extrema_adjoint_reduced` each supply one `load-store-template`. They are
governed by the direct `sub` and extrema atoms and preserve the tagged lane
contract: f32/f64 arithmetic NaNs finalize to their canonical positive quiet
NaNs, f16/bf16 arithmetic finalizes once at stored width, extrema compare
decoded values but copy selected storage bits, and extrema adjoints route the
complete stored cotangent or exact positive zero. The Phase 0 and frozen Phase
1 manifests bind these exact owners to the five named `codegen_structure`
direct-arithmetic controls and the registered ignored real-HIP
direct-arithmetic command.
Another owner, spelling, or template remains unclassified; Metal remains the
separate chelis#2338 capability gap rather than inheriting these HIP forms.

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
between foreign calls; [05-OP-44]'s entry-borrow caller preconditions continue
to own those facts. This preserves the public boundary without manufacturing a Rust
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

The native entry's private `CompiledInputs` retains the admitted lane and an
immutable `Arc<ShapeBindings>`; `execute_checked` carries that same owner into
`RawOutputs` alongside all output owners and retained Python inputs. Only
`ShapeBindings::admit` constructs its private name-to-int64 map from explicit
manifest literals and already checked input metadata. Spec/11 §1.2 and spec/04
§4.1 govern the equalities: wildcard `*` axes never enter the map. Output
adoption checks the actual descriptor against those bindings before constructing
`ValidatedTensor`. No name parser, symbolic-expression evaluator, or guessed
bijection against the code generator's interface-witness list participates.

The private device handle retains its artifact `Library` through the exact
opaque-owner finalizer. Input admission proves pointer offset and remaining
capacity against retained framework storage before importing its checked raw
packet. Each output's device comes from its own opaque owner and agrees with
the actual HIP current device. DLPack's validated synchronization request uses
the retained library's successful device barrier before capsule construction;
the protocol's explicit no-synchronization request remains distinct.

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

Until Phase 5, a hit may instead match one exact identity in the
integrity-digested Phase 0 foundation, with one owning deletion phase, and
remain in the separately generated active-debt list. The foundation is
append-only while the active list is shrink-only: regeneration preserves
retired foundation identities, and an addition, rename, signature change,
relocation, or reclassification changes the foundation digest and requires
review. At Phase 5 the debt set must be empty. This is an explicit migration
ledger, not an allow-list or a final authority class.

Anything neither final nor an unchanged frozen debt identity fails. The
inventory includes descriptor fields, `data` access, pointer casts,
width/arithmetic matches, `normalized_key` arithmetic, fixed-rank arrays,
narrow metadata fields, backend element spellings, load/store templates, and
the dtype contract itself: the `Repr` and `RuntimeDType` variants and the
element bindings that tie a Rust marker to a runtime tag. A dtype added
without its complete contract is the mutation this last enumerator exists to
catch, so each variant and each binding is its own row.
A final-form exception may name a private owner function and reason, but a
stale or unmatched entry fails and an issue citation does not authorize a raw
path.

The length-aware UTF-8 string boundary is one such exact final form. The C
backend owner `runtime_string_literal` emits fixed-width byte escapes and an
explicit byte count, and the public runtime owner `chelis_string_from_utf8`
accepts that counted `uint8_t` sequence before constructing a validated Chelis
string. Those bytes are encoded text, not tensor elements or an untyped numeric
carrier. Only those two scanner identities receive this disposition; adjacent
string accessors, constructors, and backend emitters remain unclassified unless
they independently satisfy a final form.

### The inventory's universe is a file list

The inventory's completeness claim is over an explicit, reviewed list of the
repository files that can carry a seam, held in the oracle as
`INVENTORY_SOURCES`. The oracle proves that list still equals the on-disk
contents of its declared roots, so a new file fails until someone registers it,
and it reads the filesystem rather than the git index because cargo compiles
what is on disk.

Stating the claim over a *language* instead would not be dischargeable: a
reviewer can always name one more construct. Stated over a file list it is
decidable, and every source in it is read by a real parser for its own
language: the Rust files with `syn`, and the C and Objective-C headers through
clang's front end (`clang -fsyntax-only -Xclang -ast-dump=json`). A seam's
owner is the declaration that encloses it, and producing that owner means
parsing declarations; a token walk cannot do it, because C declaration form
(tagged aggregates, unions, macro-typed declarators, multi-declarator lists,
attributes, K&R definitions, keywords inside string literals) is a grammar,
and each form a hand-written walk left unmodelled dropped a seam silently. The
compiler supplies the declarations; the reader classifies their type spellings
with the capacity census's closed type-word lists, so a C type word keeps one
classification authority in the repository, and a spelling neither list names
still fails the scan.

The parse is host-independent by construction. Each header is read under a
fixed target lane (C on `x86_64-unknown-linux-gnu`; `chelis_metal_runtime.h`
as Objective-C on `x86_64-apple-macosx14.0`), with `-ffreestanding
-nostdlibinc`, a committed stub SDK under `crates/chelis-repr-inventory/sdk-stubs/`
standing in for libc, the HIP SDK, hipBLAS, the BLAS headers, SLEEF, the
Apple frameworks, and the SIMD intrinsics headers, and a scrubbed environment,
so Linux CI, macOS CI, Devenv,
and a workstation see the same preprocessed text and produce the same rows.
The stub SDK is deliberately minimal: a runtime header that starts using an
SDK symbol the stub does not declare fails the scan until the stub declares
it.

A compiler reads one preprocessing configuration at a time, so a lane also
names the closed set of configurations its headers are parsed under (the
published headers scalar, with the AVX2 arm, with the NEON arm, with the Apple
arms, with the SLEEF arm, and with the OpenBLAS arm; the HIP header with and
without `NDEBUG` and with the hipBLAS header present; the Metal header with
and without `NDEBUG`), and a header's row set is the union over them. That set is checked rather than trusted: a marker planted at
the start of every conditional arm lets the preprocessor itself report which
arms each configuration keeps, and an arm that carries code and that no
configuration keeps fails the scan naming its directive. An arm carries code
when it holds a declaration, a quoted `#include` (a repository file), or a
`#define` with a body; an angle `#include` (an SDK or compiler header the
universe supplies), an include guard's bodiless `#define`, an `#error`, a
`#pragma`, and a linkage-specification brace carry none and need no
configuration. Two more
rules close the universe: an include that resolves, canonically, to anything
but the header itself, the stub SDK, the compiler's own headers, or the
published include directory fails the scan, and the type-word rule applies to
a block parameter and in a cast, compound literal, or `sizeof` operand exactly
as in a declaration. A `sizeof` in an array bound or bit-field width is a
width seam of the declaration that spells it. Cargo build scripts under the
inventoried crates are roots too, since cargo compiles them like any other
source.

A row's identity is its kind, its path, and its owning declaration. Reformatting
a literal inside a function does not move the freeze; adding a function, field,
or dtype variant that carries a seam does. That is the granularity Phases 3 and
4 delete at, since those phases remove declarations and call sites rather than
individual bytes.

Phase 0 inventories both backend runtime headers as structural seam sources.
Phase 2 separately inventories the HIP support root's public ABI after its
device packet moves onto the tagged carrier. The recursively discovered
published Metal header set remains in this structural inventory but not in the
capacity baseline while it exports only `static inline` definitions; its Phase
2 enrollment tripwire fails on the first attributable public declaration in any
published `.h`.

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
- register a source file's seam without registering the file;
- replace the exact arbitrary-precision product with primitive wrapping
  arithmetic;
- recover capacity through a direct or aliased legacy capacity carrier;
- use an arithmetic type spelling no vocabulary classifies;
- change a Bool8 lane spelling to `float`; and
- make a Bool8 kernel store `1.0f` or omit the device failure flag.

Every mutation must plant a real seam of its class rather than a token pattern,
and must make the phase command fail for its exact intended reason. A witness
that only a pattern-matcher would catch proves nothing about a structural
classifier. The same command also executes the scanner's own positive and
negative suite. A source scan is the completeness guard; compile-fail and
execution tests prove its sanctioned replacements work.

---

# Part II — process rules

## B1. Freeze points

- Phase 0 freezes the immutable `foundation_rows`, including each identity's
  owning deletion phase, and `source_inventory.mutations`. Each frozen mutation
  row binds its stable witness ID, exact implementation digest, target source
  path, expected seam kind, required owners, expected failure code and reason,
  and required command. Later phases may reduce raw
  hits but may not add an exception or weaken a witness without a reviewed
  freeze move. The separately stored active-debt list sits outside the digest
  so it can shrink, while regeneration preserves retired foundation identities
  and the reviewed mutation rows. An identity in the frozen foundation but
  absent from the prior active-debt list is retired; regeneration rejects its
  reappearance rather than silently restoring it. A genuinely new identity
  outside the prior foundation may still be emitted with a changed digest for
  review. `coverage_manifest()` remains code-derived configuration rather than
  a persisted baseline field: beyond the frozen mutation contract, it adds the
  source universe, release reproducers, hardware probes, counts, and ordinary execution
  configuration. At runtime the oracle verifies that live probes match the
  frozen mutation rows, verifies that the richer manifest is the exact
  projection of the current configuration, and executes every non-hardware
  mutation and reproducer. Changes only to live-derived reproducers, hardware
  probes, counts, or ordinary configuration do not move the freeze.
- Phase 1 freezes `DTypeContract`, sealed element markers, exact capacity keys,
  checked finite-count types, and a required test-identity floor. Later phases
  consume the contracts without parallel tables. Tests newly selected by an
  existing command execute and appear as additions without weakening or moving
  that floor.
- Phase 2 freezes the generated host/device schema, public-layout parity, and a
  required test-identity floor. Later field changes amend [05-OP-31] or
  [05-OP-44] first when public, regenerate every consumer, and move the
  architectural freeze in one change. Newly selected tests execute and are
  reported without requiring a floor amendment.
- Phase 3 freezes typed runtime ingress and the private data field. Reopening a
  raw accessor is a design change, not a local optimization.
- Phase 4 freezes the typed lane renderer and all-lanes representation probes.
  A new dtype or lane cannot land without extending both.
- Phase 5 freezes the composite oracle and closure receipts. No individual
  child test substitutes for it.

Moving an architectural freeze, adding or changing a Phase 0 foundation row or
frozen mutation row, or removing, renaming, or intentionally replacing a
required test identity requires changing this document, the owning numbered
spec when semantics move, and the oracle's integrity digest. A foundation
change also requires a mutation that would have accepted the forbidden
behavior. Adding a test already selected by a frozen command does not move the
required floor.

A Phase 0 source-universe, final-form, reproducer, or hardware registration that
does not add or change a foundation row or frozen mutation row does not move the
digest and does not require a B1 amendment paragraph. Its owning change still
updates code, focused positive and negative tests, and current documentation.
Adding, removing, renaming, reimplementing, retargeting, or changing the
expected seam kind, required owners, failure or command of a mutation does move
the freeze. The runtime
manifest/configuration equality check, source closure, expected-failure checks,
and execution of every mutation and reproducer remain mandatory.

The schema 7 mutation-contract amendment freezes the existing full mutation
manifest projection, adding `path`, `expected_kind`, and `expected_owners` to
each reviewed row. These fields select the mutated source, the seam kind its
rejection must identify, and every owner the rejection must name; changing
them changes the witness's rejection obligation even when its implementation
is unchanged. The frozen owner list retains the live manifest's exact order
and duplicates; execution still requires the set of named owners. This
strengthening preserves all foundation identities, deletion phases, active
debt, mutation implementations, failure expectations, and commands. It adds
no representation exception or negative witness and changes no numbered
language semantics. Live reproducers, hardware probes, source-universe
configuration, and counts remain outside the freeze.

The checked-extent staged plan registers `chelis-ir/src/host/staged.rs` in
that source universe. It composes existing tagged values and DAG carriers,
without adding a representation seam. Borrowing its checked program changes
seven existing scanner-qualified owners from `LowerCtx::method` to
`LowerCtx < 'program >::method`. This is an exact one-to-one owner rename:
each row retains its kind, source path and deletion phase. The foundation digest
includes those seven successor names; the additional source is code-derived
configuration. The existing
unregistered-source, unregistered-subdirectory, direct-data-access and
normalized-key-arithmetic mutations continue to reject new debt; the rename
adds no exception or detector admission rule.

The fixed-control host transport extends that same Phase 0 freeze with three
closed, scanner-visible owners: the single shared C dropout sampler prelude,
its private host-helper execution witness, and the inherited RNG load/store in
`CEmitter::emit_preplanned`. The immutable foundation moves from 358 to 361
rows and active debt from 244 to 247 after the generated-ABI consolidation;
no classifier, source-universe rule, or
deletion phase is weakened. The digest binds those exact identities. The
existing backend-element-spelling and load-store-template controlled mutations
remain the closed-world negative witnesses: an additional spelling or state
template still fails as an unclassified identity. This amendment changes no
public C descriptor, dtype tag, width, or numbered representation semantics.

The staged fixed-control evaluator composes those existing owners without a
new numeric carrier. Its opaque companion retains exact partition mappings
and one invocation's Random keys/counters; host sources and checked numeric
segments execute in their original order. The companion is not serialized,
and does not change the public legacy stage, kernel, or wire structures.

The opt-in native Random observer registers
`chelis-backend-c/src/random_observer.rs` in the Phase 0 source universe:
84 sources (73 Rust, eleven C/C++/Objective-C). Its private invocation parameter
and C state-observation prelude add two scanner-visible Phase 4 owners:
`backend-element-spelling` at `host_emit.rs::emit_function`, and
`load-store-template` at `random_observer.rs::SUPPORT`. The foundation therefore
extends from 361 to 363 rows and active debt from 247 to 249; every prior row,
classifier rule and mutation implementation remains unchanged. The foundation
digest binds those exact additional owners; the source count remains
code-derived configuration, with no exemption for the opt-in feature. Existing
backend-element-spelling, load-store-template, unregistered-source and
subdirectory closure mutations remain the negative witnesses. This amendment
changes no public descriptor, dtype, width or numbered
representation semantics.

Complete signature-entry planning registers two typed modules in the source
universe: `chelis-ir/src/host/signature_entry.rs` and
`chelis-backend-c/src/host_emit/entry.rs`. They compose existing DAG witnesses
and project already discharged helper guards. The structural scan finds no new
representation seam; the universe contains 86 sources (75 Rust and eleven
C/C++/Objective-C), with no foundation or mutation change.

Invocation-local literal result claims extend the Phase 0 foundation with two
`load-store-template` owners in `chelis-backend-c/src/host_emit.rs`:
`HostResultClaim::frame_lines` emits the immutable axis/extent pairs and private
frame, and `append_host_result_claim_support` reads those pairs at the selected
producer. These are host metadata templates, with no tensor element access or
public ABI change. Both retain Phase 4 as their deletion owner. The amendment
preserves every prior foundation identity, classifier rule, source-universe
rule and frozen mutation. The existing load-store-template mutation must still
reject an additional unregistered template; the selected-result eval/C corpus
checks that the admitted frames execute at the producer with the required
primitive attribution and effect order. No dtype, width or numbered semantic
contract changes.

Selected-result aggregate provenance adds two exact private metadata final
forms in `chelis-backend-c/src/host_emit.rs`:
`append_host_result_claim_checks` compares declared axes with tensor shape
metadata, and `append_host_result_interface_origin_support` traverses typed
List, Tuple, ADT and Option handles to rebuild per-field `load` origins at a
genuine interface. Neither owner reads tensor element storage or changes a
public carrier or ABI, so both are final authorities rather than Phase 4
transition debt. Exact path/kind/owner registration, wrong-path/kind/owner
negatives and the existing unregistered load/store mutation preserve the
closed inventory. The Phase 0 execution leg binds them to C claim failures,
aggregate and Option projection, interface ingress and repeated public-call
arena lifetime. Nested `Cons` execution additionally requires an O(1)
invocation-arena suffix view before both cloning and consuming List skips, so
pattern lowering and direct immutable `skip` preserve the selected child
without copying each remaining origin array or reading a consumed payload.
The frozen foundation and
active-debt rows therefore do not
grow, and their reviewed digest stays fixed; only the code-derived final-form
and execution manifest changes.

The captured activation-claim comparison adds the private
`load-store-template` owner, `CEmitter::emit_runtime_dim_sites` in
`chelis-backend-c/src/emit.rs`. It reads the captured int64 witness before the
producer guard and retains the existing tagged tensor storage contract. The
amendment preserves every previous identity, deletion phase and frozen
mutation; it adds no public ABI or numeric carrier exception. The
load-store-template mutations continue to reject any unregistered owner. The
combined Phase 0 foundation contains 366 rows, including 252 active-debt rows.

Pure-helper result claims add the private
`CEmitter::emit_inherited_result_guards` load/store owner and move the
evaluator's existing path-random-counter arithmetic into
`eval_tensor_internal_with_result_claims`. The append-only foundation extends
from 366 to 368 rows; replacement of the old evaluator owner plus the new
emitter owner moves active debt from 252 to 253. Both remain implementation
internals with no public ABI or carrier change, and the existing mutations
continue to reject an unregistered successor.

The random-key element dtype (chelis#2413) adds runtime dtype `Key = 9`, whose
representation `Repr::Word64` is one opaque 64-bit word with no arithmetic
representation, and its sealed storage marker `KeyWord`. The append-only
foundation extends from 368 to 372 rows, and active debt from 237 to 241, with
four `dtype-contract` owners: `RuntimeDType::Key` and `Repr::Word64` in
`chelis-vocab/src/lib.rs`, and `ElementStorage for KeyWord` and
`private :: Sealed for KeyWord` in `chelis-runtime/src/element.rs`. The
numbered semantics move with them: spec/04 §1.1 adds `key`, and [05-OP-31]
adds `CHELIS_DTYPE_KEY = 9` as a tensor-only tag that no `chelis_scalar`
carries. The two element owners enter as transition debt, not as final forms
beside `Bool8`'s, because the final-form list is oracle configuration this
change does not edit; promoting them is a classification for the #893 owner.
No classifier, final-form list, source-universe rule or deletion phase changes.
One frozen mutation is reimplemented: `phase0.mutate_incomplete_dtype` anchored
on the text of the last variant, `I16 = 8`, which appending `Key` removed, so
the witness could no longer run. It now anchors on the `RuntimeDType`
declaration and inserts its unregistered variant before the closing brace, so
it still follows the last variant and no longer drifts when a dtype is
appended. Its path, seam kind, owners, failure and command are unchanged; its
implementation digest and the freeze digest move. The incomplete-dtype,
incomplete-arithmetic and element-binding mutations remain the negative
witnesses: a dtype or storage marker without its complete registration still
fails.

The carried runtime (chelis#1354) registers `chelis-runtime/src/public_headers.rs`
in the Phase 0 source universe: 87 sources (76 Rust and eleven
C/C++/Objective-C). It holds the public runtime header texts as `include_str!`
constants that the CLI stages beside the carried archive. The structural scan
finds no new representation seam, with no foundation, classifier or mutation
change.

Development runtime freshness (chelis#1354) registers `chelis-runtime/build.rs`
and `chelis-runtime/src/build_record.rs`: 89 sources (78 Rust and eleven
C/C++/Objective-C). The build script hashes the runtime's declared source
inputs into a text record, and the module exposes that record as an
`include_str!` constant. The structural scan finds no new representation seam,
with no foundation, classifier or mutation change.

## B2. Invariants at every phase boundary

1. The public C ABI remains [05-OP-31]/[05-OP-44]-exact and
   configuration-invariant; the seal rides the opaque handle.
2. A correct landed receipt does not regress while its structural replacement
   is being built.
3. No phase introduces a second width, arithmetic, dtype-tag, or ABI field
   authority, even temporarily.
4. Debug/release and source/device paths have the same validation disposition.
5. Zero extents mean zero elements; rank zero means one element.
6. Equal capacity never authorizes reuse across unequal `Repr` values.
7. Observable foreign metadata validation happens before the first read,
   write, copy, render, or ownership action; no Rust reference is formed from
   a foreign carrier, and [05-OP-44]'s allocation/lifetime/synchronization
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
manifest in C6, over the frozen source list; a structural seam scanner with its
own positive and negative suite; release-profile reproducers for exact capacity
collision, count/byte overflow, zero extents, and malformed foreign metadata,
including a planner-level [#888] witness that shows the collision reaching slot
reuse rather than only key equality; one detection mutation per classifier plus
fail-closed controls for an unregistered source file and an unclassified
arithmetic spelling; source-only and hardware probe harnesses; all landed
receipts as positive controls.

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


### Phase 1 consumer and execution receipt contract

The Phase 1 implementation is `scripts/runtime_representation_phase1.py`, invoked
through the command above. `runtime_representation_phase1_tests.json` freezes
required test-identity floors and commands; its reviewed digest is in the
implementation. Removing, renaming, or intentionally replacing a required
identity requires a reviewed manifest amendment. A test newly selected by an
existing command is an addition: it executes and is reported without changing
the required floor. Neither a previous receipt nor a regenerated selection is an
acceptance input.

The integer-unary typed-lane amendment retains two inherited Phase 0 execution
legs in the Phase 1 manifest: integer-to-float finalization freezes two native
C/UBSan positive and invalid-target controls; integer device lowering freezes
four HIP and seven Metal admission, exact-storage, shape, and rejection
controls. The [05-OP-55] List-map C owners add two further legs: one native C
column-length rejection and four compiled Eval/C cotangent controls for
ordered consumer accumulation, stored-width pair rounding, inactive/empty
rows, and a false forward range claim. Fresh nextest listings established the
exact binary/test identities. The manifest retains every earlier command,
required identity floor, Python floor, native control, and planner mutation;
its reviewed digest binds the four additive commands and 18 identities.
Hardware-only HIP and Metal execution remains registered separately and cannot
be counted by these active-test receipts. These amendments do not change the
Phase 0 foundation, mutation contract, or digest. A passing selection receipt
still requires actual execution and artifact verification.

The host/C consumer audit follows the representation owners and submissions:

| Consumer mechanism | Checked authority | Executed boundary and negative controls |
| --- | --- | --- |
| Owned allocation, borrowed ingress, repurpose, read/write views | Private `ShapeMetadata`, `ElementCount`, `ByteCount`, `AllocationBytes`; immutable metadata replaced atomically | `checked_metadata`, `metadata_compile`, `exact_tagged_c_abi`, `op33_tensor_validation`, `tensor_repurpose`, `tensor_write_guard`, both profiles |
| Host byte copy, fills, reshape and indexed tensor movement | `copy_bytes(AllocationBytes)`, validated tensor metadata/index ranges and checked iteration domains | Padding, empty-domain, dtype-domain, checked C metadata/indexing/affine and movement boundary suites, both profiles |
| Generated C entry/elementwise/cast/gather/scatter/concat/stack/arange loops | Checked runtime counts and indices, validated read/write views; complete result checks before submission | Named C source-delegation controls, optimized native sanitizer matrices, dtype dispatch/reuse and CLI cast cases |
| C reductions, Count, sparse, BLAS and windows | Private checked plan domains, physical scratch projections and vendor-width validation | Each registered runtime contract in both profiles; actual emitted native bypass/ownership controls and executable examples |
| C literal, JSON ordering and snapshot/host shape scratch | Tagged literal preflight, checked tensor scratch owner, descriptor observations before submission, checked allocation-like API | All-dtype literal/shape native controls, JSON cleanup ledger mutation, shape-capture repurpose mutation and vmap axis case |
| Shared C/HIP storage planning | Private exact `CapacityKey` plus exact `Repr`; opaque linear reuse proof | Key and adapter collision/equivalence/domain tests, all 81 source/result representations through eligible expired slots, private storage proofs, both profiles |
| Metal storage planning | Typed no-reuse projection; no reusable capability is admitted | `never_reuse_boundary` in both profiles; this is planning evidence, not device execution |
| Representation vocabulary and runtime element bindings | Closed `DTypeContract` and sealed storage/arithmetic registrations | Positive external consumers, all nine representations and compile-failure/production mutations, both profiles |

Allocation and byte-copy submission signatures consume the private checked
values; public metadata construction cannot opt out of their validation. The
C codegen consumers receive validated counts, byte counts or metadata plans,
and the frozen source inventory continues to reject new arithmetic owners and
representation seams. The table names the acceptance mechanisms; it does not
claim that a finite example matrix enumerates every tensor value or shape.
Element-pointer privacy and backend element-spelling debt remain Phase 3/4,
and generated descriptor/Python/DLPack adoption remains Phase 2.

The oracle first checks clean committed source bytes/modes, executes its Python
framework controls, validates the Phase 0 inventory and rejects all its live
production seam mutations. Each Rust leg then builds and lists current tests,
requires every nonempty frozen identity while accepting current additions,
hashes every listed executable, removes any earlier JUnit output, and executes
the complete current selection with a dedicated nextest profile and zero
retries. Missing, renamed, duplicate, ignored, skipped, failing or unexecuted
required identities block. Added identities also block if ignored, skipped,
failing or unexecuted. Native failure controls and production mutation probes
retain exact one-for-one selection rather than the cohort-floor rule. The
isolated JUnit path prevents nested CLI tests from replacing the parent receipt.
Receipt schema 2 records `required`, current `selected`, exact `additions`, and
one passing `executed` row per current identity alongside the command, source
identity and executable digests.

Native execution links the runtime each consumer's own Cargo build carries. The
Rust harnesses stage it through `chelis-runtime-bundle`, and `chelis build`
stages it beside its output ([spec/08 §2.1](../08-backends.md)). Both verify the
written archive against the carried digest and link it by exact path, so no leg
takes a runtime from the target directory or the environment, and none can be
substituted. Without `CHELIS_OWNERSHIP_LEDGER_PATH` the ledger records nothing.
The one leg that sets it, the JSON ownership-ledger mutation test, builds `chelis`
with its `ownership-ledger` feature, so the runtime `chelis build` stages is the
instrumented one. The CLI-staged consumer's control names a directory holding an
empty archive as `CHELIS_RUNTIME_DIR` and must fail with the CLI's rejection before
linking: honoring the directory would fail at the linker, and ignoring it would
pass. Five missing-compiler controls prevent native compile checks from
returning early. Staging, freshness and persisted-artifact admission
negatives live with `chelis-runtime-bundle` and `chelis-python`; they replace
the seven empty-archive controls this runner held while it pinned an archive.
Selected library native cases have no availability exits: parallelism follows
the configured platform toolchain, and the canonical matmul case executes with
the BLAS hint without claiming a vendor call. Reduced-float BLAS cases use the
platform toolchain and must execute. Python fixtures isolate their target
configuration and prove that an inherited target's execution evidence survives.

Two additional optimized production mutations erase the shared planner's exact
representation or exact capacity conjunct. Their named behavioral assertions
must fail, the source is restored byte for byte, and the same cases must then
pass after rebuilding. Existing private metadata mutations independently weaken
extent/count/byte/stride/capacity/target/scratch checks. The complete oracle never
accepts `--skip-mutations` or `--regenerate` as Phase 1 success.

The greater-than-8-GiB allocation test remains the explicitly excluded #1112
manual gate. It is recorded as unexecuted, while exact wide metadata and target
projection are exercised without requesting that allocation. Capacity-key public
compile-fail doctests remain a supporting obligation. No Phase 2 device or native
Python boundary is admitted through this host/C receipt.

The hosted `runtime-representation` nightly/manual worker and optional full/local
gates invoke Phase 1; `--fast` remains the pre-push stage and does not run this
composite. Phase 0 is still directly runnable. A clean local Phase 1 pass is
supporting evidence until candidate-head hosted acceptance and a compliant fresh
review pass. #888/#889 closure requires those receipts; this change does not
close #893 or Phases 2–5.

The CPU HIP entry and owner execution rows require the exact-head runtime archive
that the Phase 2 runner pins. They are ignored in an ordinary workspace run and
executed as a complete ignored-only selection inside the Phase 2 oracle; the
header-only owner contract remains active in the ordinary suite. The equivalent
developer commands and expected outcomes are registered in
`docs/manual_gates.md`.

## Phase 2 — canonical ABI descriptors

**Requires:** Phase 1.

**Delivers:** C3 completely and the binding/device portion of C4: `chelis-abi`,
the moved shared metadata authority, generated host views/device descriptors,
freshness and layout probes, dynamic-rank
exact metadata plus private validated Python/DLPack wrappers, and the deletion
of every handwritten mirror. The current [#1289] public ABI and [#1347]
zero-extent behavior are positive receipts.

Shared extraction and spec-derived tests may be prepared against existing
checked code while the remaining Phase 1 consumers are in flight. This does
not satisfy the prerequisite: Phase 2 adoption and landing require the landed
Phase 1 base and its complete execution/mutation oracle. The callable device
layout cutover and spec/11's ABI version 2 producer/consumer admission ship
together; version 1 must fail before metadata interpretation or library loading.

**Issue exit:** [#1345] closes after Python host-to-device, device entry,
device-to-host, and DLPack paths pass rank 0, 1, 8, and greater-than-8 cases,
preserve `int64` values above `i32::MAX`, reject out-of-domain values before
copy, and have no fixed array or narrow mirror left.

**Also delivers:** the backend runtime-header capacity-census leg. The HIP
support root uses the tagged device packet and is inventoried under a fixed,
freestanding clang target with the committed Phase-0 SDK stubs and closed
canonical include attribution. Its recursively discovered support-header set is
the structural owner selection before row extraction, so every reached nested
declaration reaches final-authority comparison while SDK rows remain inputs.
Its ten final rows have their own baseline and `coverage_manifest()` entry. The
recursively discovered published Metal `.h` set exports only `static inline`
definitions, so it has no capacity-census row and no fake SDK lane. Instead an
executable enrollment gate parses every published header with the shared
C-family lexer, aggregates and sorts raw attributable rows, and fails as soon
as one gains a public declaration, requiring a hermetic Metal lane and exact
authority in that change. Phase 2 is not complete while any generated device
header carries numeric surface no census enumerates.

**Oracle:**

```sh
uv run --managed-python --python 3.11 --no-project python \
  scripts/runtime_representation_oracle.py --phase 2
```

Final line: `RUNTIME REPRESENTATION PHASE 2: PASS`.

The implementation is `scripts/runtime_representation_phase2.py`.
`runtime_representation_phase2_tests.json` stores seven required Python contract
identities and 243 explicit required Rust identities: 33 shared-ABI tests, six
metadata-plan C API tests, 79 HIP descriptor/owner tests, 109 platform-invariant
Python binding tests, and sixteen backend-header census/enrollment tests. Counts
and digests are derived summaries, not membership authority. The Python leg names its
integration binaries and relevant internal ownership tests explicitly rather
than freezing platform-only package tests. No leg receives a runtime: the HIP
harnesses stage the runtime their build carries through `chelis-runtime-bundle`,
chelis-cli legs link what `chelis build` staged, and the Python leg links the
runtime the extension carries (#1354). The command first obtains a complete
fresh Phase 1 receipt, then lists and executes each complete current Phase 2
cohort with zero retries. Every required identity must remain selected,
nonignored, executed and passing; additions are executed and reported, while a
removal or rename blocks. Receipt schema 2 distinguishes `required`, current
`selected`, exact `additions`, and passing `executed` identities, and the runner
verifies unchanged test artifacts and source identity. The only recorded
non-execution is the exact HIP hardware command above; CPU SDK-fixture execution
does not relabel it as hardware evidence. Hosted CI advances the stable
`runtime-representation-phase0-oracle` job identity to this command and uploads
both Phase 1 and Phase 2 receipts.

## Phase 3 — typed host runtime access and the field seal

**Requires:** Phase 1. It may run in parallel with Phase 2 and does not consume
the Python or device descriptor.

**Delivers:** the host-runtime and public-C portion of C4. Migrate
repository-owned allocation, views, elementwise kernels, reductions,
formatting, collection ingress/egress, and ownership paths to validated typed
views; foreign C entries use only the branded indexed access core. Delete
public unchecked pointer helpers and broad raw-byte access. Make every raw host
descriptor field private in the owner module as the last diff. Phase 2 owns the
binding/device portion, and C4 is complete only when both phases are green.

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

**Requires:** Phase 1 for the renderer and Phase 2 for device descriptors; it may
run in parallel with Phase 3 but cannot exit before both are green.

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
| public C carrier | [05-OP-44]'s opaque handle with [05-OP-31]'s views, validate before access | external C is inherently forgeable; opacity removes forgeable ownership fields but does not make foreign bytes typed |
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
