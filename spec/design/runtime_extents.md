# Runtime Extents: one resolver for non-literal tensor extents

**Status:** PROPOSED. No phase is implemented. Tracking issue: [#1277].
Code evidence in this document was rechecked on `main` at
`12c22520`; function and type names are the durable anchors.
**Owning specs:** `spec/04-type-system.md` §4.7,
`spec/05-risc-primitives.md` [05-DIM-1..3] plus the owning movement
atoms, and `spec/06-transformations.md` for `vmap`.
`spec/10-serialization.md` controls the public WireDag
encoding. This plan implements those decisions; it does not weaken them to
match a current lane. It exposes two language decisions that must land in the
numbered tier before implementation: runtime-bound behavior under `vmap`
in `spec/06` before Phase 2 and coupled positional-`expand`
settlement order in `spec/04` before Phase 3. Phase 2 also amends
`spec/05`'s closed runtime-extent representation before changing IR.
**Class fixed:** [#1277] -- a direct `shape()` extent or proved tensor-backed
dimension binder passed to `expand` or `reshape` is a folded movement-node
metadata expression. [05-OP-7] requires that folding for the direct read. The
same read used by `pad`, `shrink`, or `stride` remains the node-valued `Shape`
scalar required by `spec/05` section 2.4.1, and every other non-literal tensor
extent is ordinary typed integer dataflow. The current
checker, deferral machinery, and backend recover those values through several
incomplete provenance and symbolic-name paths.

**Interlocks, not hidden scope:**

- [#729] owns extent and axis dtype semantics. Its child [#1112] still owns the
  HIP metadata-carrier widening; this plan may not close it with checker-only
  evidence.
- [#1298] owns runtime axes and reduction-window operands as one issue with one
  authoritative oracle. Phase 2 depends on #1298 closing and that complete
  oracle passing at the exact integration head. This plan consumes its dynamic
  `shape`-axis result but does not duplicate reduction-window rows.
- [#1341] owns the reusable hash-order determinism mechanism, now developed
  separately in PR [#1366]. This document consumes that mechanism only; it
  does not duplicate or review the sibling design.
- [#731] owns witnessed checker diagnostics. [#730] owns typed
  `Unsupported` receipts.

## Summary

The controlling spec is newer than the implementation. It admits a function
parameter, local or top-level binding, record projection, user-function
result, cast, or checked arithmetic expression anywhere an `int64`
extent is expected. Eval, C, HIP, and Metal must execute the same value and
guards.

The implementation still has four separate recovery mechanisms:

1. The checker classifies an `expand` size by syntactic provenance and
   rejects forms its walk does not know.
2. Positional `expand` defers insert-versus-replace selection in
   stores whose consumers and freeze points are not total.
3. The IR carries `Expand.size` as a `DimExpr` and later
   tries to recover runtime extents by symbolic name.
4. Public and target boundaries are not one surface: the C carrier is
   `int64`, the HIP tensor metadata in [#1112] is still 32-bit, and
   exact WireDag v6 serializes `Expand.size` as a string.

The fix is ordered around representation. Phase 2 first gives every accepted
`expand` size either a structurally proved tensor-axis witness (for a direct
`shape()` extent or an in-scope dimension binder) or an ordinary scalar
value edge, migrates every lane and WireDag, and only then deletes provenance
rejections. Phase 3 totalizes the
insert-versus-replace protocol. Phase 4 replaces output-axis name recovery
with one checked source for every realized output axis. One runner records
baseline state and enforces the allowed progression from an ICE or lane
divergence, through a temporary typed implementation receipt, to exact
execution.

## Part I: contracts

### C1 The controlling extent contract

- **C1.1 Admissibility is typing, not provenance.** An `expand` size
  is any expression of exactly type `int64`. A `reshape`
  target is a `List[int64]` of statically known arity whose elements
  are arbitrary well-typed `int64` expressions. No stage may require
  a literal, a tensor-name source, or a privileged expression spelling.
- **C1.2 Identity is proof-gated; equality is guarded.** A literal may remain a
  literal. An ordinary proof may preserve a named input dimension. Otherwise
  the result type carries a fresh `(d-name {} *)` extent. A declared
  literal or name that is not statically proved equal adds a runtime equality
  guard and never makes the value illegal. The runtime value is retained even
  when a type-level identity is preserved.
- **C1.3 Zero is legal.** Static negative extents are type errors and runtime
  negative extents trap `Domain` before allocation or access. Zero is
  neither case: it produces a tensor with the declared shape and logical
  element count zero. No lane may synthesize or access a logical element.
  Physical backing is target-private: a runtime may retain a non-null,
  one-byte allocation when its allocator or device API cannot represent a
  zero-byte buffer. Pointer spelling and backing capacity are not language
  observations and are not cross-lane parity requirements.
- **C1.4 Extents are `int64` and axes are `int32`.** The
  checker, movement signatures, and C carrier have shipped this split. The HIP
  tensor still carries 32-bit shape, stride, size, and allocation metadata;
  [#1112] remains open under [#729] until that target boundary and its Metal
  audit are green. [#1367], a child of [#1277], owns only stale diagnostic and
  comment text that still recommends `cast(N, int32)` for an extent
  or cites the removed Form-3 taxonomy. Correct `int32` axis guidance
  stays.

### C2 Representation first, provenance deletion last

The target `expand` node and graph-level equality contract are:

```rust
RiscOp::Expand {
    axis: usize,
    size: RuntimeExtent,
}

struct RuntimeExtent {
    value: RtDim,
    class: Option<RuntimeDimRef>,
}

struct RuntimeDimRef {
    class: RuntimeDimId,
    use_id: RuntimeDimUseId,
}

struct RuntimeExtentClass {
    id: RuntimeDimId,
    canonical: RuntimeExtentWitness,
    equals: Vec<RuntimeExtentWitness>,
}

struct RuntimeGuardSchedule {
    execution: RuntimeExecutionOrder,
    guards: Vec<RuntimeEqualityGuard>,
}

struct RuntimeExecutionOrder {
    events: Vec<RuntimeObservableEvent>, // total semantic order
}

struct RuntimeObservableEvent {
    id: RuntimeExecutionEventId,
    origin: RuntimeOriginId,
    kind: RuntimeObservableEventKind,
}

enum RuntimeObservableEventKind {
    InterfaceBind(u32),
    Effect(NodeId),
    ExplicitTrap(NodeId),
    CheckedArithmetic(NodeId),
    EqualityGuard { class: RuntimeDimId, member: RuntimeDimSourceId },
    CapacityCheck(NodeId),
    Allocation(NodeId),
    Access(NodeId),
    PublicExposure(NodeId),
}

struct RuntimeEqualityGuard {
    class: RuntimeDimId,
    canonical: RuntimeDimSourceId,
    member: RuntimeDimSourceId,
    check_at: RuntimeExecutionEventId,
    ready_after: Vec<NodeId>,
    guarded_events: Vec<RuntimeExecutionEventId>,
}

enum RuntimeExtentWitness {
    Literal { source: RuntimeDimSourceId, value: i64 },
    Scalar { source: RuntimeDimSourceId, node: NodeId },
    TensorAxis {
        source: RuntimeDimSourceId,
        tensor: NodeId,
        axis: RuntimeWitnessAxis,
    },
}

enum RuntimeWitnessAxis {
    Literal(i32),
    Scalar(NodeId), // earlier rank-zero exact-int32 node
}

struct RuntimeDimDeclaration {
    id: RuntimeDimId,
    owner: RuntimeDimOwner,
    sources: Vec<RuntimeDimSourceDeclaration>, // nonempty, canonical order
    uses: Vec<RuntimeDimUseDeclaration>,
}

struct RuntimeDimAuthority {
    origins: Vec<RuntimeOrigin>,
    declarations: Vec<RuntimeDimDeclaration>,
    dynamic_axis_witnesses: Vec<RuntimeDynamicAxisWitness>,
    transforms: RuntimeTransformNamespace,
}

struct RuntimeTransformNamespace {
    issued_through: Option<u64>,
    next: RuntimeTransformCursor,
}

enum RuntimeTransformCursor {
    Available(u64),
    Exhausted,
}

struct RuntimeDynamicAxisWitness {
    class: RuntimeDimId,
    source: RuntimeDimSourceId,
    tensor: NodeId,
    axis: NodeId, // earlier rank-zero exact-int32 node
}

struct ProvisionalRuntimeDimAuthority {
    origins: Vec<RuntimeOrigin>,
    declarations: Vec<ProvisionalRuntimeDimDeclaration>,
    transforms: RuntimeTransformNamespace,
}

struct ProvisionalRuntimeDimDeclaration {
    key: HygienicRuntimeDim,
    owner: RuntimeDimOwner,
    sources: Vec<ProvisionalRuntimeDimSourceDeclaration>,
    uses: Vec<ProvisionalRuntimeDimUseDeclaration>,
}

struct AnnotatedDag {
    graph: SealedDag, // private; exposes no raw mutable graph
    runtime_dims: ProvisionalRuntimeDimAuthority,
}

enum RuntimeGraphMutationSite { // generated
    LoweringConstruction,
    SelectRootsAndDce,
    ReplaceNode,
    RewriteNodeInputs,
    RewriteNodeType,
    Vmap,
    Grad,
    Specialize,
    Fuse,
    CloneOrRemap,
    ImportDag,
    DecodeWireDag,
    Cse,
    Dce,
    EvalBind,
    HostActualize,
}

struct RuntimeDimOwner {
    lexical_scope: RuntimeScopeId,
    instance: RuntimeScopeInstanceId,
    binder_slot: u32,
}

enum RuntimeOrigin {
    Typed {
        lexical_scope: RuntimeScopeId,
        typed_site: TypedSiteId,
    },
    Synthesized {
        transform: TransformInstanceId,
        parent: RuntimeOriginId,
        path: GeneratedExtentPath,
    },
}

struct RuntimeDimSourceDeclaration {
    id: RuntimeDimSourceId,
    origin: RuntimeDimSourceOrigin,
}

enum RuntimeDimSourceOrigin {
    TensorAxis {
        tensor: RuntimeOriginId,
        axis: RuntimeAxisOrigin,
    },
    ScalarParameter { parameter: u32 },
    Literal { origin: RuntimeOriginId, value: i64 },
    ScalarOpOutput { origin: RuntimeOriginId },
}

enum RuntimeAxisOrigin {
    Literal(i32),
    Scalar(RuntimeOriginId), // rank-zero exact-int32 producer
}

struct RuntimeDimUseDeclaration {
    id: RuntimeDimUseId,
    origin: RuntimeDimUseOrigin,
}

enum RuntimeDimUseOrigin {
    Bound { origin: RuntimeOriginId, field: RuntimeBoundField },
    AliasAxis { origin: RuntimeOriginId, axis: usize },
    ScalarAlias { origin: RuntimeOriginId },
}
```

`ProvisionalRuntimeDimDeclaration` uses the same owner and complete source/use
algebras as the final declaration, but keys them by opaque
`HygienicRuntimeDim`, provisional source, and provisional use IDs. It is
seeded by typed inference before lowering and is primary state beside, not an
index reconstructed from, node-local annotations. The annotations point into
this authority so transforms and finalization can cross-check two independent
representations.
`SealedDag` is a module-private newtype with no `DerefMut`, `AsMut<RawDag>`, raw
constructor, or extraction method. The underlying `RawDag` implementation is
also module-private: it is not re-exported, has no public constructor or
mutator, and does not implement a public `Serialize` or `Deserialize` path.
Only the generated transaction module can construct or mutate it.
`AnnotatedDag` and `FinalizedDag` are the only public graph-state carriers;
ordinary consumers receive read-only queries through those opaque states and
can never recover a raw graph. Neither carrier has a public field or unchecked
constructor: lowering or a registered import creates `AnnotatedDag`, and only
`finalize_runtime_dims` or exact-version verified Wire decode creates
`FinalizedDag`.

Only `RtDim::Lit` and `RtDim::Node` are legal for
`Expand.size` ordinary values. A third legal variant is the structural tensor
axis witness used by a folded expand/reshape direct-shape read or a proved
in-scope dimension binder in one of those owners:

```rust
RtDim::InputAxis {
    tensor: usize,
    axis: RtAxis, // Lit(int32) or Node(input slot with rank-0 int32)
}
```

`tensor` and every `Node` are absolute slots in the owning movement
node's `inputs`. `RuntimeExtent.value` is the executed extent. A
`class` is present only when that value instantiates a hygienic dimension
binder; its class and durable use ID link the concrete use to the DAG-level
declaration whose complete witness manifest is executed independently of any
particular movement node. `RtDim::Sym` and
`RtDim::ToEnd`
remain illegal in `Expand.size`. In executable IR, a named
`reshape` target also resolves to `Lit`, `InputAxis`, or
`Node`; `Sym` may exist only in the typed pre-monomorphization
carrier and must not cross verification or WireDag.
Phase 2 changes every semantic static-extent carrier from platform-sized
`usize` to exact `i64`: `RtDim::Lit`, `DimInfo::Lit`, the known-size member of
`DimInfo::Named`, `DimExpr::Concrete`, every `TensorType`/output-type copy of
those values, and their exact Wire equivalents including `WireRtDim`,
`WireDimInfo`, and `WireDimExpr`. Executable literal extents admit only
`0..=i64::MAX`; a negative literal is a static type error, and a Wire integer
outside that range is rejected before IR construction on every host width.
There is no `as usize` lowering or serialization step. Only physical
allocation sizes, addressing counts, and host container indices receive an
extent through a checked target-capacity conversion immediately before that
physical use. Node IDs, input slots, ranks, and axis positions remain index
carriers rather than semantic extents. Axis literals remain exact `i32` and
are not extent literals.

A generated semantic-extent transit census is bijective with every place an
exact tensor extent can be stored, copied, compared, normalized, computed,
bound, returned, serialized, or converted before physical use. Its enumerators
cover struct fields and enum payloads; public artifact/schema fields; derived
equality/cache keys; function parameters and returns; map values; intermediate
folder/evaluator accumulators; every encode/decode/bind conversion site; and
every operation parameter classified by `OutputAxisRule` as defining an
output extent. Each row names its exact `i64` type and producer/consumer path,
plus the one checked physical-conversion boundary if it has one. The generated
enumerator output and typed registry are exact bijections: an unregistered
semantic transit or a registered row absent from code fails regeneration.
Adding or changing any such transit as `usize`, an unsigned Wire integer, or
another narrower type fails compilation/regeneration.

The initial frozen rows explicitly include `DimExprKey::Concrete(i64)` and its
normalization arithmetic; `DimExpr::evaluate`, `bind`, and `bind_except` with
exact-`i64` bindings, intermediate results, and returns;
`ExecutionDim.size: Option<i64>` in every serialized
`CompiledExecutionArtifact`; and `RiscOp::OneHot.vocab: i64` plus its Wire
form. They are requirements, not an exhaustive handwritten allowlist: the
generated operation/schema/API enumerators remain authoritative. Positive
cross-host fixtures above `u32::MAX` carry the same value simultaneously
through each independent transit, the movement bound, `TensorType` output
dimensions, every `DimInfo`/`DimExpr` occurrence, public execution metadata,
operation-defined extents, and Wire payload. Every copy, key, binding,
evaluation, and round trip must remain exact on 32- and 64-bit hosts before any
target-capacity rejection is allowed.

- **C2.1 Exact representation invariant.** `inputs[0]` is the tensor
  operand. `RtDim::Node(i)` is an absolute slot in the same node's
  `inputs`, with `1 <= i < inputs.len()`. The referenced node is
  earlier in topological order and has rank zero and exact `int64`
  dtype. A `shape(tensor, axis)` value used as an `Expand.size` or
  `Reshape.new_shape` element, including through a transparent binding, does
  **not** materialize a `RiscOp::Shape` value node for that use: it becomes
  `RtDim::InputAxis`, whose tensor slot is an earlier tensor node and whose
  literal or node-valued axis is exact `int32`. The owning expand or reshape
  reads that input's metadata directly. The typed producer retains this
  `TensorAxisWitness` identity structurally across bindings; it is not
  recovered by a syntax/provenance walk. If the same binding also has an
  ordinary scalar consumer, lowering materializes one `RiscOp::Shape` for
  that consumer while the expand/reshape use remains folded.

  `Pad`, `Shrink`, and `Stride` follow the distinct representation fixed by
  `spec/05` section 2.4.1: a direct or transparently bound `shape()` value is
  materialized exactly once as a rank-zero exact-`int64` `RiscOp::Shape`, and
  the bound carries `RtDim::Node` pointing to that scalar input. Other scalar
  consumers reuse the same node; lowering never creates a second `Shape`
  operation. The declaration may still identify the source as the original
  tensor-axis witness, while the materialized scalar and bound are its durable
  alias/use occurrences. A cast, arithmetic expression, or user-function
  result is ordinary scalar dataflow and likewise reaches every movement node
  as `RtDim::Node`.

  A bare in-scope dimension value such as `a` in
  `c: tensor[a, f32]` uses the same owner-specific carrier. For expand and
  reshape, a tensor witness becomes `InputAxis`; for Pad, Shrink, and Stride,
  it is materialized or reused as the exact `Shape` scalar and becomes
  `Node`. The
  typed environment maps binder identity, never spelling, to its runtime
  witnesses. A literal instantiation becomes `Lit`; a tensor witness
  becomes the tag admitted by the owning field; and a scalar term witness
  becomes `Node`.
  Typed inference gives every binder an opaque provisional
  `HygienicRuntimeDim` key containing its lexical scope and binder slot. This
  key, its witness/use role, and a `RuntimeOrigin` travel through lowering on an
  `AnnotatedDag`; the same key already exists in
  `AnnotatedDag.runtime_dims`, independently of the source node or field that
  carries its annotation. Display spelling is never consulted. Source nodes
  and uses
  that came from typed input carry `RuntimeOrigin::Typed`. Every graph-creating
  pass allocates a deterministic `TransformInstanceId` from the DAG-local
  `RuntimeTransformNamespace` and creates `RuntimeOrigin::Synthesized` for each
  new node or field.
  Its `GeneratedExtentPath` records transform kind, parent origin, deterministic
  local path, operation or field, and axis. A generated origin may therefore be
  remapped and serialized without pretending that it had a `TypedNodeId`.

  `RuntimeScopeInstanceId` is distinct from lexical scope. A root instance has
  an empty lineage; cloning a lexical scope appends the transform instance and
  deterministic clone ordinal. Two copies of the same lexical scope therefore
  have different owners. A binder whose declaring lexical-scope instance is
  inside the cloned set moves to the new instance; a captured binder owned by
  an outside instance preserves its owner. Nested clones extend the lineage.
  The same rule freshens provisional source/use identities. Transform history
  is first-class authority: `issued_through` is a serialized, hashed high-water
  mark for the contiguous local ID range that has ever been allocated. DCE may
  delete a synthesized origin only when no provisional/final declaration,
  source, use, owner lineage, annotation, or dynamic witness references it, but
  never lowers this mark.
  Allocation is checked and total: `Available(n)` returns `n`, sets
  `issued_through = Some(n)`, and becomes `Available(n + 1)`, except
  `Available(u64::MAX)` becomes `Exhausted` after returning the last ID.
  `Exhausted` rejects any graph mutation that needs a fresh transform with
  `Capacity` before mutating the graph or authority. No path wraps or reuses an
  ID. Encode/decode preserves both fields exactly. Their canonical pairs are
  `(None, Available(0))`, `(Some(n), Available(n + 1))` for
  `n < u64::MAX`, and `(Some(u64::MAX), Exhausted)`. Every live synthesized
  origin must refer to an ID at or below the high-water mark; gaps are erased
  transforms, not reusable IDs.

  A no-op decode/refinalize preserves bytes. A transform whose entire output is
  later removed by DCE intentionally changes the serialized high-water mark and
  content hash: the hash commits to deterministic transform-ID allocation
  history, not a full transform audit log or semantic equivalence of optimized
  graphs. Combining annotated DAGs imports them in declared operand order and remaps every live transform
  ID, synthesized origin, generated path, and scope lineage into a fresh
  destination range whose length is the imported history count
  (zero for `None`, otherwise `issued_through + 1`), including erased IDs. It reserves that entire range
  transactionally with widened checked arithmetic and fails `Capacity` before
  mutation if the range does not fit; accepting raw overlapping namespaces is
  invalid.

  Canonical structural ordering is complete. Typed lexical-scope and site IDs
  are assigned by preorder of the canonical typed arena, with child scopes,
  binders, fields, and axes in declared vector order. Owners sort by
  `(lexical_scope, instance_lineage[(transform, clone_ordinal)], binder_slot)`.
  Origins are topologically sorted with every parent first, then by the tuple:
  typed before synthesized; typed uses `(lexical_scope, typed_site)`;
  synthesized uses `(transform, parent_canonical_index,
  generated_path_key)`. `generated_path_key` is the lexicographic tuple of the
  frozen Wire tag bytes for transform kind, deterministic local vector index,
  canonical operation identity bytes, bound-field tag bytes, and numeric axis
  or slot components. Imported IDs are remapped first in declared operand
  order and then compared by this same rule. Declarations follow owner order;
  their source/use ordering follows the rule below. The graph-level
  dynamic-axis witness list follows declaration order and then the source's
  position inside that declaration. Encoders regenerate this order and
  decoders reject reordered tables. No hash-map iteration, graph
  hash, allocation address, or display name participates in ordering.

  Typed lowering and every production graph construction or mutation,
  including vmap, grad, specialization, fusion, cloning, remapping, CSE, and
  DCE, may mutate
  runtime-dimension state only through a closed authority API:
  register, add, remap, specialize, or discharge. Each operation updates the
  independent provisional declaration and the corresponding annotations as
  one checked transaction; no pass can edit the authority vectors directly.
  `AnnotatedDag.graph` is sealed: callers receive read-only queries and have no
  raw graph, mutable-node slice, `set_roots`, `replace_node`, or type/input
  setter. The current public `chelis_ir::Dag` re-export, its public
  construction/mutation methods and serde implementations, and every raw-DAG
  extraction method are removed rather than left as a parallel authority-free
  surface. `RuntimeGraphTransaction` stages the graph mutation, its
  source/use/origin/liveness edits, and any checked discharges, validates both
  representations, and commits atomically or leaves both unchanged. Every
  non-import transaction that actually changes graph or authority state
  allocates exactly one fresh transform ID before commit; all origins it
  creates share that ID and have distinct generated paths. `ImportDag` instead
  reserves the imported history range transactionally as specified above, plus
  one fresh ID only if combination creates new structure. A proved no-op
  allocates nothing. Thus an in-place root edit, DCE, fusion pass, or
  reconstruction remains in the same non-reusing history even when it creates
  no surviving origin.
  Those transforms preserve annotations on copied nodes and use the frozen
  `OutputAxisRule` to assign roles to generated axes. CSE may combine annotated
  nodes only when their complete runtime-origin/class/source/use identity sets
  are identical.
  When any identity differs it must retain separate nodes; there is no
  multi-source stamp carrier and no merge-then-recover alternative. DCE runs
  before authority freezes, but may discharge a binder, source, or use only
  after the authority proves that no public interface, equality obligation,
  movement bound, alias, or surviving annotation refers to it. Root selection
  and DCE are one `SelectRootsAndDce` transaction: selecting a result may remove
  an unreachable internal declaration only through an explicit proven
  discharge, while a public input/interface witness remains a liveness root.

  A generated registry is bijective with every graph-construction, mutation,
  import, and deserialization capability and every reachable production call
  site in the entire workspace, not only compiler-API and CLI. It includes
  in-place root/node/input/type edits and transitive lowering, Eval binding,
  host actualization, specialization, fusion, CSE, DCE, compiler artifacts,
  bindings, prove/offline consumers, cache paths, and C/HIP/Metal routes.
  Every row names its `RuntimeGraphMutationSite`, transaction method, affected
  authority sets, and focused positive/negative mutation. Adding a capability,
  exposing a raw mutable graph, or invoking an unregistered call site fails
  compilation/regeneration. Test fixtures construct graphs through the same
  annotated transaction builder; there is no feature, test helper, or public
  serde route that restores an external raw-graph escape. External-crate
  compile-fail controls attempt to import or deserialize `RawDag`, call its
  former constructors and mutators, extract a raw graph from a pipeline
  artifact, forge `AnnotatedDag` or `FinalizedDag`, mutate through
  `AnnotatedDag`, and submit an arbitrary graph to Eval or each public
  C/HIP/Metal entry point. Tests cover entry selection that
  discards a data root whose
  otherwise-unused public Load still owns a declared witness/class, plus
  generated
  nonidentity movement bounds, nested vmap, two clones of one lexical scope,
  captured outer binders, CSE, specialization, fusion, and post-transform DCE.

  Public `chelis-ir` lowering returns `AnnotatedDag`. Compiler pipeline
  artifacts carry an opaque annotated or finalized state instead of a public
  `dag` field, `dag()`, `into_dag()`, or raw `into_parts()` result. Lowering
  construction, entry/root selection, symbolic binding, host actualization,
  specialization, target fusion, and final DCE all complete as registered
  annotated transactions before `finalize_runtime_dims`. Eval, prove/offline,
  cache comparison, bindings, and the public C/HIP/Metal codegen functions
  accept only `FinalizedDag` or checked Wire bytes that decode to it. Cache and
  artifact serialization cover the finalized authority-bearing WireDag, never
  an independently serialized raw graph. No route reconstructs or mutates
  graph structure after finalization. A backend specialization that changes
  structure must run as a registered annotated transaction before
  finalization, or consume the finalized value back to annotated form and
  refinalize before emission.

  The migration inventory is exact; every current row must reach the stated
  target in the same Phase-2 change set:

  | current public or cross-crate surface | required target |
  |---|---|
  | `chelis_ir::Dag`, its serde impls, and `new`/`add_node`/`node_mut`/`set_roots`/`replace_node` | module-private `RawDag`; construction and mutation only inside registered transactions |
  | `lower_program`, `try_lower_program`, `lower_program_with_context`, and named-entry lowering returning `Dag` | return opaque `AnnotatedDag` |
  | public fusion, grad, vmap, specialization, DCE/CSE/folding, tier-2 lowering, symbolic binding, host actualization, and span/input/type/root mutation over `Dag` | generated `RuntimeGraphTransaction` methods with one registry row per capability and call site |
  | `LoweredLibrary::dag`/raw extraction and pipeline `LoweredParts.pub dag`, `dag()`, `into_dag()`, or raw `into_parts()` | state-specific opaque artifacts; read-only queries, annotated consumption, or finalized consumption only |
  | public Eval functions taking `&Dag` | take `&FinalizedDag` and its derived verified views |
  | public C/HIP/Metal codegen and exported backend helpers taking arbitrary `&Dag` | take `&FinalizedDag`; graph-changing specialization happens before finalization |
  | raw-DAG bincode/cache comparison | canonical finalized WireDag bytes including runtime authority and history |
  | WireDag/prove/offline/binding decode or import | verified `FinalizedDag`, or explicit consume-to-annotated import followed by refinalization |

  Read-only analyses either take `&FinalizedDag` or a non-forgeable borrowed
  view issued by `AnnotatedDag` inside a registered transaction. No public
  analysis signature can be used to manufacture, deserialize, extract, or
  submit a raw graph.

  Only after every graph-creating transform and final DCE does
  `finalize_runtime_dims(AnnotatedDag)` cross-check the independent provisional
  authority against all surviving annotations, then emit a `FinalizedDag` and
  one primary, non-optional `RuntimeDimDeclaration` for every binder instance
  that the provisional authority has not explicitly discharged.
  IDs are assigned from owner and origin order, never from display spelling.
  Each declaration enumerates the complete ordered `RuntimeDimSourceId`s and
  `RuntimeDimUseId`s plus their typed or synthesized origins. This declaration
  carrier is the authority for class existence and membership; finalization
  preserves or explicitly discharges the pre-existing membership and does not
  derive it from the set of surviving nodes, final-DAG stamps, or the
  executable class manifest.
  `FinalizedDag` has no in-place graph-transform API. A caller that must
  transform a decoded finalized DAG consumes it back into the annotated form,
  using the final declarations as the new independent provisional baseline and
  preserving its owner/origin records, history high-water, and transform
  cursor; it
  must refinalize the result.

  Every rank-zero scalar or statically selected tensor axis that independently
  witnesses the binder is stamped `WitnessOf { class, source }`; a pass-through
  occurrence is stamped `AliasOf { class, use_id }`. Tensor result metadata
  carries one optional stamp per statically known output axis, and rank-zero
  exact-`int64` scalar producers carry the scalar form. A node-valued-axis read
  cannot use either location: finalization emits exactly one graph-level
  `RuntimeDynamicAxisWitness { class, source, tensor, axis }` for it. That
  carrier executes the metadata selection, owns both node dependencies, and is
  a control/liveness root independent of every static output-axis stamp.
  Optionality means an axis may be anonymous or concrete; it does not make a
  declaration source or use optional. Every source ID named by a
  `RuntimeDimDeclaration` must appear exactly once in one of four disjoint
  placements: a declared literal, a scalar-producer stamp, a static
  tensor-output-axis stamp, or a graph-level dynamic-axis witness. Every use ID
  must appear exactly once as an alias stamp or `RuntimeDimRef`.

  `RuntimeDimSourceOrigin::TensorAxis` covers both a static output axis and a
  folded `shape(tensor, axis_value)` read. Its tensor origin is a liveness and
  ownership dependency. Its axis is either an exact `int32` literal or the
  origin of an earlier rank-zero exact-`int32` scalar producer, which is also a
  liveness dependency. Finalization resolves those origins to the
  `RuntimeExtentWitness::TensorAxis` node and static-or-node axis carrier. A
  literal axis resolves through the exact static-axis stamp; a scalar axis
  resolves through the dedicated graph-level dynamic-axis witness and is
  invalid if any static output-axis stamp claims that source.
  `ScalarOpOutput` covers rank-zero exact-`int64` results of casts, arithmetic,
  and user functions; it resolves to `RuntimeExtentWitness::Scalar` and never
  invents a meaningless tensor-axis index. Static public tensor axes use the
  same tensor-axis algebra with a literal axis, while scalar parameters keep
  their explicit signature position.

  Phase 2 generates and freezes one `OutputAxisRule` table bijective with the
  complete current `RiscOp` registry. Each possible output axis is classified
  as literal, external, structurally forwarded, scalar-derived, or
  `OpComputed(OpExtentRule)`. The row determines witness-versus-alias and, for
  an operation-computed witness, the exact pre-allocation extent formula citing
  the owning numbered operation atom. An unclassified operation, output, rank
  rule, or formula is a build failure; a missing governing atom is authored in
  Phase 2 before the row can register. This is the single table Phase 4 later
  consumes for total `AxisSource`; no second operation/source table exists.

  `derive_runtime_extent_classes(&FinalizedDag)` walks the authoritative declarations,
  resolves each durable source ID through literals, scalar/static-axis stamps,
  graph-level dynamic-axis witnesses, and the typed-origin map, and
  constructs one executable manifest per ID. It does not discover class
  membership from optional stamps. Classes follow declaration order; within a
  class, sources follow the exact order frozen in its declaration: external
  signature witnesses first in signature/source order, then literals in origin
  order, then scalar-operation and tensor-axis sources in origin order. Within
  a tensor-axis source, a literal axis sorts before a scalar-origin axis and
  the latter uses canonical origin order. Uses sort by canonical origin, then
  frozen Wire role-tag bytes (`bound`, `alias_axis`, `scalar_alias`), then the
  structural bound-field key or numeric axis. The first member is canonical
  and every remaining member is an equality obligation. A
  `RuntimeExtent` that denotes the binder carries the declared class/use pair
  and executes the canonical value; it does not carry a second, losable copy of
  the member list.
  A reusable generic may retain the
  binder in typed pre-monomorphization state, but a complete executable DAG
  must resolve it to a local class/use pair and one of these three value forms.
  Missing witness or use, wrong class, and violated witness equality fail
  loudly; no pass searches for a matching string.
  Thus an expand/reshape direct read obeys [05-OP-7]'s folded-`DimExpr` rule,
  while a Pad/Shrink/Stride read and every other scalar value obey the
  node-valued rule in `spec/05` section 2.4.1 and `spec/04` section 4.7.4.
  Both forms are real owning-node input
  dependencies, so DCE cannot lose them; a symbolic name or shape-only side
  table is not an equivalent size carrier.

  The in-memory owner matrix is exact and matches the Wire matrix; all named
  fields below store `RuntimeExtent`, not a bare `RtDim`:

  | `RiscOp` field | in-memory type | legal `RuntimeExtent.value` |
  |---|---|---|
  | `Expand.size` | `RuntimeExtent` | `Lit`, `InputAxis`, `Node` |
  | `Reshape.new_shape[*]` | `RuntimeExtent` | `Lit`, `InputAxis`, `Node` |
  | `Pad.padding[*].before/after` | `RuntimeExtent` | `Lit`, `Node` |
  | `Shrink.bounds[*].start` | `RuntimeExtent` | `Lit`, `Node` |
  | `Shrink.bounds[*].end` | `RuntimeExtent` | `Lit`, `Node`, `ToEnd` |
  | `Stride.strides[*]` | `RuntimeExtent` | `Lit`, `Node` |

  A surviving binder use requires `class: Some(RuntimeDimRef)` in every row.
  Anonymous or fully concrete values may have no class. `ToEnd` is legal only
  for `Shrink.bounds[*].end` and requires `class: None`; `Sym` is illegal in
  every executable owner. Verification maps each row through the same
  `RuntimeBoundField` identity that its provisional and final use declaration
  carries.
- **C2.2 Equality classes are executable graph structure.** The DAG stores the
  authoritative declarations, the independently mapped `WitnessOf` and
  `AliasOf` source/use stamps, the independently mapped graph-level
  dynamic-axis witnesses, and the derived canonical class manifest.
  Verification starts from the declarations and requires two exact
  bijections: declared source/use IDs to witness stamps, alias stamps,
  dynamic-axis witnesses, literals, and `RuntimeDimRef`s, then declared ordered
  classes to executable manifests. It rejects an absent declaration, source,
  use, stamp, dynamic-axis witness, class, or
  bound reference; a duplicate, unstamped,
  cross-class, split, or merged member; and any owner/source-ID inconsistency.
  It also checks every scalar/axis node, rank, dtype, topological position,
  literal, `OutputAxisRule`, and `RuntimeExtent.class` reference. This is
  deliberately graph-level:
  Load/Load, Load/operation-output, and operation-output/operation-output
  equalities exist even when no movement-bound field owns them.

  Canonical class/member order controls identity, equality pairing, and
  serialization; it does not reorder dynamic evaluation. Finalization derives
  one non-serialized `RuntimeEqualityGuard` for every noncanonical source and
  verifies an exact `RuntimeGuardSchedule`. `RuntimeExecutionOrder.events` is
  one total semantic order containing every interface binding, effect,
  explicit or checked-arithmetic trap, equality guard, capacity check,
  allocation, potentially trapping access, and public exposure exactly once.
  Typed source order and the numbered left-to-right/sequential rules order
  typed events; synthesized origins and the registered operation rule place
  transform-created events without inventing a target-specific order. Each
  guard's `check_at` names its exact `EqualityGuard` event in that list, so its
  complete independent observable predecessors and successors are structural
  facts: the immediately adjacent event IDs are the lower and upper order
  fences. Declaration/source order breaks ties only when the language leaves
  already-ready guards at one semantic point unordered; it never moves a guard
  across an existing observable event.

  `ready_after` is the duplicate-free, canonical-topological producer closure
  for the canonical and compared member. `guarded_events` is the
  duplicate-free execution-order list of every later capacity check,
  allocation, access, exposure, or other event whose correctness relies on
  that obligation. The `check_at` event must follow every producer, precede
  every guarded event, and occupy the exact source-order slot between its
  observable predecessor and successor. These three independent relations
  make producer readiness, dependent-use precedence, and unrelated
  effect/trap order separately verifiable.

  Only actual interface values and declared literals are available for the
  entry/prologue schedule. Eval, C, HIP, and Metal check those entry members in
  declared interface order before their first dependent use. A local
  `ScalarOpOutput`, node-valued-axis read, user-function result, or other local
  witness remains ordinary dataflow and is checked only after all of its
  producer dependencies have executed, then before its first dependent
  allocation, access, or exposure. An `OpComputed` member uses the
  Phase-2-frozen `OpExtentRule` at its owning operation and checks equality
  before allocating or exposing that result. Every guard is an explicit
  control dependency for its `guarded_events`, and every lane emits the total
  observable-event order as control edges between adjacent events. Every root
  return waits for all live declaration guards. Thus an independent earlier
  effect or trap is a structural predecessor of a later mismatch, while a
  mismatch that belongs before an independent later effect/trap is its
  structural predecessor. No lane may duplicate a producer, evaluate a local
  scalar in the prologue, omit either observable-order fence, delay a guard
  past a dependent event, or report the failure under a different source
  origin.

  `derive_runtime_guard_schedule(&FinalizedDag)` is the one checked view used
  by Eval and every backend. It is invalidated with the finalized carrier and
  recomputed after consume-to-annotated transformation and refinalization; it
  is not serialized or cached as a second authority. Verification rejects a
  missing/duplicate observable event or guard; a mismatched `check_at`;
  incomplete readiness, guarded-event, predecessor, or successor relations; an
  entry guard for a local producer; a dependency/event-order cycle; or any
  schedule that crosses source/effect/trap order. Each provisional declaration is a control/liveness
  root before finalization; each final declaration, class manifest, and guard
  schedule is a control/liveness root afterward. Final DCE runs on the
  annotated graph before final declarations freeze. It consults the independent
  provisional authority and must prove every explicit discharge; correlated
  removal of a computation and its annotation alone is invalid. Afterward no
  pass can remove a declared
  input, source
  owner, or guard from `FinalizedDag`. Graph transforms preserve/remap
  provisional binder keys, runtime origins, scope instances, sources, uses,
  nodes, fields, and axes and then refinalize. A decoded graph follows the same
  consume-to-annotated/refinalize path. Hashing covers the finalized authority
  and carriers.

  The executable interface maps each declared function input and scalar
  parameter to its declaration source IDs, so dropping an otherwise-data-unused
  Load cannot shrink a class. Verification compares that interface, the
  primary declarations, all source/use stamps and dynamic-axis witnesses, the
  derived class manifest, and every classed bound after each transform and
  after decode. Mutations
  immediately before final DCE remove a scalar-operation source, a tensor-axis
  source with a literal or node-valued axis, an entire provisional class, or
  only a node annotation; each fails against the still-primary provisional
  authority. Post-finalization correlated mutations that change stamps and
  manifests together still fail against the declaration authority: remove a
  member and its data-use edge, remove an
  entire Load-only class, substitute an axis from another class, split one
  class, merge two classes, duplicate or reorder members, stale a node/axis, or
  drop the guard root while leaving the primary declaration fixed. A runtime
  unequal-member mutation traps `Domain`.

  Structural validation does not claim to recover a producer's former intent
  from an arbitrary, coherently rewritten program. Removing an internal
  declaration together with every dependent operation, stamp, manifest entry,
  and use may produce a different valid DAG; ordinary decode accepts it if all
  structural rules hold. The artifact-integrity oracle separately compares the
  bytes with an expected content hash or exact fixture supplied outside the
  mutable Wire payload and must reject that replacement. Removing a public
  input while the out-of-band interface remains fixed is structural failure.
  A digest stored only inside the same replaceable payload is not integrity
  authority. Only after both structural and artifact-integrity paths are green
  may Phase 4 delete `SymbolicBinding.others` and the name-grouped equality
  loop.
- **C2.3 Static values are an optimization.** A statically proved exact `i64`
  value in `0..=i64::MAX`, including zero, may use `RtDim::Lit(i64)`. One
  checked static folder is shared or contract-tested across checker and
  lowering. The same exact `i64` is stored without a host-width cast in every
  `DimInfo`, `DimExpr`, `TensorType` output dimension, runtime bound, and Wire
  copy named by the semantic-extent transit census. Every derived key,
  binding/evaluation API, public artifact, operation-defined output extent,
  and conversion site in that census preserves the same value. Failure to fold produces the
  exact `InputAxis` or `Node` carrier dictated by C2.1; it never rejects the
  expression or guesses a value. A target may reject the resulting physical
  allocation only at the checked capacity boundary, after language- and
  Wire-level exactness has been preserved.
- **C2.4 Every result is constructed.** The checker always constructs an
  `expand` result tensor whose rank is the selected operand rank or
  operand rank plus one. It stamps complete type metadata and checks declared
  or ascribed rank and dimensions. The early exit that causes [#597] and
  [#609] is deleted.
- **C2.5 Every consumer lands before deletion.** Verification, Eval, C, HIP,
  Metal, specialization, fusion, AD, vmap, hashing, and cloning/remapping passes read
  `RuntimeExtent.value`, its optional declared class/use reference, every
  source/use stamp, every dynamic-axis witness, and the complete ordered class
  manifest before any provenance rejection is removed.
  Each ordinary bound value is exactly one tag admitted by the owner matrix;
  `ToEnd` remains the sole sentinel. [#1112]'s
  HIP carrier work and Metal audit are Phase 2
  entry requirements because a checker-only `int64` result is not an
  all-lane extent contract. Entry requires their focused width, capacity,
  guard, target-build, and hardware-availability smoke evidence; execution of
  this plan's runtime-extent group is a Phase 2 **exit**, not an entry
  requirement.
  Equality and negativity guards run before allocation or element access on
  every lane.
- **C2.6 Transforms preserve the bound slice.** This clause is the proposed
  rule for the required `spec/06` amendment; current `spec/06`
  says every node is batched, so the numbered spec wins until that amendment
  lands. Transforming a movement node
  transforms its complete bound-dependency slice, not only the tensor operand.
  Under axis-zero `vmap`, non-tensor scalar parameters remain shared, folded
  expand/reshape `InputAxis` reads observe the corresponding original tensor
  axis after the inserted batch-axis shift, and Pad/Shrink/Stride `Shape`
  nodes plus other scalar integer nodes used only by movement bounds remain
  rank-zero rather than acquiring a batch dimension.
  Every class member, use reference, and source/use stamp follows the same axis
  shift, input-slot remap, liveness, and equality semantics as its canonical
  value.
  Literal axes normalize against the original source rank and then shift;
  node-valued axes perform the same checked normalization and shift at runtime.
  When one scalar producer has both bound and ordinary value consumers, it is
  evaluated exactly once. The bound edge references that rank-zero value. If
  the ordinary branch needs a batched value, the amended `spec/06` and its
  numbered operation atom authorize this adapter:

  ```rust
  RiscOp::BroadcastScalarRef {
      scalar: usize, // earlier rank-zero scalar input slot
      batch_axes: Vec<TensorAxisRef>, // earlier tensor slots + static axes
  }

  struct TensorAxisRef {
      tensor: usize,
      axis: usize,
  }
  ```

  All slots are absolute positions in the adapter node's inputs and every
  referenced node is earlier; each static axis is in range. It outputs one axis
  per ordered `batch_axes` entry, preserves the scalar's
  exact dtype, and repeats the already-computed value. Its `OutputAxisRule`
  structurally forwards those batch axes. It never re-executes the checked
  arithmetic or user function; nested vmap extends the ordered axis list. A
  folded expand/reshape `shape` use needs no scalar adapter for the bound edge;
  `InputAxis` reads the vmapped tensor metadata. A Pad/Shrink/Stride bound
  reuses its one rank-zero `Shape` producer through `Node`, including when an
  ordinary branch also consumes that producer. The amended spec and tests
  cover literal, scalar-parameter, shape, arithmetic, and dual-use slices.
  Grad, specialization, cloning, and remapping preserve or remap every absolute
  input slot and its dtype/rank invariant. A bound computed from vmapped tensor
  *elements* could vary per example and cannot construct one regular stacked
  output shape. Phase 2 is pinned to the explicit typed-rejection route: the
  required `spec/06` amendment must reject that dependency before lowering. If
  the numbered-spec review chooses guarded equality instead, implementation
  stops and this design is amended first. Implementation may not batch a
  bound-only scalar, guess, silently share a varying value, or duplicate a
  computation before the numbered rule and its positive/negative tests land.
  Production fusion runs on `AnnotatedDag` before finalization. Every scalar,
  axis-selector, tensor-axis-read, guard, alias, or movement-bound node in a
  runtime-dimension control/dependency slice is a fusion barrier and remains an
  explicit node with its origin, source/use identity, dtype/rank, and input
  slot. For a dual-use producer, fusion retains that one producer as an
  explicit multi-consumer input and may fuse only eligible ordinary nodes
  downstream of it; no bound/control computation is cloned. Fusing differently
  stamped producers, absorbing an `RtDim::Node` producer into `FusedElem`, or
  running a target fusion pass over `FinalizedDag` is invalid. A decoded final DAG must
  consume to annotated form, fuse under the same rule, and refinalize.
  Overflow, division-by-zero, explicit traps, and effectful user-function rows
  prove exact occurrence count, order, and attribution across vmap and fusion.
  All graph-creating transforms operate on provisional hygienic annotations,
  create typed or synthesized origins as C2.1 requires, and finalize only after
  their last DCE. Focused tests finalize after each named transform to prove the
  resulting authority; no generated node borrows a nonexistent typed-site ID.
- **C2.7 The deletion is atomic with the usable replacement.** Only after
  C2.1-C2.6 and C6 are green does the phase delete `SizeClass`,
  `classify_expand_size`, `classify_arith_app`,
  `sourceless_expand_size_error`,
  `Env::size_provenance` and its plumbing, plus the lowerer's mirror
  rejections. No intermediate commit may accept a value the IR cannot carry.
  Phase 4 separately removes `SymbolicBinding.others` only after
  C4 consumes the graph-level runtime class structure. Best-effort
  identity recognition may survive only as refinement whose
  failure result is a fresh extent plus guard.

### C3 Positional expand uses one normative protocol

The implementation derives its action from `spec/04` §4.7.2 rather
than assigning semantic labels by intuition:

- **`Constrain(expected)`** is used when the context supplies an
  independently fixed tensor rank or shape equation. A declared result,
  ascription, already-instantiated user-function parameter or generic field,
  branch join with independently resolved shape evidence, or builtin relation
  with an independently resolved operand equation selects the unique candidate
  satisfying that equation. A comparison constrains a pending operand only
  when its other operand or surrounding resolved evidence supplies that
  independent equation; two pending operands merely link and propagate their
  shared choice even though the comparison result is scalar or boolean.
- **`Propagate`** is used when a context can carry the same unresolved
  monomorphic candidate without requiring either rank. It adds no evidence and
  cannot select or clone the choice.
- **`Freeze`** is used only at a freeze point named by §4.7.2, when a
  concrete tensor shape is required and no independent constraint selected a
  candidate. An axis within the input rank selects same-rank replacement;
  `axis == rank(input)` selects trailing insertion.

Every inference rule that consumes a tensor must invoke exactly one action.
This is enforced through one mandatory, typed choke point rather than inferred
from a review table. Recursive expression inference returns an opaque,
`#[must_use]` `Inferred` carrier whose type and pending-expand tokens have
private fields. A child carrier can enter its parent only through
`consume_pending(site: PendingUseSite, value: Inferred,
evidence: ExpectedShapeEvidence)`. Only that module can inspect, clone, unify,
project, or finalize a pending token; there is no raw-`Type` escape hatch.
`PendingUseSite` is a closed enum generated bijectively from the builtin
declaration registry and the Deep expression-form registry, so a new builtin
or expression form fails compilation/regeneration until it has a disposition.
Each disposition is an exhaustive decision function over a closed
`ExpectedShapeEvidence` enum (`Independent`, `Fresh`, or
`LinkedPending`), not one action baked into the syntactic site.
`Independent` carries the resolved tensor equation and yields
`Constrain`; `Fresh` and `LinkedPending` yield
`Propagate` unless the numbered freeze rule applies. Adding an evidence
variant or leaving a site/evidence cell unhandled fails compilation and the
generated table check.

`PendingExpandAction` is a private implementation detail that only the
generated exhaustive `decision_for(site, evidence)` function can construct.
No inference caller can name, import, or pass `Constrain`, `Propagate`, or
`Freeze`; `consume_pending` must call the generated decision function before
it can inspect the carrier. A compile-fail privacy test rejects direct action
construction or a helper that bypasses `decision_for`, and regeneration fails
if any site/evidence cell is missing or changed without its semantic fixture.

The recursive choke point walks tensor-bearing tuple, List, record, ADT, and
function fields rather than only a top-level tensor. The action is derived from
resolved expected-shape evidence, not from container syntax:

- an anonymous tuple field, closure capture, fresh generic field or parameter,
  singleton List, and the seed element of an inferred List
  `Propagate` because each defines or carries its own provisional type;
- a concrete declared record/ADT field, an already-instantiated generic field,
  a declared/ascribed List element type, and a later List element facing an
  independently resolved accumulated element shape `Constrain`;
- if a later concrete List element supplies the first independent shape, the
  recursive carrier routes that evidence back to constrain the earlier pending
  seed; two unresolved elements link one monomorphic pending choice and
  continue to propagate rather than selecting each other;
- tuple/record/ADT projection and pattern binding transfer the selected token;
  record update constrains an updated field only when its resolved schema is
  independent, otherwise propagating the generic field along with untouched
  fields.

Branch joins, comparisons, and builtin operand relations with
`LinkedPending` evidence propagate the shared monomorphic choice; the same
sites constrain only when `Independent` evidence is present. Declared results,
ascriptions, and independently instantiated user-function parameters provide
that independent evidence and constrain;
an undeclared closure return propagates to the call boundary; and
complete-program finalization freezes. Generated compile-fail bypass tests
prove no inference helper outside the module can unwrap or copy the carrier.
Generated semantic tests execute both candidate outcomes and a contradictory
shape for every `Constrain` row, propagation followed by later
selection for every `Propagate` row, and the documented default for
every `Freeze` row. Mutations that omit the tuple-projection,
constructor-field, List-element, record-update, pattern-binding, or
closure-return call fail before a checker result is returned. Generated rows
cross replacement, insertion, and contradiction with concrete fields, fresh
and fixed generic fields, singleton/list-seed elements, later concrete and
pending elements, declared List element types, and nested generic containers.
The matrix also crosses pending/pending comparison and branch join, those same
forms followed by independent result evidence, contradictory independent
evidence, and final freeze with no independent evidence. Mutating either a
site/evidence disposition or the private dispatch call fails compilation,
regeneration, or the corresponding exact-verdict row.

Coupled defaults expose settlement order. Before Phase 3 implementation,
`spec/04` §4.7.2 must state the order. This plan proposes first-deferred
source order because it is source-stable and requires no lookahead, but the
numbered-spec amendment decides; if it chooses another order, this document
and [#1277] must be updated before implementation. The ordered-store mechanism
comes from [#1341]; this plan owns the resulting extent verdict and the
[#1338] corpus row. If a consumer cannot be derived from the three normative
actions, implementation stops for a numbered-spec amendment rather than
adding a table exception.

### C4 Every realized output axis has one checked source

An exhaustive match over `RiscOp` variants catches a new operation
but not a missing flow through an existing operation. The structural
interface is therefore a total per-output-axis algebra:

```rust
enum AxisSource {
    Literal { value: i64 },
    ExternalAxis { load: NodeId, axis: usize },
    InputAxis { input: usize, axis: RtAxis },
    ScalarInput { input: usize },
    OpComputed { op: NodeId, axis: usize, rule: OpExtentRule },
}
```

- **C4.1 Cardinality and ownership.** Every realized node supplies exactly one
  `AxisSource` for each output axis; the vector length equals output
  rank, and omission or duplication is invalid. `ExternalAxis`
  validates that `load` is the exact external-input `Load` node
  and that `axis` exists; a `Load` may self-identify this way and
  downstream nodes retain its `NodeId`, never its string name.
  `InputAxis` validates
  the tensor input slot and its literal- or scalar-valued exact-`int32` axis.
  `ScalarInput` validates the same
  earlier-node, rank-zero, exact-`int64` contract as C2.1.
  `OpComputed` is permitted only for a closed operation-and-axis
  `OpExtentRule` whose formula is the owning numbered operation atom;
  it is not a wildcard fallback. Phase 4 instantiates each source from the
  single `OutputAxisRule` row already generated, atom-authorized, and frozen by
  Phase 2. That table is bijective with the complete current `RiscOp` registry
  and assigns every output axis its source class and formula. Adding an op,
  output, or rank rule without a complete row has already failed regeneration
  and compilation before Phase 2 exits.
  The frozen row also supplies the runtime-dimension role for the axis: an
  external or operation-computed source is `WitnessOf { class, source }`,
  while a structurally forwarded source is
  `AliasOf { class, use_id }`. Phase-4
  derivation requires that role to agree with the declaration authority, stored
  axis ID, and complete runtime-class manifest; `AxisSource` never invents,
  merges, or recovers a class from a display name. There is no Phase-4 sibling
  registry or target-specific formula match.
- **C4.2 Exact movement mappings.** Same-rank `Expand` maps every
  unchanged output axis to the same input axis and maps the replaced axis to
  its literal or scalar size. Rank-increasing `Expand` maps axes
  before the insertion unchanged, the inserted axis to its size, and later
  output axes to input axis `output_axis - 1`. Each
  `Reshape` target maps to its literal, folded input axis, or scalar input;
  a proved name remains output type metadata, not a runtime name lookup. Identity
  `Shrink`, `Stride`, and `Pad` axes use
  `InputAxis`; non-identity axes use their exact
  `OpComputed` rule. `Load` axes use
  `ExternalAxis`. Shape-preserving non-movement ops use the exact
  input-axis map, while reductions, concatenation, convolution, and every other
  computed-shape op use a closed rule citing their own numbered atom. These
  classifications and formulas are Phase-2-frozen `OutputAxisRule` content;
  Phase 4 consumes them without making a new semantic decision.
- **C4.3 Interim failure is typed.** The algebra first lands as a verifier and
  property ratchet. A currently unsupported but well-typed mapping yields the
  registered [#730] `Unsupported` receipt. It never reaches the
  occurrence-pass ICE and never substitutes an input extent.
- **C4.4 Target state consumes the algebra.** Eval and all backends consume
  `AxisSource` directly. Runtime extent flows no longer depend on
  `shape_source_for_axis`, `op_declared_output_axes`, or a
  search for a Load carrying the same string. The graph-level
  `RuntimeExtentClass` manifest supplies every runtime equality witness,
  including external axes and operation-computed output axes with no movement
  consumer, so this phase also deletes `SymbolicBinding.others` and its
  name-grouped execution loops.
  Static symbolic identities may remain in type metadata, but no runtime size
  or guard is recovered by name. This
  closes [#665] and [#592].
- **C4.5 Sources are a final-DAG derived view.** `AxisSource` is
  neither stored in `FinalizedDag`/WireDag nor carried across transforms.
  `derive_axis_sources(&FinalizedDag)` runs after the last DAG rewrite and
  before final verification/emission, producing a view tied to that exact DAG
  generation. Any mutation or transform invalidates the view; callers cannot
  reuse it because the API borrows the immutable final DAG and keeps the map's
  `NodeId`s private. Eval and each backend derive or receive the view
  only after vmap, grad, specialization, fusion, cloning/remapping, and DCE
  finish. Production fusion runs before finalization;
  Phase 4 runs derivation and cardinality verification after each transform in
  focused tests and after the complete production pipeline. This proves
  current-node identity without inventing a second remapping protocol.

Required mutation tests cover every `RiscOp` registry row, external
Load/root axes, kept axes before and after an inserted
`Expand` axis, the inserted/replaced axis, same-rank versus
rank-increasing forms, every `Reshape` target, and omitted,
duplicate, wrong-shifted, out-of-range, and wrong-node sources. Lifecycle rows
derive the view after vmap, grad, specialization, fusion, cloning/remapping,
and DCE;
a mutation that caches a pre-transform view or accepts a stale `NodeId`
fails before emission.

### C5 The class oracle has a reachable boundary

The authoritative named suite is `scripts/runtime_extent_oracle.py`
plus its two platform execution gates and the exact-head #1298 prerequisite
oracle. The runner accepts
`--phase 0` through `--phase 4` and `--phase final`.
Final automatic success exits zero with:

```text
RUNTIME EXTENT ORACLE: PASS
```

The suite records the exact commit and corpus digest so host, HIP, and Metal
evidence cannot be combined across different heads. Phase 2 and later first
run the external prerequisite at that same commit:

```sh
uv run --managed-python --python 3.11 --no-project python scripts/dtype_dynamic_axis_window_oracle.py
```

It must end with `DTYPE DYNAMIC AXIS WINDOW ORACLE: PASS`.
Its reduction-window cells remain owned by #1298 and are not copied into this
corpus; composing the full oracle is the honest cost of #1298 owning dynamic
axes and windows in one issue.

1. **Host parity.** A generated legal matrix crosses extent-producing forms
   (literal and literal arithmetic, parameter, local/top-level binding, record
   projection, user-function result, cast, checked arithmetic, a bare in-scope
   dimension binder with one or several tensor witnesses, and direct or
   indirect `shape()` reads with literal or computed axes) with `expand`,
   `reshape`, `shrink`, `pad`, and
   `stride` where each form is legal. `check`,
   `eval`, and compiled-and-executed C agree on acceptance, shape,
   values, and traps. The named-dimension rows prove literal instantiation,
   canonical signature-order witness selection, and complete class guards for
   Load/Load, Load/operation-output, and operation-output/operation-output
   classes, including a class with no movement-bound consumer. Positive guard
   rows compare a node-valued-axis shape read and a rank-zero arithmetic or
   user-function scalar result against an independently declared literal or
   named witness. The guard schedule crosses interface-only prologue checks,
   local scalar and dynamic-axis producers, user-function results, and
   operation-computed axes. One row places an earlier effect or explicit trap
   before a later mismatch and requires that earlier event to win; the converse
   places a ready mismatch before a later effect/trap and requires the mismatch
   to win without executing the later event. Eval and compiled C agree on
   occurrence count, order, and source attribution. They also prove loud
   missing/wrong-class failure. Static exactness rows use values above
   `u32::MAX` and assert identical `i64` bits in the movement bound; every
   `DimInfo`/`DimExpr`/`DimExprKey`/`TensorType` output copy; each
   evaluate/bind/bind-except map, intermediate, and return; serialized
   `ExecutionDim`; `OneHot.vocab`; and the decoded Wire graph;
   physical capacity rejection, when applicable, occurs only afterward.
   Runtime windows are absent from this
   corpus; the composed #1298 oracle owns them.
2. **GPU build and execution.** The same named rows compile and execute in the
   HIP and Metal correctness suites, not merely through capability-gate
   rejection tests. The exact manual commands are:

   ```sh
   scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
   PYO3_PYTHON="$(uv python find 3.11)" cargo test -p chelis-backend-metal --test gpu_correctness -- --ignored --test-threads=1
   ```

   Each command must report the runtime-extent group green at the same commit
   and corpus digest as the host run.
3. **Negative parity.** Static negative extents fail with the owning type
   error; runtime negative extents trap `Domain`. Wrong dtype,
   out-of-range axis, rank-contradicting ascription, malformed scalar input,
   named-binder witness mismatch, missing declaration/source/member, split or
   merged class, wrong binder or source ID, a host-width cast in any
   semantic-extent transit census row, and checked overflow fail for the owning
   reason on every applicable lane. Guard mutations that hoist a local producer into the
   prologue, cross an earlier effect/trap, delay past a dependent allocation or
   exposure, omit a control edge, duplicate a producer/guard, or change failure
   attribution are exact negatives across Eval, C, HIP, and Metal.
4. **Zero positives.** Literal-zero and runtime-zero rows cover positional
   replacement, positional insertion, and named-axis expansion. They assert
   the exact output shape, logical element count zero, no element access, and
   Eval/C/HIP/Metal agreement. They inspect logical tensor metadata, not pointer
   nullness or target-private backing capacity. C and Metal one-byte backing is
   a positive control; a HIP mutation that clamps logical `size` to one must
   fail. Acceptance alone is not sufficient.
5. **Real #569 transformation.** The runner proves a direct spelling checks,
   evaluates, and compiles; copies it to a temporary task-owned path; runs
   `chelis lint --fix` and `chelis fmt --inplace`; proves
   formatting is idempotent and parseable; runs `chelis lint --check`
   and the normal style-gated `check`, `eval`, and compiled
   C path; and compares type, rank, shape, and value with the control. A
   negative fixture proves the typed-pipeline safety gate suppresses a rewrite
   whose transformed program would not preserve the typed result.
6. **Deferral stability.** Every positional candidate row is run in K fresh
   processes. [#1338] is named and must settle to the normative source-order
   verdict every time.
7. **Transform integrity.** Positive rows cover `Lit`, expand/reshape
   direct-shape and named-binder `InputAxis`, Pad/Shrink/Stride materialized
   shape `Node`, shared non-tensor `Node`, and arithmetic `Node` dependencies,
   including dual-use producers, through vmap, grad, specialization, fusion,
   cloning/remapping, and DCE. They
   verify exact rank-zero
   bound scalars, shifted literal and node-valued axes, preserved tensor-axis
   witnesses, absolute input slots, typed and synthesized origins, lexical and
   scope-instance owners, source/use/class IDs, witness/alias roles, complete
   manifests, guards, shapes, and values.
   Mutations that
   prepend a batch axis to a bound scalar, re-evaluate a dual-use producer
   instead of referencing it through `BroadcastScalarRef`,
   omit an `Expand` bound from
   grad liveness, lose a source under DCE, fail to shift an axis, or retain a
   stale slot/ID fail before emission. Grad-created movement/output witnesses,
   specialization, and post-transform DCE finalize exactly. CSE merges only
   identical complete identity sets; a mutation that merges producers carrying
   different source IDs fails rather than encoding an unrepresentable shared
   stamp. Pre-finalization mutations remove a scalar-operation source, a static
   or node-valued tensor-axis source, a whole provisional class, and one
   annotation immediately before final DCE; every mutation fails against the
   independent provisional authority. Cloning rows
   cover nested vmap and two copies of one lexical scope: the clone lineage
   gives each local declaration a distinct scope instance, while declarations
   captured from an outside instance remain shared. Decode lifecycle rows prove
   no-op refinalization byte stability, two post-decode clones of one scope,
   nested post-decode transforms, deterministic ordered DAG import/remapping,
   and rejection of replayed transform IDs, origins, generated paths, malformed
   scope-instance lineages, or duplicate complete owner identities. The
   derived guard schedule is recomputed after every named transform. Mutations
   stale or move a `check_at` event; omit or swap its immediate observable
   predecessor/successor fence; corrupt `ready_after` or `guarded_events`;
   classify a local scalar as entry-ready; and place the guard on either side
   of an independent earlier/later effect or trap. Each fails before emission.
   The generated production-mutation
   registry covers every workspace production
   graph construction/mutation/import/deserialization capability and reachable
   call site. External-crate compile-fail rows reject raw graph import,
   construction, serde decode, root/node/input/type mutation, raw artifact
   extraction, annotated/finalized carrier forgery, and arbitrary-graph
   submission to Eval and C/HIP/Metal;
   entry-selection
   rows prove `SelectRootsAndDce` cannot lose an otherwise-unused public
   witness/class with the discarded data root. Fusion positives retain
   compute-once bound-only/dual-use scalar and dynamic-axis
   nodes while fusing eligible ordinary branches; mutations that absorb an
   `RtDim::Node` chain into `FusedElem`, fuse a dynamic-axis dependency, merge
   differently stamped producers, duplicate checked/effectful work, omit the
   structural broadcast reference, or fuse after
   finalization fail before emission. Overflow, division-by-zero, explicit-trap,
   and effectful user-function negatives assert unchanged occurrence count,
   order, and attribution through both vmap and fusion. The
   batch-varying element-derived row
   rejects exactly under the amended `spec/06` rule.
8. **Axis-source mutations.** The Phase-2-frozen `OutputAxisRule` bijection
   covers every current `RiscOp` output axis, including external
   `Load`/root axes and non-movement computed-shape operations. Phase 4
   consumes those exact rows to derive the view, which is recomputed and
   validated after vmap, grad, specialization, fusion, cloning/remapping, and
   DCE. C4
   cardinality and mapping
   corruptions fail before emission with the registered typed receipt; no
   mutation, cached pre-transform map, or stale `NodeId` is accepted,
   silently repaired, or allowed to reach an ICE.
9. **WireDag v8.** Exact JSON round-trip, stable bytes/hash, prove and offline
   extraction, compiler-API and binding consumption, and the capacity census
   are green. Structural decoder negatives include missing size, old or future
   version, illegal bound tag, missing or out-of-range input slot, later-node
   reference, non-scalar source, wrong dtype, malformed `input_axis` tensor or
   axis slot, wrong axis dtype, missing/duplicate/reordered/stale declaration
   or class member, invalid lexical/scope-instance owner, invalid synthesized
   origin or clone lineage, a noncanonical history/cursor pair or allocation
   after exhaustion,
   a duplicate complete origin/generated-path identity or conflicting
   mutation-site tags under one transform ID, malformed
   scope-instance lineage, or duplicate complete owner identity,
   missing source/use ID or stamp, member/stamp
   disagreement, correlated stamp-plus-manifest removal with its declaration
   fixed, valid-axis substitution from another class, split/merged class
   carriers that disagree with the declaration, dropped otherwise-unused
   public Load or guard root, `input_axis` in any Pad/Shrink/Stride field or
   another owner-illegal tag, classed `to_end`, unshifted
   vmap source axis, and incompatible axis/rank/output shape. Dynamic-axis
   placement mutations put a node-valued witness on one static axis, omit or
   duplicate its graph-level carrier, or reuse its source in both placements.
   Namespace mutations cover `(some(u64::MAX - 1),
   available(u64::MAX))`, allocation of the last ID, canonical transition to
   `(some(u64::MAX), exhausted)`, mismatched history/cursor pairs, another
   transform after exhaustion,
   transform-create/full-DCE/refinalize, highest-live-ID DCE, stable no-op
   decode/refinalize, near-exhaustion DCE followed by allocation, post-decode
   allocation, and an imported history range requiring more IDs than remain.
   They require the history-sensitive hash to change when a transform is issued
   and later wholly erased, while identical graph plus identical history stays
   byte-stable. Literal mutations cover negative and greater-than-`i64::MAX`
   Wire integers plus 32-bit/64-bit decode parity across `WireRtDim`,
   `WireDimInfo`, `WireDimExpr`, `WireRiscOp::OneHot`, every output-type copy,
   `DimExprKey`, evaluate/binding API, and serialized `ExecutionDim` path.
   Mutations narrow each census row independently to `usize`/`u32`, cast during
   normalization/lowering/binding/serialization, or accept an unsigned value
   above `i64::MAX`; every mutation fails before IR
   consumption or physical allocation. Origin-order mutations vary
   insertion order, reorder the serialized table, permute imported operands,
   and stale a generated-path component. Insertion order alone cannot change
   bytes, but declared operand order and transform high-water history can.

   Whole-program integrity is a separate oracle. It coherently removes or
   replaces an internal declaration and every dependent graph/carrier field;
   structural decode may succeed because the result is another valid program,
   but comparison with the caller-supplied expected fixture/content hash must
   fail. The expected commitment is outside the mutated Wire bytes. Tests never
   call coherent semantic replacement a structural decoder rejection.

Positive rows use this allowed transition lattice:

```text
nonconforming_rejection | ice | lane_divergent
    -> typed_unsupported(issue)
    -> executes_exactly
```

A positive row may move only right, although it may skip the interim receipt.
`typed_unsupported` is allowed only for a named interim Phase 4 row and must
carry the exact registered issue receipt. Negative controls remain in the
separate terminal state `rejects_exactly` with their owning diagnostic or
trap. A lane that already accepts a positive row may not regress, and an
`executes_exactly` row may not change shape, value, trap, or serialized
meaning. A phase invocation requires its owned rows at their exit state and
rejects unexplained per-lane changes in every other row.

### C6 Coordinated exact WireDag v8 migration

Phase 2 changes the public compiler API and therefore carries the complete
wire change in the same implementation change. Current `main` is v6.
Because prerequisite #1298 changes serialized `Shape.axis`, #1298
first amends `spec/10` and consumes v7. Runtime extents then refetches
that exact head, amends `spec/10` again, and consumes v8. A schema
advance before implementation shifts both numbers monotonically; the invariant
is that #1298 and runtime extents use two distinct successive versions and this
document, both trackers, fixtures, and oracles update together. A version is
never reused.

- amend `spec/10-serialization.md` and bump
  `WIRE_DAG_SCHEMA_VERSION` monotonically from #1298's v7 to v8;
- encode `WireRiscOp::Expand { axis, size: WireRuntimeExtent }`
  and use the same wrapper for every other runtime-extent owner;
  `WireRuntimeExtent` contains one `value: WireRtDim` plus an optional
  `class: WireRuntimeDimRef { class, use_id }`. `input_axis`
  carries a tensor input slot plus a
  `WireRtAxis` that is an exact int32 literal or scalar input slot. The `lit`
  tag carries an exact nonnegative `i64`, never `usize` or an unsigned JSON
  value beyond `i64::MAX`;
- change every static tensor-extent copy in the schema at the same version:
  `WireDimInfo::Lit`, the known-size value in `WireDimInfo::Named`,
  `WireDimExpr::Concrete`, `WireRiscOp::OneHot.vocab`, and all
  `WireTensorType` input/output dimensions are exact nonnegative `i64`.
  Public `CompiledExecutionArtifact` serialization likewise carries
  `ExecutionDim.size: Option<i64>`. Encode takes the exact in-memory `i64`
  without a host-width cast; decode validates the signed range without
  converting to `usize`. A checked physical-capacity conversion occurs only
  when a selected target allocates or indexes storage, never while parsing or
  constructing IR;
- encode the primary ordered `WireRuntimeDimDeclaration` list and
  `WireRuntimeTransformNamespace { issued_through: Option<u64>, next:
  available(u64) | exhausted }` at DAG level. `issued_through` is retained
  transform-history authority even when DCE removed every origin that used its
  highest IDs;
  Each non-optional declaration contains its stable local class ID, lexical
  scope, runtime scope-instance/clone lineage, binder slot, and complete
  ordered durable source/use IDs. Encode the canonical `RuntimeOrigin` table:
  a typed origin carries its lexical scope and typed site; a synthesized origin
  carries its transform instance, parent origin, and deterministic generated
  path. A tensor-axis source selects a tensor origin and a
  `literal_axis(int32)` or `scalar_axis(origin)`; the latter makes that exact
  rank-zero `int32` producer a liveness dependency. Other source/use entries
  select scalar-parameter, literal, scalar-operation-output, bound-field,
  alias-axis, or scalar-alias roles. Encode the derived
  `WireRuntimeExtentClass` separately, with the same ID and `literal`,
  `scalar`, or `tensor_axis` witnesses carrying their source IDs; a
  `tensor_axis` witness carries its tensor node and a literal or rank-zero
  `int32` scalar node for the axis. Encode one optional
  `witness { class, source }` or `alias { class, use_id }` stamp per static
  tensor output axis and the corresponding optional stamp on rank-zero
  exact-`int64` scalar producers. Encode every node-valued-axis source once in
  a separate ordered `WireRuntimeDynamicAxisWitness { class, source, tensor,
  axis }` list; no static axis stamp may repeat that source. The public input
  manifest retains every declared Load independently of data uses and maps each
  input axis/scalar to its declared source ID;
- interpret `WireRtDim::Node { input }` as an absolute index into the
  owning `WireDagNode.inputs`, then validate that referenced earlier
  node as rank-zero `int64`;
- interpret `WireRtDim::InputAxis { tensor, axis }` as a structural tensor-axis
  witness only where the owner matrix admits it: a folded direct `shape` read
  or proved dimension binder for expand/reshape. Validate the tensor slot as
  an earlier tensor node and a node-valued axis slot as an earlier rank-zero
  `int32` scalar. Pad/Shrink/Stride shape-derived bounds serialize the one
  materialized rank-zero exact-`int64` `Shape` producer through `node`;
- apply this exact owner/tag matrix at encode, decode, verification, and
  mutation generation:

  | owning field | legal `value` tags | class rule |
  |---|---|---|
  | `Expand.size` | `lit`, `input_axis`, `node` | optional; required for a surviving binder use |
  | `Reshape.new_shape[*]` | `lit`, `input_axis`, `node` | optional; required for a surviving binder use |
  | `Pad.padding[*].before/after` | `lit`, `node` | optional; required for a surviving binder use |
  | `Shrink.bounds[*].start` | `lit`, `node` | optional; required for a surviving binder use |
  | `Shrink.bounds[*].end` | `lit`, `node`, `to_end` | optional except `to_end`, which requires absent |
  | `Stride.strides[*]` | `lit`, `node` | optional; required for a surviving binder use |

  `sym` is illegal in every executable owner. An owner/tag pair absent
  from the table is invalid; there is no generic permissive arm;
- validate the tensor operand, axis, input/output ranks, and exact output-axis
  mapping. Start from the non-optional declarations, resolve every durable
  source and use through the public-input map, stamps, dynamic-axis witnesses,
  or bound references, and
  apply the Phase-2-frozen `OutputAxisRule`; then require exact equality with
  the serialized class manifest. Every lexical scope, scope instance, clone
  lineage, typed/synthesized origin, transform instance/path, source/use ID,
  member node/axis, literal, dtype, rank, class ID, uniqueness, canonical
  order, and control/liveness edge is checked before encode and after
  exact-version decode. The history/cursor pair must be exactly `(none,
  available(0))`, `(some(n), available(n + 1))` below `u64::MAX`, or
  `(some(u64::MAX), exhausted)`. Every live synthesized origin's transform is
  at or below `issued_through`; an unreferenced origin is omitted, while the
  high-water mark remains. Allocation from `exhausted` and any overflowing
  import fail `Capacity` before mutation. Duplicate complete origin/generated
  path identities, conflicting mutation-site tags under one transform ID, and
  duplicate complete owner identities are invalid; distinct binder slots may
  share one valid scope-instance lineage. Importing another annotated DAG
  canonically remaps its complete history range before combination; decode never accepts
  an overlapping raw import. A stamp/manifest pair cannot establish or resize
  a class;
- reject v6, #1298-only v7, versionless, future, string-size, executable
  `sym`, and owner-illegal `to_end` spellings before IR
  consumption; no legacy
  conversion or default exists. Named values must already be resolved through
  their structural value and declared class/use pair before encoding; display
  strings have no role in class construction or verification;
- update stable hash/prove/Beacon fixtures and every compiler-API or binding
  consumer that exposes WireDag bytes or version names. Hashing includes
  `issued_through` and is intentionally sensitive to transform-ID allocation
  history: a transform later erased by DCE changes the hash, while identical graph and
  identical history serialize identically;
- keep structural decode and artifact integrity distinct. Generic v8 decode
  accepts any internally valid program and rejects carrier inconsistencies.
  Trusted fixture/cache/offline loaders additionally accept an expected hash
  or exact bytes from outside the Wire payload and reject coherent replacement.
  No digest inside the same payload is treated as self-authentication;
- regenerate and review the typed wire capacity census. Every changed
  descriptor receives a final authority classification; the schema bump does
  not inherit or create a legacy exemption. The semantic-extent transit census
  and Wire census must be bijective over every field, enum payload, public
  artifact, derived key, function/map/intermediate value, operation-defined
  extent, and conversion path. A host-sized semantic transit in either census
  is a build failure.

## Part II: boundary law

- Each phase exit freezes its corpus rows, public type shapes, and exact oracle
  command. Changing one updates this document and [#1277] together.
- Controls never move to bless an implementation. A red row becomes green only
  when the tree changes.
- A discovery mid-phase becomes its own child issue and named corpus row rather
  than silently widening the phase.
- Language behavior is derived from the controlling numbered spec. If the
  three C3 actions do not decide a context, amend the numbered spec first.
- New or changed numeric identities follow the [05-OP-N] registration and
  rejection-registry regeneration rules. Wire changes also run the typed
  capacity census.

## Part III: phases

### Phase 0 -- oracle skeleton and measured baseline

**Deliver:** `scripts/runtime_extent_oracle.py`, generated corpus,
checked-in per-row baseline, allowed-transition validation, exact-head/corpus
digest, and host/GPU evidence handoff.

**Frozen at exit:** row identities, status vocabulary, mutation set, and
platform evidence schema.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 0` exits zero with final line
`RUNTIME EXTENT ORACLE: BASELINE OK` and no unexplained row.

### Phase 1 -- diagnostic residue only

**Deliver:** [#1367]. Remove obsolete Form-3 text and
`cast(N, int32)` extent recommendations while preserving correct
`int32` axis guidance. Do not change typing and do not close [#1112].

**Frozen at exit:** the diagnostic/comment census and its axis-domain positive
controls.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 1`. The runner executes
`cargo nextest run -p chelis-types --test issue_1112_extent_dtypes` as a
supporting leg and the diagnostic census. The focused test confirms the
language-level dtype rows only; it is not GPU completion evidence.

### Phase 2 -- runtime extent representation, WireDag v8, then one resolver

**Entry requirements:** Phase 0; [#1112]'s HIP carrier/guard/widening and Metal
audit landed with focused width, capacity, and guard tests; both target suites
build and their hardware-availability smoke reports are recorded at the exact
head; #1298 is closed and its complete dynamic-axis/window oracle passes at the
integration head; and the exact normative and WireDag changes are ready to land
atomically. The entry record verifies #1298 consumed its distinct v7 schema;
the C5 runtime-extent groups cannot be an entry gate because this
phase creates them.

**Deliver in order:** first amend `spec/05`'s closed `RtDim`
and [05-OP-7] representation to admit `RuntimeExtent` and the structural
tensor-axis witness with a static-or-node axis for expand/reshape, while
preserving section 2.4.1's node-valued `Shape` representation for
Pad/Shrink/Stride; migrate every semantic static-extent `RtDim`, `DimInfo`,
`DimExpr`, tensor-type, and Wire copy to exact nonnegative `i64` under one
generated transit census that also covers derived keys, evaluation/binding
APIs and maps, public execution artifacts, operation-defined extents,
intermediates, and conversion sites; admit exact scalar-operation-output
sources, the independent provisional authority, sealed annotated graph, and
closed atomic graph/authority transaction API,
provisional hygienic annotations, the typed/synthesized origin algebra,
the exact structural origin comparator, a serialized checked transform cursor
with retained hashed high-water history and explicit exhaustion, lexical and
runtime scope-instance ownership,
post-transform primary binder declarations with durable source/use IDs, stable
class IDs, static/scalar stamps plus the graph-level dynamic-axis witness
carrier, the complete workspace production
mutation/import/deserialization-capability and callsite registry, the removal
of public raw-DAG construction, mutation, serde, artifact-extraction, cache,
Eval, and backend escape routes,
compute-once `BroadcastScalarRef` adaptation and fusion barriers, source
witness/alias roles, the complete in-memory owner
matrix, and the complete graph-level
`RuntimeExtentClass` manifest plus the producer-ready, source-ordered
`RuntimeGuardSchedule`, and require named values resolved before
executable IR. Author any missing numbered operation atoms,
then generate and freeze the one complete `OutputAxisRule`/`OpExtentRule`
registry used both for pre-allocation operation-output guards here and total
`AxisSource` in Phase 4; amend `spec/06` for rank-zero bound slices,
compute-once dual-use scalar adaptation, and typed rejection of batch-varying
element-derived extents; and amend `spec/10`
for the next monotonic WireDag version (v8 from the current v6 baseline).
Write the derived positive, negative, equality-class/observable-event-schedule,
semantic-transit/cross-host, owner-matrix, and transform test stubs before
implementation. Then deliver C2.1-C2.6 and C6
across all in-memory, target, transform, and wire consumers, followed by C2.7
deletion. Close
[#1266], [#569], [#597], and [#609]. [#578] remains open; commits that improve
its mechanism use `Part of #578` until its complete rank-polymorphic
acceptance reproducer is green under the owning rank-polymorphism work.

**Frozen at exit:** amended `spec/05`/`spec/06` atoms;
`Expand.size: RuntimeExtent`; exact nonnegative-`i64` semantic extents in every
`RtDim`, `DimInfo`, `DimExpr`, tensor-type, and Wire copy, with `usize` only
behind checked physical-capacity conversions; structural tensor-axis and
scalar input values; the generated semantic-extent transit census over fields,
artifacts, keys, APIs/maps/intermediates, operation parameters, and conversions;
the exact `RuntimeExtent` owner matrix; provisional hygienic keys and the
independent provisional authority, sealed graph, atomic mutation transactions,
opaque annotated/finalized public states, finalized-only Eval/backend/cache
boundaries, and external-crate compile-fail raw construction, mutation,
deserialization, extraction, and submission boundaries; typed/synthesized origin and
transform-instance algebras and exact structural comparator; the serialized
checked transform cursor, retained history high-water, history-sensitive hash,
exhaustion behavior, and post-decode/import collision rules;
lexical/runtime-scope and clone-lineage
rules; post-transform primary binder declaration/ownership/source/use
identities; static-or-node tensor-axis and scalar-operation-output source
algebras; graph-level dynamic-axis witness placement; stable runtime binder IDs;
complete ordered equality-class manifests and source/use roles;
the one final-DAG-derived total observable-event order and guard schedule with
interface-only prologue checks, producer-ready local checks, exact `check_at`
events, adjacent predecessor/successor fences, guarded-event edges, and
source/effect/trap order;
the single generated `OutputAxisRule` and `OpExtentRule` variant/formula sets;
the complete workspace production mutation/import/deserialization capability
and callsite registry,
`BroadcastScalarRef`, trap/effect occurrence rules, and transform/fusion
behavior for every bound-dependency class;
WireDag v8; no provenance-rejection construct; one static folder; all-lane
guard placement.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 2`, including host, HIP, Metal,
WireDag, #569 transformation, every transform-bound class, named-dimension
witness sets and producer-ready guard schedules, greater-than-`u32::MAX`
semantic-transit/cross-host rows, generated-origin/clone-lineage rows, sealed
public
construction/mutation/deserialization/artifact/backend boundaries,
and entry-selection rows, compute-once vmap/fusion trap/effect rows, retained
high-water/DCE/hash rows, structural Wire negatives, coherent-replacement
integrity mismatches, zero, and negative rows owned by this phase, plus the
composed exact-head #1298 oracle.

### Phase 3 -- deferral totality and deterministic settlement

**Entry requirements:** Phase 2; the `spec/04` settlement-order
amendment; [#1341]'s ordered-store mechanism.

**Deliver:** the opaque `Inferred` carrier, mandatory
`consume_pending` choke point, exhaustive generated use-site registry,
closed `ExpectedShapeEvidence` decision matrix, all recursive
composite-carrier and non-builtin rows, evidence-qualified comparison/branch
routing, private action dispatch, source-order stores, compile-fail bypass
mutations, and generated action tests. Close [#1265] when its complete
reproducer is green.
Use `Part of #1338`; close #1338 only if every remaining acceptance row
owned by that issue is green, otherwise leave it open for its external work.

**Frozen at exit:** carrier privacy boundary, registry identities, evidence
variant set, normative action mapping, recursive composite dispositions,
settlement order, and K-run count.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 3`; every action row and K fresh
[#1338] processes produce one exact verdict.

### Phase 4 -- total output-axis sources

**Deliver:** consume the Phase-2-frozen complete
`RiscOp`-to-output-axis `OutputAxisRule` bijection to instantiate C4's
final-DAG `AxisSource` view; land C4.1-C4.3 including
`ExternalAxis` first as the typed interim ratchet; then land C4.4 across
Eval/C/HIP/Metal. No new role or formula table is authored here. Close [#665]
and [#592].

**Frozen at exit:** `AxisSource` variant set, final-DAG derivation point,
one-source-per-output-axis
cardinality, and removal of runtime-extent name and equality recovery.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 4` turns the named interim rows from
typed receipts to exact execution. The same managed-Python command with
`--phase final`, plus both GPU manual commands as platform legs of that
named suite, is the class completion oracle and ends with
`RUNTIME EXTENT ORACLE: PASS`.

## Part IV: bookkeeping

### Interlocks

- **[#729] / [#1112]:** owns the remaining HIP `int64` carrier
  boundary. This plan depends on it before Phase 2 all-lane claims; it neither
  reparents nor closes it.
- **[#1298]:** owns computed runtime axes and runtime reduction-window
  operands as one delivery issue. Phase 2 waits for it to close and composes
  its complete `dtype_dynamic_axis_window_oracle.py` at the exact
  integration head. No reduction-window row is duplicated here.
- **[#1341] / PR [#1366]:** owns reusable ordered-iteration mechanics and
  linting. This plan consumes the mechanism and owns the runtime-extent
  verdict. No sibling document content is part of this PR.
- **[#731]:** owns witnessed checker errors. Remaining rejections and [#609]'s
  rank error use that channel.
- **Rank-polymorphism plans:** own whether named-axis forms are legal inside a
  `..r` body. This plan provides the resolution mechanism wherever
  the normative spec permits the form. [#578] remains open until its complete
  acceptance reproducer is green; this plan never claims a half-closed issue.
- **[#730]:** owns the typed `Unsupported` receipt used by C4's
  interim transition.

### Issue map

| issue | owning clause | phase |
|---|---|---|
| [#1367] | stale `int32` extent and Form-3 guidance | 1 |
| [#1266] | record projection rejected by provenance walk | 2 |
| [#569] | real lint/fmt transformation breaks a legal extent | 2 |
| [#597] | static expand gets no evaluator type metadata | 2 |
| [#609] | wrong-rank ascription is accepted | 2 |
| [#578] | mechanism evidence only; full rank-polymorphic repro stays open | external rank-polymorphism work |
| [#1265] | comparison consumer never selects the deferred shape | 3 |
| [#1338] | coupled defaults settle nondeterministically | 3 |
| [#665] | movement-op runtime wildcard is lost across Expand | 4 |
| [#592] | grad/vmap backward Expand reaches the same missing source | 4 |
| [#1112] | HIP metadata-carrier width; external entry prerequisite | [#729], before Phase 2 |
| [#1298] | runtime axes and windows; full external prerequisite/oracle | before Phase 2 |

### Not owned here

Data-dependent output ranks or shapes ([#600]), type-level dimension
arithmetic ([#526]), grad's symbolic-window gaps ([#513]), implementation of
runtime axes and windows ([#1298], an explicit full prerequisite), the
rank-polymorphic legality half of [#578], sibling symbolic-dim defects not yet
parented to [#1277], and dtype-semantics decisions.

[#513]: https://github.com/Chelis-Lang/chelis/issues/513
[#526]: https://github.com/Chelis-Lang/chelis/issues/526
[#569]: https://github.com/Chelis-Lang/chelis/issues/569
[#578]: https://github.com/Chelis-Lang/chelis/issues/578
[#592]: https://github.com/Chelis-Lang/chelis/issues/592
[#597]: https://github.com/Chelis-Lang/chelis/issues/597
[#600]: https://github.com/Chelis-Lang/chelis/issues/600
[#609]: https://github.com/Chelis-Lang/chelis/issues/609
[#665]: https://github.com/Chelis-Lang/chelis/issues/665
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#1112]: https://github.com/Chelis-Lang/chelis/issues/1112
[#1265]: https://github.com/Chelis-Lang/chelis/issues/1265
[#1266]: https://github.com/Chelis-Lang/chelis/issues/1266
[#1277]: https://github.com/Chelis-Lang/chelis/issues/1277
[#1298]: https://github.com/Chelis-Lang/chelis/issues/1298
[#1338]: https://github.com/Chelis-Lang/chelis/issues/1338
[#1341]: https://github.com/Chelis-Lang/chelis/issues/1341
[#1366]: https://github.com/Chelis-Lang/chelis/pull/1366
[#1367]: https://github.com/Chelis-Lang/chelis/issues/1367
