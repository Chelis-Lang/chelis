# Runtime Extents: one resolver for non-literal tensor extents

**Status:** PROPOSED. No phase is implemented. Tracking issue: [#1277].
Code evidence in this document was rechecked on `main` at `53607a64`;
function and type names are the durable anchors and line numbers are a
convenience of that commit.
**Owning specs:** `spec/04-type-system.md` §4.7 (admissibility, identity,
zero and negative extents, the runtime extent guard placement rule, and the
§4.7.2 settlement order), `spec/05-risc-primitives.md` §2.4.1 (the `RtDim`
carrier set including `InputAxis`), [05-OP-7] (the folded direct read),
[05-MOV-1], and [05-DIM-1..3]; `spec/06-transformations.md` §3.7 (runtime
extents under `vmap`) and §8.6 (`batch_varying_extent`);
`spec/10-serialization.md` for the public WireDag encoding. This plan
implements those decisions and decides no language semantics of its own.
Where an earlier draft of this plan had to invent a rule, that rule now
lives in the numbered tier and this document cites it.
**Class fixed:** [#1277] - a tensor extent that is not a literal (a
`shape()` read, an `int64` parameter, a record projection, a user-function
result, checked arithmetic, or an in-scope dimension binder) is admitted by
the checker according to its syntactic provenance and is recovered by the
backend by string name. The numbered spec makes every well-typed `int64`
expression admissible and requires the same value, guard, and trap on every
lane. The four mechanisms that stand between those two states are
incomplete in ways that keep producing instances.

**Interlocks, not hidden scope:**

- [#729] owns extent and axis dtype semantics; its child [#1112] owns the
  32-bit HIP tensor metadata carrier. Slice B's HIP guard rows wait for it;
  nothing else here does.
- [#1298] owns computed runtime `shape` axes and reduction windows. This plan
  depends on it only for node-valued `InputAxis` axes and for wire-version
  ordering (Part IV).
- [#1341] owns the hash-order determinism mechanism, delivered by
  [`hash_order_determinism.md`](hash_order_determinism.md). This plan
  consumes ordered stores and owns the extent verdict they settle on.
- [#731] owns witnessed checker diagnostics; [#730] owns typed `Unsupported`
  receipts.
- [#1372] (DAG rebuild integrity: side-carried annotations surviving every
  graph rebuild) and [#1373] (exact-`i64` internal extent carriers) are
  separate work filed from this plan and named in Part IV; neither is a
  phase here.

## Summary

The controlling spec is newer than the implementation. `spec/04` §4.7 admits
a function parameter, local or top-level binding, record projection,
user-function result, cast, or checked arithmetic expression anywhere an
`int64` extent is expected, requires Eval, C, HIP, and Metal to execute the
same value and guards, and now states where a guard is evaluated and in what
order coupled positional-`expand` defaults settle. `spec/05` §2.4.1 now names
the carrier every executable extent uses. `spec/06` §3.7 now says what `vmap`
does with a runtime extent.

The implementation still has four separate recovery mechanisms, measured in
the next section. The fix is ordered around representation: Slice A gives
every accepted `expand` size a real graph carrier (a literal, a folded
tensor-axis read, or a scalar value edge), migrates every lane and the wire
format, and only then deletes the provenance rejections. Slice B makes the
equality guards that `spec/04` requires executable graph structure and gives
every realized output axis one checked source, so no runtime size is
recovered by name. Slice C makes the one genuine deferral (positional
`expand`) total and deterministic. One runner records the baseline and
enforces the allowed progression from an ICE or lane divergence, through a
typed implementation receipt, to exact execution.

## The evidence: four mechanisms, measured

1. **The provenance walk** (`classify_expand_size`,
   `crates/chelis-types/src/infer/app_shape_helpers.rs`; `SizeClass`;
   `Env::size_provenance` in `crates/chelis-types/src/env.rs:108`). Before
   accepting a runtime `expand` size the checker walks the expression looking
   for a path back to a tensor: literals and `cast` chains, `shape(t, axis)`
   on a bare or borrowed variable, `let`-bound aliases through a lexically
   scoped provenance map, and applications of a fixed arithmetic set.
   Everything else hits a fail-closed catch-all arm. A record field access
   ([#1266]), the lambda `chelis fmt` itself emits for `|> cast(int64)`
   ([#569]), and a symbolic-rank body ([#578]) are missing arms of that
   match, not defects in the language.
2. **Deferred shape candidates** (`crates/chelis-types/src/unify.rs:131`,
   `deferred_expand_constraints: Mutex<HashMap<TypeVar, _>>`). Positional
   `expand(x, axis, n)` genuinely defers a choice, insert a new axis or
   replace an extent, as constraints keyed by the unresolved result type
   variable. Selection fires only in `bind_tvar` (`unify.rs:1782-1846`) when
   that variable unifies against a non-variable type. The comparison family
   constructs its `bool` result and returns it without unifying anything
   (`infer/app_post.rs:570-589`), so a comparison can never select
   ([#1265]). The freeze-point fallback
   (`materialize_deferred_expand_defaults`, `unify.rs:996-1009`) iterates a
   `HashMap`, so coupled defaults settle in hash order and the same file is
   accepted or rejected at random ([#1338]).
3. **The IR carries the extent by name.** `RiscOp::Expand { axis, size:
   DimExpr }` (`crates/chelis-ir/src/dag.rs:870-873`) is the only movement
   operation whose bound is not an `RtDim`; `DimExpr` is `Concrete | Sym |
   Mul | Div` with no node reference (`dag.rs:140-146`). A `shape(x, k)`
   size therefore never becomes a value edge: lowering turns it into
   `DimExpr::from(x.output_type.dims[k])`, a name, and keeps `x` alive only
   through the side vector `DagNode.shape_deps`
   (`dim_expr_from_shape_arg_with_source`, `lower.rs:11385-11391`;
   `add_shape_dep`, `lower.rs:9155-9157`). Arithmetic over a read fails
   closed (`lower.rs:9109-9123`, "cannot materialize as an extent"), and a
   bare `int64` scalar with no tensor name fails after the node is built
   (`lower.rs:9169-9190`). The backend then recovers runtime extents by
   string: `symbolic_occurrences` (`dag.rs:1669`), `op_declared_output_axes`
   (`dag.rs:1901-1968`), and `shape_source_for_axis` (`dag.rs:2156-2210`)
   resolve every symbol to a `Load` or an op-declared axis and ICE on an
   undeclared occurrence. The `Expand` arm covers only the expand's own axis;
   an op-declared runtime axis arriving on the expand's input is dropped,
   which is [#665] and, through grad+vmap, [#592].
4. **The lanes disagree.** The C lane emits its equality guards in
   `emit_input_shape_preamble` (`crates/chelis-backend-c/src/emit.rs:1230-
   1274`) before any allocation and inline at op-declared sites; the HIP lane
   compiles a Load-declared symbolic `expand` (`emit_expand`,
   `crates/chelis-backend-hip/src/emit.rs:1799`, renders the size through
   `emit_dim_expr`, `3781-3784`, which prints `DimExpr::Sym` by name) but
   rejects every
   `RtDim::Node` bound and the `Shape` read itself
   (`crates/chelis-compiler-api/src/compiler.rs:4628-4694`); the Metal lane
   rejects node-valued bounds (`compiler.rs:4238-4259`). `vmap`
   (`crates/chelis-ir/src/vmap.rs:12`) prepends the batch axis to every node
   including rank-0 scalars, and the verifier then rejects the rank-1 bound
   source (`verify.rs:518-522`, `verify.rs:1471-1479`), so `vmap` over any
   runtime bound is unrepresentable today. The checker rejects a zero size
   (`infer/app_tensor.rs:985-996` and `1184-1193`) although `spec/04` §4.7.2
   prohibits only a negative one.

Three structural facts explain why fixing instances has not closed the class:

- **Three walks claim to be mirrors of each other and drift by construction.**
  The checker's `classify_expand_size`, the lowerer's static folder and
  shape-source dispatch (`fold_static_size`, `lower.rs:10933`), and the DAG's
  `shape_source_for_axis` each re-derive "where does this extent come from"
  over a different vocabulary. The checker accepts `mul(shape(x, 0), 2)`;
  the lowerer rejects exactly that spelling.
- **Polarity is inconsistent.** `SizeClass::Unknown` accepts and
  `Sourceless` rejects while the walk's catch-all rejects;
  `shape_source_for_axis` returning `None` means "op-declare the axis" at one
  call site and "ICE" at another.
- **The checker-to-backend channel is one `Dim::Name` in annotated type
  metadata.** The classifier's verdict is never serialized. The early
  return in `check_expand_signature` is simultaneously the [#597] hole (a
  static-sized expand result stays a type variable and gets no `type`
  metadata, so eval rejects what build accepts) and the [#609] hole (the
  same early return skips validating a declared rank, so a wrong-rank
  ascription is accepted and eval silently returns a contradicting rank).

## Part I: contracts

### C1 The controlling extent contract

Restated by citation. No clause below is stronger than the sentence it cites,
and the numbered spec, not this restatement, decides.

- **C1.1 Admissibility is typing, not provenance** (`spec/04` §4.7.2,
  §4.7.3, §4.7.4, §4.7.6). An `expand` size is any expression of exactly
  type `int64`; a `reshape` target is a `List[int64]` of statically known
  arity whose elements are arbitrary `int64` expressions. No stage may
  require a literal, a tensor-name source, or a privileged spelling.
- **C1.2 Identity is proof-gated; equality is guarded** (`spec/04` §4.7.2,
  §4.7.3, §4.7.6, and the identity-only movement rule in §4.7). A literal
  stays a literal; a direct read may keep an input dimension's identity only
  when ordinary type reasoning proves it; otherwise the result carries a
  fresh `(d-name {} *)` extent, and a declared literal or name that is not
  statically proved equal adds a runtime equality guard rather than making
  the value illegal. Every symbolic `shrink` axis mints a fresh extent,
  including a full-axis `(0, ToEnd)` slice whose realized extent equals the
  input's.
- **C1.3 Guard placement is a partial order** (`spec/04` §4.7, the runtime
  extent guard paragraph). A guard runs once, after its operands exist and
  before the first allocation or element access whose shape depends on it;
  interface-valued guards run at function entry in signature order; a guard
  over a locally computed value takes the source position of the operation
  that introduces the guarded extent relative to independent effects and
  traps; unrelated guards and operations may run in either order. A failing
  guard traps `Domain` under the introducing operation and names the
  disagreeing sources.
- **C1.4 Settlement is source order** (`spec/04` §4.7.2). Coupled positional
  `expand` defaults settle in the order their `expand` calls appear in the
  program, never in an implementation's iteration order.
- **C1.5 Zero is legal** (`spec/04` §4.7.2). A static negative extent is a
  type error; a runtime negative extent traps `Domain` before allocation or
  access; zero produces a tensor with the declared shape and logical element
  count zero. Physical backing is target-private and is not a cross-lane
  observation.
- **C1.6 Extents are `int64` and axes are `int32`** ([05-DIM-1..3]). The
  checker, movement signatures, and C carrier ship this split; the HIP
  metadata carrier is [#1112]'s.
- **C1.7 One carrier per owner** (`spec/05` §2.4.1 and §2.5.1). An
  executable extent is `Lit`, `Node`, `InputAxis`, or `ToEnd`; a symbolic
  name is a typed form that no executable owner admits. `expand` and
  `reshape` admit `Lit`, `Node`, and `InputAxis`; `pad`, `shrink`, and
  `stride` admit `Lit` and `Node` plus `ToEnd` for a `shrink` end. A direct
  `shape()` extent argument to `expand` or `reshape` is the folded `InputAxis`
  form; the same read bound to `pad`, `shrink`, or `stride` is a rank-0
  `Node`; a `reshape` target that restates a bystander named dimension reads
  it as `InputAxis` from the tensor that declares it.
- **C1.8 `vmap` shares extents** (`spec/06` §3.7, §8.6). A rank-0 extent
  value is not batched: it is evaluated once, shared across the batch, and
  its traps and effects occur once. An `InputAxis` read observes the axis it
  named before the batch axis was inserted. An extent that depends on the
  elements of a vmapped tensor is the type error `batch_varying_extent`.

### C2 Representation first, provenance deletion last

The target `expand` node and the equality-class carrier are:

```rust
RiscOp::Expand {
    axis: usize,
    size: RtDim,
}

enum RtDim {
    Lit(usize),         // exact width is the #729 child's work, not this plan's
    ToEnd,              // Shrink end only
    Node(usize),        // absolute input slot of a rank-0 int64 node
    Sym(String),        // transitional name-bound reshape target; deleted in Slice B
    InputAxis {         // folded tensor-axis read, expand/reshape only
        tensor: usize,  // absolute input slot of an earlier tensor node
        axis: RtAxis,
    },
}

enum RtAxis {
    Lit(i32),           // axis-domain literal, [05-DIM-1]
    Node(usize),        // absolute input slot of a rank-0 int32 node (#1298)
}

struct RuntimeDimClass {
    binder: RuntimeBinderId,          // the hygienic dimension binder
    members: Vec<RuntimeDimMember>,   // first member is canonical
}

enum RuntimeDimMember {
    LoadAxis { load: NodeId, axis: usize },
    TensorAxis { node: NodeId, axis: RtAxis },
    Scalar { node: NodeId },
    OpOutput { node: NodeId, axis: usize },
    Literal(i64),
}

struct Dag {
    // ... existing fields ...
    runtime_dim_classes: Vec<RuntimeDimClass>,
}
```

- **C2.1 Exact representation invariant.** `inputs[0]` is the tensor operand.
  `RtDim::Node(i)` is an absolute slot in the same node's `inputs` with `1 <=
  i < inputs.len()`; the referenced node is earlier in topological order, rank
  zero, and exactly `int64`. `RtDim::InputAxis { tensor, axis }` names an
  earlier tensor node that is also an input slot of the owning node: the read
  tensor becomes a shape-only input, so the owning node's data dependencies
  and shape dependencies are the same edge kind and DCE cannot lose one
  without the other. The `shape_deps` side vector survives only for uses that
  no `InputAxis` slot covers, and is deleted once
  `record_runtime_dim_shape_deps` and `op_declared_output_axes` no longer read
  it (Slice B). A literal axis is an exact `int32` already normalized into
  `0..rank(t)` (`spec/04` §4.7.1 normalizes a negative literal statically); a
  node-valued axis is an earlier rank-0 `int32` node and is admitted only
  after [#1298] lands the runtime `Shape.axis` operand. A cast, arithmetic
  expression, record projection, parameter, or user-function result is
  ordinary scalar dataflow and reaches every movement node as `RtDim::Node`;
  `pad`, `shrink`, and `stride` materialize a direct `shape()` read exactly
  once as a rank-0 `RiscOp::Shape` and bind it through `Node`, as `spec/05`
  §2.4.1 states. A bare in-scope dimension binder such as `a` in `c: tensor[a,
  f32]` uses the same owner-specific carrier: a tensor witness becomes
  `InputAxis` for `expand`/`reshape` and a materialized `Node` for the other
  three; a literal instantiation becomes `Lit`; a scalar witness becomes
  `Node`. The typed environment maps binder identity, never spelling, to its
  witnesses.

  The in-memory owner matrix is exact and matches the wire matrix:

  | `RiscOp` field | legal `RtDim` |
  |---|---|
  | `Expand.size` | `Lit`, `InputAxis`, `Node` |
  | `Reshape.new_shape[*]` | `Lit`, `InputAxis`, `Node` |
  | `Pad.padding[*].before/after` | `Lit`, `Node` |
  | `Shrink.bounds[*].start` | `Lit`, `Node` |
  | `Shrink.bounds[*].end` | `Lit`, `Node`, `ToEnd` |
  | `Stride.strides[*]` | `Lit`, `Node` |

  `Sym` has no executable owner in the decided rule (`spec/05` §2.4.1). Today
  lowering (`lower.rs:11777`), the grad adjoint reshape (`grad.rs:2182`), vmap
  (`vmap.rs:29` through `RtDim::from_dim_info`, `dag.rs:124`), and
  specialization (`specialize.rs:106`) still produce a name-bound `RtDim::Sym`
  reshape target that eval (`eval.rs:2214`), the C lane (`emit.rs:6340`), and
  `WireRtDim::Sym` execute and serialize by name. Slice A leaves that path
  untouched; Slice B migrates every producer to `InputAxis` on the declaring
  tensor (a shape-only input slot when it is not already an operand), deletes
  `RtDim::Sym` and `WireRtDim::Sym`, and makes `sym` a wire negative. - **C2.2
  Static values are an optimization.** A statically proved non-negative value,
  including zero, may use `RtDim::Lit`. One checked static folder is shared,
  or contract-tested for agreement, between the checker and lowering, closing
  the `div` drift between `INT_ARITH` and `fold_static_size`. Failure to fold
  produces the exact `InputAxis` or `Node` carrier dictated by C2.1; it never
  rejects the expression or guesses a value. The `size > 0` checks at
  `app_tensor.rs:985-996` and `1184-1193` and the verifier's `size must be >
  0` become negative-only rejections; a runtime negative value traps `Domain`
  before allocation. - **C2.3 Every result is constructed.** The checker
  always constructs an `expand` result tensor whose rank is the operand rank
  or the operand rank plus one, stamps complete type metadata, and validates a
  declared or ascribed rank and dimensions against it. The early exit that
  causes [#597] and [#609] is deleted. - **C2.4 Equality classes are
  executable graph structure.** Every hygienic dimension binder that survives
  to executable IR with more than one witness is one `RuntimeDimClass` in
  `Dag.runtime_dim_classes`. The first member is canonical; every other member
  is exactly one equality guard against it, placed by C1.3. Members reference
  nodes by `NodeId` and are remapped by the same `remap: HashMap<NodeId,
  NodeId>` every rebuild pass already threads for `shape_deps` and
  `merged_spans` (DCE `optimize.rs:185`, CSE `optimize.rs:303`, fusion
  `fuse.rs:203`, vmap `vmap.rs:3`, grad pruning `grad.rs:623`, the four
  specialization passes in `specialize.rs`, copy insertion and drop stripping
  in `lower.rs`, `splice_dag` at `lower.rs:7824`, and `bind_symbolic_dims` at
  `dag.rs:2352`). A member whose node a pass removes is discharged only when
  no surviving node depends on the guarded extent; otherwise the pass is
  invalid and the verifier says so. Two classes may reference one node, each
  with its own guard. A node that appears twice in one class, which is what
  `splice_dag` produces for `f(n, n)` because both parameter names map to one
  `NodeId` (`lower.rs:7830-7838`), keeps one member. The canonical member is
  the signature-first witness: the earliest member by node position, which for
  interface members is declared signature order; the remaining members follow
  node position, then `(kind, axis)` as tie-breakers, and classes follow their
  canonical member. No hash-map iteration and no display name participates. A
  class whose members are all interface values is never discharged: `spec/04`
  §4.7 evaluates its guard at entry regardless of data use, so DCE keeps the
  `Load`s it references even when nothing else does. This is deliberately
  graph-level: Load/Load, Load/op-output, and op-output/op-output equalities
  exist even when no movement bound owns them, which is what the C lane's
  name-grouped `SymbolicDimBinding.others` (`dag.rs:570-574`) approximates
  today and Slice B replaces. Any node referenced by a class member or by an
  `RtDim::Node` slot is a fusion barrier: it is never absorbed into a fused
  chain and keeps its identity; fusion among other nodes is unaffected. -
  **C2.5 Every consumer lands before deletion.** Verification, Eval, C, HIP,
  Metal, specialization, fusion, AD, vmap, CSE, DCE, hashing, and the wire
  encoder and decoder read `RtDim` in every owner and `runtime_dim_classes`
  before any provenance rejection is removed. Lane notes: - C:
  `emit_input_shape_preamble` already evaluates interface-valued guards at
  entry before any allocation and op-declared guards inline, which is the C1.3
  placement, though in symbol-name order because `symbolic_bindings` groups by
  `BTreeMap`; it changes from name grouping to the class list and from name
  order to signature order. - Eval: the same placement, expressed as ordinary
  dataflow plus explicit guard steps before the first dependent allocation. -
  HIP and Metal: their gates admit `Lit` and `InputAxis`, which are metadata
  reads (the same thing the HIP emitter does for `DimExpr::Sym` today), so a
  Load-declared symbolic `expand` keeps compiling. They reject `Node` with the
  existing [05-MOV-1] typed receipt until the runtime scalar path lands under
  [#1112]/[#1298]; that is a legal interim state in the C5 lattice, and an
  extent above `INT_MAX` on HIP remains [#1112]'s defect exactly as it is now.
  - Wire: `Expand.size` changes from a display string to `WireRtDim`, which
  gains an `input_axis { tensor, axis }` variant in Slice A and loses `sym` in
  Slice B; `runtime_dim_classes` is serialized in canonical order. The typed
  wire capacity census rows for every changed descriptor are regenerated and
  classified in the same change. - **C2.6 Transforms preserve the bound
  slice.** `vmap` follows `spec/06` §3.7: a rank-0 extent scalar keeps rank
  zero and is shared; an `InputAxis` literal axis shifts by one and a
  node-valued axis is normalized against the unbatched rank and then shifted;
  a bound derived from vmapped tensor elements is rejected as
  `batch_varying_extent` before lowering. When a scalar producer has both a
  bound consumer and an ordinary batched consumer it is evaluated once; if the
  ordinary branch needs a batched value, an `Expand` of the rank-0 node over
  the batch axis is already legal (the checker admits a rank-0 operand,
  `builtins.rs:1824-1842`, and the compiler emits that shape in
  `zero_tensor_node`, `lower.rs:7047-7085`), so no new operation is needed.
  `grad` preserves every absolute input slot and its rank/dtype invariant;
  bound scalars remain the zero-cotangent boundary `spec/05` §2.4.1 defines.
  Specialization, cloning, and remapping preserve or remap every slot and
  every class member through the pass's own `remap`. - **C2.7 The deletion is
  atomic with the usable replacement.** Only after C2.1-C2.6 are green for a
  lane does the slice delete `SizeClass`, `classify_expand_size`,
  `classify_arith_app`, `sourceless_expand_size_error`, `Env::size_provenance`
  and its plumbing, and the lowerer's mirror rejections at
  `lower.rs:9109-9190`. No intermediate commit may accept a value the IR
  cannot carry. Slice B separately removes `SymbolicDimBinding.others` and the
  name-grouped execution loops once C4 consumes the class list on every lane.
  Best-effort identity recognition may survive only as refinement whose
  failure result is a fresh extent plus a guard.

### C3 Positional expand uses one normative protocol

The implementation derives its action from `spec/04` §4.7.2 rather than
assigning semantic labels by intuition:

- **`Constrain(expected)`** applies when the context supplies an
  independently fixed tensor rank or shape equation: a declared result, an
  ascription, an already-instantiated user-function parameter or generic
  field, a branch join with independently resolved shape evidence, or a
  builtin relation with an independently resolved operand equation. It
  selects the unique candidate satisfying that equation. A comparison
  constrains a pending operand only when its other operand or the
  surrounding resolved evidence supplies an independent equation; two
  pending operands link and propagate their shared choice even though the
  comparison result is scalar or boolean.
- **`Propagate`** applies when a context can carry the same unresolved
  monomorphic candidate without requiring either rank: an anonymous tuple
  field, a closure capture, a fresh generic field or parameter, a singleton
  `List`, the seed element of an inferred `List`, and an undeclared closure
  return. It adds no evidence and cannot select or clone the choice.
- **`Freeze`** applies only at a freeze point §4.7.2 names, when a concrete
  tensor shape is required and no independent constraint selected a
  candidate: an axis within the input rank selects same-rank replacement and
  `axis == rank(input)` selects trailing insertion. When several results
  freeze together they settle in source order (C1.4), which the stores
  realize by keying deferred constraints in `TypeVar` allocation order (a
  `Vec`, never a `HashMap`); [#1341] owns the general mechanism and this plan
  owns the extent verdict.

Every inference rule that consumes a tensor invokes exactly one action, and
the comparison family stops constructing its result out of band: it routes
through unification so that a consumer which supplies a shape necessarily
binds its operands' pending candidates ([#1265]), and its
`expand -> add -> eq` variant closes with it. The action is derived from
resolved expected-shape evidence, not from container syntax: a concrete
declared record or ADT field, an already-instantiated generic field, a
declared `List` element type, and a later `List` element facing an
independently resolved accumulated element shape constrain; if a later
concrete element supplies the first independent shape, that evidence
constrains the earlier pending seed; tuple, record, and ADT projection and
pattern binding transfer the selected token; record update constrains an
updated field only when its resolved schema is independent.

The privacy lock is a named later deliverable of the same slice, not an
entry requirement: recursive expression inference returns an opaque
`#[must_use]` carrier whose pending tokens have private fields; a child
enters its parent only through one `consume_pending(site, value, evidence)`
choke point; the site enum is generated from the builtin registry and the
Deep expression-form registry so a new builtin or form fails compilation
until it has a disposition; and compile-fail tests reject a helper that
bypasses the choke point. Generated semantic tests execute both candidate
outcomes and a contradictory shape for every `Constrain` row, propagation
followed by later selection for every `Propagate` row, and the documented
default for every `Freeze` row.

### C4 Every realized output axis has one checked source

An exhaustive match over `RiscOp` variants catches a new operation but not a
missing flow through an existing operation; [#665] is the proof, since the
operation is already `Expand` and the missing fact is that an op-declared
axis on its input must flow through a kept output axis with an index shift.
The structural interface is therefore a total per-output-axis algebra:

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
  `AxisSource` per output axis; the vector length equals the output rank, and
  omission or duplication is invalid. `ExternalAxis` validates that `load` is
  the exact external-input `Load` and that `axis` exists; downstream nodes
  retain its `NodeId`, never its string name. `InputAxis` validates the
  tensor input slot and its literal or node-valued exact-`int32` axis.
  `ScalarInput` validates the earlier rank-0 exact-`int64` contract of C2.1.
  `OpComputed` is permitted only for a closed operation-and-axis
  `OpExtentRule` whose formula is the owning numbered operation atom; it is
  not a wildcard. The per-op extent table is bijective with the `RiscOp`
  registry: an operation, output, or rank rule without a row is a build
  failure, and a row without a governing atom is authored as a numbered-spec
  change first. Identity is decided separately and only by typed proof: a
  proved same identity keeps the source dimension's name as type metadata,
  an unproved cross-tensor read is a fresh extent with a class member and a
  guard, and equality of extent formulas alone never proves identity.
- **C4.2 Exact movement mappings.** Same-rank `Expand` maps every unchanged
  output axis to the same input axis and the replaced axis to its literal,
  `InputAxis`, or scalar size. Rank-increasing `Expand` maps axes before the
  insertion unchanged, the inserted axis to its size, and later output axes
  to input axis `output_axis - 1`. Each `Reshape` target maps to its literal,
  folded `InputAxis`, or scalar input. Only the identity movement axes that
  `spec/04` §4.7 names, `Stride` with literal step one and `Pad` with literal
  zero padding, pass an input axis through; every symbolic `Shrink` output
  axis and every other non-identity movement axis is `OpComputed` with a
  fresh class member, including a full-axis `(0, ToEnd)` slice. `Load` axes
  are `ExternalAxis`. Shape-preserving non-movement ops use the exact
  input-axis map; reductions, concatenation, convolution, and every other
  computed-shape op use a closed rule citing their own atom.
- **C4.3 Interim failure is typed.** The algebra first lands as a verifier
  and property ratchet. A currently unsupported but well-typed mapping yields
  the registered [#730] `Unsupported` receipt; it never reaches the
  occurrence-pass ICE and never substitutes an input extent.
- **C4.4 Target state consumes the algebra.** Eval and all backends consume
  `AxisSource` and `runtime_dim_classes` directly. Runtime extent flows no
  longer depend on `shape_source_for_axis`, `op_declared_output_axes`, or a
  search for a `Load` carrying the same string, and `SymbolicDimBinding.
  others` is deleted. Static symbolic identities may remain in type
  metadata, but no runtime size or guard is recovered by name. This closes
  [#665] and [#592].
- **C4.5 Sources are a final-DAG derived view.** `AxisSource` is neither
  stored in the DAG nor serialized nor carried across transforms.
  `derive_axis_sources(&Dag)` runs after the last rewrite and before final
  verification and emission, borrowing the immutable DAG so a stale view
  cannot outlive a mutation. Eval and each backend derive it only after
  vmap, grad, specialization, fusion, cloning, and DCE finish; focused tests
  derive it after each transform.

Required mutation tests cover every `RiscOp` row, external `Load` axes, kept
axes before and after an inserted `Expand` axis, the inserted or replaced axis,
same-rank versus rank-increasing forms, every `Reshape` target, and omitted,
duplicate, wrong-shifted, out-of-range, and wrong-node sources.

### C5 The class oracle has a reachable boundary

The authoritative named suite is `scripts/runtime_extent_oracle.py` plus its
two platform execution gates. The runner accepts `--phase 0` through
`--phase 4` and `--phase final`. Final automatic success exits zero with:

```text
RUNTIME EXTENT ORACLE: PASS
```

The suite records the exact commit and corpus digest so host, HIP, and Metal
evidence cannot be combined across different heads. Rows that need a
node-valued axis compose the [#1298] oracle at the same commit
(`scripts/dtype_dynamic_axis_window_oracle.py`, ending
`DTYPE DYNAMIC AXIS WINDOW ORACLE: PASS`); no other row depends on it, and
reduction windows stay wholly in that oracle.

1. **Host parity.** A generated legal matrix crosses extent-producing forms
   (literal and literal arithmetic, parameter, local and top-level binding,
   record projection, user-function result, cast, checked arithmetic, a bare
   in-scope dimension binder with one or several tensor witnesses, and direct
   or indirect `shape()` reads with literal or, once [#1298] lands, computed
   axes) with `expand`, `reshape`, `shrink`, `pad`, and `stride` where each
   form is legal. `check`, `eval`, and compiled-and-executed C agree on
   acceptance, shape, values, and traps. Named-dimension rows prove literal
   instantiation, canonical signature-order witness selection, and complete
   class guards for Load/Load, Load/op-output, and op-output/op-output
   classes, including a class with no movement-bound consumer, two classes
   sharing one member node with one execution and two guards, and the
   `f(n, n)` splice producing one member. Same-tensor direct reads keep their
   proved identity; unproved cross-tensor reads use the same `InputAxis`
   carrier and add a guard. A full-axis symbolic `shrink` `(0, ToEnd)` is a
   dedicated positive: same runtime extent, fresh type identity, fresh class
   member, and a guard before allocation when a declared equality claims it.
   Guard-order rows place an earlier independent effect or trap before a
   later mismatch and require the earlier event to win, and the converse.
2. **GPU build and execution.** The same named rows compile and execute in the
   HIP and Metal correctness suites, not merely through capability-gate
   rejection tests:

   ```sh
   scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
   PYO3_PYTHON="$(uv python find 3.11)" cargo test -p chelis-backend-metal --test gpu_correctness -- --ignored --test-threads=1
   ```

   Each command reports the runtime-extent group at the same commit and corpus
   digest as the host run. Rows a lane rejects with the [05-MOV-1] receipt sit
   at `typed_unsupported` in the lattice, never at a silent pass. 3.
   **Negative parity.** Static negative extents fail with the owning type
   error; runtime negative extents trap `Domain`. Wrong dtype, out-of-range
   axis, rank-contradicting ascription, malformed scalar input, named-binder
   witness mismatch, a missing class member, a split or merged class, and
   checked overflow fail for the owning reason on every applicable lane. Guard
   mutations that hoist a local producer to entry, cross an earlier effect or
   trap, delay past a dependent allocation, duplicate a producer or guard, or
   change failure attribution are exact negatives on every lane. A `shrink`
   mutation that forwards the input class instead of minting a fresh member
   fails before emission. 4. **Zero positives.** Literal-zero and runtime-zero
   rows cover positional replacement, positional insertion, and named-axis
   expansion; they assert the exact output shape, logical element count zero,
   no element access, and Eval/C/HIP/Metal agreement, inspecting logical
   metadata rather than pointer nullness or target-private backing. 5. **Real
   [#569] transformation.** The runner proves a direct spelling checks,
   evaluates, and compiles; copies it to a task-owned path; runs `chelis lint
   --fix` and `chelis fmt --inplace`; proves formatting is idempotent and
   parseable; runs `chelis lint --check` and the style-gated `check`, `eval`,
   and compiled C path; and compares type, rank, shape, and value with the
   control. A negative fixture proves the typed-pipeline safety gate
   suppresses a rewrite whose transformed program would not preserve the typed
   result. 6. **Deferral stability.** Every positional candidate row runs in K
   fresh processes; [#1338] is named and must settle to the source-order
   verdict every time. 7. **Transform integrity.** Positive rows cover `Lit`,
   expand/reshape direct-shape and named-binder `InputAxis`, pad/shrink/stride
   materialized `Node`, shared non-tensor `Node`, and arithmetic `Node`
   dependencies, including dual-use producers, through vmap, grad,
   specialization, fusion, CSE, DCE, cloning, and `splice_dag`. After each
   pass they assert rank-0 bound scalars, shifted literal and node-valued
   axes, absolute input slots, every class member by `NodeId`, class and
   member order, guards, shapes, and values. The fused-chain row places a
   runtime-dimension dependency on an elementwise node and proves fusion keeps
   the edge (the fused-chain branch of `rebuild_with_fusion`,
   `fuse.rs:222-288`, carries no `shape_deps` today, which is a probe item for
   the DAG rebuild integrity class). A negative-literal-axis row proves the
   stored `InputAxis` axis is the §4.7.1-normalized one, so the batch shift
   never selects the batch axis. Mutations that batch a bound scalar,
   re-evaluate a dual-use producer, drop a member under DCE, fail to shift an
   axis, shift an unnormalized axis, retain a stale slot, or merge two classes
   fail before emission. Overflow, division by zero, explicit traps, and
   effectful user-function rows prove exact occurrence count, order, and
   attribution across vmap and fusion, and the element-derived extent row
   rejects as `batch_varying_extent`. 8. **Axis-source mutations.** The C4
   cardinality and mapping corruptions fail before emission with the
   registered typed receipt; no cached pre-transform view or stale `NodeId` is
   accepted. 9. **Wire.** Exact JSON round trip, stable bytes and hash, prove
   and offline extraction, compiler-API and binding consumption, and the
   capacity census are green. Decoder negatives include a missing size, an old
   or future version, a string-valued `Expand.size`, `sym` anywhere once Slice
   B has removed it, `input_axis` in a `pad`/`shrink`/`stride` field, a
   classed `to_end`, a missing or out-of-range slot, a later-node reference, a
   non-scalar source, a wrong dtype, a malformed `input_axis` tensor or axis
   slot, a missing, duplicate, or reordered class or member, and an
   incompatible axis, rank, or output shape.

Positive rows use this allowed transition lattice:

```text
nonconforming_rejection | ice | lane_divergent
    -> typed_unsupported(issue)
    -> executes_exactly
```

A positive row may move only right, although it may skip the interim receipt.
`typed_unsupported` must carry the exact registered issue receipt. Negative
controls remain in the separate terminal state `rejects_exactly` with their
owning diagnostic or trap. A lane that already accepts a positive row may not
regress, and an `executes_exactly` row may not change shape, value, trap, or
serialized meaning. A phase invocation requires its owned rows at their exit
state and rejects unexplained per-lane changes in every other row.

## Part II: boundary law

- Each phase exit freezes its corpus rows, public type shapes, and exact
  oracle command. Changing one updates this document and [#1277] together.
- Controls never move to bless an implementation. A red row becomes green
  only when the tree changes.
- A discovery mid-phase becomes its own child issue and named corpus row
  rather than silently widening the phase.
- Language behavior is derived from the controlling numbered spec. If the
  three C3 actions do not decide a context, or a guard placement question is
  not answered by `spec/04` §4.7, amend the numbered spec first.
- New or changed numeric identities follow the [05-OP-N] registration and
  rejection-registry regeneration rules. Wire changes also run the typed
  capacity census.

## Part III: phases

Each phase names one authoritative oracle. Phase 1 needs only Phase 0's
runner; Slice C touches `chelis-types` only and may land before Slice B.

### Phase 0 - oracle skeleton and measured baseline

**Deliver:** `scripts/runtime_extent_oracle.py`, generated corpus, checked-in
per-row baseline, allowed-transition validation, exact-head and corpus
digest, and host/GPU evidence handoff.

**Frozen at exit:** row identities, status vocabulary, mutation set, and
platform evidence schema.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 0` exits zero with final line
`RUNTIME EXTENT ORACLE: BASELINE OK` and no unexplained row.

### Phase 1 - diagnostic residue only

**Entry requirements:** Phase 0's runner.

**Deliver:** [#1367]. Remove obsolete Form-3 text and `cast(N, int32)` extent
recommendations while preserving correct `int32` axis guidance. Do not change
typing and do not close [#1112].

**Frozen at exit:** the diagnostic and comment census and its axis-domain
positive controls.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 1`, which runs `cargo nextest run
-p chelis-types --test issue_1112_extent_dtypes` as a supporting leg plus the
diagnostic census. The focused test confirms language-level dtype rows only;
it is not GPU completion evidence.

### Phase 2 (Slice A) - the value edge and one resolver

**Entry requirements:** Phase 0. Nothing from [#1298] or [#1112]: the C
carrier is already `int64`, the HIP lane keeps its Load-declared symbolic
`expand` through `InputAxis`, and node-valued axes are simply not admitted
yet.

**Deliver in order:** derived test stubs for every C2 clause and every C5
row this slice owns; `Expand.size: RtDim` with `InputAxis { tensor, axis:
RtAxis::Lit }` and the exact owner matrix; the read tensor as a shape-only
input slot; one static folder; constructed results with rank validation
(C2.3); zero-extent acceptance (C2.2); the `spec/06` §3.7 vmap rule for
rank-0 extent scalars and `InputAxis` axis shifting, plus the
`batch_varying_extent` rejection; every lane, transform, and verifier
consumer (C2.5, C2.6); the wire change under the next monotonic
`WIRE_DAG_SCHEMA_VERSION` at landing, coordinated with [#1298] so the two
migrations use distinct successive versions and both trackers, `spec/10`,
fixtures, hashes, and rejected-version controls update together; the
regenerated typed wire capacity census; then C2.7's deletion. Close
[#1266], [#569], [#597], and [#609]. [#578] remains open; commits that
improve its mechanism use `Part of #578` until its complete rank-polymorphic
acceptance reproducer is green under the owning rank-polymorphism work.

**Frozen at exit:** `Expand.size: RtDim`; the `InputAxis` carrier with a
literal axis; the owner matrix in memory and on the wire; the single static
folder; the `vmap` bound-slice rule; the wire schema version; no
provenance-rejection construct in `chelis-types` or `chelis-ir`.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 2`: host parity, HIP and Metal
build-and-execute rows for `Lit` and `InputAxis` (with `Node` rows at
`typed_unsupported([05-MOV-1])`), zero rows, the [#569] transformation row,
transform rows, and wire rows this slice owns.

### Phase 3 (Slice B) - equality classes, output-axis sources, no name recovery

**Entry requirements:** Phase 2. [#1112] for the HIP guard rows only: HIP
must carry `int64` extents to compare them exactly, so host, C, and Metal
rows may exit first with the HIP rows at `typed_unsupported(#1112)`.
[#1298] for the node-valued `InputAxis` axis rows only.

**Deliver:** `Dag.runtime_dim_classes` seeded by typed inference from binder
identity, remapped by every rebuild pass, verified after each; guard
placement per C1.3 on Eval, C, HIP, and Metal; the fusion-barrier rule; the
`AxisSource` derivation and the per-op extent table (C4.1-C4.3 as a typed
ratchet first, then C4.4); migration of every name-bound `RtDim::Sym`
reshape target producer (`lower.rs:11777`, `grad.rs:2182`, `vmap.rs:29`,
`specialize.rs:106`) and consumer (`eval.rs:2214`, `emit.rs:6340`,
`WireRtDim::Sym`) to `InputAxis` on the declaring tensor, then deletion of
`RtDim::Sym`; deletion of `SymbolicDimBinding.others`, the name-grouped
execution loops, `shape_source_for_axis`, `op_declared_output_axes`, and,
once nothing reads it, `shape_deps`. Close
[#665] and [#592].

**Frozen at exit:** the `RuntimeDimClass` and `RuntimeDimMember` shapes,
canonical class and member order, the `RtDim` variant set without `Sym`, the
guard placement realization per lane, the `AxisSource` variant set, the per-op
extent table, the final-DAG derivation point, and the removal of
runtime-extent name recovery.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 3`: named-dimension and guard-order
rows on every lane, axis-source mutations, transform rows asserting class
membership after every pass, and the class wire rows.

### Phase 4 (Slice C) - deferral totality and deterministic settlement

**Entry requirements:** Phase 0; the `spec/04` §4.7.2 settlement-order
rule; [#1341]'s ordered-store mechanism, or, if it has not landed, the two
deferred stores keyed locally by `TypeVar` allocation order with a citation
to [#1341].

**Deliver:** the three-action protocol over resolved expected-shape evidence,
the comparison-family unification route, source-order settlement, the
recursive composite-carrier rows, and then the privacy lock. Close [#1265]
when its complete reproducer is green. Use `Part of #1338`; close #1338 only
if every remaining acceptance row owned by that issue is green.

**Frozen at exit:** the action mapping, evidence variant set, recursive
composite dispositions, settlement order, carrier privacy boundary, and
K-run count.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 4`; every action row and K fresh
[#1338] processes produce one exact verdict. The same command with
`--phase final`, plus both GPU manual commands as platform legs, is the
class completion oracle and ends with `RUNTIME EXTENT ORACLE: PASS`.

## Part IV: bookkeeping

### Interlocks

- **[#729] / [#1112]:** owns the remaining HIP `int64` metadata carrier.
  Slice B's HIP guard rows depend on it; nothing else here does, and this
  plan neither reparents nor closes it.
- **[#1298]:** owns computed runtime `shape` axes and runtime reduction
  windows. This plan admits `RtAxis::Node` only after that runtime axis
  lands, and composes its oracle only for those rows. Whichever of [#1298]
  and Slice A lands first takes the next monotonic `WIRE_DAG_SCHEMA_VERSION`;
  the other takes the one after; both trackers update together and no
  version is reused.
- **[#1341] / [`hash_order_determinism.md`](hash_order_determinism.md):**
  owns ordered-iteration mechanics, the lint ratchet, and the K-run harness.
  This plan consumes them in C3 and C5 leg 6 and owns which verdict
  determinism settles on. [#1338] keeps [#1277] as its structural parent with
  an `Also part of #1341` cross-link.
- **[#731]:** owns witnessed checker errors; remaining rejections and the
  [#609] rank error use that channel.
- **[#730]:** owns the typed `Unsupported` receipt used by C4's interim
  transition and by the GPU lanes' `Node` rows.
- **Rank-polymorphism plans** (`rank_polymorphism.md`,
  `rank_polymorphism_tier3_followups.md`): own whether named-axis forms are
  legal inside a `..r` body; this plan provides the resolution mechanism
  wherever the numbered spec permits the form. [#578] remains open until its
  complete acceptance reproducer is green.
- **[#1372] DAG rebuild integrity:** owns the invariant that
  side-carried annotations (`shape_deps`, `merged_spans`, and now
  `runtime_dim_classes`) survive every graph rebuild. Slice B relies on that
  invariant through each pass's `remap`; the tracker owns making it
  structural (a shared rebuild helper, or stronger) and the fused-chain
  `shape_deps` probe named in C5 leg 7.
- **[#1373] exact `i64` internal extent carriers (a [#729] child):** owns moving
  `RtDim::Lit`, `DimInfo`, `DimExpr::Concrete`, the tensor-type copies, and
  their wire forms from host-sized `usize` to exact `i64`. This plan neither
  requires nor blocks it.

### Issue map

| issue | owning clause | phase |
|---|---|---|
| [#1367] | stale `int32` extent and Form-3 guidance | 1 |
| [#1266] | record projection rejected by provenance walk | 2 (Slice A) |
| [#569] | real lint/fmt transformation breaks a legal extent | 2 (Slice A) |
| [#597] | static expand gets no evaluator type metadata | 2 (Slice A) |
| [#609] | wrong-rank ascription is accepted | 2 (Slice A) |
| [#665] | movement-op runtime wildcard is lost across Expand | 3 (Slice B) |
| [#592] | grad/vmap backward Expand reaches the same missing source | 3 (Slice B) |
| [#1265] | comparison consumer never selects the deferred shape | 4 (Slice C) |
| [#1338] | coupled defaults settle nondeterministically | 4 (Slice C) / [#1341] mechanism |
| [#578] | mechanism evidence only; full rank-polymorphic repro stays open | external rank-polymorphism work |
| [#1112] | HIP metadata-carrier width; Slice B HIP guard rows | [#729] |
| [#1298] | runtime axes and windows; `RtAxis::Node` rows and wire ordering | [#729] |

### Not owned here

Data-dependent output ranks or shapes ([#600]), type-level dimension
arithmetic ([#526]), grad's symbolic-window gaps ([#513]), runtime axes and
windows ([#1298]), the rank-polymorphic legality half of [#578], the DAG
rebuild integrity class ([#1372]), exact `i64` internal carriers ([#1373]),
sibling symbolic-dim defects not yet parented to [#1277], and dtype-semantics decisions.

## Considered and rejected

An earlier draft of this plan, reviewed through twenty-four fresh-context
red-team rounds on PR [#1343], grew from 455 to 2,762 lines. Rounds through
the review of `9fe6cdad` found defects against the numbered spec or the code
(the [05-OP-7] fold, the [#1298] dependency, the wire-version collision, the
missing binder carrier, executable equality witnesses, `Constrain` for
declared fields) and those repairs are the C1-C5 above. Later rounds found
collisions between invariants the draft had invented for itself, and each
repair added structure that the next round found a seam in. The complete
earlier text is at PR head `067352ab`; each mechanism below is recorded with
the review that motivated it, why it is unnecessary once the numbered spec
decides the underlying rule, and what replaces it.

- **Synthesized-origin algebra and clone-lineage scope instances** (review
  of `f342e330`, repair `3496f159`). Gave every generated node a durable
  origin so class members could be matched across transforms. Replaced by:
  members are `NodeId` references remapped by the same `remap` every rebuild
  pass already threads; the corpus asserts membership after each pass.
- **Independent provisional authority, five finalization bijections,
  consume-to-annotated inverse, and non-forgeable evidence IDs with exact
  destinations** (reviews of `3496f159`, `d7782405`, `d6abd362`; repairs
  `37914d77`, `d6abd362`, `c70ae93d`). Existed because final IDs were
  assigned by canonical sort order after transforms had already stored them,
  so an earlier-sorting insertion staled every stored ID. Replaced by:
  classes are a `Dag` field whose members are never renumbered except by a
  pass's own remap; canonical order is computed at serialization.
- **Transform namespace with a `u64` cursor, exhaustion, and history-sensitive
  hashing** (reviews of `3496f159`, `37914d77`, `4215f760`; repairs
  `37914d77`, `4215f760`, `3d6d4746`). Existed to keep synthesized origins
  collision-free under a self-imposed no-ID-reuse rule, and made the artifact
  hash depend on which passes had run. Replaced by: no transform IDs; the
  hash covers the graph and its classes in canonical order.
- **Workspace-wide generated transaction registry and module-private
  `RawDag`** (reviews of `4215f760`, `3d6d4746`; repairs `3d6d4746`,
  `60d595b2`). Sealed every construction and mutation route across 1,424
  `add_node` call sites and every public `Dag` signature. The defect class
  it targets is real but is a class of its own; it is filed as [#1372],
  whose proportionate first step is a shared
  rebuild helper that carries every side vector so forgetting one is a type
  error. `spec/10` requires no hash stability that a sealed graph would buy.
- **`usize -> i64` semantic-extent transit census** (reviews of `60d595b2`,
  `d5c6fe0b`; repairs `d5c6fe0b`, `a0e74485`). Closes no instance in the
  issue map and pays off only on 32-bit hosts; it is filed as [#1373], a [#729]
  child, without the generated census, which an actual defect would have to
  motivate.
- **Total observable-event order with adjacent fences** (reviews of
  `60d595b2`, `d5c6fe0b`; repairs `d5c6fe0b`, `a0e74485`). Filled a gap the
  numbered spec left: where a guard runs relative to an independent earlier
  effect or trap. A total order over every allocation and access would also
  have pinned every lane to serialized per-node dispatch. Replaced by the
  partial-order rule now in `spec/04` §4.7 (C1.3), whose placement (entry,
  before allocation) the C lane already satisfies, though not yet its
  signature order.
- **Proof-sensitive placement table with separate value and output
  occurrences** (review of `bc9c3a44`, repair `d7782405`). Separated "what
  supplies the value" from "which class the result aliases" through a 3x5
  table over occurrence bindings. The distinction survives as C4.1's two
  sentences: value source from the op table, identity only from typed proof.
- **Closed `RuntimeBoundField` tags and the outer bound/output-axis
  discriminator** (reviews of `c70ae93d`, `2ca98512`; repairs `2ca98512`,
  `d3c8d39a`). Keyed evidence records to their destinations; without
  evidence records there is nothing to key.
- **Two-artifact import namespaces and `RuntimeInterfaceInputId`** (reviews
  of `2ca98512`, `d3c8d39a`; repairs `d3c8d39a`, `1cce1470`). Remapped typed
  scopes, sites, and interface positions when combining two independently
  serialized graphs. Chelis combines graphs through `splice_dag`, which
  remaps by `NodeId` and needs no second identity domain.
- **Scalar-actual substitution and substitution-alias authority** (reviews
  of `d3c8d39a`, `1cce1470`; repairs `1cce1470`, `067352ab`). Preserved two
  declared source IDs when `f(n, n)` bound two parameters of one class to
  one actual, to satisfy a one-witness-per-occurrence invariant. `splice_dag`
  already maps both parameters to one `NodeId`; a class keeps one member per
  node and the obligation is trivially satisfied.
- **`BroadcastScalarRef`** (review of `4215f760`, repair `3d6d4746`). A new
  Tier-1 operation to repeat a computed scalar across batch axes without
  re-executing it. `Expand` of a rank-0 node is already legal and emitted by
  the compiler; the "evaluate once under `vmap`" intent is `spec/06` §3.7's
  non-batching rule, a property of the transform rather than an operation.

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
[#1343]: https://github.com/Chelis-Lang/chelis/pull/1343
[#1367]: https://github.com/Chelis-Lang/chelis/issues/1367
[#1372]: https://github.com/Chelis-Lang/chelis/issues/1372
[#1373]: https://github.com/Chelis-Lang/chelis/issues/1373
