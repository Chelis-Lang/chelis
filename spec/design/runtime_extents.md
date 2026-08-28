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
**Class fixed:** [#1277] -- a direct `shape()` extent is the
folded movement-node metadata expression required by [05-OP-7], and every
other non-literal tensor extent is ordinary typed integer dataflow. The current
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

The target `expand` node is:

```rust
RiscOp::Expand {
    axis: usize,
    size: RtDim,
}
```

Only `RtDim::Lit` and `RtDim::Node` are legal for
`Expand.size` ordinary values. A third legal variant is the structural tensor
axis witness used by a folded direct-shape read or a proved in-scope dimension
binder:

```rust
RtDim::InputAxis {
    tensor: usize,
    axis: RtAxis, // Lit(int32) or Node(input slot with rank-0 int32)
}
```

`tensor` and every `Node` are absolute slots in the owning movement
node's `inputs`. `RtDim::Sym` and `RtDim::ToEnd`
remain illegal in `Expand.size`. In executable IR, a named
`reshape` target also resolves to `Lit`, `InputAxis`, or
`Node`; `Sym` may exist only in the typed pre-monomorphization
carrier and must not cross verification or WireDag.

- **C2.1 Exact representation invariant.** `inputs[0]` is the tensor
  operand. `RtDim::Node(i)` is an absolute slot in the same node's
  `inputs`, with `1 <= i < inputs.len()`. The referenced node is
  earlier in topological order and has rank zero and exact `int64`
  dtype. A `shape(tensor, axis)` value whose use is the movement extent,
  including use through a transparent binding, does **not** materialize a
  `RiscOp::Shape` value node for that use: it becomes
  `RtDim::InputAxis`, whose tensor slot is an earlier tensor node and
  whose literal or node-valued axis is exact `int32`. The movement node
  reads that input's metadata directly. The typed producer retains this
  `TensorAxisWitness` identity structurally across bindings; it is not
  recovered by a syntax/provenance walk. If the same binding also has an
  ordinary scalar consumer, lowering materializes `RiscOp::Shape` for that consumer while
  the movement use remains folded. A cast, arithmetic expression, or
  user-function result is ordinary scalar dataflow and reaches the movement
  node as `RtDim::Node`.

  A bare in-scope dimension value such as `a` in
  `c: tensor[a, f32]` uses the same `InputAxis` carrier. The
  typed environment maps binder identity, never spelling, to its runtime
  witnesses. A literal instantiation becomes `Lit`; a tensor witness
  becomes `InputAxis`; and a scalar term witness becomes `Node`.
  When several tensor axes witness one binder, the first declaration in
  signature/source order is the canonical carrier and equality guards compare
  every additional witness before use. A reusable generic may retain the
  binder in typed pre-monomorphization state, but a complete executable DAG
  must resolve it to one of these three forms. Missing witness, wrong binder,
  and violated witness equality fail loudly; no pass searches for a matching
  string.
  Thus the direct read obeys [05-OP-7]'s folded-`DimExpr` rule while every
  scalar value obeys `spec/04` §4.7.4. Both forms are real owning-node input
  dependencies, so DCE cannot lose them; a symbolic name or shape-only side
  table is not an equivalent size carrier.
- **C2.2 Static values are an optimization.** A nonnegative statically proved
  value, including zero, may use `RtDim::Lit`. One checked static
  folder is shared or contract-tested across checker and lowering. Failure to
  fold produces the exact `InputAxis` or `Node` carrier dictated by
  C2.1; it never rejects the expression or guesses a value.
- **C2.3 Every result is constructed.** The checker always constructs an
  `expand` result tensor whose rank is the selected operand rank or
  operand rank plus one. It stamps complete type metadata and checks declared
  or ascribed rank and dimensions. The early exit that causes [#597] and
  [#609] is deleted.
- **C2.4 Every consumer lands before deletion.** Verification, Eval, C, HIP,
  Metal, specialization, AD, vmap, hashing, and cloning/remapping passes read
  `Lit`, `InputAxis`, and `Node` before any provenance
  rejection is removed. [#1112]'s HIP carrier work and Metal audit are Phase 2
  entry requirements because a checker-only `int64` result is not an
  all-lane extent contract. Entry requires their focused width, capacity,
  guard, target-build, and hardware-availability smoke evidence; execution of
  this plan's runtime-extent group is a Phase 2 **exit**, not an entry
  requirement.
  Equality and negativity guards run before allocation or element access on
  every lane.
- **C2.5 Transforms preserve the bound slice.** This clause is the proposed
  rule for the required `spec/06` amendment; current `spec/06`
  says every node is batched, so the numbered spec wins until that amendment
  lands. Transforming a movement node
  transforms its complete bound-dependency slice, not only the tensor operand.
  Under axis-zero `vmap`, non-tensor scalar parameters remain shared, direct
  `InputAxis` reads observe the corresponding original tensor axis after
  the inserted batch-axis shift, and scalar `shape`/integer nodes used only by
  movement bounds remain rank-zero rather than acquiring a batch dimension.
  Literal axes normalize against the original source rank and then shift;
  node-valued axes perform the same checked normalization and shift at runtime.
  When one scalar producer has both bound and ordinary value consumers, vmap
  splits its uses: the bound dependency is cloned as the rank-zero shared
  slice, while an ordinary result follows `spec/06`'s batched value
  rule. A direct `shape` use needs no scalar clone for the bound edge;
  `InputAxis` reads the vmapped tensor metadata. The amended spec and
  tests cover literal, scalar-parameter, shape, arithmetic, and dual-use slices.
  Grad, specialization, cloning, and remapping preserve or remap every absolute
  input slot and its dtype/rank invariant. A bound computed from vmapped tensor
  *elements* could vary per example. The same `spec/06` amendment must
  choose its regular-stack behavior (an equality guard or a typed rejection).
  Implementation may not batch a bound-only scalar, guess, or silently share a
  varying value before the numbered rule and its positive/negative tests land.
- **C2.6 The deletion is atomic with the usable replacement.** Only after
  C2.1-C2.5 and C6 are green does the phase delete `SizeClass`,
  `classify_expand_size`, `classify_arith_app`,
  `sourceless_expand_size_error`,
  `Env::size_provenance` and its plumbing, plus the lowerer's mirror
  rejections. No intermediate commit may accept a value the IR cannot carry.
  Best-effort identity recognition may survive only as refinement whose
  failure result is a fresh extent plus guard.

### C3 Positional expand uses one normative protocol

The implementation derives its action from `spec/04` §4.7.2 rather
than assigning semantic labels by intuition:

- **`Constrain(expected)`** is used when the context supplies an
  independent tensor rank or shape equation. A declared result, ascription,
  user-function parameter, branch join, generic instantiation, or builtin
  relation between tensor operands selects the unique candidate satisfying
  that equation. Comparisons constrain their operands even though their result
  is scalar or boolean.
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
action: PendingExpandUse)`. Only that module can inspect, clone, unify,
project, or finalize a pending token; there is no raw-`Type` escape hatch.
`PendingUseSite` is a closed enum generated bijectively from the builtin
declaration registry and the Deep expression-form registry, so a new builtin
or expression form fails compilation/regeneration until it has a disposition.

The recursive choke point walks tensor-bearing tuple, List, record, ADT, and
function fields rather than only a top-level tensor. Anonymous tuple
construction and closure capture propagate the corresponding token. A named
record/ADT constructor field constrains against its declared field type; List
construction constrains tensor elements through the homogeneous element
equation; tuple/record/ADT projection and pattern binding transfer the selected
token; record update constrains an updated tensor field against its declared
field shape while propagating untouched fields; branch joins, declared
results, ascriptions, user-function parameters, generic instantiation, and
builtin operand relations constrain;
an undeclared closure return propagates to the call boundary; and
complete-program finalization freezes. Generated compile-fail bypass tests
prove no inference helper outside the module can unwrap or copy the carrier.
Generated semantic tests execute both candidate outcomes and a contradictory
shape for every `Constrain` row, propagation followed by later
selection for every `Propagate` row, and the documented default for
every `Freeze` row. Mutations that omit the tuple-projection, named
constructor-field, List-element, record-update, pattern-binding, or
closure-return call fail before a checker result is returned. Constructor tests
exercise replacement, insertion, and a contradictory declared field.

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
    Literal { value: usize },
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
  it is not a wildcard fallback. A generated table is bijective with the
  complete current `RiscOp` registry and assigns every output axis to
  one of these five source classes. Adding an op, output, or rank rule without
  a complete row fails regeneration and compilation.
  If an op-computed axis has no governing numbered atom, Phase 4 authors that
  atom before registering or implementing the rule.
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
  computed-shape op use a closed rule citing their own numbered atom.
- **C4.3 Interim failure is typed.** The algebra first lands as a verifier and
  property ratchet. A currently unsupported but well-typed mapping yields the
  registered [#730] `Unsupported` receipt. It never reaches the
  occurrence-pass ICE and never substitutes an input extent.
- **C4.4 Target state consumes the algebra.** Eval and all backends consume
  `AxisSource` directly. Runtime extent flows no longer depend on
  `shape_source_for_axis`, `op_declared_output_axes`, or a
  search for a Load carrying the same string. Static symbolic identities may
  remain in type metadata, but no runtime size is recovered by name. This
  closes [#665] and [#592].

Required mutation tests cover every `RiscOp` registry row, external
Load/root axes, kept axes before and after an inserted
`Expand` axis, the inserted/replaced axis, same-rank versus
rank-increasing forms, every `Reshape` target, and omitted,
duplicate, wrong-shifted, out-of-range, and wrong-node sources.

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
   canonical signature-order tensor witness selection, all-witness equality
   guards, and loud missing/wrong-binder failure. Runtime windows are absent
   from this corpus; the composed #1298 oracle owns them.
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
   and checked overflow fail for the owning reason on every applicable lane.
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
7. **Transform integrity.** Positive rows cover `Lit`, direct-shape
   and named-binder `InputAxis`, shared non-tensor `Node`, and
   shape/arithmetic `Node` dependencies, including dual-use producers,
   through vmap, grad,
   specialization, cloning/remapping, and DCE. They verify exact rank-zero
   bound scalars, shifted literal and node-valued axes, preserved tensor-axis
   witnesses, absolute input slots, guards, shapes, and values. Mutations that
   prepend a batch axis to a bound scalar, fail to split a dual-use producer,
   omit an `Expand` bound from
   grad liveness, lose a source under DCE, fail to shift an axis, or retain a
   stale slot fail before emission. The batch-varying element-derived row
   executes or rejects exactly as the amended `spec/06` decides.
8. **Axis-source mutations.** A generated bijection covers every current
   `RiscOp` output axis, including external `Load`/root axes and
   non-movement computed-shape operations. C4 cardinality and mapping
   corruptions fail before emission with the registered typed receipt; no
   mutation is accepted, silently repaired, or allowed to reach an ICE.
9. **WireDag v7.** Exact JSON round-trip, stable bytes/hash, prove and offline
   extraction, compiler-API and binding consumption, and the capacity census
   are green. Missing size, old or future version, illegal bound tag, missing
   or out-of-range input slot, later-node reference, non-scalar source, wrong
   dtype, malformed `input_axis` tensor or axis slot, wrong axis dtype,
   unshifted vmap source axis, and incompatible axis/rank/output shape are
   negative controls.

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

### C6 Exact WireDag v7 migration

Phase 2 changes the public compiler API and therefore carries the complete
wire change in the same implementation change:

- amend `spec/10-serialization.md` and bump
  `WIRE_DAG_SCHEMA_VERSION` monotonically from 6 to 7;
- encode `WireRiscOp::Expand { axis, size: WireRtDim }`, permitting
  `lit`, `input_axis`, and `node` only for
  `expand`; `input_axis` carries a tensor input slot plus a
  `WireRtAxis` that is an exact int32 literal or scalar input slot;
- interpret `WireRtDim::Node { input }` as an absolute index into the
  owning `WireDagNode.inputs`, then validate that referenced earlier
  node as rank-zero `int64`;
- interpret `WireRtDim::InputAxis { tensor, axis }` as a
  structural tensor-axis witness for either a folded direct `shape` read
  or a proved dimension binder: validate the tensor slot as an earlier tensor
  node and a node-valued axis slot as an earlier rank-zero `int32`
  scalar;
- validate the tensor operand, axis, input/output ranks, and exact output-axis
  mapping before encode and after exact-version decode;
- reject v6, versionless, future, string-size, executable `sym`, and
  illegal `to_end` spellings before IR consumption; no legacy
  conversion or default exists. Named values must already be resolved through
  their structural witness before encoding;
- update stable hash/prove/Beacon fixtures and every compiler-API or binding
  consumer that exposes WireDag bytes or version names;
- regenerate and review the typed wire capacity census. Every changed
  descriptor receives a final authority classification; the schema bump does
  not inherit or create a legacy exemption.

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

### Phase 2 -- runtime extent representation, WireDag v7, then one resolver

**Entry requirements:** Phase 0; [#1112]'s HIP carrier/guard/widening and Metal
audit landed with focused width, capacity, and guard tests; both target suites
build and their hardware-availability smoke reports are recorded at the exact
head; #1298 is closed and its complete dynamic-axis/window oracle passes at the
integration head; and the exact normative and WireDag changes are ready to land
atomically. The C5 runtime-extent groups cannot be an entry gate because this
phase creates them.

**Deliver in order:** first amend `spec/05`'s closed `RtDim`
and [05-OP-7] representation to admit the structural tensor-axis witness and
require named values resolved before executable IR; amend `spec/06` for
rank-zero bound slices and batch-varying extents; and amend `spec/10`
for WireDag v7. Write the derived positive, negative, and transform test stubs
before implementation. Then deliver C2.1-C2.5 and C6 across all in-memory,
target, transform, and wire consumers, followed by C2.6 deletion. Close
[#1266], [#569], [#597], and [#609]. [#578] remains open; commits that improve
its mechanism use `Part of #578` until its complete rank-polymorphic
acceptance reproducer is green under the owning rank-polymorphism work.

**Frozen at exit:** amended `spec/05`/`spec/06` atoms;
`Expand.size: RtDim`; structural tensor-axis and scalar input
invariants; transform behavior for every bound-dependency class;
WireDag v7; no provenance-rejection construct; one static folder; all-lane
guard placement.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 2`, including host, HIP, Metal,
WireDag, #569 transformation, every transform-bound class, named-dimension
witnesses, zero, and negative rows owned by this phase, plus the composed
exact-head #1298 oracle.

### Phase 3 -- deferral totality and deterministic settlement

**Entry requirements:** Phase 2; the `spec/04` settlement-order
amendment; [#1341]'s ordered-store mechanism.

**Deliver:** the opaque `Inferred` carrier, mandatory
`consume_pending` choke point, exhaustive generated use-site registry,
all recursive composite-carrier and non-builtin rows, comparison-family
constraint routing, source-order stores, compile-fail bypass mutations, and
generated action tests. Close [#1265] when its complete reproducer is green.
Use `Part of #1338`; close #1338 only if every remaining acceptance row
owned by that issue is green, otherwise leave it open for its external work.

**Frozen at exit:** carrier privacy boundary, registry identities, normative
action mapping, recursive composite dispositions, settlement order, and K-run
count.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 3`; every action row and K fresh
[#1338] processes produce one exact verdict.

### Phase 4 -- total output-axis sources

**Deliver:** generate the complete `RiscOp`-to-output-axis
bijection; land C4.1-C4.3 including `ExternalAxis` first as the
typed interim ratchet; then land C4.4 across Eval/C/HIP/Metal. Close [#665]
and [#592].

**Frozen at exit:** `AxisSource` and `OpExtentRule`
variant sets, one-source-per-output-axis cardinality, and removal of
runtime-extent name recovery.

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
