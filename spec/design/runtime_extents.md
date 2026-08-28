# Runtime Extents: one resolver for non-literal tensor extents

**Status:** PROPOSED. No phase is implemented. Tracking issue: [#1277].
Code evidence in this document was rechecked on `main` at
`12c22520`; function and type names are the durable anchors.
**Owning specs:** `spec/04-type-system.md` §4.7 and
`spec/05-risc-primitives.md` [05-DIM-1..3] plus the owning movement
atoms. `spec/10-serialization.md` controls the public WireDag
encoding. This plan implements those decisions; it does not weaken them to
match a current lane. The one open language decision it exposes -- coupled
positional-`expand` settlement order -- must land in
`spec/04-type-system.md` before Phase 3.
**Class fixed:** [#1277] -- every non-literal tensor extent is ordinary typed
integer dataflow, but the current checker, deferral machinery, and backend
recover that value through several incomplete provenance and symbolic-name
paths.

**Interlocks, not hidden scope:**

- [#729] owns extent and axis dtype semantics. Its child [#1112] still owns the
  HIP metadata-carrier widening; this plan may not close it with checker-only
  evidence.
- [#1298] owns runtime axes and reduction-window operands, including its own
  authoritative oracle. Reduction windows are not rows in this plan's oracle.
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
`expand` size a scalar value edge, migrates every lane and WireDag,
and only then deletes provenance rejections. Phase 3 totalizes the
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
  neither case: it produces a zero-element tensor with the declared shape and
  the runtime's null-data/zero-capacity empty representation.
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
`Expand.size`. `RtDim::Sym` and
`RtDim::ToEnd` remain illegal in that position.

- **C2.1 Exact scalar-edge invariant.** `inputs[0]` is the tensor
  operand. `RtDim::Node(i)` is an absolute slot in the same node's
  `inputs`, with `1 <= i < inputs.len()`. The referenced
  node is earlier in topological order and has rank zero and exact
  `int64` dtype. A `shape()` read or arithmetic chain is a
  real value dependency through that slot, so DCE, specialization, AD, and
  backend emission cannot lose it. A shape-only side table is not an
  equivalent size carrier.
- **C2.2 Static values are an optimization.** A nonnegative statically proved
  value, including zero, may use `RtDim::Lit`. One checked static
  folder is shared or contract-tested across checker and lowering. Failure to
  fold produces `RtDim::Node`; it never rejects the expression or
  guesses a value.
- **C2.3 Every result is constructed.** The checker always constructs an
  `expand` result tensor whose rank is the selected operand rank or
  operand rank plus one. It stamps complete type metadata and checks declared
  or ascribed rank and dimensions. The early exit that causes [#597] and
  [#609] is deleted.
- **C2.4 Every consumer lands before deletion.** Verification, Eval, C, HIP,
  Metal, specialization, AD, vmap, hashing, and cloning/remapping passes read
  the scalar edge before any provenance rejection is removed. [#1112]'s HIP
  carrier work and Metal audit are Phase 2 entry requirements because a
  checker-only `int64` result is not an all-lane extent contract.
  Equality and negativity guards run before allocation or element access on
  every lane.
- **C2.5 The deletion is atomic with the usable replacement.** Only after
  C2.1-C2.4 and C6 are green does the phase delete `SizeClass`,
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
Builtin rules carry an exhaustive `PendingExpandUse` entry; the same
closed registry has explicit non-builtin rows for declared results and
ascriptions, user-function calls, branch joins, generic instantiation,
bindings, and complete-program finalization. Adding a builtin or a consuming
inference rule without a row fails structurally. Generated tests execute both
candidate outcomes and a contradictory shape for every `Constrain`
row, propagation followed by later selection for every
`Propagate` row, and the documented default for every
`Freeze` row.

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
    InputAxis { input: usize, axis: usize },
    ScalarInput { input: usize },
    OpComputed { rule: OpExtentRule },
}
```

- **C4.1 Cardinality and ownership.** Every realized node supplies exactly one
  `AxisSource` for each output axis; the vector length equals output
  rank, and omission or duplication is invalid. `InputAxis` validates
  the input slot and input axis. `ScalarInput` validates the same
  earlier-node, rank-zero, exact-`int64` contract as C2.1.
  `OpComputed` is permitted only for a closed operation-and-axis
  `OpExtentRule` whose formula is the owning movement atom; it is not
  a wildcard fallback.
- **C4.2 Exact movement mappings.** Same-rank `Expand` maps every
  unchanged output axis to the same input axis and maps the replaced axis to
  its literal or scalar size. Rank-increasing `Expand` maps axes
  before the insertion unchanged, the inserted axis to its size, and later
  output axes to input axis `output_axis - 1`. Each
  `Reshape` target maps to its literal or scalar input; a proved name
  remains output type metadata, not a runtime name lookup. Identity
  `Shrink`, `Stride`, and `Pad` axes use
  `InputAxis`; non-identity axes use their exact
  `OpComputed` rule.
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

Required mutation tests cover kept axes before and after an inserted
`Expand` axis, the inserted/replaced axis, same-rank versus
rank-increasing forms, every `Reshape` target, and omitted,
duplicate, wrong-shifted, out-of-range, and wrong-node sources.

### C5 The class oracle has a reachable boundary

The authoritative named suite is `scripts/runtime_extent_oracle.py`
plus its two platform execution gates. The runner accepts
`--phase 0` through `--phase 4` and `--phase final`.
Final automatic success exits zero with:

```text
RUNTIME EXTENT ORACLE: PASS
```

The suite records the exact commit and corpus digest so host, HIP, and Metal
evidence cannot be combined across different heads.

1. **Host parity.** A generated legal matrix crosses extent-producing forms
   (literal and literal arithmetic, parameter, local/top-level binding, record
   projection, user-function result, cast, checked arithmetic, and direct or
   indirect `shape()` reads) with `expand`,
   `reshape`, `shrink`, `pad`, and
   `stride` where each form is legal. `check`,
   `eval`, and compiled-and-executed C agree on acceptance, shape,
   values, and traps. Runtime windows are absent; [#1298]'s separate oracle
   owns them.
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
   the exact output shape, zero elements, null-data/zero-capacity behavior at
   the runtime boundary, and Eval/C/HIP/Metal agreement. Acceptance alone is
   not sufficient.
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
7. **Axis-source mutations.** The C4 cardinality and mapping corruptions fail
   before emission with the registered typed receipt; no mutation is accepted,
   silently repaired, or allowed to reach an ICE.
8. **WireDag v7.** Exact JSON round-trip, stable bytes/hash, prove and offline
   extraction, compiler-API and binding consumption, and the capacity census
   are green. Missing size, old or future version, illegal bound tag, missing
   or out-of-range input slot, later-node reference, non-scalar source, wrong
   dtype, and incompatible axis/rank/output shape are negative controls.

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
  `lit` and `node` only for `expand`;
- interpret `WireRtDim::Node { input }` as an absolute index into the
  owning `WireDagNode.inputs`, then validate that referenced earlier
  node as rank-zero `int64`;
- validate the tensor operand, axis, input/output ranks, and exact output-axis
  mapping before encode and after exact-version decode;
- reject v6, versionless, future, string-size, `sym`, and
  `to_end` spellings before IR consumption; no legacy conversion or
  default exists;
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

### Phase 2 -- scalar edges, WireDag v7, then one resolver

**Entry requirements:** Phase 0; [#1112]'s HIP carrier/guard/widening and Metal
audit landed with their focused capacity tests and the HIP/Metal commands in
C5 green; the exact WireDag v7 contract ready to land atomically.

**Deliver in order:** C2.1-C2.4 and C6; all in-memory, target, transform, and
wire consumers; then C2.5 deletion. Close [#1266], [#569], [#597], and [#609].
Close [#578]'s resolution-mechanism half only; rank-polymorphic legality stays
with its owning docs.

**Frozen at exit:** `Expand.size: RtDim`; scalar input invariants;
WireDag v7; no provenance-rejection construct; one static folder; all-lane
guard placement.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 2`, including host, HIP, Metal,
WireDag, #569 transformation, zero, and negative rows owned by this phase.

### Phase 3 -- deferral totality and deterministic settlement

**Entry requirements:** Phase 2; the `spec/04` settlement-order
amendment; [#1341]'s ordered-store mechanism.

**Deliver:** the exhaustive `PendingExpandUse` registry and all
non-builtin rows, comparison-family constraint routing, source-order stores,
and generated action tests. Close [#1265] and [#1338]'s extent-selection half.

**Frozen at exit:** registry identities, normative action mapping, settlement
order, and K-run count.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase 3`; every action row and K fresh
[#1338] processes produce one exact verdict.

### Phase 4 -- total output-axis sources

**Deliver:** C4.1-C4.3 first as the typed interim ratchet, then C4.4 across
Eval/C/HIP/Metal. Close [#665] and [#592].

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
  operands. Its `dtype_dynamic_axis_window_oracle.py` is independent;
  no reduction-window row appears here.
- **[#1341] / PR [#1366]:** owns reusable ordered-iteration mechanics and
  linting. This plan consumes the mechanism and owns the runtime-extent
  verdict. No sibling document content is part of this PR.
- **[#731]:** owns witnessed checker errors. Remaining rejections and [#609]'s
  rank error use that channel.
- **Rank-polymorphism plans:** own whether named-axis forms are legal inside a
  `..r` body. This plan provides the resolution mechanism wherever
  the normative spec permits the form.
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
| [#578] | resolution mechanism for permitted rank-polymorphic forms | 2 |
| [#1265] | comparison consumer never selects the deferred shape | 3 |
| [#1338] | coupled defaults settle nondeterministically | 3 |
| [#665] | movement-op runtime wildcard is lost across Expand | 4 |
| [#592] | grad/vmap backward Expand reaches the same missing source | 4 |
| [#1112] | HIP metadata-carrier width; external entry prerequisite | [#729], before Phase 2 |
| [#1298] | runtime axes and windows; separate oracle | excluded |

### Not owned here

Data-dependent output ranks or shapes ([#600]), type-level dimension
arithmetic ([#526]), grad's symbolic-window gaps ([#513]), runtime axes and
windows ([#1298]), sibling symbolic-dim defects not yet parented to [#1277],
and dtype-semantics decisions.

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
