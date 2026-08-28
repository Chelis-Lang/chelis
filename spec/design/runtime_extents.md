# Runtime Extents: one resolver for non-literal tensor extents

**Status:** PROPOSED. No slice is implemented. Tracking issue: [#1277].
Code evidence in this document was rechecked on `main` at `53607a64`;
function and type names are the durable anchors and line numbers are a
convenience of that commit.
**Owning specs:** `spec/04-type-system.md` §4.7 (admissibility, identity,
zero and negative extents, the runtime extent guard placement rule, and the
§4.7.2 settlement order), `spec/05-risc-primitives.md` §2.4.1 (the `RtDim`
carrier set including `InputAxis`), §2.5.1 and [05-OP-7] (the folded direct
read), [05-MOV-1], and [05-DIM-1..3]; `spec/06-transformations.md` §3.7
(runtime extents under `vmap`) and §8.6 (`batch_varying_extent`);
`spec/10-serialization.md` for the public WireDag encoding. This plan
implements those decisions and decides no language semantics of its own.
**Class fixed:** [#1277] - a tensor extent that is not a literal (a
`shape()` read, an `int64` parameter, a record projection, a user-function
result, checked arithmetic, or an in-scope dimension binder) is admitted by
the checker according to its syntactic provenance and its equality guards
and output-axis sources are recovered by the backend by string name. The
numbered spec makes every well-typed `int64` expression admissible and
requires the same value, guard, and trap on every lane. The four mechanisms
that stand between those two states are incomplete in ways that keep
producing instances.

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
  separate work filed from this plan; neither is a slice here.

## Summary

The controlling spec is newer than the implementation. `spec/04` §4.7 admits
a function parameter, local or top-level binding, record projection,
user-function result, cast, or checked arithmetic expression anywhere an
`int64` extent is expected, requires Eval, C, HIP, and Metal to execute the
same value and guards, and states where a guard is evaluated and in what
order coupled positional-`expand` defaults settle. `spec/05` §2.4.1 names the
carrier every executable extent uses. `spec/06` §3.7 says what `vmap` does
with a runtime extent.

The implementation still has four separate recovery mechanisms, measured in
the next section. The fix is ordered around representation. Slice A gives
every `expand` size a real graph carrier (a literal, a folded tensor-axis
read, or a scalar value edge) and migrates every lane and the wire format
without changing which programs the provenance walk accepts. Slice B gives
every realized output axis one checked source, derives the equality classes
`spec/04` requires from that and the claims the checker stamps, places the
guards on every lane, and only then deletes the provenance rejections the
guards replace. Slice C
makes the one genuine deferral
(positional `expand`) total and deterministic. One runner records the
baseline and enforces the allowed progression from an ICE or lane divergence,
through a typed implementation receipt, to exact execution.

## The evidence: four mechanisms, measured

1. **The provenance walk** (`classify_expand_size`,
   `crates/chelis-types/src/infer/app_shape_helpers.rs`; `SizeClass`;
   `Env::size_provenance` in `crates/chelis-types/src/env.rs:108`). Before
   accepting a runtime `expand` size the checker walks the expression looking
   for a path back to a tensor: literals and `cast` chains, `shape(t, axis)`
   on a bare or borrowed variable, `let`-bound aliases through a lexically
   scoped provenance map, and applications of a fixed arithmetic set.
   Everything else hits a fail-closed catch-all arm. A record field access
   ([#1266]) and the canonical pipe spelling `x |> shape(0) |> cast(int64)`,
   which `chelis fmt` keeps canonical and lint's autofix produces once the
   walk no longer rejects it ([#569]), are missing arms of that match, not
   defects in the language; the
   symbolic-rank body ([#578]) is rejected by the separate rank-polymorphism
   gate in `crates/chelis-types/src/infer/common.rs:1630-1716`, whose
   legality half belongs to the rank-polymorphism plans.
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
   `add_shape_dep`, `lower.rs:9158`). Arithmetic over a read fails
   closed (`lower.rs:9109-9123`, "cannot materialize as an extent"), and a
   bare `int64` scalar with no tensor name fails after the node is built
   (`lower.rs:9169-9190`). The backend then recovers runtime extents by
   string: `symbolic_occurrences` (`dag.rs:1669`), `op_declared_output_axes`
   (`dag.rs:1901-1968`), and `shape_source_for_axis` (`dag.rs:2156-2210`)
   resolve every symbol to a `Load` or an op-declared axis and ICE on an
   undeclared occurrence. The `Expand` arm covers only the expand's own axis;
   an op-declared runtime axis arriving on the expand's input is dropped,
   which is [#665]. [#592] is the same ICE reached through a grad-backward
   `Expand` whose `DimExpr::Sym` size cannot be traced to a declaring `Load`;
   its size carrier closes in Slice A and any kept-axis residue in Slice B.
4. **The lanes disagree.** The C lane emits its equality guards in
   `emit_input_shape_preamble` (`crates/chelis-backend-c/src/emit.rs:1230-
   1274`) before any allocation, in symbol-name order, and inline at
   op-declared sites; the HIP lane compiles a Load-declared symbolic
   `expand` because `emit_expand` (`crates/chelis-backend-hip/src/emit.rs:
   3107-3140`) never reads the size and takes the output shape from the
   node's type metadata by name (`emit_alias_view`, `1996-2001`). The
   Metal lane executes no `expand` row at all today: its emitter has no
   standalone `Expand` arm (`crates/chelis-backend-metal/src/emit.rs:
   641-646`, which defers broadcasts to the Metal backend plan) and
   `require_static_shape` (`701-727`) rejects every symbolic-dim and every
   rank-0 `Load`, so each such program builds to the M1 fallback stub
   (`174-188`), which aborts at runtime with no typed receipt while
   `chelis build --target metal` reports success. Both GPU gates run only
   on the device-DAG path (`reject_unsupported_hip_ops`,
   `crates/chelis-compiler-api/src/compiler.rs:4495`, called at `2079`;
   `reject_unsupported_metal_ops`, `4204`, called only from `chelis-cli`):
   there HIP rejects every `RtDim::Node` bound and the `Shape` read itself
   (`4628-4694`) and Metal rejects node-valued bounds (`4238-4259`) with a
   `deliberate [05-MOV-1]` receipt whose hint ("defined on eval and C; use
   `--target c`") asserts the language restriction the atom forbids. The
   host path executes `Node` bounds and `Shape` reads today through the C
   emitter (`codegen_host_program`), but the two targets route to it
   differently: `chelis build --target hip` takes it when any root
   manifests to the host lane or the DAG has no roots and no
   tensor-signature entry (`crates/chelis-cli/src/main.rs:3285-3295`),
   while `--target metal` takes it only in the second case
   (`main.rs:3359-3362`), so a host-rooted program with a tensor-signature
   `def` executes on HIP and is gate-rejected on Metal; the compiler-api
   path at `compiler.rs:2049-2066` is HIP-only and lacks the host-roots
   test. `vmap`
   (`crates/chelis-ir/src/vmap.rs:12`) prepends the batch axis
   to every node including rank-0 scalars and has no `Shape` arm;
   `chelis_ir::verify::verify`, which would reject the rank-1 bound source
   (`verify.rs:518-522`, `verify.rs:1471-1479`), runs only in tests and at
   the end of `grad_dag` (`grad.rs:562`), so on
   the build and eval paths a vmapped `shape()` bound silently reads the
   batch extent and both lanes agree on the wrong result ([#1378]). The checker
   rejects a zero size (`infer/app_tensor.rs:985-996` and `1184-1193`)
   although `spec/04` §4.7.2 prohibits only a negative one.

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
  return in `check_expand_signature` is the [#609] hole (it skips
  validating a declared rank, so a wrong-rank ascription is accepted and
  eval silently returns a contradicting rank). The [#597] family is
  lowering's, not the checker's: `fallback_expand_type`
  (`lower.rs:9139-9147`, `5413-5428`) discards the checker's stamped type
  for every `Concrete` or `Sym` size and always inserts an axis, so
  positional same-rank replacement never executes on any lane today and
  both lanes silently produce rank 2 under a declared rank 1 even when the
  checker stamped rank 1.

## Part I: contracts

### C1 The controlling extent contract

The numbered spec decides every rule below; this table only names where.
Later sections cite the clause labels.

| clause | rule | where it is decided |
|---|---|---|
| C1.1 | Admissibility is typing, not provenance: any `int64` expression is an `expand` size; a `reshape` target is a `List[int64]` of static arity | `spec/04` §4.7.2, §4.7.3, §4.7.4, §4.7.6 |
| C1.2 | Identity is proof-gated and equality is guarded; an unproved claim over a runtime extent adds a guard, never a rejection; every symbolic `shrink` axis is fresh | `spec/04` §4.7, §4.7.2, §4.7.3, §4.7.6 |
| C1.3 | Guard placement is a partial order: once, after operands, before the first dependent allocation or access; interface guards at entry in signature order; local guards at the introducing operation's source position; an equality guard traps `Domain` under the introducing operation, a non-negativity guard under the owning movement operation | `spec/04` §4.7, the runtime extent guard paragraph |
| C1.4 | Coupled positional defaults settle in the order their results are introduced into the checked program | `spec/04` §4.7.2 |
| C1.5 | Zero is legal; a static negative is a type error; a runtime negative traps `Domain` | `spec/04` §4.7.2 |
| C1.6 | Extents are `int64`, axes are `int32` | [05-DIM-1..3] |
| C1.7 | One carrier per owner: `expand` admits `Lit`, `Node`, `InputAxis`; `reshape` additionally `Sym` (a bystander named dimension); `pad`/`shrink`/`stride` admit `Lit`, `Node`, and `ToEnd` for a `shrink` end; a direct `shape()` extent argument to `expand`/`reshape` is the folded `InputAxis`; an in-scope binder instantiated by a tensor axis is `InputAxis` in an `expand` size and `Sym` in a `reshape` target; the same read bound elsewhere is a rank-0 `Node` | `spec/05` §2.4.1, §2.5.1 |
| C1.8 | `vmap` shares a rank-0 extent, evaluates it once, shifts a folded or materialized read's axis, and rejects an element-derived extent as `batch_varying_extent` | `spec/06` §3.7, §8.6 |

### C2 Representation first, provenance deletion last

The target `expand` node and the equality-class carrier are:

```rust
RiscOp::Expand {
    axis: usize,
    size: RtDim,
}

enum RtDim {
    Lit(usize),         // exact width is #1373's work, not this plan's
    ToEnd,              // Shrink end only
    Node(usize),        // absolute input slot of a rank-0 int64 node
    Sym(String),        // reshape targets only (spec/05 §2.4.1); unchanged here
    InputAxis {         // folded tensor-axis read, expand/reshape only
        tensor: usize,  // absolute input slot of an earlier tensor node
        axis: RtAxis,
    },
}

enum RtAxis {
    Lit(i32),           // normalized axis-domain literal, [05-DIM-1]
    Node(usize),        // absolute input slot of a rank-0 int32 node (#1298)
}

struct RuntimeDimClass {
    claim: Dim,                       // the stamped binder name or literal
    members: Vec<(NodeId, usize)>,    // output axes carrying the claim; first canonical
}

// Derived, never stored: computed with `output_axis_sources` from the DAG a
// lane consumes, after the last rewrite.
fn derive_runtime_dim_classes(dag: &Dag) -> Vec<RuntimeDimClass>;
```

- **C2.1 Exact representation invariant.** `inputs[0]` is the tensor operand.
  `RtDim::Node(i)` is an absolute slot in the same node's `inputs` with
  `1 <= i < inputs.len()`; the referenced node is earlier in topological
  order, rank zero, and exactly `int64`. `RtDim::InputAxis { tensor, axis }`
  names an earlier tensor node that is also an input slot of the owning node:
  the read tensor becomes a shape-only input, so the owning node's data
  dependencies and shape dependencies are the same edge kind and DCE cannot
  lose one without the other. The `shape_deps` side vector survives only for
  uses that no `InputAxis` slot covers, and is deleted once nothing reads it
  (Slice B). A literal axis is an exact `int32` already normalized into
  `0..rank(t)` (`spec/04` §4.7.1 normalizes a negative literal statically);
  a node-valued axis is an earlier rank-0 `int32` node and is admitted only
  after [#1298] lands the runtime `Shape.axis` operand. A `let` alias or a
  same-dtype `cast` of a direct `shape(x, axis)` read resolves as the direct
  read, as `shape_app_operand_axis_resolved` (`lower.rs:11414`) already
  does, so it carries the read tensor's identity; other arithmetic, a record
  projection of a scalar, a parameter, or a user-function result is ordinary
  scalar dataflow and reaches every movement node as `RtDim::Node`;
  `pad`, `shrink`, and `stride` materialize a direct `shape()` read exactly
  once as a rank-0 `RiscOp::Shape` and bind it through `Node`, as `spec/05`
  §2.4.1 states. A bare in-scope dimension binder such as `a` in `c:
  tensor[a, f32]` uses the same owner-specific carrier: a tensor witness
  becomes `InputAxis` in an `expand` size, stays `Sym` in a `reshape`
  target, and is a materialized `Node` for the other three; a literal
  instantiation becomes `Lit`. The typed
  environment maps binder identity, never
  spelling, to its witnesses. The reshape-only `Sym` target is the carrier
  `spec/05` §2.4.1 defines and is not changed by this plan; its runtime
  binding (`bind_symbolic_dims`, the C prologue variable) stays as it is.

  The in-memory owner matrix is exact and matches the wire matrix:

  | `RiscOp` field | legal `RtDim` |
  |---|---|
  | `Expand.size` | `Lit`, `InputAxis`, `Node` |
  | `Reshape.new_shape[*]` | `Lit`, `InputAxis`, `Node`, `Sym` |
  | `Pad.padding[*].before/after` | `Lit`, `Node` |
  | `Shrink.bounds[*].start` | `Lit`, `Node` |
  | `Shrink.bounds[*].end` | `Lit`, `Node`, `ToEnd` |
  | `Stride.strides[*]` | `Lit`, `Node` |

- **C2.2 Static values are an optimization.** A statically proved
  non-negative value, including zero, may use `RtDim::Lit`. One checked
  static folder is shared between the checker and lowering. Failure to
  fold produces the exact `InputAxis` or
  `Node` carrier dictated by C2.1; it never rejects the expression or guesses
  a value. The `size > 0` checks at `app_tensor.rs:985-996` and `1184-1193`,
  the verifier's `size must be > 0`, and the evaluator's `expand requires
  positive count` (`runtime/eval.rs:2236`) become negative-only rejections;
  a runtime negative value traps `Domain` before allocation.
- **C2.3 Every result is constructed.** The checker always constructs an
  `expand` result tensor whose rank is the operand rank or the operand rank
  plus one, stamps complete type metadata, and validates a declared or
  ascribed rank, and a literal claim against a literal size, against it. The
  early exit that causes [#609] is deleted.
- **C2.4 Equality classes are derived, not stored.** Every stamped `Dim`
  claim, a binder name or a literal, that the checker attached to more than
  one witness is one `RuntimeDimClass`, computed by
  `derive_runtime_dim_classes` from the DAG a lane consumes, after the last
  rewrite, at the same point as `output_axis_sources` (C4.5). Grouping is by
  the stamped claim, which is the output of the typed identity proof (C1.2)
  and is what `symbolic_bindings` (`dag.rs:2283`) groups by today for names;
  a member is an output axis `(node, axis)` carrying the claim, and what
  supplies its value is read from that axis's `output_axis_sources` entry
  (`ExternalAxis` for a `Load` axis, whose declared signature index is read
  off the `Load`; `OpComputed`; `InputAxis`; `ScalarInput`), never from a
  string search. Only an output axis that C4.2 maps to an unchanged input
  axis is pass-through and not a member; the axis an operation sets or
  inserts is a member whatever slot its `InputAxis` names, so a same-tensor
  read under a foreign claim ([#1376]) is guarded, while a same-tensor read
  under a proved identity costs no guard because C1.2's static proof leaves
  no claim. A literal claim is the class's canonical value itself. A literal
  claim on an anonymous runtime extent (`-> tensor[8, f32]` over `expand(b,
  0, mul(shape(x, 0), 2i64))`) is therefore a class whose canonical value is
  the literal and whose one member is the scalar-sourced set axis, and a
  cross-tensor folded read under a name is a class with the declaring `Load`
  axis and the `InputAxis`-sourced set axis, exactly the two guards
  `spec/04` §4.7.2 and §4.7.3 require. The first member is canonical; every
  other member is exactly one equality guard against it, placed by C1.3.
  Four rules, each from a `spec/04` §4.7 sentence:
  - the canonical member is the signature-first witness, so interface
    members order by declared signature index (node position is name order
    on the subexpression-program path, `lower.rs:1341`, and declared order on
    the `lower_fn` path, `lower.rs:12100-12108`, so it is not a reliable
    key); local members follow node position; nothing orders by hash
    iteration or display name;
  - no guard is ever discharged: an all-interface class runs its guard at
    entry regardless of data use, and a local member's producer is an
    observable root under `spec/06` §5.2 because the guard can trap, so DCE
    keeps every member's operands live;
  - two classes may share a node, each with its own guard, and derivation
    yields one member per `(node, axis)`, so `splice_dag` mapping both
    parameters of `f(n, n)` to one `NodeId` (`lower.rs:7830-7838`) produces
    one member;
  - a node an `RtDim::Node` slot or a member references must survive as a
    node, which C2.1's slot validation already enforces; a fused producer has
    no observable value, so fusion never absorbs one.
  Nothing is stored in the DAG or on the wire for classes: a rebuild pass
  that keeps stamped names and bound slots correct (every pass remaps
  `RtDim` slots through its own map, or carries them verbatim in the 1:1
  id-preserving passes `vectorize_axis0`, `vmap.rs:3`, and
  `bind_symbolic_dims`, `dag.rs:2322`) yields the same derived classes. This
  is deliberately graph-level: Load/Load, Load/op-output, and
  op-output/op-output equalities exist even when no movement bound owns
  them, which is what the C lane's `SymbolicDimBinding.others`
  (`dag.rs:570-574`) computes today from names alone and Slice B recomputes
  from names plus sources.
- **C2.5 Every consumer lands before deletion.** Verification, Eval, C, HIP,
  Metal, specialization, fusion, AD, vmap, CSE, DCE, hashing, and the wire
  encoder and decoder read `RtDim` in every owner, and every lane derives
  the classes, before any provenance rejection is removed. Lane notes:
  - C: `emit_input_shape_preamble` already evaluates interface-valued guards
    at entry before any allocation and op-declared guards inline, which is
    the C1.3 placement, though in symbol-name order because
    `symbolic_bindings` groups by `BTreeMap`; it changes from name grouping
    to the derived class list and from name order to signature order.
  - Eval: the same placement, expressed as ordinary dataflow plus explicit
    guard steps before the first dependent allocation.
  - HIP: its gate admits `Lit` and `InputAxis`, which are metadata reads
    (the same by-name metadata read `emit_expand` performs today), so a
    Load-declared symbolic `expand` keeps compiling. On the device-DAG path
    it rejects `Node` at `typed_unsupported(#1298)`, the owner the atom's
    own parenthetical names, until the device scalar path lands there; the
    same gate edit changes the receipt from `deliberate [05-MOV-1]` (a
    language-rejected authority whose hint asserts a restriction the atom
    forbids) to `unimplemented chelis#1298`, the [05-UNS-5] kind for an
    implementation gap. That is a legal interim state in the C5 lattice.
    An extent above `INT_MAX` on HIP remains [#1112]'s defect exactly as
    it is now.
  - Metal: its gate admits the same carriers and rejects `Node` at
    `typed_unsupported(#1383)` with the same receipt-kind change, but no
    device-path `expand` row and no symbolic-dim or rank-0 `Load` row
    executes on Metal today (evidence item 4): every such row is recorded
    at `lane_divergent` (the M1 abort stub) and stays there until [#1383]
    lands `expand` emission and symbolic-dim `Load` support under the
    Metal backend plan. This plan adds no Metal emission.
  - Host path: a program the CLI routes to the host lane (evidence item 4:
    host-rooted or root-free on HIP, root-free only on Metal) already
    executes `Node` bounds through the C emitter, and those rows keep
    executing.
  - Wire: `Expand.size` changes from a display string to `WireRtDim`, which
    gains an `input_axis { tensor, axis }` variant; nothing is serialized
    for classes, since every consumer derives them from the names and
    sources it already decodes; the typed wire capacity census rows for
    every changed descriptor are regenerated and classified in the same
    change.
- **C2.6 Transforms preserve the bound slice.** `vmap` follows `spec/06`
  §3.7: a rank-0 extent scalar keeps rank zero and is shared; an `InputAxis`
  literal axis shifts by one, a node-valued axis is normalized against the
  unbatched rank and then shifted, and a materialized `shape()` read shifts
  its axis the same way, so `vmap` gains a `Shape` arm; a bound derived from
  vmapped tensor elements is rejected as `batch_varying_extent` before
  lowering. When a scalar producer has both a bound consumer and an ordinary
  batched consumer it is evaluated once; if the ordinary branch needs a
  batched value, an `Expand` of the rank-0 node over the batch axis is
  already legal (the checker admits a rank-0 operand, `builtins.rs:1824-
  1842`, and the compiler emits that shape in `zero_tensor_node`,
  `lower.rs:7047-7085`), so no new operation is needed. `grad` preserves
  every absolute input slot and its rank/dtype invariant; bound scalars
  remain the zero-cotangent boundary `spec/05` §2.4.1 defines.
  Specialization, cloning, and remapping preserve or remap every `RtDim`
  slot through the pass's own map.
- **C2.7 The deletion is atomic with the usable replacement.** Slice A
  changes no acceptance decision of the provenance walk: `SizeClass`,
  `classify_expand_size`, `classify_arith_app`,
  `sourceless_expand_size_error`, `Env::size_provenance`, and the lowerer's
  two rejection sites at `lower.rs:9109-9190` keep their acceptance
  decisions (only their wording changes under [#1367]), so
  every row keeps its `main` baseline through Slice A except the
  spec-conformance rows Slice A itself owns (zero extents, the `vmap` rule,
  constructed results). Slice B deletes all of them in the same change that
  places the guards on every lane and removes lowering's
  `fallback_expand_type` override of the stamped result type
  (`lower.rs:9139-9147`), so no intermediate commit may accept a value the
  IR cannot carry or execute a claimed extent without its guard, and a row
  `main` already executes without its guard ([#1374], [#1375], [#1376],
  [#1377]) keeps that baseline, recorded as `silent_unguarded` or
  `lane_divergent`, until that change. Slice B likewise replaces
  `symbolic_occurrences`, `op_declared_output_axes`, and
  `shape_source_for_axis` with `output_axis_sources`, and
  `symbolic_bindings` with `derive_runtime_dim_classes`; a reshape `Sym`
  target keeps binding to its class's canonical value exactly as it does
  today. Best-effort identity recognition may survive only as refinement
  whose failure result is a fresh extent plus a guard.

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
  realize by keying deferred constraints by the result's source-introduction
  position (a library result at its instantiation site) in a `Vec`, never a
  `HashMap`; `TypeVar` allocation order is dependency order under the
  callee-first SCC schedule (`crates/chelis-types/src/infer/declarations.rs`,
  line 409) and is not the key. [#1341] owns the general mechanism and this
  plan owns the extent verdict.

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

One tripwire test walks the builtin registry and asserts that every builtin
consuming a tensor either unifies its operands against the pending result or
propagates it; a builtin without a disposition fails that test. The
semantic tests execute both candidate outcomes and a contradictory shape for
every `Constrain` context, propagation followed by later selection for every
`Propagate` context, and the documented default for every `Freeze` context.

### C4 Every realized output axis has one checked source

An exhaustive match over `RiscOp` variants catches a new operation but not a
missing flow through an existing operation; [#665] is the proof, since the
operation is already `Expand` and the missing fact is that an op-declared
axis on its input must flow through a kept output axis with an index shift.
The structural interface is therefore a per-output-axis source with a
cardinality check:

```rust
enum AxisSource {
    Literal { value: i64 },
    ExternalAxis { load: NodeId, axis: usize },
    InputAxis { input: usize, axis: RtAxis },
    ScalarInput { input: usize },
    OpComputed { op: NodeId, axis: usize },
}

fn output_axis_sources(dag: &Dag, node: NodeId) -> Vec<AxisSource>;
```

- **C4.1 Cardinality and ownership.** `output_axis_sources` is one
  exhaustive match over `RiscOp`; a check on every production path (eval
  and build, not only `chelis_ir::verify::verify`, which runs in tests and
  at the end of `grad_dag`) requires
  its result to have exactly the output rank with no omitted or duplicated
  axis, so a new operation fails to compile and a missing flow through an
  existing operation fails that check. `ExternalAxis` names the exact external
  `Load` by `NodeId`; no source is located by searching for a `Load` that
  carries a string, and a stamped name only groups (C2.4); `InputAxis`
  validates the tensor slot
  and its literal or node-valued exact-`int32` axis; `ScalarInput` validates
  the rank-0 exact-`int64` contract of C2.1; `OpComputed` means the
  operation computes the extent by its own output-shape rule. Identity is
  decided separately and only by typed proof: a proved same identity keeps
  the source dimension's name as type metadata, an unproved cross-tensor read
  is a fresh extent with a class member and a guard, and equality of extent
  formulas alone never proves identity. The checker already keeps the
  declared claim (`check --show-inferred` stamps `-> tensor[d0, f32]` beside
  `y: tensor[d1, f32]` for [#1374]); it is lowering's `fallback_expand_type`
  (`lower.rs:9139-9147`) that overrides the stamped result type with the
  size's identity, and Slice B removes that override so the claim survives
  to derivation.
- **C4.2 Exact movement mappings.** Same-rank `Expand` maps every unchanged
  output axis to the same input axis and the replaced axis to its literal,
  `InputAxis`, or scalar size. Rank-increasing `Expand` maps axes before the
  insertion unchanged, the inserted axis to its size, and later output axes
  to input axis `output_axis - 1`. Each `Reshape` target maps to its literal,
  folded `InputAxis`, scalar input, or reshape-only `Sym`. Only the identity
  movement axes that `spec/04` §4.7 names, `Stride` with literal step one and
  `Pad` with literal zero padding, pass an input axis through; every symbolic
  `Shrink` output axis and every other non-identity movement axis is
  `OpComputed` with a fresh class member, including a full-axis `(0, ToEnd)`
  slice. `Load` axes are `ExternalAxis`; shape-preserving non-movement ops
  use the input-axis map; every computed-shape op is `OpComputed`.
- **C4.3 Interim failure is typed.** The check first lands as a verifier
  ratchet. A currently unsupported but well-typed mapping yields the
  registered [#730] `Unsupported` receipt; it never reaches the
  occurrence-pass ICE and never substitutes an input extent.
- **C4.4 Target state consumes the sources.** Eval and all backends consume
  `output_axis_sources` and `derive_runtime_dim_classes` directly. Runtime
  extent flows no longer depend on `shape_source_for_axis`,
  `op_declared_output_axes`, or a search for a `Load` carrying the same
  string. Stamped names group; sources locate; the reshape-only `Sym`
  target binds to its class's canonical value as it does today. This closes
  [#665] and the kept-axis residue of [#592].
- **C4.5 Sources are derived after the last rewrite.** `output_axis_sources`
  is neither stored in the DAG nor serialized nor carried across transforms;
  it is computed from the DAG that Eval or a backend actually consumes, after
  vmap, grad, specialization, fusion, cloning, and DCE, so a stale `NodeId`
  cannot outlive a mutation.

### C5 The class oracle has a reachable boundary

The authoritative named suite is `scripts/runtime_extent_oracle.py` plus its
two platform execution gates. The runner accepts `--phase a`, `--phase b`,
`--phase c`, and `--phase final`. Final automatic success exits zero with:

```text
RUNTIME EXTENT ORACLE: PASS
```

The suite records the exact commit and corpus digest so host, HIP, and Metal
evidence cannot be combined across different heads. Rows that need a
node-valued axis compose the [#1298] oracle at the same commit
(`scripts/dtype_dynamic_axis_window_oracle.py`, ending
`DTYPE DYNAMIC AXIS WINDOW ORACLE: PASS`); no other row depends on it, and
reduction windows stay wholly in that oracle. The corpus is generated from
the properties below; the generator, not this document, enumerates rows.

1. **Lane parity.** Every legal combination of an extent-producing form
   (literal and literal arithmetic, parameter, local and top-level binding,
   record projection, user-function result, cast, checked arithmetic, an
   in-scope dimension binder with one or several tensor witnesses, and a
   direct or indirect `shape()` read) with `expand`, `reshape`, `shrink`,
   `pad`, and `stride` gives the same acceptance, shape, values, and traps
   on `check`, `eval`, and compiled-and-executed C. Named-dimension
   combinations cover Load/Load, Load/op-output, and op-output/op-output
   classes, a class with no movement-bound consumer, two classes sharing one
   member node, the `f(n, n)` splice, a same-tensor read that keeps its
   proved identity beside a cross-tensor read that gets a guard, and a
   full-axis symbolic `shrink` `(0, ToEnd)` whose extent equals the input's
   but whose identity and guard are fresh; [#1374], [#1375], and [#1376] are
   the named rows that start at `silent_unguarded`, and [#1377] and [#1379]
   the ones that start at `lane_divergent`.
2. **GPU build and execution.** The same rows compile and execute in the HIP
   correctness suite, not merely through capability-gate rejection tests;
   Metal device-path rows sit at their recorded `lane_divergent` baseline
   (evidence item 4) until the Metal backend plan lands `expand` emission
   and symbolic-dim `Load` support, and then run under the second command:

   ```sh
   scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness --
   --ignored --test-threads=1
   PYO3_PYTHON="$(uv python find 3.11)" cargo test -p chelis-backend-metal
   --test gpu_correctness -- --ignored --test-threads=1
   ```

   Each command reports the runtime-extent group at the same commit and
   corpus digest as the host run. A row a lane rejects with an issue
   receipt sits at `typed_unsupported`, never at a silent pass; a row that
   builds to the Metal stub sits at `lane_divergent`, never at a receipt.
3. **Negative parity.** Every C1 rule has a failing control on every
   applicable lane with the owning diagnostic or trap: static negative
   extents, runtime negative extents, wrong dtype, out-of-range axis,
   rank-contradicting ascription, a named-binder witness mismatch, and
   checked overflow. Guard-order controls place an independent effect or
   trap on each side of a mismatch and require C1.3's order in both
   directions, and a `shrink` that forwards its input's class instead of
   minting a fresh member fails before emission.
4. **Zero.** Literal-zero and runtime-zero rows cover positional replacement,
   positional insertion, and named-axis expansion and assert the declared
   shape, logical element count zero, no element access, and lane agreement
   on logical metadata; the positional-replacement rows are Slice B's,
   because lowering inserts an axis until Slice B removes its override.
5. **Real [#569] transformation.** The runner proves a direct spelling checks,
   evaluates, and compiles; copies it to a task-owned path; runs `chelis lint
   --fix` and `chelis fmt --inplace`; proves formatting is idempotent and
   parseable; runs `chelis lint --check` and the style-gated `check`, `eval`,
   and compiled C path; and compares type, rank, shape, and value with the
   control. A negative fixture proves the typed-pipeline safety gate
   suppresses a rewrite that would not preserve the typed result.
6. **Deferral stability.** Every positional candidate row runs in K fresh
   processes and settles to the source-order verdict every time, including
   [#1338]'s spelling and its operand-swapped form
   `sub(expand(t1, 0, 3i64), reshape(expand(t0, 0, 6i64), [3i64, 2i64]))`,
   which passes only when `reshape`'s literal target list counts as
   independent evidence.
7. **Rebuild survival.** After each rebuild pass (vmap, grad, specialization,
   fusion, CSE, DCE, cloning, `splice_dag`), every bound slot and shifted
   axis is still present by `NodeId`, the stamped names survive on the
   rebuilt nodes so the derived classes are the same set in the same order,
   and shapes and values agree; traps and effects of a bound
   producer occur exactly once through vmap and fusion; the fused-chain row
   (an elementwise node carrying a runtime-dimension dependency; the
   fused-chain branch of `rebuild_with_fusion`, `fuse.rs:222-288`, carries no
   `shape_deps` today) and the negative-literal-axis row are named
   instances of this property.
8. **Axis-source cardinality.** A mutation that omits or duplicates an
   output-axis source fails the C4.1 check with the registered typed receipt
   before emission; a wrong shift or a misdirected source has the right
   cardinality and is caught by property 1's lane-parity rows.
9. **Wire.** Exact JSON round trip, stable bytes and hash, prove and offline
   extraction, compiler-API and binding consumption, and the capacity census
   are green; the decoder rejects any owner/tag pair the C2.1 matrix
   forbids, a missing or out-of-range slot, a later-node reference, a wrong
   dtype or rank, and an old or future schema version.

Positive rows use this allowed transition lattice:

```text
nonconforming_rejection | silent_unguarded | ice | lane_divergent
    -> typed_unsupported(issue)
    -> executes_exactly
```

A positive row may move only right, although it may skip the interim receipt.
`typed_unsupported` must carry the exact registered issue receipt. Negative
controls remain in the separate terminal state `rejects_exactly` with their
owning diagnostic or trap. A row at `executes_exactly` may not regress or
change shape, value, trap, or serialized meaning; `silent_unguarded` to
`typed_unsupported` is a rightward move. A slice invocation requires its owned
rows at their exit
state and rejects unexplained per-lane changes in every other row.

## Part II: boundary law

- Each slice exit freezes its corpus rows, public type shapes, and exact
  oracle command. Changing one updates this document and [#1277] together.
- Controls never move to bless an implementation. A red row becomes green
  only when the tree changes.
- A discovery mid-slice becomes its own child issue and named corpus row
  rather than silently widening the slice.
- Language behavior is derived from the controlling numbered spec. If the
  three C3 actions do not decide a context, or a guard placement question is
  not answered by `spec/04` §4.7, amend the numbered spec first.
- New or changed numeric identities follow the [05-OP-N] registration and
  rejection-registry regeneration rules. Wire changes also run the typed
  capacity census.

## Part III: slices

Each slice names one authoritative oracle. Slice C touches `chelis-types`
only and may land before Slice B.

### Slice A - the value edge

**Entry requirements:** none from [#1298] or [#1112]: the C carrier is
already `int64`, the HIP lane keeps its Load-declared symbolic `expand`
through `InputAxis`, and node-valued axes keep their `main` baseline
(`check` accepts, `chelis eval` executes, C, HIP, and Metal reject at
lowering; recorded `lane_divergent`) until [#1298] lands.

**Deliver in order:** the oracle runner with its generated corpus,
checked-in per-row baseline, allowed-transition validation, and exact-head
and corpus digest; [#1367]'s diagnostic residue (remove obsolete Form-3 text
and `cast(N, int32)` extent recommendations while preserving correct `int32`
axis guidance, without changing typing or closing [#1112]); derived test
stubs for every C2 clause and every C5 property this slice owns;
`Expand.size: RtDim` with `InputAxis { tensor, axis: RtAxis::Lit }` and the
exact owner matrix; the read tensor as a shape-only input slot; one static
folder; constructed results with rank validation (C2.3); zero-extent
acceptance (C2.2); the `spec/06` §3.7 vmap rule for rank-0 extent scalars and
axis shifting, plus the `batch_varying_extent` rejection; every lane,
transform, and verifier consumer (C2.5, C2.6); the wire change under the
next monotonic `WIRE_DAG_SCHEMA_VERSION` at landing, coordinated with [#1298]
so the two migrations use distinct successive versions and both trackers,
`spec/10`, fixtures, hashes, and rejected-version controls update together;
and the regenerated typed wire capacity census. The provenance walk and the
lowerer's rejection sites keep their acceptance decisions (C2.7). Close
[#1367], [#609], [#1378], [#1382], and [#592] if its reproducer is green
once the size carrier lands; [#597] waits for Slice B's removal of the lowering
override that inserts the extra axis. [#578] remains open; commits that
improve its mechanism use `Part of #578` until its complete
rank-polymorphic acceptance reproducer is green under the owning
rank-polymorphism work.

**Frozen at exit:** corpus row identities and status vocabulary;
`Expand.size: RtDim`; the `InputAxis` carrier with a literal axis; the owner
matrix in memory and on the wire; the single static folder; the `vmap`
bound-slice rule; the wire schema version; the provenance walk's acceptance
decisions unchanged.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase a`: lane parity for every row
whose recorded `main` baseline is `executes_exactly`, HIP build-and-execute
rows for `Lit` and `InputAxis` (with device-path `Node` rows at
`typed_unsupported(#1298)`, host-path `Node` rows at their executing
baseline, and Metal device-path rows at `typed_unsupported(#1383)` for the
gate-rejected bounds or their recorded `lane_divergent` baseline for the
stub rows), the bare-binder row ([#1382], from `ice` to `executes_exactly`
through the binder fold), the insertion and named-axis zero rows,
rebuild-survival rows, and wire rows this slice owns; every other row,
including every positional same-rank replacement row (`silent_unguarded`,
owner [#597]), stays at its recorded baseline.

### Slice B - one resolver: sources, classes, guards, and the walk's deletion

**Entry requirements:** Slice A. [#1112] for the HIP guard rows over
device-resident extents only: HIP must carry `int64` device extents to
compare them exactly, so host, C, host-path Metal, and host-path HIP rows
may exit first with the device-resident HIP rows at
`typed_unsupported(#1112)`. [#1383] for the device-path Metal guard rows:
they stay at their recorded `lane_divergent` baseline until Metal can
load a symbolic-dim tensor and emit `expand` (evidence item 4), and this
slice does not wait for them.
[#1298] for the node-valued `InputAxis` axis rows only, which sit at
`typed_unsupported(#1298)` on every lane until then.

**Deliver:** `output_axis_sources` and its production-path cardinality check
(C4.1-C4.3 as a typed ratchet first, then C4.4); `derive_runtime_dim_classes`
with the four C2.4 rules; removal of lowering's `fallback_expand_type`
override of the stamped result type so the declared claim survives to
derivation (what closes [#1374] and [#1376]); guard placement per C1.3 on
Eval, C, and HIP (and on Metal once [#1383] lands); replacement of
`symbolic_occurrences`,
`op_declared_output_axes`, `shape_source_for_axis`, and `symbolic_bindings`
by the two derivations; then, in the same change, deletion of `SizeClass`,
`classify_expand_size`, `classify_arith_app`,
`sourceless_expand_size_error`, `Env::size_provenance`, and the lowerer's
two rejection sites, and, once nothing reads it, of `shape_deps`. Close
[#1266], [#569], [#597], [#665], [#1374], [#1375], [#1376], [#1377], [#1379],
and any residue of [#592].

**Frozen at exit:** the `RuntimeDimClass` shape, canonical class and member
order, the guard placement realization per lane, the `AxisSource` variant
set, the one derivation point for both, the removal of string searches for
a declaring `Load`, and no provenance-rejection construct in `chelis-types`
or `chelis-ir`.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase b`: every parity row including the
forms the walk rejected and the positional same-rank replacement rows,
named-dimension and guard-order rows on every lane, the [#569]
transformation row, the positional-replacement zero rows, axis-source
cardinality, and rebuild-survival rows asserting the derived classes after
every pass.

### Slice C - deferral totality and deterministic settlement

**Entry requirements:** the Slice A oracle runner; the `spec/04` §4.7.2
settlement-order rule; [#1341]'s ordered-store mechanism, or, if it has not
landed, the two deferred stores keyed locally by the result's
source-introduction position with a citation to [#1341].

**Deliver:** the three-action protocol over resolved expected-shape evidence,
the comparison-family unification route, source-order settlement, the
recursive composite-carrier rows, and the builtin-consumer tripwire; where
allocation order implements introduction order, the K-run rows prove the two
coincide at every freeze point. Close [#1265] when its complete reproducer
is green. Use `Part of #1338`; close #1338 only if every remaining
acceptance row owned by that issue is green.

**Frozen at exit:** the action mapping, evidence variant set, recursive
composite dispositions, settlement order, and K-run count.

**Oracle:** `uv run --managed-python --python 3.11 --no-project python
scripts/runtime_extent_oracle.py --phase c`; every action row and K fresh
[#1338] processes produce one exact verdict. The same command with
`--phase final`, plus both GPU manual commands as platform legs, is the
class completion oracle and ends with `RUNTIME EXTENT ORACLE: PASS`.

## Part IV: bookkeeping

### Interlocks

- **[#729] / [#1112]:** owns the remaining HIP `int64` metadata carrier.
  Slice B's HIP guard rows over device-resident extents depend on it;
  nothing else here does, and this plan neither reparents nor closes it.
- **[#1298]:** owns computed runtime `shape` axes, runtime reduction
  windows, and, as [05-MOV-1]'s own parenthetical records, the device
  scalar path that device-path `Node` bound rows wait on
  (`typed_unsupported(#1298)`). This plan admits `RtAxis::Node` only after
  that runtime axis lands, and composes its oracle only for those rows.
  Whichever of [#1298]
  and Slice A lands first takes the next monotonic `WIRE_DAG_SCHEMA_VERSION`;
  the other takes the one after; both trackers update together and no
  version is reused.
- **[#1341] / [`hash_order_determinism.md`](hash_order_determinism.md):**
  owns ordered-iteration mechanics, the lint ratchet, and the K-run harness.
  This plan consumes them in C3 and C5 property 6 and owns which verdict
  determinism settles on. [#1338] keeps [#1277] as its structural parent with
  an `Also part of #1341` cross-link.
- **[#731]:** owns witnessed checker errors; remaining rejections and the
  [#609] rank error use that channel.
- **[#730]:** owns the typed `Unsupported` receipt used by C4's interim
  transition and by the GPU lanes' `Node` rows.
- **[#1383] / Metal backend plan**
  ([`chelis_metal_backend_plan.md`](chelis_metal_backend_plan.md) §4, the
  `expand` row; run evidence under [#737]): owns Metal `expand` emission,
  symbolic-dim and rank-0 `Load` support, and the device-path receipt kind.
  Every device-path Metal row here sits at its recorded baseline until that
  lands; this plan adds no Metal emission and gates its Metal guard rows on
  that owner.
- **Rank-polymorphism plans** (`rank_polymorphism.md`,
  `rank_polymorphism_tier3_followups.md`): own whether named-axis forms are
  legal inside a `..r` body; this plan provides the resolution mechanism
  wherever the numbered spec permits the form. [#578] remains open until its
  complete acceptance reproducer is green.
- **[#1372] DAG rebuild integrity:** owns the invariant that side-carried
  annotations (`shape_deps`, `merged_spans`) survive every graph rebuild.
  This plan stores no class annotation, so it relies on that invariant only
  for `shape_deps` until Slice B deletes it; the tracker owns making the
  invariant structural and the fused-chain probe named in C5 property 7.
- **[#1373] exact `i64` internal extent carriers (a [#729] child):** owns
  moving `RtDim::Lit`, `DimInfo`, `DimExpr::Concrete`, the tensor-type
  copies, and their wire forms from host-sized `usize` to exact `i64`. This
  plan neither requires nor blocks it.

### Issue map

| issue | owning clause | slice |
|---|---|---|
| [#1367] | stale `int32` extent and Form-3 guidance | A |
| [#1266] | record projection rejected by provenance walk | B |
| [#569] | real lint/fmt transformation breaks a legal extent | B |
| [#597] | positional same-rank replacement never executes: lowering always inserts | B |
| [#609] | wrong-rank ascription is accepted | A |
| [#665] | movement-op runtime wildcard is lost across Expand | B |
| [#592] | grad-backward Expand size cannot be traced to a Load | A (size carrier); B for any kept-axis residue |
| [#1374] | cross-tensor read under a named claim is silently identified, no guard | B |
| [#1375] | node-valued reshape target under a named claim executes unguarded | B |
| [#1376] | same-tensor read on the set axis under a foreign claim, no guard | B |
| [#1377] | eval executes a literal claim over a cross-tensor read that C guards | B |
| [#1378] | vmap batches a `shape()` bound so it reads the batch extent | A |
| [#1379] | arithmetic size under a named claim: eval unguarded, compiled lanes reject | B |
| [#1382] | bare binder as an `expand` size: no witness in eval, compiled lanes ICE | A |
| [#1265] | comparison consumer never selects the deferred shape | C |
| [#1338] | coupled defaults settle nondeterministically | C / [#1341] mechanism |
| [#578] | mechanism evidence only; full rank-polymorphic repro stays open | external rank-polymorphism work |
| [#1112] | HIP metadata-carrier width; Slice B HIP guard rows | [#729] |
| [#1298] | runtime axes and windows; `RtAxis::Node` rows and wire ordering | [#729] |
| [#1383] | Metal device-path runtime extents: emission, `Load` support, receipt kind | Metal backend plan |

### Not owned here

Data-dependent output ranks or shapes ([#600]), type-level dimension
arithmetic ([#526]), grad's symbolic-window gaps ([#513]), runtime axes and
windows ([#1298]), the rank-polymorphic legality half of [#578], the DAG
rebuild integrity class ([#1372]), exact `i64` internal carriers ([#1373]),
Metal `expand` emission and symbolic-dim `Load` support ([#1383]), sibling
symbolic-dim defects not yet parented to [#1277], and
dtype-semantics decisions.

## Considered and rejected

### Removed after the trimmed plan's own review

The trimmed plan went through three fresh-context rounds on PR [#1343]
(heads `3de7b370`, `4cec575c`, `112c3b6f`). Those rounds corrected false
code claims and spec contradictions and added no mechanism, but a cold
reread with the stricter question "what is the smallest thing that
satisfies the citation" removed four items that had survived from the
earlier draft or entered as repairs:

- **The C3 privacy lock** (an opaque `#[must_use]` inference carrier, a
  `consume_pending` choke point, a site enum generated from the builtin and
  Deep expression-form registries, compile-fail bypass tests). It entered as
  a repair to a hypothetical bypass of the design's own dispatch, not to an
  instance; [#1265] closes by routing the comparison family through
  unification. Replaced by the builtin-consumer tripwire test in C3.
- **The generated `OpExtentRule` registry bijective with `RiscOp`**, with
  build failure on a missing row and every formula citing an atom. It grew
  from a review remark that `OpComputed` "could not cover every output axis";
  the cardinality check on one exhaustive match is what catches a missing
  flow. Replaced by `output_axis_sources` and the verifier check in C4.1.
- **The reshape `Sym` migration and its transitional matrix row.** A review
  found the trimmed plan contradicting `spec/05` §2.4.1, which makes `Sym`
  legal as a reshape target; the plan was aligned to the spec instead of the
  spec to the plan's "no name recovery" principle. No instance involves a
  reshape bystander target, so the carrier and its binding are unchanged.
- **Enumerated mutation and wire-negative lists in C5.** Replaced by the
  properties the corpus generator derives rows from.
- **A lowering-time interim receipt through Slice A** (rounds five to eight
  of the strict review). Slice A originally deleted the provenance walk and
  kept a narrowing receipt for claimed sizes; four consecutive rounds found
  a seam in that clause (Node-only, then reshape, then `InputAxis` reads the
  walk rejects, then `let`/`cast` reads `main` executes). Replaced by
  sequencing: Slice A changes no acceptance decision of the walk, and Slice
  B deletes the walk in the same change that places the guards, so no row
  moves anywhere before its guard exists.
- **A stored `Dag.runtime_dim_classes` field with per-pass remaps and wire
  encoding** (removed after the fourth round). The four `spec/04` §4.7
  sentences it cited fix placement and order, not storage; deriving the
  classes at the consumption point from the stamped names and
  `output_axis_sources`, as C4.5 already does for sources and as
  `symbolic_bindings` does today from names alone, satisfies them without a
  DAG field, a remap obligation in every pass, a wire encoding, or a
  dependence on [#1372].
- **A discharge clause for local class members** (removed after the fourth
  round). It contradicted `spec/04` §4.7's "evaluated exactly once" and
  `spec/06` §5.2's rule that a potentially trapping node is an observable
  root.
- **`InputAxis` for a binder-instantiated `reshape` target** (removed after
  the fourth round). With `Sym` kept as the reshape-only carrier, the binder
  fold applies to an `expand` size only; two conforming implementations
  would otherwise have produced different bytes for one program.

### Removed from the earlier draft

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
  classes are derived at the consumption point, so there is nothing to
  renumber; canonical order is computed where they are derived.
- **Transform namespace with a `u64` cursor, exhaustion, and history-sensitive
  hashing** (reviews of `3496f159`, `37914d77`, `4215f760`; repairs
  `37914d77`, `4215f760`, `3d6d4746`). Existed to keep synthesized origins
  collision-free under a self-imposed no-ID-reuse rule, and made the artifact
  hash depend on which passes had run. Replaced by: no transform IDs; the
  hash covers the graph, and classes are derived rather than hashed.
- **Workspace-wide generated transaction registry and module-private
  `RawDag`** (reviews of `4215f760`, `3d6d4746`; repairs `3d6d4746`,
  `60d595b2`). Sealed every construction and mutation route across 1,424
  `add_node` call sites and every public `Dag` signature. The defect class
  it targets is real but is a class of its own; it is filed as [#1372],
  whose proportionate first step is a shared rebuild helper that carries
  every side vector so forgetting one is a type error. `spec/10` requires no
  hash stability that a sealed graph would buy.
- **`usize -> i64` semantic-extent transit census** (reviews of `60d595b2`,
  `d5c6fe0b`; repairs `d5c6fe0b`, `a0e74485`). Closes no instance in the
  issue map and pays off only on 32-bit hosts; it is filed as [#1373], a
  [#729] child, without the generated census, which an actual defect would
  have to motivate.
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
  sentences: value source from the op, identity only from typed proof.
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
[#737]: https://github.com/Chelis-Lang/chelis/issues/737
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
[#1374]: https://github.com/Chelis-Lang/chelis/issues/1374
[#1375]: https://github.com/Chelis-Lang/chelis/issues/1375
[#1376]: https://github.com/Chelis-Lang/chelis/issues/1376
[#1377]: https://github.com/Chelis-Lang/chelis/issues/1377
[#1378]: https://github.com/Chelis-Lang/chelis/issues/1378
[#1379]: https://github.com/Chelis-Lang/chelis/issues/1379
[#1382]: https://github.com/Chelis-Lang/chelis/issues/1382
[#1383]: https://github.com/Chelis-Lang/chelis/issues/1383
