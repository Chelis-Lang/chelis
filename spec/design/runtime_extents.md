# Runtime Extents: values, claims, and their witnesses

**Status:** IN PROGRESS. Slice A and the first Slice B mechanisms have
landed. The remaining Slice B work is specified below. Slice C is withdrawn:
`expand` broadcasts a singleton axis and `insert` adds an axis, so neither
operation produces a deferred shape. Tracking parent: [#1277].

**Authority:** `spec/04-type-system.md` §4.7 decides admissibility, identity,
claims, and guard placement; `spec/05-risc-primitives.md` §2.4.1,
[05-MOV-1], and [05-DIM-1..3] decide carriers and movement execution;
`spec/06-transformations.md` §3.7 and §5.2–5.4 decide batching and rewrite
legality; `spec/10-serialization.md` controls the public wire contract.
This document selects an implementation of those rules. It changes no
language rule and does not relax the numbered specs to match a baseline.

## Current state and remaining work

The reference implementation state for the merged inventory is `4f3812e2f`.
The following are merged mechanisms, not a claim of complete class coverage.

| delivery | merged PR | established behavior |
|---|---|---|
| Slice A | #1467 (`627313120`) | `Expand.size: RtDim`, folded axis and scalar edges, static folder, zero/rank checks, typed vmap bound handling, WireDag v7 |
| B1 | #1510 (`801f92c02`) | `output_axis_sources` and a production-path cardinality check; #1480 closed |
| B2a | #1536 (`55ec86524`) | derived classes/witnesses and entry guards; signature scope is approximated by root reachability |
| B2h | #1531 (`1253d7653`) | host def application on eval uses the kernel decision C uses; entry checks reach this route |
| B2r | #1597 (`12c04c66a`) | reshape leaves the kernel keep-list; local carrier guards have C and eval consumers; inlined #1375 residue belongs to #1686 |
| expand/insert decision and implementation | #1532, #1547, #1590 (`c8a5f1a75`) | one meaning per primitive; same-rank unit-extent guards; the lowering override is removed |
| S2c | #1605 (`42ce46cdc`) | deferral recorder, stores, executor, settlement registry, and source-ordinal index removed |
| B2b-0 | #1616 (`f6cfd2d72`) | seven existing phase-B rows receive passing receipts; no guard mechanism changes |
| B2b-0b broadcast preparation | #1658 (`3fbc1df49`) | anonymous broadcast axes retain their own sources; the 11-case broadcast attribution contract passes; inlined unit-check residue belongs to #1687 |
| B2b-0b numeric local guards | #1662 (`5dde8373c`) | literal/resolved claims compare independent runtime carriers; the 32-lane local matrix passes |
| B2b-0b local guard order | #1666 (`e0fa5ccc0`) | multi-axis local reshape guards use declaration order on Eval and C |
| B2b-1 literal claim transport | #1668 (`84ae9bd7f`) | explicit caller-axis witnesses and invocation dependencies retain literal claims through direct, nested and discarded calls |
| exact wire migration | #1664 (`ad9b6c248`) | WireDag v9 uses exact numeric codecs and validated fixed-width references; stdlib/library/context caches are 16/12/18 |
| helper guard order | #1688 (`4f3812e2f`) | declared helper input order and one shared IR comparison schedule; the 50-case exported/binding/main oracle passes |

The extent carrier is no longer a display name, and sources/classes already
exist. What remains is preservation of a claim and its caller witnesses,
complete consumption of the derived sources, and removal of the provenance
walk only after the guards can protect the newly admitted forms.

The shipped `DimInfo::Named(String, Option<usize>)` conflates a name with its
optional resolved size. `derive_dim_witnesses` serves the C/HIP prologues;
`derive_runtime_dim_classes` serves the evaluator and local sites. They share
`output_axis_sources` and `split_by_scope`, but are distinct groupings. The
old `symbolic_bindings` path still supplies evaluator bindings and declaration
consumers. Passing a test of one grouping does not test the other.

The checked transport in C2.4 implements the restored #1686/#1687 host
obligations: computed reshape claims and broadcast unit preconditions survive
inlining, graph rewrites and cache transport. Their bounded oracle retains
independent declaration, value and failure assertions. Remaining failure boundaries:

- #1374/#1376: lowering may drop the argument whose axis witnesses the result
  claim. Root reachability cannot recover a signature that is no longer
  represented, and an unread argument still owes its signature check.
- #1397: checked function stamps now retain the declared result; movement
  execution still owes its guard and general wildcard-returning roots can
  disappear from eval and entry emission. #1378's public vmap witness remains
  masked by that root failure.
- #1266/#569: the provenance walk still rejects equivalent field/pipe forms.
- #1379: local op-computed extents need guard sites beyond the folded-axis
  and scalar-target forms B2r serves.
- #665/#1556: declaration consumers still use legacy name recovery. #1482
  additionally needs an actual shape source for a synthesized constant.
- #1566: evaluator binding inference can identify separate signatures by
  binder spelling.
- #1512: the old deferred-expand witnesses are unreachable, but unresolved
  variables have other origins. The remaining early-return validation audit
  is open; neither the old census nor four `sum` probes closes it.

B2b-0b's merged broadcast preparation repair addresses #1619: for `Expand` outputs
without explicit named dimensions, C's anonymous-axis rewrite uses
`output_axis_sources`, filling anonymous axes from their own size/kept-axis
sources. Literal result claims survive, and an anonymous resolved number
remains a literal obligation. The previous
same-rank shortcut copied the operand's unit extent onto the output, creating
a false result claim against the size-source tensor. The unit-precondition
derivation already read the correct operand and is unchanged. Outputs with
explicit named dimensions retain their existing preparation path: preserving
the name without its unread declaring witness can newly execute a wrong
shape. B2b-1 owns preservation and enforcement of those scoped claims.

The bounded acceptance command is `singleton_broadcast_contract` in C5.
Literal call/inlining obligations, op-computed local guards, and scoped claim transport
have separate receipts below. C2.4's checked transport closes the inlined
non-unit host obligation (#1687); #1658's anonymous-output rewrite alone did
not establish that claim preservation through calls.

## Part I: implementation contracts

### C1 Controlling rules

| clause | implementation obligation | authority |
|---|---|---|
| C1.1 | Admit every correctly typed int64 extent expression, including record projection, calls, casts and arithmetic | spec/04 §4.7.2–4.7.6 |
| C1.2 | Preserve an independently stated literal or named claim; require proof of equality or its execution-time guard | spec/04 §4.7 |
| C1.3 | Evaluate guards once, after operands and before dependent allocation/access; interface guards at entry in signature order, local guards at the introducing operation's source position | spec/04 §4.7 |
| C1.4 | Use the exact Domain trap line and accompanying source/axis/value context; preserve trap occurrence and attribution through rewrites | [04-NUM-9..12], spec/06 §5.2–5.4 |
| C1.5 | Zero is legal; negative static sizes reject and runtime negatives trap | spec/04 §4.7.2 |
| C1.6 | Extents are int64 and axes int32; retain the exact owner-specific carrier | [05-DIM-1..3], spec/05 §2.4.1 |
| C1.7 | Share rank-0 extent producers under vmap; shift shape reads past the batch axis; reject element-derived extents | spec/06 §3.7, §8.6 |
| C1.8 | `expand` is same-rank and requires a unit operand axis; `insert` raises rank | spec/04 §4.7.2, spec/05 §2.4.1 |

### C2 Values and claims are separate

#### C2.1 Existing value carriers

`RtDim::Node(i)` names an absolute input slot containing an earlier rank-0
int64 node. `InputAxis { tensor, axis }` names the exact tensor input slot and
its normalized int32 axis. These are real dependencies; a tensor used only
for its shape is still an input. `Lit` is an independently folded value,
not a copy of the result declaration. The carrier matrix is unchanged:

| IR field | legal carriers |
|---|---|
| `Expand.size` (both surface operations) | `Lit`, `InputAxis`, `Node` |
| `Reshape.new_shape[*]` | `Lit`, `InputAxis`, `Node`, `Sym` |
| `Pad.padding[*].before/after` | `Lit`, `Node` |
| `Shrink.bounds[*].start` | `Lit`, `Node` |
| `Shrink.bounds[*].end` | `Lit`, `Node`, `ToEnd` |
| `Stride.strides[*]` | `Lit`, `Node` |

`ToEnd` requires a paired `Lit(0)` start. `Sym` remains the reshape-only
carrier required by spec/05; its display name must resolve through the typed
binding environment, never by searching unrelated Loads with that spelling.
A direct shape read folds to `InputAxis` only in its permitted owners;
other owners materialize `Shape` once and use `Node`. Failure of the shared
static folder retains dataflow rather than rejecting provenance.

#### C2.2 Scoped binding identity

The remaining implementation uses an opaque dimension-binding identity in
the checked type and lowered axis contract. It identifies a declaration in
one signature instantiation; the human name is diagnostic text. Lowercase
polymorphic dimension variables retain their existing instantiation semantics
and are not converted into concrete symbolic names by this work.

Allocate identities in the checked signature's declaration order, and retain
an explicit association with its ordered parameter-axis witnesses. At a call,
instantiate the callee's bindings using a fresh per-call substitution map;
map each formal witness to the actual argument axis. Two calls do not share
callee-private identities, and separate signatures spelling `seq` never
become one claim merely because their text matches. Repeated uses of one
binding within the same instantiation share its identity. A loop or repeated
runtime invocation executes that call's obligations anew; the compile-time
identity is not an execution counter.

Call substitution may prove a formal/actual equality, but it may not replace
a result claim with the actual result's inferred extent. In particular,
`f(x: tensor[rows], y: tensor[cols]) -> tensor[rows]` keeps the relationship
to the caller's `x` when its body derives a size from `y`. Binding both
parameters to one actual (`f(n, n)`) remaps both witnesses to that actual
axis; deduplicate that same source within an obligation without manufacturing
a new alias identity. Equal numerical extents alone do not merge binders.

These are typed artifact-local references, not strings, source-span guesses,
process-global counters, or a new durable-origin algebra. In-memory graph
renumbering leaves binding identity unchanged. Combining independently checked
artifacts explicitly renames their reference domains through one import map,
as ordinary graph import already remaps node references. A fresh call
instantiation is distinct from copying a graph for an optimization: the latter
preserves the same logical bindings and remaps its node references only.

#### C2.3 Axis contract and independent evidence

Each lowered axis retains BOTH its computed extent source and all outstanding
claims attached by declarations/ascriptions or calls. The implementation must
replace the single-name-as-both-facts representation at this boundary; an
axis may carry more than one obligation. A claim refers to a scoped binding
or a literal required value, its introducing operation/source position, and
the signature witness/order information needed for C1.3. Derived equality
classes are not stored or serialized.

The independent evidence comes from the operation and its inputs:
`output_axis_sources`, a validated literal carrier, or checked arithmetic
folding. A `DimInfo::Named(_, Some(k))` or `Lit(k)` copied from a declared
result is a CLAIM, not proof about a caller's tensor or the produced extent.
Likewise, `bind_symbolic_dims` resolving a name does not discharge its guard.
Record whether an extent fact was obtained independently; never infer proof
from the presence of a number in result metadata or from interface/local
placement alone.

For each obligation, compare the independent fact with the required value:

- Proven equal: no dynamic equality check is necessary, but evaluating any
  potentially trapping/effectful bound expression still obeys spec/06.
- Unknown: retain the runtime equality guard and every value it compares.
- Contradictory literals in source typing: report the owning type error.
- A modular runtime claim whose disagreement becomes known only after call
  inlining or optimization: preserve the already-required Domain failure and
  its source order/attribution. It may become an unconditional trap at the
  same observable point; it must not become successful execution, disappear,
  or be reclassified as an implementation-unsupported construct.

The last row decides #1377: a modular `-> tensor[4]` over a runtime read
must still fail when the inlined actual supplies 5. It does not authorize a
new checker phase to reject a previously runtime-dependent program. The
required failure follows spec/04 §4.7 and spec/06's observation-preserving
rewrite rule; no normative amendment is needed for that implementation.

#### C2.4 Witness retention and rebuilds

Before body-only parameter pruning, lowering materializes every signature
axis needed by a claim as an interface witness, with its scoped identity and
original signature position. A published wrapper must pass those witnesses
to its kernel even when the body reads none of their elements. They are shape
inputs, not copied payloads. Inlining replaces the formal Load by the actual
axis dependency and retains the obligation and its introducing call position.
No guard attempts to reconstruct a dropped argument from a similarly named
survivor. #1374/#1376's caller preservation is explicitly owned by B2b-1.

The same claim/source derivation is available before any deleting rewrite
for its liveness decision and after the last rewrite for emission. An
unproved extent guard is potentially trapping and therefore observable under
spec/06 §5.2. DCE retains its witness and bound-producer dependencies even if
its tensor result is unused. A genuinely total, proven-satisfied claim adds
no trap root. A rewrite that replaces a computation transfers the obligation
to the corresponding replacement axis or preserves the necessary guard
computation; it does not keep both the obsolete tensor computation and its
replacement merely to preserve a stale node id.

The typed rebuild interface must require remapping the axis contract together
with the node's inputs/type. No pass may copy just a printable dimension and
silently default the claims to empty. This change is bounded to extent
contracts; #1372 continues to own the general side-annotation rebuild class.

| boundary | preservation obligation and negative control |
|---|---|
| checked calls and `splice_dag` | instantiate scoped bindings once per call; remap formal witnesses to actual axes; independent same-name calls stay independent; `f(n,n)` keeps its obligation |
| cloning/import | preserve identities on ordinary clone; rename independent artifact domains on import; do not collide same-spelled names |
| DCE | unread signature witnesses and potentially trapping extent checks survive; deleting either must make a mismatch test fail |
| CSE/fusion | preserve each observable guard occurrence and bound evaluation; never fuse away a scalar whose value a bound needs; spec/06 §5.3 still forbids merging potentially trapping nodes |
| specialization/constant folding | carry claims onto replacement axes; keep independent extent facts separate; a known mismatch still fails at the required point |
| grad | remap all bound slots and preserve primal obligations; bounds keep spec/05's zero-cotangent boundary |
| vmap | preserve the binding/claim relationship with shifted axes, share the rank-0 bound, and execute it once as spec/06 §3.7 requires |
| wire/cache | preserve claim references, ordered witnesses and independent facts or reject the artifact; no missing-field empty default |

This table is an implementation acceptance obligation. The preparation
fixtures below cover public call/root witnesses; they do not already prove
all rebuild/import rows. The existing B2b-0 receipt proves seven named IR
passes on transforming fixtures, not lowering-side `splice_dag`, imports or
the new contract carrier. B2b-1 extends those tests before changing the carrier.

#### C2.5 Guards consume the observed quantity

Derive equality classes by scoped claim identity and literal obligations,
with sources supplied by `output_axis_sources`. Signature order selects the
canonical interface witness; local-only classes use introducing source order.
A literal is the required value itself, not the first observed member.
Every consumer obtains scope, source, proof and placement from this derivation;
legacy evaluator binding inference must migrate with the prologue consumers
so #1566 does not survive behind a second name-keyed answer.

Local sites support folded `InputAxis`, scalar `Node`, and op-computed
extents. A carrier is readable before the movement executes. An extent
computed by the operation must be computed/validated before its first
shape-dependent allocation/access, not recovered from a tensor allocated
using the unvalidated claim. Merely checking after a wrong allocation is not
a conforming implementation of C1.3. C and Eval consume the same observation
instruction and placement; the C emitter's current five movement callers are
not a proof that every derived site has a consumer.

The unit precondition on `expand` is a separate obligation from its result
extent. Its observed value is the OPERAND axis before replacement. Its result
source is `size`. #1619's symbolic-size and folded-shape-size rows must prove
that these cannot be swapped. `insert` has no unit-operand precondition.
Multiple obligations on one axis survive independently; coalesce a duplicate
representation of the same obligation, not distinct trapping operations.

#### C2.6 Atomic integration and wire ordering

B2b-1 first implements literal call claims through an explicit IR witness.
This bounded change owns #1377's shape-derived `insert` call and its nested
and discarded-result controls. A literal identifies its own required value;
it does not need to identify a named binder by spelling. The remaining named
claim migration still owns scoped binding identities, unread named witnesses,
and #1374/#1376/#1566. Both changes retain the C2.3 distinction between a
requirement and an independently observed extent.

The literal transport uses `RiscOp::ExtentWitness { site, parameter, axis,
requirements }`. Its one tensor input is the actual argument; its result is
the observed axis extent as a rank-zero `int64`. `parameter` is diagnostic
text, `axis: RtAxis` selects the observed axis, and
`requirements: Vec<ScalarValue>` retains ordered, tagged `int64` literal
claims. Requirements are explicit fields, with no missing-field default.
The operation reads shape metadata without copying the argument's elements.
Its existing `span_id` records the introducing call.

Lowering creates these witnesses in parameter/axis order before lowering the
callee body. A parameter shape read uses its witness through an ordinary
`RtDim::Node` input slot. The checked declared result supplies the literal
obligation, while the witness input supplies the observed value. A fresh call
creates fresh witness nodes; copying or importing a graph remaps their ordinary
inputs and retains their claims. No name-keyed extent grouping is involved.
Within each lexical environment, a value binding carries its original tensor
node and witnesses together. Binding an alias forwards that metadata; rebinding
replaces it and leaving the scope restores it. Borrowing an alias for a shape
read consumes the same binding metadata. Thus alias
chains retain the declaring witness without conflating different parameters
that happen to receive the same actual tensor. Shape reads consume this binding
metadata rather than requiring the parameter's original spelling.
Once a guard establishes equality, a consumer may use that checked literal;
the result annotation alone never licenses the substitution.

The enclosing invocation result retains its call witnesses through explicit
`shape_deps`, including witnesses of nested calls whose tensor results are
discarded. A fresh `Copy` return carrier follows the result value and every
required witness; an existing returned value is never mutated to depend on
a later call. These dependencies participate in root-scoped evaluation and DCE:
an unrelated export does not activate another invocation's checks. A witness
without a requirement adds no trap dependency. CSE, specialization and folding preserve the check or discharge
it from independent evidence; they cannot infer success from its requirement.
Grad retains primal checks and assigns zero cotangent to shape values. Vmap
keeps the result scalar and shifts its observed input axis past the batch axis.

The construction/consumer inventory for this change is:

| boundary | concrete owners |
|---|---|
| checked declaration | `infer/common.rs` declaration owner and `infer/annotate.rs` function stamp; `CheckedProgram::signature_inference` retains the declared result |
| construction and calls | `lower.rs`: `lower_fn`, `lower_plain_callable_app`, parameter shape-read lowering, block/let invocation dependencies and declared-result preservation |
| graph transport | `Dag::add_node`/`replace_node`, `lower.rs::splice_dag`, `optimize.rs` DCE/CSE/folding, `specialize.rs`, `fuse.rs`, `grad.rs`, `vmap.rs`, `tier2.rs` and contextual library import |
| validation and sources | `verify.rs`, `axis_sources.rs`, `dag.rs` operation properties and runtime-dimension consumers |
| execution and ownership | `eval.rs`, `ownership/mod.rs`, `ownership/storage.rs`, C/HIP/Metal emitters and compiler target classification |
| public transport and caches | compiler-api `schema.rs` wire op and `compiler.rs` conversions, stdlib/library cache versions and build identity; capacity/rejection registries and structural inventory |

The existing six-case `literal_result_claim_contract` is extended by nested
calls and discarded results in
`literal_claim_transport_survives_nested_and_unused_calls`. These public
fixtures preceded implementation and both runners now pass.
`runtime_extent_literal_transport` checks independent requirements, invalid
carriers, root liveness, CSE/folding, grad and vmap. `wire_extent_witness`
checks exact claims, invocation edges and source provenance on roundtrip,
plus rejection of missing fields and malformed claims/edges. WireDag v8
introduced these fields; #1664 moves their numeric transport to v9 and
stdlib/library/context cache versions 16/12/18.
The wider named-claim and op-computed-source exits remain separate. HIP and
Metal retain their existing runtime scalar shape-read exclusions; these
host execution receipts do not certify device execution.

The checked-extent integration owns #1686 and #1687. It adds two checked
carriers to the construction and consumer inventory above:

- `CheckedReshapeExtent { claims, axis }` consumes an ordinary scalar `int64`
  input for the independently computed target and one for each required value.
  Each requirement is a tagged literal constant or the declaring parameter's
  `ExtentWitness`, selected within the current signature activation before
  substitution. The nonempty `claims` list contains ordered diagnostic labels,
  not identities. Each requirement is checked in list order. The checked scalar
  becomes the reshape target before allocation. All shape-list expressions
  lower before any of its check carriers; existing `shape_deps` retain every
  target producer when only a discarded result's check remains live. Thus a
  later target's arithmetic failure precedes a reshape claim check, as §4.7.3
  requires. Existing static arithmetic folding is retained only when producing
  literals, external literal axes checked at entry, constant scalar dataflow,
  or a prior checked scalar with a literal requirement independently establish
  the source extents. Folded source dependencies and the scalar claim carrier remain;
  computed result metadata cannot supply a proof. Its computed axis has a fresh
  runtime identity; the declared requirement remains an explicit checked edge.
- `CheckedUnitAxis { axis }` consumes the original tensor and that same
  tensor-axis witness carrying requirement one. Verification requires both
  edges to agree, requires the unit obligation, and permits only the checked
  axis to refine to one. It forwards the tensor after the witness succeeds;
  `expand` then consumes that checked tensor without repeating the guard.

`ExtentWitness.site` explicitly distinguishes `Caller` from `LocalExpand`.
Caller failures retain the declaring parameter and `load` trap; a local
broadcast retains the observed input node and `expand` trap. Rebuilding and
vmap preserve the site while remapping the input and axis. Local unit
refinements are reused only for the same input node and axis within the
current activation; entering and leaving a call saves and restores that map.
An inserted axis derived from a witness also retains a fresh runtime identity,
so an independently known actual cannot make intermediate ownership validation
preempt the witness's runtime check.

Result requirements are resolved before entering the body. After lowering,
the returned tensor's per-axis source derivation attaches them to unique
computed reshape scalar carriers. This follows aliases, shape-preserving
operations and helper results without forwarding raw binders through syntax.
A scalar starts as an identity carrier and becomes checked before its consuming
reshape; inner and outer requirements remain distinct input edges. A claim
introduced after a value was already produced executes at that call boundary.
Literal-condition host functions use existing DAG branch pruning while retaining
their full declaring signature; dynamic host control flow keeps its existing route.

Each activation owns fresh witness nodes. The lowering environment maps the
signature's binders to these exact nodes and restores that map on return;
ordinary graph edges, rather than spelling or reachability, carry identity
through rebuilding and import. Every checked scalar and required witness is
retained by the invocation's fresh return carrier, even when its result is
discarded. CSE and folding preserve independent checks and call provenance.
`LoweredLibrary.program_signatures` retains authored declarations separately
from inferred function metadata. `ResolvedFunction` carries that declaration
through callable aliases and transforms; local shadowing cannot select a
same-spelled global declaration. Checker wildcard narrowing retains a named
dimension only when that declaration binds it in a parameter. A shape-only
argument remains a witness even when the body does not read its data.
Eval composes the checked library and new program before imported kernel
lookup, using the existing checked-library proof; an absent proof is an error.

Grad retains primal checks and restores a unit operand's cotangent shape using
its checked witness, so a known non-unit actual cannot make backward graph
validation preempt the primal Domain failure. Vmap shares scalar checks and shifts tensor-axis
witnesses and unit refinements together. The verifier and exact wire decoder
reject missing claims, wrong arity or scalar types, and unsupported refinements.
WireDag v10 carries both checked operations. Stdlib/library/context cache
versions 17/13/19 require the authored signature ledger and revalidate it
against fresh lowering. Missing fields and a forged ledger with a valid
checksum and unchanged proof identity reject at admission.

The completion oracle for these two host obligations is:

```sh
cargo nextest run -p chelis-cli -p chelis-ir -p chelis-compiler-api --lib \
  --test runtime_extent_claim_preparation --test runtime_extent_checked_transport \
  --test wire_extent_witness --test disk_cache \
  --test issue_513_symbolic_axis_adjoints \
  -E 'binary(runtime_extent_claim_preparation) | binary(runtime_extent_checked_transport) | binary(wire_extent_witness) | test(=cached_imports_preserve_computed_claims_and_unit_preconditions) | test(=previous_checked_extent_cache_is_rejected_before_payload_decode) | test(context_decode_rejects_missing_or_forged_authored_signatures) | binary(issue_513_symbolic_axis_adjoints) | test(static_reshape_folding_requires_independent_axis_sources)'
```

The new public matrix has 117 initial exported/binding/main fixtures, 69
result-graph fixtures, 48 complete-shape-list scheduling fixtures,
six folded-source caller-contract fixtures and 24 producing-source expression fixtures,
three executable example controls, 24 grad/vmap controls
and 12 imported-call controls. Every
case checks declarations independently of actual shape/value or required
failure. Claim mismatches require Domain/reshape/int64. Scheduling fixtures
independently require Eval's division-by-zero/floor_div/int64 diagnostic and
the C integer helper's existing division-by-zero failure; they do not certify
that helper's diagnostic parity. The same command retains the earlier literal and helper-order
receipts and the existing reshape arithmetic gradient/finite-difference controls,
checks independent static-source proofs, IR rewrites and malformed wire edges, and executes matching
and mismatching calls from both disk and worker caches. The ignored full-class
`claimed_extent_contract` is a separate, still-pending #1277 exit, not a receipt
for these two issues. The named `insert` preparation cases now retain declared
signatures and execute their roots, but their missing caller equality checks
remain #1374/#1376 work; their measured negative failures remain in the baseline.

`scripts/runtime_extent_cache_compatibility.py` supplies additional two-binary
evidence: an actual previous producer reads its own cache, the current consumer
rejects those bytes even at its own cache path, and current/current executes
exact results or Domain failures without rewriting the cache. Its committed
v18 fixture comes from that actual producer. The pre-implementation public run
executed 90 fixtures and failed 42 contract assertions. HIP/Metal execution
remains with the documented platform handoff.

B2b-1 changes the checked-to-lowered claim carrier and every consumer together.
Its PR must name the concrete type fields and all construction/rebuild/decode
sites before implementation; compilation and negative tests reject omitted
claim transport. Any serialized carrier change amends spec/10 and uses the
next available exact WireDag version at landing, coordinated with #1298 and
other wire work then in flight. No version is reserved here. Update all
public consumers, caches, hashes, rejection controls and the typed capacity
census in that same carrier change. A decoder may not infer missing scope
from display names or accept a legacy payload by dropping obligations.
Serialize reference identities in deterministic declaration/call order, with
explicit domain remapping on import; allocation addresses or fresh-process
counters may not change the bytes/hash of the same checked artifact.

No numeric carrier receives an exception: actual extents keep their exact
tagged numeric contract, while identity references keep their distinct
reference domain. Any changed public numeric operation owes its exact
[05-OP-N] registration and generated rejection-registry membership.

### C3 No deferred expand settlement

`expand` and `insert` have exactly one result shape. Slice C, its ordered
stores, registry, source-ordinal index and completion phase are withdrawn.
No remaining Slice B entry gate depends on #1341's former settlement stores
or K-process settlement oracle. General hash-order enforcement remains that
tracker's work. #1512's remaining unresolved-variable validation audit must
start from live non-expand producers, not resurrect the deleted candidate
model. #1489's reject-unresolved gates are a separate issue and remain open.

### C4 Output sources and provenance deletion

`output_axis_sources` remains one exhaustive operation match with exactly one
source per realized output axis, checked on production paths. `ExternalAxis`
names an exact Load and axis; `InputAxis` and `ScalarInput` name validated
input slots; `OpComputed` names the operation/axis; `ClassSupplied` describes
an axis sized from an already-established claim rather than an independent
witness. Cardinality alone does not establish that a source is the right one.

An unchanged axis forwards its exact input axis. `insert` shifts later axes;
`expand` replaces only the selected one. Non-identity movement axes obtain
fresh extents, including symbolic shrink; literal stride one and zero pad
retain identity under spec/04 §4.7. Derive sources after each final rewrite,
never retain stale node ids in a cached class list.

Replace both binding and DECLARATION consumers of `symbolic_occurrences`,
`op_declared_output_axes`, `shape_source_for_axis` and `symbolic_bindings`.
#665 is a declaration-consumer failure and cannot close merely because an
entry guard passes. A synthesized Const's value supplies no shape: #1482
needs an actual axis source, not a guessed dimension or a bypass of the
cardinality check. Recheck the current sigmoid/silu/gelu witnesses; the ReLU
mechanism was removed by #1313.

Record projection also needs an executable lowering route: the preparation
alias fixture passes checking and C execution after #1658, but eval still
refuses runtime record `access`. B2b-2 owns materializing the field's tensor
as the actual shape input on that runtime-record route; #1266 requires both
spelling variants to execute, not merely that the provenance error disappear.

Only after these consumers and guards protect the admitted domain may B2b-2
remove `SizeClass`, `classify_expand_size`, `classify_arith_app`,
`sourceless_expand_size_error`, `Env::size_provenance`, and the lowerer's
provenance rejection sites. Remove `shape_deps` once each remaining use has
an actual typed dependency and no reader remains. Guards may land earlier;
acceptance may not widen earlier. A best-effort identity recognizer may
remain as a refinement whose miss yields a fresh guarded extent.

### C5 Acceptance and the preparation fixtures

The class completion command remains:

```sh
.venv/bin/python scripts/runtime_extent_oracle.py --phase final
```

Automatic success is exit zero ending `RUNTIME EXTENT ORACLE: PASS`, with
applicable HIP and Metal hardware receipts at the same head/corpus digest.
The recorded phase-B corpus has 57 rows: 38 at exit,
19 short. This is baseline metadata, not a fresh execution receipt. The final
command currently fails because `SLICE_PHASES` still requires unregistered
`c`. B2b-3 retires that requirement and its tests; it does not add a fake
passing phase or erase outstanding B rows. `--phase a` and `--phase b`
retain their names and row-transition checks.

The preparation suite is `crates/chelis-cli/tests/runtime_extent_claim_preparation.rs`.
Its baseline and full pending acceptance remain separate:

```sh
cargo nextest run -p chelis-cli --test runtime_extent_claim_preparation
cargo test -p chelis-cli --test runtime_extent_claim_preparation \
  claimed_extent_contract -- --ignored --exact --nocapture
```

The first locks measured current behavior with explicit issue-owned gaps and
runs the repaired broadcast and literal-call subsets. It must fail on an unexplained behavior
change. The second is the manual
acceptance runner over the SAME fixtures and asserts the decided contract.
It is intentionally red until the owning fixes land, reports every failed
cell, and must run before any fixture is claimed repaired. An ignored
acceptance test, a clean checker, object-only output, or equal wrong answers
on Eval/C is never a passing completion receipt. Missing compiler/toolchain
prerequisites fail the suite rather than skip a lane.

The bounded #1619 attribution receipt is the unignored `singleton_broadcast_contract` test:

```sh
cargo test -p chelis-cli --test runtime_extent_claim_preparation \
  singleton_broadcast_contract -- --exact --nocapture
```

Its 11 cases check the original literal/symbolic/folded comparisons, a static
non-unit rejection, numeric broadcasts through export/binding/inlined-main
routes, and satisfied/refuted runtime unit operands through export/binding
routes. They assert exact signatures where applicable, shapes, values and
failure context. Phase B registers this execution receipt as
`expand.positional.replacement.shape_size.eval_c`. The broader 55-case
baseline changes only the two #1619 C results and #1266's record-alias C
result: all three now execute with their expected values. The record-alias
Eval failure and direct-field failures remain; #1266 is still open.

The merged B2b-0b numeric kernel repair compares literal and resolved named
claims against independent nonnegative runtime sizes at live `Expand` and
`Reshape` nodes. A number in result metadata is a requirement, not a proof
about the carrier. Both lanes consume the shared local sites, and C checks
reshape claims before its legacy size-mismatch failure. Its acceptance command is:

```sh
cargo test -p chelis-backend-c --test exec_compile \
  numeric_local_extent_claims_execute_exactly -- --exact --nocapture
```

The matrix executes 32 lane cases: literal/resolved-name claims, expand/reshape,
computed scalar/computed tensor-axis carriers, and matching/mismatching sizes,
on Eval and C. Positive cases assert exact shape and values; negative cases
require the Domain trap, claimed value and observed node/axis value. Phase B
registers it as `guard.local.numeric_carriers.eval_c`. This receipt does not
cover claim formation from anonymous resolved metadata, negative-extent
rendering, op-computed sources without carriers, or preservation through
calls, inlining and dead-code elimination.

The local guard-order repair groups C guards by introducing operation and
retains the derivation's declaration order within that operation. Movement
consumers supply their supported extent expressions together, so output axis
order cannot reorder simultaneous claims. Legacy declarations remain separate
from this guard schedule. Its bounded acceptance command is:

```sh
cargo test -p chelis-backend-c --test exec_compile \
  local_reshape_guards_follow_declaration_order -- --exact --nocapture
```

This eight-lane matrix reverses two resolved named axes in a runtime reshape.
It asserts the exact first failing claim for each single-axis mismatch and
for simultaneous mismatches, plus exact shape and values when both agree.
Phase B records `guard.local.declaration_order.eval_c`. This closes the P2
multi-axis reshape ordering witness recorded during #1662; it does not claim
the remaining op-computed guard coverage or call/witness transport.

The #1377 exit runs six direct-call cases and 34 nested/discarded/alias
cases, with declared types and exact outputs or required traps:

```sh
cargo test -p chelis-cli --test runtime_extent_claim_preparation \
  literal_ -- --nocapture
```

It asserts the declared result and exact execution or required failure on
exported calls, top-level bindings and inlined main, including returning an
existing value after a discarded call, alias chains, borrows and shadowing.
All 40 cases pass.
Phase B attaches the direct public runner to its two inlined-root rows and
adds `claim.literal.nested_and_unused.eval_c`. The 55-case preparation
baseline now has 51 unmet cells: 24 declared signatures are preserved and
two formerly silent #1377 inlined failures trap. The remaining 18 repaired
signature cells belong to #1374/#1376/#1397; their execution gaps stay open.

The suite's rows distinguish:

- cross-tensor named claims (#1374), foreign claims on an inserted axis
  (#1376), and literal claims (#1377), each through an exported C call,
  a top-level value binding and an inlined `main`, with satisfied controls;
  named/foreign cases retain the original polymorphic `n`/`m` binders
  alongside concrete named-dimension controls;
- declared-result preservation and independent wildcard-root execution
  (#1397), plus the masked public vmap witness and its literal-bound and
  element-derived controls (#1378);
- direct record-field reads versus their local-alias spelling (#1266);
- genuine singleton broadcasts with literal and shape-derived sizes (#1619)
  versus the static non-unit rejection;
- independent same-spelled signature binders and an unread witness, rather
  than a same-name grouping assertion with no caller;
- resolved validation controls for #1512 under today's one-shape primitives.
  These are controls for its future non-expand audit, not a completed census
  or a live unresolved-variable reproducer.

Each applicable check records the declared/inferred signature, and each
execution asserts exact shape/data or the expected typed failure. Exported
C calls use runtime inputs through the public function, never a textual
assertion that an unused helper contains a guard. Rooted C cases must link
and execute a program; the wildcard-root row explicitly records the missing
entry as a failure. Failure controls name their owning diagnostic/trap, so
unrelated parsing, style, lowering or link failures cannot satisfy them.

The class oracle's remaining obligations include all legal producer/consumer
forms, negative dtype/axis/rank/extent/overflow controls, zero, real lint/fmt
transformation (#569), guard order, rebuilds that actually rewrite, source
cardinality and attribution, and exact wire round trips/rejection. These
preparation rows are a bounded subset. Implementation slices attach their
passing per-test receipts to the corresponding class rows, preserving the
transition lattice:

```text
nonconforming_rejection | silent_unguarded | ice | lane_divergent
    -> typed_unsupported(issue) -> executes_exactly
```

Invalid-program controls remain `rejects_exactly`. A typed implementation
receipt is interim capability evidence, not execution of a legal case. Repair
stale measured dispositions with fresh evidence and an explicit baseline
correction; do not call relabeling a behavioral improvement.

## Part II: remaining delivery sequence

Each row below is an owner of concrete work, not a claim that a PR exists.
All are Slice B work under #1277 unless expressly separated.

| owner | entry | deliverable and exit |
|---|---|---|
| B2b-0b: remaining local guards | merged B2r/S2b and #1658's broadcast preparation repair | guard literal and resolved numeric claims from independent local size sources; op-computed local extents; exact positive/negative C/Eval rows |
| B2b-1: claim transport | C2 contract and red fixtures; integrates B2b-0b | preserve the shipped helper-order and C2.4 checked-reshape/unit receipts (#1686/#1687); finish general scoped checked/lowered identities, explicit caller witnesses, multi-claim axes, rebuild/wire transport and migrated binding consumers; #1374/#1376/#1566, #1397's declaration-erasure half, with #1377's literal call/inlined-root exit established by the witness subset |
| B2b-root: root execution | can start independently; acceptance composes B2b-1 | #1397's general wildcard-root boundary; run or diagnose every accepted root; unlock and reverify #1378's exact public value witness |
| B2b-2: sources and acceptance | guards and claim transport for every newly admitted row | finish declaration sources (#665/#1556), supply #1482's missing shape source, remove provenance restrictions (#1266/#569/#1379), then remove unused `shape_deps` |
| B2b-3: phase exit | preceding host repairs and per-row platform dispositions | register actual passing receipts, correct measured stale baselines, retire phase c from final selection; phase b/final remain red until their named obligations pass |
| #1512 audit | no dependency on the B2b carrier or withdrawn C | enumerate reachable non-expand unresolved producers and consumer decisions; resolved/unresolved positive and negative pairs; distinguish error cascade suppression; assign each surviving defect a repair under #1512 |

B2b-0b and B2b-1 can be developed as separate changes, but their shared local
site integration must preserve both claim kinds. Claim transport is not an
out-of-scope caller problem: it is precisely B2b-1's closure requirement.
The #1377 exit now retains the declared type and an explicit call-entry
witness, with invocation dependencies preserving discarded checks. B2b-1
still owns the named scoped identities and unread named caller witnesses
needed by #1374/#1376/#1566. Kernel guard tests alone do not close those
public contracts.
B2b-root is separately bounded within #1397 so a declaration fix cannot
silently close its broader root failure. #1378 stays open until the public
witness executes; its typed Slice A mechanism need not be reimplemented.

The helper signature-order repair shipped in #1688. A helper lowered from a declared
function receives tensor inputs in that function's parameter order, including
shape-only parameters. A signatureless subexpression retains its assigned,
deterministic ABI order. Pruning and rebuilding preserve relative input order;
host callers continue mapping actual arguments by input label. A literal is
its own canonical value: when its checked value is a folded input-axis read,
class ordering uses the source input's signature slot and axis, not the later
consumer node. Named classes retain their declaring canonical witnesses.
A shared IR schedule orders individual checks across classes and claim kinds;
Eval and the C prologue consume it without regrouping:
a repeated literal cannot pull its later witness ahead of an intervening
parameter. Canonical values and witness deduplication remain attached to their
checks. The acceptance command is:

```sh
cargo nextest run -p chelis-cli --test runtime_extent_claim_preparation \
  -E 'test(=helper_signature_guard_order_contract)'
```

This covers signature `b,z,a` with satisfied, individually failing and
simultaneously failing claims through exported C, bindings and inlined main,
plus interleaved named and mixed named/literal binding checks, repeated literal
claims at nonadjacent parameters, nested discarded
calls, aliases and both executable-example variants. Internal tests cover signatureless
ABI order and reconstruction. These checks do not discharge the computed
reshape or unit-precondition obligations.

The subsequent checked-extent integration owns #1686/#1687 together. It captures
scoped claims and declaring witnesses before substitution/folding; a checked
scalar compares an independently computed reshape target before allocation,
and a checked tensor enforces an operand-axis precondition before refining that
axis. Their explicit dependencies retain nested/discarded checks through
rewrites and wire/cache boundaries. `computed_claim_result_graph_contract` adds
69 exact-type/value/failure cases for copy, negation, static conditionals, inferred
helpers, aliases, same-spelled binders in different signatures and an untaken
invalid branch, each across exports, bindings and inlined main.
`computed_claim_complete_shape_list_precedes_guards` adds 48 cases covering
later target-expression failures, first-axis matching/mismatching controls,
exact positive values, wrappers and discarded results on those same routes.
The `omitted_extent_claim_contract` runner
must assert declarations, actual shape/values and runtime Domain failures on
exported calls, bindings and inlined main before either issue closes. HIP/Metal
execution remains with the platform owners described below.

The B2b-1 carrier is one atomic integration change because dropping scope,
claims or witnesses at any checker/lowerer/rebuild/wire boundary loses the
same obligation. Separable fixture, local-guard, root, and provenance work
remain separate PRs. At implementation, size is justified by those shared
invariants and by the cases the oracle proves, not by inherited slice names.

### Platform and class interlocks

- #1112 under #729 owns exact HIP device extent metadata. Only applicable
  device-resident HIP guard rows wait; host/C rows can exit first with honest
  recorded device dispositions.
- #1298 owns runtime shape axes, windows and the device scalar path. Compose
  its oracle only for the rows requiring those facilities, using the exact
  integration head. Coordinate wire-version allocation at landing.
- #1383 / the Metal backend plan owns device expand and symbolic/rank-0
  Load support. Unsupported/stub baselines are not hardware execution.
- #730 owns typed Unsupported and execution-boundary failures; #731 owns
  witnessed checker errors. #1522's inline host expressions still need their
  own routing/guard assessment; B2h covered def application only.
- #1372 owns general DAG side-annotation rebuilding; C2.4 makes this
  contract's transport explicit without waiting for that entire class.
- #1373's exact-i64 internal carriers are independent. #1341 no longer
  supplies an extent-settlement entry gate.
- Rank-polymorphism plans own legality inside `..r`; #578 stays open until
  its complete acceptance reproducer passes, not merely an extent subcase.

Platform execution commands (manual; not implied by the ordinary suite):

```sh
scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- \
  --ignored --test-threads=1
PYO3_PYTHON="$(uv python find 3.11)" cargo test -p chelis-backend-metal \
  --test gpu_correctness -- --ignored --test-threads=1
```

Both must identify the runtime-extent group and agree with the host head and
corpus digest. A host program emitted by C for a GPU target is host evidence,
not a device execution receipt.

### Scope and freeze discipline

A slice freezes the rows and public carrier shapes it actually establishes.
Amending one updates this plan and #1277 together, with a concrete owning
PR and the required frozen-contract acknowledgement. A discovery that requires
separable work remains a named issue/row rather than an implicit expansion.

Not owned here: data-dependent output rank/shape (#600), type-level dimension
arithmetic (#526), grad symbolic-window gaps (#513), runtime axes/windows
(#1298), the rank-polymorphic legality half of #578, general DAG integrity
(#1372), exact internal i64 carriers (#1373), Metal emission (#1383),
capacity and reuse equality
over typed extent expressions
([`runtime_representation.md`](runtime_representation.md), [#888]),
#1489's reject-unresolved gates, and dtype-semantics decisions. These
boundaries do not exempt this plan's callers, claims, or accepted roots from
their named Slice B obligations.

## Considered and rejected

The following records earlier designs and reviews. Current C1–C5 and the
remaining delivery table above control this plan where those records describe
an older implementation. Scoped claim transport required by the named caller
instances is not the old global origin/transaction architecture.

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
[#888]: https://github.com/Chelis-Lang/chelis/issues/888
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
[#1380]: https://github.com/Chelis-Lang/chelis/issues/1380
[#1382]: https://github.com/Chelis-Lang/chelis/issues/1382
[#1383]: https://github.com/Chelis-Lang/chelis/issues/1383
[#1397]: https://github.com/Chelis-Lang/chelis/issues/1397
[#1467]: https://github.com/Chelis-Lang/chelis/pull/1467
[#1480]: https://github.com/Chelis-Lang/chelis/issues/1480
[#1482]: https://github.com/Chelis-Lang/chelis/issues/1482
[#1313]: https://github.com/Chelis-Lang/chelis/issues/1313
[#1318]: https://github.com/Chelis-Lang/chelis/issues/1318
[#1328]: https://github.com/Chelis-Lang/chelis/issues/1328
[#1489]: https://github.com/Chelis-Lang/chelis/issues/1489
