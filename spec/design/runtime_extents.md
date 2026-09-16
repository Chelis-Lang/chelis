# Runtime Extents: values, claims, and their witnesses

**Status:** IN PROGRESS. Slice A and Slice B's mechanisms have landed, and
every non-deferred row of the recorded phase-A and phase-B corpus is at an exit
state: `runtime_extent_oracle.py --phase final` passes on the host lanes, with
the HIP and Metal hardware receipts still owed separately at the same head and
corpus digest. Slice C is withdrawn: `expand` broadcasts a singleton axis and
`insert` adds an axis, so neither operation produces a deferred shape. A green
class oracle is not a closed class: the recorded rows are a bounded subset, and
the named residual work in Part II and the open sub-issues of [#1277] remain.
Tracking parent: [#1277].

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
Deliveries after that sha are recorded in [#1277]'s ledger rather than copied
here; the executable statement of what is landed is the oracle's own row
report.

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
| B2h memo representation | #1910 (chelis#1835) | the kernel decision's per-program facts are fields of a `HostLoweringSession` bound to its program; the thread-local flag, the pointer keys and the arming guard are deleted, and the callee summary probe is the last host-lane predicate asked |

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

The eval before/after-effect rows in `runtime_extent_slice_b` assert actual
transcript bytes across failure (#1585); C now flushes observable output before
its traps (#1591).

The checked transport in C2.4 implements the restored #1686/#1687 host
obligations: computed reshape claims and broadcast unit preconditions survive
inlining, graph rewrites and cache transport. Their bounded oracle retains
independent declaration, value and failure assertions. Remaining failure boundaries:

- #1397: the declared result of a runtime-bound movement is retained and
  guarded at the outermost activation, and a root whose result type carries a
  runtime extent executes on both lanes. #1798 closed the retention gap for a
  declared axis that passes through an op-computed extent. Its original receipt
  attributed a returned `add`'s claim to the operand-side `shrink`; #1948
  corrects that attribution under spec/04 §4.7, which names the primitive that
  produced the returned value. `axis_sources::op_computed_axis_origin` remains
  the pass-through resolver when the reached operation is itself the returned
  value's producer after administrative copies/casts. It is not a selector
  among a same-shape producer's operands.
  #1800 is closed: a named claim whose declaring witness observes a graph-fixed
  extent is stamped resolved, and a single resolved op-computed member is a
  complete class. #1801's arm of the same predicate is closed too: a root that
  kept an unresolved dim VARIABLE rather than a runtime extent was dropped from
  both lanes because the checker left the variable free at a concretely applied
  root, and `spec/04-type-system.md` §3.2 now makes a dimension variable an
  application's instantiation minted, that unifies with a runtime extent and
  that no argument of that application binds to a literal or named dimension,
  denote that extent. What denotes the extent is the variable's ALIAS CLASS, so
  a polymorphic def passed as an argument, which mints the variable that
  becomes the class's root, is covered, and so is a class of three or more
  members whose meeting was recorded on a member a later binding re-rooted
  over: the wildcard meeting and the authored-name pin are properties of the
  CLASS, maintained on its root and merged at every union, so argument order
  cannot change the verdict. A PARAMETER-bound binder keeps its
  name. A RESULT-ONLY binder is absorbed to the runtime extent it met, and that
  case was LANE DIVERGENT on `0820ee28e`: `def outer(t: tensor[3, f32]) ->
  tensor[seq, f32] = apply1(h, g(t))` with a root built, linked and printed
  correctly on C while eval refused it for a missing `seq` binding, and the
  one-call-shallower `= g(t)` spelling already published `tensor[*, f32]` on
  both lanes. §4.7 requires every execution mode to observe the same values,
  §4.7.3 forbids a verdict that turns on a function boundary, and §4.4.1 makes
  a dimension that occurs only in the declared result output-inferred from what
  the body produced; the `root.dim_variable.result_only_binder` corpus pair is
  the receipt. The manifest's
  `DeepTag::DVar` refusal is unchanged: an uninstantiated variable still has no
  ABI.
  #1378's public vmap witness is no longer masked and executes with its exact
  value on both lanes.
- #1266/#569 are admitted: the walk resolves a `shape` operand by its TYPE, so
  a record field's tensor answers where the ADT base could not, and it folds a
  `pipe` into the staged application it denotes.
- #1379 is closed. Local op-computed extents have guard sites, `shrink` is the
  admitted owner, and the lowering now admits an arithmetic `expand`/`insert`
  size as an ordinary `RtDim::Node`, so the guard has the value to check. The
  checker's provenance walk stopped letting a sourceless operand poison an
  expression that already carried a real shape source, and its operator set now
  agrees with the shared static folder's. A size with no admissible operand at
  all is still sourceless with its unchanged diagnostic. Since chelis#1791 the
rule also runs in pipe position, and no rule moved to achieve it: the operand
of a pipe stage used to reach `check_expand_signature` as an unresolved type
variable, whose arm returns before the size rule, and after the fold the
operand is the real expression. Hoisting the provenance rule above the
operand-type match would also have closed the hole, and was measured to
replace the direct-position rendering for `insert(b, m, cast(k, int64), n)`
with the sourceless one: a silent change to an established diagnostic that
nothing asked for. Stating the pipe's meaning once cannot have that effect.
With #1266/#569
  admitted beside it, no provenance restriction remains and removing the walk
  itself is all that is left of B2b-2's acceptance half.
- #1482 needs an actual shape source for a synthesized constant. The
  declaration consumers no longer use legacy name recovery: every name a lane
  renders resolves through `ExtentOrigin`, and a name that resolves to none is
  a typed receipt. #665's kept-axis instance and #1566's binder-spelling
  instance are closed; #1556's published instance does not reproduce in its
  current spelling.
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
derivation already read the correct operand and is unchanged. An explicit name
ON THE EXPANDED AXIS retains its existing preparation path: preserving the name
without its unread declaring witness can newly execute a wrong shape. A name on
a KEPT axis does not, and chelis#1822 is why that distinction is the rule rather
than "outputs with explicit named dimensions": gating on every axis sent an
`expand` whose bystander axis carries a signature binder to the same-rank
shortcut, which stamped the operand's pre-expand extent onto the axis `spec/05`
section 2.4 replaces. B2b-1 owns preservation and enforcement of those scoped
claims.

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
survivor. A declared result's NAMED claim is carried by the witness that
observed the produced extent, as a named claim against the witness that
declares the binder, so the obligation survives every call form and reaches a
declared-but-unread parameter, whose interface witness the claim retains
(#1374, #1376, #1566's unread residual).

The same claim/source derivation is available before any deleting rewrite
for its liveness decision and after the last rewrite for emission. An
unproved extent guard is potentially trapping and therefore observable under
spec/06 §5.2. DCE retains its witness and bound-producer dependencies even if
its tensor result is unused. A genuinely total, proven-satisfied claim adds
no trap root. A rewrite that replaces a computation transfers the obligation
to the corresponding replacement axis or preserves the necessary guard
computation; it does not keep both the obsolete tensor computation and its
replacement merely to preserve a stale node id.

The #1821 independent-activation repair separates the producer's physical
shape from declaration claims. A printable binder or checked caller label
cannot join different activations or supply the runtime extent of a producing
operation. Existing parameter-witness equality checks own result obligations
only where the observed result source is that exact checked witness; this
discharge requires the source relationship, never equal spelling. Such an
entry-owned equality executes once in signature order and creates no duplicate
local guard.

For a locally computed named result, lowering captures the declaring witness
before entering nested callees. A fresh shape-only `ExtentWitness` with a
`ResultClaim` site records its diagnostic label and claimed result axis,
observes the declaring tensor axis, and retains the declaring witness as a
dependency. The actual introducing primitive retains each claim token through
`shape_deps`. Guard derivation compares the primitive's observed axis against
the token's scalar by node identity, preserving multiple requirements in
declaration order and the original primitive's trap attribution. It never
uses the printed label as a runtime dimension lookup or overwrites physical
result metadata with that label. Existing literal and checked-reshape carriers
retain their own obligations. Checked helper-label views remain a separate
concern; adding a view is not evidence that a claim survived.

Literal declared results transferred into the private pure-helper slice use a
distinct `LiteralResultClaim` witness role. Lowering selects that ownership mode
from the authored declaration identity and its top-level call relationship, not
from later graph reachability. The same mode is explicit in the ordinary
private gradient subcontext so zero and unused cotangents retain the forward
guard; mapped-gradient transport remains separate. Direct and root lowering,
movement and device paths, and mapped-gradient lowering retain their established
literal carriers rather than gaining a token because an equal dimension is
reachable.

In the selected mode, lowering allocates one token containing the exact tagged
requirement before lowering that declaration's body, then attaches it to the
returned producing operation only when that producer's extent source is already
admitted. A non-unit-stride op-computed extent that was previously unadmitted
therefore retains its legacy carrier. Nested declarations attach their
obligations before enclosing ones; guard order follows those ordered
attachments, not token allocation order. The shared producer-site derivation
places casts at their source while retaining cast attribution. An
already-produced result gets an invocation boundary carrier rather than a
backward dependency to its old producer. Only the exact axis whose token is
installed stops using the legacy literal stamp; administrative copies and casts
may carry that ownership, while physical extent metadata is not a substitute
for it. DAG rebuilding preserves token identity and order, and operation
rewrites must retain or transfer the guard's primitive provenance. Literal
tokens have no input tensor and no element derivative.

The additive site role requires exact wire transport and admission, including
its axis and dependency validation; predecessor artifacts must reject. AD,
DCE, graph splicing and vectorization must preserve or correctly remap both
the token and its owning producer edge. Storage lifetime planning follows
`shape_deps` as well as value inputs: a claim token's buffer remains live until
its producer consumes the guard, so later shape computations cannot overwrite
the required extent. Ownership borrowing and storage reuse must agree on those
same edges. These are implementation obligations, not completed receipts. The regression selection must include independent
same-spelled calls, computed-only claims, unused/zero cotangents, and #1991's
lost parameter/result equality and original-`shrink` attribution controls.
No arbitrary eager value or effect is encoded as a shape-only dependency.

Six additional eval/C corpus rows preserve independent-call agreement, ordered
entry failures and computed-result claim attribution. Their tests exercise
same and distinct callees, renamed binders, written target order and zero or
nonzero cotangents. The computed case requires the original `insert` failure
before a later elementwise operation; the agreeing case checks every gradient
value. Existing #1991 helper-label controls retain their historical result,
including the separately tracked missing `seq` binding.

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
| grad | keep authored signatures separate from inferred expression types before lowering the activation; form every declaration claim against its ordered interface witnesses, then retain the forward activation as a shape dependency of every selected cotangent, including zero cotangents. Tensor, aggregate, and primitive-scalar single/multiple selections share this evaluator contract (#1920/#1924/#1934). Native primitive-scalar selections use the existing DAG cotangent reconstruction route under the same contract (#1934). The dependency is unconditional (#1935); bounds keep spec/05's zero-cotangent boundary |
| vmap | preserve the binding/claim relationship with shifted axes, share the rank-0 bound, and execute it once as spec/06 §3.7 requires; `vmap(grad(...))` retains the batched forward activation as the batched cotangents' shape dependency, remaps the complete ordered entry-witness set and every rendered dimension origin through the batched node map, and fails before publishing an artifact when any mapped root, witness or dimension declaration is unresolved (#1932) |
| wire/cache | preserve claim references, ordered witnesses and independent facts or reject the artifact; no missing-field empty default |

This table is an implementation acceptance obligation. The preparation
fixtures below cover public call/root witnesses; they do not already prove
all rebuild/import rows. The existing B2b-0 receipt proves seven named IR
passes on transforming fixtures, not lowering-side `splice_dag`, imports or
the new contract carrier. B2b-1 extends those tests before changing the carrier.

##### Mapped gradient entry witnesses and artifact closure (#1932)

Ordinary `grad` closure does not discharge the mapped path. Before
differentiation, `vmap(grad(f))` lowers the authored activation signature and
forms its complete ordered entry-witness set exactly as the corresponding
unmapped call does. Vectorization remaps each witness input, claim,
requirement, shape dependency and rendered dimension origin through the
`vectorize_axis0_with_node_map` result. Cotangent packing and the final
`splice_dag` retain those remapped dependencies even when the cotangent is
zero or is the only result root.

Root correspondence is necessary but not sufficient. After the splice, the
shared verifier checks that every live claim has its witnesses and that every
dimension name a lane can render has one live declaration or producer.
Failure is a fatal typed lowering result before Eval starts or C artifacts are
reported successful. The host fallback may not absorb that failure, Eval may
not translate it into an unstructured "no roots" message, and
`build --target c` may not exit zero after writing a source file that refers
to an undeclared temporary.

The #1932 exit matrix contains the exact named-entry-witness reproducer and an
agreeing control, each on Eval and compiled C. It also covers a zero cotangent,
a nonzero cotangent, a non-identity node map, and a reordered or aliased
callable path. Success means exact gradients or the activation's exact
[04-NUM-9] trap; a supported-fragment refusal must be the same typed nonzero
lowering failure on both lanes and produce no purportedly successful C
artifact. The C receipt compiles, links and runs every successful build.
Deleting any witness remap, mapped shape dependency, root correspondence or
dimension-origin declaration must make a named negative control fail.

#### C2.5 Guards consume the observed quantity

Derive equality classes by scoped claim identity and literal obligations,
with sources supplied by `output_axis_sources`. Signature order selects the
canonical interface witness; local-only classes use introducing source order
to schedule distinct obligations. Neither rule chooses a declared-result
diagnostic owner: spec/04 §4.7 assigns that role to the primitive that produced
the returned value. A literal is the required value itself, not the first
observed member.
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

An op-computed site therefore states a THIRD read instruction beside the
carrier and the realized extent: the extent the operation is about to compute,
expressed from that operation's own bounds and evaluated before it runs. The
derivation owns which owners supply one, and that one answer also gates
lowering's declared-result stamp, so a claim cannot be written onto an axis
with no site to check it. `shrink` and `pad` with non-zero padding supply one.
A non-unit `stride` supplies `ComputedAxisExtent::StrideSpan` after the
independent stride-step precondition below has established a positive step
(#1907/#1931).

Two further rules bound what the stamp may write when an op-computed operation
itself produces the returned value. A resolved op-computed claim is its own
canonical value, exactly as C2.4 makes a literal one: a single member whose
source is op-computed and whose dim carries a resolved name has both a number
to compare against and an extent to compare, so it is a complete class, while
an unresolved single member still forms none. And a claim the owner's own
rule statically PROVES a different value for is not runtime-checkable at all:
section 4.7.2 conditions its guard on a claim "that is not statically proven
equal to `size`", the IR verifier's per-owner size check rejects a graph that
states a refuted one, and the verdict therefore belongs to sections 4.4/4.5.
A user-spelled name already on the origin axis is another signature's claim
and is not relabeled either; the compiler-minted spellings for a fresh extent
are.

When such a class's canonical value is a binder no interface witness declares,
the FIRST site in derivation order declares it from its observed extent and
every later site guards against that value. This is C2.4's canonical-member
rule reaching the local sites, and it is the rule the C emitter already applied
through its declare-then-guard split; stating it here removes the asymmetry in
which the evaluator skipped such a site while C emitted its comparison.

The unit precondition on `expand` is a separate obligation from its result
extent. Its observed value is the OPERAND axis before replacement. Its result
source is `size`. #1619's symbolic-size and folded-shape-size rows must prove
that these cannot be swapped. `insert` has no unit-operand precondition.
Multiple obligations on one axis survive independently; coalesce a duplicate
representation of the same obligation, not distinct trapping operations.

##### Stride preconditions and result extents (#1907 and #1931)

A stride step is read as its signed `int64` carrier and validated before any
conversion to an index type, ceil-division, allocation or element access. A
runtime step less than or equal to zero executes one stride
operation-precondition guard and raises `Domain` with the exact
`numeric trap: domain in stride at int64` line and its stride-step context.
Eval and C consume the same observation and neither substitutes step one,
returns the input extent, or reaches a result-claim or generic movement-target
check first.

Only a validated positive step reaches
`ComputedAxisExtent::StrideSpan`. That site reads the realized operand axis and
the same step carrier, computes `ceil(operand_extent / step)` without an
overflowing `extent + step - 1` intermediate, and is available before the
stride executes. Step one remains the identity-only pass-through form; every
other positive literal or runtime step introduces a fresh extent whose
declared literal or named claims are preserved and checked at the stride.
This ordering is the semantic interlock between #1907's operation
precondition and #1931's result claim: an invalid step wins, while a valid step
whose span disagrees with its claim reports the claim rather than the
allocation backstop.

The combined matrix crosses literal and runtime step carriers with step one,
positive non-unit, zero and negative values; agreeing, disagreeing and free
result extents; and one- and multi-axis tensors. Each legal row executes exact
values on Eval and compiled C. Each invalid row checks the exact first failure,
source/axis/value context and exit status. The final runtime-extent corpus
keeps independent #1907 and #1931 closure rows even when one implementation
change delivers both.

##### Same-shape declared-result ownership (#1948)

A returned same-shape operation is the primitive that produced the returned
value, so it owns the declared-result guard. The guard does not inherit the
identity of whichever rank-matching operand `shape_preserving` encounters, and
`op_computed_axis_origin` does not recurse through the same-shape producer to
choose one. For the exact witness this makes `add`, not either input-side
`shrink`, the context and [04-NUM-9] operation.

Lowering attaches the existing literal/named result-claim token to the
same-shape producer itself through a dedicated `result_claim_deps` lane. That
lane is an execution obligation, not a value input or an ordinary
`shape_deps` consumer: it keeps the token live and ordered without changing
tensor fanout, copy insertion or storage ownership. Guard derivation
represents the observation as the complete nonempty set of positive-rank
inputs whose rank equals the result rank, paired with the result axis. Rank-0
scalar inputs remain outside that set. The representation has no
selected-origin field: repeated edges may be deduplicated as the same
agreement member, while distinct operand paths remain distinct members. A
producer carrying a claim without a complete nonempty agreement relation is a
verifier error rather than an empty observation, so a missing or malformed
relation cannot drop a claim. Rank-erased staged-control handles that carry no
producer claim remain outside this envelope. Multiple claim-capable operand
origins are not collapsed or treated as a request to choose: the complete
member set remains the observation, and successful operand agreement
establishes its one result extent.

At execution the operation's independent operand-rank/shape agreement runs
first across that complete set. Only after agreement succeeds does the
declared-result guard compare the agreed output-axis extent. A disagreement
there reports the same-shape operation and traps before the operation's
element kernel; an operand disagreement reports the operand guard and never
reaches the result claim. A graph-fixed result extent that contradicts the
declaration remains the pre-execution `DimensionMismatch` disposition.

`output_axis_sources` may retain one representative input for physical shape
transport and unrelated dimension-class derivation. That implementation
representative has no diagnostic authority. The result-claim observation is
derived in memory from the complete agreement relation, while the existing
`LiteralResultClaim` / `ResultClaim` token remains the durable graph identity.
WireDag publication projects the dedicated internal lane into the existing
role-tagged non-value dependency transport; the witness site distinguishes a
result claim from an ordinary shape dependency at admission. No new
`ExtentWitnessSite`, WireDag field or schema version is required for #1948,
and arbitrary producer-origin selection is unrepresentable.

The #1948 matrix includes the exact operand-1 witness, its reversed-operand
twin, an agreeing declaration, runtime operand disagreement and precedence,
rank-0 exclusion, repeated edges to one path, distinct operand paths, no
claim-capable operand origin, and static refutation. Eval and compiled C must
both name the returned same-shape operation for a result-claim failure. These
rows have receipt identities independent from the local-ascription leaf below.

##### Local tensor ascriptions (#2110, split from #1948)

A local tensor ascription is an extent claim under C2.3 even though it is not a
function result. Lowering records a `LocalAscriptionClaim` at the annotated
binding, attaches it to the initializer's producing operation before aliases
or inlining can erase that relationship, and retains it through the same
rebuild and liveness paths as result claims. A graph-fixed disagreement is the
spec/04 §4.7 pre-execution rejection. A runtime-dependent disagreement is a
guard at the initializer's introducing operation. Physical result metadata,
the C verifier's target-shape assertion and a later result declaration are not
substitutes for that obligation.

`LocalAscriptionClaim` is an additive exact site role under C2.3. Any checked
artifact or WireDag that carries it must allocate a versioned field, preserve
it exactly and reject a missing or unknown representation; a default-empty
decode is forbidden. An internal-only representation must reject publication
at a boundary that cannot encode the role rather than silently erase it.

The #2110 matrix covers direct, aliased and inlined bindings with agreeing,
runtime-disagreeing and statically refuted extents, including the incidental
function-result failure and the wildcard-result form that currently succeeds
silently. Check disposition, Eval and compiled C must agree with the
numbered-spec verdict; no row may exit zero with an undeclared extent or
terminate through an internal assertion.

##### Host declared-result guards (#1771)

A declared literal result creates an obligation independently of whether its
return expression names one producer. The obligation retains the declaring
result and its ordered literal axes. Producer attribution is selected when
the producing expression executes: [04-NUM-9] requires that primitive's name,
not the enclosing user function's name. Lexical aliases retain their defining
scope; an `if` or `match` forwards the obligation only into its selected arm.
An untaken arm neither evaluates its producer nor checks its obligation.

The implementation selected for #1771/#1945 carries inherited obligations
through the private owned-function calling interface. A shared callee has one
body and receives its caller's obligations per invocation, alongside its own
declared-result obligations. Published wrappers start without inherited
obligations; no public ABI parameter or per-literal specialization is needed.
The evaluator carries the same invocation-local information. Argument
evaluation does not inherit a claim on the call's result, and one invocation's
claims must not leak into a later invocation.

After successful helper lowering, the lowering result records whether that
exact authored declaration transferred its literal obligations into its tensor
helper. The host emitter consumes this explicit ownership record when deciding
whether to construct the declaration's frame; DAG reachability, a whole
`TensorCall` body, or equal literal dimensions do not establish ownership.
Direct and root lowering retain their legacy carriers, while inherited caller
claims remain invocation-local plans.

The selected producer consumes the applicable obligations before effects that
follow it in source order. Distinct declarations remain distinct obligations;
forwarding the same obligation twice does not create another check. A helper's
existing DAG guard owns only the obligation it actually represents, not every
claim reaching a call that happens to use that helper. Entry guards retain
their separate position before body execution. The acceptance receipts must
check branch selection, primitive attribution, lexical shadowing, effects on
both sides of the producer, and different callers of one shared callee on Eval
and compiled C. These receipts cover host-builtin producing expressions. An inherited claim
ending in a lowered tensor helper still needs distinct producer-site transport;
the helper's existing local claim is not evidence for that inherited claim.

The continuation carries inherited literal obligations across tensor helpers
and supported callable invocations. Each invocation supplies its obligations;
the selected producing operation consumes them independently of helper-local
claims and signature-entry checks. Preserve producer provenance through aliases
and dynamic selection. When the selector is available before production, only
its selected producer receives the inherited claim. When a later computation
determines selection, retain provenance and check the selected value once the
guard operands are available, before subsequent effects. This follows section
4.7's readiness and source-order requirements: do not speculate selector effects
or impose an inherited claim on an unselected alternative. Acceptance covers
both selector timings with exact primitive attribution and effects before and
after the guard. The existing host-builtin receipts alone do not establish this
continuation's completion.
The pure-helper slice enrolls `return.pure_helper.literal.{eval,c}`. Callable
transport and selected-alias provenance remain separate continuation slices;
their pending witnesses do not count as passing pure-helper receipts.
The receipts do not establish general preallocation coverage for every host
primitive. Named host declared-result claims remain #1900.

##### Host entry guards (#1788)

Retain the expanded checked signature before helper extraction, body
refinement, inlining, or parameter pruning. Construct one ordered entry plan
from that retained declaration. The plan retains each literal obligation and
repeated binder's ordered parameter-axis
witnesses, source labels, and scoped identity. Compare every later binder
witness with its first witness; equal spellings in independent signatures do
not create an equality. Signature order, not helper extraction order, selects
the first semantic failure.

The owning invocation executes the complete plan before entry ownership drops
or body operations. This includes obligations whose witnesses the body never
reads and preserved monomorphized signatures. A private helper may omit only
the exact signature obligations already executed by its dominating host entry;
its local operation and result guards remain independent. Standalone helper
entry retains the complete checks.
Executable beta reduction retains this invocation boundary: evaluate every
actual once in caller order, including unused actuals, then check the authored
lambda signature before its substituted body. The enclosing function and
callback signatures keep separate obligations. Shape-only substitution does
not authorize erasing the executable argument or entry boundary. Ownership must account for every obligation
exactly once, rather than suppressing the enclosing plan when any helper owns
one obligation.

Higher-order inlining must retain an invocation boundary: evaluate actual
arguments once in caller order, map the original signature witnesses to those
values, execute its entry plan, then run the substituted body and callbacks.
An indirect invocation likewise retains its checked callable contract.
Host specialization represents a callable formal with entry obligations as an
explicit checked adapter around the supplied callable syntax. Eval retains the
same authored formal signature as an invocation contract on every supported
runtime callable value (ordinary closures and `grad`/`vmap` transforms). Both
forms consume one `SignatureEntryPlan` before the supplied callable's own
entry, transform, kernel, or body, without re-evaluating payload actuals. Inline
collection callbacks retain a signature-entry node before their body, so each
iteration checks even a parameter whose value the callback never reads. Eval
and C consume the same ordered obligations. The #1991 `apply4` control compares
the authored parameter `p` witnesses before callbacks; independent result
labels named `k` cannot supply that equality. #1991's unpublished draft is not
a dependency of this implementation.

The shipped-example census keeps result guards and entry guards as distinct
contracts. Its result-guard set remains empty. Entry obligations are derived
independently from checked source signatures in parameter/axis order, using
the authored signature before body refinement where present and expanding
checked type aliases. The reader compares that expectation against every
emitted owned function, including a definition with no discovered checks.
The comparison includes each observed and canonical witness axis, literal or
binder claim, order and multiplicity. Missing, duplicate, reordered and
wrong-axis controls must fail. This replaces the obsolete empty entry-guard
expectation when complete signature checking adds wrapper guards; it neither
introduces a second example roster nor derives authority from emitted code.
The same source-directory traversal and asserted capability refusals retain
the enumeration witness. Existing example parity and signature-entry runtime
receipts remain required alongside this structural emission check.

#### C2.6 Atomic integration and wire ordering

B2b-1 first implements literal call claims through an explicit IR witness.
This bounded change owns #1377's shape-derived `insert` call and its nested
and discarded-result controls. A literal identifies its own required value;
it does not need to identify a named binder by spelling. The named half
followed and delivered scoped binding identities, unread named witnesses, and
#1374/#1376/#1566 for TENSOR-typed parameters; a binder reached only through a
container type mints no witness. At an inlined root the checker infers the
result dimension by instantiating the callee's binder, so the root RESTATED
that binder as a literal and the restatement rendered first; a literal claim
whose comparison a named claim already makes now records no requirement, and
the guard the user sees names both sources (#1782). That declination is
bounded on the ENTAILING side: the other witness of the named claim must
observe an axis whose extent the lowered graph FIXES, which at an inlined root
is the argument's own `ConstTensor`. The bound is decided by PROVENANCE, with
`resolve_axis_extent`, and its four origins are total: `Literal` is fixed,
while `ExternalAxis`, `ScalarInput` and `OpComputed` are not. An ABI
parameter's axis is an interface obligation the entry guard checks rather than
a fact of the graph, so it never entails a literal however many PASS-THROUGH
hops separate the parameter from the witness, and those spellings keep the
requirement. An operation that fixes the extent itself, a `reshape` to a
literal target among them, has a `Literal` origin and does entail it; such an
operation imposes that extent or traps before the declared result exists. Deciding this by the
neighbouring operation instead does not hold: matching the observed node's op
against `Load` loses the bound at the first intervening `mul` or `cast`, which
is what a syntactic stand-in for provenance costs. The declared dimension is
still stamped on the result type in both cases; only the requirement is
declined. Both changes
retain the C2.3 distinction between a requirement and an independently
observed extent.

The transport uses `RiscOp::ExtentWitness { site, parameter, axis,
requirements, claims }`. Its FIRST input is the actual argument, followed by
one input per named claim, each an earlier `ExtentWitness` of the same
activation; its result is the observed axis extent as a rank-zero `int64`.
`parameter` is diagnostic text, `axis: RtAxis` selects the observed axis,
`requirements: Vec<ScalarValue>` retains ordered, tagged `int64` literal
claims, and `claims` retains one entry per requirement input, each carrying the
dimension binder and a `requirement_declares` flag saying which side declares
it, because either side can be the later witness and the edges do not recover
that role. Requirements and claims are explicit fields, with no missing-field
default. The operation reads shape metadata without copying the argument's
elements. Its existing `span_id` records the introducing call.

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

Host-sourced extents use a shared staged function plan. A supported scalar
`int64` expression that cannot execute in the tensor DAG keeps its checked
source expression and captures the current activation's
values explicitly. Its result supplies a fresh typed scalar input in the logical
DAG. Lowering attaches result claims through that complete graph before splitting
it into executable helpers; a helper's result metadata cannot reconstruct the
lost provenance after a split. Existing native arithmetic lowering remains in
place where it already carries the source correctly.

Stages execute at their original source positions. The preceding graph segment
executes eager expressions even when their values are unused and exports only
values required after the cut. A completion dependency retains that execution
without exporting every intermediate tensor. It materializes each capture once;
the existing host evaluator or host C lowering evaluates the scalar expression
once, and the following segment consumes its
tagged result. Complete shape-list evaluation precedes checked reshape carriers
and allocation, including mixed host/DAG producers and discarded results. The
same plan drives Eval and C; host C emitted for HIP follows it too. Scalar control
inside a source expression stays within that expression. The plan does not move
a source out of a tensor branch or make tensor control flow eager.

Plan validation requires exactly one producer for each staged input, available
captures, matching types, and no unresolved placeholders in executable helpers.
After cache admission, the plan is regenerated from the checked program and
mandatory authored-signature ledger before definition execution. Any admitted
transform must preserve stage/value mapping and execution multiplicity. These
obligations are not discharged by the
native remainder path: the active `host_produced_reshape_targets_preserve_declared_claims`
fixture covers bitwise, metadata, list, helper and scalar-conditional producers,
with direct and bound/copied results. Non-tensor locals retain typed host values;
capture identity is keyed by the producing node or host value, never the binder
spelling. Scalar/tensor conversions receive distinct view nodes before partition,
so both aliases retain their host surfaces. Native aggregates retain their checked host type and constructor metadata beside
their field graph. Captures reconstruct that typed structure from already evaluated
leaves, including nested lists and tuples; tensor consumers retain the original
leaf graph. Packing does not replay arithmetic or effects. Host literals use typed
source values. Once a source is selected, capture failure cannot fall back to
unclaimed execution.
Static function aliases retain their resolved definition at the binding
position: Eval captures the existing function value, and C projects that same
identity into direct calls while respecting nested binders. Later shadowing cannot
retarget the call. Calls into staged definitions retain the shared plan instead of
re-extracting a tensor-only helper. Host control boundaries are an explicit
planner result, so a fallback cannot silently retry whole-function DAG lowering.
Random handlers retain host scope, each tensor segment consumes the live handled
stream, and CSE preserves distinct activated draws.

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
The admitted extent arithmetic includes signed remainder. `RiscOp::Mod` uses
the existing `Numeric:mod:TableA` registration under [05-OP-64] and exact tagged
integer kernels. It stays materialized through fusion, preserves its inputs
through rebuilding, and uses checked C arithmetic. Integer zero-divisors retain DivZero; remainder
by -1 is exactly zero without forming an unrepresentable quotient. This closes the host-route
claim bypass for remainder targets: selecting `mod` cannot discard the authored
reshape obligation. HIP entry selection retains the checked realizability lane
through helper extraction on both CLI and API paths. The host C artifact is
executed independently in the oracle; genuine tensor roots still select device
emission. HIP device execution remains with the platform owners.
WireDag v12 carries checked operations, integer remainder, witness-to-witness
requirements and the exact `ResultClaim` site with its declaring and producer
edges. Stdlib/dependency-library/context cache formats 21/16/23 retain the
authored signature ledger, checked operation-family restrictions and lowered
graph result-claim roles. The ledger is revalidated against fresh lowering.
Earlier serialized-graph formats reject before payload decoding, including
formats 20/22 used by the declaration-admission change before integration.
Missing fields and a forged ledger with a valid checksum and unchanged proof
identity reject at admission.

The completion oracle for these two host obligations is:

```sh
cargo nextest run -p chelis-cli -p chelis-ir -p chelis-compiler-api -p chelis-backend-c --lib \
  --test runtime_extent_claim_preparation --test runtime_extent_checked_transport \
  --test wire_extent_witness --test disk_cache \
  --test issue_513_symbolic_axis_adjoints --test exec_compile \
  --test runtime_extent_slice_b --test issue_912_root_boundary --test cli \
  -E 'binary(runtime_extent_claim_preparation) | binary(runtime_extent_checked_transport) | binary(wire_extent_witness) | test(=cached_imports_preserve_computed_claims_and_unit_preconditions) | test(=previous_checked_extent_cache_is_rejected_before_payload_decode) | test(context_decode_rejects_missing_or_forged_authored_signatures) | binary(issue_513_symbolic_axis_adjoints) | test(static_reshape_folding_requires_independent_axis_sources) | test(checked_remainder) | test(staged_plan_) | test(cse_preserves_executed_random_draws) | test(=a_local_unit_extent_claim_is_guarded_on_the_hip_host_lowering) | test(=hip_tensor_root_uses_the_manifest_to_emit_a_gpu_executable) | test(=build_hip_executes_host_reduce_window_with_exact_shape_and_values) | test(=build_hip_host_rejects_unimplemented_window_dtype_cleanly)'
```

The new public matrix has 117 initial exported/binding/main fixtures, 69
result-graph fixtures, 48 complete-shape-list scheduling fixtures,
six folded-source caller-contract fixtures, 48 producing-source expression fixtures,
six dynamic remainder fixtures and 12 HIP host CLI/API executions,
three executable example controls, 24 grad/vmap controls
and 16 imported-call controls. The staged-source coverage adds 60 producer
fixtures, 60 capture/order fixtures, six eager-source fixtures, six scoped-witness
fixtures, 12 HIP host CLI/API executions and 36 handled-Random fixtures. Thirty tuple and scalar/tensor-view fixtures
cover structured captures and both aliases at a stage cut. Twenty-four native-list
and host-literal fixtures retain constructor types and claims. These
583 public fixture variants check declarations independently of actual
shape/value or required failure. Random values are checked at exact f32 bits. Two integration controls execute a
checked HIP host window entry with exact shape/data and require a clean error
for its unimplemented bf16 cell. Current [05-RWIN-2] permits the operation;
these controls distinguish the selected host implementation from device support.
Claim mismatches require Domain/reshape/int64. Scheduling fixtures
independently require Eval's division-by-zero/floor_div/int64 diagnostic and
the C integer helper's existing division-by-zero failure; they do not certify
that helper's diagnostic parity. The same command retains the earlier literal and helper-order
receipts and the existing reshape arithmetic gradient/finite-difference controls,
checks independent static-source proofs, IR rewrites and malformed wire edges, and executes matching
and mismatching calls from both disk and worker caches. The full-class
`claimed_extent_contract` is a separate #1277 exit, met in B2b-3, and was
never a receipt for these two issues. The named `insert` preparation cases now retain declared
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

Both binding and DECLARATION consumers now read the derivation.
`symbolic_occurrences`, `bind_symbol_from_any_load` and `symbolic_bindings` are
deleted; `op_declared_output_axes`, `shape_source_for_axis` and
`op_internal_symbolic_dims` retain five other callers and their migration is a
separate slice. A declaration comes from `resolve_axis_extent`'s terminal
`ExtentOrigin`: an input tensor's axis goes in the prologue, and an extent an
operation produces is declared at that operation, which is how a kept axis
forwards its exact input axis without renaming the claim it carries. Choosing
among a name's candidate axes follows where a declaration can GO, not which
answer is most certain: an input axis wins, an operation-produced extent comes
next because it names a site, and a literal comes last because no lane declares
an entry literal today, so preferring one over an available site would leave
the name undeclared.

A declaration is keyed by NAME across the whole graph, which is correct only
while one name means one extent in one emitted function. Two roots merged into
one function can each declare the same binder from their own signature, and
scoping deliberately keeps those two witnesses in separate classes, so nothing
compares them; before #1788 they nevertheless shared one C variable and the
second root sized its work with the first root's extent. The repair is upstream
of every declaration consumer rather than inside one: `prepare_dag_for_codegen`
gives each scope after the first its own identity, `<name>__s<k>`, on output
types and op-internal symbol payloads alike, so one name again means one extent
and both loops declare what they always declared. It renames only scopes that
share no node, because a node two roots reach cannot carry two names for one
axis, and it abandons the rename rather than emit a half-renamed graph if an op
payload still carries the old identity.
#665 was a declaration-consumer failure and did not close because an entry
guard passed; it closes because the kept name is declared from its source. A
name that resolves to no origin is a typed receipt from
`check_rendered_dim_origins`, which is an emission obligation rather than a
lowering one, and `CEmitter::declared_dim_names` is the executable invariant
that the two declaration loops cover the rendered set between them. A
synthesized Const's value supplies no shape: #1482 needs an actual axis source,
not a guessed dimension or a bypass of the cardinality check. Recheck the
current sigmoid/silu/gelu witnesses; the ReLU mechanism was removed by #1313.

Three rules bind the binding consumer, and each is here because its absence
was measured. A binding consumer skips a class whose extent an operation
computes only when `op_declared_dim_names` carries the name, because that is
exactly the set `bind_symbolic_dims` leaves unbound, and the two decisions
read one set rather than two lists that can drift. A required name that no
class speaks for takes its value from its resolved origin, which is how an
op-internal `Sym` carrier or a statically bound axis is reached at all; a
class that answered, by binding or by declining, is never overruled by that
fallback. A name the scope split finds in more than one class has no single
pre-eval extent, so when its scopes disagree it binds to nothing and every
axis carrying it is computed from actual values, and that tolerance stops at
the type: a live node reading the name BY VALUE, a `Reshape` target's
`RtDim::Sym`, still refuses. Which scopes disagree is a property of the
supplied values rather than of the graph, so the set is the caller's to name
and the tolerance covers exactly it; a multi-scope name whose scopes AGREE has
one extent and an omitted binding for it is refused like any other. The
declarations are not yet scoped the way these guards are: two scopes of one
binder lowered into ONE emitted function still share one declaration, which
#1788 records as residual and the per-scope rename in the claim transport
fixes. Deleting a declaration mechanism that holds a
loud-unsupported census site shrinks that site's baseline in the same change,
under `spec/design/loud_unsupported.md` B1, which owns that rule.

Record projection also needs an executable lowering route, and the route
B2b-2 built is a routing decision plus a host local rather than a runtime
record in the DAG. `body_form_the_dag_cannot_carry` already reported a `match`
on a runtime scrutinee; the `access` sibling was missing, so the two lanes
answered "is this def a kernel" differently -- C absorbed the failed kernel
lowering and fell through to host code (#1515) while eval propagated it. With
the decision shared, a tensor-typed projection of a runtime record inside a
host def body binds to a local before the tensor helper is attempted, and the
helper takes that local as its own tensor input. That IS materializing the
field's tensor as the actual shape input, and it is the prologue-local rewrite
#1266 reports downstream applying by hand.

One decision, asked once per session. The per-program facts behind the shared
decision, the program's definitions and call graph among them, are fields of a
`HostLoweringSession` that borrows the program they describe, so every caller
establishes one and none can read a fact derived from a different program. The
earlier arrangement keyed those facts on the program's address and gated them
on a thread-local flag, which three entry points grew into and the third never
armed (#1829); a flag can be forgotten at the next entry point, and a type
cannot.

What the type enforces and what it does not, stated at the granularity it
earns. It enforces that a session EXISTS wherever the memo is read: forgetting
one is a compile error, which is what the flag could never give. It does not
enforce that a caller HOLDS one for a program's lifetime, because that is a
statement about the extent of a value rather than about its type, and no Rust
visibility construct bounds it: the interpreter is a separate crate and
legitimately constructs a session, so the constructor cannot be narrowed.
Each session's extent is therefore its owner's, and the interpreter's is its
program's lifetime by construction of `EvalContext`. A future entry point that
built a session per ask would re-derive everything, and that cost is visible at
its call site but is not a compile error; #1921 tracks that residual. What the
representation changed is therefore precise: the session makes forgetting
impossible and makes misuse visible at the call site, where the thread-local
flag made both invisible (#1835). What follows from that, and is worth stating because
it is the reason a cheaper second predicate was rejected: nothing may answer
"is this def a kernel" except this decision. A syntactic surrogate for the
callee summary probe would be a second definition of one question, and the two
would drift. The probe may be asked LATER, and now is, last among the
host-lane predicates, because every cheaper predicate that answers first is a
callee lowering not run; asking it earlier changed no answer and cost a
lowering per non-tensor definition (#1835).

The binding is not unconditional, and both exits are loud rather than wrong. A
base name an inner binder also rebinds keeps ALL of its projections where they
are, because the walk asks whether the name is bound anywhere under the body
rather than reconstructing lexical scope; such a program needs the
prologue-local rewrite it needed before, and the C lane names the construct
under [04-TOT-2] instead of substituting the wrong tensor. A binder position
the walk cannot read a name from abandons the hoist for that def entirely, on
the same reasoning. The enumerable part of the claim is the oracle beside the
reader: it builds seven binder spellings from Surf source through the parser
and the desugarer, and asserts each yields its name. Those seven are a typed
parameter, an untyped parameter, two typed parameters, a typed parameter named
after a Deep tag (which the desugarer emits as `^{:type ..} name`), a `let`
binding, a `match` pattern binder and a pipe stage. Every fixture goes through
the parser rather than being constructed, because a reader checked against a
shape the parser never emits proves nothing about the shape it always emits. The necessity is the one #1266
names: both spelling variants must execute, and the direct spelling otherwise
leaves `expand` to a C host vocabulary that deliberately has no emission for
it ([04-TOT-2]). A record whose constructor is a compile-time fact keeps its
DAG route, which is what `is_static_constructor` now answers for a `(record
..)` literal as well as for an uppercase constructor application.

Only after these consumers and guards protect the admitted domain may B2b-2
remove `SizeClass`, `classify_expand_size`, `classify_arith_app`,
`sourceless_expand_size_error`, `Env::size_provenance`, and the lowerer's
provenance rejection sites. `shape_deps` stays: `root_reach` traverses it, so
removing it would silently merge scopes and reintroduce #1566, and it is a
declared WireDag v9 transport whose removal is a schema change with its own
numeric census obligations. It carries a third obligation since B2b-0b: a named
claim over an op-computed result axis records its declaring parameter there, so
an unread declaring parameter survives elimination and reaches the kernel. That
dependency is what makes the claim comparable at all, and #1372's removal must
migrate it rather than drop it. That removal is residual under #1372. Guards may land earlier;
acceptance may not widen earlier. A best-effort identity recognizer may
remain as a refinement whose miss yields a fresh guarded extent.

### C5 Acceptance and the preparation fixtures

The class completion command remains:

```sh
.venv/bin/python scripts/runtime_extent_oracle.py --phase final
```

Automatic success is exit zero ending `RUNTIME EXTENT ORACLE: PASS`, with
applicable HIP and Metal hardware receipts at the same head/corpus digest.
The command selects both registered slice phases, runs their
deduplicated targets once, and prints a digest per phase beside the combined
one. The nightly `runtime-extent-oracle` job runs exactly it, as one step, so
this class has one acceptance oracle rather than two phase steps whose
relationship a reader has to work out. B2b-3 retired the withdrawn phase `c`
from `SLICE_PHASES`, which is what the command was refusing on; naming `c`
still reaches the oracle's own
"corpus is not implemented" refusal. Retiring it added no passing phase and
erased no row: `--phase a` and `--phase b` keep their names, corpora and
row-transition checks, and each still has to PASS on its own.

The phase-B corpus contains 206 rows. Completion requires `--phase b` to
report `RUNTIME EXTENT ORACLE: PASS` without `--allow-shortfall`; enrollment
and a hand count do not establish that execution result. The JSON's `phase_b`
column contains 30 non-`executes_exactly` values against 176
`executes_exactly`; the dispositions below account for the thirty.
Twenty-nine rows are `rejects_exactly`, an exit state, since those programs
are SUPPOSED to be rejected and a row that stopped rejecting them would be the
defect. Nine of the twenty-nine predate B2c:
`expand.positional.replacement.non_unit_source_static`,
`shrink.to_end.nonzero_start`, and the seven `route.untied` rows
(`gather.gate`, `matmul.match`, `scatter_replace.gate`, `sum.copy`,
`sum.match`, `sum.record` and `trace.gate`), whose operand is still unresolved
where the shape-computed route runs, so the route returns a result nothing ties
to the shape it computes and any declared shape is admitted. Fourteen are
B2c's. One is `expand.sourceless_size.pipe_position`, chelis#1791's half B: a
size with no tensor source was accepted in pipe position and rejected written
directly, because the size rule matched the operand's type first and a pipe
stage's operand was unresolved. The other thirteen are
the two `concat.literal_claim.inlined_root` rows, which B2c moved off
`silent_unguarded`, plus five more lane pairs of the same class
(`pad.identity_axis.literal_claim.inlined_root`, `claim.literal.identity_root`,
`pad.literal_claim.inlined_root`, `claim.named.resolved.inlined_root` and
`claim.literal.nameless_activation`) and the single-row control
`claim.literal.kernel_entry`, which is one row rather than a pair because a
checker verdict no lane varies is one row and the program never reaches a
lane. That control is also the only `rejects_exactly` phase-B row whose baseline
EQUALS its exit state: it was already refused, correctly, before B2c, so it is this section's
"Invalid-program controls remain `rejects_exactly`" rather than a defect that
moved. Six more rejection rows cover `dtype.late_precision`:
`instantiation`, `binds_one_application_later`, `declared_bound`, and
`family_routes`, whose tensor operand's precision was unresolved when the
dtype policy first ran, plus `authored_contract` and
`indirect_and_transitive`, which require definition-time admission and
restriction transport instead of callee-body inspection. One row,
`shrink.elementwise_const.build`, is a registered `typed_unsupported(#1482)`,
an owned receipt rather than an unexplained gap.

The five evaluator gradient rows already have executable receipts from #2070.
This slice moves the two formerly deferred native primitive-scalar rows to
`executes_exactly`; all seven now execute their required forward failures and
agreeing-gradient controls. `PHASE_B_DEFERRED` is empty.

The `wrt` ARGUMENT KIND is an axis of this corpus, enumerated from the SPEC's
category rather than from what a review round happened to find. Three
consecutive rounds of chelis#1821 each found an unrecorded execution mode by
varying that kind, which is what a witness-by-witness list cannot stop:
`spec/04-type-system.md` lines 991-995 defines the category as a
"differentiable target" and admits a float tensor of any rank, a float PRIM
scalar, and an aggregate of those; only a non-differentiable `wrt` is a type
error. The axis is therefore **every `wrt` kind the spec admits, three of them,
crossed with single and multi target: six cells, fourteen rows, all
at an exit state.**

None of the three kinds folds into another, and that is measured rather than
asserted. A rank-0 `tensor[f32]` cotangent retains tensor identity while a float
prim cotangent uses the typed tensor-to-scalar boundary; the mixed-target
receipt checks their distinct consumers and exact values. Evaluator controls
cover tuples, records, one-field records, nested records and records with a
non-differentiable leaf. Compiled-C controls cover tuple/record groups and
mixed scalar/aggregate target ordering.

The evaluator now preserves activation obligations for all six cells. The
#1920 trace established the shared defect: both formal interface witnesses
were created, but transform lowering supplied an inferred result binder in
place of the authored result signature. With parameter binders `n` and `m`
and an inferred result binder such as `d43`, the declared equality never
formed before differentiation and dead-code elimination. Passing authored
signatures separately from inferred checked types repairs #1920, the five
aggregate layouts under #1924, and both primitive-scalar evaluator selections
under #1934. The existing witness preparation and #1912 forward-activation
dependency remain in place; no extra primal execution is added.

The undifferentiated and differentiated routes compare the same original
witnesses and report the same `load` Domain failure. Native primitive-scalar
admission uses the existing DAG cotangent reconstruction route: primitive
leaves cross the typed tensor-to-scalar boundary and tensor leaves retain
their rank, including rank zero. Complete result groups follow the written
target list, including repeated targets, while primal arguments are evaluated
once in parameter order. The six-cell eval/C matrix checks agreeing exact
gradients and rejected activations, including unused and zero cotangents.

chelis#1821's forward-activation dependency is recorded unconditionally, so a
gradient whose forward carries no obligation retains and emits it anyway: 891
to 931 emitted C lines for an obligation-free `grad`, measured against a
revert. chelis#1935 owns conditioning the edge.

The late-precision remediation (#1805/#1942) implements spec/04 §3.1.5's
authored generic contract rule through ordinary checking. Operation family
requirements constrain checked precision variables. The declaration check
compares the resulting requirements with the bounds the signature supplied,
including requirements obtained by calling another bounded function; it
rejects an absent or broader authored bound rather than publishing a
silently narrowed signature. Omitted signatures and parameter holes do not
authorize new inferred generic admission contracts either. The checker keeps
new parameter holes and call-operand/result requirements monomorphic while
the enclosing declaration is checked, then rejects an unresolved family requirement unless
an authored enclosing bound supplies it. Concrete local binding retains
[04-INF-1]'s inference rules; aliasing an already-checked function value is
contract transport, not a newly authored wrapper. Existing scheme restrictions
carry admitted families through instantiation, unification, generalization
and imported checked contexts.

This is the dtype implementation of the general policy in
[PR #2074](https://github.com/Chelis-Lang/chelis/pull/2074), not an implementation
of collection relations or the checker-wide protocol investigated by
[#2073](https://github.com/Chelis-Lang/chelis/issues/2073).

The `mod`/bitwise/shift acceptance controls in this slice use scalar operands;
[#2076](https://github.com/Chelis-Lang/chelis/issues/2076) owns the existing
bounded-tensor integer validator limitation. Checked function-value transport
also does not certify evaluator resolution through every aggregate
([#2077](https://github.com/Chelis-Lang/chelis/issues/2077)).

The call-site body walker is retired with this integration, not extended to
follow more syntax. The acceptance matrix covers each family route,
including [05-OP-39]'s specialized window-shape path and its ordinary
function-value alias path, standalone and inline signatures, wrappers,
higher-order values, local lambdas and aggregate projections, with
sufficient-bound controls and invalid concrete instantiations on both
ingresses. Window-shape failures retain diagnostic precedence over dtype
admission. The #1940/#1941
reproductions become definition errors; the corresponding bounded functions
must still execute with exact eval/C values. Existing lexical-shadowing
controls remain. The original empty-literal precision witness must also
receive a checker verdict, independently of authored generic definitions.
These are delivery obligations, not an execution receipt.

Phase B has no named deferrals after the two native primitive-scalar rows join
the five evaluator rows at their exit receipts. B2c removed the old
`concat.literal_claim.inlined_root.{c,eval}` deferrals: a claim the lowered
graph proves wrong is rejected before execution. A deferred row stays at its
measured start state with its reason; `exit_shortfall` skips it in every
phase, `final` included, so no phase reports it as a shortfall or fails for it.
`rows_at_exit` still enforces its receipt, so the test that pins the disposition
has to keep passing. `--phase final` refuses `--allow-shortfall`, which a
named deferral never needed.

The op-computed local guards moved seven rows
(`expand.shape_derived.declared_result_survives.{c,eval}`,
`class.load_op_output.eval`, `class.op_output_op_output.{c,eval}` and
`class.splice_f_of_n_n.{c,eval}`) and the six `shrink.*` preparation cells, and
#1379's acceptance moved `expand.arith_size.named_claim.{c,eval}`, which were
the last two.

The pipe fold reduces only direct first-argument forwarding stages into a
named call or its dedicated cast/copy/realize form. The fold consumes the
call-stage origin marker when a lambda becomes an ordinary callee. Other
lambda stages remain ordinary applications, preserving
lexical bindings and evaluation of the input before the body. The CLI oracle
receipts cover capture, sequential shadowing, conditional and deferred uses,
and trap order, with direct-call and agreeing-input controls.

Eight rows arrived with the pipe fold (chelis#1923 and chelis#1791).
`spec/02-surf-syntax.md` section 0.1 says `x |> f(y)` MEANS `f(x, y)`; every
consumer that met a `pipe` node reconstructed that application for itself, and
they did not all reconstruct it the same way. The checker typed a bare-name
stage from the callee's function type instead of as the application, which
lost every rule keyed on an application's arguments
(`pipe.bare_name_stage.to_tensor.{expand,sum}.{eval,c}`, rejected at check on a
program the direct spelling accepts) and dropped the section 4.7.2 size rule
on an operand that stayed unresolved
(`expand.sourceless_size.pipe_position`, which accepted a sourceless size
before chelis#1909 and rejected it with the wrong diagnostic after).
The lowerer bound the accumulator to a synthesized variable, so a callee's own
shape source stopped resolving and the lanes disagreed
(`pipe.bare_name_stage.expand_source.{eval,c}`, and
`pipe.bare_name_stage.lint_fix.c` for the same program as `chelis lint --fix`
writes it). `chelis_deep::pipe::fold_pipes` states the sentence once, over
every checker entry's input, and each of those passes lost its own pipe arm
rather than gaining a rule.

The two `staged.dynamic_to_tensor.vmap_column` rows are chelis#1779 and are
adjacent rather than the same defect. A runtime-shaped `to_tensor` lowers to a
deliberate rank-0 placeholder whose documented contract is to be refused so
the definition routes to the host lane; chelis#1693's staged host-source
partition runs ahead of the decision that reads that signal and cannot carry
the marker, so both lanes refused a program that checks at 1.0. The repair
reads the same fact the tensor-helper extractor reads. The partition's
exactly-one check is unchanged: it is what caught this.

What a passing `--phase final` claims is exactly this: every non-deferred row of the two
recorded corpora is at an exit state, every named receipt executed and passed
at one clean exact head, and both baselines match their generated corpora. It
claims nothing about the rows those corpora do not contain, and the class's
own remaining obligations are the list at the end of this section and the open
sub-issues of [#1277]. The HIP and Metal hardware halves are separate receipts
run by hand at the same head and digest; the oracle prints the HIP command it
expects rather than executing it.

Each deferred row carries its measured state in both columns, because
`validate_transition` permits only a move from a start state to an exit class:
the lattice is one-way, so a start-to-start move is refused by design and is
not expressible. `validate_deferral_keys` refuses a deferral key that names no
row, so a typo cannot silently defer nothing. Both lanes of every deferred
cell are pinned by a lock, and each issue's fix flips its lock, moves its rows
and removes its deferral.

`--phase a` reaches its row report and PASSes. Its `symbolic_window` target runs
`issue_368_runtime_symbolic_window_grad_is_half_everywhere`, whose `grad`
lowering failed backward-DAG verification until chelis#1775 landed in #1780.
`--phase a` has to PASS for `--phase final` to, which it now does; a red
there is again the signal that a row is short rather than a known outstanding
regression.

Phase A's `vmap.shared_shape_bound.concrete_c_emit` row left `PHASE_A_DEFERRED`
with chelis#1397's wildcard-root repair: the guard it was waiting on is that
repair, so the row exits at `EXECUTES` instead of repeating its start state.
`PHASE_A_DEFERRED` retains one entry, `expand.input_axis.metal_device`.

Phase A's wire capacity leg executes the Python binding facade through the
interpreter `PYO3_PYTHON` names, falling back to the checkout's `.venv`, and
not through the interpreter running the oracle. Install that facade's
dependencies into it with
`uv pip install --python <that interpreter> -r bindings/python/pyproject.toml`;
without them the leg reports a `ModuleNotFoundError` that reads like a census
defect.

Each phase's expected per-test receipts live in the reviewed manifest
`scripts/runtime_extent_oracle_targets.json`, which the oracle reads to build
its commands and which `crates/chelis-types/tests/runtime_extent_target_manifest.rs`
checks against the named sources inside `scripts/gate.py --fast`. After a
reviewed edit to a phase's generated corpus, regenerate that phase's checked
baseline with
`.venv/bin/python scripts/runtime_extent_oracle.py --phase <p> --write-baseline`;
never hand-edit the file or its digest. `--allow-shortfall` reports a
recorded row shortfall instead of failing on it, ending `RUNTIME EXTENT
ORACLE: RECEIPTS PASS, ROWS SHORT OF EXIT` rather than PASS; `--phase final`
refuses it outright. No registered phase records a shortfall today, and the
nightly job cannot be given the flag at all now that it runs `--phase final`.
The flag stays for the next phase to register a corpus, which starts with rows
short of exit, and because it is the only thing separating a landing row from a
drifted receipt; the oracle's self-tests hold its behaviour on synthetic short
rows.

The preparation suite is `crates/chelis-cli/tests/runtime_extent_claim_preparation.rs`.
Its acceptance runner asserts the decided contract over every cell:

```sh
cargo nextest run -p chelis-cli --test runtime_extent_claim_preparation
cargo test -p chelis-cli --test runtime_extent_claim_preparation \
  claimed_extent_contract -- --exact --nocapture
```

`claimed_extent_contract` reports every failed cell rather than stopping at
one, and must run before any fixture is claimed repaired. B2b-3 took its
`#[ignore]` off: it was there while the B2b repairs landed, beside a baseline
test that locked the measured gaps and whose own last assertion said to retire
it once no cell was unmet. That assertion fired on `main`, so the baseline test
and `fixtures/runtime_extent_claim_baseline.json` are gone and the contract is
an ordinary test. A clean checker, object-only output, or equal wrong answers
on Eval/C is never a passing completion receipt, and an `#[ignore]` back on
this test would be a claim withdrawn rather than a gate relaxed. Missing
compiler/toolchain prerequisites fail the suite rather than skip a lane.

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
baseline changed at #1619 only in the two #1619 C results and #1266's
record-alias C result: all three now execute with their expected values.

chelis#1397's wildcard-root repair moves two further cells of that 55, taking
the unmet count from 10 to 8. `wildcard.root` and `vmap.shape` both execute on
Eval and C with their contract values, `main = tensor(shape=[2], data=[2.0,
3.0])` and `main = tensor(shape=[2, 2], data=[2.0, 3.0, 5.0, 6.0])`. The
remaining eight were `polymorphic.named.root.mismatch.{eval,c}` (#1374) and
`polymorphic.foreign.root.mismatch.{eval,c}` (#1376), which #1782's deferred
root restatement then met, and `record.direct.{check,eval,c}` with
`record.alias.eval` (#1266), which #1266/#569 meet below. #1397's own
`shrink.*` declaration cells are met.

chelis#1266/#569 move the four record cells: `record.direct` on check, Eval
and C, and `record.alias` on Eval. All four now execute `[0.25, 0.25]`. The
runner reports `55 cases, 0 unmet contract cells` and passes: the whole
preparation matrix is met.

The sentence this replaces said four cells remained,
`polymorphic.named.root.mismatch.{eval,c}` (#1374) and
`polymorphic.foreign.root.mismatch.{eval,c}` (#1376). Those left with #1811's
deferral of a root's restated literal claim to the callee's named guard, which
landed between that sentence being written and its change merging. Measured
rather than counted by hand, and measured on both sides: reverting
`crates/chelis-ir/src` and `crates/chelis-types/src` to `ccd684643` and
rerunning gives `0 unmet` as well, so #1379's acceptance moves none of these
cells and the count was already zero before it.

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
baseline at that delivery had 51 unmet cells: 24 declared signatures were
preserved and two formerly silent #1377 inlined failures trapped. C2.4's
checked transport reduces the same preparation baseline to 35 unmet cells.
The preserved named `insert` declarations and executable roots do not prove
caller equality or attribution. #1374/#1376's caller equality landed with the
named claim and its inlined-root attribution with #1782; #1397's general
wildcard-root boundary and #1378's public value witness landed with B2b-root.
Four cells remained at that delivery, all #1266's; the acceptance runner,
`#[ignore]`d at the time, reported them, and #1266/#569 met them below.

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
| B2b-0b: remaining local guards | merged B2r/S2b and #1658's broadcast preparation repair | guard literal and resolved numeric claims from independent local size sources; op-computed local extents; exact positive/negative C/Eval rows. The op-computed admission and #1397's declaration half are delivered for `shrink` and for `pad` at the OUTERMOST activation (exported def, value binding, inlined root), and for a declared axis that passes an op-computed extent through. A helper whose NAMED result is consumed inside another def's body is guarded through its resolved binder; the spellings that bind the enclosing result to a rigid dim parameter are checker rejections under section 4.4.1. A claim the owner's own rule statically REFUTES is not stamped and is not executed either: B2c REJECTS it when the activation is lowered, with one fatal diagnostic both host lanes render byte-identically at exit 1, which is section 4.7's "A violation proven from literals is a type error" reaching the case the checker cannot see. The checker keeps that verdict wherever the extent IS visible to it, which a literal parameter extent makes it (`claim.literal.kernel_entry.checker`); what it cannot see is an extent that becomes literal only because a call supplied concrete arguments, and `tensor_concat_result_type`'s `Dim::Wildcard` under section 4.5.4 rule 3 is why `concat`'s DAG path is the sharpest instance. #526's `n + n` checker-tier repair is unchanged by this and remains the right fix for the type it would give. Non-unit stride delivery is separated into #1907/#1931; same-shape result ownership is #1948; local tensor ascriptions are #2110. None is evidence that B2b-0b already closed them. An out-of-domain span on the only claim-failing axis of a `shrink` is not reported as a claim failure: the local guard declines a span whose end runs past its operand, so both lanes report spec/05 §2.4.1's overshoot as the runtime's `Domain: shrink bounds outside input extent` line followed by [04-NUM-9]'s trap line, under a disagreeing literal claim, an agreeing one and a free dim alike, wherever the evaluator raises that diagnostic directly (#1797). Two pre-existing divergence classes remain outside that statement and are not closed by it: a SECOND axis whose in-domain span disagrees with its own claim is still reported as that claim on eval while C reports the overshoot, and a host transform such as `grad` prefixes its own wrapper to the eval text. A span that is empty as well as out of domain is still refused first by #616's operation-level admission rule and renders per lane under #1795 |
| B2b-1: claim transport | C2 contract and red fixtures; integrates B2b-0b | preserve the shipped helper-order and C2.4 checked-reshape/unit receipts (#1686/#1687); finish general scoped checked/lowered identities, explicit caller witnesses, multi-claim axes, rebuild/wire transport and migrated binding consumers; #1397's declaration-erasure half, with #1377's literal call/inlined-root exit established by the witness subset. Named result claims and the unread signature witness (#1374, #1376, #1566) are delivered, and a root's restated literal claim defers to them when a graph-fixed extent entails it, never when an ABI parameter's axis does, decided by `resolve_axis_extent`'s origin rather than by the neighbouring operation (#1782) |
| B2b-root: root execution | can start independently; acceptance composes B2b-1 | #1397's general wildcard-root boundary is closed: a nullary root whose result type carries a runtime extent is kept in the root manifest, so eval renders it and the C host emits an entry. On eval and C such a root is admitted and sized by the runtime rather than needing a sizing diagnosis, because the manifest print path sizes from the realized extent and never materializes a static buffer; guards and device capability diagnostics still apply, and an empty realized bound renders differently per lane under #1795. #1378's exact public value witness is unlocked and reverified. A root that keeps an unresolved dim variable is sized from the extent its callee's instantiation absorbed (#1801) |
| B2b-2: sources and acceptance | guards and claim transport for every newly admitted row | declaration sources are finished (#665/#1556/#1566); supply #1482's missing shape source. No provenance restriction remains: #1266/#569's field and pipe spellings and #1379's arithmetic sizes are all admitted, so what is left of this row's acceptance half is deleting the walk itself. `shape_deps` removal moves out of this row and is residual under #1372, which must now also migrate B2b-0b's declaring-parameter dependency rather than drop it |
| B2b-3: phase exit | preceding host repairs and per-row platform dispositions | DELIVERED. Every recorded phase-A and phase-B row has a registered receipt that executes and passes; `--phase b` demands PASS with no `--allow-shortfall`; the withdrawn phase `c` is out of `SLICE_PHASES`, so `--phase final` reaches its row report, passes on the host lanes, and is the nightly `runtime-extent-oracle` job's one step; and `claimed_extent_contract` runs as an ordinary test over all 55 cells with its preparation baseline retired. It moved no row, met no cell, and closes none of the issues listed below |
| #1907/#1931 stride closure | typed `Stride.strides[*]` carriers and C2.5's operation/result ordering | validate every runtime step before span computation; add `ComputedAxisExtent::StrideSpan` for positive non-unit steps; preserve independent operation-precondition and result-claim failures; replace the old disposition lock with exact positive/negative Eval/C compile-run receipts and final-corpus rows |
| #1948 same-shape result claims | C2.3's independent claim contract and spec/04 §4.7's returned-value producer rule | attach each declared-result obligation to the returned same-shape operation; represent every positive-rank agreement member without a selected operand origin; run operand agreement before the result guard; prove the dedicated check/Eval/compiled-C matrix and update the earlier #1798 attribution receipts |
| #2110 local tensor ascriptions | C2.3's independent claim contract and C2.5's introducing-site rule | retain authored local tensor annotations as explicit checked obligations; attach them at the initializer producer; preserve them through aliases, inlining, rebuilds and artifact boundaries; prove the independent static/runtime/agreeing matrix on check, Eval and compiled C |
| #1932 mapped-gradient artifact closure | C2.4's authored witnesses, batched node map and fail-loud artifact boundary | remap and retain the complete entry-witness/dimension-origin set through `vmap(grad(...))`, cotangent packing and splice; reject unresolved roots or rendered identifiers before success; execute the exact witness and controls on Eval and compiled C, including compile/link/run and mutation negatives |
| #1512 audit | no dependency on the B2b carrier or withdrawn C | enumerate reachable non-expand unresolved producers and consumer decisions; resolved/unresolved positive and negative pairs; distinguish error cascade suppression; assign each surviving defect a repair under #1512 |

B2b-0b and B2b-1 can be developed as separate changes, but their shared local
site integration must preserve both claim kinds. Claim transport is not an
out-of-scope caller problem: it is precisely B2b-1's closure requirement.
The #1377 exit now retains the declared type and an explicit call-entry
witness, with invocation dependencies preserving discarded checks. B2b-1
still owns the named scoped identities and unread named caller witnesses
needed by #1374/#1376/#1566. Kernel guard tests alone do not close those
public contracts.
B2b-root was separately bounded within #1397 so a declaration fix could not
silently close its broader root failure. Both halves have now landed as
separate changes, and #1378's public witness executes, so the bound has served
its purpose. Its typed Slice A mechanism was not reimplemented.

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

The residual work under [#1277] is its open sub-issues, which a green class
oracle does not touch. As of 2026-09-11, read with

```sh
gh api graphql -H "GraphQL-Features: sub_issues" -f query='{ repository(owner:"Chelis-Lang", name:"chelis") { issue(number:1277) { subIssues(first:100) { nodes { number state title } } } } }'
```

twenty-six are open: #578, #1397, #1482, #1522, #1535, #1559, #1574, #1575,
#1743, #1771, #1779, #1786, #1788, #1791, #1794, #1795 (with #1481), #1797,
#1798, #1800, #1801, #1802, #1805, #1814, #1815, #1821 and #1822. Re-read that
query rather than this sentence: the list is a measurement, and the tracker
moves. Several of them are rows the recorded corpora do not contain, which is
the difference between the oracle passing and the class closing.

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
