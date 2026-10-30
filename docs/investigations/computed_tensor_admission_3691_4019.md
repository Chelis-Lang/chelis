# Computed operands at the host/tensor boundary

Investigation for [#3691](https://github.com/Chelis-Lang/chelis/issues/3691)
and [#4019](https://github.com/Chelis-Lang/chelis/issues/4019), recorded on
2026-10-29 against `03c3698856f41d2f230ef3384160f1f76eb06a52`.

## Contract and bounded reproductions

`spec/04-type-system.md` §4.7.2–4.7.6 gives reshape targets ordinary i64
dataflow, preserves their checked list arity, and requires the same admission
across inline expressions, bindings and helper calls. Expand retains its
singleton-axis precondition. `[05-OP-11]` preserves mean's operand dtype and
rejects an empty reduced axis at execution when its extent is dynamic.

The range-expand witness is also pinned in Hydronnx at
[`ea4a78515d767bb714ccf9f957f785f8b8fb3eec`](https://github.com/Chelis-Lang/hydronnx/blob/ea4a78515d767bb714ccf9f957f785f8b8fb3eec/tests_blocked/range_expand.ch.in),
beside its
[literal control](https://github.com/Chelis-Lang/hydronnx/blob/ea4a78515d767bb714ccf9f957f785f8b8fb3eec/tests_blocked/range_expand_control.ch.in).
Its typed `[2, 3]` result contains `[0, 1, 2, 0, 1, 2]` and reduces to `6`.
The mean witness flattens a symbolic input using `[numel(x)]`, then reduces
`[1, 0, 0]` to one third. The f16/f32/f64 evaluator carriers are respectively
`3555`, `3eaaaaab`, and `3fd5555555555555`.

The baseline focused CLI target reaches unsupported host emission for inline
expand and for the symbolic reshape scalar mean at every tested float width.
Direct mean and fixed-size controls execute. This is an admission failure,
not evidence of an incorrect computed result.

## Representation

The host application extractor recognizes an operand whose root stays in the
host lane, but its root-only test misses host work below an otherwise numeric
operation. Dynamic List-to-tensor production below reshape reaches the helper
preflight barrier. A runtime `numel` in a reshape target has a host scalar
producer; ordinary tensor-only extraction cannot supply that value.

Operand preparation uses the existing scoped admission walk, with checked
parameter types and lexical/callee facts, to identify these boundaries before
helper extraction. It evaluates the selected operand once on the host and
passes its typed result to the numeric helper. Earlier working operands also
move when necessary to retain source evaluation order. The helper uses its
ordinary tensor/scalar inputs and runtime extent edges; no emitter arm or
precision conversion is added.

A literal shape list retains its Cons/Nil spine. Its runtime leaves cross the
boundary separately, preserving both checked rank and independently computed
i64 extents. Integer constants and named axes retain their static selector
evidence; only runtime computations need a new boundary. Fresh
binding names are reserved across recursive list preparation.

Physical tensor types come from the produced host value; a logical generic
argument signature cannot reduce their rank. Computed inputs carry their local
producer identity into numeric lowering. An expand precondition on such a
capture remains a LocalExpand witness, rather than becoming an authored Caller
claim merely because partitioning introduced a Load.

## Oracle and limits

`computed_tensor_c_admission` checks score `1`, empty diagnostics and no
untyped nodes before comparing actual generated C binaries with evaluator
observations. It covers inline/bound/literal range-expand, the 48-case
f16/f32/f64 direct/reshape, symbolic/fixed, scalar/tensor, guarded/unguarded
mean matrix, helper/tuple activations, empty domains, runtime product guards,
axis checks, effects, independent dimensions, precision and ownership errors.
Inline and bound linspace/insert operands also feed matmul and return `6.0`
through linked C binaries with the same evaluator observations.
`examples/computed_tensor_consumers.ch` supplies an executable corpus example.

This establishes the tested producer/consumer boundaries. It does not certify
every computed tensor family, close the broader computed-tensor admission
tracker, or establish a compiler resource bound. Test wall times on a contended
developer machine are not throughput measurements.

The focused execution commands are:

```sh
CARGO_BUILD_JOBS=1 cargo nextest run -p chelis-cli --test computed_tensor_c_admission --test issue_2939_host_tensor_calls --test issue_2373_computed_concat_c_parity --test key_pair_host_staging --test-threads 1
CARGO_BUILD_JOBS=1 cargo nextest run -p chelis-ir --test runtime_extent_slice_b_eval_guard --test runtime_extent_checked_transport --test-threads 1
```

The CLI checks pass all 15 new tests and 29 neighboring controls; the two IR
targets pass all 24 tests. The new regression target executes 84 generated C
programs, including failing domain controls and untaken branches for scalar
and tensor results, without skips. The example also passes canonical formatting, lint,
strict check, evaluator execution and linked C execution; both lanes produce
the six integer cells, total `6`, and f32 one third `0.33333334`.

## Checked numeric work at a prepared boundary

Red-team round 1 finds a regression on `f83c239b7081e505587bcf3f8a141c36fa5eb918`:
`[cast(cast(128i64, i8), i64), later(marker)]` must trap in the checked cast
before the effectful second leaf. The syntactic integer reader recognizes the
underlying `128`, so the all-operand exemption skips that computation while
moving `later`. The numeric reshape extractor then reads `128` from the
retained syntax. Native C prints `later` and traps in reshape; Eval traps in
the cast with empty stdout. This is lost numeric work and argument order,
under spec/02 and spec/04 [04-NUM-14] and §4.7.3.

Operand preparation classifies every cast as work. The builtin registration's
axis layout determines which argument retains compile-time selector syntax;
it also covers direct-only builtins through their unshadowed declarations.
If that selector's work moves before a later host operand, its host binding
executes the checks and the helper keeps its checked selector form. Numeric
shape leaves instead pass their actual evaluated values. This leaves the
reshape extractor and its opaque-target fallback unchanged.

The paired regression cases cover above-range i8/i16/i32 casts, a below-range
i8 cast inside a non-negative target expression, exact casts, and a cast
between two effectful leaves. The first effect executes once and a failing
cast suppresses the later effect. Direct and nested cast-wrapped static axes
remain accepted; runtime-axis and forbidden builtin-value forms remain
rejected. These tests extend the new target to 15 tests and 84 linked C
executions. The unrelated pre-existing cast-through-f16 shape-value defect is
outside this repair's claim.

## Corpus and counted-work integration

The settled-head hosted fast suite exposes five integration assertions that need migration.
The executable example needs an owning `parity.rs` input as well as its focused
admission target. Annotated concat/softmax also crosses the repaired boundary:
its computed concat operand is prepared before softmax helper extraction.
Its old C-rejection assertion and refusal-census exemption therefore migrate
to linked executable parity, while the invalid-axis case rejects on Eval and C.
The example enters the ordinary full guard census rather than being skipped.

The counted #1205 nested fixture at 20/40/80/160 operations reports
2059/3979/7819/15499 application-clone nodes and total work
10129/19409/37969/75089. Boundary preparation adds a bounded application
rewrite per selected producer; total application cloning is linear rather than
constant. The oracle measures the maximum nodes in one rewritten application
and requires that maximum to remain invariant with nesting, separately bounds
total clone growth by 2.2 for each doubled source size, and retains the original
2.2 total-work growth bound. Repeatedly copying a growing descendant subtree
fails the maximum-size invariant. No compiler semantic path changes in this
integration repair.

The flat fixture reports 1057/2017/3937/7777 cloned nodes and total work
7771/14731/28651/56491 at the same sizes. Every nested and flat maximum is
23 nodes, and all six owning host-work controls pass. A temporary negative
control adds an actual redundant clone of the original input application at
each rewrite and includes that clone in the work counter. The revised oracle
rejects it: maximum copied size grows from 842 to 1642 nodes at 20 to 40
operations, and aggregate cloning becomes quadratic. The mutation is removed
before candidate validation.

## Scoped admission work

A source audit of the counted oracle finds an unmeasured regression in
`670c558a47d7f07591567316aa4b23edefc38bbe`: operand preparation runs a complete
scoped classification for each working argument and materializes the whole
incoming type scope per application. The clone counts remain linear, so the
old counter sum does not expose either cost. This is the P1 finding class
"repeated operand classification and whole-scope materialization introduce
quadratic host-lowering work" in red-team round 2.

Exact-base attribution uses `3632c07d7b3a9e880c198538551af25e2c59bbc5` and the
same admitted nested/flat #1205 inputs on both heads. The source generator is
byte-identical, SHA-256
`45f7260a8af5f11a527e9e0de1c6d0a4ede2497e25f7c40f5858164dae176fef`.
At 20/40/80/160 operations, base nested admission visits are
724/1364/2644/5204; the candidate visits are
17881/61241/224761/859001. Base flat visits are 808/1528/2968/5848. The new
candidate flat scope materialization reads 1966/7086/26926/105006 entries;
that materialization is absent from the base.

The extended oracle counts scoped admission visits, incoming binding work
and captured fact nodes in addition to the previous work categories, and
bounds classification and scope growth separately. It fails before the
structural repair. Operand preparation then shares scoped facts through an
expression-owner guard, reads incoming types only for referenced names, and
keeps callee/callback contexts out of the caller's address cache. The guard is
thread-owned and drops its facts with that lowering. No numeric operation,
guard, runtime value, precision or source effect is replaced by a cached fact.

Initial repaired nested visits are 2302/4382/8542/16862 with 23 incoming
binding reads at every size. Flat visits are 1946/3706/7226/14266 and incoming
binding reads are 59/99/179/339. The maximum cloned application remains 23
nodes. These counts describe the measured categories and fixture families;
they do not establish an end-to-end compiler resource bound.

A negative mutation preserves the cache but adds the old complete classifier
and whole-scope construction at every working operand. The extended oracle
rejects that actual repeated work: nested admission visits reach
64259/230659/870659 at 40/80/160 operations, while the largest copied
application remains 23 nodes. The mutation is removed. The lifecycle units
also require independent root scopes, restored outer facts after a nested
guard, no facts after drop, and no caller-cache entries for named-callee or
anonymous-callback bodies. The native branch control shadows a scalar extent
with a tensor, checks the chosen producer, and retains both the product trap
and the untaken branch's lack of a trap.

## Persistent scopes and helper input transport

Counting actual `scope.clone()` entries exposes another cost in the same class.
At 20/40/80/160 operations, exact-base host-hoist prefix copies are 2/2/2/2
for both fixture families. The initial cache repair copies
332/652/1292/2572 nested entries and 928/3448/13288/52168 flat entries.
The growth oracle rejects the flat path without changing its 2.2 bound.

Host scopes therefore share persistent bindings and maintain dimension-owner
indices at binding updates. Signatureless helpers borrow the complete lookup
domain instead of collecting every binding and creating dead `Load` nodes at
each boundary. Runtime free inputs are materialized before branches and
closures; metadata-only extent observations are resolved by indexed canonical
source. Authored signature interfaces retain their original order and claims.

The oracle counts scope lookups, binding updates, constant-time scope clones,
ordered scope transport, actual helper input loads and free-input scan nodes.
Counting lookups also exposes repeated record-subterm scans beneath nested
operands. The same scoped owner folds those predicates once, independently of
callee and callback bodies. The extended measured sum remains a fixture-level
work oracle rather than an end-to-end compiler resource guarantee.

With that representation, nested lookups are 5949/11549/22749/45149 and input
loads are 324/644/1284/2564. Flat lookups are 3191/6031/11711/23071 and input
loads are 164/324/644/1284. Both families copy zero prefix entries and enumerate
zero full-scope transport entries. Total measured work is
23504/45264/88784/175824 nested and 15668/29668/57668/113668 flat.
The maximum application clone remains 23. Each new work category is bounded
separately, in addition to the aggregate growth control.

A shape-only local-ascription probe uses an otherwise unread `basis:
tensor[n, f32]` to constrain a reshaped wildcard input before mean. Equal and
unequal extents retain the same C capability refusal on the immutable e60
baseline and the repaired author branch; the unequal Eval case traps at the
named reshape claim. This pre-existing local-ascription admission boundary is
not promoted into the accepted matrix or claimed resolved by this repair.

Three additional real-work mutations reject after the representation repair.
Rebuilding the persistent scope's bindings reports 652 copied prefix entries
and violates the zero-copy invariant. Re-running the old record reader reports
12716 then 37836 nested scope lookups and fails the lookup growth bound.
An actual discarded full-input helper compilation adds flat transport counts
1048/3688/13768/53128 and input-load counts 1212/4012/14412/54412; the
transport growth check rejects it. The ordinary input scan and lowering remain
active in that mutation, so the failure is introduced repeated work rather
than lost positive coverage. All mutations are removed before final checks.

The indexed source lookup also preserves the ordered interface's wildcard
rule: empty names and `*` reserve dimension identities but declare no named
witness. The paired namespace/source unit fails before that exclusion and
passes after it; named-source ordering and persistent shadowing remain its
positive controls.

## Source-owned operand reservations

Standing round 2 finds an omitted category at local head
`46e04f30cd0f1e6dc631a6af823af32cc8c17c6f`. The added
`collect_deep_var_names` call before every operand preparation repeats whole
application scans. Actual nested visits at n=20/40/80/160 are
28234/107434/419434/1657834; flat visits are 1754/3274/6314/12394. The exact
3632 base has no attributable call. The admitted source generator remains
byte-identical, SHA256
`45f7260a8af5f11a527e9e0de1c6d0a4ede2497e25f7c40f5858164dae176fef`.
Counting this actual traversal in both the total and its own unchanged 2.2
category bound makes the previously green oracle fail. This is the same
repeated-preparation P1 class, not a fresh review round.

The reservation domain is part of the existing checked-program name index,
which is already source-owned and shared read-only by prepared sessions.
On first demand, an iterative borrowed traversal records all source names, including unread
binders and annotations. A preparation owns only a local numeric suffix
cursor; recursive list leaves share it, and checked-name/visible-scope point
lookups reject collisions. Numeric suffixes avoid rescanning a growing set
of previously minted underscore suffixes. No mutable allocator or new
expression-pointer cache survives the attempt. An unwind discards the local
cursor; cached source ownership remains immutable.

Reservation traversal, hash lookups/inserts, and copied/generated name bytes
are separate work categories and contribute to the aggregate. Repaired
nested source visits are 1037/1837/3437/6637, lookups 599/1119/2159/4239, and
name bytes 7589/14869/29429/58550. Flat source visits are
1137/2037/3837/7437, lookups 441/801/1521/2961, and name bytes
3529/6729/13129/26050. Totals including these categories are
33065/63745/125105/247826 nested and 20951/39571/76811/151412 flat.
These totals include additional lexical-ownership lookups and global-type
lookup steps (168/328/648/1288 nested, 88/168/328/648 flat). Existing
classification, clone and transport fields remain unchanged, including the
maximum application-copy size of 23.

Discriminating controls reserve unread source parameters, globals and local
binders, plus visible generated bindings. Independent programs start fresh;
prepared sessions and unwind/retry choose the same names. Native paired
programs exercise a legal colliding global used by later shape leaves,
effect order, repeated lowering and the product-guard failure twin.


## Numeric global availability in signatureless helpers

The new namespace controls expose a distinct input-availability regression in
local head `46e04f30`: a direct integer global in a literal reshape target is
not materialized for the geometry reader, although `add(global, 0i64)` is.
Both a legal `__host_tensor_arg_1_0` spelling and a noncolliding `global_rows`
spelling behave alike. The immutable e60 CLI builds and executes all four
sources; local 46 builds only the two wrapped controls and refuses the direct
forms with an anonymous-extent declaration diagnostic. The retained first published candidate f83 reproduces the same direct/wrapped
distinction. The reservation repair does not introduce this regression; the
earlier operand-preparation change does, so it remains in scope for the same
consolidated repair.

The signatureless input resolver now reads a global declaration through its
resolved identity and complete checked numeric type. The lexical scope of a
function contains its parameters, so that domain alone cannot supply globals.
A signatureless helper materializes an already initialized global
through its resolved top-level Load identity, preserving its checked physical
type without adding a lexical ABI input. Ordinary lexical bindings retain
precedence, and nonnumeric globals remain on their existing host path. This
repairs input availability; it changes neither the reshape fallback/result-claim
policy nor an emitter. The paired C controls check direct and wrapped reads,
colliding and noncolliding names, product failure, source effects, and repeated
independent lowering.

The actual redundant-rescan mutation calls the new borrowed collector before
each application preparation. It grows nested reservation visits from 70965
to 269205 at n=20 to 40 and the unchanged 2.2 category bound rejects it. The
mutation is removed before final validation. Declaration-owned global types
also reject unresolved precision without reading caller substitutions; lexical
payload and callable bindings shadow same-spelled globals. Global-initializer
controls require one IO effect and an earlier checked-cast trap that suppresses
all later effects. No type is fabricated for a nonnumeric global.


## Scalar host calls have no preparation consumer

The package-expansion witness
`overflow_cli_ingress::nested_surf_calls_build_without_a_stack_abort` builds
`f(len([previous, 2i64]))` nested 40 times. On `d70bcbca7`, ordinary scalar
calls enter tensor operand preparation even though neither tensor-helper
branch can consume its result. Preparation lowers host-owned arguments and
discards them; the ordinary host fallback then lowers the original arguments.
Nesting repeats that work. This is an introduced admission-cost regression:
the unchanged witness emits C on `786146e4d` in a bounded local probe, while
`d70bcbca7` exceeds a ten-second cap already at depth 10.

The spec-derived counter oracle records the actual host lowering and copied
application nodes before the repair:

| nesting depth | host expression visits | operand facts | application copies |
| --- | ---: | ---: | ---: |
| 2 | 63 | 98 | 133 |
| 4 | 1023 | 1100 | 2653 |
| 8 | 262143 | 237368 | 688093 |

Every row makes zero tensor-helper attempts. Preparation therefore belongs
behind the existing checked tensor-result candidate. Scalar and container
host calls keep their original eager argument-lowering route. The repaired
depths 2, 4, 8, 16, 32 and 40 visit 14, 24, 44, 84, 164 and 204 host
expressions and build no operand facts or copied applications. The counter
test checks the invariant at each
depth before advancing, so a reintroduced preparation fails on the small
case instead of running the deep exponential case.

Spec/03 §4.4 requires eager ordered List element work, and [05-OP-54] observes
the collection's size. An effectful element control prints each reached
marker exactly once; its division-by-zero twin traps before later elements.
The CLI regression compares complete Eval/native stdout and checks the exact
numeric trap identity. These bounded tests retain the unchanged #1205
resource thresholds and do not establish a whole-compiler complexity bound.
The unchanged depth-40 CLI test passes in 0.562 seconds; a separate bounded
C-emission probe takes 0.467 seconds, and Eval and the linked C executable
both return `a = 2`. The profiling unit uses the same 64 MiB thread-stack
setup as the #1205 profiles; the production CLI runs with its normal setup.

## Rejection-support ownership after admission

The native concat/softmax parity migration removes the final Rust module
consumer of `tests/common/unsupported_wording.rs`. Its sole snapshot row is
the superseded C softmax rejection, and the reviewed-wording generator owns
only that row. Configuration closure therefore exposes an orphaned Rust test
helper after the admission change.

Retire that helper, its snapshot, the exclusively owned generator/test, and
their regeneration and hook rows together. The remaining regeneration tests
retain their stale-leg, fail-fast, ordering and read-only checks using live
owners. Both tracked hook copies keep identical bytes. The general exact-wording
review rule remains; live unsupported diagnostics, their negative controls,
softmax native parity and invalid-axis rejection are unchanged.


## Physical values and checked helper frames

Package expansion on d70 identifies eight non-resource regressions: a matmul
guard reports insert instead, wildcard where/comparison/generic-grad calls
reach graph contradictions, two symbolic adjoints acquire concrete output
axes beside ToEnd bounds, a metadata-only insert binder is unavailable, and
the key-form fixture retains an obsolete host-RNG route assertion. Root's
qualified numeric expansion 41014854125 executes all nine failing identities,
including the separate List/len cost witness, successfully on a frozen tree
byte-equal to main `786146e4d01a177aa4a6631b494c749f067975ed`.

The formed-value type and the helper's parameter view have separate roles.
Physical HostBinding/value types retain actual rank, dtype and geometry.
The helper scope may widen axes through compatible checked parameter frames,
or through an argument's checked annotation when no compatible parameter frame
is available. Unknown physical axes never become claimed literals. Keeping this
checked view restores the runtime guard owners and symbolic adjoint frame;
no emitter default, new primitive or speculative whole-subtree retry is added.
The new axis-view work is counted in the existing 2.2 category/aggregate oracle.

A checked dimension read that has no runtime free-variable occurrence demands
its lexical tensor source through the existing dimension index. Missing sources
still reject, repeated reads reuse the Load, and a callee cannot consult its
caller's borrowed domain. The key-form fixture keeps its Eval/C/key-carrier and
reference-word checks; a host-produced computed template is an input of a DAG
draw, so its route assertion names that actual preparation boundary.

## Related composition regressions

The #4074 inserted-scalar scale/mean-addback cases and #4083 symbolic
reshape/extrema cases fail C admission on main `786146e4d` and execute through
the existing #4085 operand boundary. Independent probes establish this on both
the original d70 candidate and the repaired checked-frame candidate; these
issues add regression coverage without another production mechanism. The
`issue_4074_generic_insert` target covers dimension/precision specialization,
independent instantiations and shape/precision/extent rejections. The
`issue_4083_symbolic_extrema` target covers both extrema across f16/f32/f64,
fixed/symbolic rank-one/rank-two shapes, empty/untaken branches, shape-product
traps and effect order.

The static-empty checker requirement remains distinct: main and the original
candidate both accept direct and reshaped statically empty extrema despite
[05-OP-12]/[05-OP-13]. Runtime-empty negative controls stay in the regression
target. The finite admission corpus neither changes that normative rule nor
qualifies the downstream catalog or the broader lowering trackers.
