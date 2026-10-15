# Contextual WireDag ingress work

Wire shape questions are discovered after positional-domain validation, grouped by
exact shape authority, resolved with one temporary memo per group, and consumed
by the original ordered contract validator. Discovery cannot admit a graph or
report a later contract error ahead of an earlier shape error. It stops at the
first non-shape error, whose disposition belongs to ordered replay.

Axis-origin authority projects relevant-source membership onto the dependencies
read by Const and ConstTensor. Semantic-dimension authority instead projects onto
shape dependencies of **all** node kinds. Both preserve dependency-list order;
projection changes membership representation, never which successful alternative
wins. The witness epoch derives from the original unfiltered maximum source ID.
Witness observations retain their separate input-agreement mode, and equality
requiring a witness still requires an edge even when origins are identical.

Anonymous identical-input Add axes and identical-branch Where axes have precomputed representatives with exact
edge distances. Fixed and named output dimensions stop aliasing; validation and
semantic-dimension questions still refer to the original nodes. Where dimensions
can jump only when every shape dependency names the same identical branch, so
no competing ordered alternative is erased. An ordered
first-position index intersects relevant sources with each node's dependencies,
preserving failed alternatives and dependency order without a full list scan.

The semantic-dimension resolver uses an iterative ordered dependency search with
memoized successes and failures. Its fuel key is bounded by the earlier node ID:
every dependency decreases that ID. The authority is checked before reuse.

## Reproducible bounded witnesses

Run native checks serially with `CARGO_BUILD_JOBS=1`,
`NEXTEST_TEST_THREADS=1`, and `MAKEFLAGS=-j1`:

```text
cargo nextest run -p chelis-compiler-api --lib -E 'test(public_contextual)|test(public_interleaved)|test(public_where_and_logical)|test(grouped_shape_discovery)|test(semantic_dimensions_keep)' --no-capture
cargo nextest run -p chelis-compiler-api --test contextual_ingress_performance --no-capture
cargo nextest run -p chelis-compiler-api --lib -E 'test(witness_)|test(historical_witness)|test(large_historical)|test(disconnected_strict)' --test-threads 1 --no-capture
```

The public validated-JSON witnesses cover a repeated Const/Add prefix, alternating
A/B source choices, genuinely changing source choices, intervening witnesses with
actual connectivity, Where and Logical prefixes, warmed-context rejection,
malformed references and first-diagnostic ordering. The allocation owner measures
allocated bytes, allocation calls and peak live-heap delta during public ingress;
it bounds retained memory for all cases and allocation for repeated/alternating
contexts. The unit owner reports axis queries/expansions, semantic visits and memo
probes/entries, positional lookups and fallback scans, dependency and membership
steps, witness scans/edges/connectivity, authority construction, shape questions
and answers, group count, shape-question/authority key comparisons, source-index
probes, alias-plan steps and jumps, validation-node visits and connectivity seen
membership. These counters describe shape resolution and ordered replay, not
every instruction in JSON parsing or domain validation.

For 16/32/64 alternating A/B comparisons over equally deep shared prefixes,
axis expansions are 37/69/133 with exactly two groups. For 32/64/128 Where
prefixes, semantic visits are 259/515/1027; with dependencies naming the identical
branch and 32/64/128 authorities, they are 414/830/1662. Logical prefixes use
195/387/771.
Dictionary key comparisons grow with ordered-map operations rather than forming
a strict linear bound on every machine instruction.

## Residual class

This mechanism bounds retained contextual memo memory and shares work across
interleaved equivalent authorities. It does **not** prove universal whole-ingress
linear work. With 16/32/64 genuinely distinct choices over the anonymous identical-input Add
prefix, axis expansions are 65/129/257. Other context-dependent graph forms can require another traversal for each
distinct authority. Witness prefixes with opaque contextual observations retain
the contextual edge reconstruction and connectivity fallback described below.
Issue #3231 remains open for that context-count dependence.

## Matched source control

A debug control uses the same benchmark harness against main `2e56e9026` and the
grouped resolver `4e5d11d33`. Five samples alternate parent/candidate execution
order for each fixture class. CPU is median user-plus-system time for each direct
test process, including lengths 32/64/128, fixture construction, rejection checks
and startup. It is not isolated resolver CPU or a universal speedup.
The machine also runs independent workloads, so CPU and elapsed timings are
shared-host observations. Deterministic work counters and allocator bytes
provide the structural evidence.
Each sample executes one allocation case in a separate test process. The
committed owner also serializes its process-wide counters when Cargo runs cases
together.

| Fixture | Parent/grouped CPU (ms) | Parent/grouped allocated bytes at 128 | Parent/grouped peak bytes at 128 |
| --- | --- | --- | --- |
| Repeated authority | 131.2 / 46.8 | 6,850,052 / 850,818 | 255,535 / 362,243 |
| A/B alternation | 135.1 / 43.4 | 6,850,789 / 851,783 | 255,701 / 345,939 |
| Distinct choices | 152.6 / 68.3 | 6,956,999 / 1,184,387 | 288,827 / 377,925 |
| Active witnesses | 247.2 / 152.8 | 12,858,200 / 8,943,908 | 520,940 / 712,702 |

Retained peak heap remains linear in the tested classes but increases against
main. Grouped questions, answers and immutable plans trade retained storage for
reuse; this is not a peak-memory reduction claim against main. The
all-context-retention prototype instead fails the committed peak bound.

The benchmark temporarily omits only allocation bound assertions in both
control harnesses; rejection and exact-diagnostic checks remain active. The
committed owner retains every bound. Both schema and harness bytes are restored
before normal validation. Shape-first, binding-first and domain-first controls
produce identical full diagnostics against both implementations.

## Strict-observation witness prefixes

A lazy index is shared across ingress authority groups. It examines original
witness IDs monotonically up to the greatest requested **unfiltered** cutoff.
Only the original claimed input slots and earlier requirement witnesses can
supply edges. Each endpoint is observed at the original tensor, literal axis,
full graph fuel and false input-agreement mode, with strict resolution and empty
relevant sources. Strict resolution can still require inner input agreement
through runtime metadata; an outer false mode alone does not prove independence.

The index owns a separate strict memo of successes **and failures**. Every origin
edge traverses an earlier node and consumes at most its exact edge distance,
including precomputed alias hops. Starting at graph-length fuel therefore leaves
at least `node.id + 1` at every reached state in a validated DAG. Only states with
that sufficient budget enter this memo. Limited probes and production memo
entries never populate it, and index entries never populate production caches.
Shared unsupported prefixes consequently share their internal failed states,
not merely their outer witness observations.

Exact semantic origins are interned. Union by size assigns an attached root one
parent and the original declaring witness ID; there is no path compression.
Parent stamps increase towards final roots. Following only links stamped before
a requested cutoff reconstructs that historical partition, including descending
cutoff requests after a later prefix has been indexed. Component doubling bounds
parent depth logarithmically. Earliest incident-edge IDs preserve self edges and
out-and-back paths: identical origins requiring a witness still need an actual
prior incident edge.

The first eligible edge lacking independently resolved endpoints marks an
incomplete prefix. Before that declaration, the indexed graph is exactly the
original eligible graph, so disconnected and same-origin-without-edge queries
can return false directly. Later incomplete prefixes prove only indexed positive
paths; all other questions use the unchanged contextual algorithm. This changes
neither domain validation nor ordered diagnostic replay.

With W examined witnesses, E claimed edge slots, V interned origins, R distinct
sufficient strict states, Q equality queries, and T actual strict-resolution
work, setup accounts for W/E/T and ordered-map comparisons; union and historical
lookup work is O((E + Q) log V). Retained index storage is O(W + V + R).
This does not make T automatically linear in graph nodes across arbitrary rank,
metadata or state fanout. Incomplete contextual prefixes can still pay the
original fallback cost after index setup. Issue #3231 remains open.

The differential owner retains the original algorithm and exercises all small
prefix cutoffs, descending requests, both witness requirements, empty sources,
root changes, cycles, duplicates, disconnected origins, self edges, strict mode,
exact origin kinds, contextual Const alternatives, runtime-metadata agreement,
limited fuel and malformed slots. Public JSON controls compare complete errors
with the original resolver, including domain-first malformed axes/claims and
shape-first versus binding-first contract errors. A 10,000-origin historical
index runs on a 512 KiB worker stack; existing deep-ingress controls remain.

The public changing-authority Load-observation witness at 16/32/64 comparisons
constructs 15/31/63 edges once, takes 120/248/504 historical parent steps, resolves
16/32/64 strict states and performs 865/2329/6061 new-map key comparisons.
Observation entries and interned origins each grow as 16/32/64. It uses
no contextual fallback. The exact `b0f01b650` control reconstructs 240/992/4032
edges and connectivity steps. Internal unsupported-prefix setup over 129
witnesses and 128 claimed edges resolves 255 states: one success and 254 failures.
A disconnected strict-only component at 64 witnesses answers 128 rejecting
queries with 63 constructed edges, 64 strict states, 126 parent steps and no
contextual fallback.

The same public allocation harness compares exact `b0f01b650` with the index:

| Comparisons | Main/index allocated bytes | Main/index allocation calls | Main/index peak bytes |
| --- | --- | --- | --- |
| 32 | 988,780 / 605,228 | 9,136 / 4,687 | 179,014 / 182,246 |
| 64 | 2,810,788 / 1,213,764 | 26,720 / 9,359 | 357,590 / 363,214 |
| 128 | 8,943,908 / 2,431,428 | 86,717 / 18,696 | 712,702 / 723,734 |

The retained index raises peak heap in this bounded family. The allocation owner
keeps its existing peak bound and extends its allocation bound to this strict
witness family; both new work/allocation assertions reject the main control.
The strict-only contextual-observation control retains fallback and its existing
peak bound without claiming linear allocation. Raw receipts belong to the author
report. No wall-clock ratio is claimed on the shared host.

The contextual-observation fallback fixture uses the identical harness against
main and the strict-only index, with each observed Const carrying its own Load dependency:

| Comparisons | Main/index allocated bytes | Main/index peak bytes |
| --- | --- | --- |
| 32 | 825,484 / 836,412 | 197,350 / 202,326 |
| 64 | 1,950,772 / 1,972,404 | 394,278 / 403,814 |
| 128 | 5,127,060 / 5,171,396 | 786,518 / 806,470 |

Strict-only index setup adds allocation and retained state in this family.
Conditional plans below address its one-layer source choices; opaque contextual
observations retain fallback. The control temporarily substitutes
only the exact main schema source in the exclusive author tree, retains the same
allocation harness and all assertions, and restores the candidate source bytes
before focused validation. Neither measurement is a timing-speedup claim.

## Conditional observation plans

Conditional witness endpoints use shared plans for one ordered Const choice layer
whose candidates resolve independently under full-fuel strict observation.
Existing exact-hop aliases transfer these plans without changing fixed dimensions,
input-agreement mode or candidate order. Other observations remain opaque and
retain the original contextual fallback.

Eligible original claim slots map to canonical unordered plan pairs stamped with
the earliest declaring witness. Source postings select active plans in an exact
shape-authority group. Direct queries combine those conditional edges with the
historical strict partition in an ephemeral overlay at the original unfiltered
cutoff. Conditional pairs are enumerated by the smaller of the active pair budget and eligible active
incidence count; this avoids both rescanning an always-active anchor's degree and
quadratic pair enumeration for dense active plans with sparse edges. Self edges
remain actual evidence. Negative answers require a complete eligible prefix;
opaque observations preserve fallback, including metadata-dependent agreement.

Construction, ordered candidates, source postings, map comparisons, pair probes,
incidence visits, strict-component walks and overlay queries are counted separately.
Retained plans and edge indexes are shared; only the current authority's overlay
is retained. This is a bounded conditional-observation mechanism, not a universal
whole-ingress linearity or peak-memory reduction claim. Metadata proof reentry
remains an uninvestigated question outside this mechanism.

With W witness observations, E original claim slots, P compiled observation
states, D explicit ordered candidate records, R sufficient strict states and V
strict origins, retained setup storage is O(W + E + P + D + R + V). Strict
resolution work T and map comparisons remain separately counted. For each direct
query authority group, C relevant keys select S source postings and K active conditional plans. Overlay construction includes S + K, prefix-count searches, the smaller of
K(K+1)/2 and eligible conditional incidence count, active conditional-to-strict
incidences, historical strict-root walks and equality queries, with ordered-map
comparison factors. Each equality query also projects, sorts and compares its
relevant-source key: charge its full key length and sorting comparisons for every
query, even when the overlay is reused. Only one overlay survives per group. Dense postings, many
conditional-to-strict edges and opaque fallback can still require repeated work;
this bound is not automatically linear in graph nodes.

The conditional oracle compares every small cutoff in both directions, ordered
alternatives, opaque candidates, fixed dimensions, mixed source precision, self
edges and duplicate claims with the independent original helper. Public JSON
controls cover changing and alternating authorities, unrelated-origin rejection,
malformed references and an earlier shape error followed by a later non-shape
error. A dense-active/sparse-edge control forces incidence enumeration; a long
shared-choice control requires aliases to share candidate and posting storage.

The exact `43cd1becdfb` control and conditional plans use the same public fixture
builder. The control adds only a test-only work reporter to the parent schema.
For 16/32/64 distinct singleton source choices, the control scans
480/1,984/8,064 witness nodes and expands 279/815/2,655 axis states. Conditional
plans scan no fallback witnesses and expand 128/256/512 axis states. At 64 they
compile 64 plans, visit 128 candidate records, select 504 source/query postings,
probe 186 pairs and two incidences, and construct 63 overlay edges. Strict setup
accounts separately for 64 successful and 64 failed states. Semantic visits remain
1,016. Ordered-map comparisons can increase: at 64 the new maps make 22,424
comparisons versus 2,900 in the strict-only control; shape/authority comparisons
are 24,532 versus 24,343. This is a structural witness-scan claim, not a CPU ratio.
The alternating control scans 256/1,024/4,096 witness nodes; conditional plans
retain zero fallback and bounded pair/incidence work in that family too.

| Comparisons | Strict-only/conditional allocated bytes | Strict-only/conditional allocation calls | Strict-only/conditional peak bytes |
| --- | --- | --- | --- |
| 32 | 836,412 / 725,332 | 7,704 / 5,424 | 202,326 / 220,566 |
| 64 | 1,972,404 / 1,456,148 | 19,672 / 10,842 | 403,814 / 440,662 |
| 128 | 5,171,396 / 2,918,924 | 56,412 / 21,673 | 806,470 / 879,606 |

The existing allocation bound now covers conditional observations and rejects
the exact parent at 64. For the full 128 control receipt only its parent guard
disposition is used; fixture construction, peak bounds and malformed rejection
remain unchanged. Both source files are restored byte-for-byte before candidate
validation. Retained peak increases; no peak-reduction or shared-host timing claim
is made. The dense-active/sparse-edge control takes 30/62/126 incidences and no
pair probes at 16/32/64 active plans. A 4,096-plan chain validates and rejects a
missing source on a 512 KiB worker stack.

The public `public_conditional_strict_incidence_residual_is_counted` control
observes one Const depending on Load0 through successive witnesses linked to
strict Load anchors, followed by Compare(Load0, Load_j). Its 16/32/64 cases encode
63/127/255 nodes. Stable-membership replay visits 15/31/63 conditional-to-strict
incidences and genuine edges, with zero fallback; the unchanged baseline visits
120/496/2,016. Clearing the Const source dependency retains the original exact
shape rejection. Selection, event preparation/replay and complete counted work
have separate assertions; this is a bounded family oracle, not ingress closure.

The remaining opaque/contextual fallback can also cause polynomial repeated
prefix work. No remaining exponential expanded-path acceptance witness is
established by these probes. Arbitrary nested metadata proof reentry and universal
whole-ingress bounds remain unknown; this mechanism makes no termination or
acceptance claim for that separate question. Issue #3231 remains open.

## Stable-membership ordered prefix replay

Supported one-layer conditional choices have a constant ordered assignment when
all original relevant-source vectors project to the same semantic membership.
Discovery groups these questions for scheduling without their epoch, orders them
by original unfiltered cutoff, and retains the original epoch in every semantic
and axis answer cache. Ordered validation still replays the original questions.
Only one bucket's sparse selected assignment and connectivity state is retained.

Connectivity uses stable raw origins. Timestamped successful strict unions and
selected conditional edges replay once in declaring order; every edge remains
strictly earlier than the original cutoff. Historical component roots are never
persistent vertex keys. Genuine endpoint incidence, including self and redundant
edges, is separate from synthetic strict forest links. An opaque eligible edge
or selected opaque observation prevents a negative proof and preserves fallback.
Unexpected membership, missing witness authority, cold and descending queries
retain the existing resolver. No semantic answer crosses an epoch boundary.

The stable-membership conditional-to-strict fixture has one active choice and
n-1 edges. Its incidence visits are n-1 instead of repeated triangular prefixes;
ordered maps and event/question ordering bound this family by O(n log n) work
and O(n) storage. Count source projection, key comparisons, postings, sparse
selection, strict preparation, event ordering, DSU walks, queries and fallback
separately. With G distinct memberships and V strict origins, strict preparation
requires O(G V) origin/forest visits plus ordered-map costs. Dense postings,
distinct assignments, opaque fallback and large query keys remain input-sensitive. Equal assignments across different membership
vectors are not interned. Allocation totals and peak heap require matched
measurements and need not improve together. Metadata proof reentry remains
uninvestigated. This mechanism does not establish whole-ingress linearity.

The matched stable-membership allocation owner at 32/64/128 uses the same public
JSON and measurement window for both derivations. Total allocated bytes change
476,009/1,061,929/2,560,745 to 395,673/803,681/1,610,457; peak heap increases
114,733/226,405/448,701 to 118,645/235,373/466,477. Other allocation owners retain
their bounds but pay small grouping bookkeeping increases. These measurements
support neither a peak reduction nor a shared-host CPU ratio. Single-epoch buckets
retain the existing overlay because they have no repeated prefix to amortize.
