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
distinct authority. Witness prefixes with unsupported strict observations retain
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
A separate contextual-observation fixture retains fallback and its existing peak
bound without claiming linear allocation. Raw receipts belong to the author
report. No wall-clock ratio is claimed on the shared host.

The contextual-observation fallback fixture uses the identical harness against
main and the index, with each observed Const carrying its own Load dependency:

| Comparisons | Main/index allocated bytes | Main/index peak bytes |
| --- | --- | --- |
| 32 | 825,484 / 836,412 | 197,350 / 202,326 |
| 64 | 1,950,772 / 1,972,404 | 394,278 / 403,814 |
| 128 | 5,127,060 / 5,171,396 | 786,518 / 806,470 |

Index setup adds allocation and retained state in this unsupported family;
its contextual fallback remains a residual. The control temporarily substitutes
only the exact main schema source in the exclusive author tree, retains the same
allocation harness and all assertions, and restores the candidate source bytes
before focused validation. Neither measurement is a timing-speedup claim.
