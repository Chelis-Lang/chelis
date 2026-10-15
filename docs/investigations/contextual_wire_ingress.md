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

Run with two build jobs:

```text
cargo nextest run -p chelis-compiler-api --lib -E 'test(public_contextual)|test(public_interleaved)|test(public_where_and_logical)|test(grouped_shape_discovery)|test(semantic_dimensions_keep)' --no-capture
cargo nextest run -p chelis-compiler-api --test contextual_ingress_performance --no-capture
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
prefix, axis expansions are 65/129/257. Adding intervening witnesses and
connectivity questions gives
240/992/4032 witness edges and connectivity steps. Other context-dependent graph forms can require another traversal for each
distinct authority, and witness equivalence is rebuilt per question.
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
