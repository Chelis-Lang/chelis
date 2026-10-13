# Scalar range properties through Beacon

`chelis prove --tier beacon-only` selects a fail-closed real-arithmetic lane.
The self-contained tensor route uses a rank-zero `tensor[f64]` carrier so its
operations enter the compiler graph lane. Its property parameters have finite,
closed intervals from explicit f64 literal `where` inequalities on
`tensor_to_scalar(x)`. Its body compares `tensor_to_scalar(output_expression)`
with one or two finite f64 bounds.
The package scalar route admits named `f32` or `f64` scalar parameters with
direct finite, closed `where` bounds. Its body states a lower bound, an upper
bound, or both comparisons joined by `&&` over the same output expression.
It extracts a proof-only rank-zero graph from the checked linked source
declarations, including imported pure scalar functions. The original scalar
functions keep their ordinary host classification. This scalar extractor
checks the linked declaration set before a property can receive a Beacon
result, including when the shared runner is called directly or through Tide. It
accepts finite typed scalar literals, variables, unary negation, arithmetic,
scalar local bindings, literal casts to the source dtype, finite scalar
conditionals, selected numeric builtins, and direct calls in a pure same-dtype
closure. `standard_normal_cdf` uses the shared §3.3 activation graph; unsupported
expressions stay unsupported. Typed `f32` values are lifted
exactly to the f64 proof graph; operations in that graph have real-arithmetic
semantics rather than f32 execution semantics.
Missing, duplicate or unrecognized constraints are rejected;
there is no SMT or fuzz fallback in this explicit lane.

For a standalone in-memory source, the source must be self-contained. The
tensor route also rejects shadowing of the `tensor_to_scalar` bridge. The
CLI's file path may resolve a Reef package first; imported rank-zero tensor
or pure scalar functions then lower from the checked linked declarations.
The graph inlines those functions, so the property result records the selected
linked source declarations and a digest of the exact authored output
expression. These are provenance for the compiler's source-to-graph step.
Beacon's certificate addresses the graph hash and root.

The gallery build generates its network source before dispatch.
[The tensor example](../../examples/beacon_scalar_range.ch) is run
with `CHELIS_BEACON_BIN=/absolute/path/to/chelis-beacon chelis prove
examples/beacon_scalar_range.ch --tier beacon-only --beacon-budget 2000 --json`.
[The scalar example](../../examples/beacon_scalar_host.ch) exercises f64 and
f32 two-sided host properties with the same explicit tier.

The shared property runner lowers the expression's reachable source closure.
`GoalShape::ScalarUpperBound` carries the named input box and a tagged f64
`ScalarValue` threshold. An f32 source endpoint or threshold is represented by
its exact stored value in that tagged proof carrier. Its `IntervalBox` endpoints
and the separate `BoxRange` output endpoints also use tagged `ScalarValue`.
A one-sided bound does not acquire a fictitious finite lower limit.
The exact WireDag bytes are addressed by SHA256 and a selected node ID. The shim verifies
that binding, names and scalar types before adding the folded expression
`output - upper` or `lower - output`. Its request keeps both the original output
and folded roots. The linear request's unsafe clause references only the folded
root; the certified Arb request selects that root explicitly. A
two-sided property dispatches both obligations and passes only when both prove;
evidence records each side separately. `selected_root` names the goal's selected
node, `folded_root` names the appended comparison node, and authored property
evidence also names the original `source_output_root`. For a lower bound, the
selected node is an appended negation of that source output. A confirmed violation refutes the range,
while an unknown or unsupported side cannot be masked by the other side.

The arithmetic/ReLU cone uses Beacon's directed-rounded linear relaxation.
Graphs with other admitted operations use Beacon's certified Arb box cover.
Both routes require a strict negative upper bound on the folded root to certify
the authored non-strict bound; equality can remain unknown. The returned proof has
`RealArith` and renders `proven_modulo_real_arithmetic`. This covers real
arithmetic on stored weights, not float32 or float64 execution.

Unknown preserves the engine's reason and available search evidence. A
confirmed Arb singleton witness strictly beyond a stated bound maps to refuted
with its real-arithmetic qualification. A witness exactly on the boundary
cannot refute the non-strict property. The certified route checks the response's
DAG hash, selected folded root, strict limit, finite bound, complete nonempty
leaf certificate, and in-domain singleton counterexamples before changing a
property status.
Unrecognized reports, unconfirmed witnesses, mismatched source bindings and
abnormal proof exits fail closed. Large output captures must not block on a
full pipe. The outer process deadline leaves five seconds beyond the requested
Beacon budget for output and cleanup. `--beacon-wall-budget` (Tide:
`beacon_wall_budget`) optionally includes compiler preparation, up to one day.
After lowering, the search budget is reduced to the remaining wall time minus
five seconds. An expired deadline requests zero search time and preserves the
engine's unknown. Compiler preparation itself is not interruptible by this
cooperative deadline. An unknown before the first search evaluation has no hull;
its tree and reason still travel through the result.

The CLI and notebook use the same property runner and expose the actual folded
goal, engine evidence and semantic qualifier. Deterministic request identity
excludes elapsed time. Evidence includes exact graph and request hashes, the
engine binary hash, input bounds, typed threshold and effective search settings.
Consumers combine the engine hash and request hash for replay identity. Run
artifact identity is separate.

Acceptance: typed positive/negative goal tests; real current-main shim tests;
unknown, witness, qualifier, hull and large-tree transport tests; a generated
gallery network through both the CLI and a deployed downstream consumer. Source-only and mock-only
success do not certify the deployed path.
