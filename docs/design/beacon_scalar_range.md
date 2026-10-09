# Scalar range properties through Beacon

`chelis prove --tier beacon-only` selects a fail-closed real-arithmetic lane.
The self-contained tensor route uses a rank-zero `tensor[f64]` carrier so its
operations enter the compiler graph lane. Its property parameters have finite,
closed intervals from explicit f64 literal `where` inequalities on
`tensor_to_scalar(x)`. Its body is
`tensor_to_scalar(output_expression) <= upper_f64_literal`.
The package scalar route admits named `f64` scalar parameters with direct
finite, closed `where` bounds and an `output_expression <= upper_f64_literal`
body. It extracts a proof-only rank-zero graph from the checked linked source
declarations, including imported pure f64 functions. The original scalar
functions keep their ordinary host classification. This scalar extractor
accepts finite typed f64 literals, variables, unary negation, addition,
subtraction, multiplication, division, and direct calls in a pure f64
closure; other expressions report unsupported.
Missing, duplicate or unrecognized constraints are rejected;
there is no SMT or fuzz fallback in this explicit lane.

For a standalone in-memory source, the source must be self-contained. The
tensor route also rejects shadowing of the `tensor_to_scalar` bridge. The
CLI's file path may resolve a Reef package first; imported rank-zero tensor
or pure f64 scalar functions then lower from the checked linked declarations.
The graph inlines those functions, so the property result records the selected
linked source declarations and a digest of the exact authored output
expression. These are provenance for the compiler's source-to-graph step.
Beacon's certificate addresses the graph hash and root.

The gallery build generates its network source before dispatch.
[The tensor example](../../examples/beacon_scalar_range.ch) is run
with `CHELIS_BEACON_BIN=/absolute/path/to/chelis-beacon chelis prove
examples/beacon_scalar_range.ch --tier beacon-only --beacon-budget 2000 --json`.
[The scalar example](../../examples/beacon_scalar_host.ch) exercises the
proof-only f64 host extraction with the same explicit tier.

The shared property runner lowers the expression's reachable source closure.
`GoalShape::ScalarUpperBound` carries the named input box and a tagged f64
`ScalarValue` threshold. Its `IntervalBox` endpoints and the separate
`BoxRange` output endpoints also use tagged `ScalarValue`. A one-sided bound
does not acquire a fictitious finite lower limit.
The exact WireDag bytes are addressed by SHA256 and a node ID. The shim verifies
that binding, names and scalar types before adding the folded expression
`output - threshold`. Its request keeps both output and folded roots, and the
unsafe clause references only the folded root.

Beacon's directed-rounded relaxation must exclude the folded nonnegative
region to certify. This is a sufficient, deliberately strict check for a
non-strict upper bound; equality can remain unknown. The returned proof has
`RealArith` and renders `proven_modulo_real_arithmetic`. This covers real
arithmetic on stored weights, not float32 or float64 execution.

Unknown preserves reason, accumulated hull and the whole split tree. A
confirmed witness strictly above the upper bound maps to refuted with its
real-arithmetic qualification. A witness exactly on the boundary cannot refute
the non-strict property.
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
