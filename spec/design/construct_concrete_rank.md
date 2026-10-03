# Concrete-rank tensor construction

Owning issue: [#1416](https://github.com/Chelis-Lang/chelis/issues/1416).
Owner: Jeff Smith (`jeffreyksmithjr`). Normative authority: [05-OP-35] and its
stdlib manifest, subject to spec/04's decidable-fragment rules.

## Decision and release boundary

Keep the single public identities `stack`, `squeeze`, and `unsqueeze`. Instantiate
an ordinary concrete-rank tensor signature from the resolved callable identity,
its input shape, and a static i32 axis. Known literal List counts produce a literal
stack extent; unknown counts produce an ordinary wildcard at the known insertion
position, with declared output claims guarded against the runtime count. Mathematical shape schemas replace the
old invalid adjacent-spread and literal-anchor signatures. This records the
required public decision; it does not implement callable shape instantiation.

The current generic stdlib wrappers still reject ordinary concrete calls loudly.
They remain outside the executable release feature set and are recorded on the
public known-issues page. #1416 remains open as implementation work after the
reviewed decision clears its `freeze` and `launch:required` modifiers. Its P2
classification remains. No existing named-anchor, List-rank, dynamic-concat, or
runtime-insert rule is weakened, and no rank-specific aliases are introduced.

For supported concrete programs, use `reshape` with an explicit target dimension
list to remove or add a singleton; use `insert(x, axis, 1i64)` to add a singleton;
stack a known fixed collection by inserting singleton axes and using `concat`.
These alternatives do not claim a generic replacement for arbitrary List input.

## Implementation acceptance oracle

The future authoritative oracle is a named CLI suite,
`cargo test -p chelis-cli --test issue_1416_construct`, to be authored before
implementation. It must cover all three resolved exports, imports and aliases,
leading/interior/trailing and normalized negative positions, rank-zero insertion,
all active dtype families, and evaluator/generated-C parity with full values.
Negative cases: dynamic axes, out-of-range axes, independent wrong result shape,
non-singleton squeeze, empty stack, inconsistent shapes/dtypes and invalid
non-tensor input. Include direct and binding-carried literal counts, unknown List
parameter counts, and exact/incorrect declared output count claims. Runtime
shape/count/empty failures must trap before observation.
The stdlib executable corpus must exercise all three exports again.

This PR's decision oracle is the phase4B contract validator and its independent
mutation controls. The release-boundary evidence additionally executes the
original refused calls and the supported concrete primitive alternatives; it is
not evidence that the future implementation oracle is green.
