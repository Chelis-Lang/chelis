A discarded trapping node is no longer eliminated from the evaluator, so its
trap occurs under `chelis eval`.
`spec/06-transformations.md` §5.2 is prescriptive — mark every effectful
node, every potentially trapping node, every `Store` and every designated
output as live, and "purity alone does not make a possible trap dead" — but
dead-code elimination derived liveness purely from value reachability, so a
discarded integer overflow, division by zero, or out-of-domain cast simply
did not happen. Those now join `[05-OP-68]` guarded aborts as observable
roots, identified by one dtype-aware predicate: integer arithmetic and
division can trap, float arithmetic cannot, and §5.2's own example of a
removable dead float `add` still holds. Integer reductions and the
movement-op domain traps are not yet covered, so `[05-OP-68]` keeps a
non-normative note that its rule is not fully implemented for every trapping
operation. §5.2 now also states what it always meant by an observable root:
they are scoped to the activation being executed, so a node reachable only
from a root the evaluation did not select belongs to a declaration it is not
running. That is the rule [#2476](https://github.com/Chelis-Lang/chelis/issues/2476)
implemented, and it is what makes this seed safe to carry; without it the
trap seed would demand an uncalled declaration's parameters. The compiled lane is not fully
converged: a declaration that returns its own parameter still emits an
identity kernel with the discarded trap absent, so `chelis eval` traps where
the built binary exits 0. That shape is recorded on
[#2440](https://github.com/Chelis-Lang/chelis/issues/2440), which owns the
compiled-lane acceptance and which this change is part of.
