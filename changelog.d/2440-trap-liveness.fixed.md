A discarded trapping node is no longer eliminated, so its trap occurs.
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
operation. Because a retained trap must execute, an input read only by a
discarded integer node is now a required input under root-scoped strict
evaluation, where its float twin still is not; that narrows the invariant
stated in [#991](https://github.com/Chelis-Lang/chelis/issues/991), whose
own reproduction is unaffected. Part of
[#2440](https://github.com/Chelis-Lang/chelis/issues/2440).
