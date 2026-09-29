A discarded `gather`, `scatter`, `scatter_replace`, `scatter_elements` or
`one_hot` is no longer eliminated, so its out-of-bounds index reports in the
evaluator and in compiled C instead of vanishing. `spec/06-transformations.md`
§5.2 makes a potentially trapping node an observable root, and [05-OP-52]'s
index domain is data, so no static fact rules the failure out for a non-literal
index; a literal one the checker rejects before lowering. These five operations
now form their own run-time check class, which the one seed predicate reads,
and each member is pinned individually against reclassification.

Still not covered for this class: the check is not gated by its node's
activation, so it runs in an untaken `if` arm, consumed or discarded, as it did
before. See [#2440](https://github.com/Chelis-Lang/chelis/issues/2440).
