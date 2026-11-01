`argmax_reduce` and `argmin_reduce` produce i64 indices in compiled C builds
when their operand comes from a computed reshape. Previously the compiled lane
stored those indices at the operand's dtype (for example i32) and a following
reshape copied them by byte count into an i64 result, so two rows `[1, 2]` read
back as `[8589934593, 0]` while `chelis eval` returned `[1, 2]`. The graph
verifier now rejects an index reduction whose result is not i64 and a reshape,
permute, expand, shrink or stride whose result dtype differs from its operand,
so a mislabeled graph fails the build instead of reinterpreting bytes. See
[#4105](https://github.com/Chelis-Lang/chelis/issues/4105).
