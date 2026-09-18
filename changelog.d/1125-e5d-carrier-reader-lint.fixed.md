`chelis lint` now rejects new reader-side `Expr::List` patterns and
explicit `chelis_deep::Node::to_list` paths outside `chelis-deep`. The
structural rule permits carrier-total matches without prescribing a helper,
catches carrier-arm deletion, checks newly added source without freezing
[#1125](https://github.com/Chelis-Lang/chelis/issues/1125) E5e debt by source
identity, and requires a site-specific producer or proven-symmetric necessity
for exceptions.
