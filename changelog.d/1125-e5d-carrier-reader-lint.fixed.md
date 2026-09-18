`chelis lint` now rejects new reader-side `Expr::List` patterns and
`Node::to_list` bridges outside `chelis-deep`. The structural rule permits
carrier-total matches without prescribing a helper, checks newly added source
without freezing [#1125](https://github.com/Chelis-Lang/chelis/issues/1125)
E5e debt by source identity, and requires a site-specific producer or
proven-symmetric necessity for exceptions.
