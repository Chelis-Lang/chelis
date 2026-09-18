`chelis lint` now rejects new reader-side `Expr::List` patterns and
`Node::to_list` bridges outside `chelis-deep`. The structural rule permits
carrier-total matches without prescribing a helper, freezes existing
[#1125](https://github.com/Chelis-Lang/chelis/issues/1125) E5e reader debt by
exact site identity, and requires a site-specific rationale for exceptions.
