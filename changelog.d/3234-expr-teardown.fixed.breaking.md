Direct Surf parser callers can release deeply nested expression ASTs on small
worker stacks without a native stack-overflow abort. Rust callers must borrow
`Expr` fields or take them through mutable references instead of moving them
out through by-value matches. See [#3234](https://github.com/Chelis-Lang/chelis/issues/3234).
