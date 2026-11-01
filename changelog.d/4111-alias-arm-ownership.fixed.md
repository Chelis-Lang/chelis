`chelis build` accepts a program in which a join arm (an `if` or `match` arm,
or a block's result) returns an alias of a binding the enclosing scope reads
afterwards, when that binding holds a block's result. Ownership lowering
recorded such a result at the depth its block minted it rather than the scope
that binds it, so behind enough nested bindings the arm moved the binding and
the build failed with `owner %N ... is not live` for a program `chelis check`
and `chelis eval` accepted. The arm now receives a copy and the binding stays
live for its later read. See
[#4111](https://github.com/Chelis-Lang/chelis/issues/4111).
