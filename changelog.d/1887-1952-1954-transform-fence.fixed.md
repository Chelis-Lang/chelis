`check` now rejects unsafe transform targets before evaluation: aliases of
top-level functions at module or local scope, local bindings that shadow
transform targets, and inline or locally bound `vmap` lambdas with untyped
parameters, including local alias chains. See [#1887](https://github.com/Chelis-Lang/chelis/issues/1887),
[#1952](https://github.com/Chelis-Lang/chelis/issues/1952), and
[#1954](https://github.com/Chelis-Lang/chelis/issues/1954), with the
local-lambda fence completed by
[#2109](https://github.com/Chelis-Lang/chelis/issues/2109).
