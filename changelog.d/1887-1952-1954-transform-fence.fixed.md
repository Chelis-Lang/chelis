`check` now rejects unsafe transform targets before evaluation: local aliases of
top-level functions, local bindings that shadow transform targets, and inline
`vmap` lambdas. See [#1887](https://github.com/Chelis-Lang/chelis/issues/1887),
[#1952](https://github.com/Chelis-Lang/chelis/issues/1952), and
[#1954](https://github.com/Chelis-Lang/chelis/issues/1954).
