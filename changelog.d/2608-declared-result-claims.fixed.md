A declared result extent is now checked when it names a dimension binder, not
only when it is a literal. `def f[n](a: tensor[n, f32], t0: tensor[*, f32]) ->
tensor[n, f32] = t0` returned a three-element tensor for a two-element `a` on
both `chelis eval` and compiled C; it now traps with
``extent `n`: a axis 0 = 2, load axis 0 = 3`` and
`numeric trap: domain in load at i64`, naming the binder and the parameter axis
that declares it. A literal result claim on a block-bodied pass-through of a
wildcard parameter, which both lanes also dropped, now traps at the parameter
when the function runs as its own kernel.

A tensor held in the list or tuple that a List operation returns (`map`,
`filter`, `partition`, `fold`, `scan`, `flat_map`, `flatten`, `zip`,
`enumerate`, `append`, `concat` or `chunk`) now has that operation as its
producer for a declared-result check, while `index`, `take` and `skip` keep
the element's own producer (spec/04 §4.7). A valid program that indexes such a
result no longer aborts with an internal provenance error, and a wrong extent
traps as, for example, `numeric trap: domain in map at i64` on both lanes.

In a hand-built tensor graph, an operation that forwards an input axis under
another binder's name now checks the two extents itself and traps in that
operation. See [#2608](https://github.com/Chelis-Lang/chelis/issues/2608),
[#1900](https://github.com/Chelis-Lang/chelis/issues/1900),
[#2598](https://github.com/Chelis-Lang/chelis/issues/2598) and
[#2512](https://github.com/Chelis-Lang/chelis/issues/2512).
