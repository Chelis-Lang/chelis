A declared result extent is now checked when it names a dimension binder, not
only when it is a literal. `def f[n](a: tensor[n, f32], t0: tensor[*, f32]) ->
tensor[n, f32] = t0` returned a three-element tensor for a two-element `a` on
both `chelis eval` and compiled C; it now traps with
``extent `n`: a axis 0 = 2, load axis 0 = 3`` and
`numeric trap: domain in load at i64`, naming the binder and the parameter axis
that declares it. A literal result claim on a block-bodied pass-through of a
wildcard parameter, which both lanes also dropped, now traps at the parameter.
In a hand-built tensor graph, an operation that forwards an input axis under
another binder's name now checks the two extents itself and traps in that
operation. See [#2608](https://github.com/Chelis-Lang/chelis/issues/2608),
[#1900](https://github.com/Chelis-Lang/chelis/issues/1900) and
[#2512](https://github.com/Chelis-Lang/chelis/issues/2512).
