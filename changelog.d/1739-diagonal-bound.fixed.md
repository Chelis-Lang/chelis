`diagonal` over an axis pair with one symbolic and one literal extent no longer admits a
declared result extent the runtime can never produce. `[05-OP-33]` takes the smaller selected
extent, so the literal axis is an upper bound: a declared extent strictly greater than it is
now a `DimensionMismatch` at check time, on both the `def`/`sig` route and a block-scoped
ascription. A declaration at or below the bound stays accepted and is checked at run time: a
host-lane function whose return expression is a direct builtin application and whose declared
tensor result carries a literal extent now traps `Domain` on disagreement, identically on
`chelis eval` and the C backend, instead of silently returning a differently shaped tensor.
