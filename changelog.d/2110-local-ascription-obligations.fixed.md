Authored local tensor ascriptions now retain runtime-dependent extent claims
through checking, lowering, graph transformations, caches, WireDag artifacts,
Eval, and generated C. Disagreement traps at the initializer operation with the
typed `Domain` diagnostic; static disagreement remains `DimensionMismatch`.
See [#2110](https://github.com/Chelis-Lang/chelis/issues/2110).
