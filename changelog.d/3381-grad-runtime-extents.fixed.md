`grad` now differentiates programs whose backward pass needs runtime extents in
two more places. A constant that an adjoint synthesizes, such as the `2` in the
`sqrt` derivative, now takes its extents from the value it sits beside. Before,
a group RMS normalisation over extents computed from shape reads failed with
`0 extent source(s) ... (op const)`. A `concat` whose parts have a runtime
extent on the concat axis, from a signature name or a wildcard, now splits its
cotangent at the runtime part boundaries inside `grad` and `vmap` bodies.
Before, it was refused with `tensor concat cannot be represented by the static
tensor DAG`. See [#3381](https://github.com/Chelis-Lang/chelis/issues/3381) and
[#3390](https://github.com/Chelis-Lang/chelis/issues/3390).
