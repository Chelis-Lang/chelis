`grad`, `vmap`, and `vmap(grad)` now differentiate through float `diagonal`,
`trace`, `cumsum`, and `einsum` in `chelis eval` and compiled C at every float
dtype. Previously each was rejected with "application of `<op>` has no numeric
IR lowering". Each adjoint follows its atom's stated order: the reverse
inclusive scan at the sum accumulator for `cumsum`, the zero-filled diagonal
scatter for `diagonal` and `trace`, and the contraction of the cotangent with
the other operand in forward output-then-reduction order for `einsum`. Graphs
over runtime extents, and integer forms used inside a differentiated body, are
still rejected. `clamp`, `sort`, and add-mode `scatter` remain rejected under
`grad`. See [#3362](https://github.com/Chelis-Lang/chelis/issues/3362).
