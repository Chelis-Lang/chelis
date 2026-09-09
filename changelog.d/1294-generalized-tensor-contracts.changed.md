Replace the `conv2d` builtin with `conv`, using explicit per-axis int64
strides and `(low, high)` padding pairs for any positive spatial rank.
Static convolution lowering now contracts the complete window matrix in
the specified order; zero-sized contractions retain their RISC graph.

Generalize the normative `to_list` contract to nested Lists at every
positive tensor rank and `tensor_scan` to fixed-shape tensor state.
Their runtime implementations still have rank-one/scalar restrictions,
recorded as implementation gaps in the dtype plan.

Require an explicit same-dtype epsilon for `layer_norm`, including its
adjoint. Correct standard lowering recipes to insert axes explicitly and
select embeddings and masked values with `gather` and `where`.

Run host `matmul` through the typed rank-generic lowering and preserve a
tensor boundary for nested host operands in generated C. Batched attention
and broadcast matrix calls now execute in both lanes.

Preserve canonical contraction trees during C selection instead of implicitly
substituting vendor GEMM, including generated convolution and hosted matmul.
Remove derived BLAS helper/wrapper shortcuts before binding C ownership.
Ordinary sum now uses the canonical adjacent-pair tree in evaluation and
generated C, preserving odd tails, signed zero, and checked integer overflow.
