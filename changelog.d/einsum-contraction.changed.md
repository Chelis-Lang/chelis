Compiled `einsum` runs much faster. A float `einsum` with one reduction label
shared by both operands, an output ending in a row of one operand and a column of
the other, no repeated label, and static extents on every label both operands
carry (for example `amk,kn->amn`, `mk,nk->mn` or `bqd,bkd->bqk`) now compiles to
the same emitted product-sum kernel as `matmul`, at every float width and
accumulator. Every other `einsum`, integer ones included, keeps the runtime
kernel, which now computes each operand's offsets from per-label strides,
checking the largest offset once per call instead of an index per product.
Results are bit-identical, in the canonical product and reduction order. At
`[1,1024,256] x [256,256]`, a compiled `amk,kn->amn` drops from about 19.3e9 to
1.1e9 instructions, and the same contraction with runtime contraction extents,
which stays on the runtime kernel, drops to 1.5e9. See
[#3736](https://github.com/Chelis-Lang/chelis/issues/3736),
[#3703](https://github.com/Chelis-Lang/chelis/issues/3703) and
[#3702](https://github.com/Chelis-Lang/chelis/issues/3702).
