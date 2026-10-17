A compiled f16 or bf16 `einsum` contraction with an empty output and an
enormous reduction, for example `mk,nk->mn` over `[0, 2^62] x [0, 2^62]`, now
returns the empty tensor as `chelis eval` does. It previously trapped with
`Overflow: einsum reduction buffer byte size exceeds i64`. See
[#3783](https://github.com/Chelis-Lang/chelis/issues/3783).
