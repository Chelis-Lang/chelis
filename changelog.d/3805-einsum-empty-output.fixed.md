A compiled `einsum` with an empty output no longer fails with `Overflow: einsum
reduction buffer byte size exceeds i64` when its reduction extent is very large,
for example `einsum("bkn,bmk->bmn", p, q, accumulator=f64)` with `b = 0` and
`k = 2^61`. It returns the empty tensor, as `chelis eval` already did. See
[#3805](https://github.com/Chelis-Lang/chelis/issues/3805).
