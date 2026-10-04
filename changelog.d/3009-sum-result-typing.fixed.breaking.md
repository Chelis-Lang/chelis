The checker types `cumsum`, `trace` and `einsum` by
`sum_result(p, default(p))`, the dtype both evaluation lanes produce: an i8 or
i16 operand gives an i32 result. A program that declared the operand dtype
for such a result is now rejected, as is `cumsum`, `trace` or `einsum` over
bool. `einsum`'s result extents now follow its equation when both operands
have settled tensor types. See
[#3009](https://github.com/Chelis-Lang/chelis/issues/3009).
