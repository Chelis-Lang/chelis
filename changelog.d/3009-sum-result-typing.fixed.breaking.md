The checker types `cumsum`, `trace` and `einsum` by
`sum_result(p, default(p))`, the dtype both evaluation lanes produce: an i8 or
i16 operand gives an i32 result. A program that declared the operand dtype
for such a result is now rejected, as is `cumsum`, `trace` or `einsum` over
bool. `einsum`'s result extents now follow its equation when both operands
have settled tensor types. Over a precision variable, `sum`, `cumsum`,
`trace` and `einsum` keep the variable only when `sum_result` keeps every
dtype its bound admits (`Float`, for example), take one concrete dtype when
`sum_result` maps them all there (`{i8, i16}` gives i32), and are otherwise
rejected at the definition: a generic `[p: Int]` or unbounded `cumsum`
declared to return `p` no longer checks. See
[#3009](https://github.com/Chelis-Lang/chelis/issues/3009).
