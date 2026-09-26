The compiled C of a function whose body reads a rank-zero top-level tensor
value, such as `v = mul(copy(y), scalar_to_tensor(3.0f32))` read by
`def g(x: tensor[f32]) -> tensor[f32] = add(x, copy(v))`, now compiles and
reads the value. The function passed the value's global to its tensor kernel
as a host scalar, which the C compiler rejected. See
[#2413](https://github.com/Chelis-Lang/chelis/issues/2413).
