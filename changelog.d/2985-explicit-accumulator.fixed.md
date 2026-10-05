`matmul`, `sum` and `einsum` accept their explicit accumulator in Surf as a
final `accumulator=<dtype>` argument, as in `sum(x, 0i32, accumulator=f64)`,
which previously failed to parse. It desugars to the call's Deep
`accumulator` metadata, formats and resugars unchanged, and is checked
against the spec/04 §5.7.1 permitted-pairs table: an accumulator narrower
than the default, of the other numeric kind, or on any other call is a type
error. Eval and compiled C agree at every permitted pair. See
[#2985](https://github.com/Chelis-Lang/chelis/issues/2985).
