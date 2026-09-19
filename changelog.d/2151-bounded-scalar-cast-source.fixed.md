`cast` and `cast_trunc` now accept a scalar source whose type is a
dtype-family-bounded binder, such as `u: p` under `[p: Float]`. The checker
previously rejected it with "cast requires tensor or prim type, got `p`",
though the tensor form over the same binder was accepted and [05-OP-6] gives
both surfaces the same semantics. Evaluation and generated C agree on the
result at `f32` and `f64`.

The same holds when the bound reaches the cast through an inference variable,
for example a lambda parameter later identified with the binder or passed
through a `Float`-bounded function. The verdict no longer depends on which
operand inference visits first.

An invalid scalar target, or a non-integer `cast_trunc` target, is rejected
with its real reason. A `cast_trunc` from a scalar bounded by `Int` or
`Numeric` is still rejected, with the same message as before, and so is an
unbounded binder source. See
[#2151](https://github.com/Chelis-Lang/chelis/issues/2151).
