`cast` and `cast_trunc` now accept a scalar source whose type is a
dtype-family-bounded binder, such as `u: p` under `[p: Float]`. The checker
previously rejected it with "cast requires tensor or prim type, got `p`",
though the tensor form over the same binder was accepted and [05-OP-6] gives
both surfaces the same semantics. Evaluation and generated C agree on the
result at `f32` and `f64`.

An invalid scalar target, or a non-integer `cast_trunc` target, is now rejected
at the cast with its real reason. A `cast_trunc` from a scalar bounded by `Int`
or `Numeric` is still rejected, with the same message as before. An unbounded
binder source stays rejected. See
[#2151](https://github.com/Chelis-Lang/chelis/issues/2151).
