`cast` and `cast_trunc` now accept a scalar source whose type is a
dtype-family-bounded binder, such as `u: p` under `[p: Float]`. The checker
previously rejected it with "cast requires tensor or prim type, got `p`",
though the tensor form over the same binder was accepted and [05-OP-6] gives
both surfaces the same semantics. Evaluation and generated C agree on the
result at `f32` and `f64`.

`cast_trunc` from a scalar bounded by `Int` or `Numeric` is now rejected at the
declaration, naming [05-OP-6]'s float-source requirement, instead of with the
generic message. An unbounded binder source stays rejected. See
[#2151](https://github.com/Chelis-Lang/chelis/issues/2151).
