`chelis build` now lowers a scalar `cast` to a dtype-family-bounded binder, such
as `cast(numel(v), p)` under `[n, p: Float]`, inside a generic function that
takes a `tensor[.., p]` parameter. Such calls are inlined at the call site, and
the inlined body was never given the call site's type bindings, so every
consumer build, f32 included, failed with
``unsupported: dtype `p` on a `cast` target in host lowering``. Evaluation and
generated C now agree at `f32` and `f64`, including when the cast is reached
through another generic function.

Not yet covered: a scalar cast that is lowered inside an extracted tensor
kernel, rather than in the host lane, still fails with "a non-primitive cast
target expression". See
[#2152](https://github.com/Chelis-Lang/chelis/issues/2152) and
[#1564](https://github.com/Chelis-Lang/chelis/issues/1564).
