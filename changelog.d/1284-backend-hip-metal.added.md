HIP now executes direct tensor comparisons, Bool8 logical operations, and
stored-bit `where` selection across all active device dtypes. Metal reports a
stable typed unsupported diagnostic for these operations until its exact
kernels land. See [#1284](https://github.com/Chelis-Lang/chelis/issues/1284).
