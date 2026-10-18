`chelis prove`'s fuzz tier compiles a property whose binders are scalars,
booleans or strings once, as functions of its binders, and evaluates every
sample, guard and shrink trial as one call of them. Verdicts, counterexamples,
shrink steps, seeds and error messages are unchanged. Properties with a tensor
or opaque-invariant binder keep the chunked compile. See
[#3801](https://github.com/Chelis-Lang/chelis/issues/3801).
