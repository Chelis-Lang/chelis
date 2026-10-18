`chelis prove`'s fuzz check of an opaque type's invariant compiles each
producer whose parameters are scalars, booleans or literal-shape tensors once,
as a function of its parameters, and evaluates every sample and shrink trial as
one call of it. Verdicts, counterexamples, shrink steps, seeds and error
messages are unchanged. A producer that takes an opaque-invariant value keeps
the per-sample evaluation. See
[#3800](https://github.com/Chelis-Lang/chelis/issues/3800).
