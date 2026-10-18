`chelis prove`'s fuzz check of an opaque type's invariant compiles each
producer whose parameters are scalars, booleans or literal-shape tensors once,
as a function of its parameters, and evaluates every sample and shrink trial as
one call of it. For every producer the per-sample probe could evaluate,
verdicts, counterexamples, shrink steps, seeds and error messages are
unchanged. Some producers it could not evaluate now check. A rank-0 tensor
input, for example, used to error on every sample (with messages such as
"expected rank-0 tensor, got rank-1 tensor"), and it now reaches a verdict.
An input the old path rejected may still be rejected, with a different
message. A producer that takes an opaque-invariant value keeps the per-sample
evaluation. See [#3800](https://github.com/Chelis-Lang/chelis/issues/3800).
