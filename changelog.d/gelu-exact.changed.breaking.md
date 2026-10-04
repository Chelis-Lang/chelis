`gelu` is now the exact Gaussian error linear unit `x * Phi(x)`, where `Phi` is
the standard normal CDF. The tanh approximation it used to compute keeps its
values under the new name `gelu_tanh`, for models trained with it (GPT-2 and
its descendants). Programs that relied on the old `gelu` values, such as parity
tests against `torch.nn.functional.gelu(approximate="tanh")`, must call
`gelu_tanh` instead. See
[#2957](https://github.com/Chelis-Lang/chelis/issues/2957).

What else changes:

- `erf` and `erfc` are new correctly rounded primitives ([05-OP-46]), taken from
  the vendored CORE-MATH kernels at f32 and f64, with f16 and bf16 rounded once
  from the f32 result. A user `def erf` or `def erfc` outside a package now
  shadows a builtin and is rejected.
- `standard_normal_cdf` is a new builtin: the standard normal CDF `Phi`, built from
  `erfc` with a correction for the rounding of `-x/sqrt(2)`. It stays within
  about one unit in the last place at f32 and f64, including the deep left tail.
- `Std.Contracts.normal_cdf` calls `standard_normal_cdf`. It replaces the
  Abramowitz-and-Stegun polynomial, so its values change at every width.
- `silu`, `gelu`, and `gelu_tanh` return `-0.0` at `-inf` instead of NaN, and
  `+inf` at `+inf`. Every finite input keeps its result.
- The wire DAG schema version is 25. A version-24 reader rejects a graph, which
  may now contain `Erf` or `Erfc`.
