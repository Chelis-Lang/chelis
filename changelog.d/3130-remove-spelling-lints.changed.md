`chelis lint` no longer has the `prefer-pipe-operator`,
`redundant-linearity-call`, or `prefer-typed-literal` rules. Each one rewrote
source from one spelling into another, and type-checking the rewrite does not
show that it means the same program: `prefer-pipe-operator` changed
`2.0 * outer(inner(x), 1.0)` into `2.0 * x |> inner |> outer(1.0)`, which
evaluates differently, and `redundant-linearity-call` turned
`result = drop(f(1.0))` into `result = f(1.0)`. `chelis lint` and
`chelis check` no longer print these warnings, `chelis lint --fix` no longer
changes these spellings, and `--rule` or `--rules` naming one of the three ids
reports that no such rule exists. With them goes the typed-pipeline check that
gated their fixes, so `chelis lint` no longer type-checks the files it reads.
See [#3119](https://github.com/Chelis-Lang/chelis/issues/3119) and
[#3130](https://github.com/Chelis-Lang/chelis/issues/3130).
