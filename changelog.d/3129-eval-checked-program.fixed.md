`chelis eval --file` evaluates the program it type-checked. A single-file
program was checked as parsed, then printed back to Surf and evaluated from a
second parse, so a Surf printer defect changed the printed value: `(if true
then 1.0 else 3.0) |> inner` evaluated to `1.0` where `chelis check` and
`chelis deep` followed by `eval` give `11.0`. `chelis test` (module
initialization and test roots), the `chelis prove` sampled-property lane, its
producer obligations, and the `normal_cdf` reference evaluation now also run
the declarations they hold instead of a printed copy. See
[#3129](https://github.com/Chelis-Lang/chelis/issues/3129).
