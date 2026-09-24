Piped transform stages retain their own metadata, so a stage such as
`x |> grad(f, wrt=v)` checks, evaluates, and builds. Type ascriptions written
on a pipe result are also enforced. See [#2430](https://github.com/Chelis-Lang/chelis/issues/2430).
