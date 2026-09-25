The DAG evaluator raises the typed `Overflow` trap for an `expand` or
`split_keys` whose extents give a result it cannot represent, as compiled C
does, where it used to panic with `capacity overflow`. A representable result
the evaluator cannot allocate fails as the C runtime's allocation failure
does. Fixes [#2491](https://github.com/Chelis-Lang/chelis/issues/2491).
