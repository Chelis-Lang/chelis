`chelis lint` gains the warning rule `prefer-typed-literal`
(`spec/01-nomenclature.md` §12.6). It reports `cast(<literal>, <dtype>)` where
the suffixed literal binds the same value at the same dtype, such as
`cast(1.0, f32)` or `cast(3, i64)`, and `chelis lint --fix` rewrites it to
`1.0f32` or `3i64`, keeping the literal body as written. It does not report a
negative operand, a float body under an integer dtype, a radix body under a
float dtype, or a dtype outside the suffix set. The rule is non-blocking, so
`chelis check`, `build`, and `eval` print the warning and still accept the
program. The executable examples and the specification's Surf examples now use
typed literals except where an example demonstrates `cast` itself.
