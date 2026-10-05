A type error caused by `sum`, `cumsum`, `trace` or `einsum` returning i32 for
an i8 or i16 operand now says so: it names the operation, the operand and
result dtypes and `spec/04` §5.7.1, and suggests declaring the result as i32
(or, for `sum` and `einsum`, passing `accumulator=i64`) or narrowing it with
an explicit `cast`. This covers a declared result, a
`let` ascription, a downstream operation that needs the operand dtype, and a
generic precision variable whose bound has no single result. The type
reference and the surface reference state the rule with a table of the four
operations.
