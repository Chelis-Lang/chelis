Compiled C now lowers a `match` on every scalar dtype and on tuples, and a
nested sub-pattern constrains its arm. Before, a match on an `i8`, `i16`,
`i32`, `f16` or `bf16` value or on a tuple was rejected at build with the
misattributed "an Option match has no `Some` arm"
([#2446](https://github.com/Chelis-Lang/chelis/issues/2446)), a `bool` match
without a wildcard arm was rejected for lacking a default, and a nested
sub-pattern under `Some` or a user constructor, such as `Some(Some(v))` or
`Wrap(None)`, was ignored so the arm was selected anyway
([#2450](https://github.com/Chelis-Lang/chelis/issues/2450)). Compiled programs
with such nested patterns can return different values.
