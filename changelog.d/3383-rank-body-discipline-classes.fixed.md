A rank-polymorphic (`..r`) body admits every operation whose result shape the
checker tracks by name, as spec/04 §4.5.3 now states by property: `fail`,
`drop`, `print`, `debug`, `write_file`, and the `test_assert` family (they
produce no new tensor), `dropout` (shape-identity), and `layer_norm` over a
named trailing axis. They were previously rejected as "shape-rewriting". A
`layer_norm` operand whose row ends in a spread is rejected with a diagnostic
that names the missing trailing axis, and the rejection of an untracked builtin
no longer calls it shape-rewriting. See
[#3383](https://github.com/Chelis-Lang/chelis/issues/3383).
