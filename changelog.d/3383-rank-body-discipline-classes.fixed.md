A rank-polymorphic (`..r`) body admits the four kinds of builtin that spec/04
§4.5.3 now defines: shape-identity, named-axis, ordered-prefix, and inert, where
an inert builtin returns unit, returns its operand unchanged, or never returns.
`fail`, `drop`, `print`, `debug`, `write_file`, and the `test_assert` family
are inert, `dropout` is shape-identity, and `layer_norm` over a named trailing
axis is named-axis; they were previously rejected as "shape-rewriting". A
builtin that returns another value, such as a string or a scalar, is still
rejected. `dropout`, `uniform_like`, and `test_assert_close_tensor`, whose
schemes take a lone-spread row, are admitted on a spread-only operand such as
`tensor[..r, f32]`; on an anchored row such as `tensor[..r, hidden, f32]` the
rank-spread unifier still rejects them
([#3529](https://github.com/Chelis-Lang/chelis/issues/3529)). A `layer_norm`
operand whose row ends in a spread is rejected with a diagnostic that names
the missing trailing axis, and the rejection of any other builtin states the
admitted kinds rather than calling it shape-rewriting. See
[#3383](https://github.com/Chelis-Lang/chelis/issues/3383).
