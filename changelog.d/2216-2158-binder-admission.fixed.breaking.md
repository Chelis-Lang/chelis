The checker decides a builtin's operand rule when the operand's type is an
authored type binder, at every instantiation the binder admits
([04-INF-6], [04-DTYPE-2]). Previously a call whose operand was a binder, such
as `numel(k)` with `k: p` and `p: Float`, checked with score 1, then trapped in
`eval` and emitted C that does not compile. It is now rejected with the
operation's own diagnostic, naming the instantiation where it fails, for
example `p := f32`. A binder with no dtype-family bound is decided at an
arbitrary type rather than at any one type, so an operation that admits only
some types, such as `eq`, is rejected on it. An operation that every
instantiation admits, such as `take(xs, n)` with `n: p` and `p: Int`, is still
accepted.
See [#2216](https://github.com/Chelis-Lang/chelis/issues/2216).

`cast_trunc` rejects a source whose dtype is an `Int` or `Numeric` binder, on
tensor and scalar sources and with a concrete or binder target, and names the
`Float` bound to declare ([05-OP-6]). A cast whose target is a binder applies
the float-source and integer-target rule to a concrete source or a source
whose type is a binder at the cast, and rejects a source that is a binder with
no dtype-family bound; a source that is still an inference variable at the cast,
such as a lambda parameter, is not yet checked
([#2534](https://github.com/Chelis-Lang/chelis/issues/2534)). A tensor
precision that a later binding makes an integer is rejected too. Previously
these programs checked with score 1, except a scalar binder source with a
concrete target and a tensor source with a binder target, which were already
rejected with another diagnostic; the issue's programs then panicked in `eval`
or C emission. A cast to a binder target now accepts a `bool` source
([04-NUM-14], [05-OP-63]), which the scalar form rejected before, and a
variable-target `cast_trunc` whose source precision a later binding makes a
float is now accepted instead of rejected. A dtype-family requirement that
reaches an authored binder through a lambda parameter is reported once, against
the binder, and no longer again against the parameter.
See [#2158](https://github.com/Chelis-Lang/chelis/issues/2158).
