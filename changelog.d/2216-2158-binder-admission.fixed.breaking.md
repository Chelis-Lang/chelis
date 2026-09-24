The checker decides a builtin's operand rule when the operand's type is an
authored type binder, at every instantiation the binder admits
([04-INF-6], [04-DTYPE-2]). Previously a call whose operand was a binder, such
as `numel(k)` with `k: p` and `p: Float`, checked with score 1, then trapped in
`eval` and emitted C that does not compile. It is now rejected with the
operation's own diagnostic, naming the instantiation where it fails, for
example `p := f32`. An operation that every instantiation admits, such as
`take(xs, n)` with `n: p` and `p: Int`, is still accepted.
See [#2216](https://github.com/Chelis-Lang/chelis/issues/2216).

`cast_trunc` rejects a source whose dtype is an `Int` or `Numeric` binder, on
tensor and scalar sources and with a concrete or binder target, and names the
`Float` bound to declare ([05-OP-6]). A cast whose target is a binder applies
the float-source and integer-target rule to a concrete or binder source, and
requires a numeric source. A tensor precision that a later binding makes an
integer is rejected too. Previously these programs checked with score 1 and
panicked in `eval` or C emission. A variable-target `cast_trunc` whose source
precision a later binding makes a float is now accepted instead of rejected.
See [#2158](https://github.com/Chelis-Lang/chelis/issues/2158).
