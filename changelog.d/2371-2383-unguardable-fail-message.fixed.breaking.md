A `fail(...)` under `grad` or `vmap` whose message is not a string literal written
at the `fail` itself is now rejected instead of silently becoming a value.
[05-OP-68] carries the abort message as part of the operation's identity and takes
no string operand, so such a `fail` has no guarded form; lowering substituted a
zero placeholder, and under a transform that placeholder became part of the
answer — `grad` over `sum(add(x, fail(string_concat("bad: ", tag()))), 0i32)`
returned `1.0` at exit 0, and `vmap` echoed its inputs with the abort gone
entirely. A `let`-bound literal message had the same effect. The rejection is
taken by the consuming transform rather than at the `fail`, so an untransformed
`fail` with a computed message — including one reached through a `string`
parameter — still aborts at run time with its message. A literal message still
builds its guard and still aborts, and an untaken guard still differentiates.
A direct `fail(...)` whose message cannot be used no longer reports the
indirect-`fail` diagnostic that told the author to write `fail` directly as the
branch when that is what they had written; one classifier now returns either a
usable message or a typed reason, and a genuinely indirect `fail` keeps the
indirect wording
([#2383](https://github.com/Chelis-Lang/chelis/issues/2383)). Part of
[#1464](https://github.com/Chelis-Lang/chelis/issues/1464) and
[#730](https://github.com/Chelis-Lang/chelis/issues/730).
