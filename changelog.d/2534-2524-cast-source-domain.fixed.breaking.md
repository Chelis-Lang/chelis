A checked `cast` admits exactly the numeric dtypes and `bool` as its source
([05-OP-63], [04-NUM-14]), whether the source is written directly or reaches the
cast through a lambda parameter. Previously a `string` source checked with score
1: `cast(s, i64)` with `s: string`, and `(fn (y) -> cast(y, i64))("s")`
([#2524](https://github.com/Chelis-Lang/chelis/issues/2524)). A cast whose
target is a declaration's dtype binder also accepted, unchecked, a source that
was still an inference variable at the cast, so a `string` or an unbounded
binder reaching it through a lambda checked with score 1 and trapped in `eval`
([#2534](https://github.com/Chelis-Lang/chelis/issues/2534)). Such a source is
now decided when it binds, by the rule a direct source gets, and rejected at the
declaration boundary if it never binds.

A `let`-bound lambda whose operand check is still waiting (a `cast`, `copy`,
`gather`, `scatter`, `trace` or host-slot operand) now stays monomorphic until
its first application binds the operand ([04-INF-1]). It used to generalize, so
each application bound a fresh copy and the check was left on a variable nothing
bound: `g = fn (y) -> cast(y, f64)` followed by `g(x)` was rejected with
`got ?N`, and a lambda over `gather` or `trace` lost the direct form's verdict.
Those now check, `g("s")` is rejected naming `string`, and a second application
at a different type is rejected, as for any lambda that carries an obligation.
As a consequence a tensor reaching a
binder-target `cast` through a lambda now keeps its dimensions, as the direct
cast does, where it was rejected before; the `cast_trunc` form is unchanged
([#2535](https://github.com/Chelis-Lang/chelis/issues/2535)).
