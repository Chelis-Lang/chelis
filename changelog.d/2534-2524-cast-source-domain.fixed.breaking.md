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
declaration boundary if it never binds. As a consequence a tensor reaching a
binder-target `cast` through a lambda now keeps its dimensions, as the direct
cast does, where it was rejected before; the `cast_trunc` form is unchanged
([#2535](https://github.com/Chelis-Lang/chelis/issues/2535)).
