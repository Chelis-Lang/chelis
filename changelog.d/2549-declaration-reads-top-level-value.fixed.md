Two or more top-level function declarations may now read the same top-level
tensor-carrying value. The linearity checker walked each `def f() = value` as a
closure created in the top-level scope, so the first declaration consumed the
value and every later reader, whether another declaration, a top-level
initializer, or a second module of a package importing the same library value,
was rejected with "already consumed by closure capture" in `chelis eval`,
`chelis build`, `chelis check` and `chelis test`. A function declaration now
reads top-level values without consuming them, and consumes inside one
declaration body, or by a closure an eager value creates, are still rejected.
See [#2549](https://github.com/Chelis-Lang/chelis/issues/2549).
