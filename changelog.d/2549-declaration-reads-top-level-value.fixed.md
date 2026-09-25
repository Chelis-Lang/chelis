Two or more top-level function declarations may now read the same top-level
tensor-carrying value. The linearity checker walked each `def f() = value` as a
closure created in the top-level scope, so the first declaration consumed the
value and every later reader, whether another declaration, a top-level
initializer, or a second module of a package importing the same library value,
was rejected with "already consumed by closure capture" in `chelis eval`,
`chelis build`, `chelis check` and `chelis test`. A function declaration's body
is now checked against the ownership state once every top-level initializer has
run: it reads a value no initializer consumes without consuming it, and a
declaration that reads a value some initializer consumes is rejected wherever
the def sits in the source. Consumes inside one declaration body, and by a
closure an eager value creates, are still rejected.
See [#2549](https://github.com/Chelis-Lang/chelis/issues/2549).
