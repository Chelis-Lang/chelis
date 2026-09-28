`chelis build` now accepts a `filter` or `partition` whose predicate passes the
loop item to a named definition, as in `filter(fn (x: string) -> keep(x), xs)`,
`filter(keep, xs)`, or a predicate received as a parameter. Previously the
build failed in the ownership stage with `owner %N ... is not live`. See
[#2577](https://github.com/Chelis-Lang/chelis/issues/2577).
