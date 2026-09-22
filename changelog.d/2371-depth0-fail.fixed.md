A `fail(...)` with no enclosing `if` no longer returns a zero placeholder
inside `grad` or `vmap`. It is unconditional there, so it now lowers to an
[05-OP-68] guarded abort whose condition is a constant true, and the program
aborts with the authored message in both lanes. Previously `grad` over
`sum(add(x, fail("m")), 0i32)` returned `1.0` with exit 0 and emitted no
`chelis_fail` at all, so the fabricated zero silently changed the answer
rather than replacing it. The same route through a helper, a whole
transformed body, `vmap`, a `where` arm, and a statically-selected `match`
arm ([#2369](https://github.com/Chelis-Lang/chelis/issues/2369)) is fixed
with it, and `fail("")` is now rejected by name instead of becoming a
placeholder. Two shapes remain: a message that is not a compile-time literal
([#2383](https://github.com/Chelis-Lang/chelis/issues/2383)), and a guard
whose result is never consumed, which dead-code elimination still removes
([#2368](https://github.com/Chelis-Lang/chelis/issues/2368)). Part of
[#2371](https://github.com/Chelis-Lang/chelis/issues/2371).
