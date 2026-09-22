A `fail(...)` with no enclosing `if` no longer returns a zero placeholder
inside `grad` or `vmap`. It is unconditional there, so it now lowers to an
[05-OP-68] guarded abort whose condition is a constant true: outside a
transform the guard is discarded with the rest of the speculative DAG and the
host lane still owns the abort, and inside one it fires with the authored
message. Previously `grad` over `sum(add(x, fail("m")), 0i32)` returned `1.0`
with exit 0 and emitted no `chelis_fail` at all, so the fabricated zero
silently changed the answer rather than replacing it. The same route through a
helper, a whole transformed body, `vmap`, and a `where` arm is fixed with it.
A message that is not a compile-time literal still has no guarded form
([#2383](https://github.com/Chelis-Lang/chelis/issues/2383)). Fixes
[#2371](https://github.com/Chelis-Lang/chelis/issues/2371).
