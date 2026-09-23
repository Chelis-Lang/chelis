An [05-OP-68] guarded abort whose result nothing consumes is no longer
removed by dead-code elimination. The atom says the abort may not be
removed, but an abort has no consumer by design, and every liveness
computation in the compiler derived "live" from value reachability — so
`grad` over `{ ignored = fail("m")  sum(x, 0i32) }` swept the guard and
returned a value with exit 0. Effect nodes are now identified by one shared
predicate and seeded at each of the three liveness sites (dead-code
elimination, grad's own pruner, and the evaluator's root mask), and the
verifier treats a guarded abort like `Store` and `Drop` rather than
requiring it to have a consumer. The generated C lane was broken the same
way and is fixed with it. Note one consequence for programs with no firing
guard: an untaken discarded guard now evaluates its fallback, so a fallback
that traps on its own will trap, per [05-OP-68]'s rule that the fallback is
an ordinary operand. Fixes
[#2368](https://github.com/Chelis-Lang/chelis/issues/2368).
