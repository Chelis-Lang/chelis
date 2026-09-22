A `fail(...)` branch inside `grad` or `vmap` now aborts with its authored
message instead of returning a zero placeholder. Lowering represented the
failing branch as a value, so a taken abort was selected as the result and
the program exited successfully with `0.0` in both the evaluator and the
generated C binary. The new [05-OP-68] guarded-abort identity keeps the trap
in the graph, so its occurrence survives differentiation, batching and code
generation, while an untaken guard still computes and differentiates
unchanged. A guard whose value is never consumed can still be removed by
dead-code elimination
([#2368](https://github.com/Chelis-Lang/chelis/issues/2368)), and a `fail`
in a statically-selected `match` arm is unchanged
([#2369](https://github.com/Chelis-Lang/chelis/issues/2369)). Part of
[#1464](https://github.com/Chelis-Lang/chelis/issues/1464).
