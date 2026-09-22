A `uniform_like` bound written as a chain of casts now applies every cast's
rounding in the compiled lanes, matching `chelis eval`. Previously both compiled
lanes folded the chain by value and kept the innermost literal, so a bound such
as `cast(cast(0.30000001, f16), f32)` compiled to `0.30000001` where the program
declares `0.300048828125` — about 1600 f32 ULPs — and the binary sampled an
interval its source never declared. The two compiled lanes agreed with each
other and diverged from `eval`, so the error did not show up as a lane
inconsistency.

`spec/04-type-system.md` [04-NUM-14] already required a float-target cast to be
total IEEE-754 round-to-nearest, ties-to-even at the target width, and
[04-LIT-1] already required a literal to be finalized once at its declared
width without passing through f64 first. Both compiled folds now stage a bound
the way the evaluator does — an integer leaf stays exact through i64 and a
float leaf stays at its source dtype until a cast finalizes it — so they apply
the same roundings in the same order. This also stops an exact integer bound
beyond f64's integer range from losing a bit on its way to f32.
See [#2316](https://github.com/Chelis-Lang/chelis/issues/2316).
