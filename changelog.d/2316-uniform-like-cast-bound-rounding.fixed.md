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
[04-NUM-5] already required a compile-time fold to use exact per-dtype semantics
or decline to fold. The rounding now has a single definition shared by both fold
sites. See [#2316](https://github.com/Chelis-Lang/chelis/issues/2316).
