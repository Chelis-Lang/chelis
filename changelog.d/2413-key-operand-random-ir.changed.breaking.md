Random draws take their controls and their key as operands ([#2413](https://github.com/Chelis-Lang/chelis/issues/2413), phase 3). Every lane now evaluates the same key-operand graph, and these behaviors change:
- a runtime `dropout` rate and runtime `uniform_like` bounds are accepted in every lane; the static-rate requirement and its diagnostics are gone ([#2411](https://github.com/Chelis-Lang/chelis/issues/2411));
- compiled C validates `uniform_like` bounds under [05-OP-8] and traps an invalid bound before the draw, as eval does; an invalid literal `dropout` rate in compiled C now builds and traps at the draw, as eval does, where the entry lane used to refuse it at code generation;
- a public C entry that draws with no `with seed` handler of its own aborts, where it used to draw at seed 0;
- compiled C honors a helper's own `with seed` inside an active handler, where the enclosing handler's stream used to override it;
- `vmap` over a function that draws is refused in every lane ([#2409](https://github.com/Chelis-Lang/chelis/issues/2409));
- `grad` through a `dropout` rate that a differentiated parameter reaches is rejected with `RandomSelectionParameter` ([#2421](https://github.com/Chelis-Lang/chelis/issues/2421));
- a discarded `dropout` still takes its ordinal, but its data input is no longer a required entry input.

BREAKING: WireDag moves to schema version 17. Random nodes carry no fields; `DrawKey`, `DropoutReplay` and `UniformBoundAdjoint` are new, and a version-16 payload is rejected. The cached standard-library and compiled-context formats move to 27 and 29, so existing caches are rebuilt.
