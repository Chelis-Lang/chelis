Selected Host evaluation entries now receive their required caller-supplied tensor
arguments, including inputs used only by lowered shape witnesses. Missing inputs
leave the declaration unentered, while dead and unrelated bindings remain
unobserved. Prepared library calls preserve fresh actuals, existing Random
behavior, and runtime error ordering. Fixes
[#2013](https://github.com/Chelis-Lang/chelis/issues/2013).
