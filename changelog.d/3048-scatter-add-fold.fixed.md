A built program now computes the gradient of a `gather` with repeated indices
exactly as `chelis eval` does. The C backend adds each destination's
contributions with the balanced adjacent-pair tree that [05-OP-33] specifies,
instead of folding them left to right; for example, contributions
`[1, B, -B, 0.5]` with large `B` now give 1.5 in both lanes, where the built
program previously gave 0.5. Integer scatter-add in the C backend now checks
overflow at every addition. See
[#3048](https://github.com/Chelis-Lang/chelis/issues/3048).
