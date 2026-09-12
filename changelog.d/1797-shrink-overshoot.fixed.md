`chelis eval` reports a `shrink` whose runtime end runs past the operand's
extent as `spec/05-risc-primitives.md` §2.4.1's overshoot error, in the same
words and with the same exit status as the compiled lane: `Domain: shrink
bounds outside input extent` followed by the `numeric trap: domain in shrink at
int64` line. Previously the evaluator panicked and exited 101 when the declared
result claimed the span's arithmetic width or claimed nothing, and reported a
mismatch against a claim that was not the defect when it claimed something
else. See [#1797](https://github.com/Chelis-Lang/chelis/issues/1797).
