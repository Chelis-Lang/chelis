Generated C checks simultaneous local reshape extent claims in declaration
order, including when the output reverses the claimed axes. Eval and C now
report the same first failing claim for that case. Part of
[#1277](https://github.com/Chelis-Lang/chelis/issues/1277).
