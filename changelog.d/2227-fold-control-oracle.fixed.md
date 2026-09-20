The static-condition regression matrix now executes a three-row exact-integer
truth table above 2^53, checking exact stdout and stderr for taken, equal, and
reversed comparisons. Mutation controls cover comparator, operand, rounding,
dataflow, fail-message, fail-target, and branch-loss regressions. See
[#2227](https://github.com/Chelis-Lang/chelis/issues/2227).
