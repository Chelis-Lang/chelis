Add IR evaluator APIs for preparing strict-root and selected-plan inputs through
a fallible provider before execution. Existing evaluator signatures and input
selection remain unchanged. This is a prerequisite for, not a fix to,
[#1956](https://github.com/Chelis-Lang/chelis/issues/1956).
