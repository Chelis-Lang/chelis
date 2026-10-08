Compiled C programs check the current thread's stack before recursive user
function calls and report a located `RuntimeStackBudgetExhausted` error instead
of dying silently when the reserved budget is exhausted. See
[#2312](https://github.com/Chelis-Lang/chelis/issues/2312).
