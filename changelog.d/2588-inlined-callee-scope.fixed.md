A called function now reads its free names in the scope it was written in,
wherever the compiler inlines it. A caller's local, parameter or local function
alias that shares a name with a top-level value or function the callee reads no
longer replaces it, so `chelis eval` and compiled C return the lexically scoped
result instead of silently using the caller's binding. This holds through
nested calls, `grad`, `vmap` and function-valued arguments, in function bodies
and in top-level values' blocks. Where a function's parameter, or a function
local in scope at the call, shares its name with a top-level value that an
inlined callee reads, and the compiled graph would have to name both the same
way, the compiler now fails loudly instead: `chelis eval` reports that the
graph shares the name (for `grad` over such a function), and a compiled build
can report that it cannot lower or link the call. Both lanes previously
returned the caller's value there. See
[#2588](https://github.com/Chelis-Lang/chelis/issues/2588) and
[#2604](https://github.com/Chelis-Lang/chelis/issues/2604).
