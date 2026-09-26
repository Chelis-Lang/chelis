A called function now reads its free names in the scope it was written in,
wherever the compiler inlines it. A caller's local, parameter or local function
alias that shares a name with a top-level value or function the callee reads no
longer replaces it, so `chelis eval` and compiled C return the lexically scoped
result instead of silently using the caller's binding. This holds through
nested calls, `grad`, `vmap` and function-valued arguments. Where a compiled
tensor graph would have to name the top-level value and a same-named input
with one name, the compiler now declines that graph, and a `grad` or `vmap`
in a top-level value's block that previously built with the caller's value
now reports that it cannot be lowered. See
[#2588](https://github.com/Chelis-Lang/chelis/issues/2588).
