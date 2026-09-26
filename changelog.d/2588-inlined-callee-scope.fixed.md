A called function now reads its free names in the scope it was written in,
wherever the compiler inlines it. A caller's local, parameter or local function
alias that shares a name with a top-level value or function the callee reads no
longer replaces it, so `chelis eval` and compiled C return the lexically scoped
result instead of silently using the caller's binding. This holds through
nested calls, `grad`, `vmap` and function-valued arguments. One case now fails
loudly instead: a tensor graph that must read a top-level value while also
taking a same-named input, for example `grad` over a function whose parameter
is named like a top-level value one of its callees reads. `chelis eval` reports
that the graph shares the name, and a compiled build can report that it cannot
lower the definition, where both previously returned the caller's value. See
[#2588](https://github.com/Chelis-Lang/chelis/issues/2588) and
[#2604](https://github.com/Chelis-Lang/chelis/issues/2604).
