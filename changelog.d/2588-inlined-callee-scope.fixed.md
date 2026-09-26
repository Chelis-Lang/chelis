A called function now reads its free names in the scope it was written in,
wherever the compiler inlines it. A caller's local, parameter or local function
alias that shares a name with a top-level value or function the callee reads no
longer replaces it, so `chelis eval` and compiled C return the lexically scoped
result instead of silently using the caller's binding. This holds through
nested calls, `grad`, `vmap` and function-valued arguments, with one exception
that is not yet fixed: when `chelis eval` or `chelis test` applies a `grad` or
`vmap` to a function literal, in a top-level value's block or a test body, a
caller's local or local function can still replace a top-level name that the
literal, or a declaration it calls, reads
([#2619](https://github.com/Chelis-Lang/chelis/issues/2619)). Where the
compiled tensor graph would have to name the top-level value and a same-named
local or parameter in scope at the call the same way, the compiler now fails
loudly instead: `chelis eval` reports that the graph shares the name (for
`grad` over a function whose parameter is named like a value one of its
callees reads), and a compiled build can report that it cannot lower or link
the call, including a `grad` or `vmap` in a top-level value's block beside a
shadowing local. Both lanes previously returned the caller's value there. See
[#2588](https://github.com/Chelis-Lang/chelis/issues/2588) and
[#2604](https://github.com/Chelis-Lang/chelis/issues/2604).
