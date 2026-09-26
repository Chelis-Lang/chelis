Names keep the meaning they have where they are written on more routes.
When `chelis eval` or `chelis test` applies `grad` or `vmap` to a function
literal, the literal reads the caller's locals, and each closure it reaches
reads what it closed over, while a declaration the literal calls reads
top-level names. Previously a caller's local or local function could replace a
top-level value or function the declaration read, a later rebinding could
replace a closure's captured value, and a captured scalar could go missing
([#2619](https://github.com/Chelis-Lang/chelis/issues/2619),
[#2380](https://github.com/Chelis-Lang/chelis/issues/2380)). A let-bound list
literal or `shape(x, axis)` keeps the values it was bound from when a name it
reads is rebound before its use
([#2603](https://github.com/Chelis-Lang/chelis/issues/2603)). A tensor
parameter named like a top-level function, and a local function named like a
builtin, are no longer taken for the function or builtin
([#1949](https://github.com/Chelis-Lang/chelis/issues/1949),
[#1964](https://github.com/Chelis-Lang/chelis/issues/1964)). A top-level value
defined by a block now has a type, so a named-axis call that reads it evaluates
and compiles ([#2547](https://github.com/Chelis-Lang/chelis/issues/2547)).
Repeated `grad` and `vmap` applications in the interpreter no longer re-scan
the whole program each time
([#2439](https://github.com/Chelis-Lang/chelis/issues/2439)).
