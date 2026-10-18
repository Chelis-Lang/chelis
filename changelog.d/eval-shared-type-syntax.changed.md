`chelis eval` no longer copies the type syntax its bindings carry each time
it applies a function. Each call shares the declared types of its
parameters and arguments rather than copying them, keeps its per-binding type
scope in a structure whose snapshot at each `let` is constant-time, derives
a closure's nested claim patterns once instead of at every application, and
reads a definition's body without copying it. A fold step's allocations no
longer grow with the size of its parameter types: a 64-component tuple
accumulator drops from 1,161 allocations per step to 28. `linspace` over 1,000
elements drops from about 18.0 G to 9.2 G instructions. Results and
diagnostics are unchanged. See
[#3778](https://github.com/Chelis-Lang/chelis/issues/3778).
