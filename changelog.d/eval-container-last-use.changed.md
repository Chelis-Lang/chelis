`chelis eval` and `chelis test` no longer copy a List or Dict that `append`,
`concat`, `dict_insert` or `dict_merge` receives from a variable read for the
last time; the write changes it in place. A `fold` or recursion that grows a
container one element per step used to copy the whole container at every step.
It now copies each element at most a constant number of times. A container that
is read again after the write is still copied, so every program computes what
it did before. Folds that append tensors to a `List[tensor[...]]` now cost time
linear in their length. Folds and recursions over a `List[i64]` copy nothing
per step, but each step still costs time that grows with the length of the
list (#3761). See [#3720](https://github.com/Chelis-Lang/chelis/issues/3720).

Applying a function value passed through a function-typed parameter no longer
normalizes that parameter's signature again on every call, and copying the
function value no longer copies those signatures. On the Nautilus `ode` and
`curvefit` test files, interpretation now costs 16% and 9% fewer instructions.
That recovers the slowdown #3629 introduced on these files. See
[#3741](https://github.com/Chelis-Lang/chelis/issues/3741).
