`chelis eval` and `chelis test` change a List or Dict in place when `append`,
`concat`, `dict_insert` or `dict_merge` receives it from a variable read for the
last time. A `fold` or recursion that grows a container one element per step
copied the whole container on every step and was quadratic in the step count;
it is now linear. A container that is read again after the write is still
copied, so every program computes what it did before. See
[#3720](https://github.com/Chelis-Lang/chelis/issues/3720).

Applying a function value passed through a function-typed parameter no longer
re-normalizes that parameter's signature on every call, which had made
interpretation 8 to 16% slower on the Nautilus test files. See
[#3741](https://github.com/Chelis-Lang/chelis/issues/3741).
