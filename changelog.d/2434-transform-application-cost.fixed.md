A `grad` or `vmap` application in the interpreter no longer re-validates every
definition in the program. The pipe fold rebuilds only the nodes above a pipe
it folds, and a definition table with no pipe in it is shared rather than
copied when a lowering context is prepared. #2427 had made each application in
a package that includes the standard library roughly twice as slow. See
[#2434](https://github.com/Chelis-Lang/chelis/issues/2434).
