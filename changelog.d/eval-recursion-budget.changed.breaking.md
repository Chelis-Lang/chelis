`chelis eval` stops a recursion at 10,000 nested function applications.
Each application of a `def` or of a function value counts one level while it
runs, a callback that `fold`, `map` or `scan` applies included, and the
application that would begin level 10,001 reports `evaluation recursion budget
exhausted: more than 10000 nested function applications`, keeping the output
printed before it. The limit used to be wherever the native stack ran out:
about 68,000 levels for a simple recursion in a release build, about 11,000 in
a debug build, and less for wider bodies. It is now the same 10,000 in every
build and on every thread, so a recursion between 10,000 and 68,000 levels
deep that used to evaluate in a release build now fails. A runaway recursion
also stops with far less memory: about 108 MiB instead of 637 MiB for a
`List` recursion and 115 MiB instead of 672 MiB for a tensor one. A call the
evaluator runs as one compiled unit, such as a kernel, a named-axis reduction
or a `grad` or `vmap` target, counts one level like any other call; the
functions compiled into it are not counted, and the compiler's own inlining
limits bound them instead. The native stack check remains for recursive calls that sit inside deeply nested
expressions: calls nested up to 6 expressions deep reach the 10,000 limit in
every build, and up to 16 in a release build. See
[#3842](https://github.com/Chelis-Lang/chelis/issues/3842).
