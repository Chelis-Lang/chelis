`chelis eval` keeps less heap for each live function call in a deep
recursion, and evaluates blocks with many local bindings faster. A call
frame's bindings no longer reserve a full ordered-map node of unboxed values,
a call drops its entry checks' working state before its body runs, a static
type read shares its declaration's type rather than copying it, and a call no
longer keeps its caller's claim list alive beside its own. A recursion level
now keeps about 1.24 KB (an `i64` argument), 1.31 KB (`List[i64]`) or
1.86 KB (`tensor[2, f32]`) of heap, down from about 3.0, 3.36 and 6.15 KB. A
`let` or `match` arm now shares the bindings it saves instead of copying
them, so a function whose body binds 40 locals in sequence takes about 23%
fewer instructions per call, and one that binds 100 about 33% fewer. In a
release build a runaway recursion trips the stack budget about 2% deeper,
with the same message, `evaluation stack budget exhausted during a function
call`, and uses less memory on the way: about 637 MB instead of 788 MB for a
`List` recursion and 672 MB instead of 1,001 MB for a tensor one. See
[#3842](https://github.com/Chelis-Lang/chelis/issues/3842).
