`chelis eval` keeps less heap for each live function call in a deep
recursion. A call frame's bindings no longer reserve a full ordered-map node
of unboxed values, a call drops its entry checks' working state before its
body runs, a static type read shares its declaration's type rather than
copying it, and a call no longer keeps its caller's claim list alive beside
its own. A recursion level now keeps about 1.14 KB (an `i64` argument),
1.21 KB (`List[i64]`) or 1.77 KB (`tensor[2, f32]`) of heap, down from about
3.0, 3.36 and 6.15 KB. A runaway recursion trips the stack budget about 1%
deeper, with the same message, `evaluation stack budget exhausted during a
function call`, and uses less memory on the way: about 611 MB instead of
788 MB for a `List` recursion and 649 MB instead of 1,001 MB for a tensor
one. See [#3842](https://github.com/Chelis-Lang/chelis/issues/3842).
