`chelis eval` keeps less heap for each live function call in a deep
recursion. A call frame's bindings no longer reserve a full ordered-map node
of unboxed values, and a call drops its entry checks' working state before
its body runs instead of keeping it until the body returns. A recursion
level now keeps about 1.1 KB (an `i64` argument), 1.5 KB (`List[i64]`) or
2.5 KB (`tensor[2, f32]`) of heap, down from about 3.0, 3.4 and 6.2 KB. A
runaway recursion still stops at the same depth with
`evaluation stack budget exhausted during a function call`, using less
memory on the way: about 652 MB instead of 782 MB for the `List` case and
719 MB instead of 978 MB for the tensor case. See
[#3842](https://github.com/Chelis-Lang/chelis/issues/3842).
