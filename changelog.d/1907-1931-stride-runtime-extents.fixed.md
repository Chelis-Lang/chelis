Eval and compiled C now reject runtime zero or negative `stride` steps with
the same typed domain failure, and non-unit strides enforce declared result
extents at the operation with the exact numeric-trap context. See
[#1907](https://github.com/Chelis-Lang/chelis/issues/1907) and
[#1931](https://github.com/Chelis-Lang/chelis/issues/1931).
