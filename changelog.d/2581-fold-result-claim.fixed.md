Compiled C now checks a tensor returned by `fold` against the function's
declared result extent and raises `numeric trap: domain in fold at i64` on a
mismatch, as `chelis eval` does. Previously the compiled program returned the
mismatched tensor, so the declared result type was false of the value. See
[#2581](https://github.com/Chelis-Lang/chelis/issues/2581).
