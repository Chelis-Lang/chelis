`conv` and `layer_norm` now compile when an operand extent, a stride, or a
padding amount is known only at run time. A dimension-generic def that applies
`conv` with a declared return type, a generic `layer_norm` with or without a
declared return type, `grad` through either, and `conv` with `i64` strides or
padding passed as arguments used to fail at compile time with "requires a
concrete extent" or "checked conv requires literal per-axis metadata". They
now evaluate and build to the same values as their literal-shaped twins. A
stride that is not positive, a negative padding amount, or a kernel larger
than its padded input stops the program with the same message in `chelis eval`
and in the built executable. With `--target hip` or `--target metal`, a
`layer_norm` whose hidden extent is known only at run time is refused with a
message that names `--target c`. See
[#3379](https://github.com/Chelis-Lang/chelis/issues/3379) and
[#3380](https://github.com/Chelis-Lang/chelis/issues/3380).
