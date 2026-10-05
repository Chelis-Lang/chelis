A multi-axis named `sum` over an i8 or i16 tensor, such as
`sum(x, seq, head)`, returns its i32 total in eval and in compiled C. Each
stage used to cast its total back to the operand dtype, so eval trapped on
overflow and the build failed with a tensor helper type mismatch.
