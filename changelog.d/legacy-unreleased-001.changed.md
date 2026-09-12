**ReLU now retains its dedicated [05-OP-43] identity and adjoint
(chelis#1313).** Reverse-mode AD returns the complete incoming cotangent
only where `0 < x`, and exact positive zero at both signed zeros and NaN;
direct `max_elem`/`min_elem` keep their first-operand tie rule. Eval and the
C backend implement exact f16/bf16/f32/f64 stored-bit selection, HIP does so
through all-width kernels, and Metal admits f32/f16/bf16 while retaining its
target-wide f64 rejection. WireDag v7 carries `Relu` and `ReluAdjoint` as
distinct validated identities.
