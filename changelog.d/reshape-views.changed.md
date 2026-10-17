A compiled product-sum kernel no longer copies its operands or its result at
the reshapes around it. When an `einsum` contraction flattens row labels, the
kernel reads the row operand in place through the flattening reshape and
stores each total directly in the output reshape's buffer. The same store
applies to any compiled `matmul` or product sum whose result is reshaped in the
same compiled function. A reshape's element-count check and its trap order are
unchanged, and results are bit-identical.

A compiled f32 `amk,kn->amn` over `x: tensor[16384, 3, 256]` now peaks at
97.8 MiB instead of 145.8 MiB, and an f16 one over `tensor[65536, 3, 64]` at
354.1 MB instead of 379.3 MB. `einsum` rows whose flattened extent is a
runtime extent beside a static one still use the runtime kernel. See
[#3802](https://github.com/Chelis-Lang/chelis/issues/3802).
