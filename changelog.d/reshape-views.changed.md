A compiled product-sum kernel no longer copies its operands or its result at
the reshapes around it. When an `einsum` contraction flattens row labels, the
kernel reads the row operand in place through the flattening reshape and
stores each total directly in the output reshape's buffer, so it allocates
only its result, as the runtime kernel does. The same store applies to any
compiled `matmul` or product sum whose result is reshaped. A reshape's
element-count check and its trap order are unchanged.

A compiled f32 `amk,kn->amn` over `x: tensor[16384, 3, 256]` now peaks at
97.8 MiB instead of 145.8 MiB. With the copies gone, `einsum` rows whose
flattened extent is one runtime extent beside static ones, for example
`tensor[a, 3, k]`, also take the product-sum kernel: at a = 16384 that drops
from about 67.3e9 to 44.0e9 instructions with the same 97.8 MiB peak. Rows with
two or more runtime extents keep the runtime kernel. Results are bit-identical.
See [#3802](https://github.com/Chelis-Lang/chelis/issues/3802) and
[#3781](https://github.com/Chelis-Lang/chelis/issues/3781).
