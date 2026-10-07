`chelis eval` builds flat list literals with bounded native stack depth and
linear element-copy cost, so lists with thousands of values evaluate instead
of exhausting the stack. The same path handles lists passed to `to_tensor`.
See [#906](https://github.com/Chelis-Lang/chelis/issues/906) and
[#1630](https://github.com/Chelis-Lang/chelis/issues/1630).
