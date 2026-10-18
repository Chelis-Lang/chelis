`chelis eval`'s einsum now feeds each output's products into the
adjacent-pair sum tree as it forms them, as compiled code does, instead of
storing every product of the output first. Memory no longer grows with the
reduction: `einsum("i,j->", x, y)` over two 4096-element `i64` vectors drops
from about 529 MB peak RSS to 17 MB. An output no longer reserves a buffer
the size of its reduction, so a reduction too large to store is computed
rather than refused. Results are bit-identical, and an integer overflow still
reports `overflow in einsum` at the accumulator dtype. See
[#3807](https://github.com/Chelis-Lang/chelis/issues/3807).
