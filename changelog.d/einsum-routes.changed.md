More compiled `einsum` contractions run on the emitted product-sum kernel
instead of the runtime kernel:

- f16 and bf16 operands at their default f32 accumulator, and any float operand
  at an explicit `accumulator=f64`. The kernel converts each operand element as
  it reads it and rounds each total once to the result dtype as it stores it,
  so it allocates no accumulator-width copy of an operand or of the result.
  Compiled f16 and bf16 `matmul` stores its result the same way.
- Row labels whose flattened extent is a runtime extent beside a static one, or
  several runtime extents, for example `amk,kn->amn` over `tensor[a, 3, k]`.

Results are bit-identical to `chelis eval`. A bf16 `amk,kn->amn` at
`[1,64,4096] x [4096,4096]` drops from about 32.3e9 to 14.3e9 instructions with
the same peak memory. See
[#3781](https://github.com/Chelis-Lang/chelis/issues/3781) and
[#3783](https://github.com/Chelis-Lang/chelis/issues/3783).
