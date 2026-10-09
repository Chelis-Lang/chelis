A tensor whose element count, stride, or byte size does not fit `i64` is
now refused consistently across `chelis check`, `chelis eval` and
`chelis build`.

- A size proven from literals is a `DimensionMismatch` type error that names
  the overflowing count or byte size, so every lane rejects the program
  before it runs. Examples are `expand(x, 0i32, 4611686018427387904i64)` on
  `f32`, or a parameter declared `tensor[4611686018427387904, f32]`.
  Previously the checker accepted it, `chelis eval` panicked with
  `capacity overflow`, and `chelis build` failed with an internal live-byte
  bound error.
- A size computed at run time traps `Overflow` before allocation in both
  `chelis eval` and compiled C. For `expand`, `insert`, `pad` and `einsum`
  the two print the same report; other operations can render the trap
  differently between the lanes (for example `reshape`, `split_keys` and
  `pad_sequences`), which
  [#3436](https://github.com/Chelis-Lang/chelis/issues/3436) tracks.
  `reshape` refuses an empty target whose strides do not fit.
- A program whose tensors each fit, but whose total live bytes exceed 64
  bits, now builds.
- Where the host evaluator sizes a result itself (`expand`, `insert`, `pad`,
  the reductions and other movement operations, `concat`, `einsum`), a
  representable size the machine cannot allocate fails with C's
  `Domain: chelis_alloc tensor allocation failed`.

`split_keys`, and the operations `chelis eval` evaluates through the DAG
evaluator (such as `matmul`), can still abort the process on such a size;
see [#3435](https://github.com/Chelis-Lang/chelis/issues/3435). See
[#3418](https://github.com/Chelis-Lang/chelis/issues/3418).
