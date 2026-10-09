A tensor whose element count or byte size does not fit `i64` now fails the
same way in every lane. A size proven from literals, such as
`expand(x, 0i32, 4611686018427387904i64)` on `f32` or a parameter declared
`tensor[4611686018427387904, f32]`, is a `DimensionMismatch` type error that
names the overflowing count or byte size, so `chelis check`, `chelis eval`
and `chelis build` all reject the program before it runs. Previously the
checker accepted it, `chelis eval` panicked with `capacity overflow`, and
`chelis build` failed with an internal live-byte bound error. A size
computed at run time traps `numeric trap: overflow in <op> at i64` before
allocation in `chelis eval`, as compiled C already did, and a representable
size the machine cannot allocate fails with C's allocation failure in both.
A program whose tensors each fit but whose total live bytes exceed 64 bits
now builds. See [#3418](https://github.com/Chelis-Lang/chelis/issues/3418).
