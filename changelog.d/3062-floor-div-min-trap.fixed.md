A compiled integer `floor_div` on tensors now traps
`numeric trap: overflow in floor_div at <dtype>` when the dtype's minimum is
divided by `-1`, at i8, i16, i32 and i64, as `chelis eval` does. It used to
evaluate C's undefined `MIN / -1` and print the minimum. A zero divisor now
prints `numeric trap: division by zero in floor_div at <dtype>` in the compiled
program too, instead of `integer division or remainder by zero`. See
[#3062](https://github.com/Chelis-Lang/chelis/issues/3062).
