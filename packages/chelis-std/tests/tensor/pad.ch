module Std.Tests.Tensor.Pad
import Std.Test (assert_close_tensor, assert_eq)
def test_pad_uses_supplied_fill_value() -> unit ! { Test } = {
  padded = pad_sequences_to([[cast(5.0, f32)]], cast(3, int64), cast(99.0, f32))
  flat = reshape(padded, [cast(3, int64)])
  expected = to_tensor([cast(5.0, f32), cast(99.0, f32), cast(99.0, f32)])
  assert_close_tensor(flat, expected, cast(1e-6, f32), "pad fill 99.0 -> [5, 99, 99] (catches fill->1.0/0.0/ignored)")
}
def test_pad_ragged_rows_fills_independently() -> unit ! { Test } = {
  padded = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32)]], cast(3, int64), cast(7.0, f32))
  flat = reshape(padded, [cast(6, int64)])
  expected = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(7.0, f32), cast(3.0, f32), cast(7.0, f32), cast(7.0, f32)])
  assert_close_tensor(flat, expected, cast(1e-6, f32), "ragged [[1,2],[3]] padded to width 3 with fill 7")
}
def test_pad_truncates_when_width_lt_row_len() -> unit ! { Test } = {
  padded = pad_sequences_to([[cast(10.0, f32), cast(20.0, f32), cast(30.0, f32), cast(40.0, f32)]], cast(2, int64), cast(0.0, f32))
  flat = reshape(padded, [cast(2, int64)])
  expected = to_tensor([cast(10.0, f32), cast(20.0, f32)])
  assert_close_tensor(flat, expected, cast(1e-6, f32), "width=2 truncates row of length 4")
}
def test_pad_int64_uses_supplied_fill_value() -> unit ! { Test } = {
  padded = pad_sequences_to([[cast(5, int64)]], cast(3, int64), cast(7, int64))
  flat = reshape(padded, [cast(3, int64)])
  xs = to_list(flat)
  _ = assert_eq(cast(index(xs, cast(0, int64)), int64), cast(5, int64), "int64 pad [0] = 5")
  _ = assert_eq(cast(index(xs, cast(1, int64)), int64), cast(7, int64), "int64 pad [1] = 7 (fill)")
  assert_eq(cast(index(xs, cast(2, int64)), int64), cast(7, int64), "int64 pad [2] = 7 (fill)")
}
