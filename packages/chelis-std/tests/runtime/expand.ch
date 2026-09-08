module Std.Tests.Runtime.Expand
import Std.Test (assert_close_tensor)
def test_expand_broadcasts_a_unit_axis() -> unit ! { Test } = {
  b = to_tensor([[cast(1.0, f32)], [cast(2.0, f32)]])
  out = expand(b, cast(1, int32), cast(3, int64))
  flat = reshape(out, [cast(6, int64)])
  assert_close_tensor(flat, to_tensor([cast(1.0, f32), cast(1.0, f32), cast(1.0, f32), cast(2.0, f32), cast(2.0, f32), cast(2.0, f32)]), cast(1e-6, f32), "expand([[1],[2]], 1, 3) repeats each row across the unit axis")
}
def test_expand_leading_unit_axis() -> unit ! { Test } = {
  b = to_tensor([[cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]])
  out = expand(b, cast(0, int32), cast(2, int64))
  flat = reshape(out, [cast(6, int64)])
  assert_close_tensor(flat, to_tensor([cast(4.0, f32), cast(5.0, f32), cast(6.0, f32), cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]), cast(1e-6, f32), "expand([[4,5,6]], 0, 2) repeats the single row")
}
def test_expand_to_one_is_the_identity() -> unit ! { Test } = {
  b = to_tensor([[cast(7.0, f32)], [cast(8.0, f32)]])
  out = expand(b, cast(1, int32), cast(1, int64))
  flat = reshape(out, [cast(2, int64)])
  assert_close_tensor(flat, to_tensor([cast(7.0, f32), cast(8.0, f32)]), cast(1e-6, f32), "expanding a unit axis to 1 preserves values")
}
def test_expand_bias_broadcast_pattern() -> unit ! { Test } = {
  b = to_tensor([[cast(10.0, f32), cast(100.0, f32)]])
  bias = expand(b, cast(0, int32), cast(2, int64))
  rows = to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]])
  flat = reshape(add(rows, bias), [cast(4, int64)])
  assert_close_tensor(flat, to_tensor([cast(11.0, f32), cast(102.0, f32), cast(13.0, f32), cast(104.0, f32)]), cast(1e-6, f32), "a row bias broadcast down the batch axis")
}
