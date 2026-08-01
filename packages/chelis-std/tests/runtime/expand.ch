module Std.Tests.Runtime.Expand
import Std.Test (assert_close_tensor)
def test_expand_replicates_1d_to_3x2() -> () ! { Test } = {
  b = to_tensor([cast(1.0, f32), cast(2.0, f32)])
  out = expand(b, cast(0, int32), cast(3, int32))
  flat = reshape(out, [cast(6, int64)])
  assert_close_tensor(flat, to_tensor([cast(1.0, f32), cast(2.0, f32), cast(1.0, f32), cast(2.0, f32), cast(1.0, f32), cast(2.0, f32)]), cast(1e-6, f32), "expand([1,2], 0, 3) replicates as 3 rows")
}
def test_expand_trailing_axis() -> () ! { Test } = {
  b = to_tensor([cast(1.0, f32), cast(2.0, f32)])
  out = expand(b, cast(1, int32), cast(2, int32))
  flat = reshape(out, [cast(4, int64)])
  assert_close_tensor(flat, to_tensor([cast(1.0, f32), cast(1.0, f32), cast(2.0, f32), cast(2.0, f32)]), cast(1e-6, f32), "expand([1,2], 1, 2) -> [[1,1],[2,2]]")
}
def test_expand_count_one_keeps_values() -> () ! { Test } = {
  b = to_tensor([cast(7.0, f32), cast(8.0, f32), cast(9.0, f32)])
  out = expand(b, cast(0, int32), cast(1, int32))
  flat = reshape(out, [cast(3, int64)])
  assert_close_tensor(flat, to_tensor([cast(7.0, f32), cast(8.0, f32), cast(9.0, f32)]), cast(1e-6, f32), "expand with count 1 preserves values")
}
def test_expand_linear_bias_pattern() -> () ! { Test } = {
  b = to_tensor([cast(10.0, f32), cast(100.0, f32)])
  expanded = expand(b, cast(0, int32), cast(2, int32))
  flat = reshape(expanded, [cast(4, int64)])
  assert_close_tensor(flat, to_tensor([cast(10.0, f32), cast(100.0, f32), cast(10.0, f32), cast(100.0, f32)]), cast(1e-6, f32), "Linear.forward bias-expand pattern")
}
