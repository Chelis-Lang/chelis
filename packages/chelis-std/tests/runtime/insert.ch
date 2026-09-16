module Std.Tests.Runtime.Insert
import Std.Test (assert_close_tensor)
def test_insert_replicates_1d_to_3x2() -> unit ! { Test } = {
  b = to_tensor([cast(1.0, f32), cast(2.0, f32)])
  out = insert(b, cast(0, i32), cast(3, i64))
  flat = reshape(out, [cast(6, i64)])
  assert_close_tensor(flat, to_tensor([cast(1.0, f32), cast(2.0, f32), cast(1.0, f32), cast(2.0, f32), cast(1.0, f32), cast(2.0, f32)]), cast(1e-6, f32), "insert([1,2], 0, 3) replicates as 3 rows")
}
def test_insert_trailing_axis() -> unit ! { Test } = {
  b = to_tensor([cast(1.0, f32), cast(2.0, f32)])
  out = insert(b, cast(1, i32), cast(2, i64))
  flat = reshape(out, [cast(4, i64)])
  assert_close_tensor(flat, to_tensor([cast(1.0, f32), cast(1.0, f32), cast(2.0, f32), cast(2.0, f32)]), cast(1e-6, f32), "insert([1,2], 1, 2) -> [[1,1],[2,2]]")
}
def test_insert_count_one_keeps_values() -> unit ! { Test } = {
  b = to_tensor([cast(7.0, f32), cast(8.0, f32), cast(9.0, f32)])
  out = insert(b, cast(0, i32), cast(1, i64))
  flat = reshape(out, [cast(3, i64)])
  assert_close_tensor(flat, to_tensor([cast(7.0, f32), cast(8.0, f32), cast(9.0, f32)]), cast(1e-6, f32), "insert with count 1 preserves values")
}
def test_insert_linear_bias_pattern() -> unit ! { Test } = {
  b = to_tensor([cast(10.0, f32), cast(100.0, f32)])
  expanded = insert(b, cast(0, i32), cast(2, i64))
  flat = reshape(expanded, [cast(4, i64)])
  assert_close_tensor(flat, to_tensor([cast(10.0, f32), cast(100.0, f32), cast(10.0, f32), cast(100.0, f32)]), cast(1e-6, f32), "Linear.forward bias-expand pattern")
}
