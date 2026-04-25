module Std.Tests.Runtime.Sum
import Std.Test (assert_close_tensor, assert_close)
def test_sum_axis1_2x3() -> unit ! { Test } = {
  mat = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
  out = sum(mat, cast(1, int32))
  assert_close_tensor(out, to_tensor([cast(6.0, f32), cast(15.0, f32)]), cast(0.000001, f32), "sum-axis1-2x3=[6,15]")
}
def test_sum_axis0_2x3() -> unit ! { Test } = {
  mat = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
  out = sum(mat, cast(0, int32))
  assert_close_tensor(out, to_tensor([cast(5.0, f32), cast(7.0, f32), cast(9.0, f32)]), cast(0.000001, f32), "sum-axis0-2x3=[5,7,9]")
}
def test_sum_singleton_row() -> unit ! { Test } = {
  mat = pad_sequences_to([[cast(7.0, f32)]], cast(1, int64), cast(0.0, f32))
  out = sum(mat, cast(1, int32))
  assert_close_tensor(out, to_tensor([cast(7.0, f32)]), cast(0.000001, f32), "sum-[7]=[7]")
}
def test_sum_handles_negatives() -> unit ! { Test } = {
  mat = pad_sequences_to([[cast(-1.0, f32), cast(2.0, f32), cast(-3.0, f32)]], cast(3, int64), cast(0.0, f32))
  out = sum(mat, cast(1, int32))
  assert_close_tensor(out, to_tensor([cast(-2.0, f32)]), cast(0.000001, f32), "sum-[-1,2,-3]=-2")
}
