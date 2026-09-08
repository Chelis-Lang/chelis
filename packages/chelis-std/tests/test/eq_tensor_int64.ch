module Std.Tests.Test.EqTensorInt64
import Std.Test (assert_eq_tensor, fail)
def test_eq_tensor_int64_basic() -> unit ! { Test } = {
  actual = to_tensor([cast(0, int64), cast(1, int64), cast(2, int64)])
  expected = to_tensor([cast(0, int64), cast(1, int64), cast(2, int64)])
  assert_eq_tensor(actual, expected, "[0,1,2] equals itself")
}
def test_eq_tensor_int64_negatives_and_zero() -> unit ! { Test } = {
  actual = to_tensor([cast(-3, int64), cast(0, int64), cast(7, int64)])
  expected = to_tensor([cast(-3, int64), cast(0, int64), cast(7, int64)])
  assert_eq_tensor(actual, expected, "negative and zero values compare exactly")
}
def test_eq_tensor_int64_singleton() -> unit ! { Test } = {
  actual = to_tensor([cast(42, int64)])
  expected = to_tensor([cast(42, int64)])
  assert_eq_tensor(actual, expected, "single-element int64 tensor")
}
