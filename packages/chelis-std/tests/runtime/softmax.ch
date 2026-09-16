module Std.Tests.Runtime.Softmax
import Std.Test (assert_close, assert_close_tensor)
def test_softmax_uniform_zeros_is_uniform() -> unit ! { Test } = {
  x = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
  out = softmax(x, cast(0, i32))
  third = cast(0.3333333, f32)
  assert_close_tensor(out, to_tensor([third, third, third]), cast(0.001, f32), "softmax([0,0,0]) = [1/3,1/3,1/3]")
}
def test_softmax_two_class_matches_reference() -> unit ! { Test } = {
  x = to_tensor([cast(1.0, f32), cast(0.0, f32)])
  out = softmax(x, cast(0, i32))
  assert_close_tensor(out, to_tensor([cast(0.7310585, f32), cast(0.2689414, f32)]), cast(0.001, f32), "softmax([1,0]) ≈ [0.731, 0.269]")
}
def test_softmax_large_inputs_remain_stable() -> unit ! { Test } = {
  x = to_tensor([cast(1000.0, f32), cast(1000.0, f32)])
  out = softmax(x, cast(0, i32))
  assert_close_tensor(out, to_tensor([cast(0.5, f32), cast(0.5, f32)]), cast(0.001, f32), "softmax([1000,1000]) is uniform 0.5 (numerical stability)")
}
def test_softmax_normalizes_to_one() -> unit ! { Test } = {
  x = to_tensor([cast(2.0, f32), cast(1.0, f32), cast(0.5, f32)])
  out = softmax(x, cast(0, i32))
  total = tensor_to_scalar(sum(out, cast(0, i32)))
  assert_close(total, cast(1.0, f32), cast(0.001, f32), "softmax probabilities sum to 1")
}
