module Std.Tests.Nn.Silu
import Std.Nn.Silu (forward, sigmoid_scalar)
import Std.Test (assert_close, assert_close_tensor)
def test_silu_zero_is_zero() -> unit ! { Test } = {
  x = cast(0.0, f32)
  out = mul(x, sigmoid_scalar(x))
  assert_close(out, cast(0.0, f32), cast(0.000001, f32), "silu(0) == 0 * sigmoid(0) == 0")
}
def test_sigmoid_zero_is_one_half() -> unit ! { Test } = {
  out = sigmoid_scalar(cast(0.0, f32))
  assert_close(out, cast(0.5, f32), cast(0.000001, f32), "sigmoid(0) == 1 / (1 + exp(0)) == 0.5")
}
def test_sigmoid_pins_closed_form_values_at_minus3_and_plus3() -> unit ! { Test } = {
  s_neg = sigmoid_scalar(cast(-3.0, f32))
  s_pos = sigmoid_scalar(cast(3.0, f32))
  _ = assert_close(s_neg, cast(0.04742587, f32), cast(0.0001, f32), "sigmoid(-3) ≈ 0.0474")
  assert_close(s_pos, cast(0.95257413, f32), cast(0.0001, f32), "sigmoid(3) ≈ 0.9526")
}
def test_silu_large_positive_saturates_to_identity() -> unit ! { Test } = {
  x = cast(10.0, f32)
  out = mul(x, sigmoid_scalar(x))
  assert_close(out, x, cast(0.01, f32), "silu(10) ~= 10 (sigmoid saturates to 1, so x * sigmoid(x) -> x; residual is x*(1-sigmoid(x)))")
}
def test_silu_large_negative_saturates_to_zero() -> unit ! { Test } = {
  x = cast(-10.0, f32)
  out = mul(x, sigmoid_scalar(x))
  assert_close(out, cast(0.0, f32), cast(0.01, f32), "silu(-10) ~= 0 (sigmoid saturates to 0, so x * sigmoid(x) -> 0)")
}
def test_forward_is_elementwise_x_times_sigmoid() -> unit ! { Test } = {
  xs = to_tensor([cast(1.0, f32), cast(-1.0, f32)])
  actual = forward(xs)
  v0 = cast(1.0, f32)
  v1 = cast(-1.0, f32)
  expected = to_tensor([mul(v0, sigmoid_scalar(v0)), mul(v1, sigmoid_scalar(v1))])
  assert_close_tensor(actual, expected, cast(0.000001, f32), "forward(x) == x * sigmoid(x) elementwise")
}
