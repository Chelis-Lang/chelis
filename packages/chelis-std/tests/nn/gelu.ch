module Std.Tests.Nn.Gelu
import Std.Nn.Gelu (forward, gelu_scalar, tanh_scalar)
import Std.Test (assert_close, assert_close_tensor)
def test_gelu_zero_is_zero() -> unit ! { Test } = {
  out = gelu_scalar(cast(0.0, f32))
  assert_close(out, cast(0.0, f32), cast(0.000001, f32), "gelu(0) == 0")
}
def test_gelu_large_positive_saturates_to_identity() -> unit ! { Test } = {
  x = cast(5.0, f32)
  out = gelu_scalar(x)
  assert_close(out, x, cast(0.001, f32), "gelu(5) ~= 5 (saturates to identity for large +x)")
}
def test_gelu_large_negative_saturates_to_zero() -> unit ! { Test } = {
  out = gelu_scalar(cast(-5.0, f32))
  assert_close(out, cast(0.0, f32), cast(0.001, f32), "gelu(-5) ~= 0 (saturates to 0 for large -x)")
}
def test_forward_is_elementwise_gelu_scalar() -> unit ! { Test } = {
  xs = to_tensor([cast(1.0, f32), cast(-1.0, f32), cast(2.0, f32)])
  actual = forward(xs)
  expected = to_tensor([gelu_scalar(cast(1.0, f32)), gelu_scalar(cast(-1.0, f32)), gelu_scalar(cast(2.0, f32))])
  assert_close_tensor(actual, expected, cast(0.000001, f32), "forward(x) == map(gelu_scalar, x) elementwise")
}
def test_tanh_scalar_zero_is_zero() -> unit ! { Test } = {
  out = tanh_scalar(cast(0.0, f32))
  assert_close(out, cast(0.0, f32), cast(0.000001, f32), "tanh(0) == 0")
}
def test_tanh_scalar_saturates_to_plus_one_and_minus_one() -> unit ! { Test } = {
  hi = tanh_scalar(cast(10.0, f32))
  lo = tanh_scalar(cast(-10.0, f32))
  _ = assert_close(hi, cast(1.0, f32), cast(0.0001, f32), "tanh(10) ~= 1")
  assert_close(lo, cast(-1.0, f32), cast(0.0001, f32), "tanh(-10) ~= -1")
}
