module Std.Tests.Nn.RmsNorm
import Std.Nn.RmsNorm (forward)
import Std.Test (assert_close_tensor, assert_shape)
def test_uniform_input_yields_unit_output() -> unit ! { Test } = {
  c = cast(2.0, f32)
  x = to_tensor([c, c, c])
  unit_gain = to_tensor([cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)])
  out = forward(x, unit_gain, cast(0.0, f32))
  expected = to_tensor([cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)])
  assert_close_tensor(out, expected, cast(0.000001, f32), "RMS(c)/c = 1 for c=2.0 with unit gain")
}
def test_zero_input_yields_zero_output() -> unit ! { Test } = {
  zero = cast(0.0, f32)
  x = to_tensor([zero, zero, zero])
  unit_gain = to_tensor([cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)])
  out = forward(x, unit_gain, cast(0.000001, f32))
  expected = to_tensor([zero, zero, zero])
  assert_close_tensor(out, expected, cast(0.0, f32), "zero input with eps>0 must produce exact zero output")
}
def test_weight_scaling_is_linear() -> unit ! { Test } = {
  x = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)])
  unit_gain = to_tensor([cast(1.0, f32), cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)])
  scaled_gain = to_tensor([cast(2.0, f32), cast(2.0, f32), cast(2.0, f32), cast(2.0, f32)])
  eps = cast(0.000001, f32)
  out_unit = forward(copy(x), unit_gain, eps)
  out_scaled = forward(x, scaled_gain, eps)
  doubled = to_tensor(map(fn (v: f32) -> mul(cast(2.0, f32), v), to_list(out_unit)))
  assert_close_tensor(out_scaled, doubled, cast(0.000001, f32), "forward(x, 2*w, eps) == 2 * forward(x, w, eps)")
}
def test_shape_matches_input() -> unit ! { Test } = {
  x = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32), cast(5.0, f32)])
  unit_gain = to_tensor([cast(1.0, f32), cast(1.0, f32), cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)])
  out = forward(x, unit_gain, cast(0.000001, f32))
  assert_shape(out, cast(5, int64), "rank-1 RmsNorm preserves input length")
}
def test_unit_gain_matches_closed_form_on_one_two_three_four() -> unit ! { Test } = {
  x = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)])
  unit_gain = to_tensor([cast(1.0, f32), cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)])
  out = forward(x, unit_gain, cast(0.0, f32))
  -- mean(x^2) = (1 + 4 + 9 + 16) / 4 = 7.5; expected[i] = x[i] / sqrt(7.5).
  inv_rms = div(cast(1.0, f32), sqrt(cast(7.5, f32)))
  expected = to_tensor([mul(cast(1.0, f32), inv_rms), mul(cast(2.0, f32), inv_rms), mul(cast(3.0, f32), inv_rms), mul(cast(4.0, f32), inv_rms)])
  assert_close_tensor(out, expected, cast(0.000001, f32), "forward([1,2,3,4], unit_gain, 0) == [1,2,3,4] / sqrt(7.5)")
}
