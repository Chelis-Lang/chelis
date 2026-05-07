module Std.Tests.Loss.Bce
import Std.Loss.Bce (bce_with_logits)
import Std.Test (assert_close_tensor, assert_true)
def expected_bce_shape[n](z: tensor[n, f32], y: tensor[n, f32]) -> tensor[n, f32] = bce_with_logits(z, y)
def test_bce_at_zero_is_log_two() -> unit ! { Test } = {
  z = to_tensor([cast(0.0, f32), cast(0.0, f32)])
  y = to_tensor([cast(0.0, f32), cast(0.0, f32)])
  actual = bce_with_logits(z, y)
  log_two = log(cast(2.0, f32))
  expected = to_tensor([log_two, log_two])
  assert_close_tensor(actual, expected, cast(0.000001, f32), "BCE(z=0, y=0) == log(2) elementwise")
}
def test_bce_correct_positive_saturates_to_zero() -> unit ! { Test } = {
  z = to_tensor([cast(10.0, f32)])
  y = to_tensor([cast(1.0, f32)])
  actual = bce_with_logits(z, y)
  expected = to_tensor([cast(0.0, f32)])
  assert_close_tensor(actual, expected, cast(0.001, f32), "BCE(z=10, y=1) saturates near 0 (correct positive prediction)")
}
def test_bce_correct_negative_saturates_to_zero() -> unit ! { Test } = {
  z = to_tensor([cast(-10.0, f32)])
  y = to_tensor([cast(0.0, f32)])
  actual = bce_with_logits(z, y)
  expected = to_tensor([cast(0.0, f32)])
  assert_close_tensor(actual, expected, cast(0.001, f32), "BCE(z=-10, y=0) saturates near 0 (correct negative prediction)")
}
def test_bce_wrong_positive_penalty_grows_with_z() -> unit ! { Test } = {
  z = to_tensor([cast(-10.0, f32)])
  y = to_tensor([cast(1.0, f32)])
  actual = bce_with_logits(z, y)
  expected = to_tensor([cast(10.0, f32)])
  assert_close_tensor(actual, expected, cast(0.001, f32), "BCE(z=-10, y=1) ≈ 10 (wrong-direction confident prediction)")
}
def test_bce_typechecks_with_shared_dim_variable() -> unit ! { Test } = { assert_true(true, "Std.Loss.Bce.bce_with_logits shared-n shape contract type-checks") }
