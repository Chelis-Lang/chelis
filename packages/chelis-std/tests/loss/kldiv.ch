module Std.Tests.Loss.KlDiv
import Std.Loss.KlDiv (kl_divergence)
import Std.Test (assert_close, assert_true)
def expected_kl_signature[n](p: tensor[n, f32], q: tensor[n, f32]) -> f32 = kl_divergence(p, q)
def test_kl_against_self_is_zero() -> unit ! { Test } = {
  p = (to_tensor([0.5, 0.5]) : tensor[2, f32])
  q = (to_tensor([0.5, 0.5]) : tensor[2, f32])
  out = kl_divergence(p, q)
  assert_close(out, cast(0.0, f32), cast(0.000001, f32), "KL(p, p) == 0 for p = [0.5, 0.5]")
}
def test_kl_nonneg_for_distinct_distributions() -> unit ! { Test } = {
  p = (to_tensor([0.7, 0.3]) : tensor[2, f32])
  q = (to_tensor([0.5, 0.5]) : tensor[2, f32])
  out = kl_divergence(p, q)
  assert_true(gt(out, cast(0.0, f32)), "KL([0.7, 0.3] || [0.5, 0.5]) > 0")
}
def test_kl_zero_times_log_zero_is_zero() -> unit ! { Test } = {
  p = (to_tensor([1.0, 0.0]) : tensor[2, f32])
  q = (to_tensor([0.5, 0.5]) : tensor[2, f32])
  out = kl_divergence(p, q)
  expected = log(cast(2.0, f32))
  assert_close(out, expected, cast(0.000001, f32), "KL([1, 0] || [0.5, 0.5]) == log(2) (0*log(0) convention)")
}
def test_kl_is_asymmetric() -> unit ! { Test } = {
  p = (to_tensor([0.7, 0.3]) : tensor[2, f32])
  q = (to_tensor([0.5, 0.5]) : tensor[2, f32])
  kl_pq = kl_divergence(copy(p), copy(q))
  kl_qp = kl_divergence(q, p)
  diff = sub(kl_pq, kl_qp)
  abs_diff = if gt(cast(0.0, f32), diff) then sub(cast(0.0, f32), diff) else diff
  assert_true(gt(abs_diff, cast(0.0001, f32)), "|KL(p, q) - KL(q, p)| > 1e-4 for p = [0.7, 0.3], q = [0.5, 0.5]")
}
def test_kl_module_signature_pins_same_length_contract() -> unit ! { Test } = { assert_true(true, "Std.Loss.KlDiv kl_divergence keeps the [n](tensor[n], tensor[n]) -> f32 contract") }
