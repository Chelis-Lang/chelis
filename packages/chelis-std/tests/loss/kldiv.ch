module Std.Tests.Loss.KlDiv
import Std.Loss.KlDiv (kl_divergence)
import Std.Test (assert_close, assert_true)
-- Acceptance surface for `Std.Loss.KlDiv` at Phase 3t A1 Wave 2.
--
-- The shipped signature is
--   def kl_divergence[n](p: tensor[n, f32], q: tensor[n, f32]) -> f32
-- so a runtime length-mismatch case (the `fail("kl_divergence: p and q have
-- different lengths")` branch in the body) is not reachable through any
-- type-checked Chelis source: a call where `p` and `q` have different
-- statically-known sizes is rejected by the dim-var unifier before any
-- evaluator code runs. That static rejection is pinned by the CLI test
-- `phase3j_pre_batch4_kl_divergence_rejects_shape_mismatch` in
-- `crates/chelis-cli/tests/phase3j_pre_std_batch4.rs`, which compiles a
-- driver where `p : tensor[2, f32]` and `q : tensor[3, f32]` are passed
-- to `kl_divergence` and asserts `chelis eval --file` exits with failure.
-- Below, `expected_kl_signature` is a typed wrapper that re-states the
-- same-length contract: if the published signature drifts and the dim
-- var stops unifying, this def stops compiling and the whole file fails
-- with a single `<file>` row, surfacing the regression.

def expected_kl_signature[n](p: tensor[n, f32], q: tensor[n, f32]) -> f32 = kl_divergence(p, q)

def test_kl_against_self_is_zero() -> unit ! { Test } = {
  -- KL(p || p) = sum(p * (log p - log p)) = sum(p * 0) = 0 for any
  -- distribution p. Use the uniform binary distribution p = [0.5, 0.5].
  p = (to_tensor([0.5, 0.5]) : tensor[2, f32])
  q = (to_tensor([0.5, 0.5]) : tensor[2, f32])
  out = kl_divergence(p, q)
  assert_close(out, cast(0.0, f32), cast(0.000001, f32), "KL(p, p) == 0 for p = [0.5, 0.5]")
}

def test_kl_nonneg_for_distinct_distributions() -> unit ! { Test } = {
  -- Gibbs' inequality: KL(p || q) >= 0 for all valid distributions, with
  -- equality iff p == q. With p = [0.7, 0.3] and q = [0.5, 0.5] (both
  -- proper, p != q), the divergence must be strictly positive. This is a
  -- weak-but-useful sanity check that the sum-of-terms has the right
  -- sign overall (a sign flip or swapped argument order would surface
  -- here).
  p = (to_tensor([0.7, 0.3]) : tensor[2, f32])
  q = (to_tensor([0.5, 0.5]) : tensor[2, f32])
  out = kl_divergence(p, q)
  assert_true(gt(out, cast(0.0, f32)), "KL([0.7, 0.3] || [0.5, 0.5]) > 0")
}

def test_kl_zero_times_log_zero_is_zero() -> unit ! { Test } = {
  -- The 0 * log(0) = 0 convention is implemented in the body as
  --   if eq(pair.0, 0.0) then 0.0 else pair.0 * (log pair.0 - log pair.1)
  -- For p = [1.0, 0.0], q = [0.5, 0.5]:
  --   term0 = 1.0 * (log 1.0 - log 0.5) = -log 0.5 = log 2
  --   term1 = (0 * (log 0 - log 0.5)) -> short-circuited to 0 by the
  --           convention (without the guard, log 0 would produce -inf
  --           and the result would be NaN, blowing past any tolerance).
  -- So KL = log 2. Compute the reference value via the same `log`
  -- builtin so this test pins the convention, not a hard-coded constant
  -- that could mask a `log` regression.
  p = (to_tensor([1.0, 0.0]) : tensor[2, f32])
  q = (to_tensor([0.5, 0.5]) : tensor[2, f32])
  out = kl_divergence(p, q)
  expected = log(cast(2.0, f32))
  assert_close(out, expected, cast(0.000001, f32), "KL([1, 0] || [0.5, 0.5]) == log(2) (0*log(0) convention)")
}

def test_kl_is_asymmetric() -> unit ! { Test } = {
  -- KL is not a metric: KL(p, q) != KL(q, p) in general. Use the same
  -- non-uniform pair as the nonneg test so the asymmetry must be
  -- numerically large enough to clear a generous tolerance, since
  -- exact-not-equal on floats is fragile. With p = [0.7, 0.3] and
  -- q = [0.5, 0.5], the two divergences differ by roughly 0.05 nats,
  -- well above the 1e-4 tolerance below.
  p = (to_tensor([0.7, 0.3]) : tensor[2, f32])
  q = (to_tensor([0.5, 0.5]) : tensor[2, f32])
  kl_pq = kl_divergence(copy(p), copy(q))
  kl_qp = kl_divergence(q, p)
  diff = sub(kl_pq, kl_qp)
  abs_diff = if gt(cast(0.0, f32), diff) then sub(cast(0.0, f32), diff) else diff
  assert_true(gt(abs_diff, cast(0.0001, f32)), "|KL(p, q) - KL(q, p)| > 1e-4 for p = [0.7, 0.3], q = [0.5, 0.5]")
}

def test_kl_module_signature_pins_same_length_contract() -> unit ! { Test } = {
  -- The file-level compile pass type-checks `expected_kl_signature`
  -- above. If the shipped `kl_divergence` signature drifts off
  -- `[n](tensor[n, f32], tensor[n, f32]) -> f32`, that wrapper stops
  -- compiling and `chelis test` aborts with a `<file>` compile failure
  -- before this test runs. Reaching this body therefore witnesses that
  -- the same-length, scalar-result contract still holds; the runtime
  -- mismatch branch is unreachable through type-checked source and is
  -- pinned at the CLI level (see file header).
  assert_true(true, "Std.Loss.KlDiv kl_divergence keeps the [n](tensor[n], tensor[n]) -> f32 contract")
}
