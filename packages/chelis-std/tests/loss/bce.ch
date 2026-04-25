module Std.Tests.Loss.Bce
import Std.Loss.Bce (bce_with_logits)
import Std.Test (assert_close_tensor, assert_true)
-- Acceptance surface for `Std.Loss.Bce` at Phase 3t A1 Wave 2.
--
-- Unlike the `Std.Loss.CrossEntropy` companion file, BCE compiles to host
-- runtime: its body is `to_list` + `map` + scalar `log`/`exp`, all of which
-- are implemented in the host runtime that backs `chelis test`. So this
-- file pins both the shape contract (file-compile typed wrapper) and the
-- numeric contract (4 runtime tests against identities derivable from the
-- numerically-stable BCE form).
--
-- Numeric form pinned (numerically-stable, see `src/loss/bce.ch`):
--
--     BCE(z, y) = max(z, 0) - z*y + log(1 + exp(-|z|))
--
-- which agrees mathematically with the standard `softplus(z) - z*y` form.
--
-- Shape-mismatch path: `bce_with_logits[n](z: tensor[n, f32],
-- y: tensor[n, f32])` rejects mismatched lengths at the type-checker level
-- via the shared `n` dimension variable. The corresponding negative test
-- lives in `crates/chelis-cli/tests/phase3j_pre_std_batch4.rs::
-- phase3j_pre_batch4_bce_rejects_shape_mismatch`, which asserts that
-- `chelis eval --file` fails on a mismatched program. Putting the same
-- negative inline here would cascade the entire test file to a single
-- `<file>` compile failure (per the test-runner contract in
-- `crates/chelis-cli/src/main.rs::run_test_file`), so the static-rejection
-- contract is pinned by the typed wrapper below — if `bce_with_logits`
-- ever drifts away from `tensor[n, f32] -> tensor[n, f32] -> tensor[n, f32]`
-- (e.g. relaxes the shared `n`), this file stops compiling.
def expected_bce_shape[n](z: tensor[n, f32], y: tensor[n, f32]) -> tensor[n, f32] = bce_with_logits(z, y)
def test_bce_at_zero_is_log_two() -> unit ! { Test } = {
  -- BCE(z=0, y=0) = max(0,0) - 0*0 + log(1 + exp(-0)) = 0 + log(2). Both
  -- elements should equal log(2). Compare against an expected tensor whose
  -- entries are computed via Chelis's own log/cast pipeline so we are not
  -- pinning an external numeric lookup, only the BCE invariant.
  z = to_tensor([cast(0.0, f32), cast(0.0, f32)])
  y = to_tensor([cast(0.0, f32), cast(0.0, f32)])
  actual = bce_with_logits(z, y)
  log_two = log(cast(2.0, f32))
  expected = to_tensor([log_two, log_two])
  assert_close_tensor(actual, expected, cast(0.000001, f32), "BCE(z=0, y=0) == log(2) elementwise")
}
def test_bce_correct_positive_saturates_to_zero() -> unit ! { Test } = {
  -- BCE(z=10, y=1) = max(10,0) - 10*1 + log(1 + exp(-10))
  --                = 10 - 10 + log(1 + tiny) ≈ 0.
  -- The residual is `log(1 + exp(-10)) ≈ 4.54e-5`, well inside 1e-3.
  z = to_tensor([cast(10.0, f32)])
  y = to_tensor([cast(1.0, f32)])
  actual = bce_with_logits(z, y)
  expected = to_tensor([cast(0.0, f32)])
  assert_close_tensor(actual, expected, cast(0.001, f32), "BCE(z=10, y=1) saturates near 0 (correct positive prediction)")
}
def test_bce_correct_negative_saturates_to_zero() -> unit ! { Test } = {
  -- BCE(z=-10, y=0) = max(-10,0) - (-10)*0 + log(1 + exp(-|-10|))
  --                 = 0 - 0 + log(1 + exp(-10)) ≈ 0.
  -- Same residual as the y=1 mirror; symmetric saturation.
  z = to_tensor([cast(-10.0, f32)])
  y = to_tensor([cast(0.0, f32)])
  actual = bce_with_logits(z, y)
  expected = to_tensor([cast(0.0, f32)])
  assert_close_tensor(actual, expected, cast(0.001, f32), "BCE(z=-10, y=0) saturates near 0 (correct negative prediction)")
}
def test_bce_wrong_positive_penalty_grows_with_z() -> unit ! { Test } = {
  -- BCE(z=-10, y=1) = max(-10,0) - (-10)*1 + log(1 + exp(-10))
  --                 = 0 + 10 + log(1 + exp(-10)) ≈ 10.
  -- The model said "very confident negative" but the label is positive, so
  -- the loss is essentially -z = 10. Tolerance covers the softplus residual.
  z = to_tensor([cast(-10.0, f32)])
  y = to_tensor([cast(1.0, f32)])
  actual = bce_with_logits(z, y)
  expected = to_tensor([cast(10.0, f32)])
  assert_close_tensor(actual, expected, cast(0.001, f32), "BCE(z=-10, y=1) ≈ 10 (wrong-direction confident prediction)")
}
def test_bce_typechecks_with_shared_dim_variable() -> unit ! { Test } = {
  -- The file-level compile pass type-checks `expected_bce_shape` above. If
  -- the `bce_with_logits` signature drifts away from sharing the same `n`
  -- across z, y, and the result, `chelis test` aborts with a `<file>`
  -- compile failure before this test runs. Reaching this body therefore
  -- witnesses that the shared-`n` dimension contract is still in force,
  -- which is what makes mismatched-shape calls a static type error (see
  -- `phase3j_pre_batch4_bce_rejects_shape_mismatch` for the external
  -- runner-level pinning of the rejection).
  assert_true(true, "Std.Loss.Bce.bce_with_logits shared-n shape contract type-checks")
}
