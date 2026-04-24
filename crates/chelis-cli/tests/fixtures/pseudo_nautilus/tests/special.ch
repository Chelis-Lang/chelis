module PseudoNautilus.Tests.Special

import PseudoNautilus.Special (erf_approx)
import Std.Test (assert_close, assert_true)

-- erf(0) must be zero. A-S 7.1.26 gives this exactly in theory; rounding
-- through f32 arithmetic leaves <1e-6 residual.
def test_erf_zero() -> unit ! { Test } = assert_close(erf_approx(cast(0.0, f32)), cast(0.0, f32), cast(0.00001, f32), "erf(0)")

-- Odd-function identity: erf(x) + erf(-x) == 0 for any x.
def test_erf_symmetry() -> unit ! { Test } = {
  x = cast(0.7, f32)
  pos = erf_approx(x)
  negv = erf_approx(sub(cast(0.0, f32), x))
  total = add(pos, negv)
  assert_close(total, cast(0.0, f32), cast(0.00001, f32), "erf-odd-identity")
}

-- Bounded by the unit interval: 0 < erf(3) < 1. The A-S approximation has
-- max abs error ~1.5e-7, so erf(3) lands safely inside (0, 1).
def test_erf_bounded() -> unit ! { Test } = {
  y = erf_approx(cast(3.0, f32))
  _ = assert_true(gt(y, cast(0.0, f32)), "erf(3) > 0")
  assert_true(gt(cast(1.0, f32), y), "erf(3) < 1")
}

-- Monotonicity on the positive side: erf is strictly increasing.
def test_erf_monotonic() -> unit ! { Test } = {
  lo = erf_approx(cast(0.3, f32))
  hi = erf_approx(cast(0.7, f32))
  assert_true(gt(hi, lo), "erf(0.7) > erf(0.3)")
}
