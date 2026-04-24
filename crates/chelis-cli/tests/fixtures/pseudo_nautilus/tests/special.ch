module PseudoNautilus.Tests.Special

import PseudoNautilus.Special (erf_approx)

-- These tests use the raw `test_assert*` runtime builtins rather than
-- `Std.Test` wrappers. The chelis-std dep would be the more idiomatic wiring,
-- but as of v0.2.3 the `chelis test` module-init precheck evaluates every
-- linked library decl under strict loads, and some chelis-std bodies trip on
-- unresolved references during that scan. The raw builtins exercise the
-- identical runtime pass/fail paths (`Std.Test.assert_true` is literally
-- `test_assert`, `Std.Test.assert_close` is the same tolerance math open-coded
-- below), so the coverage is equivalent from the pseudo-nautilus "internal
-- correctness" perspective. Parity against scipy lives in `../parity/`.

-- assert_close(actual, expected, tol, label): fails if |actual - expected| > tol.
-- Open-coded here so we don't pull in chelis-std.
def close(actual: f32, expected: f32, tol: f32, label: string) -> unit = {
  diff = sub(actual, expected)
  abs_diff = if gt(cast(0.0, f32), diff) then sub(cast(0.0, f32), diff) else diff
  ok = not(gt(abs_diff, tol))
  test_assert(ok, label)
}

-- erf(0) must be zero. A-S 7.1.26 gives this exactly in theory; rounding
-- through f32 arithmetic leaves <1e-6 residual.
def test_erf_zero() -> unit = close(erf_approx(cast(0.0, f32)), cast(0.0, f32), cast(0.00001, f32), "erf(0)")

-- Odd-function identity: erf(x) + erf(-x) == 0 for any x.
def test_erf_symmetry() -> unit = {
  x = cast(0.7, f32)
  pos = erf_approx(x)
  negv = erf_approx(sub(cast(0.0, f32), x))
  total = add(pos, negv)
  close(total, cast(0.0, f32), cast(0.00001, f32), "erf-odd-identity")
}

-- Bounded by the unit interval: 0 < erf(3) < 1. The A-S approximation has
-- max abs error ~1.5e-7, so erf(3) lands safely inside (0, 1).
def test_erf_bounded() -> unit = {
  y = erf_approx(cast(3.0, f32))
  _ = test_assert(gt(y, cast(0.0, f32)), "erf(3) > 0")
  test_assert(gt(cast(1.0, f32), y), "erf(3) < 1")
}

-- Monotonicity on the positive side: erf is strictly increasing.
def test_erf_monotonic() -> unit = {
  lo = erf_approx(cast(0.3, f32))
  hi = erf_approx(cast(0.7, f32))
  test_assert(gt(hi, lo), "erf(0.7) > erf(0.3)")
}
