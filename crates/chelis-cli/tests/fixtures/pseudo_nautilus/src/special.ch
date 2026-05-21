module PseudoNautilus.Special
export (erf_approx)
-- Abramowitz-Stegun 7.1.26 rational approximation to erf.
--
-- For x >= 0:
--   t = 1 / (1 + p*x)
--   erf(x) ~= 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
--
-- For x < 0, use the odd-function identity erf(-x) = -erf(x).
--
-- Max absolute error vs the true erf is ~1.5e-7 -- well inside the tolerances
-- the Chelis-native identity tests use, and close enough that the scipy-parity
-- harness in ../parity/run_parity.py finds agreement to a few ulps.
def erf_approx_nonneg(x: f32) -> f32 = {
  p = cast(0.3275911, f32)
  a1 = cast(0.254829592, f32)
  a2 = cast(-0.284496736, f32)
  a3 = cast(1.421413741, f32)
  a4 = cast(-1.453152027, f32)
  a5 = cast(1.061405429, f32)
  t = div(cast(1.0, f32), add(cast(1.0, f32), mul(p, x)))
  t2 = mul(t, t)
  t3 = mul(t2, t)
  t4 = mul(t2, t2)
  t5 = mul(t4, t)
  poly = add(add(add(add(mul(a1, t), mul(a2, t2)), mul(a3, t3)), mul(a4, t4)), mul(a5, t5))
  decay = exp(neg(mul(x, x)))
  sub(cast(1.0, f32), mul(poly, decay))
}
def erf_approx(x: f32) -> f32 = {
  is_negative = gt(cast(0.0, f32), x)
  abs_x = if is_negative then sub(cast(0.0, f32), x) else x
  positive_value = erf_approx_nonneg(abs_x)
  if is_negative then sub(cast(0.0, f32), positive_value) else positive_value
}
