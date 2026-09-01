module Std.Contracts
export (normal_cdf, normal_cdf_implementation_symbol, normal_cdf_range_contract, normal_cdf_reflection_contract, normal_cdf_monotonicity_contract, exp_positivity_contract, exp_monotonicity_contract, exp_zero_contract, log_monotonicity_contract, log_one_contract, standard_contract_tolerance, normal_cdf_contract_samples, normal_cdf_contract_seed)
def abs_float[p: Float](x: p) -> p = if lt(x, 0.0) then neg(x) else x
def erf_approx[p: Float](x: p) -> p = {
  ax = abs_float(x)
  if lt(ax, 0.00001) then mul(x, 1.1283791670955126) else {
    t = div(1.0, add(1.0, mul(0.3275911, ax)))
    poly = mul(t, add(0.254829592, mul(t, add(-0.284496736, mul(t, add(1.421413741, mul(t, add(-1.453152027, mul(t, 1.061405429)))))))))
    y = sub(1.0, mul(poly, exp(neg(mul(ax, ax)))))
    if lt(x, 0.0) then neg(y) else y
  }
}
def normal_cdf[p: Float](x: p) -> p = mul(0.5, sub(1.0, erf_approx(neg(mul(x, 0.7071067811865475)))))
def normal_cdf_implementation_symbol() -> string = "Std.Contracts.normal_cdf"
def normal_cdf_range_contract() -> string = "std.normal_cdf.range"
def normal_cdf_reflection_contract() -> string = "std.normal_cdf.reflection"
def normal_cdf_monotonicity_contract() -> string = "std.normal_cdf.monotonicity"
def exp_positivity_contract() -> string = "std.exp.positivity"
def exp_monotonicity_contract() -> string = "std.exp.monotonicity"
def exp_zero_contract() -> string = "std.exp.zero"
def log_monotonicity_contract() -> string = "std.log.monotonicity"
def log_one_contract() -> string = "std.log.one"
def standard_contract_tolerance() -> f32 = cast(1e-10, f32)
def normal_cdf_contract_samples() -> int64 = cast(8192, int64)
def normal_cdf_contract_seed() -> int64 = add(mul(cast(3235848, int64), cast(1000, int64)), cast(230, int64))
