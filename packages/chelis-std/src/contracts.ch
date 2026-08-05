module Std.Contracts
export (normal_cdf, normal_cdf_implementation_symbol, normal_cdf_range_contract, normal_cdf_reflection_contract, normal_cdf_monotonicity_contract, exp_positivity_contract, exp_monotonicity_contract, exp_zero_contract, log_monotonicity_contract, log_one_contract, standard_contract_tolerance, normal_cdf_contract_samples, normal_cdf_contract_seed)
def abs_f32(x: f32) -> f32 = if lt(x, cast(0.0, f32)) then neg(x) else x
def erf_approx(x: f32) -> f32 = {
  ax = abs_f32(x)
  if lt(ax, cast(0.00001, f32)) then mul(x, cast(1.1283791670955126, f32)) else {
    t = div(cast(1.0, f32), add(cast(1.0, f32), mul(cast(0.3275911, f32), ax)))
    poly = mul(t, add(cast(0.254829592, f32), mul(t, add(cast(-0.284496736, f32), mul(t, add(cast(1.421413741, f32), mul(t, add(cast(-1.453152027, f32), mul(t, cast(1.061405429, f32))))))))))
    y = sub(cast(1.0, f32), mul(poly, exp(neg(mul(ax, ax)))))
    if lt(x, cast(0.0, f32)) then neg(y) else y
  }
}
def erfc_approx(x: f32) -> f32 = sub(cast(1.0, f32), erf_approx(x))
def normal_cdf(x: f32) -> f32 = mul(cast(0.5, f32), erfc_approx(neg(mul(x, cast(0.7071067811865475, f32)))))
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
