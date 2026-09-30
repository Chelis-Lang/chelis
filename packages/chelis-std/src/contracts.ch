module Std.Contracts
export (normal_cdf, normal_cdf_implementation_symbol, normal_cdf_range_contract, normal_cdf_reflection_contract, normal_cdf_monotonicity_contract, exp_positivity_contract, exp_monotonicity_contract, exp_zero_contract, log_monotonicity_contract, log_one_contract, standard_contract_tolerance, normal_cdf_contract_samples, normal_cdf_contract_seed)
def abs_float[p: Float](x: p) -> p = if lt(x, cast(0.0, p)) then neg(x) else x
def erf_approx[p: Float](x: p) -> p = {
  ax = abs_float(x)
  if lt(ax, cast(0.00001, p)) then mul(x, cast(1.1283791670955126, p)) else {
    t = 1.0 |> cast(p) |> div(add(cast(1.0, p), mul(cast(0.3275911, p), ax)))
    poly = mul(t, add(cast(0.254829592, p), mul(t, add(cast(-0.284496736, p), mul(t, add(cast(1.421413741, p), mul(t, add(cast(-1.453152027, p), mul(t, cast(1.061405429, p))))))))))
    y = 1.0 |> cast(p) |> sub(mul(poly, exp(neg(mul(ax, ax)))))
    if lt(x, cast(0.0, p)) then neg(y) else y
  }
}
def normal_cdf[p: Float](x: p) -> p = mul(cast(0.5, p), sub(cast(1.0, p), erf_approx(neg(mul(x, cast(0.7071067811865475, p))))))
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
def normal_cdf_contract_samples() -> i64 = cast(8192, i64)
def normal_cdf_contract_seed() -> i64 = add(mul(cast(3235848, i64), cast(1000, i64)), cast(230, i64))
