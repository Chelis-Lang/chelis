module Std.Contracts
export (normal_cdf, normal_cdf_implementation_symbol, normal_cdf_range_contract, normal_cdf_reflection_contract, normal_cdf_monotonicity_contract, exp_positivity_contract, exp_monotonicity_contract, exp_zero_contract, log_monotonicity_contract, log_one_contract, standard_contract_tolerance, normal_cdf_contract_samples, normal_cdf_contract_seed)
def normal_cdf[p: Float](x: p) -> p = standard_normal_cdf(x)
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
