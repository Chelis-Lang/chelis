module Std.Tests.Contracts
import Std.Contracts (normal_cdf, normal_cdf_implementation_symbol, normal_cdf_reflection_contract, standard_contract_tolerance, normal_cdf_contract_samples, normal_cdf_contract_seed)
import Std.Test (assert_close, assert_eq_string, assert_eq_int, assert_true)
def test_normal_cdf_reflection_id() -> () ! { Test } = assert_eq_string(normal_cdf_reflection_contract(), "std.normal_cdf.reflection", "normal CDF reflection contract id")
def test_normal_cdf_implementation_symbol() -> () ! { Test } = assert_eq_string(normal_cdf_implementation_symbol(), "Std.Contracts.normal_cdf", "normal CDF implementation symbol")
def test_normal_cdf_reference_values() -> () ! { Test } = {
  _ = assert_close(normal_cdf(cast(0.0, f32)), cast(0.5, f32), cast(0.00001, f32), "N(0) = 0.5")
  _ = assert_close(normal_cdf(cast(1.0, f32)), cast(0.8413447, f32), cast(0.0002, f32), "N(1)")
  reflected = add(normal_cdf(cast(-1.0, f32)), normal_cdf(cast(1.0, f32)))
  assert_close(reflected, cast(1.0, f32), cast(0.0002, f32), "N(-x) + N(x) = 1")
}
def test_contract_fuzz_metadata() -> () ! { Test } = {
  _ = assert_eq_int(normal_cdf_contract_samples(), cast(8192, int64), "normal CDF fuzz sample count")
  expected_seed = add(mul(cast(3235848, int64), cast(1000, int64)), cast(230, int64))
  _ = assert_eq_int(normal_cdf_contract_seed(), expected_seed, "normal CDF fuzz seed")
  assert_true(gt(standard_contract_tolerance(), cast(0.0, f32)), "contract tolerance is positive")
}
