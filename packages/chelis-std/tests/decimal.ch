module Std.Tests.Decimal
import Std.Decimal (decimal, decimal_add, decimal_div, decimal_eq, decimal_from_int, decimal_gt, decimal_gte, decimal_lt, decimal_lte, decimal_mul, decimal_sub, decimal_to_float, decimal_to_string, round_half_even, try_decimal)
import Std.Test (assert_close, assert_eq, assert_false, assert_true)
def test_add_tenths_is_exact() -> unit ! { Test } = {
  sum = decimal_add(decimal("0.1"), decimal("0.2"))
  assert_true(decimal_eq(sum, decimal("0.3")), "0.1 + 0.2 == 0.3")
}
def test_add_from_int() -> unit ! { Test } = {
  lhs = decimal_from_int(cast(42, int64))
  rhs = decimal_from_int(cast(8, int64))
  expected = decimal_from_int(cast(50, int64))
  assert_true(decimal_eq(decimal_add(lhs, rhs), expected), "42 + 8 == 50")
}
def test_sub_inverts_add() -> unit ! { Test } = {
  a = decimal("1.25")
  b = decimal("0.75")
  recovered = decimal_sub(decimal_add(a, b), b)
  assert_true(decimal_eq(recovered, a), "(a + b) - b == a")
}
def test_mul_one_and_a_half_by_two() -> unit ! { Test } = {
  product = decimal_mul(decimal("1.5"), decimal("2.0"))
  assert_true(decimal_eq(product, decimal("3.0")), "1.5 * 2.0 == 3.0")
}
def test_ordering_lt_strict() -> unit ! { Test } = assert_true(decimal_lt(decimal("0.1"), decimal("0.2")), "0.1 < 0.2")
def test_ordering_gt_strict() -> unit ! { Test } = assert_true(decimal_gt(decimal("0.2"), decimal("0.1")), "0.2 > 0.1")
def test_ordering_lte_reflexive() -> unit ! { Test } = assert_true(decimal_lte(decimal("0.1"), decimal("0.1")), "0.1 <= 0.1")
def test_ordering_gte_reflexive() -> unit ! { Test } = assert_true(decimal_gte(decimal("0.1"), decimal("0.1")), "0.1 >= 0.1")
def test_ordering_lt_irreflexive() -> unit ! { Test } = assert_false(decimal_lt(decimal("0.1"), decimal("0.1")), "0.1 < 0.1 must be false")
def test_string_roundtrip() -> unit ! { Test } = {
  rendered = decimal_to_string(decimal("3.14"))
  assert_eq(rendered, "3.14", "decimal_to_string(decimal(\"3.14\")) == \"3.14\"")
}
def test_string_roundtrip_negative() -> unit ! { Test } = assert_eq(decimal_to_string(decimal("-2.50")), "-2.5", "negative trailing-zero normalises")
def test_string_roundtrip_zero() -> unit ! { Test } = assert_eq(decimal_to_string(decimal("0")), "0", "zero renders as 0")
def test_to_float_half_exact() -> unit ! { Test } = {
  approx = decimal_to_float(decimal("0.5"))
  assert_close(cast(approx, f32), cast(0.5, f32), cast(0.0, f32), "decimal(0.5) -> f32 0.5 exactly")
}
def test_try_decimal_rejects_garbage() -> unit ! { Test } =
  match try_decimal("not_a_number") with {
    | Some(_) => assert_true(false, "try_decimal(\"not_a_number\") must return None")
    | None => assert_true(true, "try_decimal(\"not_a_number\") returns None")
  }
def test_div_one_by_three_round_half_even() -> unit ! { Test } = {
  q = decimal_div(decimal("1"), decimal("3"), cast(0, int64), round_half_even())
  assert_true(decimal_eq(q, decimal_from_int(cast(0, int64))), "1 / 3 @ scale 0, half-even == 0")
}
def test_div_five_by_two_half_even_ties_to_even() -> unit ! { Test } = {
  q = decimal_div(decimal_from_int(cast(5, int64)), decimal_from_int(cast(2, int64)), cast(0, int64), round_half_even())
  assert_true(decimal_eq(q, decimal_from_int(cast(2, int64))), "5 / 2 half-even == 2 (banker's)")
}
def test_mul_scale_accumulation_value() -> unit ! { Test } = assert_true(decimal_eq(decimal_mul(decimal("0.1"), decimal("0.1")), decimal("0.01")), "0.1 * 0.1 == 0.01 (value)")
def test_mul_scale_accumulation_string() -> unit ! { Test } = assert_eq(decimal_to_string(decimal_mul(decimal("0.1"), decimal("0.1"))), "0.01", "decimal_to_string normalises 0.1*0.1 to \"0.01\"")
