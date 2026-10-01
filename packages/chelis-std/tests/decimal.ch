module Std.Tests.Decimal
import Std.Decimal (Decimal, decimal, try_decimal, decimal_to_string, decimal_to_fixed_string, decimal_from_i64, decimal_to_i64, try_decimal_to_i64, decimal_from_f64, try_decimal_from_f64, decimal_to_f64, decimal_to_f32, decimal_scale, decimal_add, decimal_sub, decimal_mul, decimal_round, decimal_div, try_decimal_div, decimal_lt, decimal_lte, decimal_gt, decimal_gte)
import Std.Rounding (Rounding, RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)
import Std.Test (assert_eq, assert_true, assert_false)
-- Self-tests for [05-OP-76]. Expected values come from exact rational
-- arithmetic (Python's fractions.Fraction); every rejection is checked
-- through a `try_` twin, which returns None exactly where its twin fails
-- `domain`.
def i64_minimum() -> i64 = sub(-9223372036854775807i64, 1i64)
def zeros(count: i64) -> string = fold(fn (acc: string, unused: i64) -> string_concat(acc, "0"), "", range(0i64, count))
def shown(value: Option[Decimal]) -> string =
  match value with {
    | Some(d) => decimal_to_string(d)
    | None => "none"
  }
def shown_int(value: Option[i64]) -> string =
  match value with {
    | Some(v) => to_string(v)
    | None => "none"
  }
def parsed(text: string) -> string = shown(try_decimal(text))
def rounded(text: string, places: i64, mode: Rounding) -> string = decimal_to_string(decimal_round(decimal(text), places, mode))
def divided(a: string, b: string, places: i64, mode: Rounding) -> string = shown(try_decimal_div(decimal(a), decimal(b), places, mode))
def narrowed(text: string, mode: Rounding) -> string = shown_int(try_decimal_to_i64(decimal(text), mode))
def ingested(value: f64, places: i64, mode: Rounding) -> string = shown(try_decimal_from_f64(value, places, mode))
def single(text: string) -> f64 = cast(decimal_to_f32(decimal(text)), f64)
-- The value set and canonical form.
def test_canonical_form_drops_trailing_zeros() -> unit ! { Test } = {
  _ = assert_eq(parsed("1.50"), "1.5", "1.50 is 1.5")
  _ = assert_eq(decimal_scale(decimal("1.50")), 1i64, "1.50 has scale 1")
  _ = assert_eq(parsed("100"), "100", "integers keep their zeros")
  _ = assert_eq(decimal_scale(decimal("100")), 0i64, "100 has scale 0")
  _ = assert_eq(parsed("12.340000"), "12.34", "12.340000 is 12.34")
  _ = assert_eq(parsed("1.000"), "1", "1.000 is 1")
  _ = assert_eq(decimal_scale(decimal("1.000")), 0i64, "1.000 has scale 0")
  assert_eq(decimal("1.50"), decimal("1.5"), "one value has one representation")
}
def test_canonical_form_keeps_distinct_values_distinct() -> unit ! { Test } = {
  _ = assert_eq(parsed("1.05"), "1.05", "1.05 keeps its interior zero")
  _ = assert_eq(decimal_scale(decimal("1.05")), 2i64, "1.05 has scale 2")
  _ = assert_true(decimal_lt(decimal("1.5"), decimal("1.51")), "1.5 is below 1.51")
  assert_false(decimal_lt(decimal("1.50"), decimal("1.5")), "1.50 is not below 1.5")
}
def test_zero_is_unsigned_with_scale_zero() -> unit ! { Test } = {
  _ = assert_eq(parsed("-0"), "0", "-0 is zero")
  _ = assert_eq(parsed("-0.000"), "0", "-0.000 is zero")
  _ = assert_eq(decimal_scale(decimal("0.000")), 0i64, "zero has scale 0")
  _ = assert_eq(decimal("-0.0"), decimal("0"), "one zero")
  _ = assert_eq(decimal_to_string(decimal_mul(decimal("-1.5"), decimal("0"))), "0", "a zero product is unsigned")
  _ = assert_eq(decimal_to_string(decimal_add(decimal("-1.5"), decimal("1.5"))), "0", "a zero sum is unsigned")
  _ = assert_eq(rounded("-0.4", 0i64, RoundTowardZero), "0", "rounding to zero is unsigned")
  assert_eq(ingested(neg(0.0f64), 2i64, RejectInexact), "0", "negative zero ingests as zero")
}
def test_nonzero_values_keep_their_sign() -> unit ! { Test } = {
  _ = assert_eq(parsed("-0.4"), "-0.4", "-0.4 keeps its sign")
  _ = assert_eq(rounded("-0.4", 0i64, RoundTowardNegative), "-1", "-0.4 down is -1")
  assert_eq(decimal_to_string(decimal_mul(decimal("-1.5"), decimal("2"))), "-3", "a negative product")
}
-- The envelope: |c| <= 10^38 - 1 and 0 <= s <= 38.
def test_envelope_bounds_are_accepted() -> unit ! { Test } = {
  _ = assert_eq(parsed("99999999999999999999999999999999999999"), "99999999999999999999999999999999999999", "10^38 - 1")
  _ = assert_eq(parsed("-99999999999999999999999999999999999999"), "-99999999999999999999999999999999999999", "-(10^38 - 1)")
  _ = assert_eq(parsed("0.00000000000000000000000000000000000001"), "0.00000000000000000000000000000000000001", "10^-38")
  _ = assert_eq(decimal_scale(decimal("1e-38")), 38i64, "10^-38 has scale 38")
  _ = assert_eq(parsed("-1e-38"), "-0.00000000000000000000000000000000000001", "-10^-38")
  _ = assert_eq(parsed("0.99999999999999999999999999999999999999"), "0.99999999999999999999999999999999999999", "38 digits at scale 38")
  _ = assert_eq(parsed("9.9999999999999999999999999999999999999e37"), "99999999999999999999999999999999999999", "38 digits through an exponent")
  assert_eq(parsed("1e37"), "10000000000000000000000000000000000000", "10^37")
}
def test_envelope_rejects_values_outside() -> unit ! { Test } = {
  _ = assert_eq(parsed("100000000000000000000000000000000000000"), "none", "10^38")
  _ = assert_eq(parsed("1e38"), "none", "10^38 through an exponent")
  _ = assert_eq(parsed("-1e38"), "none", "-10^38")
  _ = assert_eq(parsed("1e-39"), "none", "10^-39")
  _ = assert_eq(parsed("0.000000000000000000000000000000000000001"), "none", "39 fractional digits")
  _ = assert_eq(parsed("1.00000000000000000000000000000000000001"), "none", "39 significant digits")
  assert_eq(parsed("0.000000000000000000000000000000000000015"), "none", "a scale-39 value")
}
def test_removable_zeros_are_accepted() -> unit ! { Test } = {
  _ = assert_eq(parsed("9223372036854775807.0"), "9223372036854775807", "the i64 maximum with a zero fraction")
  _ = assert_eq(parsed("99999999999999999999999999999999999999.000"), "99999999999999999999999999999999999999", "the maximum with a zero fraction")
  _ = assert_eq(parsed("0.100000000000000000000000000000000000000000000000"), "0.1", "48 fractional digits, one significant")
  _ = assert_eq(parsed("10000000000000000000000000000000000000e-38"), "0.1", "38 integer digits scaled down")
  assert_eq(parsed("0.0001e41"), "10000000000000000000000000000000000000", "a fraction scaled up to 10^37")
}
def test_removable_zeros_do_not_rescue_out_of_range_values() -> unit ! { Test } = {
  _ = assert_eq(parsed("100000000000000000000000000000000000000.0"), "none", "10^38 with a zero fraction")
  _ = assert_eq(parsed("0.0001e42"), "none", "a fraction scaled up to 10^38")
  assert_eq(parsed("10000e-43"), "none", "10^-39 written with zeros")
}
-- Parsing: one RFC 8259 number token of at most 1000 scalars.
def test_parser_accepts_number_tokens() -> unit ! { Test } = {
  _ = assert_eq(parsed("0"), "0", "zero")
  _ = assert_eq(parsed("-123.456"), "-123.456", "a negative fraction")
  _ = assert_eq(parsed("1e5"), "100000", "a lowercase exponent")
  _ = assert_eq(parsed("1E5"), "100000", "an uppercase exponent")
  _ = assert_eq(parsed("1e+5"), "100000", "a signed exponent")
  _ = assert_eq(parsed("1.5e-3"), "0.0015", "a negative exponent")
  _ = assert_eq(parsed("12.5E+1"), "125", "a fraction times ten")
  _ = assert_eq(parsed("1e0005"), "100000", "exponent leading zeros")
  _ = assert_eq(parsed("1e-0000000000000000000000000000000000000002"), "0.01", "many exponent leading zeros")
  _ = assert_eq(parsed("-0.0e5"), "0", "a signed zero with an exponent")
  _ = assert_eq(parsed("0e99999999999999999999"), "0", "zero with a huge exponent")
  _ = assert_eq(parsed("0e-9223372036854775808"), "0", "zero with the i64 minimum exponent")
  assert_eq(parsed("0.00e-99999999999999999999"), "0", "zero with a huge negative exponent")
}
def test_parser_rejects_other_text() -> unit ! { Test } = {
  _ = assert_eq(parsed("+1"), "none", "a plus sign")
  _ = assert_eq(parsed("01"), "none", "a leading zero")
  _ = assert_eq(parsed("00"), "none", "a doubled zero")
  _ = assert_eq(parsed("-01.5"), "none", "a signed leading zero")
  _ = assert_eq(parsed(".5"), "none", "a bare fraction")
  _ = assert_eq(parsed("-.5"), "none", "a signed bare fraction")
  _ = assert_eq(parsed("1."), "none", "a bare point")
  _ = assert_eq(parsed("1.e5"), "none", "a bare point before an exponent")
  _ = assert_eq(parsed(" 1"), "none", "leading whitespace")
  _ = assert_eq(parsed("1 "), "none", "trailing whitespace")
  _ = assert_eq(parsed("\t1"), "none", "a leading tab")
  _ = assert_eq(parsed("1e"), "none", "an empty exponent")
  _ = assert_eq(parsed("1e+"), "none", "a signed empty exponent")
  _ = assert_eq(parsed("e5"), "none", "no significand")
  _ = assert_eq(parsed("NaN"), "none", "NaN")
  _ = assert_eq(parsed("inf"), "none", "inf")
  _ = assert_eq(parsed("-Infinity"), "none", "-Infinity")
  _ = assert_eq(parsed("-"), "none", "a bare sign")
  _ = assert_eq(parsed(""), "none", "empty text")
  _ = assert_eq(parsed("1.5.5"), "none", "two points")
  _ = assert_eq(parsed("1_000"), "none", "a digit separator")
  _ = assert_eq(parsed("1,5"), "none", "a decimal comma")
  _ = assert_eq(parsed("0x10"), "none", "hexadecimal")
  _ = assert_eq(parsed("1e5.5"), "none", "a fractional exponent")
  _ = assert_eq(parsed("1e5e5"), "none", "two exponents")
  _ = assert_eq(parsed("--1"), "none", "two signs")
  _ = assert_eq(parsed("١"), "none", "an Arabic-Indic digit")
  assert_eq(parsed("１"), "none", "a fullwidth digit")
}
def test_parser_length_bound() -> unit ! { Test } = {
  _ = assert_eq(string_len(string_concat("1.", zeros(998i64))), 1000i64, "the probe has 1000 characters")
  _ = assert_eq(parsed(string_concat("1.", zeros(998i64))), "1", "1000 characters are accepted")
  _ = assert_eq(parsed(string_concat("-0.", zeros(997i64))), "0", "a 1000-character zero")
  assert_eq(parsed(string_concat("0.1", zeros(997i64))), "0.1", "1000 characters with one significant digit")
}
def test_parser_rejects_over_long_text() -> unit ! { Test } = {
  _ = assert_eq(parsed(string_concat("1.", zeros(999i64))), "none", "1001 characters")
  _ = assert_eq(parsed(string_concat("0.", zeros(999i64))), "none", "a 1001-character zero")
  _ = assert_eq(parsed(string_concat("1", zeros(999i64))), "none", "1000 characters naming 10^999")
  assert_eq(parsed(string_concat(string_concat("0.", zeros(997i64)), "1")), "none", "1000 characters naming 10^-998")
}
def test_parser_exponent_extremes() -> unit ! { Test } = {
  _ = assert_eq(parsed("1e-9223372036854775808"), "none", "the i64 minimum exponent")
  _ = assert_eq(parsed("1e9223372036854775807"), "none", "the i64 maximum exponent")
  _ = assert_eq(parsed("1e99999999999999999999"), "none", "an exponent past i64")
  _ = assert_eq(parsed("1e-99999999999999999999"), "none", "a negative exponent past i64")
  _ = assert_eq(parsed("1e-38"), "0.00000000000000000000000000000000000001", "the smallest exponent that fits")
  assert_eq(parsed("1e00000000000000000000000000000000000000000000037"), "10000000000000000000000000000000000000", "a padded exponent that fits")
}
-- Text rendering.
def test_to_string_is_canonical() -> unit ! { Test } = {
  _ = assert_eq(decimal_to_string(decimal("5e-1")), "0.5", "a leading integer zero")
  _ = assert_eq(decimal_to_string(decimal("-5e-3")), "-0.005", "fractional zeros")
  _ = assert_eq(decimal_to_string(decimal("1234567890.123456789")), "1234567890.123456789", "a limb boundary")
  _ = assert_eq(decimal_to_string(decimal("1000000000")), "1000000000", "10^9")
  _ = assert_eq(decimal_to_string(decimal("999999999")), "999999999", "10^9 - 1")
  assert_eq(decimal_to_string(decimal("1000000000000000000.000000001")), "1000000000000000000.000000001", "zero limbs inside")
}
def test_to_string_round_trips() -> unit ! { Test } = {
  _ = assert_eq(decimal(decimal_to_string(decimal("-123.456"))), decimal("-123.456"), "a negative fraction")
  _ = assert_eq(decimal(decimal_to_string(decimal("1e-38"))), decimal("1e-38"), "10^-38")
  _ = assert_eq(decimal(decimal_to_string(decimal("-99999999999999999999999999999999999999"))), decimal("-99999999999999999999999999999999999999"), "the minimum")
  assert_eq(decimal(decimal_to_string(decimal("0"))), decimal("0"), "zero")
}
def test_to_fixed_string_pads() -> unit ! { Test } = {
  _ = assert_eq(decimal_to_fixed_string(decimal("1.5"), 2i64), "1.50", "1.50")
  _ = assert_eq(decimal_to_fixed_string(decimal("1"), 3i64), "1.000", "an integer gains a point")
  _ = assert_eq(decimal_to_fixed_string(decimal("0"), 2i64), "0.00", "zero")
  _ = assert_eq(decimal_to_fixed_string(decimal("-0.5"), 1i64), "-0.5", "no padding needed")
  _ = assert_eq(decimal_to_fixed_string(decimal("12"), 0i64), "12", "scale zero")
  _ = assert_eq(decimal_to_fixed_string(decimal("1e-38"), 38i64), "0.00000000000000000000000000000000000001", "scale 38")
  assert_eq(decimal_to_fixed_string(decimal("7"), 38i64), string_concat("7.", zeros(38i64)), "38 zeros")
}
def test_to_fixed_string_never_rounds() -> unit ! { Test } = {
  _ = assert_eq(decimal_to_fixed_string(decimal("1.25"), 2i64), "1.25", "the exact scale")
  assert_false(eq(decimal_to_fixed_string(decimal("1.25"), 3i64), "1.25"), "padding is not truncation")
}
-- Integers.
def test_from_i64_is_exact() -> unit ! { Test } = {
  _ = assert_eq(decimal_to_string(decimal_from_i64(0i64)), "0", "zero")
  _ = assert_eq(decimal_to_string(decimal_from_i64(-1i64)), "-1", "-1")
  _ = assert_eq(decimal_to_string(decimal_from_i64(1000000000i64)), "1000000000", "10^9")
  _ = assert_eq(decimal_to_string(decimal_from_i64(9223372036854775807i64)), "9223372036854775807", "the i64 maximum")
  _ = assert_eq(decimal_to_string(decimal_from_i64(i64_minimum())), "-9223372036854775808", "the i64 minimum")
  assert_eq(decimal_from_i64(-42i64), decimal("-42.000"), "an integer value")
}
def test_to_i64_round_trips_the_boundaries() -> unit ! { Test } = {
  _ = assert_eq(decimal_to_i64(decimal_from_i64(9223372036854775807i64), RejectInexact), 9223372036854775807i64, "the i64 maximum")
  _ = assert_eq(decimal_to_i64(decimal_from_i64(i64_minimum()), RejectInexact), i64_minimum(), "the i64 minimum")
  _ = assert_eq(decimal_to_i64(decimal("9223372036854775807.0"), RejectInexact), 9223372036854775807i64, "a removable zero")
  _ = assert_eq(narrowed("9223372036854775807.4", RoundTowardZero), "9223372036854775807", "truncation toward the maximum")
  _ = assert_eq(narrowed("-9223372036854775808.5", RoundTiesToEven), "-9223372036854775808", "a tie rounds to the even minimum")
  _ = assert_eq(narrowed("-9223372036854775808.9", RoundTowardZero), "-9223372036854775808", "truncation toward the minimum")
  assert_eq(narrowed("9223372036854775806.5", RoundTiesToEven), "9223372036854775806", "a tie below the maximum")
}
def test_to_i64_rejects_outside_i64_and_inexact() -> unit ! { Test } = {
  _ = assert_eq(narrowed("9223372036854775808", RoundTiesToEven), "none", "the i64 maximum plus one")
  _ = assert_eq(narrowed("-9223372036854775809", RoundTiesToEven), "none", "the i64 minimum minus one")
  _ = assert_eq(narrowed("9223372036854775807.5", RoundTiesToEven), "none", "a tie rounding past the maximum")
  _ = assert_eq(narrowed("9223372036854775807.1", RoundTowardPositive), "none", "rounding up past the maximum")
  _ = assert_eq(narrowed("-9223372036854775808.5", RoundTiesToAway), "none", "a tie rounding past the minimum")
  _ = assert_eq(narrowed("99999999999999999999999999999999999999", RoundTowardZero), "none", "the decimal maximum")
  _ = assert_eq(narrowed("1.5", RejectInexact), "none", "an inexact value")
  assert_eq(narrowed("1e-38", RejectInexact), "none", "the smallest fraction")
}
def test_to_i64_rounds_by_mode() -> unit ! { Test } = {
  _ = assert_eq(decimal_to_i64(decimal("2.5"), RoundTiesToEven), 2i64, "2.5 ties to even")
  _ = assert_eq(decimal_to_i64(decimal("-2.5"), RoundTiesToAway), -3i64, "-2.5 ties away")
  _ = assert_eq(decimal_to_i64(decimal("1e-38"), RoundTowardPositive), 1i64, "10^-38 up")
  _ = assert_eq(decimal_to_i64(decimal("1e-38"), RoundTowardNegative), 0i64, "10^-38 down")
  _ = assert_eq(decimal_to_i64(decimal("-1e-38"), RoundTowardNegative), -1i64, "-10^-38 down")
  assert_eq(decimal_to_i64(decimal("-1e-38"), RoundTowardZero), 0i64, "-10^-38 toward zero")
}
-- Rounding to a quantum, every mode in both signs.
def test_round_toward_negative() -> unit ! { Test } = {
  _ = assert_eq(rounded("2.5", 0i64, RoundTowardNegative), "2", "2.5")
  _ = assert_eq(rounded("-2.5", 0i64, RoundTowardNegative), "-3", "-2.5")
  _ = assert_eq(rounded("2.6", 0i64, RoundTowardNegative), "2", "2.6")
  assert_eq(rounded("-2.4", 0i64, RoundTowardNegative), "-3", "-2.4")
}
def test_round_toward_positive() -> unit ! { Test } = {
  _ = assert_eq(rounded("2.5", 0i64, RoundTowardPositive), "3", "2.5")
  _ = assert_eq(rounded("-2.5", 0i64, RoundTowardPositive), "-2", "-2.5")
  _ = assert_eq(rounded("2.4", 0i64, RoundTowardPositive), "3", "2.4")
  assert_eq(rounded("-2.6", 0i64, RoundTowardPositive), "-2", "-2.6")
}
def test_round_toward_zero() -> unit ! { Test } = {
  _ = assert_eq(rounded("2.5", 0i64, RoundTowardZero), "2", "2.5")
  _ = assert_eq(rounded("-2.5", 0i64, RoundTowardZero), "-2", "-2.5")
  _ = assert_eq(rounded("2.6", 0i64, RoundTowardZero), "2", "2.6")
  assert_eq(rounded("-2.6", 0i64, RoundTowardZero), "-2", "-2.6")
}
def test_round_away_from_zero() -> unit ! { Test } = {
  _ = assert_eq(rounded("2.5", 0i64, RoundAwayFromZero), "3", "2.5")
  _ = assert_eq(rounded("-2.5", 0i64, RoundAwayFromZero), "-3", "-2.5")
  _ = assert_eq(rounded("2.4", 0i64, RoundAwayFromZero), "3", "2.4")
  assert_eq(rounded("-2.4", 0i64, RoundAwayFromZero), "-3", "-2.4")
}
def test_round_ties_to_even() -> unit ! { Test } = {
  _ = assert_eq(rounded("2.5", 0i64, RoundTiesToEven), "2", "2.5")
  _ = assert_eq(rounded("-2.5", 0i64, RoundTiesToEven), "-2", "-2.5")
  _ = assert_eq(rounded("3.5", 0i64, RoundTiesToEven), "4", "3.5")
  _ = assert_eq(rounded("-3.5", 0i64, RoundTiesToEven), "-4", "-3.5")
  _ = assert_eq(rounded("2.4", 0i64, RoundTiesToEven), "2", "2.4")
  _ = assert_eq(rounded("-2.6", 0i64, RoundTiesToEven), "-3", "-2.6")
  _ = assert_eq(rounded("1.005", 2i64, RoundTiesToEven), "1", "1.005 to cents")
  assert_eq(rounded("1.015", 2i64, RoundTiesToEven), "1.02", "1.015 to cents")
}
def test_round_ties_to_away() -> unit ! { Test } = {
  _ = assert_eq(rounded("2.5", 0i64, RoundTiesToAway), "3", "2.5")
  _ = assert_eq(rounded("-2.5", 0i64, RoundTiesToAway), "-3", "-2.5")
  _ = assert_eq(rounded("2.4", 0i64, RoundTiesToAway), "2", "2.4")
  _ = assert_eq(rounded("-2.4", 0i64, RoundTiesToAway), "-2", "-2.4")
  _ = assert_eq(rounded("1.005", 2i64, RoundTiesToAway), "1.01", "1.005 to cents")
  assert_eq(rounded("-0.125", 2i64, RoundTiesToAway), "-0.13", "-0.125 to cents")
}
def test_round_reject_inexact_accepts_multiples() -> unit ! { Test } = {
  _ = assert_eq(rounded("1.5", 3i64, RejectInexact), "1.5", "a coarser value is unchanged")
  _ = assert_eq(rounded("2", 0i64, RejectInexact), "2", "an integer at scale 0")
  assert_eq(rounded("1.25", 2i64, RejectInexact), "1.25", "the exact scale")
}
def test_round_carries_and_envelope_edges() -> unit ! { Test } = {
  _ = assert_eq(rounded("9.995", 2i64, RoundTowardPositive), "10", "a carry into the integer")
  _ = assert_eq(rounded("9.995", 2i64, RoundTowardZero), "9.99", "no carry")
  _ = assert_eq(rounded("99999999999999999999999999999999999.999", 0i64, RoundAwayFromZero), "100000000000000000000000000000000000", "a carry across limbs")
  _ = assert_eq(rounded("9999999999999999999999999999999999999.9", 0i64, RoundTiesToEven), "10000000000000000000000000000000000000", "a carry to 10^37")
  _ = assert_eq(rounded("0.99999999999999999999999999999999999999", 37i64, RoundTiesToEven), "1", "38 nines to scale 37")
  _ = assert_eq(rounded("0.00000000000000000000000000000000000005", 37i64, RoundTiesToEven), "0", "a tie at 10^-37 to even zero")
  _ = assert_eq(rounded("0.00000000000000000000000000000000000005", 37i64, RoundTiesToAway), "0.0000000000000000000000000000000000001", "a tie at 10^-37 away")
  assert_eq(rounded("0.00000000000000000000000000000000000015", 37i64, RoundTiesToEven), "0.0000000000000000000000000000000000002", "a tie at 10^-37 to even two")
}
-- Exact arithmetic.
def test_add_is_exact() -> unit ! { Test } = {
  _ = assert_eq(decimal_to_string(decimal_add(decimal("0.1"), decimal("0.2"))), "0.3", "0.1 + 0.2")
  _ = assert_eq(decimal_to_string(decimal_add(decimal("999999999"), decimal("1"))), "1000000000", "a limb carry")
  _ = assert_eq(decimal_to_string(decimal_add(decimal("999999999999999999.999999999"), decimal("0.000000001"))), "1000000000000000000", "a carry chain")
  _ = assert_eq(decimal_to_string(decimal_add(decimal("0.30"), decimal("0.70"))), "1", "a canonical integer sum")
  _ = assert_eq(decimal_to_string(decimal_add(decimal("5"), decimal("-7.5"))), "-2.5", "mixed signs")
  _ = assert_eq(decimal_to_string(decimal_add(decimal("1e-38"), decimal("1e-38"))), "0.00000000000000000000000000000000000002", "the smallest values")
  assert_eq(decimal_to_string(decimal_add(decimal("99999999999999999999999999999999999998"), decimal("1"))), "99999999999999999999999999999999999999", "up to the maximum")
}
def test_sub_is_exact() -> unit ! { Test } = {
  _ = assert_eq(decimal_to_string(decimal_sub(decimal("0.3"), decimal("0.1"))), "0.2", "0.3 - 0.1")
  _ = assert_eq(decimal_to_string(decimal_sub(decimal("1"), decimal("1e-38"))), "0.99999999999999999999999999999999999999", "1 - 10^-38")
  _ = assert_eq(decimal_to_string(decimal_sub(decimal("1000000000"), decimal("1"))), "999999999", "a limb borrow")
  _ = assert_eq(decimal_to_string(decimal_sub(decimal("-5"), decimal("-7.5"))), "2.5", "negative operands")
  _ = assert_eq(decimal_to_string(decimal_sub(decimal("0"), decimal("1.5"))), "-1.5", "negation")
  assert_eq(decimal_to_string(decimal_sub(decimal("-99999999999999999999999999999999999998"), decimal("1"))), "-99999999999999999999999999999999999999", "down to the minimum")
}
def test_mul_is_exact() -> unit ! { Test } = {
  _ = assert_eq(decimal_to_string(decimal_mul(decimal("999999999"), decimal("999999999"))), "999999998000000001", "a limb product")
  _ = assert_eq(decimal_to_string(decimal_mul(decimal("9999999999999999999"), decimal("9999999999999999999"))), "99999999999999999980000000000000000001", "38 digits")
  _ = assert_eq(decimal_to_string(decimal_mul(decimal("123456789.123456789"), decimal("987654321.987654321"))), "121932631356500531.347203169112635269", "scale 18")
  _ = assert_eq(decimal_to_string(decimal_mul(decimal("1e-19"), decimal("1e-19"))), "0.00000000000000000000000000000000000001", "scale 38")
  _ = assert_eq(decimal_to_string(decimal_mul(decimal("0.5"), decimal("0.2"))), "0.1", "a canonical product")
  assert_eq(decimal_to_string(decimal_mul(decimal("-1.5"), decimal("-2"))), "3", "two negatives")
}
-- Division to a quantum.
def test_div_rounds_by_mode() -> unit ! { Test } = {
  _ = assert_eq(divided("2", "3", 2i64, RoundTowardNegative), "0.66", "2/3 down")
  _ = assert_eq(divided("2", "3", 2i64, RoundTowardPositive), "0.67", "2/3 up")
  _ = assert_eq(divided("-2", "3", 2i64, RoundTowardNegative), "-0.67", "-2/3 down")
  _ = assert_eq(divided("-2", "3", 2i64, RoundTowardPositive), "-0.66", "-2/3 up")
  _ = assert_eq(divided("2", "-3", 2i64, RoundTowardZero), "-0.66", "2/-3 toward zero")
  _ = assert_eq(divided("-2", "-3", 2i64, RoundAwayFromZero), "0.67", "-2/-3 away")
  _ = assert_eq(divided("1", "8", 2i64, RoundTiesToEven), "0.12", "1/8 ties to even")
  _ = assert_eq(divided("-1", "8", 2i64, RoundTiesToEven), "-0.12", "-1/8 ties to even")
  _ = assert_eq(divided("3", "8", 2i64, RoundTiesToEven), "0.38", "3/8 ties to even")
  _ = assert_eq(divided("1", "8", 2i64, RoundTiesToAway), "0.13", "1/8 ties away")
  _ = assert_eq(divided("-1", "8", 2i64, RoundTiesToAway), "-0.13", "-1/8 ties away")
  assert_eq(divided("1", "4", 2i64, RejectInexact), "0.25", "an exact quotient")
}
def test_div_long_quotients() -> unit ! { Test } = {
  _ = assert_eq(divided("10", "3", 37i64, RoundTiesToEven), "3.3333333333333333333333333333333333333", "38 digits")
  _ = assert_eq(divided("1", "7", 38i64, RoundTiesToEven), "0.14285714285714285714285714285714285714", "scale 38")
  _ = assert_eq(divided("1", "999999999999999999999", 38i64, RoundTiesToEven), "0.000000000000000000001", "a three-limb divisor")
  _ = assert_eq(divided("99999999999999999999999999999999999999", "99999999999999999999", 18i64, RoundTiesToEven), "1000000000000000000.01", "a long divisor")
  _ = assert_eq(divided("99999999999999999999999999999999999999", "1000000000000000000000000000000000001", 36i64, RoundTowardZero), "99.999999999999999999999999999999999899", "a five-limb divisor")
  _ = assert_eq(divided("440918321037564607897474984", "500000049999999074", 0i64, RoundTowardZero), "881836553", "a quotient limb estimated two too high")
  _ = assert_eq(divided("440918321037564607897474984", "500000049999999074", 0i64, RoundTowardPositive), "881836554", "the same quotient rounded up")
  _ = assert_eq(divided("375270389798487538356409535", "500000813999999184", 0i64, RoundTowardZero), "750539557", "another estimate two too high")
  _ = assert_eq(divided("440918321037564607897474984", "500000049999999074", 18i64, RoundTiesToEven), "881836553.891475459808701794", "a two-correction quotient at scale 18")
  _ = assert_eq(divided("99999999999999999999999999999999999999", "99999999999999999999999999999999999999", 0i64, RejectInexact), "1", "the maximum by itself")
  _ = assert_eq(divided("1e-38", "99999999999999999999999999999999999999", 38i64, RoundTowardZero), "0", "the smallest by the largest, toward zero")
  assert_eq(divided("1e-38", "99999999999999999999999999999999999999", 38i64, RoundAwayFromZero), "0.00000000000000000000000000000000000001", "the smallest by the largest, away")
}
def test_div_rejects_domain_failures() -> unit ! { Test } = {
  _ = assert_eq(divided("1", "0", 2i64, RoundTiesToEven), "none", "a zero divisor")
  _ = assert_eq(divided("1", "-0.000", 2i64, RoundTiesToEven), "none", "a written negative zero divisor")
  _ = assert_eq(divided("1", "3", 39i64, RoundTiesToEven), "none", "scale 39")
  _ = assert_eq(divided("1", "3", -1i64, RoundTiesToEven), "none", "scale -1")
  _ = assert_eq(divided("1", "3", i64_minimum(), RoundTiesToEven), "none", "the i64 minimum scale")
  assert_eq(divided("1", "3", 2i64, RejectInexact), "none", "an inexact quotient")
}
-- Order.
def test_order_is_exact() -> unit ! { Test } = {
  _ = assert_true(decimal_lt(decimal("-2"), decimal("-1.5")), "-2 < -1.5")
  _ = assert_true(decimal_lt(decimal("-1e-38"), decimal("0")), "-10^-38 < 0")
  _ = assert_true(decimal_lte(decimal("1.50"), decimal("1.5")), "1.50 <= 1.5")
  _ = assert_true(decimal_gt(decimal("1e-38"), decimal("0")), "10^-38 > 0")
  _ = assert_true(decimal_gte(decimal("99999999999999999999999999999999999999"), decimal("-99999999999999999999999999999999999999")), "maximum >= minimum")
  assert_true(decimal_gt(decimal("0.1"), decimal("0.09999999999999999999999999999999999999")), "different scales")
}
def test_order_rejects_false_relations() -> unit ! { Test } = {
  _ = assert_false(decimal_lt(decimal("1.5"), decimal("1.50")), "1.5 is not < 1.50")
  _ = assert_false(decimal_gt(decimal("-0"), decimal("0")), "-0 is not > 0")
  _ = assert_false(decimal_lte(decimal("1"), decimal("0.99999999999999999999999999999999999999")), "1 is not <= 0.999...")
  _ = assert_false(decimal_gte(decimal("-1.5"), decimal("-1")), "-1.5 is not >= -1")
  assert_false(decimal_lt(decimal("0"), decimal("-1e-38")), "0 is not < -10^-38")
}
-- Binary floats.
def test_to_f64_rounds_correctly() -> unit ! { Test } = {
  _ = assert_eq(decimal_to_f64(decimal("0.1")), 0.1f64, "0.1")
  _ = assert_eq(decimal_to_f64(decimal("-1e-38")), neg(1e-38f64), "-10^-38")
  _ = assert_eq(decimal_to_f64(decimal("99999999999999999999999999999999999999")), 1e38f64, "the maximum")
  _ = assert_eq(decimal_to_f64(decimal("9007199254740993")), 9007199254740992.0f64, "2^53 + 1 ties to even")
  _ = assert_eq(decimal_to_f64(decimal("9007199254740995")), 9007199254740996.0f64, "2^53 + 3 ties to even")
  assert_eq(decimal_to_f64(decimal("0")), 0.0f64, "zero")
}
def test_to_f32_rounds_once() -> unit ! { Test } = {
  _ = assert_eq(single("0.1"), 0.10000000149011612f64, "0.1")
  _ = assert_eq(single("-0.3"), neg(0.30000001192092896f64), "-0.3")
  _ = assert_eq(single("16777217"), 16777216.0f64, "2^24 + 1 ties to even")
  _ = assert_eq(single("16777219"), 16777220.0f64, "2^24 + 3 ties to even")
  _ = assert_eq(single("-16777217"), neg(16777216.0f64), "-(2^24 + 1)")
  _ = assert_eq(single("1.000000178813934326171875"), 1.000000238418579f64, "an exact tie to even")
  _ = assert_eq(single("1.000000059604644775390625000001"), 1.0000001192092896f64, "just above a tie")
  _ = assert_eq(single("16777215.4999999999"), 16777215.0f64, "just below a tie under 2^24")
  _ = assert_eq(single("16777215.99999999999999"), 16777216.0f64, "just below 2^24")
  _ = assert_eq(single("33554431.99999999999999999"), 33554432.0f64, "just below 2^25")
  _ = assert_eq(single("99999999999999999999999999999999999999"), 9.999999680285692e37f64, "the maximum")
  _ = assert_eq(single("1e-38"), 9.999999350456404e-39f64, "10^-38 is subnormal")
  _ = assert_eq(single("-1e-38"), neg(9.999999350456404e-39f64), "-10^-38 is subnormal")
  _ = assert_eq(single("2e-38"), 2.0000000102211272e-38f64, "2 * 10^-38 is normal")
  assert_eq(single("0"), 0.0f64, "zero")
}
def test_to_f32_differs_from_double_rounding() -> unit ! { Test } = {
  _ = assert_eq(cast(decimal_to_f64(decimal("1.000000059604644775390625000001")), f32), cast(1.0f64, f32), "through f64 the value lands on a tie")
  _ = assert_false(eq(single("1.000000059604644775390625000001"), 1.0f64), "the direct rounding is not 1")
  _ = assert_eq(cast(decimal_to_f64(decimal("16777215.4999999999")), f32), cast(16777216.0f64, f32), "through f64 a value below a tie lands on it")
  assert_false(eq(single("16777215.4999999999"), 16777216.0f64), "the direct rounding stays below")
}
def test_from_f64_is_exact_then_rounded() -> unit ! { Test } = {
  _ = assert_eq(ingested(0.1f64, 2i64, RoundTiesToEven), "0.1", "0.1 to cents")
  _ = assert_eq(ingested(0.1f64, 38i64, RoundTowardZero), "0.10000000000000000555111512312578270211", "0.1 at scale 38 down")
  _ = assert_eq(ingested(0.1f64, 38i64, RoundTiesToEven), "0.10000000000000000555111512312578270212", "0.1 at scale 38 nearest")
  _ = assert_eq(ingested(1e38f64, 0i64, RejectInexact), "99999999999999997748809823456034029568", "the double nearest 10^38")
  _ = assert_eq(ingested(2.5f64, 0i64, RoundTiesToEven), "2", "2.5 ties to even")
  _ = assert_eq(ingested(2.5f64, 0i64, RoundTiesToAway), "3", "2.5 ties away")
  _ = assert_eq(ingested(neg(2.5f64), 0i64, RoundTiesToEven), "-2", "-2.5 ties to even")
  _ = assert_eq(ingested(neg(2.5f64), 0i64, RoundTiesToAway), "-3", "-2.5 ties away")
  _ = assert_eq(ingested(0.5f64, 1i64, RejectInexact), "0.5", "an exact half")
  _ = assert_eq(ingested(div(1.0f64, 3.0f64), 20i64, RoundTiesToEven), "0.33333333333333331483", "the double nearest a third")
  _ = assert_eq(ingested(5e-324f64, 38i64, RoundAwayFromZero), "0.00000000000000000000000000000000000001", "the smallest subnormal away")
  _ = assert_eq(ingested(5e-324f64, 38i64, RoundTowardZero), "0", "the smallest subnormal toward zero")
  _ = assert_eq(ingested(neg(5e-324f64), 38i64, RoundTowardNegative), "-0.00000000000000000000000000000000000001", "the negative smallest subnormal down")
  _ = assert_eq(ingested(neg(5e-324f64), 38i64, RoundTowardPositive), "0", "the negative smallest subnormal up")
  assert_eq(ingested(9007199254740992.0f64, 0i64, RejectInexact), "9007199254740992", "2^53")
}
def test_from_f64_rejects_domain_failures() -> unit ! { Test } = {
  _ = assert_eq(ingested(0.1f64, 2i64, RejectInexact), "none", "0.1 is not a multiple of 0.01")
  _ = assert_eq(ingested(div(0.0f64, 0.0f64), 2i64, RoundTiesToEven), "none", "NaN")
  _ = assert_eq(ingested(div(1.0f64, 0.0f64), 2i64, RoundTiesToEven), "none", "infinity")
  _ = assert_eq(ingested(div(-1.0f64, 0.0f64), 2i64, RoundTiesToEven), "none", "negative infinity")
  _ = assert_eq(ingested(1.0f64, 39i64, RoundTiesToEven), "none", "scale 39")
  _ = assert_eq(ingested(1.0f64, -1i64, RoundTiesToEven), "none", "scale -1")
  _ = assert_eq(ingested(2e38f64, 0i64, RoundTowardZero), "none", "2 * 10^38")
  _ = assert_eq(ingested(1.0000000000000002e38f64, 0i64, RoundTowardZero), "none", "the double above 10^38")
  assert_eq(ingested(1.7976931348623157e308f64, 38i64, RoundTowardZero), "none", "the largest double")
}
