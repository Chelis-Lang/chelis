module Std.Decimal
export (Decimal, RoundingMode, round_half_up, round_half_even, round_down, round_up, decimal, try_decimal, decimal_from_int, decimal_add, decimal_sub, decimal_mul, decimal_div, decimal_eq, decimal_lt, decimal_lte, decimal_gt, decimal_gte, decimal_to_float, decimal_to_string)
type RoundingMode =
  | RoundHalfUp
  | RoundHalfEven
  | RoundDown
  | RoundUp
def round_half_up() -> RoundingMode = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def round_half_even() -> RoundingMode = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def round_down() -> RoundingMode = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def round_up() -> RoundingMode = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
type Decimal =
  | Decimal { coefficient: i64, scale: i64 }
def decimal(text: string) -> Decimal = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def try_decimal(text: string) -> Option[Decimal] = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_from_int(value: i64) -> Decimal = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_add(lhs: Decimal, rhs: Decimal) -> Decimal = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_sub(lhs: Decimal, rhs: Decimal) -> Decimal = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_mul(lhs: Decimal, rhs: Decimal) -> Decimal = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_div(lhs: Decimal, rhs: Decimal, result_scale: i64, mode: RoundingMode) -> Decimal = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_eq(lhs: Decimal, rhs: Decimal) -> bool = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_lt(lhs: Decimal, rhs: Decimal) -> bool = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_lte(lhs: Decimal, rhs: Decimal) -> bool = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_gt(lhs: Decimal, rhs: Decimal) -> bool = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_gte(lhs: Decimal, rhs: Decimal) -> bool = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_to_float(value: Decimal) -> f64 = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
def decimal_to_string(value: Decimal) -> string = fail("Std.Decimal is unavailable: exact decimal arithmetic is not implemented (#2778)")
