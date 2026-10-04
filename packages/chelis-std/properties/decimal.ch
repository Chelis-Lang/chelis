module Std.Properties.Decimal
import Std.Decimal (Decimal, decimal, try_decimal, decimal_to_string, decimal_from_i64, decimal_to_i64, try_decimal_from_f64, decimal_to_f64, decimal_scale, decimal_add, decimal_sub, decimal_round, decimal_lt, decimal_lte)
import Std.Rounding (Rounding, RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)
-- [05-OP-76]'s laws as `chelis prove` properties at the fuzz tier. A property
-- binds only scalars, so each one builds its decimals inside the property from
-- number text assembled out of `i64` and `string` binders, or from an `f64`
-- binder. A `where` guard keeps exactly the inputs a law speaks about, so the
-- harness counts every other sample as rejected rather than as a pass.
-- `value` modulo `size`, in 0..size-1 for every i64.
def wrapped(value: i64, size: i64) -> i64 =
  value
  |> mod(size)
  |> add(size)
  |> mod(size)
-- The decimal digits of a text, in order; every other character is dropped.
def digits_of(text: string) -> string =
  fold(fn (acc: string, idx: i64) -> {
    piece = string_slice(text, idx, 1i64)
    code = char_code(piece)
    if code |> gte(48i64) |> and(lte(code, 57i64)) then string_concat(acc, piece) else acc
  }, "", range(0i64, string_len(text)))
-- Number text `<whole>.<digits of fraction>e<exponent>`, with the exponent in
-- -40..40. It is a number token exactly when `fraction` holds a digit.
def number_text(whole: i64, fraction: string, exponent: i64) -> string = string_concat(string_concat(to_string(whole), "."), string_concat(digits_of(fraction), string_concat("e", to_string(sub(wrapped(exponent, 81i64), 40i64)))))
def accepted(text: string) -> bool =
  match try_decimal(text) with {
    | Some(x) => true
    | None => false
  }
def magnitude(x: Decimal) -> Decimal = if decimal_lt(x, decimal("0")) then decimal_sub(decimal("0"), x) else x
-- The largest decimal at `scale`: 38 nines times 10^-scale.
def largest_at(scale: i64) -> Decimal = decimal(string_concat("99999999999999999999999999999999999999e-", to_string(scale)))
-- |x| + |y| is at most the largest decimal at the wider scale, so x + y and
-- x - y are both in the value set. The bound minus |y| is formed only once
-- |y| is known to be within the bound, where it cannot overflow.
def sums_fit(x: Decimal, y: Decimal) -> bool = {
  bound = largest_at(if gte(decimal_scale(x), decimal_scale(y)) then decimal_scale(x) else decimal_scale(y))
  if decimal_lte(magnitude(y), bound) then decimal_lte(magnitude(x), decimal_sub(bound, magnitude(y))) else false
}
def summable(left: string, right: string) -> bool =
  match (try_decimal(left), try_decimal(right)) with {
    | (Some(x), Some(y)) => sums_fit(x, y)
    | _ => false
  }
-- A rounding mode other than `RejectInexact`, chosen by `choice`.
def inexact_mode(choice: i64) -> Rounding = index([RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway], wrapped(choice, 6i64))
-- Any rounding mode, chosen by `choice`.
def any_mode(choice: i64) -> Rounding = index([RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact], wrapped(choice, 7i64))
def ingested(value: f64, places: i64) -> bool =
  match try_decimal_from_f64(value, wrapped(places, 39i64), RoundTiesToEven) with {
    | Some(x) => true
    | None => false
  }
-- Whether rounding `y` again at `places` by `mode` leaves it unchanged, and
-- `RejectInexact` finds it exact there.
def settles(y: Decimal, places: i64, mode: Rounding) -> bool =
  eq(decimal_round(y, places, mode), y)
  |> and(eq(decimal_round(y, places, RejectInexact), y))
-- An i64 of up to 19 digits from three binders, whatever their range.
def assembled(high: i64, middle: i64, low: i64) -> i64 =
  high
  |> mod(9000i64)
  |> mul(1000000000000000i64)
  |> add(mul(mod(middle, 1000000000i64), 1000000i64))
  |> add(mod(low, 1000000i64))
-- Text the parser accepts prints back to text that parses to the same decimal.
@property decimal_text_round_trips forall(whole: i64, fraction: string, exponent: i64) where accepted(number_text(whole, fraction, exponent)):
  {
    x = decimal(number_text(whole, fraction, exponent))
    eq(decimal(decimal_to_string(x)), x)
  }
@property decimal_add_commutes forall(whole: i64, fraction: string, exponent: i64, other_whole: i64, other_fraction: string, other_exponent: i64) where summable(number_text(whole, fraction, exponent), number_text(other_whole, other_fraction, other_exponent)):
  {
    x = decimal(number_text(whole, fraction, exponent))
    y = decimal(number_text(other_whole, other_fraction, other_exponent))
    eq(decimal_add(x, y), decimal_add(y, x))
  }
@property decimal_sub_inverts_decimal_add forall(whole: i64, fraction: string, exponent: i64, other_whole: i64, other_fraction: string, other_exponent: i64) where summable(number_text(whole, fraction, exponent), number_text(other_whole, other_fraction, other_exponent)):
  {
    x = decimal(number_text(whole, fraction, exponent))
    y = decimal(number_text(other_whole, other_fraction, other_exponent))
    eq(decimal_sub(decimal_add(x, y), y), x)
    |> and(eq(decimal_add(decimal_sub(x, y), y), x))
  }
-- Text the parser accepts converts to the f64 that `to_float` reads from it.
@property decimal_to_f64_agrees_with_to_float forall(whole: i64, fraction: string, exponent: i64) where accepted(number_text(whole, fraction, exponent)):
  {
    text = number_text(whole, fraction, exponent)
    x = decimal(text)
    match to_float(text) with {
      | Some(expected) => eq(decimal_to_f64(x), expected)
      | None => false
    }
  }
-- Rounding a rounded value again at the same scale and mode changes nothing,
-- and the rounded value is exact at that scale.
@property decimal_round_is_idempotent forall(value: f64, places: i64, digits: i64, choice: i64) where ingested(value, places):
  {
    n = wrapped(digits, 39i64)
    mode = inexact_mode(choice)
    match try_decimal_from_f64(value, wrapped(places, 39i64), RoundTiesToEven) with {
      | Some(x) => settles(decimal_round(x, n, mode), n, mode)
      | None => false
    }
  }
@property decimal_from_i64_round_trips forall(high: i64, middle: i64, low: i64, choice: i64):
  {
    v = assembled(high, middle, low)
    eq(decimal_to_i64(decimal_from_i64(v), any_mode(choice)), v)
  }
