module Std.Properties.Decimal
import Std.Decimal (Decimal, decimal, try_decimal, decimal_to_string, decimal_from_i64, decimal_to_i64, try_decimal_from_f64, decimal_to_f64, decimal_scale, decimal_add, decimal_sub, decimal_round, decimal_lt, decimal_lte)
import Std.Rounding (Rounding, RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)
-- [05-OP-76]'s laws as `chelis prove` properties at the fuzz tier. A property
-- binds only scalars, so each one builds its decimals inside the property from
-- number text assembled out of `i64` and `string` binders, or from an `f64`
-- binder. A `where` guard keeps exactly the inputs a law speaks about, so the
-- harness counts every other sample as rejected rather than as a pass.
-- The fuzz generator draws an `i64` binder from [-1000, 1000], a `string`
-- binder as `s` followed by an integer in 0..999, and an `f64` binder from
-- [-10, 10], so each property validates its law only over the sampled domain
-- its comment states. Each comment also names the deterministic tests that
-- check the callables at the extremes outside that domain.
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
-- `text` written `copies` times in a row.
def repeated(text: string, copies: i64) -> string = fold(fn (acc: string, unused: i64) -> string_concat(acc, text), "", range(0i64, copies))
-- Number text `<whole>.<fraction digits>e<exponent>`: the fraction digits are
-- the digits of `fraction` written 1 to 12 times, as `copies` selects, so a
-- coefficient reaches 40 significant digits and spans every limb, and the
-- exponent is in -40..40. It is a number token exactly when `fraction` holds
-- a digit.
def number_text(whole: i64, fraction: string, copies: i64, exponent: i64) -> string = string_concat(string_concat(to_string(whole), "."), string_concat(repeated(digits_of(fraction), add(wrapped(copies, 12i64), 1i64)), string_concat("e", to_string(sub(wrapped(exponent, 81i64), 40i64)))))
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
-- The i64 `high * 10^15 + middle * 10^6 + low`, after reducing the binders
-- modulo 9000, 10^9 and 10^6 so the result stays inside i64 for any binders.
def assembled(high: i64, middle: i64, low: i64) -> i64 =
  high
  |> mod(9000i64)
  |> mul(1000000000000000i64)
  |> add(mul(mod(middle, 1000000000i64), 1000000i64))
  |> add(mod(low, 1000000i64))
-- Text the parser accepts prints back to text that parses to the same decimal.
-- Sampled domain: the decimals `w.D * 10^e` in the value set, with `w` in
-- [-1000, 1000], `D` the digits of an integer in 0..999 written 1 to 12
-- times, and `e` in [-40, 40]: coefficients of up to 38 significant digits
-- spanning every limb, negative only when `w` is. Outside it, the differential's `row_parse` and
-- `row_text` rows (scripts/decimal_differential.py) check parsing and
-- canonical text exactly against the reference on the envelope ends, limb
-- boundaries and random canonical decimals of 1 to 38 digits at scales 0 to 38,
-- and the self-test `test_to_string_round_trips` round-trips the minimum and
-- 10^-38.
@property decimal_text_round_trips forall(whole: i64, fraction: string, copies: i64, exponent: i64) where accepted(number_text(whole, fraction, copies, exponent)):
  {
    x = decimal(number_text(whole, fraction, copies, exponent))
    eq(decimal(decimal_to_string(x)), x)
  }
-- `decimal_add` is commutative. Sampled domain: pairs of the text-built
-- decimals described for `decimal_text_round_trips` whose magnitudes sum to at
-- most the largest decimal at their wider scale. Outside it, the
-- differential's `arith` rows check `decimal_add` and `decimal_sub` exactly
-- against the reference, a stronger contract than either law, on envelope,
-- limb-carry and random 1 to 38 digit operand pairs, including overflow, as do
-- the self-tests `test_add_is_exact` and `test_sub_is_exact`.
@property decimal_add_commutes forall(whole: i64, fraction: string, copies: i64, exponent: i64, other_whole: i64, other_fraction: string, other_copies: i64, other_exponent: i64) where summable(number_text(whole, fraction, copies, exponent), number_text(other_whole, other_fraction, other_copies, other_exponent)):
  {
    x = decimal(number_text(whole, fraction, copies, exponent))
    y = decimal(number_text(other_whole, other_fraction, other_copies, other_exponent))
    eq(decimal_add(x, y), decimal_add(y, x))
  }
-- `decimal_sub` inverts `decimal_add`. Sampled domain and coverage outside it
-- as for `decimal_add_commutes`.
@property decimal_sub_inverts_decimal_add forall(whole: i64, fraction: string, copies: i64, exponent: i64, other_whole: i64, other_fraction: string, other_copies: i64, other_exponent: i64) where summable(number_text(whole, fraction, copies, exponent), number_text(other_whole, other_fraction, other_copies, other_exponent)):
  {
    x = decimal(number_text(whole, fraction, copies, exponent))
    y = decimal(number_text(other_whole, other_fraction, other_copies, other_exponent))
    eq(decimal_sub(decimal_add(x, y), y), x)
    |> and(eq(decimal_add(decimal_sub(x, y), y), x))
  }
-- Text the parser accepts converts to the f64 that `to_float` reads from it.
-- Sampled domain: the text-built decimals described for
-- `decimal_text_round_trips`. Outside it, the differential's `row_floats` rows
-- check `decimal_to_f64` bit for bit against the once-rounded reference on the
-- envelope ends, limb boundaries, i64 edges, ties, f64 and f32 midpoint
-- witnesses and random decimals of 1 to 38 digits, and the self-test
-- `test_to_f64_rounds_correctly` covers the maximum, -10^-38 and the 2^53 ties.
@property decimal_to_f64_agrees_with_to_float forall(whole: i64, fraction: string, copies: i64, exponent: i64) where accepted(number_text(whole, fraction, copies, exponent)):
  {
    text = number_text(whole, fraction, copies, exponent)
    x = decimal(text)
    match to_float(text) with {
      | Some(expected) => eq(decimal_to_f64(x), expected)
      | None => false
    }
  }
-- Rounding a rounded value again at the same scale and mode changes nothing,
-- and the rounded value is exact at that scale. Sampled domain: the exact value
-- of an f64 in [-10, 10] rounded to 0 to 38 places with ties to even and kept
-- when in the value set, so |x| <= 10, rounded again at 0 to 38 places by
-- each mode but `RejectInexact`. Outside it, the differential's `round` rows
-- check `decimal_round` exactly against the reference on the envelope ends,
-- ties and i64 edges of up to 38 digits at scales 0, 1, 2, 5, 18, 37 and 38 in
-- every mode, and the self-test `test_round_carries_and_envelope_edges`
-- covers carries to 10^37.
@property decimal_round_is_idempotent forall(value: f64, places: i64, digits: i64, choice: i64) where ingested(value, places):
  {
    n = wrapped(digits, 39i64)
    mode = inexact_mode(choice)
    match try_decimal_from_f64(value, wrapped(places, 39i64), RoundTiesToEven) with {
      | Some(x) => settles(decimal_round(x, n, mode), n, mode)
      | None => false
    }
  }
-- `decimal_from_i64` and `decimal_to_i64` round-trip under every mode.
-- Sampled domain: `v = a * 10^15 + b * 10^6 + c` with `a`, `b` and `c` in
-- [-1000, 1000], so |v| <= 1000000001000001000 and the digits at 10^10 to 10^14
-- are zero or nine. Outside it, the self-test
-- `test_to_i64_round_trips_the_boundaries` round-trips the i64 maximum and
-- minimum, and the differential checks `decimal_from_i64` on those, on
-- +-10^18 and on random i64 values over the whole range, and `decimal_to_i64`
-- on the i64 edges in every mode.
@property decimal_from_i64_round_trips forall(high: i64, middle: i64, low: i64, choice: i64):
  {
    v = assembled(high, middle, low)
    eq(decimal_to_i64(decimal_from_i64(v), any_mode(choice)), v)
  }
