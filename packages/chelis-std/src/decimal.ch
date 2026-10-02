module Std.Decimal
export (Decimal, decimal, try_decimal, decimal_to_string, decimal_to_fixed_string, decimal_from_i64, decimal_to_i64, try_decimal_to_i64, decimal_from_f64, try_decimal_from_f64, decimal_to_f64, decimal_to_f32, decimal_scale, decimal_add, decimal_sub, decimal_mul, decimal_round, decimal_div, try_decimal_div, decimal_lt, decimal_lte, decimal_gt, decimal_gte)
import Std.Rounding (Rounding, RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)
-- Std.Decimal: exact decimal numbers, governed by [05-OP-76]. A `Decimal` is
-- the rational c / 10^s with |c| <= 10^38 - 1 and 0 <= s <= 38, held in
-- canonical form: no trailing fractional zero, and zero is (0, 0). The type is
-- opaque, so the exported producers below are its only construction path.
-- Every range and validity check runs before the arithmetic it protects, so no
-- primitive numeric trap escapes a call; failures use `fail` with the message
-- grammar `<function>: <kind>: <detail>`.
@opaque
type Decimal =
  | Decimal { negative: bool, limb0: i64, limb1: i64, limb2: i64, limb3: i64, limb4: i64, scale: i64 }
-- Constants. A magnitude is a little-endian List[i64] of base-10^9 limbs, so
-- every limb product plus carry stays below 10^18 < 2^63.
def dec_base() -> i64 = 1000000000i64
def dec_max_scale() -> i64 = 38i64
def dec_min(x: i64, y: i64) -> i64 = if lt(x, y) then x else y
def dec_max(x: i64, y: i64) -> i64 = if gt(x, y) then x else y
-- Text assembly.
def dec_joined(parts: List[string]) -> string = fold(fn (acc: string, part: string) -> string_concat(acc, part), "", parts)
def dec_failure(function: string, kind: string, detail: string) -> string = dec_joined([function, ": ", kind, ": ", detail])
def dec_zeros_text(count: i64) -> string = fold(fn (acc: string, unused: i64) -> string_concat(acc, "0"), "", range(0i64, count))
def dec_padded(value: i64, width: i64) -> string = {
  digits = to_string(value)
  string_concat(dec_zeros_text(sub(width, string_len(digits))), digits)
}
-- Text named in a failure: its first 40 characters, then `...` when longer.
def dec_shown(text: string) -> string = if text |> string_len |> gt(40i64) then text |> string_slice(0i64, 40i64) |> string_concat("...") else text
def dec_scale_outside(places: i64) -> bool = places |> lt(0i64) |> or(gt(places, dec_max_scale()))
def dec_scale_detail(places: i64) -> string = dec_joined(["scale ", to_string(places), " is outside 0..38"])
def dec_inexact_detail(value: string, places: i64) -> string = dec_joined([value, " is not a multiple of 10^-", to_string(places)])
def dec_range_detail(value: string) -> string = string_concat(value, " is outside the decimal range")
-- Limb arithmetic. Every loop is a fold or scan over a precomputed range;
-- nothing recurses over limbs or digits. Results are trimmed to their
-- highest nonzero limb, and zero is [0].
def dec_limb(xs: List[i64], idx: i64) -> i64 = if lt(idx, len(xs)) then index(xs, idx) else 0i64
def dec_wider(xs: List[i64], ys: List[i64]) -> i64 = xs |> len |> dec_max(len(ys))
-- The number of limbs up to and including the highest nonzero one.
def dec_used(xs: List[i64]) -> i64 = fold(fn (acc: i64, pair: (i64, i64)) -> if neq(pair.1, 0i64) then add(pair.0, 1i64) else acc, 0i64, enumerate(xs))
def dec_trim(xs: List[i64]) -> List[i64] = {
  used = dec_used(xs)
  if eq(used, 0i64) then [0i64] else take(xs, used)
}
def dec_is_zero_limbs(xs: List[i64]) -> bool = xs |> dec_used |> eq(0i64)
def dec_is_odd(xs: List[i64]) -> bool = eq(mod(dec_limb(xs, 0i64), 2i64), 1i64)
-- The sign of xs - ys: the highest differing limb decides.
def dec_cmp(xs: List[i64], ys: List[i64]) -> i64 =
  fold(fn (acc: i64, idx: i64) -> {
    x = dec_limb(xs, idx)
    y = dec_limb(ys, idx)
    if gt(x, y) then 1i64 else if lt(x, y) then -1i64 else acc
  }, 0i64, range(0i64, dec_wider(xs, ys)))
def dec_add(xs: List[i64], ys: List[i64]) -> List[i64] = {
  steps = scan(fn (state: (i64, i64), idx: i64) -> {
    total = add(add(dec_limb(xs, idx), dec_limb(ys, idx)), state.1)
    (mod(total, dec_base()), trunc_div(total, dec_base()))
  }, (0i64, 0i64), range(0i64, xs |> dec_wider(ys) |> add(1i64)))
  dec_trim(map(fn (state: (i64, i64)) -> state.0, steps))
}
-- xs - ys for xs >= ys.
def dec_sub(xs: List[i64], ys: List[i64]) -> List[i64] = {
  steps = scan(fn (state: (i64, i64), idx: i64) -> {
    difference = sub(sub(dec_limb(xs, idx), dec_limb(ys, idx)), state.1)
    if lt(difference, 0i64) then (add(difference, dec_base()), 1i64) else (difference, 0i64)
  }, (0i64, 0i64), range(0i64, len(xs)))
  dec_trim(map(fn (state: (i64, i64)) -> state.0, steps))
}
-- xs * factor for 0 <= factor <= 10^9.
def dec_mul_small(xs: List[i64], factor: i64) -> List[i64] = {
  steps = scan(fn (state: (i64, i64), idx: i64) -> {
    total = add(mul(dec_limb(xs, idx), factor), state.1)
    (mod(total, dec_base()), trunc_div(total, dec_base()))
  }, (0i64, 0i64), range(0i64, xs |> len |> add(1i64)))
  dec_trim(map(fn (state: (i64, i64)) -> state.0, steps))
}
-- (floor(xs / divisor), xs mod divisor) for 1 <= divisor <= 10^9, walking
-- from the top limb down and prepending each quotient limb.
def dec_divmod_small(xs: List[i64], divisor: i64) -> (List[i64], i64) = {
  top = xs |> len |> sub(1i64)
  folded = fold(fn (state: (List[i64], i64), step: i64) -> {
    current = state.1 |> mul(dec_base()) |> add(dec_limb(xs, sub(top, step)))
    (concat([trunc_div(current, divisor)], state.0), mod(current, divisor))
  }, ([0i64], 0i64), range(0i64, len(xs)))
  (dec_trim(folded.0), folded.1)
}
-- xs * (10^9)^count.
def dec_shift_limbs(xs: List[i64], count: i64) -> List[i64] = concat(map(fn (unused: i64) -> 0i64, range(0i64, count)), xs)
def dec_small_pow10(exponent: i64) -> i64 = index([1i64, 10i64, 100i64, 1000i64, 10000i64, 100000i64, 1000000i64, 10000000i64, 100000000i64, 1000000000i64], exponent)
-- xs * 10^exponent for exponent >= 0.
def dec_scale_up(xs: List[i64], exponent: i64) -> List[i64] =
  xs
  |> dec_mul_small(dec_small_pow10(mod(exponent, 9i64)))
  |> dec_shift_limbs(trunc_div(exponent, 9i64))
  |> dec_trim
-- floor(xs / 10^exponent) for exponent >= 0.
def dec_scale_down(xs: List[i64], exponent: i64) -> List[i64] =
  (xs
  |> skip(trunc_div(exponent, 9i64))
  |> dec_trim
  |> dec_divmod_small(dec_small_pow10(mod(exponent, 9i64)))).0
def dec_pow10(exponent: i64) -> List[i64] = dec_scale_up([1i64], exponent)
-- xs * 2^exponent for exponent >= 0, in steps of at most 2^29 < 10^9.
def dec_times_pow2(xs: List[i64], exponent: i64) -> List[i64] = fold(fn (acc: List[i64], step: i64) -> dec_mul_small(acc, shl(1i64, dec_min(29i64, sub(exponent, mul(step, 29i64))))), xs, range(0i64, exponent |> trunc_div(29i64) |> add(1i64)))
-- Schoolbook product: one small multiply per limb of xs, each normalized
-- before it is added into the running sum.
def dec_mul(xs: List[i64], ys: List[i64]) -> List[i64] = fold(fn (acc: List[i64], idx: i64) -> dec_add(acc, ys |> dec_mul_small(dec_limb(xs, idx)) |> dec_shift_limbs(idx)), [0i64], range(0i64, len(xs)))
-- Long division (Knuth's algorithm D) for a divisor of `size` >= 2 limbs.
-- Both operands are scaled so the divisor's top limb is at least 10^9 / 2;
-- each quotient limb is estimated from the remainder's top two limbs, which
-- overestimates it by at most two, and corrected by two fixed steps.
def dec_long_divmod(num: List[i64], den: List[i64], size: i64) -> (List[i64], List[i64]) = {
  factor = trunc_div(dec_base(), den |> dec_limb(sub(size, 1i64)) |> add(1i64))
  divisor = dec_mul_small(den, factor)
  lead = dec_limb(divisor, sub(size, 1i64))
  count = add(sub(len(num), size), 1i64)
  folded = fold(fn (state: (List[i64], List[i64]), step: i64) -> {
    position = count |> sub(1i64) |> sub(step)
    rest = state.1
    top = add(mul(dec_limb(rest, add(position, size)), dec_base()), dec_limb(rest, sub(add(position, size), 1i64)))
    guess = top |> trunc_div(lead) |> dec_min(sub(dec_base(), 1i64))
    digit = fold(fn (q: i64, unused: i64) -> if gt(dec_cmp(dec_shift_limbs(dec_mul_small(divisor, q), position), rest), 0i64) then sub(q, 1i64) else q, guess, range(0i64, 2i64))
    (concat([digit], state.0), dec_sub(rest, divisor |> dec_mul_small(digit) |> dec_shift_limbs(position)))
  }, ([0i64], dec_mul_small(num, factor)), range(0i64, count))
  (dec_trim(folded.0), dec_divmod_small(folded.1, factor).0)
}
-- (floor(num / den), num mod den) for a nonzero den.
def dec_divmod(num: List[i64], den: List[i64]) -> (List[i64], List[i64]) = {
  size = dec_used(den)
  if eq(size, 1i64) then {
    (quotient, remainder) = dec_divmod_small(num, dec_limb(den, 0i64))
    (quotient, [remainder])
  } else num |> dec_trim |> dec_long_divmod(dec_trim(den), size)
}
-- Trailing decimal zeros of a nonzero limb.
def dec_limb_trailing_zeros(limb: i64) -> i64 = fold(fn (acc: i64, exponent: i64) -> if limb |> mod(dec_small_pow10(exponent)) |> eq(0i64) then exponent else acc, 0i64, range(1i64, 9i64))
-- Trailing decimal zeros of a nonzero magnitude.
def dec_trailing_zeros(xs: List[i64]) -> i64 = fold(fn (state: (i64, bool), limb: i64) -> if state.1 then state else if eq(limb, 0i64) then (add(state.0, 9i64), false) else (add(state.0, dec_limb_trailing_zeros(limb)), true), (0i64, false), xs).0
-- Rounding. `dec_rounds_up` decides whether an inexact magnitude quotient
-- moves one quantum away from zero, given the sign and how twice the
-- remainder compares with the divisor.
def dec_rounds_up(mode: Rounding, negative: bool, half: i64, odd: bool) -> bool =
  match mode with {
    | RoundTowardNegative => negative
    | RoundTowardPositive => not(negative)
    | RoundTowardZero => false
    | RoundAwayFromZero => true
    | RoundTiesToEven => half |> gt(0i64) |> or(and(eq(half, 0i64), odd))
    | RoundTiesToAway => gte(half, 0i64)
    | RejectInexact => false
  }
def dec_rejects(mode: Rounding) -> bool =
  match mode with {
    | RejectInexact => true
    | _ => false
  }
-- num / den rounded to an integer by `mode`: (magnitude, exact).
def dec_round_quotient(num: List[i64], den: List[i64], negative: bool, mode: Rounding) -> (List[i64], bool) = {
  (quotient, remainder) = dec_divmod(num, den)
  exact = dec_is_zero_limbs(remainder)
  half = remainder |> dec_mul_small(2i64) |> dec_cmp(den)
  bumped = if exact then false else dec_rounds_up(mode, negative, half, dec_is_odd(quotient))
  (if bumped then dec_add(quotient, [1i64]) else quotient, exact)
}
-- Canonical form and the value set.
def dec_canonical(xs: List[i64], scale: i64) -> (List[i64], i64) =
  if dec_is_zero_limbs(xs) then ([0i64], 0i64) else {
    strip = xs |> dec_trailing_zeros |> dec_min(scale)
    (dec_scale_down(xs, strip), sub(scale, strip))
  }
def dec_fits(xs: List[i64], scale: i64) -> bool = and(and(lte(dec_used(xs), 5i64), lt(dec_limb(xs, 4i64), 100i64)), lte(scale, dec_max_scale()))
def dec_value(negative: bool, xs: List[i64], scale: i64) -> Decimal = Decimal { negative: and(negative, xs |> dec_is_zero_limbs |> not), limb0: dec_limb(xs, 0i64), limb1: dec_limb(xs, 1i64), limb2: dec_limb(xs, 2i64), limb3: dec_limb(xs, 3i64), limb4: dec_limb(xs, 4i64), scale }
def dec_zero() -> Decimal = Decimal { negative: false, limb0: 0i64, limb1: 0i64, limb2: 0i64, limb3: 0i64, limb4: 0i64, scale: 0i64 }
def dec_magnitude(x: Decimal) -> List[i64] = dec_trim([x.limb0, x.limb1, x.limb2, x.limb3, x.limb4])
def dec_is_zero(x: Decimal) -> bool = x |> dec_magnitude |> dec_is_zero_limbs
-- x rounded to a multiple of 10^-places by `mode`: canonical (magnitude,
-- scale, exact).
def dec_round_to(x: Decimal, places: i64, mode: Rounding) -> (List[i64], i64, bool) =
  if lte(x.scale, places) then (dec_magnitude(x), x.scale, true) else {
    rounded = dec_round_quotient(dec_magnitude(x), dec_pow10(sub(x.scale, places)), x.negative, mode)
    kept = dec_canonical(rounded.0, places)
    (kept.0, kept.1, rounded.1)
  }
-- Outcomes of the callables that have `try_` twins: (kind, detail, value),
-- with an empty kind on success.
def dec_ok(value: Decimal) -> (string, string, Decimal) = ("", "", value)
def dec_domain(detail: string) -> (string, string, Decimal) = ("domain", detail, dec_zero())
def dec_overflow(detail: string) -> (string, string, Decimal) = ("overflow", detail, dec_zero())
def dec_result(function: string, outcome: (string, string, Decimal)) -> Decimal = if eq(outcome.0, "") then outcome.2 else function |> dec_failure(outcome.0, outcome.1) |> fail
def dec_try_result(function: string, outcome: (string, string, Decimal)) -> Option[Decimal] = if eq(outcome.0, "") then Some(outcome.2) else if eq(outcome.0, "domain") then None else function |> dec_failure(outcome.0, outcome.1) |> fail
-- Text rendering.
def dec_digits(xs: List[i64]) -> string = {
  top = xs |> dec_used |> sub(1i64)
  if lt(top, 0i64) then "0" else fold(fn (acc: string, step: i64) -> string_concat(acc, xs |> dec_limb(sub(sub(top, 1i64), step)) |> dec_padded(9i64)), xs |> dec_limb(top) |> to_string, range(0i64, top))
}
def dec_text(negative: bool, digits: string, scale: i64) -> string = {
  count = string_len(digits)
  body = if eq(scale, 0i64) then digits else if gt(count, scale) then dec_joined([string_slice(digits, 0i64, sub(count, scale)), ".", string_slice(digits, sub(count, scale), scale)]) else dec_joined(["0.", scale |> sub(count) |> dec_zeros_text, digits])
  if negative then string_concat("-", body) else body
}
-- Text parsing works on the token's scalar codes, so every character is
-- checked before any digit is read.
def dec_code(codes: List[i64], idx: i64) -> i64 = if lt(idx, len(codes)) then index(codes, idx) else -1i64
def dec_is_digit_code(code: i64) -> bool = code |> gte(48i64) |> and(lte(code, 57i64))
-- The index of the first non-digit at or after `start`.
def dec_digit_run_end(codes: List[i64], start: i64) -> i64 = fold(fn (acc: i64, pair: (i64, i64)) -> if acc |> eq(pair.0) |> and(dec_is_digit_code(pair.1)) then add(acc, 1i64) else acc, start, enumerate(codes))
def dec_leading_zeros(codes: List[i64]) -> i64 = fold(fn (state: (i64, bool), code: i64) -> if state.1 then state else if eq(code, 48i64) then (add(state.0, 1i64), false) else (state.0, true), (0i64, false), codes).0
def dec_trailing_zero_codes(codes: List[i64]) -> i64 = fold(fn (acc: i64, code: i64) -> if eq(code, 48i64) then add(acc, 1i64) else 0i64, 0i64, codes)
def dec_code_slice(codes: List[i64], start: i64, end: i64) -> List[i64] = codes |> skip(start) |> take(sub(end, start))
-- The magnitude of at most 38 digit codes.
def dec_digit_magnitude(codes: List[i64]) -> List[i64] = fold(fn (acc: List[i64], code: i64) -> acc |> dec_mul_small(10i64) |> dec_add([sub(code, 48i64)]), [0i64], codes)
-- The value of at most 4 digit codes.
def dec_small_digits(codes: List[i64]) -> i64 = fold(fn (acc: i64, code: i64) -> acc |> mul(10i64) |> add(sub(code, 48i64)), 0i64, codes)
-- The value of a well-formed token. An all-zero significand is zero before
-- the exponent is read; otherwise the significant digits are counted and the
-- exponent's digit count is compared with 4 before its value is read, so
-- no step does arithmetic on an unbounded exponent.
def dec_token_value(text: string, negative: bool, significand: List[i64], fraction: i64, exponent_negative: bool, exponent_digits: List[i64]) -> (string, string, Decimal) = {
  leading = dec_leading_zeros(significand)
  if eq(leading, len(significand)) then dec_ok(dec_zero()) else {
    trailing = dec_trailing_zero_codes(significand)
    count = sub(sub(len(significand), leading), trailing)
    exponent_leading = dec_leading_zeros(exponent_digits)
    exponent_count = exponent_digits |> len |> sub(exponent_leading)
    if count |> gt(dec_max_scale()) |> or(gt(exponent_count, 4i64)) then dec_domain(dec_range_detail(dec_shown(text))) else {
      exponent_size = exponent_digits |> skip(exponent_leading) |> dec_small_digits
      exponent = if exponent_negative then neg(exponent_size) else exponent_size
      point = exponent |> sub(fraction) |> add(trailing)
      if point |> lt(neg(dec_max_scale())) |> or(gt(add(point, count), dec_max_scale())) then dec_domain(dec_range_detail(dec_shown(text))) else {
        coefficient = dec_digit_magnitude(take(skip(significand, leading), count))
        if lt(point, 0i64) then negative |> dec_value(coefficient, neg(point)) |> dec_ok else negative |> dec_value(dec_scale_up(coefficient, point), 0i64) |> dec_ok
      }
    }
  }
}
-- `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`, at most 1000 scalars.
def dec_parse(text: string) -> (string, string, Decimal) = {
  size = string_len(text)
  if gt(size, 1000i64) then dec_domain(dec_joined(["number text has ", to_string(size), " characters, more than 1000"])) else {
    codes = map(fn (idx: i64) -> text |> string_slice(idx, 1i64) |> char_code, range(0i64, size))
    negative = codes |> dec_code(0i64) |> eq(45i64)
    int_start = if negative then 1i64 else 0i64
    int_end = dec_digit_run_end(codes, int_start)
    has_point = codes |> dec_code(int_end) |> eq(46i64)
    frac_start = if has_point then add(int_end, 1i64) else int_end
    frac_end = if has_point then dec_digit_run_end(codes, frac_start) else int_end
    marker = dec_code(codes, frac_end)
    has_exponent = marker |> eq(101i64) |> or(eq(marker, 69i64))
    sign = dec_code(codes, add(frac_end, 1i64))
    signed = and(has_exponent, sign |> eq(43i64) |> or(eq(sign, 45i64)))
    sign_width = if signed then 1i64 else 0i64
    exp_start = if has_exponent then frac_end |> add(1i64) |> add(sign_width) else frac_end
    exp_end = if has_exponent then dec_digit_run_end(codes, exp_start) else frac_end
    integer_ok = and(gt(int_end, int_start), or(neq(dec_code(codes, int_start), 48i64), eq(int_end, add(int_start, 1i64))))
    fraction_ok = has_point |> not |> or(gt(frac_end, frac_start))
    exponent_ok = has_exponent |> not |> or(gt(exp_end, exp_start))
    well_formed =
      integer_ok
      |> and(fraction_ok)
      |> and(and(exponent_ok, eq(exp_end, size)))
    significand = concat(dec_code_slice(codes, int_start, int_end), dec_code_slice(codes, frac_start, frac_end))
    if well_formed then dec_token_value(text, negative, significand, sub(frac_end, frac_start), and(signed, eq(sign, 45i64)), dec_code_slice(codes, exp_start, exp_end)) else ["malformed number text \"", dec_shown(text), "\""] |> dec_joined |> dec_domain
  }
}
-- Binary floats. `dec_binary_parts(y)` writes a finite y > 0 as m * 2^e with
-- 2^52 <= m < 2^53, scaling only by powers of two (each step exact) and then
-- casting the integral m exactly.
def dec_two_powers() -> List[(i64, f64)] = {
  squares = concat([2.0f64], scan(fn (p: f64, unused: i64) -> mul(p, p), 2.0f64, range(0i64, 9i64)))
  map(fn (idx: i64) -> (shl(1i64, sub(9i64, idx)), index(squares, sub(9i64, idx))), range(0i64, 10i64))
}
def dec_binary_parts(y: f64) -> (i64, i64) = {
  steps = dec_two_powers()
  low = 4503599627370496.0f64
  high = 9007199254740992.0f64
  lowered = fold(fn (state: (f64, i64), step: (i64, f64)) -> if gte(state.0, mul(low, step.1)) then (div(state.0, step.1), add(state.1, step.0)) else state, (y, 0i64), steps)
  raised = fold(fn (state: (f64, i64), step: (i64, f64)) -> if lt(state.0, div(high, step.1)) then (mul(state.0, step.1), sub(state.1, step.0)) else state, lowered, concat([index(steps, 0i64)], steps))
  (cast(raised.0, i64), raised.1)
}
-- 2^exponent as an f64 for |exponent| < 1023, exact.
def dec_pow2_f64(exponent: i64) -> f64 = {
  size = if lt(exponent, 0i64) then neg(exponent) else exponent
  power = fold(fn (acc: f64, step: (i64, f64)) -> if size |> bitand(step.0) |> eq(0i64) then acc else mul(acc, step.1), 1.0f64, dec_two_powers())
  if lt(exponent, 0i64) then div(1.0f64, power) else power
}
def dec_is_finite(value: f64) -> bool = value |> sub(value) |> eq(0.0f64)
-- The magnitude of any i64, the minimum included, read through its negation.
def dec_i64_magnitude(value: i64) -> List[i64] = {
  down = if gt(value, 0i64) then neg(value) else value
  rest = trunc_div(down, dec_base())
  dec_trim([down |> mod(dec_base()) |> neg, rest |> mod(dec_base()) |> neg, rest |> trunc_div(dec_base()) |> neg])
}
-- A finite nonzero f64 is m * 2^e. At e >= 75 it is at least 2^127 > 10^38.
-- Otherwise m * 2^e * 10^places is rounded in limbs; a divisor exponent past
-- 182 is clamped, since m * 10^places < 2^180 keeps the quotient zero and
-- the remainder below half either way.
def dec_from_f64_outcome(value: f64, places: i64, mode: Rounding) -> (string, string, Decimal) =
  if value |> dec_is_finite |> not then dec_domain(string_concat(to_string(value), " is not finite")) else if dec_scale_outside(places) then places |> dec_scale_detail |> dec_domain else if eq(value, 0.0f64) then dec_ok(dec_zero()) else {
    negative = lt(value, 0.0f64)
    parts = value |> abs |> dec_binary_parts
    if gte(parts.1, 75i64) then dec_domain(dec_range_detail(to_string(value))) else {
      num = dec_times_pow2(dec_scale_up(dec_i64_magnitude(parts.0), places), dec_max(parts.1, 0i64))
      den = dec_times_pow2([1i64], dec_min(dec_max(neg(parts.1), 0i64), 182i64))
      rounded = dec_round_quotient(num, den, negative, mode)
      if rounded.1 |> not |> and(dec_rejects(mode)) then dec_domain(dec_inexact_detail(to_string(value), places)) else {
        kept = dec_canonical(rounded.0, places)
        if dec_fits(kept.0, kept.1) then negative |> dec_value(kept.0, kept.1) |> dec_ok else dec_domain(dec_range_detail(to_string(value)))
      }
    }
  }
-- c * 2^-shift / 10^s as an integer fraction (numerator, denominator).
def dec_binary_fraction(xs: List[i64], scale: i64, shift: i64) -> (List[i64], List[i64]) = if lte(shift, 0i64) then (dec_times_pow2(xs, neg(shift)), dec_pow10(scale)) else (xs, scale |> dec_pow10 |> dec_times_pow2(shift))
-- Arithmetic outcomes.
def dec_div_outcome(a: Decimal, b: Decimal, places: i64, mode: Rounding) -> (string, string, Decimal) =
  if dec_is_zero(b) then dec_domain("division by zero") else if dec_scale_outside(places) then places |> dec_scale_detail |> dec_domain else {
    shift = places |> add(b.scale) |> sub(a.scale)
    num = if gte(shift, 0i64) then a |> dec_magnitude |> dec_scale_up(shift) else dec_magnitude(a)
    den = if gte(shift, 0i64) then dec_magnitude(b) else b |> dec_magnitude |> dec_scale_up(neg(shift))
    negative = neq(a.negative, b.negative)
    rounded = dec_round_quotient(num, den, negative, mode)
    quotient = dec_joined([decimal_to_string(a), " divided by ", decimal_to_string(b)])
    if rounded.1 |> not |> and(dec_rejects(mode)) then quotient |> dec_inexact_detail(places) |> dec_domain else {
      kept = dec_canonical(rounded.0, places)
      if dec_fits(kept.0, kept.1) then negative |> dec_value(kept.0, kept.1) |> dec_ok else dec_overflow(dec_range_detail(dec_joined([quotient, " rounded to scale ", to_string(places)])))
    }
  }
def dec_i64_bound(negative: bool) -> List[i64] = if negative then [854775808i64, 223372036i64, 9i64] else [854775807i64, 223372036i64, 9i64]
-- (kind, detail, value) for decimal_to_i64. The value is accumulated toward
-- negative so the i64 minimum is reached without overflow.
def dec_to_i64_outcome(x: Decimal, mode: Rounding) -> (string, string, i64) = {
  rounded = dec_round_to(x, 0i64, mode)
  if rounded.2 |> not |> and(dec_rejects(mode)) then ("domain", x |> decimal_to_string |> dec_inexact_detail(0i64), 0i64) else if rounded.0 |> dec_cmp(dec_i64_bound(x.negative)) |> gt(0i64) then ("overflow", x |> decimal_to_string |> string_concat(" rounds to an integer outside i64"), 0i64) else {
    lowered = fold(fn (acc: i64, step: i64) -> acc |> mul(dec_base()) |> sub(dec_limb(rounded.0, sub(2i64, step))), 0i64, range(0i64, 3i64))
    ("", "", if x.negative then lowered else neg(lowered))
  }
}
def dec_order(a: Decimal, b: Decimal) -> i64 =
  if neq(a.negative, b.negative) then if a.negative then -1i64 else 1i64 else {
    scale = dec_max(a.scale, b.scale)
    order = dec_cmp(dec_scale_up(dec_magnitude(a), sub(scale, a.scale)), dec_scale_up(dec_magnitude(b), sub(scale, b.scale)))
    if a.negative then neg(order) else order
  }
-- a + (-1)^b_negative * |b| at the wider scale.
def dec_sum(function: string, a: Decimal, b: Decimal, b_negative: bool, verb: string) -> Decimal = {
  scale = dec_max(a.scale, b.scale)
  left = a |> dec_magnitude |> dec_scale_up(sub(scale, a.scale))
  right = b |> dec_magnitude |> dec_scale_up(sub(scale, b.scale))
  order = dec_cmp(left, right)
  same = eq(a.negative, b_negative)
  negative = if or(same, gte(order, 0i64)) then a.negative else b_negative
  total = if same then dec_add(left, right) else if gte(order, 0i64) then dec_sub(left, right) else dec_sub(right, left)
  kept = dec_canonical(total, scale)
  if dec_fits(kept.0, kept.1) then dec_value(negative, kept.0, kept.1) else fail(dec_failure(function, "overflow", dec_range_detail(dec_joined([decimal_to_string(a), verb, decimal_to_string(b)]))))
}
-- Text.
def decimal(text: string) -> Decimal = dec_result("decimal", dec_parse(text))
def try_decimal(text: string) -> Option[Decimal] = dec_try_result("try_decimal", dec_parse(text))
def decimal_to_string(x: Decimal) -> string = dec_text(x.negative, x |> dec_magnitude |> dec_digits, x.scale)
def decimal_to_fixed_string(x: Decimal, places: i64) -> string =
  if dec_scale_outside(places) then fail(dec_failure("decimal_to_fixed_string", "domain", dec_scale_detail(places))) else if gt(x.scale, places) then fail(dec_failure("decimal_to_fixed_string", "domain", dec_joined([decimal_to_string(x), " has ", to_string(x.scale), " fractional digits, more than ", to_string(places)]))) else {
    point = if x.scale |> eq(0i64) |> and(gt(places, 0i64)) then "." else ""
    dec_joined([decimal_to_string(x), point, places |> sub(x.scale) |> dec_zeros_text])
  }
-- Integers.
def decimal_from_i64(value: i64) -> Decimal = value |> lt(0i64) |> dec_value(dec_i64_magnitude(value), 0i64)
def decimal_to_i64(x: Decimal, mode: Rounding) -> i64 = {
  outcome = dec_to_i64_outcome(x, mode)
  if eq(outcome.0, "") then outcome.2 else "decimal_to_i64" |> dec_failure(outcome.0, outcome.1) |> fail
}
def try_decimal_to_i64(x: Decimal, mode: Rounding) -> Option[i64] = {
  outcome = dec_to_i64_outcome(x, mode)
  if eq(outcome.0, "") then Some(outcome.2) else None
}
-- Binary floats.
def decimal_from_f64(value: f64, places: i64, mode: Rounding) -> Decimal = dec_result("decimal_from_f64", dec_from_f64_outcome(value, places, mode))
def try_decimal_from_f64(value: f64, places: i64, mode: Rounding) -> Option[Decimal] = dec_try_result("try_decimal_from_f64", dec_from_f64_outcome(value, places, mode))
-- The canonical text is always inside `to_float`'s grammar, which rounds it
-- correctly once.
def decimal_to_f64(x: Decimal) -> f64 =
  match x |> decimal_to_string |> to_float with {
    | Some(value) => value
    | None => fail(dec_failure("decimal_to_f64", "domain", dec_joined(["canonical text \"", decimal_to_string(x), "\" is not float text"])))
  }
-- The exact value rounded once to f32 with ties to even. The f64 nearest the
-- magnitude v has binary exponent p = floor(log2 v), or p + 1 when it rounded
-- up to 2^(p+1); v then lies within 2^(p-53) of 2^(p+1), so rounding on that
-- one-bit-coarser grid also gives 2^(p+1), the f32 result. The significand
-- q = round(c * 2^-shift / 10^s) is formed exactly in limbs, with the shift
-- clamped at -149 for subnormal results, so q * 2^shift is exact in f64 and
-- in f32 and the final cast does not round.
def decimal_to_f32(x: Decimal) -> f32 =
  if dec_is_zero(x) then 0.0f32 else {
    magnitude = dec_magnitude(x)
    nearest = false |> dec_value(magnitude, x.scale) |> decimal_to_f64
    shift = dec_binary_parts(nearest).1 |> add(29i64) |> dec_max(-149i64)
    fraction = dec_binary_fraction(magnitude, x.scale, shift)
    rounded = dec_round_quotient(fraction.0, fraction.1, false, RoundTiesToEven).0
    significand = cast(add(mul(dec_limb(rounded, 1i64), dec_base()), dec_limb(rounded, 0i64)), f64)
    result = mul(significand, dec_pow2_f64(shift))
    cast(if x.negative then neg(result) else result, f32)
  }
def decimal_scale(x: Decimal) -> i64 = x.scale
-- Arithmetic.
def decimal_add(a: Decimal, b: Decimal) -> Decimal = dec_sum("decimal_add", a, b, b.negative, " plus ")
def decimal_sub(a: Decimal, b: Decimal) -> Decimal = dec_sum("decimal_sub", a, b, not(b.negative), " minus ")
def decimal_mul(a: Decimal, b: Decimal) -> Decimal = {
  negative = neq(a.negative, b.negative)
  kept = dec_canonical(dec_mul(dec_magnitude(a), dec_magnitude(b)), add(a.scale, b.scale))
  if dec_fits(kept.0, kept.1) then dec_value(negative, kept.0, kept.1) else fail(dec_failure("decimal_mul", "overflow", dec_range_detail(dec_joined([decimal_to_string(a), " times ", decimal_to_string(b)]))))
}
def decimal_round(x: Decimal, places: i64, mode: Rounding) -> Decimal =
  if dec_scale_outside(places) then "decimal_round" |> dec_failure("domain", dec_scale_detail(places)) |> fail else {
    rounded = dec_round_to(x, places, mode)
    if rounded.2 |> not |> and(dec_rejects(mode)) then fail(dec_failure("decimal_round", "domain", x |> decimal_to_string |> dec_inexact_detail(places))) else dec_value(x.negative, rounded.0, rounded.1)
  }
def decimal_div(a: Decimal, b: Decimal, places: i64, mode: Rounding) -> Decimal = dec_result("decimal_div", dec_div_outcome(a, b, places, mode))
def try_decimal_div(a: Decimal, b: Decimal, places: i64, mode: Rounding) -> Option[Decimal] = dec_try_result("try_decimal_div", dec_div_outcome(a, b, places, mode))
-- Order.
def decimal_lt(a: Decimal, b: Decimal) -> bool = a |> dec_order(b) |> lt(0i64)
def decimal_lte(a: Decimal, b: Decimal) -> bool = a |> dec_order(b) |> lte(0i64)
def decimal_gt(a: Decimal, b: Decimal) -> bool = a |> dec_order(b) |> gt(0i64)
def decimal_gte(a: Decimal, b: Decimal) -> bool = a |> dec_order(b) |> gte(0i64)
