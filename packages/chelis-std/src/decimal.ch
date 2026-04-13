module Std.Decimal
export (Decimal, RoundingMode, round_half_up, round_half_even, round_down, round_up, decimal, try_decimal, decimal_from_int, decimal_add, decimal_sub, decimal_mul, decimal_div, decimal_eq, decimal_lt, decimal_lte, decimal_gt, decimal_gte, decimal_to_float, decimal_to_string)
type RoundingMode =
  | RoundHalfUp
  | RoundHalfEven
  | RoundDown
  | RoundUp
def round_half_up() -> RoundingMode = RoundHalfUp
def round_half_even() -> RoundingMode = RoundHalfEven
def round_down() -> RoundingMode = RoundDown
def round_up() -> RoundingMode = RoundUp
type Decimal =
  | Decimal { coefficient: int64, scale: int64 }
def decimal(text: string) -> Decimal = { match try_decimal(text) with {
  | Some(value) => value
  | None => fail(string_concat("decimal: invalid literal ", text))
} }
def try_decimal(text: string) -> Option[Decimal] = {
  trimmed = string_trim(text)
  if eq(string_len(trimmed), cast(0, int64)) then None else {
    first = char_at(trimmed, cast(0, int64))
    negative = eq(first, "-")
    positive = eq(first, "+")
    start = if or(negative, positive) then cast(1, int64) else cast(0, int64)
    match parse_decimal_chars(trimmed, start, cast(0, int64), cast(0, int64), false, false) with {
      | Some(parts) => {
      (coefficient, scale, seen_digit) = parts
      if not(seen_digit) then None else {
        signed_coeff = if negative then sub(cast(0, int64), coefficient) else coefficient
        Some(normalize(Decimal { coefficient: signed_coeff, scale: scale }))
      }
    }
      | None => None
    }
  }
}
def decimal_from_int(value: int64) -> Decimal = { Decimal { coefficient: value, scale: cast(0, int64) } }
def decimal_add(lhs: Decimal, rhs: Decimal) -> Decimal = {
  scale = max_int(lhs.scale, rhs.scale)
  left = align_coeff(lhs.coefficient, lhs.scale, scale)
  right = align_coeff(rhs.coefficient, rhs.scale, scale)
  normalize(Decimal { coefficient: add(left, right), scale: scale })
}
def decimal_sub(lhs: Decimal, rhs: Decimal) -> Decimal = {
  scale = max_int(lhs.scale, rhs.scale)
  left = align_coeff(lhs.coefficient, lhs.scale, scale)
  right = align_coeff(rhs.coefficient, rhs.scale, scale)
  normalize(Decimal { coefficient: sub(left, right), scale: scale })
}
def decimal_mul(lhs: Decimal, rhs: Decimal) -> Decimal = { normalize(Decimal { coefficient: mul(lhs.coefficient, rhs.coefficient), scale: add(lhs.scale, rhs.scale) }) }
def decimal_div(lhs: Decimal, rhs: Decimal, result_scale: int64, mode: RoundingMode) -> Decimal = { if eq(rhs.coefficient, cast(0, int64)) then fail("decimal_div: division by zero") else decimal_div_nonzero(lhs, rhs, result_scale, mode) }
def decimal_eq(lhs: Decimal, rhs: Decimal) -> bool = {
  scale = max_int(lhs.scale, rhs.scale)
  eq(align_coeff(lhs.coefficient, lhs.scale, scale), align_coeff(rhs.coefficient, rhs.scale, scale))
}
def decimal_lt(lhs: Decimal, rhs: Decimal) -> bool = {
  scale = max_int(lhs.scale, rhs.scale)
  gt(align_coeff(rhs.coefficient, rhs.scale, scale), align_coeff(lhs.coefficient, lhs.scale, scale))
}
def decimal_lte(lhs: Decimal, rhs: Decimal) -> bool = { or(decimal_lt(lhs, rhs), decimal_eq(lhs, rhs)) }
def decimal_gt(lhs: Decimal, rhs: Decimal) -> bool = { not(decimal_lte(lhs, rhs)) }
def decimal_gte(lhs: Decimal, rhs: Decimal) -> bool = { not(decimal_lt(lhs, rhs)) }
def decimal_to_float(value: Decimal) -> f64 = { div(cast(value.coefficient, f64), cast(pow10(value.scale), f64)) }
def decimal_to_string(value: Decimal) -> string = {
  normalized = normalize(value)
  negative = gt(cast(0, int64), normalized.coefficient)
  coeff = abs_int(normalized.coefficient)
  digits = to_string(coeff)
  rendered = render_decimal_digits(normalized.scale, digits)
  if and(negative, neq(coeff, cast(0, int64))) then string_concat("-", rendered) else rendered
}
def normalize(value: Decimal) -> Decimal = { if and(gt(value.scale, cast(0, int64)), eq(mod(abs_int(value.coefficient), cast(10, int64)), cast(0, int64))) then normalize(Decimal { coefficient: div(value.coefficient, cast(10, int64)), scale: sub(value.scale, cast(1, int64)) }) else value }
def parse_decimal_chars(text: string, idx: int64, coefficient: int64, scale: int64, seen_dot: bool, seen_digit: bool) -> Option[(int64, int64, bool)] = { if gte(idx, string_len(text)) then Some((coefficient, scale, seen_digit)) else {
  ch = char_at(text, idx)
  if eq(ch, ".") then if seen_dot then None else parse_decimal_chars(text, add(idx, cast(1, int64)), coefficient, scale, true, seen_digit) else match digit_value(ch) with {
    | Some(digit) => parse_decimal_chars(text, add(idx, cast(1, int64)), add(mul(coefficient, cast(10, int64)), digit), if seen_dot then add(scale, cast(1, int64)) else scale, seen_dot, true)
    | None => None
  }
} }
def digit_value(ch: string) -> Option[int64] = { match ch with {
  | "0" => Some(cast(0, int64))
  | "1" => Some(cast(1, int64))
  | "2" => Some(cast(2, int64))
  | "3" => Some(cast(3, int64))
  | "4" => Some(cast(4, int64))
  | "5" => Some(cast(5, int64))
  | "6" => Some(cast(6, int64))
  | "7" => Some(cast(7, int64))
  | "8" => Some(cast(8, int64))
  | "9" => Some(cast(9, int64))
  | _ => None
} }
def align_coeff(coefficient: int64, current_scale: int64, target_scale: int64) -> int64 = { if gte(current_scale, target_scale) then coefficient else mul(coefficient, pow10(sub(target_scale, current_scale))) }
def scaled_numerator(coefficient: int64, scale_delta: int64) -> int64 = { if gte(scale_delta, cast(0, int64)) then mul(coefficient, pow10(scale_delta)) else coefficient }
def scaled_denominator(coefficient: int64, scale_delta: int64) -> int64 = { if gte(scale_delta, cast(0, int64)) then coefficient else mul(coefficient, pow10(sub(cast(0, int64), scale_delta))) }
def render_decimal_digits(scale: int64, digits: string) -> string = { if eq(scale, cast(0, int64)) then digits else if gt(string_len(digits), scale) then render_decimal_split(scale, digits) else string_concat("0.", string_concat(repeat_text("0", sub(scale, string_len(digits))), digits)) }
def render_decimal_split(scale: int64, digits: string) -> string = {
  head_len = sub(string_len(digits), scale)
  string_concat(string_slice(digits, cast(0, int64), head_len), string_concat(".", string_slice(digits, head_len, scale)))
}
def decimal_div_nonzero(lhs: Decimal, rhs: Decimal, result_scale: int64, mode: RoundingMode) -> Decimal = {
  scale_delta = sub(add(result_scale, rhs.scale), lhs.scale)
  numerator = scaled_numerator(lhs.coefficient, scale_delta)
  denominator = scaled_denominator(rhs.coefficient, scale_delta)
  normalize(Decimal { coefficient: divide_round(numerator, denominator, mode), scale: result_scale })
}
def divide_round(numerator: int64, denominator: int64, mode: RoundingMode) -> int64 = {
  same_sign = eq(gte(numerator, cast(0, int64)), gte(denominator, cast(0, int64)))
  q = div(abs_int(numerator), abs_int(denominator))
  r = mod(abs_int(numerator), abs_int(denominator))
  rounded = round_abs(q, r, abs_int(denominator), mode)
  if same_sign then rounded else sub(cast(0, int64), rounded)
}
def round_abs(q: int64, r: int64, denominator: int64, mode: RoundingMode) -> int64 = { if eq(r, cast(0, int64)) then q else match mode with {
  | RoundDown => q
  | RoundUp => add(q, cast(1, int64))
  | RoundHalfUp => if gte(mul(r, cast(2, int64)), denominator) then add(q, cast(1, int64)) else q
  | RoundHalfEven => {
  twice = mul(r, cast(2, int64))
  if gt(twice, denominator) then add(q, cast(1, int64)) else if gt(denominator, twice) then q else if eq(mod(q, cast(2, int64)), cast(0, int64)) then q else add(q, cast(1, int64))
}
  | _ => q
} }
def repeat_text(text: string, count: int64) -> string = { if lte(count, cast(0, int64)) then "" else string_concat(text, repeat_text(text, sub(count, cast(1, int64)))) }
def pow10(exp: int64) -> int64 = { if lte(exp, cast(0, int64)) then cast(1, int64) else mul(cast(10, int64), pow10(sub(exp, cast(1, int64)))) }
def abs_int(value: int64) -> int64 = { if gt(cast(0, int64), value) then sub(cast(0, int64), value) else value }
def max_int(lhs: int64, rhs: int64) -> int64 = { if gt(lhs, rhs) then lhs else rhs }
def char_at(text: string, idx: int64) -> string = { string_slice(text, idx, cast(1, int64)) }
