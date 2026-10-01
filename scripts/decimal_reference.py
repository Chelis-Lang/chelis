#!/usr/bin/env python3
"""Independent reference semantics for `Std.Decimal` ([05-OP-76], chelis#2778).

This module is the oracle the differential harness compares `chelis eval` and
compiled C against. It is written from the contract alone, the [05-OP-76] atom
and the rounding table of `spec/design/std_decimal.md` §5, never by translating
Chelis source. Python integers and `fractions.Fraction` are exact, so every
value here is the exact mathematical answer, and each rounding is applied once,
directly to that exact rational:

- a decimal rounding (`decimal_round`, `decimal_div`, `decimal_from_f64`,
  `decimal_to_i64`) scales the rational by `10^n`, splits it into its floor and
  a remainder in `[0, 1)`, and chooses the multiple by the §5 table;
- a binary rounding (`decimal_to_f64`, `decimal_to_f32`) finds the exponent of
  the rational's leading bit, fixes the unit in the last place at
  `2^(e - p + 1)` for precision `p` (53 or 24) but never below the smallest
  subnormal (`2^-1074` or `2^-149`), and rounds the scaled rational to the
  nearest integer with ties to even. No value passes through another float
  format on the way.

A `Decimal` is its canonical `(coefficient, scale)` pair. Failures raise
`DecimalError`, whose `function` and `kind` the contract determines. The
`<detail>` follows the message shapes pinned for the module's tests: `detail`
holds the exact text where the shape determines it, and `pattern` is a regular
expression over the detail that every conforming spelling matches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
import math
import re
import unicodedata

MAX_DIGITS = 38
MAX_SCALE = 38
MAX_COEFFICIENT = 10**MAX_DIGITS - 1
TEXT_LIMIT = 1000
TRUNCATE_AT = 40
LIMB_BASE = 10**9
LIMB_COUNT = 5
I64_MIN = -(2**63)
I64_MAX = 2**63 - 1

ROUNDINGS = (
    "RoundTowardNegative",
    "RoundTowardPositive",
    "RoundTowardZero",
    "RoundAwayFromZero",
    "RoundTiesToEven",
    "RoundTiesToAway",
    "RejectInexact",
)

# The RFC 8259 number token the atom states, matched against the whole text.
TOKEN = re.compile(r"(-?)(0|[1-9][0-9]*)(?:\.([0-9]+))?(?:[eE]([+-]?[0-9]+))?", re.ASCII)

# Binary interchange formats: (significand precision, exponent field width).
F64 = (53, 11)
F32 = (24, 8)


# Why a call fails, one name per pinned message shape.
REASONS = ("malformed", "too_long", "outside", "scale", "division_by_zero", "inexact", "not_finite", "i64",
           "fixed_digits", "overflow")


class DecimalError(Exception):
    """A failure `<function>: <kind>: <detail>`, with the `reason` that selects its shape."""

    def __init__(self, function: str, kind: str, reason: str, detail: str | None = None,
                 pattern: str | None = None) -> None:
        if kind not in ("domain", "overflow"):
            raise ValueError(f"unknown failure kind {kind!r}")
        if reason not in REASONS:
            raise ValueError(f"unknown failure reason {reason!r}")
        if detail is None and pattern is None:
            raise ValueError("a failure needs an exact detail or a detail pattern")
        self.function = function
        self.kind = kind
        self.reason = reason
        self.detail = detail
        self.pattern = re.escape(detail) if detail is not None else pattern
        super().__init__(self.message)

    @property
    def message(self) -> str:
        shown = self.detail if self.detail is not None else f"/{self.pattern}/"
        return f"{self.function}: {self.kind}: {shown}"

    def matches(self, message: str) -> bool:
        """Whether a module's message has this function, kind, and an admitted detail."""
        prefix = f"{self.function}: {self.kind}: "
        if not message.startswith(prefix):
            return False
        return re.fullmatch(self.pattern, message[len(prefix):], re.DOTALL) is not None


def domain(function: str, reason: str, detail: str | None = None, pattern: str | None = None) -> DecimalError:
    return DecimalError(function, "domain", reason, detail, pattern)


def overflow(function: str, reason: str, detail: str | None = None, pattern: str | None = None) -> DecimalError:
    return DecimalError(function, "overflow", reason, detail, pattern)


OUTSIDE_RANGE = r".+ is outside the decimal range"


def inexact_pattern(n: int) -> str:
    return rf".+ is not a multiple of 10\^-{n}"


def fits_i64(value: int) -> bool:
    return I64_MIN <= value <= I64_MAX


# ---------------------------------------------------------------------------
# The value set and canonical form.


@dataclass(frozen=True)
class Decimal:
    """The canonical representation `coefficient / 10^scale`."""

    coefficient: int
    scale: int

    def __post_init__(self) -> None:
        if not 0 <= self.scale <= MAX_SCALE or abs(self.coefficient) > MAX_COEFFICIENT:
            raise ValueError(f"({self.coefficient}, {self.scale}) is outside the value set")
        if self.coefficient == 0 and self.scale != 0:
            raise ValueError("zero has scale 0")
        if self.scale > 0 and self.coefficient % 10 == 0:
            raise ValueError(f"({self.coefficient}, {self.scale}) has a removable trailing zero")

    @property
    def value(self) -> Fraction:
        return Fraction(self.coefficient, 10**self.scale)

    def fields(self) -> tuple[bool, int, int, int, int, int, int]:
        """The registered field tuple `(negative, limb0, ..., limb4, scale)`."""
        magnitude = abs(self.coefficient)
        limbs = []
        for _ in range(LIMB_COUNT):
            magnitude, limb = divmod(magnitude, LIMB_BASE)
            limbs.append(limb)
        assert magnitude == 0 and limbs[-1] < 100
        return (self.coefficient < 0, *limbs, self.scale)

    def __str__(self) -> str:
        return decimal_to_string(self)


ZERO = Decimal(0, 0)


def from_fields(negative: bool, limb0: int, limb1: int, limb2: int, limb3: int, limb4: int, scale: int) -> Decimal:
    """The inverse of `Decimal.fields`, for tests of the encoding."""
    limbs = (limb0, limb1, limb2, limb3, limb4)
    if any(not 0 <= limb < LIMB_BASE for limb in limbs) or limb4 >= 100:
        raise ValueError("a limb is outside its range")
    magnitude = sum(limb * LIMB_BASE**k for k, limb in enumerate(limbs))
    if negative and magnitude == 0:
        raise ValueError("zero is not negative")
    return Decimal(-magnitude if negative else magnitude, scale)


def canonical(q: Fraction) -> Decimal | None:
    """The canonical decimal denoting `q`, or None when `q` is outside the value set.

    `q = n/d` in lowest terms has a finite decimal expansion exactly when `d` is
    `2^a 5^b`, and then its least scale is `max(a, b)`: the canonical scale.
    """
    if q == 0:
        return ZERO
    d = q.denominator
    twos = fives = 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        return None
    scale = max(twos, fives)
    if scale > MAX_SCALE:
        return None
    coefficient = q.numerator * 10**scale // q.denominator
    if abs(coefficient) > MAX_COEFFICIENT:
        return None
    return Decimal(coefficient, scale)


def in_value_set(q: Fraction) -> bool:
    return canonical(q) is not None


# ---------------------------------------------------------------------------
# Text.


def truncated(text: str) -> str:
    return text[:TRUNCATE_AT] + "..." if len(text) > TRUNCATE_AT else text


def needs_escaping(text: str) -> bool:
    """Whether the contract leaves this text's spelling inside a message open."""
    return any(ch in '"\\' or unicodedata.category(ch) == "Cc" for ch in text)


def malformed(function: str, text: str) -> DecimalError:
    shown = truncated(text)
    if needs_escaping(shown):
        return domain(function, "malformed", pattern=r'malformed number text ".*"')
    return domain(function, "malformed", f'malformed number text "{shown}"')


def decimal(text: str, function: str = "decimal") -> Decimal:
    """`decimal(text)`: one RFC 8259 number token of at most 1000 scalar values.

    The length bound is the cost guard, so it is checked first, for every text,
    before the grammar.
    """
    if len(text) > TEXT_LIMIT:
        raise domain(function, "too_long", f"number text has {len(text)} characters, more than {TEXT_LIMIT}")
    match = TOKEN.fullmatch(text)
    if match is None:
        raise malformed(function, text)
    sign, whole, fraction, exponent_text = match.groups()
    fraction = fraction or ""
    digits = (whole + fraction).lstrip("0")
    if not digits:
        return ZERO
    stripped = digits.rstrip("0")
    # The value is ±stripped·10^exponent, and `stripped` ends in a nonzero digit.
    exponent = int(exponent_text or "0") - len(fraction) + (len(digits) - len(stripped))
    outside = domain(function, "outside", f"{truncated(text)} is outside the decimal range")
    if exponent >= 0:
        if len(stripped) + exponent > MAX_DIGITS:
            raise outside
        coefficient, scale = int(stripped) * 10**exponent, 0
    else:
        if -exponent > MAX_SCALE or len(stripped) > MAX_DIGITS:
            raise outside
        coefficient, scale = int(stripped), -exponent
    return Decimal(-coefficient if sign else coefficient, scale)


def try_decimal(text: str) -> Decimal | None:
    return try_domain(lambda: decimal(text))


def render(coefficient: int, scale: int) -> str:
    """`coefficient / 10^scale` with exactly `scale` fractional digits."""
    digits = str(abs(coefficient)).rjust(scale + 1, "0")
    sign = "-" if coefficient < 0 else ""
    if scale == 0:
        return sign + digits
    return f"{sign}{digits[:-scale]}.{digits[-scale:]}"


def decimal_to_string(x: Decimal) -> str:
    return render(x.coefficient, x.scale)


def check_scale(function: str, n: int) -> None:
    if not 0 <= n <= MAX_SCALE:
        raise domain(function, "scale", f"scale {n} is outside 0..{MAX_SCALE}")


def decimal_to_fixed_string(x: Decimal, n: int) -> str:
    function = "decimal_to_fixed_string"
    check_scale(function, n)
    if x.scale > n:
        raise domain(function, "fixed_digits", f"{decimal_to_string(x)} has {x.scale} fractional digits, more than {n}")
    return render(x.coefficient * 10 ** (n - x.scale), n)


def decimal_scale(x: Decimal) -> int:
    return x.scale


# ---------------------------------------------------------------------------
# Rounding to a decimal quantum (§5).


def round_to_quantum(v: Fraction, n: int, mode: str, function: str, shown: Callable[[], str] | None = None) -> Fraction:
    """`v` rounded once to an integer multiple `k·10^-n` by `mode`.

    `shown` renders `v` for a `RejectInexact` failure when the contract
    determines its spelling; otherwise the failure carries a pattern.
    """
    if mode not in ROUNDINGS:
        raise ValueError(f"unknown rounding {mode!r}")
    scaled = v * 10**n
    k = scaled.numerator // scaled.denominator
    rest = scaled - k
    if rest == 0:
        return v
    half = Fraction(1, 2)
    if mode == "RejectInexact":
        if shown is None:
            raise domain(function, "inexact", pattern=inexact_pattern(n))
        raise domain(function, "inexact", f"{shown()} is not a multiple of 10^-{n}")
    if mode == "RoundTowardNegative":
        up = False
    elif mode == "RoundTowardPositive":
        up = True
    elif mode == "RoundTowardZero":
        up = v < 0
    elif mode == "RoundAwayFromZero":
        up = v > 0
    elif mode == "RoundTiesToEven":
        up = rest > half or (rest == half and k % 2 == 1)
    else:  # RoundTiesToAway
        up = rest > half or (rest == half and v > 0)
    return Fraction(k + 1 if up else k, 10**n)


def decimal_round(x: Decimal, n: int, mode: str) -> Decimal:
    function = "decimal_round"
    check_scale(function, n)
    result = canonical(round_to_quantum(x.value, n, mode, function, lambda: decimal_to_string(x)))
    if result is None:
        raise AssertionError(f"decimal_round({x}, {n}, {mode}) left the value set, which the contract excludes")
    return result


# ---------------------------------------------------------------------------
# Integers.


def decimal_from_i64(v: int) -> Decimal:
    if not fits_i64(v):
        raise ValueError(f"{v} is not an i64")
    result = canonical(Fraction(v))
    assert result is not None
    return result


def decimal_to_i64(x: Decimal, mode: str, function: str = "decimal_to_i64") -> int:
    rounded = round_to_quantum(x.value, 0, mode, function, lambda: decimal_to_string(x))
    assert rounded.denominator == 1
    if not fits_i64(rounded.numerator):
        raise overflow(function, "i64", f"{decimal_to_string(x)} rounds to an integer outside i64")
    return rounded.numerator


def try_decimal_to_i64(x: Decimal, mode: str) -> int | None:
    try:
        return decimal_to_i64(x, mode, "try_decimal_to_i64")
    except DecimalError:
        return None


# ---------------------------------------------------------------------------
# Binary floats.


def float_text(x: float) -> str:
    """`to_string` of a non-finite f64, whose spellings spec/05 §8.1 pins."""
    if math.isnan(x):
        return "NaN"
    return "inf" if x > 0 else "-inf"


def decimal_from_f64(x: float, n: int, mode: str, function: str = "decimal_from_f64") -> Decimal:
    if math.isnan(x) or math.isinf(x):
        raise domain(function, "not_finite", f"{float_text(x)} is not finite")
    check_scale(function, n)
    exact = Fraction(*x.as_integer_ratio())
    result = canonical(round_to_quantum(exact, n, mode, function))
    if result is None:
        raise domain(function, "outside", pattern=OUTSIDE_RANGE)
    return result


def try_decimal_from_f64(x: float, n: int, mode: str) -> Decimal | None:
    return try_domain(lambda: decimal_from_f64(x, n, mode, "try_decimal_from_f64"))


def floor_log2(q: Fraction) -> int:
    """The exponent `e` with `2^e <= q < 2^(e+1)`, for `q > 0`."""
    if q <= 0:
        raise ValueError("floor_log2 needs a positive rational")
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if q < Fraction(2) ** e:
        e -= 1
    elif q >= Fraction(2) ** (e + 1):
        e += 1
    assert Fraction(2) ** e <= q < Fraction(2) ** (e + 1)
    return e


def round_binary(q: Fraction, fmt: tuple[int, int]) -> Fraction:
    """`q` correctly rounded once to the binary format, ties to even, subnormals included."""
    precision, exponent_bits = fmt
    bias = 2 ** (exponent_bits - 1) - 1
    min_normal = 1 - bias
    lowest_unit = min_normal - (precision - 1)
    if q == 0:
        return Fraction(0)
    magnitude = abs(q)
    unit = max(floor_log2(magnitude) - (precision - 1), lowest_unit)
    scaled = magnitude / Fraction(2) ** unit
    m = scaled.numerator // scaled.denominator
    rest = scaled - m
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and m % 2 == 1):
        m += 1
    rounded = m * Fraction(2) ** unit
    largest = (2 - Fraction(2) ** (1 - precision)) * Fraction(2) ** bias
    if rounded > largest:
        raise OverflowError("the rounded value exceeds the format's finite range")
    return rounded if q > 0 else -rounded


def binary_bits(v: Fraction, fmt: tuple[int, int], negative: bool | None = None) -> int:
    """The interchange encoding of a value the format represents exactly.

    `negative` gives zero's sign; a nonzero value carries its own.
    """
    precision, exponent_bits = fmt
    fraction_bits = precision - 1
    bias = 2 ** (exponent_bits - 1) - 1
    sign = (v < 0) if v != 0 or negative is None else negative
    magnitude = abs(v)
    if magnitude == 0:
        field_exponent, field_fraction = 0, 0
    else:
        e = floor_log2(magnitude)
        if e < 1 - bias:
            scaled = magnitude / Fraction(2) ** (1 - bias - fraction_bits)
            field_exponent = 0
        else:
            scaled = magnitude / Fraction(2) ** (e - fraction_bits)
            field_exponent = e + bias
        if scaled.denominator != 1:
            raise ValueError(f"{v} is not exactly representable")
        field_fraction = scaled.numerator - (2**fraction_bits if field_exponent else 0)
        if field_exponent >= 2**exponent_bits - 1 or not 0 <= field_fraction < 2**fraction_bits:
            raise ValueError(f"{v} is not a finite value of the format")
    return (int(sign) << (exponent_bits + fraction_bits)) | (field_exponent << fraction_bits) | field_fraction


def bits_value(bits: int, fmt: tuple[int, int]) -> Fraction:
    """The exact value of a finite interchange encoding (the inverse of `binary_bits`)."""
    precision, exponent_bits = fmt
    fraction_bits = precision - 1
    bias = 2 ** (exponent_bits - 1) - 1
    sign = bits >> (exponent_bits + fraction_bits)
    field_exponent = (bits >> fraction_bits) & (2**exponent_bits - 1)
    field_fraction = bits & (2**fraction_bits - 1)
    if field_exponent == 2**exponent_bits - 1:
        raise ValueError("not a finite encoding")
    if field_exponent == 0:
        magnitude = field_fraction * Fraction(2) ** (1 - bias - fraction_bits)
    else:
        magnitude = (2**fraction_bits + field_fraction) * Fraction(2) ** (field_exponent - bias - fraction_bits)
    return -magnitude if sign else magnitude


def decimal_to_f64(x: Decimal) -> float:
    rounded = round_binary(x.value, F64)
    return float(rounded)  # exact: `rounded` is an f64 value


def decimal_to_f64_bits(x: Decimal) -> int:
    return binary_bits(round_binary(x.value, F64), F64, negative=False)


def decimal_to_f32_bits(x: Decimal) -> int:
    return binary_bits(round_binary(x.value, F32), F32, negative=False)


def decimal_to_f32(x: Decimal) -> float:
    """The f32 result as the Python float of the same exact value."""
    return float(round_binary(x.value, F32))


def text_bits(text: str, fmt: tuple[int, int]) -> int | None:
    """The encoding a printed shortest round-trip float text denotes at its own width.

    Printed floats round-trip at their width, so rounding the text's exact value
    once recovers the stored bits; zero keeps the printed sign.
    """
    negative = text.startswith("-")
    body = text[1:] if negative else text
    if body in ("inf", "NaN"):
        return None
    try:
        magnitude = Fraction(body)
    except (ValueError, ZeroDivisionError):
        return None
    try:
        rounded = round_binary(-magnitude if negative else magnitude, fmt)
    except OverflowError:
        return None
    return binary_bits(rounded, fmt, negative=negative)


# ---------------------------------------------------------------------------
# Arithmetic and order.


def arithmetic(function: str, exact: Fraction) -> Decimal:
    result = canonical(exact)
    if result is None:
        raise overflow(function, "overflow", pattern=OUTSIDE_RANGE)
    return result


def decimal_add(a: Decimal, b: Decimal) -> Decimal:
    return arithmetic("decimal_add", a.value + b.value)


def decimal_sub(a: Decimal, b: Decimal) -> Decimal:
    return arithmetic("decimal_sub", a.value - b.value)


def decimal_mul(a: Decimal, b: Decimal) -> Decimal:
    return arithmetic("decimal_mul", a.value * b.value)


def decimal_div(a: Decimal, b: Decimal, n: int, mode: str, function: str = "decimal_div") -> Decimal:
    if b.coefficient == 0:
        raise domain(function, "division_by_zero", "division by zero")
    check_scale(function, n)
    rounded = round_to_quantum(a.value / b.value, n, mode, function)
    result = canonical(rounded)
    if result is None:
        raise overflow(function, "overflow", pattern=OUTSIDE_RANGE)
    return result


def try_decimal_div(a: Decimal, b: Decimal, n: int, mode: str) -> Decimal | None:
    """None where the twin fails `domain`; an `overflow` still fails, under this callable's name."""
    return try_domain(lambda: decimal_div(a, b, n, mode, "try_decimal_div"))


def decimal_lt(a: Decimal, b: Decimal) -> bool:
    return a.value < b.value


def decimal_lte(a: Decimal, b: Decimal) -> bool:
    return a.value <= b.value


def decimal_gt(a: Decimal, b: Decimal) -> bool:
    return a.value > b.value


def decimal_gte(a: Decimal, b: Decimal) -> bool:
    return a.value >= b.value


def decimal_eq(a: Decimal, b: Decimal) -> bool:
    """[05-OP-36] structural equality, which canonical form makes equality of the rationals."""
    return a.fields() == b.fields()


def try_domain(compute: Callable[[], object]):
    try:
        return compute()
    except DecimalError as error:
        if error.kind == "domain":
            return None
        raise
