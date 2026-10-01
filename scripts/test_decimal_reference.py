#!/usr/bin/env python3
"""Unit tests for the `Std.Decimal` reference ([05-OP-76], chelis#2778).

The reference is the oracle the compiler is held to, so these tests hold it to
evidence it does not share code with: a second reading of the §5 rounding table
that compares the two neighbouring multiples, Python's `decimal` module where
its results are exact (a `ROUND_05UP` intermediate one digit past the quantum
before the one final rounding, never two roundings to nearest), Python's
correctly rounded `float(Fraction)` for f64, and for f32 a nearest-neighbour
search over the encodings around the value. Each accepting check has a
rejecting partner.
"""

from __future__ import annotations

import decimal as pydecimal
from fractions import Fraction
import math
import random
import struct
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decimal_reference as ref  # noqa: E402
from decimal_reference import Decimal, DecimalError, I64_MAX, I64_MIN, ROUNDINGS  # noqa: E402

SEED = 2778
NINES = "9" * 38
MAX_TEXT = NINES
MIN_TEXT = "-" + NINES
TINY_TEXT = "0." + "0" * 37 + "1"
PYTHON_MODES = {
    "RoundTowardNegative": pydecimal.ROUND_FLOOR,
    "RoundTowardPositive": pydecimal.ROUND_CEILING,
    "RoundTowardZero": pydecimal.ROUND_DOWN,
    "RoundAwayFromZero": pydecimal.ROUND_UP,
    "RoundTiesToEven": pydecimal.ROUND_HALF_EVEN,
    "RoundTiesToAway": pydecimal.ROUND_HALF_UP,
}
WIDE = pydecimal.Context(prec=400, Emax=pydecimal.MAX_EMAX, Emin=pydecimal.MIN_EMIN, traps=[])


def d(text: str) -> Decimal:
    return ref.decimal(text)


def failure(compute) -> DecimalError:
    try:
        compute()
    except DecimalError as error:
        return error
    raise AssertionError("expected a DecimalError")


def random_decimal(rng: random.Random) -> Decimal:
    scale = rng.randint(0, 38)
    digits = rng.randint(1, 38)
    coefficient = rng.randrange(10 ** (digits - 1), 10**digits) * rng.choice((1, -1))
    result = ref.canonical(Fraction(coefficient, 10**scale))
    assert result is not None
    return result


def table_round(v: Fraction, n: int, mode: str) -> Fraction | None:
    """The §5 table read literally: choose between the two neighbouring multiples."""
    q = Fraction(1, 10**n)
    k_low = math.floor(v / q)
    low, high = k_low * q, (k_low if k_low * q == v else k_low + 1) * q
    if low == high:
        return v
    if mode == "RejectInexact":
        return None
    nearer_zero, farther = (low, high) if abs(low) < abs(high) else (high, low)
    if mode == "RoundTowardNegative":
        return low
    if mode == "RoundTowardPositive":
        return high
    if mode == "RoundTowardZero":
        return nearer_zero
    if mode == "RoundAwayFromZero":
        return farther
    if v - low != high - v:
        return low if v - low < high - v else high
    if mode == "RoundTiesToEven":
        return low if (low / q) % 2 == 0 else high
    return farther


def python_round(v: pydecimal.Decimal, n: int, mode: str) -> pydecimal.Decimal:
    """One final rounding of an exact or ROUND_05UP-sticky Python decimal."""
    return v.quantize(pydecimal.Decimal(1).scaleb(-n), rounding=PYTHON_MODES[mode], context=WIDE)


def python_quotient(a: Decimal, b: Decimal, n: int, mode: str) -> pydecimal.Decimal:
    """`a / b` to one digit past `10^-n` with ROUND_05UP, then rounded once by `mode`."""
    sticky = pydecimal.Context(prec=400, rounding=pydecimal.ROUND_05UP, Emax=pydecimal.MAX_EMAX,
                               Emin=pydecimal.MIN_EMIN, traps=[])
    quotient = sticky.divide(pydecimal.Decimal(ref.decimal_to_string(a)), pydecimal.Decimal(ref.decimal_to_string(b)))
    intermediate = quotient.quantize(pydecimal.Decimal(1).scaleb(-(n + 1)), rounding=pydecimal.ROUND_05UP, context=sticky)
    return python_round(intermediate, n, mode)


def from_python(value: pydecimal.Decimal) -> Fraction:
    return Fraction(value)


def f32_by_search(q: Fraction) -> int:
    """The f32 encoding nearest `q` (ties to an even significand), by comparing neighbours."""
    if q == 0:
        return 0
    start = struct.unpack("<I", struct.pack("<f", float(q)))[0]
    candidates = []
    for bits in range(max(start - 4, 0), start + 5):
        if (bits >> 23) & 0xFF == 0xFF:
            continue
        value = Fraction(struct.unpack("<f", struct.pack("<I", bits))[0])
        candidates.append((abs(value - q), bits & 1, bits))
    candidates.sort()
    best = candidates[0]
    assert len(candidates) < 2 or candidates[1][0] != best[0] or best[1] == 0
    return best[2]


class ValueSet(unittest.TestCase):
    def test_canonical_form_is_unique(self) -> None:
        self.assertEqual(ref.canonical(Fraction(3, 2)), Decimal(15, 1))
        self.assertEqual(ref.canonical(Fraction(10)), Decimal(10, 0))
        self.assertEqual(ref.canonical(Fraction(0)), Decimal(0, 0))
        self.assertEqual(ref.canonical(Fraction(-1, 8)), Decimal(-125, 3))
        for bad in ((10, 1), (0, 3), (1, 39), (10**38, 0), (-(10**38), 2)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Decimal(*bad)

    def test_value_set_edges(self) -> None:
        self.assertEqual(ref.canonical(Fraction(10**38 - 1)), d(MAX_TEXT))
        self.assertEqual(ref.canonical(Fraction(1, 10**38)), d(TINY_TEXT))
        self.assertIsNone(ref.canonical(Fraction(10**38)))
        self.assertIsNone(ref.canonical(Fraction(1, 10**39)))
        self.assertIsNone(ref.canonical(Fraction(1, 3)))
        self.assertIsNone(ref.canonical(Fraction(10**38 + 1, 10)))
        self.assertIsNotNone(ref.canonical(Fraction(10**38 - 1, 10**38)))
        self.assertIsNone(ref.canonical(Fraction(10**38 + 1, 10**38)))
        # A large integer keeps its zeros in the coefficient: 10^37 fits, 10^38 does not.
        self.assertEqual(ref.canonical(Fraction(10**37)), Decimal(10**37, 0))

    def test_limb_encoding_round_trips_and_names_the_limbs(self) -> None:
        self.assertEqual(d("1").fields(), (False, 1, 0, 0, 0, 0, 0))
        self.assertEqual(d("1000000000").fields(), (False, 0, 1, 0, 0, 0, 0))
        self.assertEqual(d("-1e36").fields(), (True, 0, 0, 0, 0, 1, 0))
        self.assertEqual(d(MAX_TEXT).fields(), (False, 999999999, 999999999, 999999999, 999999999, 99, 0))
        self.assertEqual(d("0").fields(), (False, 0, 0, 0, 0, 0, 0))
        rng = random.Random(SEED)
        for _ in range(500):
            x = random_decimal(rng)
            self.assertEqual(ref.from_fields(*x.fields()), x)
        for bad in ((False, 10**9, 0, 0, 0, 0, 0), (False, 0, 0, 0, 0, 100, 0), (True, 0, 0, 0, 0, 0, 0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ref.from_fields(*bad)

    def test_structural_equality_is_rational_equality(self) -> None:
        self.assertTrue(ref.decimal_eq(d("1.50"), d("15e-1")))
        self.assertTrue(ref.decimal_eq(d("-0"), d("0e99")))
        self.assertFalse(ref.decimal_eq(d("1.5"), d("1.51")))
        self.assertFalse(ref.decimal_eq(d("1"), d("-1")))


class Grammar(unittest.TestCase):
    ACCEPTED = {
        "0": "0", "-0": "0", "0.0": "0", "-0.000e-99": "0", "0e999999999999999999999": "0",
        "1": "1", "-1": "-1", "1.50": "1.5", "1.000": "1", "100e-2": "1", "0.10e1": "1",
        "12e-3": "0.012", "1E5": "100000", "1e+5": "100000", "1e05": "100000", "1e-0005": "0.00001",
        "123.456": "123.456", "-0.5": "-0.5", "1e37": "1" + "0" * 37, MAX_TEXT: MAX_TEXT, MIN_TEXT: MIN_TEXT,
        TINY_TEXT: TINY_TEXT, "-" + TINY_TEXT: "-" + TINY_TEXT, "1e-38": TINY_TEXT, "10e-39": TINY_TEXT,
        "0." + "9" * 38: "0." + "9" * 38, "1." + "0" * 60: "1", "99999999999999999999999999999999999999.0": NINES,
        "1000000000000000000000000000000000000000e-2": "1" + "0" * 37,
    }
    MALFORMED = [
        "", "-", "+1", " 1", "1 ", "1\n", "01", "-01", "00", ".5", "5.", "1.e5", "1e", "1e+", "1e-", "--1",
        "1_000", "0x10", "Infinity", "-Infinity", "NaN", "inf", "1,5", "1.5.5", "1e5e5", "1e5.5", "e5", "1ee5",
        "١", "１", "1 ", "−1", "١٢", "1d", "１.５", "1.5f", "+0", "0.", "-.5", "1e+-5",
    ]
    OUTSIDE = ["1e38", "-1e38", "1e-39", "-1e-39", "1" + "0" * 38, "0." + "0" * 38 + "1",
               "1.234567890123456789012345678901234567891", "1e100000000000000000000", "1e-100000000000000000000",
               "12345678901234567890.1234567890123456789", "1.5e-38", "99999999999999999999999999999999999999.5"]

    def test_accepted_spellings_denote_their_exact_value(self) -> None:
        for text, canonical_text in self.ACCEPTED.items():
            with self.subTest(text=text):
                x = d(text)
                self.assertEqual(ref.decimal_to_string(x), canonical_text)
                self.assertEqual(ref.try_decimal(text), x)
                if "e" not in text.lower() or abs(int(text.lower().split("e")[1])) < 1000:
                    self.assertEqual(x.value, Fraction(pydecimal.Decimal(text)))

    def test_malformed_text_fails_domain_and_names_the_text(self) -> None:
        for text in self.MALFORMED:
            with self.subTest(text=text):
                error = failure(lambda: d(text))
                self.assertEqual((error.function, error.kind), ("decimal", "domain"))
                self.assertTrue(error.matches(f'decimal: domain: malformed number text "{text}"'))
                self.assertIsNone(ref.try_decimal(text))

    def test_out_of_range_tokens_fail_domain_and_name_the_token(self) -> None:
        for text in self.OUTSIDE:
            with self.subTest(text=text):
                error = failure(lambda: d(text))
                self.assertEqual(error.message, f"decimal: domain: {ref.truncated(text)} is outside the decimal range")
                self.assertIsNone(ref.try_decimal(text))

    def test_truncation_keeps_forty_characters(self) -> None:
        text = "x" * 41
        self.assertEqual(failure(lambda: d(text)).detail, f'malformed number text "{"x" * 40}..."')
        self.assertEqual(failure(lambda: d("x" * 40)).detail, f'malformed number text "{"x" * 40}"')
        long_token = "1" + "0" * 45
        self.assertEqual(failure(lambda: d(long_token)).detail, f"{long_token[:40]}... is outside the decimal range")

    def test_escaped_spellings_are_matched_by_shape(self) -> None:
        error = failure(lambda: d('1"'))
        self.assertIsNone(error.detail)
        self.assertTrue(error.matches('decimal: domain: malformed number text "1\\""'))
        self.assertTrue(error.matches('decimal: domain: malformed number text "1""'))
        self.assertFalse(error.matches('decimal: domain: 1" is outside the decimal range'))

    def test_the_bound_is_one_thousand_scalar_values(self) -> None:
        at_limit = "1." + "0" * 998
        self.assertEqual(len(at_limit), 1000)
        self.assertEqual(d(at_limit), d("1"))
        zero_at_limit = "0e" + "0" * 998
        self.assertEqual(d(zero_at_limit), ref.ZERO)
        small_at_limit = "1e-" + "0" * 996 + "5"
        self.assertEqual(len(small_at_limit), 1000)
        self.assertEqual(d(small_at_limit), d("0.00001"))
        over = "1." + "0" * 999
        error = failure(lambda: d(over))
        self.assertEqual(error.message, "decimal: domain: number text has 1001 characters, more than 1000")
        self.assertIsNone(ref.try_decimal(over))
        self.assertEqual(failure(lambda: d("0e" + "0" * 999)).detail, "number text has 1001 characters, more than 1000")

    def test_long_malformed_text_reads_as_malformed(self) -> None:
        # Literal reading: text that is not a token fails as "other text"; the
        # length message is for a well-formed token longer than the bound.
        text = "1." + "0" * 998 + "x"
        self.assertEqual(failure(lambda: d(text)).detail, f'malformed number text "{text[:40]}..."')

    def test_huge_exponents_are_decided_without_materialising_them(self) -> None:
        started = time.monotonic()
        self.assertIsNotNone(failure(lambda: d("1e" + "9" * 990)))
        self.assertIsNotNone(failure(lambda: d("1e-" + "9" * 990)))
        self.assertEqual(d("0.0e" + "9" * 990), ref.ZERO)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_text_round_trips(self) -> None:
        rng = random.Random(SEED + 1)
        for _ in range(2000):
            x = random_decimal(rng)
            text = ref.decimal_to_string(x)
            self.assertEqual(d(text), x)
            self.assertNotIn("e", text)
            self.assertFalse(text.startswith("+"))
            if "." in text:
                self.assertFalse(text.endswith("0"))
                self.assertEqual(len(text.split(".")[1]), x.scale)
            self.assertEqual(Fraction(pydecimal.Decimal(text)), x.value)


class Rendering(unittest.TestCase):
    def test_canonical_text(self) -> None:
        cases = {"0": "0", "-0.5": "-0.5", "0.05": "0.05", "1e3": "1000", "-123.4500": "-123.45", TINY_TEXT: TINY_TEXT}
        for text, expected in cases.items():
            self.assertEqual(ref.decimal_to_string(d(text)), expected)

    def test_fixed_text_pads_and_never_rounds(self) -> None:
        self.assertEqual(ref.decimal_to_fixed_string(d("1.5"), 3), "1.500")
        self.assertEqual(ref.decimal_to_fixed_string(d("-1.5"), 1), "-1.5")
        self.assertEqual(ref.decimal_to_fixed_string(d("7"), 0), "7")
        self.assertEqual(ref.decimal_to_fixed_string(d("0"), 2), "0.00")
        self.assertEqual(ref.decimal_to_fixed_string(d("-0.01"), 38), "-0.01" + "0" * 36)
        self.assertEqual(ref.decimal_to_fixed_string(d(MAX_TEXT), 38), MAX_TEXT + "." + "0" * 38)
        error = failure(lambda: ref.decimal_to_fixed_string(d("1.25"), 1))
        self.assertEqual(error.message, "decimal_to_fixed_string: domain: 1.25 has 2 fractional digits, more than 1")
        for n in (-1, 39, I64_MIN, I64_MAX):
            self.assertEqual(failure(lambda: ref.decimal_to_fixed_string(d("1"), n)).message,
                             f"decimal_to_fixed_string: domain: scale {n} is outside 0..38")

    def test_scale_is_the_canonical_scale(self) -> None:
        self.assertEqual(ref.decimal_scale(d("1.50")), 1)
        self.assertEqual(ref.decimal_scale(d("100")), 0)
        self.assertEqual(ref.decimal_scale(d(TINY_TEXT)), 38)


class Rounding(unittest.TestCase):
    def test_every_mode_matches_the_table_reading(self) -> None:
        rng = random.Random(SEED + 2)
        values = [Fraction(k, 8) for k in range(-40, 41)] + [Fraction(k, 3) for k in range(-10, 11)]
        values += [random_decimal(rng).value for _ in range(400)]
        for v in values:
            for n in (0, 1, 2, 5, 38):
                for mode in ROUNDINGS:
                    expected = table_round(v, n, mode)
                    if expected is None:
                        error = failure(lambda: ref.round_to_quantum(v, n, mode, "f"))
                        self.assertEqual((error.function, error.kind), ("f", "domain"))
                    else:
                        self.assertEqual(ref.round_to_quantum(v, n, mode, "f"), expected, (v, n, mode))

    def test_ties_in_every_mode_and_sign(self) -> None:
        expected = {
            "RoundTowardNegative": ("2", "-3", "0", "-1", "2.4", "-2.5"),
            "RoundTowardPositive": ("3", "-2", "1", "-0", "2.5", "-2.4"),
            "RoundTowardZero": ("2", "-2", "0", "-0", "2.4", "-2.4"),
            "RoundAwayFromZero": ("3", "-3", "1", "-1", "2.5", "-2.5"),
            "RoundTiesToEven": ("2", "-2", "0", "-0", "2.4", "-2.4"),
            "RoundTiesToAway": ("3", "-3", "1", "-1", "2.5", "-2.5"),
        }
        inputs = (("2.5", 0), ("-2.5", 0), ("0.5", 0), ("-0.5", 0), ("2.45", 1), ("-2.45", 1))
        for mode, results in expected.items():
            for (text, n), want in zip(inputs, results):
                with self.subTest(mode=mode, text=text):
                    self.assertEqual(ref.decimal_round(d(text), n, mode), d(want))
        self.assertEqual(ref.decimal_round(d("3.5"), 0, "RoundTiesToEven"), d("4"))
        self.assertEqual(ref.decimal_round(d("-3.5"), 0, "RoundTiesToEven"), d("-4"))
        self.assertEqual(ref.decimal_round(d("2.5000000000000000000000000000000000001"), 0, "RoundTiesToEven"), d("3"))
        self.assertEqual(ref.decimal_round(d("-2.4999999999999999999999999999999999999"), 0, "RoundTiesToAway"), d("-2"))

    def test_reject_inexact_names_the_value_and_quantum(self) -> None:
        self.assertEqual(ref.decimal_round(d("1.25"), 2, "RejectInexact"), d("1.25"))
        self.assertEqual(ref.decimal_round(d("1.25"), 5, "RejectInexact"), d("1.25"))
        error = failure(lambda: ref.decimal_round(d("1.25"), 1, "RejectInexact"))
        self.assertEqual(error.message, "decimal_round: domain: 1.25 is not a multiple of 10^-1")

    def test_round_agrees_with_python_quantize(self) -> None:
        rng = random.Random(SEED + 3)
        for _ in range(3000):
            x = random_decimal(rng)
            n = rng.randint(0, 38)
            mode = rng.choice(tuple(PYTHON_MODES))
            want = python_round(pydecimal.Decimal(ref.decimal_to_string(x)), n, mode)
            self.assertEqual(ref.decimal_round(x, n, mode).value, from_python(want), (x, n, mode))

    def test_round_result_is_always_in_the_value_set(self) -> None:
        for text in (MAX_TEXT, MIN_TEXT, "9" * 37 + ".9", "0." + "9" * 38, "-0." + "9" * 38, "9.5"):
            for n in range(0, 39):
                for mode in PYTHON_MODES:
                    ref.decimal_round(d(text), n, mode)
        self.assertEqual(ref.decimal_round(d("0." + "9" * 38), 0, "RoundAwayFromZero"), d("1"))
        self.assertEqual(ref.decimal_round(d("9" * 37 + ".9"), 0, "RoundTowardPositive"), d("1e37"))

    def test_scale_outside_range_fails_first(self) -> None:
        for n in (-1, 39, I64_MIN, I64_MAX):
            self.assertEqual(failure(lambda: ref.decimal_round(d("1.25"), n, "RejectInexact")).message,
                             f"decimal_round: domain: scale {n} is outside 0..38")


class Arithmetic(unittest.TestCase):
    def test_exact_sum_difference_product_match_python(self) -> None:
        rng = random.Random(SEED + 4)
        ops = {"decimal_add": (ref.decimal_add, WIDE.add), "decimal_sub": (ref.decimal_sub, WIDE.subtract),
               "decimal_mul": (ref.decimal_mul, WIDE.multiply)}
        counts = {"ok": 0, "overflow": 0}
        for _ in range(3000):
            a, b = random_decimal(rng), random_decimal(rng)
            for name, (mine, python) in ops.items():
                exact = from_python(python(pydecimal.Decimal(ref.decimal_to_string(a)), pydecimal.Decimal(ref.decimal_to_string(b))))
                if ref.in_value_set(exact):
                    self.assertEqual(mine(a, b).value, exact)
                    counts["ok"] += 1
                else:
                    error = failure(lambda: mine(a, b))
                    self.assertEqual((error.function, error.kind), (name, "overflow"))
                    counts["overflow"] += 1
        self.assertGreater(counts["ok"], 1000)
        self.assertGreater(counts["overflow"], 100)

    def test_removable_zeros_and_carries(self) -> None:
        self.assertEqual(ref.decimal_mul(d("2.5"), d("4")), d("10"))
        self.assertEqual(ref.decimal_scale(ref.decimal_mul(d("2.5"), d("4"))), 0)
        self.assertEqual(ref.decimal_add(d("0.25"), d("0.75")), d("1"))
        self.assertEqual(ref.decimal_add(d("999999999"), d("1")).fields(), (False, 0, 1, 0, 0, 0, 0))
        self.assertEqual(ref.decimal_add(d("9" * 36), d("1")).fields(), (False, 0, 0, 0, 0, 1, 0))
        self.assertEqual(ref.decimal_sub(d("1e36"), d("1")).fields(), (False, 999999999, 999999999, 999999999, 999999999, 0, 0))

    def test_overflow_at_the_envelope(self) -> None:
        cases = [
            (ref.decimal_add, MAX_TEXT, "1"), (ref.decimal_sub, MIN_TEXT, "1"), (ref.decimal_mul, "1e37", "10"),
            (ref.decimal_mul, "1e-20", "1e-19"), (ref.decimal_add, "1e37", "0.1"), (ref.decimal_mul, "1.5", TINY_TEXT),
        ]
        for op, a, b in cases:
            with self.subTest(op=op.__name__, a=a, b=b):
                error = failure(lambda: op(d(a), d(b)))
                self.assertEqual((error.function, error.kind, error.detail), (op.__name__, "overflow", None))
                self.assertTrue(error.matches(f"{op.__name__}: overflow: {a} plus {b} is outside the decimal range"))
        self.assertEqual(ref.decimal_add(d(MAX_TEXT), d("-1")), d("9" * 37 + "8"))
        self.assertEqual(ref.decimal_mul(d("1e-19"), d("1e-19")), d(TINY_TEXT))

    def test_order_is_the_order_of_the_rationals(self) -> None:
        rng = random.Random(SEED + 5)
        pairs = [(d("1.5"), d("1.50")), (d("-0"), d("0")), (d("-1"), d("1")), (d(TINY_TEXT), d("0"))]
        pairs += [(random_decimal(rng), random_decimal(rng)) for _ in range(500)]
        for a, b in pairs:
            self.assertEqual(ref.decimal_lt(a, b), a.value < b.value)
            self.assertEqual(ref.decimal_lte(a, b), a.value <= b.value)
            self.assertEqual(ref.decimal_gt(a, b), a.value > b.value)
            self.assertEqual(ref.decimal_gte(a, b), a.value >= b.value)
        self.assertFalse(ref.decimal_lt(d("1.5"), d("1.50")))
        self.assertTrue(ref.decimal_lte(d("1.5"), d("1.50")))


class Division(unittest.TestCase):
    def test_quotient_matches_a_sticky_python_quotient(self) -> None:
        rng = random.Random(SEED + 6)
        outcomes = {"ok": 0, "overflow": 0}
        for _ in range(3000):
            a, b = random_decimal(rng), random_decimal(rng)
            n = rng.randint(0, 38)
            mode = rng.choice(tuple(PYTHON_MODES))
            want = from_python(python_quotient(a, b, n, mode))
            if ref.in_value_set(want):
                self.assertEqual(ref.decimal_div(a, b, n, mode).value, want, (a, b, n, mode))
                outcomes["ok"] += 1
            else:
                error = failure(lambda: ref.decimal_div(a, b, n, mode))
                self.assertEqual((error.function, error.kind), ("decimal_div", "overflow"))
                outcomes["overflow"] += 1
        self.assertGreater(outcomes["ok"], 500)
        self.assertGreater(outcomes["overflow"], 100)

    def test_the_sticky_intermediate_catches_what_double_rounding_misses(self) -> None:
        # 0.2499...9: rounding to two digits first gives the false tie 0.25.
        a, b = d("0.24999999999999999999"), d("1")
        self.assertEqual(ref.decimal_div(a, b, 1, "RoundTiesToAway"), d("0.2"))
        self.assertEqual(from_python(python_quotient(a, b, 1, "RoundTiesToAway")), Fraction(2, 10))
        double = pydecimal.Decimal("0.24999999999999999999").quantize(pydecimal.Decimal("0.01"), pydecimal.ROUND_HALF_UP)
        self.assertEqual(double.quantize(pydecimal.Decimal("0.1"), pydecimal.ROUND_HALF_UP), pydecimal.Decimal("0.3"))

    def test_ties_and_exactness(self) -> None:
        self.assertEqual(ref.decimal_div(d("1"), d("8"), 2, "RoundTiesToEven"), d("0.12"))
        self.assertEqual(ref.decimal_div(d("-1"), d("8"), 2, "RoundTiesToEven"), d("-0.12"))
        self.assertEqual(ref.decimal_div(d("1"), d("8"), 2, "RoundTiesToAway"), d("0.13"))
        self.assertEqual(ref.decimal_div(d("-1"), d("8"), 2, "RoundTiesToAway"), d("-0.13"))
        self.assertEqual(ref.decimal_div(d("1"), d("8"), 3, "RejectInexact"), d("0.125"))
        error = failure(lambda: ref.decimal_div(d("1"), d("3"), 5, "RejectInexact"))
        self.assertEqual((error.function, error.kind, error.detail), ("decimal_div", "domain", None))
        self.assertTrue(error.matches("decimal_div: domain: 1 / 3 is not a multiple of 10^-5"))
        self.assertFalse(error.matches("decimal_div: domain: 1 / 3 is not a multiple of 10^-6"))

    def test_failures_and_their_order(self) -> None:
        self.assertEqual(failure(lambda: ref.decimal_div(d("1"), d("0"), 2, "RoundTiesToEven")).message,
                         "decimal_div: domain: division by zero")
        self.assertEqual(failure(lambda: ref.decimal_div(d("1"), d("0"), 39, "RejectInexact")).message,
                         "decimal_div: domain: division by zero")
        self.assertEqual(failure(lambda: ref.decimal_div(d("1"), d("3"), -1, "RejectInexact")).message,
                         "decimal_div: domain: scale -1 is outside 0..38")
        big = failure(lambda: ref.decimal_div(d("1e37"), d(TINY_TEXT), 0, "RoundTiesToEven"))
        self.assertEqual((big.function, big.kind), ("decimal_div", "overflow"))
        # Rounding (and RejectInexact) comes before the range check of the rounded quotient.
        inexact = failure(lambda: ref.decimal_div(d("1e37"), d("3e-38"), 0, "RejectInexact"))
        self.assertEqual((inexact.function, inexact.kind), ("decimal_div", "domain"))
        digits = failure(lambda: ref.decimal_div(d("10"), d("3"), 38, "RoundTiesToEven"))
        self.assertEqual((digits.function, digits.kind), ("decimal_div", "overflow"))
        self.assertEqual(ref.decimal_div(d("1"), d("3"), 38, "RoundTiesToEven"), d("0." + "3" * 38))

    def test_try_div_is_none_exactly_on_domain(self) -> None:
        self.assertIsNone(ref.try_decimal_div(d("1"), d("0"), 2, "RoundTiesToEven"))
        self.assertIsNone(ref.try_decimal_div(d("1"), d("3"), 2, "RejectInexact"))
        self.assertIsNone(ref.try_decimal_div(d("1"), d("3"), 39, "RoundTiesToEven"))
        self.assertEqual(ref.try_decimal_div(d("1"), d("4"), 2, "RejectInexact"), d("0.25"))
        error = failure(lambda: ref.try_decimal_div(d("1e37"), d(TINY_TEXT), 0, "RoundTiesToEven"))
        self.assertEqual((error.function, error.kind), ("try_decimal_div", "overflow"))


class Integers(unittest.TestCase):
    def test_from_i64_is_exact(self) -> None:
        for v in (0, 1, -1, I64_MAX, I64_MIN, 10**18, -(10**18)):
            self.assertEqual(ref.decimal_from_i64(v).value, v)
        self.assertEqual(ref.decimal_to_string(ref.decimal_from_i64(I64_MIN)), "-9223372036854775808")
        self.assertEqual(ref.decimal_from_i64(1000).scale, 0)
        with self.assertRaises(ValueError):
            ref.decimal_from_i64(I64_MAX + 1)

    def test_to_i64_rounds_then_checks_the_range(self) -> None:
        self.assertEqual(ref.decimal_to_i64(d("9223372036854775807.4"), "RoundTiesToEven"), I64_MAX)
        self.assertEqual(ref.decimal_to_i64(d("-9223372036854775808.5"), "RoundTowardZero"), I64_MIN)
        self.assertEqual(ref.decimal_to_i64(d("-2.5"), "RoundTiesToEven"), -2)
        self.assertEqual(ref.decimal_to_i64(d("-2.5"), "RoundTiesToAway"), -3)
        self.assertEqual(ref.decimal_to_i64(d("-2.5"), "RoundTowardPositive"), -2)
        self.assertEqual(ref.decimal_to_i64(d("7"), "RejectInexact"), 7)
        up = failure(lambda: ref.decimal_to_i64(d("9223372036854775807.5"), "RoundTiesToEven"))
        self.assertEqual(up.message, "decimal_to_i64: overflow: 9223372036854775807.5 rounds to an integer outside i64")
        down = failure(lambda: ref.decimal_to_i64(d("-9223372036854775808.5"), "RoundTiesToAway"))
        self.assertEqual((down.function, down.kind), ("decimal_to_i64", "overflow"))
        inexact = failure(lambda: ref.decimal_to_i64(d("1.5"), "RejectInexact"))
        self.assertEqual(inexact.message, "decimal_to_i64: domain: 1.5 is not a multiple of 10^-0")
        both = failure(lambda: ref.decimal_to_i64(d("1000000000000000000000.5"), "RejectInexact"))
        self.assertEqual(both.kind, "domain")
        self.assertEqual(failure(lambda: ref.decimal_to_i64(d("1e30"), "RejectInexact")).kind, "overflow")

    def test_try_to_i64_is_none_on_domain_and_overflow(self) -> None:
        self.assertIsNone(ref.try_decimal_to_i64(d("1.5"), "RejectInexact"))
        self.assertIsNone(ref.try_decimal_to_i64(d("1e30"), "RoundTowardZero"))
        self.assertEqual(ref.try_decimal_to_i64(d("1.5"), "RoundTowardZero"), 1)


class BinaryRounding(unittest.TestCase):
    def test_f64_matches_python_correct_rounding(self) -> None:
        rng = random.Random(SEED + 7)
        values = [random_decimal(rng) for _ in range(5000)]
        values += [d(t) for t in (MAX_TEXT, MIN_TEXT, TINY_TEXT, "0", "9007199254740993", "9007199254740995", "0.1")]
        for x in values:
            mine = ref.decimal_to_f64(x)
            self.assertEqual(struct.pack("<d", mine), struct.pack("<d", float(x.value)), ref.decimal_to_string(x))
            self.assertEqual(ref.decimal_to_f64_bits(x), struct.unpack("<Q", struct.pack("<d", mine))[0])

    def test_f32_matches_a_nearest_neighbour_search(self) -> None:
        rng = random.Random(SEED + 8)
        values = [random_decimal(rng) for _ in range(5000)]
        values += [d(t) for t in (MAX_TEXT, MIN_TEXT, TINY_TEXT, "-" + TINY_TEXT, "2e-38", "16777217", "16777219",
                                  "1.000000059604644775390625", "0.0691026858985424")]
        for x in values:
            q = x.value
            want = f32_by_search(abs(q)) | ((1 << 31) if q < 0 else 0)
            self.assertEqual(ref.decimal_to_f32_bits(x), want, ref.decimal_to_string(x))

    def test_f32_rounds_directly_not_through_f64(self) -> None:
        x = d("0.0691026858985424")
        direct = ref.decimal_to_f32_bits(x)
        through_f64 = struct.unpack("<I", struct.pack("<f", float(x.value)))[0]
        self.assertEqual(direct, 0x3D8D85B5)
        self.assertEqual(through_f64, 0x3D8D85B6)
        # Just below an f32 midpoint by far less than half an f64 unit: f64 lands on
        # the midpoint, whose tie then goes to even, while the exact value rounds down.
        witnesses = 0
        for k in range(200):
            midpoint = Fraction(2**24 + 2 * k + 1)
            y = ref.canonical(midpoint - Fraction(1, 10**20))
            direct = ref.decimal_to_f32_bits(y)
            self.assertEqual(ref.bits_value(direct, ref.F32), midpoint - 1)
            if direct != struct.unpack("<I", struct.pack("<f", float(y.value)))[0]:
                witnesses += 1
        self.assertEqual(witnesses, 100)

    def test_binary_ties_go_to_even(self) -> None:
        self.assertEqual(ref.decimal_to_f64(d("9007199254740993")), 9007199254740992.0)
        self.assertEqual(ref.decimal_to_f64(d("9007199254740995")), 9007199254740996.0)
        self.assertEqual(ref.decimal_to_f64(d("-9007199254740993")), -9007199254740992.0)
        self.assertEqual(ref.decimal_to_f32(d("16777217")), 16777216.0)
        self.assertEqual(ref.decimal_to_f32(d("16777219")), 16777220.0)
        self.assertEqual(ref.decimal_to_f32(d("1.000000059604644775390625")), 1.0)
        self.assertEqual(ref.decimal_to_f32(d("1.000000178813934326171875")), 1.0 + 2.0**-22)
        self.assertEqual(ref.decimal_to_f32(d("16777217.000000000000001")), 16777218.0)

    def test_f32_subnormal_results(self) -> None:
        tiny = d(TINY_TEXT)
        bits = ref.decimal_to_f32_bits(tiny)
        self.assertEqual(bits >> 23, 0)
        self.assertEqual(bits, round(Fraction(1, 10**38) / Fraction(2) ** -149))
        self.assertEqual(ref.decimal_to_f32_bits(d("-" + TINY_TEXT)), bits | (1 << 31))
        self.assertEqual(ref.decimal_to_f32_bits(d("2e-38")) >> 23, 1)
        # Only ±10^-38 of the value set lies below f32's smallest normal 2^-126.
        self.assertLess(Fraction(1, 10**38), Fraction(2) ** -126)
        self.assertGreater(Fraction(2, 10**38), Fraction(2) ** -126)

    def test_zero_and_extremes(self) -> None:
        self.assertEqual(ref.decimal_to_f64_bits(ref.ZERO), 0)
        self.assertEqual(ref.decimal_to_f32_bits(ref.ZERO), 0)
        self.assertEqual(ref.decimal_to_f32(d(MAX_TEXT)), struct.unpack("<f", struct.pack("<f", 1e38))[0])
        self.assertLess(Fraction(10**38), (2 - Fraction(2) ** -23) * Fraction(2) ** 127)

    def test_round_binary_rejects_values_past_the_finite_range(self) -> None:
        with self.assertRaises(OverflowError):
            ref.round_binary(Fraction(2) ** 128, ref.F32)
        self.assertEqual(ref.round_binary(Fraction(2) ** -150, ref.F32), 0)
        self.assertEqual(ref.round_binary(Fraction(3) * Fraction(2) ** -151, ref.F32), Fraction(2) ** -149)

    def test_bits_encoding_round_trips(self) -> None:
        for fmt, pack, unpack in ((ref.F64, "<d", "<Q"), (ref.F32, "<f", "<I")):
            for value in (1.0, -2.5, 1e-38, 5e-324 if fmt == ref.F64 else 1e-45, 3.0e38):
                bits = struct.unpack(unpack, struct.pack(pack, value))[0]
                exact = ref.bits_value(bits, fmt)
                self.assertEqual(ref.binary_bits(exact, fmt), bits)

    def test_printed_text_recovers_the_bits(self) -> None:
        self.assertEqual(ref.text_bits("0.1", ref.F32), struct.unpack("<I", struct.pack("<f", 0.1))[0])
        self.assertEqual(ref.text_bits("-0.0", ref.F64), 1 << 63)
        self.assertEqual(ref.text_bits("0.0", ref.F64), 0)
        self.assertEqual(ref.text_bits("1e-38", ref.F32), ref.decimal_to_f32_bits(d(TINY_TEXT)))
        self.assertEqual(ref.text_bits("5e-324", ref.F64), 1)
        self.assertIsNone(ref.text_bits("NaN", ref.F64))
        self.assertIsNone(ref.text_bits("1.2.3", ref.F64))
        self.assertIsNone(ref.text_bits("4e38", ref.F32))
        self.assertIsNotNone(ref.text_bits("4e38", ref.F64))


class FromF64(unittest.TestCase):
    def test_exact_binary_value_rounded_once(self) -> None:
        x = ref.decimal_from_f64(0.1, 38, "RoundTiesToEven")
        self.assertEqual(ref.decimal_to_string(x), "0.10000000000000000555111512312578270212")
        self.assertEqual(ref.decimal_from_f64(0.1, 1, "RoundTowardNegative"), d("0.1"))
        self.assertEqual(ref.decimal_from_f64(0.1, 17, "RoundTowardNegative"), d("0.1"))
        self.assertEqual(ref.decimal_from_f64(0.1, 18, "RoundTowardPositive"), d("0.100000000000000006"))
        self.assertEqual(ref.decimal_from_f64(-0.0, 2, "RejectInexact"), ref.ZERO)
        self.assertEqual(ref.decimal_from_f64(1e38, 0, "RejectInexact"), d("99999999999999997748809823456034029568"))
        self.assertEqual(ref.decimal_from_f64(5e-324, 38, "RoundAwayFromZero"), d(TINY_TEXT))
        self.assertEqual(ref.decimal_from_f64(-5e-324, 38, "RoundTowardZero"), ref.ZERO)

    def test_ties_on_exact_binary_halves(self) -> None:
        for mode, want in (("RoundTiesToEven", ("2", "-2", "0.12")), ("RoundTiesToAway", ("3", "-3", "0.13")),
                           ("RoundTowardZero", ("2", "-2", "0.12")), ("RoundAwayFromZero", ("3", "-3", "0.13"))):
            self.assertEqual(ref.decimal_from_f64(2.5, 0, mode), d(want[0]))
            self.assertEqual(ref.decimal_from_f64(-2.5, 0, mode), d(want[1]))
            self.assertEqual(ref.decimal_from_f64(0.125, 2, mode), d(want[2]))

    def test_python_decimal_from_float_agrees(self) -> None:
        rng = random.Random(SEED + 10)
        for _ in range(3000):
            x = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
            if not math.isfinite(x) or abs(x) > 1e40:
                continue
            n = rng.randint(0, 38)
            mode = rng.choice(tuple(PYTHON_MODES))
            want = from_python(python_round(pydecimal.Decimal(x), n, mode))
            if ref.in_value_set(want):
                self.assertEqual(ref.decimal_from_f64(x, n, mode).value, want)
            else:
                self.assertEqual(failure(lambda: ref.decimal_from_f64(x, n, mode)).kind, "domain")

    def test_failures_and_their_order(self) -> None:
        for x, shown in ((math.nan, "NaN"), (math.inf, "inf"), (-math.inf, "-inf")):
            self.assertEqual(failure(lambda: ref.decimal_from_f64(x, 39, "RejectInexact")).message,
                             f"decimal_from_f64: domain: {shown} is not finite")
            self.assertIsNone(ref.try_decimal_from_f64(x, 2, "RoundTiesToEven"))
        self.assertEqual(failure(lambda: ref.decimal_from_f64(0.5, -1, "RejectInexact")).message,
                         "decimal_from_f64: domain: scale -1 is outside 0..38")
        inexact = failure(lambda: ref.decimal_from_f64(0.1, 2, "RejectInexact"))
        self.assertTrue(inexact.matches("decimal_from_f64: domain: 0.1 is not a multiple of 10^-2"))
        self.assertEqual(ref.decimal_from_f64(0.5, 1, "RejectInexact"), d("0.5"))
        for x, n in ((1e39, 0), (1.7976931348623157e308, 0), (-1e39, 5), (1.1, 38), (2.0**127, 0)):
            error = failure(lambda: ref.decimal_from_f64(x, n, "RoundTiesToEven"))
            self.assertEqual((error.function, error.kind), ("decimal_from_f64", "domain"))
            self.assertTrue(error.matches(f"decimal_from_f64: domain: {x} is outside the decimal range"))
            self.assertIsNone(ref.try_decimal_from_f64(x, n, "RoundTiesToEven"))


class Messages(unittest.TestCase):
    def test_every_failure_has_the_three_part_shape(self) -> None:
        errors = [
            failure(lambda: d("x")), failure(lambda: d("1e99")), failure(lambda: d("1" * 1001)),
            failure(lambda: ref.decimal_round(d("1"), 40, "RoundTiesToEven")),
            failure(lambda: ref.decimal_div(d("1"), d("0"), 1, "RoundTiesToEven")),
            failure(lambda: ref.decimal_add(d(MAX_TEXT), d(MAX_TEXT))),
            failure(lambda: ref.decimal_to_i64(d("1e20"), "RoundTiesToEven")),
            failure(lambda: ref.decimal_to_fixed_string(d("0.5"), 0)),
            failure(lambda: ref.decimal_from_f64(math.nan, 1, "RoundTiesToEven")),
        ]
        for error in errors:
            self.assertRegex(error.message, r"^[a-z0-9_]+: (domain|overflow): \S")

    def test_a_message_with_another_function_or_kind_does_not_match(self) -> None:
        error = failure(lambda: ref.decimal_round(d("1"), 40, "RoundTiesToEven"))
        self.assertTrue(error.matches("decimal_round: domain: scale 40 is outside 0..38"))
        self.assertFalse(error.matches("decimal_div: domain: scale 40 is outside 0..38"))
        self.assertFalse(error.matches("decimal_round: overflow: scale 40 is outside 0..38"))
        self.assertFalse(error.matches("decimal_round: domain: scale 41 is outside 0..38"))
        with self.assertRaises(ValueError):
            DecimalError("f", "trap", "scale", "x")
        with self.assertRaises(ValueError):
            DecimalError("f", "domain", "unnamed", "x")


if __name__ == "__main__":
    unittest.main()
