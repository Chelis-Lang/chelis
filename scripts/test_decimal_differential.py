#!/usr/bin/env python3
"""Unit tests for the `Std.Decimal` differential harness ([05-OP-76], chelis#2778).

The harness's own logic is tested without a build: a fake runner prints what a
conforming module would print, computed from the reference, and each test then
breaks one observation the way a defective lane could and checks that the
comparator reports it under the right class. The generator tests check that
the corpus is deterministic, covers each category the harness promises, and
renders literals that denote exactly the intended inputs. The canary's defect
models perturb the reference the way a defective module would compute, and
each must change an expected output of the canary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
from fractions import Fraction
import functools
import inspect
import io
import math
import operator
import os
import random
import re
import struct
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decimal_differential as harness  # noqa: E402
import decimal_reference as ref  # noqa: E402
from decimal_differential import FloatBits, LaneResult, Program  # noqa: E402

CORPUS = harness.build_corpus()
CANARY = harness.build_canary_corpus()


def shortest(element: FloatBits) -> str:
    """The shortest text that reads back to the element's bits at its width."""
    special = harness.special_text(element)
    if special is not None:
        return special
    precision, exponent_bits = element.fmt
    negative = element.bits >> (precision + exponent_bits - 1)
    value = ref.bits_value(element.bits, element.fmt)
    if value == 0:
        return "-0.0" if negative else "0.0"
    for digits in range(1, 18):
        text = f"{float(value):.{digits}g}"
        if ref.text_bits(text, element.fmt) == element.bits:
            return text
    raise AssertionError(element)


def sample_detail(error: ref.DecimalError) -> str:
    """A detail a conforming module could print for this failure."""
    if error.detail is not None:
        return error.detail
    samples = {
        "overflow": "the result is outside the decimal range",
        "outside": "1e39 is outside the decimal range",
        "malformed": 'malformed number text "1\\""',
    }
    if error.reason == "inexact":
        n = re.search(r"10\\\^-(\d+)", error.pattern).group(1)
        return f"1 / 3 is not a multiple of 10^-{n}"
    return samples[error.reason]


def conforming_output(program: Program) -> tuple[int, str, str]:
    if program.failure is not None:
        error = program.failure.error
        return 1, "", f"error: {error.function}: {error.kind}: {sample_detail(error)}\n"
    lines = []
    for binding in program.bindings:
        elements = [shortest(e) if isinstance(e, FloatBits) else e for e in binding.expected]
        lines.append(f"{binding.name} = [{', '.join(elements)}]")
    return 0, "\n".join(lines) + "\n", ""


class FakeRunner:
    """Prints what a conforming module prints, with optional per-lane edits."""

    def __init__(self, edit=None) -> None:
        self.edit = edit
        self.calls: list[tuple[str, str]] = []

    def lane(self, lane: str, program: Program) -> LaneResult:
        self.calls.append((lane, program.name))
        status, stdout, stderr = conforming_output(program)
        if self.edit is not None:
            status, stdout, stderr = self.edit(lane, program, status, stdout, stderr)
        return LaneResult(lane, status, stdout, stderr, "run")


@functools.cache
def default_programs() -> tuple[Program, ...]:
    return tuple(harness.make_programs(CORPUS, 150, 2))


def programs() -> list[Program]:
    return list(default_programs())


def run(edit=None, selected: list[Program] | None = None) -> harness.Report:
    return harness.run_all(FakeRunner(edit), selected or programs(), ("eval", "c"), 2, lambda _: None)


def first_program(row: str) -> Program:
    return next(p for p in programs() if p.bindings and p.bindings[0].row == row)


def failure_program(function: str, reason: str) -> Program:
    return next(p for p in programs() if p.failure and p.failure.function == function and p.failure.error.reason == reason)


class Comparator(unittest.TestCase):
    def test_a_conforming_module_passes_on_both_lanes(self) -> None:
        report = run()
        self.assertEqual(report.problems, [])
        self.assertEqual(report.cases, len(CORPUS.cases))
        self.assertEqual(report.failures_checked, sum(1 for p in programs() if p.failure))
        line = harness.summary(report, ("eval", "c"))
        self.assertTrue(line.startswith(harness.PASS_MARKER))
        self.assertTrue(line.endswith("on lanes eval+c; default corpus)"))
        self.assertTrue(harness.summary(report, ("c",), "large").endswith("on lanes c; large corpus)"))

    def test_a_wrong_text_element_names_its_case(self) -> None:
        program = first_program("row_add")

        def edit(lane, prog, status, stdout, stderr):
            if lane == "c":
                stdout = stdout.replace(" = [", " = [WRONG", 1)
            return status, stdout, stderr

        report = run(edit, [program])
        self.assertEqual(report.classes, {"value": 1})
        case = program.bindings[0].cases[0]
        self.assertIn(f"[c] {program.bindings[0].name}[0] {case.category} row_add{case.args!r}", report.problems[0])
        self.assertTrue(harness.summary(report, ("eval", "c")).startswith(harness.FAIL_MARKER))

    def test_a_float_one_unit_off_is_a_disagreement(self) -> None:
        program = first_program("row_floats")
        binding = program.bindings[0]
        target = next(i for i, e in enumerate(binding.expected) if isinstance(e, FloatBits) and e.fmt == ref.F32 and e.bits)
        element = binding.expected[target]
        neighbour = FloatBits(element.fmt, element.bits + 1)

        def edit(lane, prog, status, stdout, stderr):
            line = next(l for l in stdout.splitlines() if l.startswith(binding.name + " = "))
            items = line.split(" = ", 1)[1][1:-1].split(", ")
            items[target] = shortest(neighbour)
            return status, stdout.replace(line, f"{binding.name} = [{', '.join(items)}]"), stderr

        report = run(edit, [program])
        self.assertEqual(report.classes, {"value": 2})
        self.assertIn(f"element {target % 2} expected {element}", report.problems[0])

    def test_negative_zero_is_not_zero(self) -> None:
        zero = FloatBits(ref.F64, 0)
        self.assertTrue(harness.element_matches(zero, "0.0"))
        self.assertFalse(harness.element_matches(zero, "-0.0"))
        self.assertFalse(harness.element_matches(FloatBits(ref.F64, 1 << 63), "0.0"))
        self.assertFalse(harness.element_matches(zero, "NaN"))

    def test_non_finite_echoes_compare_by_class(self) -> None:
        nan = harness.float_echo(math.nan)
        self.assertTrue(harness.element_matches(nan, "NaN"))
        self.assertTrue(harness.element_matches(FloatBits(ref.F64, 0x7FF0000000000001), "NaN"))
        self.assertFalse(harness.element_matches(nan, "inf"))
        self.assertTrue(harness.element_matches(harness.float_echo(-math.inf), "-inf"))
        self.assertFalse(harness.element_matches(harness.float_echo(math.inf), "-inf"))
        self.assertFalse(harness.element_matches(harness.float_echo(math.inf), "1.7976931348623157e308"))

    def test_float_text_is_read_at_its_own_width(self) -> None:
        f32_tenth = FloatBits(ref.F32, struct.unpack("<I", struct.pack("<f", 0.1))[0])
        self.assertTrue(harness.element_matches(f32_tenth, "0.1"))
        self.assertFalse(harness.element_matches(FloatBits(ref.F64, struct.unpack("<Q", struct.pack("<d", 0.1))[0]),
                                                 "0.10000000149011612"))

    def test_missing_extra_and_stray_output(self) -> None:
        program = first_program("row_text")
        name = program.bindings[0].name

        def drop(lane, prog, status, stdout, stderr):
            return status, "\n".join(l for l in stdout.splitlines() if not l.startswith(name + " = ")), stderr

        self.assertIn(f"{name}: no output line", run(drop, [program]).problems[0])

        def extra(lane, prog, status, stdout, stderr):
            return status, stdout.replace(f"{name} = [", f"{name} = [0, ", 1), stderr

        self.assertTrue(any("elements expected" in p for p in run(extra, [program]).problems))

        def stray(lane, prog, status, stdout, stderr):
            return status, stdout + "constant = 1\nnoise\n", stderr

        report = run(stray, [program])
        self.assertEqual(report.classes, {"stray-output": 2})

    def test_a_grid_program_that_fails_is_reported(self) -> None:
        def fail(lane, prog, status, stdout, stderr):
            return (2, "", "error: Type errors") if lane == "eval" else (status, stdout, stderr)

        report = run(fail, [first_program("row_eq")])
        self.assertEqual(report.classes, {"program": 1})
        self.assertIn("[eval/run]: status 2", report.problems[0])

    def test_a_failure_that_succeeds_or_traps_is_missing(self) -> None:
        program = failure_program("decimal_add", "overflow")

        def succeed(lane, prog, status, stdout, stderr):
            return 0, f"{prog.failure.name} = [1]\n", ""

        self.assertEqual(run(succeed, [program]).classes, {"failure-missing": 2})

        def trap(lane, prog, status, stdout, stderr):
            return 1, "", "error: Overflow: integer add overflowed\n"

        self.assertEqual(run(trap, [program]).classes, {"failure-missing": 2})

    def test_wrong_function_or_kind(self) -> None:
        program = failure_program("decimal_div", "division_by_zero")

        def kind(lane, prog, status, stdout, stderr):
            return status, stdout, stderr.replace(": domain: ", ": overflow: ")

        self.assertEqual(run(kind, [program]).classes, {"failure-kind": 2})

        def function(lane, prog, status, stdout, stderr):
            return status, stdout, stderr.replace("decimal_div:", "try_decimal_div:") if lane == "c" else stderr

        report = run(function, [program])
        self.assertEqual(report.classes, {"failure-kind": 1, "lane-message": 1})

    def test_a_detail_off_the_pinned_shape(self) -> None:
        program = failure_program("decimal_round", "scale")

        def detail(lane, prog, status, stdout, stderr):
            return status, stdout, re.sub(r"scale (-?\d+) is outside", r"scale \1 lies outside", stderr)

        self.assertEqual(run(detail, [program]).classes, {"failure-detail": 2})
        overflow = failure_program("decimal_mul", "overflow")

        def no_shape(lane, prog, status, stdout, stderr):
            return status, stdout, stderr.replace("is outside the decimal range", "overflowed")

        self.assertEqual(run(no_shape, [overflow]).classes, {"failure-detail": 2})

    def test_failure_message_parsing(self) -> None:
        result = LaneResult("c", 1, "", "warning: x\ndecimal: domain: division by zero\n", "run")
        self.assertEqual(harness.failure_message(result), "decimal: domain: division by zero")
        result = LaneResult("eval", 1, "", "error: decimal_add: overflow: a is outside the decimal range\n", "eval")
        self.assertEqual(harness.failure_message(result), "decimal_add: overflow: a is outside the decimal range")
        self.assertIsNone(harness.failure_message(LaneResult("c", 1, "", "error: Domain: trap\n", "run")))

    def test_bindings_and_elements_parse(self) -> None:
        found, stray = harness.parse_bindings("a_1 = [x, y]\nnot a binding\nb = []\na_1 = [z]\n")
        self.assertEqual(found, {"a_1": "[x, y]", "b": "[]"})
        self.assertEqual(stray, ["not a binding", "a_1 = [z]"])
        self.assertEqual(harness.split_elements("[x, y]"), ["x", "y"])
        self.assertEqual(harness.split_elements("[]"), [])
        self.assertIsNone(harness.split_elements("x"))


class Generator(unittest.TestCase):
    def test_the_corpus_is_deterministic(self) -> None:
        again = harness.build_corpus()
        self.assertEqual([(c.row, c.args, c.expected) for c in again.cases],
                         [(c.row, c.args, c.expected) for c in CORPUS.cases])
        self.assertEqual([(f.function, f.args) for f in again.failures], [(f.function, f.args) for f in CORPUS.failures])

    def test_the_promised_categories_are_present(self) -> None:
        counts = CORPUS.composition()["cases by category"]
        for category in ("envelope", "limbs", "removable_zeros", "i64_edges", "ties", "float_edges", "random",
                         "parse_accepted", "parse_malformed", "parse_outside", "parse_too_long", "parse_spelled",
                         "fixed", "from_i64", "to_i64", "from_f64", "arith", "round", "div", "div_random",
                         "div_quotient_correction", "div_quotient_clamp", "order", "failure_order"):
            self.assertGreater(counts[category], 0, category)
        rows = CORPUS.composition()["cases by row"]
        self.assertEqual(set(rows), set(harness.ROWS))

    def test_every_long_division_witness_divides_in_the_default_corpus(self) -> None:
        for category, witnesses in (("div_quotient_correction", harness.QUOTIENT_CORRECTION_WITNESSES),
                                    ("div_quotient_clamp", harness.QUOTIENT_CLAMP_WITNESSES)):
            divided = {(args[0].lstrip("-"), args[1], args[2])
                       for args in [c.args for c in CORPUS.cases if c.category == category and c.row == "row_try_div"]
                       + [f.args for f in CORPUS.failures if f.category == category and f.function == "try_decimal_div"]}
            self.assertEqual(divided, set(witnesses), category)
        self.assertEqual((len(harness.QUOTIENT_CORRECTION_WITNESSES), len(harness.QUOTIENT_CLAMP_WITNESSES)), (60, 30))

    def test_every_exported_callable_is_called(self) -> None:
        text = "\n".join(p.source for p in programs())
        for name in harness.DECIMAL_NAMES[1:]:
            self.assertRegex(text, rf"\b{name}\(", name)

    def test_every_failure_path_is_sampled(self) -> None:
        paths = {(f.function, f.error.kind, f.error.reason) for f in CORPUS.failures}
        sampled = {(p.failure.function, p.failure.error.kind, p.failure.error.reason) for p in programs() if p.failure}
        self.assertEqual(paths, sampled)
        expected = {
            ("decimal", "domain", "malformed"), ("decimal", "domain", "too_long"), ("decimal", "domain", "outside"),
            ("decimal_to_fixed_string", "domain", "scale"), ("decimal_to_fixed_string", "domain", "fixed_digits"),
            ("decimal_to_i64", "domain", "inexact"), ("decimal_to_i64", "overflow", "i64"),
            ("decimal_from_f64", "domain", "not_finite"), ("decimal_from_f64", "domain", "scale"),
            ("decimal_from_f64", "domain", "inexact"), ("decimal_from_f64", "domain", "outside"),
            ("decimal_add", "overflow", "overflow"), ("decimal_sub", "overflow", "overflow"),
            ("decimal_mul", "overflow", "overflow"), ("decimal_round", "domain", "scale"),
            ("decimal_round", "domain", "inexact"), ("decimal_div", "domain", "division_by_zero"),
            ("decimal_div", "domain", "scale"), ("decimal_div", "domain", "inexact"),
            ("decimal_div", "overflow", "overflow"), ("try_decimal_div", "overflow", "overflow"),
        }
        self.assertEqual(paths, expected)

    def test_every_failure_order_witness_runs_as_a_program(self) -> None:
        witnesses = [f for f in CORPUS.failures if f.category == "failure_order"]
        self.assertEqual(len(witnesses), 10)
        sampled = {p.failure.name for p in programs() if p.failure}
        self.assertTrue({f.name for f in witnesses} <= sampled)
        reasons = {(f.function, f.args[:2]): (f.error.kind, f.error.reason) for f in witnesses}
        self.assertEqual(reasons[("decimal_to_i64", ("9223372036854775808.5", "RejectInexact"))], ("domain", "inexact"))
        self.assertEqual(reasons[("decimal_div", ("5", "0"))], ("domain", "division_by_zero"))
        self.assertEqual(reasons[("decimal_div", ("1e37", "3e-38"))], ("domain", "inexact"))
        self.assertEqual(reasons[("decimal_from_f64", (1.1, 38))], ("domain", "inexact"))
        self.assertEqual(reasons[("decimal_round", ("2.5", 39))], ("domain", "scale"))

    def test_the_equality_row_runs_by_default(self) -> None:
        default = harness.make_programs(CORPUS, 150, 1)
        equality = [p for p in default if any(b.row == "row_eq" for b in p.bindings)]
        self.assertTrue(equality)
        self.assertTrue(all(b.row == "row_eq" for p in equality for b in p.bindings))
        placed = sum(len(b.cases) for p in default for b in p.bindings)
        self.assertEqual(placed, len(CORPUS.cases))

    def test_every_case_lands_in_exactly_one_run(self) -> None:
        for chunk in (150, 7, 1000):
            selected = harness.make_programs(CORPUS, chunk, 1)
            placed = [case for p in selected for binding in p.bindings for case in binding.cases]
            self.assertEqual(len(placed), len(CORPUS.cases))
            for p in selected:
                self.assertLessEqual(len(p.bindings), 1)
                for binding in p.bindings:
                    self.assertEqual(binding.name, harness.GRID_ROOT)
                    self.assertLessEqual(len(binding.cases), chunk)

    def test_each_run_reads_its_cases_through_its_image(self) -> None:
        by_image: dict[str, set[str]] = {}
        for p in programs():
            by_image.setdefault(p.image, set()).add(p.source)
            if p.failure is None:
                binding = p.bindings[0]
                params = harness.ROWS[binding.row].params
                self.assertEqual(p.image, binding.row)
                self.assertEqual(p.inputs.split("\n"),
                                 [line for case in binding.cases for line in harness.case_lines(params, case.args)])
            else:
                self.assertEqual(p.image, f"fail_{p.failure.function}")
                self.assertEqual(p.inputs.split("\n"),
                                 harness.case_lines(harness.FAILURE_EXPRS[p.failure.function], p.failure.args))
        self.assertTrue(all(len(sources) == 1 for sources in by_image.values()))
        self.assertEqual(set(by_image), set(harness.ROWS) | {f"fail_{f}" for f in harness.FAILURE_EXPRS})

    def test_the_default_trims_only_cross_products_and_twins(self) -> None:
        # The try_ forms see every input; a twin sees every fourth accepted input.
        for twin, try_row in (("row_from_f64", "row_try_from_f64"), ("row_div", "row_try_div"),
                              ("row_to_i64", "row_try_to_i64")):
            twin_args = {c.args for c in CORPUS.cases if c.row == twin}
            try_args = {c.args for c in CORPUS.cases if c.row == try_row}
            self.assertTrue(twin_args and twin_args <= try_args, twin)
            self.assertLess(len(twin_args), len(try_args) // 2, twin)
        # Every edge input keeps every scale and mode; a seeded random input keeps
        # every scale and one mode per scale.
        rounded: dict[str, set[tuple[int, str]]] = {}
        for args in ([c.args for c in CORPUS.cases if c.row == "row_round" and c.category == "round"]
                     + [f.args for f in CORPUS.failures if f.function == "decimal_round" and f.category == "round"]):
            rounded.setdefault(args[0], set()).add(args[1:])
        for text in harness.TIES + harness.ENVELOPE:
            self.assertEqual(len(rounded[text]), 49, text)
        self.assertTrue(any(len(pairs) == 7 for text, pairs in rounded.items()))

    def test_the_large_corpus_extends_the_default(self) -> None:
        large = harness.build_corpus(large=True)
        self.assertGreater(len(large.cases), 10 * len(CORPUS.cases))
        default_args = {(c.row, c.args) for c in CORPUS.cases if c.category not in ("random", "parse_spelled")}
        large_args = {(c.row, c.args) for c in large.cases}
        self.assertLessEqual(len(default_args - large_args), len(default_args) // 2)

    def test_f32_witnesses_separate_direct_rounding_from_rounding_through_f64(self) -> None:
        witnesses = harness.f32_midpoint_witnesses(8)
        disagree = 0
        for text in witnesses:
            x = ref.decimal(text)
            through = struct.unpack("<I", struct.pack("<f", float(x.value)))[0]
            disagree += ref.decimal_to_f32_bits(x) != through
        self.assertGreaterEqual(disagree, len(witnesses) // 3)
        corpus_floats = {c.args[0] for c in CORPUS.cases if c.row == "row_floats"}
        self.assertTrue(set(harness.f32_midpoint_witnesses(4)) <= corpus_floats)
        self.assertIn(ref.decimal_to_string(ref.decimal(harness.TINY_TEXT)), corpus_floats)

    def test_the_eval_and_c_runner_sees_every_program_on_both_lanes(self) -> None:
        runner = FakeRunner()
        selected = programs()[:5]
        harness.run_all(runner, selected, ("eval", "c"), 1, lambda _: None)
        self.assertEqual(sorted(runner.calls), sorted((lane, p.name) for p in selected for lane in ("eval", "c")))


def canary_programs() -> list[Program]:
    return harness.make_programs(CANARY, 100, 0, combined=True)


class Canary(unittest.TestCase):
    def test_the_canary_is_deterministic(self) -> None:
        again = harness.build_canary_corpus()
        self.assertEqual([(c.row, c.args, c.expected) for c in again.cases],
                         [(c.row, c.args, c.expected) for c in CANARY.cases])

    def test_every_row_and_callable_runs(self) -> None:
        self.assertEqual({c.row for c in CANARY.cases}, set(harness.ROWS))
        text = "\n".join(p.source for p in canary_programs())
        for name in harness.DECIMAL_NAMES[1:]:
            self.assertRegex(text, rf"\b{name}\(", name)

    def test_the_canary_is_one_program_with_no_failure_program(self) -> None:
        selected = canary_programs()
        self.assertEqual(len(selected), 1)
        self.assertFalse(any(p.failure for p in selected))

    def test_every_row_has_one_binding_in_the_program(self) -> None:
        selected = canary_programs()
        rows = [binding.row for p in selected for binding in p.bindings]
        self.assertEqual(sorted(rows), sorted(harness.ROWS))
        self.assertEqual(sum(len(b.cases) for p in selected for b in p.bindings), len(CANARY.cases))
        for p in selected:
            self.assertEqual(p.name, p.image)
            for binding in p.bindings:
                self.assertEqual(binding.name, f"out_{binding.row}")
                self.assertIn(f"{binding.name} = {{", p.source)
            others = set(harness.ROWS) - {b.row for b in p.bindings}
            for row in others:
                self.assertNotIn(f"out_{row} =", p.source)

    def test_each_block_reads_its_own_lines(self) -> None:
        for p in canary_programs():
            first = 0
            lines = p.inputs.split("\n")
            for binding in p.bindings:
                params = harness.ROWS[binding.row].params
                block = [line for case in binding.cases for line in harness.case_lines(params, case.args)]
                self.assertEqual(lines[first:first + len(block)], block)
                self.assertIn(f"add({first}i64, mul(k, {len(params)}i64))", p.source)
                self.assertIn(f"range(0i64, {len(binding.cases)}i64)", p.source)
                first += len(block)
            self.assertEqual(first, len(lines))

    def test_a_conforming_module_passes_the_canary(self) -> None:
        report = harness.run_all(FakeRunner(), canary_programs(), ("eval", "c"), 2, lambda _: None)
        self.assertEqual(report.problems, [])
        self.assertEqual(report.cases, len(CANARY.cases))
        self.assertEqual(report.failures_checked, 0)
        self.assertTrue(harness.summary(report, ("eval", "c"), "canary").endswith("on lanes eval+c; canary corpus)"))

    def test_a_wrong_or_missing_row_in_a_shared_program_is_named(self) -> None:
        def wrong(lane, program, status, stdout, stderr):
            if lane == "c":
                stdout = re.sub(r"^(out_row_round = \[)[^,\]]*", r"\g<1>7", stdout, flags=re.M)
            return status, stdout, stderr
        report = harness.run_all(FakeRunner(wrong), canary_programs(), ("eval", "c"), 2, lambda _: None)
        self.assertEqual(report.classes, {"value": 1})
        self.assertIn("[c] out_row_round[0]", report.problems[0])

        def missing(lane, program, status, stdout, stderr):
            if lane == "eval":
                stdout = "".join(line for line in stdout.splitlines(True) if not line.startswith("out_row_eq = "))
            return status, stdout, stderr
        report = harness.run_all(FakeRunner(missing), canary_programs(), ("eval", "c"), 2, lambda _: None)
        self.assertEqual(report.classes, {"value": 1})
        self.assertIn("out_row_eq: no output line", report.problems[0])


# ---------------------------------------------------------------------------
# Defect models for the canary. A model is a set of `decimal_reference`
# attributes to replace, so that the reference computes what a module with that
# defect would. Magnitudes are little-endian base-10^9 limbs of the canonical
# coefficient, as Std.Decimal stores them.

LIMB = ref.LIMB_BASE


def limb(x: int, k: int) -> int:
    return x // LIMB**k % LIMB


def limb_count(x: int) -> int:
    """The limbs up to the highest nonzero one; zero has one."""
    count = 1
    while x >= LIMB**count:
        count += 1
    return count


def limbwise(op: Callable[[int, int], int]) -> Callable[[int, int], int]:
    """`op` limb by limb, each result reduced to its own limb, so that no carry or borrow crosses a limb."""
    def apply(x: int, y: int) -> int:
        return sum(op(limb(x, k), limb(y, k)) % LIMB * LIMB**k for k in range(max(limb_count(x), limb_count(y))))
    return apply


def carryless_product(x: int, y: int) -> int:
    """The schoolbook product with the carry out of each limb product dropped."""
    return sum(limb(x, i) * limb(y, j) % LIMB * LIMB ** (i + j)
               for i in range(limb_count(x)) for j in range(limb_count(y)))


def sums(add: Callable[[int, int], int], subtract: Callable[[int, int], int]) -> dict[str, object]:
    """`decimal_add` and `decimal_sub` that combine the magnitudes, aligned to the wider scale,
    with `add` when the signs agree and otherwise with `subtract`, larger minus smaller."""
    def signed(function: str, negate: bool) -> Callable[[ref.Decimal, ref.Decimal], ref.Decimal]:
        def call(a: ref.Decimal, b: ref.Decimal) -> ref.Decimal:
            scale = max(a.scale, b.scale)
            left, right = (abs(x.coefficient) * 10 ** (scale - x.scale) for x in (a, b))
            a_negative, b_negative = a.coefficient < 0, (b.coefficient < 0) != negate
            if a_negative == b_negative:
                magnitude, negative = add(left, right), a_negative
            elif left >= right:
                magnitude, negative = subtract(left, right), a_negative
            else:
                magnitude, negative = subtract(right, left), b_negative
            return ref.arithmetic(function, Fraction(-magnitude if negative else magnitude, 10**scale))
        return call
    return {"decimal_add": signed("decimal_add", False), "decimal_sub": signed("decimal_sub", True)}


def product(multiply: Callable[[int, int], int]) -> dict[str, object]:
    """`decimal_mul` that multiplies the magnitudes with `multiply`."""
    def decimal_mul(a: ref.Decimal, b: ref.Decimal) -> ref.Decimal:
        magnitude = multiply(abs(a.coefficient), abs(b.coefficient))
        negative = (a.coefficient < 0) != (b.coefficient < 0)
        return ref.arithmetic("decimal_mul", Fraction(-magnitude if negative else magnitude, 10 ** (a.scale + b.scale)))
    return {"decimal_mul": decimal_mul}


def long_division(corrections: int) -> Callable[[int, int], tuple[int, int]]:
    """Std.Decimal's floor quotient and remainder of magnitudes: by a one-limb divisor
    directly, otherwise by Knuth's algorithm D, which estimates each quotient limb from
    the remainder's top two limbs, clamps it to 10^9 - 1 and lowers it by at most
    `corrections` steps. A step that subtracts too much keeps only the remainder's own
    limbs, as the limb subtraction does."""
    def divide(num: int, den: int) -> tuple[int, int]:
        size = limb_count(den)
        if size == 1:
            return divmod(num, den)
        factor = LIMB // (limb(den, size - 1) + 1)
        divisor, rest = den * factor, num * factor
        lead = limb(divisor, size - 1)
        quotient = 0
        for position in reversed(range(limb_count(num) - size + 1)):
            digit = min((limb(rest, position + size) * LIMB + limb(rest, position + size - 1)) // lead, LIMB - 1)
            for _ in range(corrections):
                if divisor * digit * LIMB**position > rest:
                    digit -= 1
            quotient += digit * LIMB**position
            rest = (rest - divisor * digit * LIMB**position) % LIMB ** limb_count(rest)
        return quotient, rest // factor
    return divide


def quotient(divide: Callable[[int, int], tuple[int, int]]) -> dict[str, object]:
    """`decimal_div` that takes its magnitude quotient and remainder from `divide` and
    rounds them by the reference. A remainder of at least the divisor counts as more
    than half of it, as the module's comparison of twice the remainder with the
    divisor does."""
    def decimal_div(a: ref.Decimal, b: ref.Decimal, n: int, mode: str, function: str = "decimal_div") -> ref.Decimal:
        if b.coefficient == 0:
            raise ref.domain(function, "division_by_zero", "division by zero")
        ref.check_scale(function, n)
        shift = n + b.scale - a.scale
        num, den = abs(a.coefficient) * 10 ** max(shift, 0), abs(b.coefficient) * 10 ** max(-shift, 0)
        whole, remainder = divide(num, den)
        magnitude = whole + (Fraction(remainder, den) if remainder < den else Fraction(3, 4))
        negative = (a.coefficient < 0) != (b.coefficient < 0)
        result = ref.canonical(ref.round_to_quantum((-magnitude if negative else magnitude) / 10**n, n, mode, function))
        if result is None:
            raise ref.overflow(function, "overflow", pattern=ref.OUTSIDE_RANGE)
        return result
    return {"decimal_div": decimal_div}


# How a tie is broken: the magnitude it rounds to, from its floor and its sign.
TIES: dict[str, Callable[[int, bool], int]] = {
    "toward zero": lambda floor, negative: floor,
    "away from zero": lambda floor, negative: floor + 1,
    "toward positive": lambda floor, negative: floor + (not negative),
    "toward negative": lambda floor, negative: floor + negative,
    "to even": lambda floor, negative: floor + floor % 2,
}
WRONG_TIES = ("toward zero", "away from zero", "toward positive", "toward negative")
MODE_CALLABLES = ("decimal_to_i64", "decimal_from_f64", "decimal_round", "decimal_div")
DIRECTED = ("RoundTowardNegative", "RoundTowardPositive", "RoundTowardZero", "RoundAwayFromZero")


def rounding(callable_name: str, mode: str, as_mode: str | None = None, tie: str | None = None) -> dict[str, object]:
    """`round_to_quantum` where `callable_name`, or its `try_` twin, rounds under `mode`
    as under `as_mode`, or breaks an exact tie by `TIES[tie]`."""
    original = ref.round_to_quantum

    def round_to_quantum(v: Fraction, n: int, given: str, function: str, shown=None) -> Fraction:
        if given != mode or function.removeprefix("try_") != callable_name:
            return original(v, n, given, function, shown)
        floor, rest = divmod(abs(v) * 10**n, 1)
        if tie is not None and rest == Fraction(1, 2):
            magnitude = TIES[tie](floor, v < 0)
            return Fraction(-magnitude if v < 0 else magnitude, 10**n)
        return original(v, n, as_mode or given, function, shown)
    return {"round_to_quantum": round_to_quantum}


def binary_rounding(fmt: tuple[int, int], tie: str) -> dict[str, object]:
    """`round_binary` that breaks an exact tie in `fmt` by `TIES[tie]`, in units in the last place."""
    original = ref.round_binary

    def round_binary(q: Fraction, given: tuple[int, int]) -> Fraction:
        rounded = original(q, given)
        if given != fmt or q == 0:
            return rounded
        precision, exponent_bits = fmt
        lowest = 2 - 2 ** (exponent_bits - 1) - (precision - 1)
        unit = Fraction(2) ** max(ref.floor_log2(abs(q)) - (precision - 1), lowest)
        floor, rest = divmod(abs(q) / unit, 1)
        if rest != Fraction(1, 2):
            return rounded
        magnitude = TIES[tie](floor, q < 0) * unit
        return -magnitude if q < 0 else magnitude
    return {"round_binary": round_binary}


def f32_through_f64(x: ref.Decimal) -> int:
    return struct.unpack("<I", struct.pack("<f", ref.decimal_to_f64(x)))[0]


def unpadded_text(x: ref.Decimal) -> str:
    """`decimal_to_string` printing each limb below the top without its leading zeros."""
    magnitude = abs(x.coefficient)
    digits = "".join(str(limb(magnitude, k)) for k in reversed(range(limb_count(magnitude))))
    return ref.render(-int(digits) if x.coefficient < 0 else int(digits), x.scale)


# Each defect class the canary's docstring names, with its models by name.
CANARY_DEFECTS: dict[str, dict[str, dict[str, object]]] = {
    "the reference's single digit limit or scale limit moved by one": {
        "37 digits": {"MAX_DIGITS": 37, "MAX_COEFFICIENT": 10**37 - 1},
        "39 digits": {"MAX_DIGITS": 39, "MAX_COEFFICIENT": 10**39 - 1},
        "scale 37": {"MAX_SCALE": 37},
        "scale 39": {"MAX_SCALE": 39},
    },
    "every inner limb printed without its leading zeros": {"unpadded": {"decimal_to_string": unpadded_text}},
    "every limb carry dropped in addition": {"carry": sums(limbwise(operator.add), operator.sub)},
    "every limb borrow dropped in subtraction": {"borrow": sums(operator.add, limbwise(operator.sub))},
    "every limb product's carry dropped in multiplication": {"carry": product(carryless_product)},
    "one directed mode of one callable rounding as another directed mode on every input": {
        f"{name} {mode} as {other}": rounding(name, mode, as_mode=other)
        for name in MODE_CALLABLES for mode in DIRECTED for other in DIRECTED if other != mode},
    "every RoundTiesToEven tie of one callable broken toward zero, away from zero, toward positive or toward negative": {
        f"{name} {tie}": rounding(name, "RoundTiesToEven", tie=tie) for name in MODE_CALLABLES for tie in WRONG_TIES},
    "every RoundTiesToAway tie of one callable broken to even, toward zero, toward positive or toward negative": {
        f"{name} {tie}": rounding(name, "RoundTiesToAway", tie=tie)
        for name in MODE_CALLABLES for tie in ("to even", "toward zero", "toward positive", "toward negative")},
    "one callable's RejectInexact rounding toward zero instead of failing, on every input": {
        name: rounding(name, "RejectInexact", as_mode="RoundTowardZero") for name in MODE_CALLABLES
        if name != "decimal_round"},
    "every binary conversion tie broken toward zero, away from zero, toward positive or toward negative": {
        f"f{width} {tie}": binary_rounding(fmt, tie)
        for width, fmt in ((64, ref.F64), (32, ref.F32)) for tie in WRONG_TIES},
    "every f32 conversion rounding through f64": {"through f64": {"decimal_to_f32_bits": f32_through_f64}},
    "every long division taking at most one quotient-digit correction, or none": {
        f"{n} corrections": quotient(long_division(n)) for n in (1, 0)},
}

# The same models without their defect, which must agree with the reference.
CONTROLS: dict[str, dict[str, object]] = {
    "sums": sums(operator.add, operator.sub),
    "product": product(operator.mul),
    "long division": quotient(long_division(2)),
    "exact division": quotient(divmod),
    **{f"f{width} ties to even": binary_rounding(fmt, "to even") for width, fmt in ((64, ref.F64), (32, ref.F32))},
    **{f"{name} {mode} ties {tie}": rounding(name, mode, tie=tie) for name in MODE_CALLABLES
       for mode, tie in (("RoundTiesToEven", "to even"), ("RoundTiesToAway", "away from zero"))},
}


def changed_cases(patches: dict[str, object], cases: Sequence[harness.Case]) -> list[harness.Case]:
    """The cases whose expected output the patched reference changes. A call that now
    fails changes its case: the canary would stop with that failure."""
    with contextlib.ExitStack() as stack:
        for name, value in patches.items():
            stack.enter_context(mock.patch.object(ref, name, value))
        changed = []
        for case in cases:
            try:
                outcome = tuple(harness.ROWS[case.row].expect(*case.args))
            except ref.DecimalError as error:
                outcome = (error.message,)
            if outcome != case.expected:
                changed.append(case)
        return changed


def docstring_classes() -> list[str]:
    lines = inspect.cleandoc(harness.build_canary_corpus.__doc__ or "").splitlines()
    return [line.removeprefix("- ") for line in lines if line.startswith("- ")]


class CanaryDefects(unittest.TestCase):
    def test_the_canary_claims_exactly_the_modelled_classes(self) -> None:
        self.assertEqual(docstring_classes(), list(CANARY_DEFECTS))

    def test_every_model_changes_an_expected_canary_output(self) -> None:
        for defect, models in CANARY_DEFECTS.items():
            for name, patches in models.items():
                with self.subTest(defect=defect, model=name):
                    self.assertTrue(changed_cases(patches, CANARY.cases))

    def test_the_models_without_their_defect_agree_with_the_reference(self) -> None:
        for name, patches in CONTROLS.items():
            with self.subTest(name):
                self.assertEqual(changed_cases(patches, CORPUS.cases), [])


class Literals(unittest.TestCase):
    def test_integer_literals(self) -> None:
        self.assertEqual(harness.lit(5), "5i64")
        self.assertEqual(harness.lit(-5), "(-5i64)")
        self.assertEqual(harness.lit(ref.I64_MIN), "(-9223372036854775807i64 - 1i64)")
        with self.assertRaises(ValueError):
            harness.lit(ref.I64_MAX + 1)

    def test_float_literals_denote_the_exact_value(self) -> None:
        rng = random.Random(7)
        values = [0.1, -0.0, 0.0, 5e-324, 1e16, 1.7976931348623157e308, 2.0**53, -2.5]
        values += [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0] for _ in range(500)]
        for x in values:
            if not math.isfinite(x):
                continue
            text = harness.f64_lit(x)
            self.assertRegex(text, r"^\(?-?\d+\.\d+(e-?\d+)?f64\)?$")
            body = text.strip("()").removesuffix("f64")
            self.assertEqual(struct.pack("<d", float(body)), struct.pack("<d", x), text)
        self.assertEqual(harness.f64_lit(-0.0), "(-0.0f64)")
        self.assertEqual(harness.f64_lit(1e16), "1.0e16f64")
        self.assertEqual(harness.f64_lit(math.nan), "(0.0f64 / 0.0f64)")
        self.assertEqual(harness.f64_lit(-math.inf), "(-1.0f64 / 0.0f64)")

    def test_string_literals_escape_what_the_grammar_requires(self) -> None:
        self.assertEqual(harness.text_lit('a"b\\c'), '"a\\"b\\\\c"')
        self.assertEqual(harness.text_lit("1\n\t\r\0"), '"1\\n\\t\\r\\0"')
        self.assertEqual(harness.text_lit("\x08\x7f"), '"\\u{8}\\u{7f}"')
        self.assertEqual(harness.text_lit("١"), '"١"')

    def test_row_definitions(self) -> None:
        self.assertEqual(harness.ROWS["row_parse"].definition(),
                         "def row_parse(p0: string) -> List[string] = [opt_text(try_decimal(p0))]\n")
        text = harness.ROWS["row_div"].definition()
        self.assertTrue(text.startswith("def row_div(t: (string, string, i64, Rounding)) -> List[string] = {\n"
                                        "  (p0, p1, p2, p3) = t\n"))
        self.assertEqual(harness.ROWS["row_div"].literal(("1", "3", -2, "RejectInexact")),
                         '("1", "3", (-2i64), RejectInexact)')
        with self.assertRaises(ValueError):
            harness.value_lit("Rounding", "RoundHalfUp")

    def test_programs_carry_only_the_definitions_they_reach(self) -> None:
        source = harness.program_source(["x = [opt_int(try_decimal_to_i64(decimal(\"1\"), RejectInexact))]"])
        self.assertIn("def opt_int(", source)
        self.assertNotIn("def opt_text(", source)
        self.assertNotIn("def row_", source)
        self.assertIn("import Std.Decimal (Decimal, decimal, try_decimal,", source)
        self.assertIn("import Std.Rounding (Rounding, RoundTowardNegative,", source)
        source = harness.grid_source("row_try_div")
        for name in ("row_try_div", "opt_text", "cases", "text_field", "coded_text", "int_field", "rounding_field"):
            self.assertIn(f"def {name}(", source)
        self.assertNotIn("def float_field(", source)
        self.assertIn(f'read_lines("{harness.CASES_FILE}")', source)
        self.assertIn(f"{harness.GRID_ROOT} = {{", source)
        self.assertIn("range(0i64, trunc_div(len(lines), 4i64))", source)
        self.assertIn("row_parse(text_field(index(lines, k)))", harness.grid_source("row_parse"))
        floats = harness.grid_source("row_from_f64")
        self.assertIn('if eq(line, "nan") then (0.0f64 / 0.0f64)', floats)
        self.assertIn("def finite_field(", floats)
        failure = harness.failure_source("decimal_from_f64")
        self.assertIn(f"{harness.FAILURE_ROOT} = {{", failure)
        self.assertIn("decimal_from_f64(float_field(index(lines, 0i64)), int_field(index(lines, 1i64)), "
                      "rounding_field(index(lines, 2i64)))", failure)

    def test_case_file_fields_read_back_exactly(self) -> None:
        self.assertEqual(harness.field_line("string", " 1.5e-3"), "s: 1.5e-3")
        self.assertEqual(harness.field_line("string", ""), "s:")
        self.assertEqual(harness.field_line("string", "1\n"), "c:49,10")
        self.assertEqual(harness.field_line("string", "é"), "c:233")
        self.assertEqual(harness.field_line("i64", ref.I64_MIN), "-9223372036854775808")
        with self.assertRaises(ValueError):
            harness.field_line("i64", ref.I64_MAX + 1)
        for x in (0.1, -0.0, 5e-324, 1.7976931348623157e308, 1e22, 2.0**53 + 2):
            line = harness.field_line("f64", x)
            self.assertEqual(struct.pack("<d", float(line)), struct.pack("<d", x), line)
        self.assertEqual([harness.field_line("f64", x) for x in (math.nan, math.inf, -math.inf)], ["nan", "inf", "-inf"])
        self.assertEqual(harness.field_line("Rounding", "RejectInexact"), "RejectInexact")
        with self.assertRaises(ValueError):
            harness.field_line("Rounding", "RoundHalfUp")
        for case in CORPUS.cases:
            for line in harness.case_lines(harness.ROWS[case.row].params, case.args):
                self.assertTrue(line.isascii() and line.isprintable(), line)

    def test_failure_expressions(self) -> None:
        self.assertEqual(harness.failure_expr("decimal", ("1x",)), 'decimal_to_string(decimal("1x"))')
        self.assertEqual(harness.failure_expr("decimal_to_fixed_string", ("1.5", 0)),
                         'decimal_to_fixed_string(decimal("1.5"), 0i64)')
        self.assertEqual(harness.failure_expr("decimal_from_f64", (math.nan, 2, "RejectInexact")),
                         "decimal_to_string(decimal_from_f64((0.0f64 / 0.0f64), 2i64, RejectInexact))")
        self.assertEqual(harness.failure_expr("try_decimal_div", ("1", "3", 38, "RoundTiesToEven")),
                         'opt_text(try_decimal_div(decimal("1"), decimal("3"), 38i64, RoundTiesToEven))')


class Invocation(unittest.TestCase):
    def test_the_printed_compile_command_is_retargeted(self) -> None:
        stdout = ("Wrote out/main.c and out/main.h\n"
                  "Compile: clang -O2 -march=native out/main.c out/libchelis_runtime.a -lm -o out/main\n")
        self.assertEqual(harness.printed_compile_command(stdout),
                         ["clang", "-O2", "-march=native", "out/main.c", "out/libchelis_runtime.a", "-lm", "-o", "out/case"])
        self.assertIsNone(harness.printed_compile_command("Wrote out/main.c\n"))

    def test_the_binary_comes_from_the_flag_or_the_environment(self) -> None:
        self.assertEqual(harness.resolve_chelis(Path("/x/chelis")), Path("/x/chelis"))
        with mock.patch.dict(os.environ, {"CHELIS_BIN": "/y/chelis"}):
            self.assertEqual(harness.resolve_chelis(None), Path("/y/chelis"))
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit):
            harness.resolve_chelis(None)

    def test_list_prints_the_composition_without_a_binary(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(harness.main(["--list"]), 0)
        self.assertIn("cases by category:", out.getvalue())
        self.assertIn("decimal_from_f64/domain/not_finite", out.getvalue())

    def test_lanes_are_validated(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            harness.main(["--list", "--lanes", "eval,hip"])

    def test_an_image_is_built_once_and_run_per_case_file(self) -> None:
        runs = [p for p in programs() if p.image == "row_round"]
        self.assertGreater(len(runs), 1)
        for keep in (False, True):
            with harness.tempfile.TemporaryDirectory() as scratch:
                runner = harness.Runner(Path("/bin/false"), Path(scratch), 5, Path(scratch) / "home", "0.4.0",
                                        keep_artifacts=keep, compiler_version="9.9.9")
                calls: list[tuple[list[str], Path]] = []

                def fake_run(argv, cwd):
                    calls.append((argv, cwd))
                    if argv[1:2] == ["build"]:
                        (cwd / "out").mkdir()
                        (cwd / "out" / "libchelis_runtime.a").write_bytes(b"x")
                        return harness.subprocess.CompletedProcess(argv, 0, "Compile: cc out/main.c -o out/main\n", "")
                    if argv[0] == "cc":
                        (cwd / "out" / "case").write_bytes(b"binary")
                    return harness.subprocess.CompletedProcess(argv, 0, "", "")

                runner.run = fake_run
                with harness.ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(runner.c_lane, runs))
                self.assertEqual({(r.status, r.stage) for r in results}, {(0, "run")})
                builds = [argv for argv, _ in calls if argv[1:2] == ["build"]]
                self.assertEqual(len(builds), 1)
                self.assertIn("--emit-c", builds[0])
                image = Path(scratch) / "c" / "row_round"
                self.assertEqual(sorted(cwd for argv, cwd in calls if argv[0] == str(image / "out" / "case")),
                                 sorted(Path(scratch) / "c-runs" / p.name for p in runs))
                for p in runs:
                    self.assertEqual((Path(scratch) / "c-runs" / p.name / harness.CASES_FILE).read_text(encoding="utf-8"),
                                     p.inputs)
                self.assertTrue((image / "src" / "main.ch").exists())
                self.assertTrue((image / "out" / "case").exists())
                self.assertEqual((image / "out" / "libchelis_runtime.a").exists(), keep)

    def test_a_failed_image_build_stops_every_run_of_it(self) -> None:
        runs = [p for p in programs() if p.image == "row_round"]
        with harness.tempfile.TemporaryDirectory() as scratch:
            runner = harness.Runner(Path("/bin/false"), Path(scratch), 5, Path(scratch) / "home", "0.4.0",
                                    compiler_version="9.9.9")
            runner.run = lambda argv, cwd: harness.subprocess.CompletedProcess(argv, 1, "", "error: no")
            results = [runner.c_lane(p) for p in runs]
        self.assertEqual({(r.status, r.stage, r.stderr) for r in results}, {(1, "build", "error: no")})

    def test_programs_are_packages_depending_on_chelis_std(self) -> None:
        program = programs()[0]
        with harness.tempfile.TemporaryDirectory() as scratch:
            home = Path(scratch) / "home"
            runner = harness.Runner(Path("/bin/false"), Path(scratch), 5, home, "0.4.0", compiler_version="9.9.9")
            app, main = runner.app(Path(scratch) / "app", program.source)
            self.assertEqual(main, "src/main.ch")
            self.assertEqual((app / "src" / "main.ch").read_text(encoding="utf-8"), program.source)
            manifest = (app / "reef.toml").read_text(encoding="utf-8")
            self.assertIn('compiler = "=9.9.9"', manifest)
            self.assertIn('chelis-std = { version = "0.4.0" }', manifest)
            self.assertEqual(runner.env()["CHELIS_REEF_HOME"], str(home))
        self.assertRegex(harness.std_version(), r"^\d+\.\d+\.\d+$")

    def test_publishing_failure_stops_the_run(self) -> None:
        with harness.tempfile.TemporaryDirectory() as scratch, self.assertRaises(SystemExit):
            harness.publish_std(Path("/usr/bin/false"), Path(scratch))

if __name__ == "__main__":
    unittest.main()
