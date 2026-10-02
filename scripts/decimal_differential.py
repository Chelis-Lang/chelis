#!/usr/bin/env python3
"""Differential oracle for `Std.Decimal` ([05-OP-76], chelis#2778): eval and C against the reference.

The harness generates golden vectors from a fixed seed, renders them into
Chelis programs that call `Std.Decimal`, runs each program through
`chelis eval` and through `chelis build --target c` plus the native compiler,
and compares each lane's complete output with the values that
`decimal_reference.py` computes independently. A lane agreeing with the other
lane is not enough: each must equal the reference.

Observations:

- Each row function and each failing function is one program that reads its
  inputs from `cases.txt` at run time, so the C lane builds it once and runs
  the binary once per case file, and the eval lane evaluates it once per case
  file. The canary instead runs every row in one program, where row `r` reads
  its own block of the case file and prints `out_r`.
- A grid run maps a row function over a chunk of input tuples and prints a
  flat `List[string]`. Each case contributes a fixed number of elements: a decimal as its canonical text, an `Option` as that
  text or `None`, an i64 or bool through `to_string`, and a float through
  `to_string`, whose shortest round-trip text the comparator reads back to its
  bit pattern at the float's own width. Rows that use a twin and its `try_`
  form observe every input through the `try_` form and the twin on the inputs
  the reference accepts (every fourth of them in the default corpus).
- A failure run makes one failing call, which must fail. Each lane must exit
  nonzero with a message `<function>: <kind>: <detail>` whose function and
  kind equal the reference's, whose detail matches the pinned message shape,
  and which both lanes spell identically. A primitive trap escaping instead
  shows up as a message of the wrong shape. Repeated paths are sampled.

Every printed line must belong to a binding of the program.

Profiles: the default corpus is the edge corpus nightly CI runs (envelope
boundaries, limb boundaries and carry chains, removable zeros, i64 boundaries, ties in
every mode and sign, subnormal and double-rounding f32 witnesses, long divisions
that need the quotient digit's second correction or its clamp, the parser's
accepted and rejected spellings, and seeded random values, which take one
rounding mode per scale in rotation); `--large` adds many more seeded random
cases, crosses each with every mode, observes every twin, and samples more
failures, for the manual gate of `docs/manual_gates.md`; `--canary` is a few
hand-picked edge inputs with no failure programs, for pull-request CI, run in
one program that covers every row.

The supported entry point is `crates/chelis-cli/tests/std_decimal_oracle.rs`,
which supplies the freshly built binary, a published standard library and the
strict reference toolchain (`chelis_backend_c::toolchain`). Run directly, the
binary under test comes from `--chelis` or the `CHELIS_BIN` environment
variable. Each program is a package depending on `chelis-std`, the way a shell
uses it: `--reef-home` names a reef home where it is published, and without it
the harness publishes this checkout's `packages/chelis-std` into a fresh one
first. The C lane emits each program's source with `chelis build --emit-c` and
compiles it with `--toolchain-json` when given, and otherwise with the command
`chelis build --emit-c` prints.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tomllib

import decimal_reference as ref
from decimal_reference import Decimal, DecimalError, I64_MAX, I64_MIN, ROUNDINGS

SEED = 2778
PASS_MARKER = "STD DECIMAL ORACLE: PASS"
FAIL_MARKER = "STD DECIMAL ORACLE: FAIL"
BOTH = ("eval", "c")
# A program reads its inputs from this file in its working directory, so one
# compiled program serves every chunk of its row and every failing call of its
# function.
CASES_FILE = "cases.txt"
# The root a grid program prints, and the root a failure program must fail in.
GRID_ROOT = "outputs"
FAILURE_ROOT = "outcome"
REPO = Path(__file__).resolve().parent.parent

DECIMAL_NAMES = (
    "Decimal", "decimal", "try_decimal", "decimal_to_string", "decimal_to_fixed_string", "decimal_from_i64",
    "decimal_to_i64", "try_decimal_to_i64", "decimal_from_f64", "try_decimal_from_f64", "decimal_to_f64",
    "decimal_to_f32", "decimal_scale", "decimal_add", "decimal_sub", "decimal_mul", "decimal_round", "decimal_div",
    "try_decimal_div", "decimal_lt", "decimal_lte", "decimal_gt", "decimal_gte",
)
ROUNDING_NAMES = ("Rounding", *ROUNDINGS)

NINES = "9" * 38
MAX_TEXT = NINES
TINY_TEXT = "0." + "0" * 37 + "1"


# ---------------------------------------------------------------------------
# Chelis source builders.


def lit(n: int) -> str:
    if not ref.fits_i64(n):
        raise ValueError(f"{n} does not fit in i64")
    if n == I64_MIN:
        return "(-9223372036854775807i64 - 1i64)"
    return f"({n}i64)" if n < 0 else f"{n}i64"


def text_lit(s: str) -> str:
    """A Chelis string literal: the named escapes, `\\u{h}` for other controls."""
    named = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\t": "\\t", "\r": "\\r", "\0": "\\0"}
    out = []
    for ch in s:
        if ch in named:
            out.append(named[ch])
        elif ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F:
            out.append(f"\\u{{{ord(ch):x}}}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def f64_lit(x: float) -> str:
    """An f64 literal that denotes exactly `x`; non-finite values are IEEE quotients."""
    if math.isnan(x):
        return "(0.0f64 / 0.0f64)"
    if math.isinf(x):
        return "(1.0f64 / 0.0f64)" if x > 0 else "(-1.0f64 / 0.0f64)"
    text = repr(abs(x))
    mantissa, _, exponent = text.partition("e")
    if "." not in mantissa:
        mantissa += ".0"
    body = mantissa + (f"e{int(exponent)}" if exponent else "") + "f64"
    return f"(-{body})" if math.copysign(1.0, x) < 0 else body


def f64_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


@dataclass(frozen=True)
class FloatBits:
    """An expected float element, compared by bit pattern at its width."""

    fmt: tuple[int, int]
    bits: int

    def __str__(self) -> str:
        width = 64 if self.fmt == ref.F64 else 32
        if self.fmt == ref.F64:
            shown = repr(struct.unpack("<d", struct.pack("<Q", self.bits))[0])
        else:
            shown = repr(struct.unpack("<f", struct.pack("<I", self.bits))[0])
        return f"f{width}:{self.bits:#x}({shown})"


Element = str | FloatBits


def opt_text(x: Decimal | None) -> str:
    return "None" if x is None else ref.decimal_to_string(x)


def opt_int(v: int | None) -> str:
    return "None" if v is None else str(v)


def text_and_scale(x: Decimal) -> list[Element]:
    return [ref.decimal_to_string(x), str(x.scale)]


def b(v: bool) -> str:
    return "true" if v else "false"


def float_echo(x: float) -> FloatBits:
    return FloatBits(ref.F64, f64_bits(x))


@dataclass(frozen=True)
class RowKind:
    """A Chelis row function and its Python mirror over the same input tuple."""

    name: str
    params: tuple[str, ...]
    lines: tuple[str, ...]
    expect: Callable[..., list[Element]]

    def definition(self) -> str:
        body = list(self.lines)
        if len(self.params) == 1:
            head = f"def {self.name}(p0: {self.params[0]}) -> List[string] ="
            if len(body) == 1:
                return f"{head} {body[0]}\n"
            return head + " {\n" + "".join(f"  {line}\n" for line in body) + "}\n"
        names = ", ".join(f"p{k}" for k in range(len(self.params)))
        head = f"def {self.name}(t: {self.tuple_type()}) -> List[string] = {{\n  ({names}) = t\n"
        return head + "".join(f"  {line}\n" for line in body) + "}\n"

    def tuple_type(self) -> str:
        if len(self.params) == 1:
            return self.params[0]
        return "(" + ", ".join(self.params) + ")"

    def literal(self, args: tuple) -> str:
        parts = [value_lit(kind, value) for kind, value in zip(self.params, args)]
        return parts[0] if len(parts) == 1 else "(" + ", ".join(parts) + ")"


def value_lit(kind: str, value: object) -> str:
    if kind == "string":
        return text_lit(value)
    if kind == "i64":
        return lit(value)
    if kind == "f64":
        return f64_lit(value)
    if kind == "Rounding":
        if value not in ROUNDINGS:
            raise ValueError(value)
        return value
    raise ValueError(kind)


def d(text: str) -> Decimal:
    return ref.decimal(text)


ROWS: dict[str, RowKind] = {}


def row(name: str, params: Sequence[str], lines: Sequence[str], expect: Callable[..., list[Element]]) -> None:
    ROWS[name] = RowKind(name, tuple(params), tuple(lines), expect)


row("row_parse", ["string"], ["[opt_text(try_decimal(p0))]"], lambda t: [opt_text(ref.try_decimal(t))])
row("row_text", ["string"], ["x = decimal(p0)", "[decimal_to_string(x), to_string(decimal_scale(x))]"],
    lambda t: text_and_scale(d(t)))
row("row_floats", ["string"], ["x = decimal(p0)", "[to_string(decimal_to_f64(x)), to_string(decimal_to_f32(x))]"],
    lambda t: [FloatBits(ref.F64, ref.decimal_to_f64_bits(d(t))), FloatBits(ref.F32, ref.decimal_to_f32_bits(d(t)))])
row("row_fixed", ["string", "i64"], ["[decimal_to_fixed_string(decimal(p0), p1)]"],
    lambda t, n: [ref.decimal_to_fixed_string(d(t), n)])
row("row_from_i64", ["i64"], ["x = decimal_from_i64(p0)", "[decimal_to_string(x), to_string(decimal_scale(x))]"],
    lambda v: text_and_scale(ref.decimal_from_i64(v)))
row("row_to_i64", ["string", "Rounding"], ["[to_string(decimal_to_i64(decimal(p0), p1))]"],
    lambda t, r: [str(ref.decimal_to_i64(d(t), r))])
row("row_try_to_i64", ["string", "Rounding"], ["[opt_int(try_decimal_to_i64(decimal(p0), p1))]"],
    lambda t, r: [opt_int(ref.try_decimal_to_i64(d(t), r))])
row("row_from_f64", ["f64", "i64", "Rounding"], ["[to_string(p0), decimal_to_string(decimal_from_f64(p0, p1, p2))]"],
    lambda x, n, r: [float_echo(x), ref.decimal_to_string(ref.decimal_from_f64(x, n, r))])
row("row_try_from_f64", ["f64", "i64", "Rounding"], ["[to_string(p0), opt_text(try_decimal_from_f64(p0, p1, p2))]"],
    lambda x, n, r: [float_echo(x), opt_text(ref.try_decimal_from_f64(x, n, r))])
row("row_add", ["string", "string"], ["[decimal_to_string(decimal_add(decimal(p0), decimal(p1)))]"],
    lambda a, c: [ref.decimal_to_string(ref.decimal_add(d(a), d(c)))])
row("row_sub", ["string", "string"], ["[decimal_to_string(decimal_sub(decimal(p0), decimal(p1)))]"],
    lambda a, c: [ref.decimal_to_string(ref.decimal_sub(d(a), d(c)))])
row("row_mul", ["string", "string"], ["[decimal_to_string(decimal_mul(decimal(p0), decimal(p1)))]"],
    lambda a, c: [ref.decimal_to_string(ref.decimal_mul(d(a), d(c)))])
row("row_round", ["string", "i64", "Rounding"], ["[decimal_to_string(decimal_round(decimal(p0), p1, p2))]"],
    lambda t, n, r: [ref.decimal_to_string(ref.decimal_round(d(t), n, r))])
row("row_div", ["string", "string", "i64", "Rounding"], ["[decimal_to_string(decimal_div(decimal(p0), decimal(p1), p2, p3))]"],
    lambda a, c, n, r: [ref.decimal_to_string(ref.decimal_div(d(a), d(c), n, r))])
row("row_try_div", ["string", "string", "i64", "Rounding"], ["[opt_text(try_decimal_div(decimal(p0), decimal(p1), p2, p3))]"],
    lambda a, c, n, r: [opt_text(ref.try_decimal_div(d(a), d(c), n, r))])
row("row_order", ["string", "string"],
    ["x = decimal(p0)", "y = decimal(p1)",
     "[to_string(decimal_lt(x, y)), to_string(decimal_lte(x, y)), to_string(decimal_gt(x, y)), to_string(decimal_gte(x, y))]"],
    lambda a, c: [b(ref.decimal_lt(d(a), d(c))), b(ref.decimal_lte(d(a), d(c))), b(ref.decimal_gt(d(a), d(c))),
                  b(ref.decimal_gte(d(a), d(c)))])
# [05-OP-36] structural equality. Outside the canary every row kind runs in
# programs of its own, so a rejection of `==` on the opaque type cannot hide any
# other row; in the canary's one program it fails every row.
row("row_eq", ["string", "string"], ["[to_string(decimal(p0) == decimal(p1))]"],
    lambda a, c: [b(ref.decimal_eq(d(a), d(c)))])


# How a failing call is spelled in a program of its own, by function.
FAILURE_EXPRS = {
    "decimal": ("string",), "decimal_to_fixed_string": ("string", "i64"), "decimal_to_i64": ("string", "Rounding"),
    "decimal_from_f64": ("f64", "i64", "Rounding"), "decimal_add": ("string", "string"),
    "decimal_sub": ("string", "string"), "decimal_mul": ("string", "string"),
    "decimal_round": ("string", "i64", "Rounding"), "decimal_div": ("string", "string", "i64", "Rounding"),
    "try_decimal_div": ("string", "string", "i64", "Rounding"),
}


def failure_expr(function: str, args: tuple) -> str:
    """The failing call with its arguments as literals, as a report names it."""
    return failure_call(function, [value_lit(kind, value) for kind, value in zip(FAILURE_EXPRS[function], args)])


def failure_call(function: str, lits: Sequence[str]) -> str:
    """The failing call over argument expressions of the function's declared kinds."""
    kinds = FAILURE_EXPRS[function]
    decimals = [f"decimal({x})" if kind == "string" else x for kind, x in zip(kinds, lits)]
    if function == "decimal":
        return f"decimal_to_string(decimal({lits[0]}))"
    if function == "decimal_to_fixed_string":
        return f"decimal_to_fixed_string({', '.join(decimals)})"
    if function == "decimal_to_i64":
        return f"to_string(decimal_to_i64({', '.join(decimals)}))"
    if function == "try_decimal_div":
        return f"opt_text(try_decimal_div({', '.join(decimals)}))"
    return f"decimal_to_string({function}({', '.join(decimals)}))"


PRELUDE = {
    "opt_text": (
        "def opt_text(o: Option[Decimal]) -> string = match o with {\n"
        "  | Some(v) => decimal_to_string(v)\n"
        '  | None => "None"\n'
        "}\n"
    ),
    "opt_int": (
        "def opt_int(o: Option[i64]) -> string = match o with {\n"
        "  | Some(v) => to_string(v)\n"
        '  | None => "None"\n'
        "}\n"
    ),
    # Reading the case file: one line per field (`field_line`).
    "cases": f'def cases() -> List[string] ! {{ IO }} = read_lines("{CASES_FILE}")\n',
    "int_field": (
        "def int_field(line: string) -> i64 = match to_int(line) with {\n"
        "  | Some(v) => v\n"
        '  | None => fail(string_concat("not an i64 case field: ", line))\n'
        "}\n"
    ),
    # The non-finite values are quotients, as in `f64_lit`; the compiled lane's
    # `to_float` does not read their spellings yet (chelis#2870).
    "float_field": (
        'def float_field(line: string) -> f64 = if eq(line, "nan") then (0.0f64 / 0.0f64) '
        'else if eq(line, "inf") then (1.0f64 / 0.0f64) else if eq(line, "-inf") then (-1.0f64 / 0.0f64) '
        "else finite_field(line)\n"
    ),
    "finite_field": (
        "def finite_field(line: string) -> f64 = match to_float(line) with {\n"
        "  | Some(v) => v\n"
        '  | None => fail(string_concat("not an f64 case field: ", line))\n'
        "}\n"
    ),
    "coded_text": (
        "def coded_text(body: string) -> string = {\n"
        "  (text, digits) = fold(fn (acc: (string, string), at: i64) -> {\n"
        "    (done, pending) = acc\n"
        "    ch = string_slice(body, at, 1i64)\n"
        '    if eq(ch, ",") then (string_concat(done, char_from_code(int_field(pending))), "") '
        "else (done, string_concat(pending, ch))\n"
        '  }, ("", ""), range(0i64, string_len(body)))\n'
        "  string_concat(text, char_from_code(int_field(digits)))\n"
        "}\n"
    ),
    "text_field": (
        "def text_field(line: string) -> string = {\n"
        "  body = string_slice(line, 2i64, sub(string_len(line), 2i64))\n"
        '  if string_starts_with(line, "c:") then coded_text(body) else body\n'
        "}\n"
    ),
    "rounding_field": (
        "def rounding_field(line: string) -> Rounding = "
        + "".join(f'if eq(line, "{mode}") then {mode} else ' for mode in ROUNDINGS)
        + 'fail(string_concat("not a Rounding case field: ", line))\n'
    ),
}

FIELD_DECODERS = {"string": "text_field", "i64": "int_field", "f64": "float_field", "Rounding": "rounding_field"}


def field_line(kind: str, value: object) -> str:
    """One case-file line for a field: text as `s:` and its characters when they
    are all printable ASCII, otherwise `c:` and its code points; numbers in a
    form `to_int` and `to_float` read back exactly; a mode by its name."""
    if kind == "string":
        if all(0x20 <= ord(ch) <= 0x7E for ch in value):
            return "s:" + value
        return "c:" + ",".join(str(ord(ch)) for ch in value)
    if kind == "i64":
        if not ref.fits_i64(value):
            raise ValueError(f"{value} does not fit in i64")
        return str(value)
    if kind == "f64":
        return repr(value)
    if kind == "Rounding":
        if value not in ROUNDINGS:
            raise ValueError(value)
        return value
    raise ValueError(kind)


def case_lines(kinds: Sequence[str], args: tuple) -> list[str]:
    return [field_line(kind, value) for kind, value in zip(kinds, args)]


def field_reads(kinds: Sequence[str], first: int | str) -> list[str]:
    """Expressions decoding the fields of one case, which starts at line `first`."""
    def line(k: int) -> str:
        if isinstance(first, int):
            return f"{first + k}i64"
        return first if k == 0 else f"add({first}, {k}i64)"
    return [f"{FIELD_DECODERS[kind]}(index(lines, {line(k)}))" for k, kind in enumerate(kinds)]


def prelude_for(body: str) -> str:
    """The helper and row definitions `body` reaches, in a fixed order."""
    defs = dict(PRELUDE)
    defs.update({name: kind.definition() for name, kind in ROWS.items()})
    needed: set[str] = set()
    pending = [body]
    while pending:
        text = pending.pop()
        for name, block in defs.items():
            if name not in needed and re.search(rf"\b{name}\(", text):
                needed.add(name)
                pending.append(block)
    return "\n".join(block for name, block in defs.items() if name in needed)


def program_source(body: Sequence[str]) -> str:
    text = "\n".join(body)
    return (
        "module Demo.Main\n\n"
        f"import Std.Decimal ({', '.join(DECIMAL_NAMES)})\n"
        f"import Std.Rounding ({', '.join(ROUNDING_NAMES)})\n\n"
        + prelude_for(text) + "\n" + text + "\n"
    )


def grid_source(row_name: str) -> str:
    """The program that maps a row function over every case in the case file."""
    kind = ROWS[row_name]
    width = len(kind.params)
    reads = field_reads(kind.params, "k" if width == 1 else f"mul(k, {width}i64)")
    argument = reads[0] if width == 1 else "(" + ", ".join(reads) + ")"
    count = "len(lines)" if width == 1 else f"trunc_div(len(lines), {width}i64)"
    return program_source([
        f"{GRID_ROOT} = {{",
        "  lines = cases()",
        f"  flat_map(fn (k: i64) -> {row_name}({argument}), range(0i64, {count}))",
        "}",
    ])


def failure_source(function: str) -> str:
    """The program that makes the one failing call the case file describes."""
    reads = field_reads(FAILURE_EXPRS[function], 0)
    return program_source([f"{FAILURE_ROOT} = {{", "  lines = cases()", f"  [{failure_call(function, reads)}]", "}"])


def combined_grid_source(blocks: Sequence[tuple[str, int, int]]) -> str:
    """One program mapping each row function over its own block of the case file.

    A block is `(row, first line, case count)`, and row `r` prints `out_r`.
    """
    body = []
    for row_name, first, count in blocks:
        kind = ROWS[row_name]
        width = len(kind.params)
        reads = field_reads(kind.params, f"add({first}i64, mul(k, {width}i64))")
        argument = reads[0] if width == 1 else "(" + ", ".join(reads) + ")"
        body += [f"out_{row_name} = {{", "  lines = cases()",
                 f"  flat_map(fn (k: i64) -> {row_name}({argument}), range(0i64, {count}i64))", "}"]
    return program_source(body)


# ---------------------------------------------------------------------------
# The corpus.


@dataclass(frozen=True)
class Case:
    category: str
    row: str
    args: tuple
    expected: tuple[Element, ...]


@dataclass(frozen=True)
class Failure:
    name: str
    category: str
    function: str
    args: tuple
    error: DecimalError

    @property
    def expr(self) -> str:
        return failure_expr(self.function, self.args)


@dataclass
class Corpus:
    cases: list[Case] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    # A twin that has a `try_` form is observed on every `twin_stride`-th input
    # the reference accepts; the `try_` form observes every input.
    twin_stride: int = 1
    twin_calls: Counter = field(default_factory=Counter)

    def add(self, category: str, row_name: str, args: tuple) -> None:
        kind = ROWS[row_name]
        self.cases.append(Case(category, row_name, args, tuple(kind.expect(*args))))

    def fail(self, category: str, function: str, args: tuple, error: DecimalError) -> None:
        if error.function != function:
            raise AssertionError(f"{function} failed as {error.function}")
        name = f"f{len(self.failures):05d}_{function}_{error.kind}"
        self.failures.append(Failure(name, category, function, args, error))

    def outcome(self, category: str, function: str, row_name: str, args: tuple, compute: Callable[[], object],
                keep_case: bool = True) -> None:
        """A grid case when the reference accepts the call (if `keep_case`), a failure
        program when it fails."""
        try:
            compute()
        except DecimalError as error:
            self.fail(category, function, args, error)
            return
        if keep_case:
            self.add(category, row_name, args)

    def twin_kept(self, function: str) -> bool:
        self.twin_calls[function] += 1
        return (self.twin_calls[function] - 1) % self.twin_stride == 0

    # One method per callable family; each records the twin and its `try_` form.

    def parse(self, category: str, text: str) -> None:
        """Every text through `try_decimal`; the twin on accepted text, and in a failure
        program on rejected text unless a control character leaves the message's
        line structure open."""
        self.add(category, "row_parse", (text,))
        if not any(ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in text):
            self.outcome(category, "decimal", "row_text", (text,), lambda: d(text))

    def value(self, category: str, text: str) -> None:
        """A valid decimal's parse, text, scale, and both float conversions."""
        d(text)
        self.add(category, "row_parse", (text,))
        self.add(category, "row_text", (text,))
        self.add(category, "row_floats", (text,))

    def fixed(self, category: str, text: str, n: int) -> None:
        self.outcome(category, "decimal_to_fixed_string", "row_fixed", (text, n), lambda: ref.decimal_to_fixed_string(d(text), n))

    def from_i64(self, category: str, v: int) -> None:
        self.add(category, "row_from_i64", (v,))

    def to_i64(self, category: str, text: str, mode: str) -> None:
        self.add(category, "row_try_to_i64", (text, mode))
        self.outcome(category, "decimal_to_i64", "row_to_i64", (text, mode), lambda: ref.decimal_to_i64(d(text), mode),
                     self.twin_kept("decimal_to_i64"))

    def from_f64(self, category: str, x: float, n: int, mode: str) -> None:
        self.add(category, "row_try_from_f64", (x, n, mode))
        self.outcome(category, "decimal_from_f64", "row_from_f64", (x, n, mode), lambda: ref.decimal_from_f64(x, n, mode),
                     self.twin_kept("decimal_from_f64"))

    def arith(self, category: str, op: str, a: str, c: str) -> None:
        compute = {"add": ref.decimal_add, "sub": ref.decimal_sub, "mul": ref.decimal_mul}[op]
        self.outcome(category, f"decimal_{op}", f"row_{op}", (a, c), lambda: compute(d(a), d(c)))

    def round(self, category: str, text: str, n: int, mode: str) -> None:
        self.outcome(category, "decimal_round", "row_round", (text, n, mode), lambda: ref.decimal_round(d(text), n, mode))

    def div(self, category: str, a: str, c: str, n: int, mode: str) -> None:
        args = (a, c, n, mode)
        self.outcome(category, "try_decimal_div", "row_try_div", args, lambda: ref.try_decimal_div(d(a), d(c), n, mode))
        self.outcome(category, "decimal_div", "row_div", args, lambda: ref.decimal_div(d(a), d(c), n, mode),
                     self.twin_kept("decimal_div"))

    def order(self, category: str, a: str, c: str) -> None:
        self.add(category, "row_order", (a, c))
        self.add(category, "row_eq", (a, c))

    def composition(self) -> dict[str, Counter]:
        return {
            "cases by category": Counter(case.category for case in self.cases),
            "cases by row": Counter(case.row for case in self.cases),
            "failures by function, kind and reason": Counter(f"{f.function}/{f.error.kind}/{f.error.reason}"
                                                            for f in self.failures),
        }


# ---------------------------------------------------------------------------
# Corpus generators.


def random_text(rng: random.Random, max_digits: int = 38) -> str:
    """A random canonical decimal with 1 to `max_digits` significant digits and a random scale."""
    digits = rng.randint(1, max_digits)
    coefficient = rng.randrange(10 ** (digits - 1), 10**digits) * rng.choice((1, -1))
    scale = rng.randint(0, 38)
    value = ref.canonical(ref.Fraction(coefficient, 10**scale))
    assert value is not None
    return ref.decimal_to_string(value)


def spelled(rng: random.Random, text: str) -> str:
    """Another token for the same value: an exponent with leading zeros, or padding zeros."""
    x = d(text)
    sign = "-" if x.coefficient < 0 else ""
    if x.coefficient == 0:
        return "0"
    digits = str(abs(x.coefficient))
    choice = rng.randrange(3)
    if choice == 0:
        shift = rng.randint(0, 5)
        mantissa = digits + "0" * shift
        exponent = -x.scale - shift
        e = rng.choice("eE")
        exp_sign = "-" if exponent < 0 else rng.choice(("", "+"))
        return f"{sign}{mantissa}{e}{exp_sign}{'0' * rng.randint(0, 3)}{abs(exponent)}"
    if choice == 1:
        return ref.decimal_to_string(x) + ("" if x.scale else ".") + "0" * rng.randint(1, 4)
    head, tail = digits[0], digits[1:]
    exponent = len(tail) - x.scale
    return f"{sign}{head}.{tail or '0'}e{exponent}"


ENVELOPE = [
    MAX_TEXT, "-" + MAX_TEXT, TINY_TEXT, "-" + TINY_TEXT, "0." + NINES, "-0." + NINES, "1" + "0" * 37,
    "-1" + "0" * 37, "9." + "9" * 37, "9" * 37 + ".9", "0", "1", "-1", "0." + "0" * 36 + "99",
    "1" + "0" * 36 + "1", "5" + "0" * 37, "0." + "1" * 38,
]
LIMBS = (
    [("999999999" + "0" * k) for k in range(0, 30)]
    + [str(10 ** (9 * j) + delta) for j in (1, 2, 3, 4) for delta in (-1, 0, 1)]
    + ["9" * n for n in (18, 27, 36, 37)]
    + [ref.decimal_to_string(ref.canonical(ref.Fraction(999999999, 10**s))) for s in (9, 18, 27, 36, 38)]
    + ["-999999999999999999.999999999", "99" + "0" * 36, "1" + "0" * 36]
)
REMOVABLE = ["1.000", "100e-2", "1.50", "0.10e1", "2.50e1", "-0.0", "-0.000e-5", "120.00", "1e0", "1000e-3",
             "0.00100", "12.3400e2"]
I64_EDGES = ["9223372036854775807", "-9223372036854775808", "9223372036854775807.5", "-9223372036854775808.5",
             "9223372036854775808", "-9223372036854775809", "9223372036854775806.5", "-9223372036854775807.5",
             "9223372036854775807.4999999999999999999", "-9223372036854775808.4999999999999999999",
             "9223372036854775807.0000000000000000001", "-9223372036854775808.0000000000000000001"]
TIES = ["2.5", "-2.5", "0.5", "-0.5", "1.5", "-1.5", "3.5", "-3.5", "0.125", "-0.125", "2.45", "-2.45",
        "0.0000000000000000000000000000000000005", "-0.0000000000000000000000000000000000005",
        "4999999999999999999999999999999999999.5", "-4999999999999999999999999999999999999.5",
        "2.5000000000000000000000000000000000001", "-2.4999999999999999999999999999999999999",
        "0." + "0" * 36 + "15", "-0." + "0" * 36 + "25"]
FLOAT_EDGES = ["2e-38", "-2e-38", "0.0691026858985424", "-0.0691026858985424", "9007199254740993",
               "9007199254740995", "-9007199254740993", "18014398509481986", "16777217", "16777219", "-16777217",
               "1.000000059604644775390625", "1.000000178813934326171875", "0.1", "0.2", "0.3", "1e37", "3.4e37",
               "123456789.123456789", "0.00000000000000000000000000000000000117"]


def f32_midpoint_witnesses(count: int) -> list[str]:
    """Decimals just either side of f32 midpoints, closer to them than half an f64 unit.

    An f32 midpoint of the binade `[2^e, 2^(e+1))` is `(2m+1)·2^(e-24)`. Rounding
    a value just beside it to f64 lands exactly on the midpoint, whose tie then
    goes to the even neighbour, so a conversion through f64 disagrees with the
    direct rounding on every other `m`.
    """
    offsets = ((24, ref.Fraction(1, 10**20)), (0, ref.Fraction(1, 10**37)), (-10, ref.Fraction(1, 10**37)),
               (40, ref.Fraction(1, 10**5)), (100, ref.Fraction(1)), (125, ref.Fraction(1)))
    out = []
    for e, delta in offsets:
        for k in range(count):
            midpoint = (2 * (2**23 + k) + 1) * ref.Fraction(2) ** (e - 24)
            for value in (midpoint - delta, midpoint + delta):
                x = ref.canonical(value)
                if x is not None:
                    out.append(ref.decimal_to_string(x))
    return out


ACCEPTED_TEXTS = [
    "0", "-0", "0.0", "-0.000e-99", "0e999999999999999999999", "1", "-1", "1.50", "1.000", "100e-2", "0.10e1",
    "12e-3", "1E5", "1e+5", "1e05", "1e-0005", "123.456", "-0.5", "1e37", MAX_TEXT, "-" + MAX_TEXT, TINY_TEXT,
    "1e-38", "10e-39", "0." + NINES, "1." + "0" * 60, NINES + ".0", "1" + "0" * 39 + "e-2", "-0E+0", "0e-0",
    "1." + "0" * 998, "0e" + "0" * 998, "1e-" + "0" * 996 + "5", "0.0e" + "9" * 990,
]
MALFORMED_TEXTS = [
    "", "-", "+1", " 1", "1 ", "1\n", "01", "-01", "00", ".5", "5.", "1.e5", "1e", "1e+", "1e-", "--1", "1_000",
    "0x10", "Infinity", "-Infinity", "NaN", "inf", "1,5", "1.5.5", "1e5e5", "1e5.5", "e5", "1ee5", "١",
    "１", "1 ", "−1", "1d", "1.5f", "+0", "0.", "-.5", "1e+-5", "1\t", 'x"y', "a\\b",
    "0123456789012345678901234567890123456789012345", "1." + "0" * 998 + "x", "-" * 1001,
    # The bound and the 40-character cut count Unicode scalar values, not UTF-8 bytes.
    "\u0661" * 1000, "\u00e9" * 501, "\u00e9" * 45, "1" + "\u4e00" * 50, "\U0001f600" * 41,
]
OUTSIDE_TEXTS = [
    "1e38", "-1e38", "1e-39", "-1e-39", "1" + "0" * 38, "0." + "0" * 38 + "1",
    "1.234567890123456789012345678901234567891", "1e100000000000000000000", "1e-100000000000000000000",
    "12345678901234567890.1234567890123456789", "1.5e-38", NINES + ".5", "1e" + "9" * 990, "-1e-" + "9" * 990,
]
TOO_LONG_TEXTS = ["1." + "0" * 999, "0e" + "0" * 999, "1" * 1001, "-" + "1" * 1000, "0." + "0" * 1500,
                  "\u0661" * 1001]

FLOATS = [
    0.0, -0.0, 0.1, -0.1, 0.5, 2.5, -2.5, 0.125, -0.125, 1 / 3, -2 / 3, 1e-300, 5e-324, -5e-324, 1e38, -1e38,
    1e39, 2.0**127, 1.7976931348623157e308, 1.1, 123456.789, 2.0**53, 2.0**53 + 2, 1e-38, 9.5, 1e22, 1e23,
    0.30000000000000004, math.nan, math.inf, -math.inf,
]
SCALES = [0, 1, 2, 3, 17, 18, 37, 38]
BAD_SCALES = [-1, 39, I64_MIN, I64_MAX]
# Long divisions `dividend * 10^scale / divisor`, as (dividend, divisor, scale), on which
# Std.Decimal's long division needs its second quotient-digit correction
# (QUOTIENT_CORRECTION_WITNESSES) or clamps its digit estimate to 10^9 - 1
# (QUOTIENT_CLAMP_WITNESSES). Random division reaches neither branch.
QUOTIENT_CORRECTION_WITNESSES = [
    ("999999999999999999999999999", "682348825961024742", 18),
    ("9438469427994449664758872201810", "690999099990999099090999999", 37),
    ("99999999999999999999", "500909990909990090999009090", 30),
    ("59909909999000099900990999", "611003572960301859", 28),
    ("309999999099099999090090009990000", "649962543979608040871492331", 29),
    ("683223864990997998278665867", "590000009909909999090999909", 19),
    ("4990009090000900999", "6094481534565536523", 15),
    ("62340295459640712430325460456532030170", "572517726903781899936616638343141961", 19),
    ("9999999999999999999999999999999999", "1090090999900099990", 28),
    ("461670743713552483455806339", "599999000900090900009900090", 11),
    ("2965883462933767", "609909900909009000", 34),
    ("999999999999999999999", "5099000090099009999", 16),
    ("8009090999090090090909009990", "540033413817343991", 10),
    ("5099999000099009099099000090", "2099900999", 16),
    ("29181665513482841658228519", "573822655955942579802632894", 36),
    ("7416534480261130484288135337605394", "609990090999990099900000900", 20),
    ("500090009999909990900900000909990", "800009990990990009", 35),
    ("84061769328477672051975126828126138", "2004005362987501365248272630", 34),
    ("44093691827667005116900962993951728205", "509990009909099999", 3),
    ("9999999999999999999999999", "500099000900009009000090999099090900", 29),
    ("66250415854599858767502722909152", "528181590953901013131420507", 30),
    ("4450500998330044927829938851943973980", "587863420904585179568250296", 6),
    ("999999999999999999999999999999", "1338538625958496816309204082", 32),
    ("7117414460708282", "1037783547901736391", 38),
    ("901951447227190155996668", "513708694987166129446355759", 33),
    ("99999999999999999999999999999", "609909009909999900900909999", 38),
    ("99999999999999999999999999999", "599909099900099000", 22),
    ("89909000090099909900099090090990000909", "700090009909909990009909009999000099", 12),
    ("2616616528959786218879524459460624", "552544844803439644", 29),
    ("999999999999999999999999999999", "2023849214", 21),
    ("867099983", "200099990990009009", 37),
    ("9999999999999999999999999999999999", "909990099990999990", 31),
    ("99999999999999999999999999999999", "509999900999909990909999099", 15),
    ("18571921221623668381810864", "1090009999999099090009999090", 24),
    ("999999999999999999999999999999999", "512170171880118330053065238106805469", 21),
    ("28254079003023414588028506326486773431", "509437501705392253", 30),
    ("9999999999999999999999999999999", "709999090990009999", 33),
    ("99999999999999999", "543546777747143833194261400616178141", 33),
    ("99999999999999999999999999999", "543585995705833117", 35),
    ("99999999999999999999999999999999", "9900999909999009009999990099000090", 14),
    ("26359166224795901719", "848048180996605431", 26),
    ("999999999", "1001677301833329194854111150790878814", 38),
    ("409965328993863462038178", "500999099909009009", 20),
    ("717830224", "599090999909999909000900990", 38),
    ("3213894835931448521423076897341478930", "538168845979784162", 9),
    ("9208113", "615878968975002894", 28),
    ("999999999", "589085667852113309951532373", 36),
    ("511746013044492239554597571187107175", "400909990999990009", 7),
    ("43745403355541479020187353858685", "527709169867244915", 7),
    ("246505824839472290210673087079524303", "2090999090000000090099909090909099090", 29),
    ("4469992490646160837513015608", "600909999909909000", 35),
    ("7909099909909900909990990090909090", "509909999990999009009900009000909000", 24),
    ("999999999999999999999", "799990000909990999000099999", 32),
    ("9999999999999999999999", "1205796125688576409444302215189371184", 37),
    ("99999999999999999999999", "8099099099", 15),
    ("999999999999999999999", "600099999900909090009900009", 34),
    ("309099000000090000909900990009", "590090009999000009", 21),
    ("99999999", "525709862913607983727355670", 33),
    ("709099000099909990000000909000", "1000009009999090999", 7),
    ("9999999999999999999999", "522618471801071207608168780", 37),
]
QUOTIENT_CLAMP_WITNESSES = [
    ("9999999999999999999999999", "99999999999999999999999999", 19),
    ("999999999999999999999999999", "99999999999999999999999999999", 15),
    ("99999999999999999", "999999999999999999999999", 24),
    ("9999999999999999999999999999", "999999999999999999999999999999", 26),
    ("999999999999", "9999999999999", 28),
    ("999999999999999", "99999999999999999999999999999999999999", 33),
    ("9999999999999999999999999", "99999999999999999999999999", 20),
    ("999999999999999999", "99999999999999999999999", 24),
    ("999999999999999999999", "9999999999999999999999999999999", 32),
    ("999999999999999999", "99999999999999999999999999999", 28),
    ("9999999999999999999999999", "9999999999999999999999999999999999999", 34),
    ("999999999999999999", "9999999999999999999999", 33),
    ("999999999999999999999", "99999999999999999999999999", 26),
    ("999999999999999999", "9999999999999999999999999", 16),
    ("99999999999999999999999999999999", "999999999999999999999999999999999", 23),
    ("99999999999999999", "99999999999999999999999999999999999", 36),
    ("99999999999999999", "9999999999999999999999999", 28),
    ("999999999999999999999999", "9999999999999999999999999999999", 30),
    ("999999999999999999999", "999999999999999999999999", 15),
    ("99999999999999999999", "9999999999999999999999999999", 20),
    ("999999999999999999999999999", "9999999999999999999999999999999999", 26),
    ("999999999999999999999999", "99999999999999999999999999999999", 35),
    ("99999999999999999999999999999999999999", "9999999999999", 21),
    ("99999999999", "99999999999999999999", 28),
    ("999999999999999999999999999999999999", "99999999999999999999", 38),
    ("99999999999999999999999999", "99999999999999", 27),
    ("999999999999999999999999999999999999", "99999999999999999999999", 38),
    ("99999999999999999999999999", "99999999999999", 23),
    ("99999999999999999999999999", "9999999999999999999999999999", 26),
    ("99999999999999", "99999999999999999999999", 38),
]


def random_float(rng: random.Random) -> float:
    exponent = rng.randint(-140, 140)
    return rng.choice((1, -1)) * rng.random() * 2.0**exponent


def build_corpus(large: bool = False) -> Corpus:
    """The default (nightly CI) corpus, or with `large` the manual gate's.

    The large corpus crosses every input with every scale and rounding mode and
    observes every twin. To keep the default run short, the default corpus
    observes a twin on every fourth input the reference accepts (its `try_`
    form still sees every input), and gives seeded random inputs of rounding
    and float conversion one mode per scale in rotation instead of all seven;
    the fixed edge inputs keep the full cross product.
    """
    rng = random.Random(SEED)
    corpus = Corpus(twin_stride=1 if large else 4)
    scale = 25 if large else 1

    def modes_for(k: int, j: int, edge: bool) -> Sequence[str]:
        """Every mode for an edge input or in the large corpus; one in rotation otherwise."""
        return ROUNDINGS if large or edge else (ROUNDINGS[(k + j) % len(ROUNDINGS)],)

    pools = {
        "envelope": ENVELOPE, "limbs": LIMBS, "removable_zeros": REMOVABLE, "i64_edges": I64_EDGES, "ties": TIES,
        "float_edges": FLOAT_EDGES + f32_midpoint_witnesses(8 * scale),
        "random": [random_text(rng) for _ in range(200 * scale)],
    }
    for category, texts in pools.items():
        for text in texts:
            corpus.value(category, text)

    # Parser: accepted, malformed, outside the value set, longer than the bound,
    # and random values spelled with exponents and padding.
    for category, texts in (("parse_accepted", ACCEPTED_TEXTS), ("parse_malformed", MALFORMED_TEXTS),
                            ("parse_outside", OUTSIDE_TEXTS), ("parse_too_long", TOO_LONG_TEXTS)):
        for text in texts:
            corpus.parse(category, text)
    for _ in range(150 * scale):
        corpus.parse("parse_spelled", spelled(rng, random_text(rng)))
    for _ in range(50 * scale):
        digits = rng.randint(39, 45)
        corpus.parse("parse_outside", str(rng.randrange(10 ** (digits - 1), 10**digits)) + "e-" + str(rng.randint(0, 45)))

    # Fixed-scale rendering.
    fixed_inputs = ENVELOPE + REMOVABLE + TIES + pools["random"][: 40 * scale]
    for text in fixed_inputs:
        x = d(text)
        for n in sorted({x.scale, min(x.scale + 1, 38), 38, 0, max(x.scale - 1, 0)}):
            corpus.fixed("fixed", text, n)
    for n in BAD_SCALES:
        corpus.fixed("fixed_bad_scale", "1.5", n)

    # Integers.
    ints = [0, 1, -1, I64_MAX, I64_MIN, I64_MAX - 1, I64_MIN + 1, 10**18, -(10**18), 999999999, 10**9, -(10**9),
            10**18 - 1, 1000]
    ints += [rng.randint(I64_MIN, I64_MAX) for _ in range(50 * scale)]
    for v in ints:
        corpus.from_i64("from_i64", v)
    to_i64_inputs = I64_EDGES + TIES + ["0", "-0.9", "0.9", "1e30", "-1e30", MAX_TEXT, TINY_TEXT, "-" + TINY_TEXT]
    to_i64_inputs += [random_text(rng, 20) for _ in range(30 * scale)]
    for text in to_i64_inputs:
        for mode in ROUNDINGS:
            corpus.to_i64("to_i64", text, mode)

    # Float to decimal.
    floats = FLOATS + [random_float(rng) for _ in range(20 * scale)]
    for k, x in enumerate(floats):
        for j, n in enumerate(SCALES):
            for mode in modes_for(k, j, k < len(FLOATS)):
                corpus.from_f64("from_f64", x, n, mode)
    for x in (0.5, math.nan, -math.inf):
        for n in BAD_SCALES:
            corpus.from_f64("from_f64_bad_scale", x, n, "RoundTiesToEven")

    # Arithmetic.
    pairs = [
        (MAX_TEXT, "1"), ("-" + MAX_TEXT, "-1"), (MAX_TEXT, "-1"), ("1e37", "10"), ("1e-19", "1e-19"),
        ("1e-20", "1e-19"), ("2.5", "4"), ("0.25", "0.75"), ("1e37", "0.1"), ("1.5", TINY_TEXT), ("0." + NINES, TINY_TEXT),
        ("9" * 36, "1"), ("1e36", "-1"), (MAX_TEXT, MAX_TEXT), ("-" + MAX_TEXT, MAX_TEXT), ("0", "0"),
        ("3.14", "-3.14"), ("99999999999999999999", "99999999999999999999"), ("9999999999999999999", "9999999999999999999"),
    ]
    pairs += [("999999999" + "0" * k, "1" + "0" * k) for k in range(0, 30, 3)]
    pairs += [(random_text(rng), random_text(rng)) for _ in range(150 * scale)]
    pairs += [(random_text(rng, 19), random_text(rng, 19)) for _ in range(100 * scale)]
    for a, c in pairs:
        for op in ("add", "sub", "mul"):
            corpus.arith("arith", op, a, c)

    # Rounding to a scale.
    round_edges = TIES + ENVELOPE + I64_EDGES[:4]
    round_inputs = round_edges + pools["random"][: 30 * scale]
    for k, text in enumerate(round_inputs):
        for j, n in enumerate((0, 1, 2, 5, 18, 37, 38)):
            for mode in modes_for(k, j, k < len(round_edges)):
                corpus.round("round", text, n, mode)
    for n in BAD_SCALES:
        corpus.round("round_bad_scale", "2.5", n, "RoundTiesToEven")

    # Division.
    div_pairs = [("1", "3"), ("2", "3"), ("1", "8"), ("-1", "8"), ("1", "7"), ("10", "3"), ("-10", "3"), ("1e37", TINY_TEXT),
                 ("1", "0"), ("0", "7"), ("-5", "2"), ("5", "-2"), (MAX_TEXT, "0.5"), (TINY_TEXT, "2"), ("1e37", "3e-38"),
                 ("7", "7"), (MAX_TEXT, MAX_TEXT), (TINY_TEXT, MAX_TEXT)]
    for a, c in div_pairs:
        for n in (0, 1, 2, 3, 38):
            for mode in ROUNDINGS:
                corpus.div("div", a, c, n, mode)
    for _ in range(300 * scale):
        corpus.div("div_random", random_text(rng), random_text(rng, rng.choice((3, 12, 38))), rng.choice(SCALES),
                   rng.choice(ROUNDINGS))
    exact_modes = [mode for mode in ROUNDINGS if mode != "RejectInexact"]
    for category, witnesses in (("div_quotient_correction", QUOTIENT_CORRECTION_WITNESSES),
                                ("div_quotient_clamp", QUOTIENT_CLAMP_WITNESSES)):
        for k, (a, c, n) in enumerate(witnesses):
            corpus.div(category, "-" + a if k % 2 else a, c, n, exact_modes[k % len(exact_modes)])
    for n in BAD_SCALES:
        corpus.div("div_bad_scale", "1", "3", n, "RoundTiesToEven")
        corpus.div("div_bad_scale", "1", "0", n, "RoundTiesToEven")

    # Several failing checks at once: each argument's own validity left to right,
    # then RejectInexact, then the result's range. Each case is its own sampled path.
    corpus.to_i64("failure_order", "9223372036854775808.5", "RejectInexact")
    corpus.to_i64("failure_order", "-100000000000000000000.5", "RejectInexact")
    corpus.div("failure_order", "5", "0", 40, "RejectInexact")
    corpus.div("failure_order", "5", "0", I64_MIN, "RoundTiesToEven")
    corpus.div("failure_order", "1e37", "3e-38", 0, "RejectInexact")
    corpus.from_f64("failure_order", math.inf, 40, "RejectInexact")
    corpus.from_f64("failure_order", math.nan, -1, "RoundTiesToEven")
    corpus.from_f64("failure_order", 1.1, 38, "RejectInexact")
    corpus.round("failure_order", "2.5", 39, "RejectInexact")
    corpus.fixed("failure_order", "1.25", -1)

    # Order and equality.
    order_pairs = [("1.5", "1.50"), ("-0", "0"), ("-1", "1"), (TINY_TEXT, "0"), ("-" + TINY_TEXT, "0"),
                   (MAX_TEXT, "-" + MAX_TEXT), ("0.1", "0.10000000000000000000000000000000000001"),
                   ("100", "1e2"), ("2", "10"), ("-2", "-10")]
    order_pairs += [(random_text(rng), random_text(rng)) for _ in range(100 * scale)]
    texts = pools["random"]
    order_pairs += [(t, t) for t in texts[: 10 * scale]]
    for a, c in order_pairs:
        corpus.order("order", a, c)
    return corpus


def build_canary_corpus() -> Corpus:
    """The canary corpus: a fixed hand-picked handful of the default corpus's edge inputs.

    It runs every row, so it calls every callable and observes every twin, in
    one program with no failure programs and no seeded random inputs. It
    detects a defect only where one of its expected outputs changes; the
    default corpus is the complete check. Each defect class below changes at
    least one of them, which `scripts/test_decimal_differential.py` checks
    against models of the class that perturb the reference as a module with the
    defect would compute:

    - a coefficient or scale bound off by one
    - an inner limb rendered without its leading zeros
    - a limb carry dropped in addition
    - a limb borrow dropped in subtraction
    - a limb product's carry dropped in multiplication
    - a directed rounding mode rounding as another directed mode
    - a RoundTiesToEven tie broken toward zero, away from zero, toward positive or toward negative
    - a RoundTiesToAway tie broken to even, toward zero, toward positive or toward negative
    - a RejectInexact call accepting an inexact value
    - a binary conversion tie broken toward zero, away from zero, toward positive or toward negative
    - an f32 conversion rounding through f64
    - a long division taking one quotient-digit correction or none

    The binary conversions are `decimal_to_f64` and `decimal_to_f32`. The
    rounding classes hold for every callable that takes a mode, except that a
    `RejectInexact` rounding of an inexact value is a failure, which the canary
    does not run: it meets that of `decimal_to_i64`, `decimal_from_f64` and
    `decimal_div` only through their `try_` twins, and that of `decimal_round`,
    which has no twin, not at all.
    """
    corpus = Corpus(twin_stride=1)
    limbs = [str(10 ** (9 * j) + delta) for j in (1, 2, 4) for delta in (-1, 0)]
    for category, texts in (("envelope", [MAX_TEXT, "-" + MAX_TEXT, TINY_TEXT, "-" + TINY_TEXT, "0." + NINES]),
                            ("limbs", limbs), ("removable_zeros", ["-0.000e-5"]),
                            ("float_edges", ["9007199254740993", "-9007199254740993", "16777217", "-16777217",
                                             *f32_midpoint_witnesses(1)[:2]])):
        for text in texts:
            corpus.value(category, text)
    # Halfway cases of f64 and of f32 whose even neighbour is the larger.
    for text in ("9007199254740995", "16777219"):
        corpus.add("float_edges", "row_floats", (text,))
    for category, texts in (("parse_accepted", ["1e-0005"]), ("parse_malformed", ["+1", "1e"]),
                            ("parse_outside", ["1e38", "1e-39"])):
        for text in texts:
            corpus.parse(category, text)
    for text, n in (("1.25", 2), (MAX_TEXT, 0), (TINY_TEXT, 38)):
        corpus.fixed("fixed", text, n)
    for v in (I64_MAX, I64_MIN):
        corpus.from_i64("from_i64", v)
    for sign in (1, -1):
        for mode in ROUNDINGS:
            corpus.to_i64("to_i64", ("-" if sign < 0 else "") + "2.5", mode)
            corpus.from_f64("from_f64", sign * 2.5, 0, mode)
            corpus.round("round", ("-" if sign < 0 else "") + "4999999999999999999999999999999999999.5", 0, mode)
            corpus.div("div", str(5 * sign), "2", 0, mode)
        corpus.to_i64("to_i64", ("-" if sign < 0 else "") + "1.5", "RoundTiesToEven")
        corpus.from_f64("from_f64", sign * 1.5, 0, "RoundTiesToEven")
        corpus.div("div", str(3 * sign), "2", 0, "RoundTiesToEven")
    for text, mode in (("9223372036854775807.5", "RoundTowardZero"), ("-9223372036854775808.5", "RoundTiesToEven"),
                       (TINY_TEXT, "RoundAwayFromZero")):
        corpus.to_i64("to_i64", text, mode)
    for x, n in ((0.1, 38), (5e-324, 38), (math.nan, 2)):
        corpus.from_f64("from_f64", x, n, "RoundTiesToEven")
    for mode in ("RoundTiesToEven", "RoundTiesToAway"):
        corpus.round("round", "-0." + "0" * 36 + "25", 37, mode)
    for a, c in [("999999999" + "0" * k, "1" + "0" * k) for k in (0, 18)] + [
            (MAX_TEXT, "-1"), ("99999999999999999999", "99999999999999999999"), ("1e-19", "1e-19"),
            ("0." + NINES, TINY_TEXT)]:
        for op in ("add", "sub", "mul"):
            corpus.arith("arith", op, a, c)
    # Limb products that carry, and a subtraction that borrows across limbs.
    corpus.arith("arith", "mul", "999999999999999999", "999999999999999999")
    corpus.arith("arith", "sub", "1000000000000000000", "1")
    for a, c, n in (("1", "3", 38), (MAX_TEXT, "0.5", 0), (TINY_TEXT, MAX_TEXT, 38)):
        corpus.div("div", a, c, n, "RoundTiesToEven")
    for category, witnesses in (("div_quotient_correction", QUOTIENT_CORRECTION_WITNESSES),
                                ("div_quotient_clamp", QUOTIENT_CLAMP_WITNESSES)):
        a, c, n = witnesses[0]
        corpus.div(category, a, c, n, "RoundTiesToEven")
    for a, c in (("1.5", "1.50"), ("-0", "0"), (MAX_TEXT, "-" + MAX_TEXT), ("2", "10")):
        corpus.order("order", a, c)
    return corpus


# ---------------------------------------------------------------------------
# Programs.


@dataclass(frozen=True)
class Binding:
    """A printed root and the cases whose expected elements it lists, in order."""

    name: str
    row: str
    cases: tuple[Case, ...]

    @property
    def expected(self) -> list[Element]:
        return [element for case in self.cases for element in case.expected]


@dataclass(frozen=True)
class Program:
    """One run: the program `image` names, reading `inputs` as its case file.

    Every run of an image has the same source, so the C lane builds an image
    once and runs the binary once per run.
    """

    name: str
    image: str
    source: str
    inputs: str
    bindings: tuple[Binding, ...] = ()
    failure: Failure | None = None


def representative_failures(failures: list[Failure], per_path: int) -> list[Failure]:
    """Up to `per_path` evenly spaced failures per category, function, kind, and reason."""
    groups: dict[tuple[str, str, str, str], list[Failure]] = {}
    for failure in failures:
        key = (failure.category, failure.function, failure.error.kind, failure.error.reason)
        groups.setdefault(key, []).append(failure)
    chosen = []
    for members in groups.values():
        if len(members) <= per_path:
            chosen += members
        else:
            step = (len(members) - 1) / (per_path - 1) if per_path > 1 else 0
            chosen += [members[round(k * step)] for k in range(per_path)]
    return sorted(chosen, key=lambda f: f.name)


def make_programs(corpus: Corpus, chunk: int, failures_per_path: int, combined: bool = False) -> list[Program]:
    """A run per `chunk` cases of a row, and per sampled failure.

    With `combined`, one program runs every case of every row instead, so the
    C lane builds once however many rows there are.
    """
    programs = []
    by_row: dict[str, list[Case]] = {}
    for case in corpus.cases:
        by_row.setdefault(case.row, []).append(case)
    if not combined:
        for row_name, cases in by_row.items():
            source = grid_source(row_name)
            params = ROWS[row_name].params
            for k in range(0, len(cases), chunk):
                group = tuple(cases[k:k + chunk])
                inputs = "\n".join(line for case in group for line in case_lines(params, case.args))
                programs.append(Program(f"{row_name}_{k // chunk:04d}", row_name, source, inputs,
                                        (Binding(GRID_ROOT, row_name, group),)))
    elif by_row:
        blocks: list[tuple[str, int, int]] = []
        lines: list[str] = []
        bindings = []
        for row_name, cases in by_row.items():
            blocks.append((row_name, len(lines), len(cases)))
            lines += [line for case in cases for line in case_lines(ROWS[row_name].params, case.args)]
            bindings.append(Binding(f"out_{row_name}", row_name, tuple(cases)))
        programs.append(Program("rows", "rows", combined_grid_source(blocks), "\n".join(lines), tuple(bindings)))
    sources: dict[str, str] = {}
    for failure in representative_failures(corpus.failures, failures_per_path):
        source = sources.setdefault(failure.function, failure_source(failure.function))
        inputs = "\n".join(case_lines(FAILURE_EXPRS[failure.function], failure.args))
        programs.append(Program(failure.name, f"fail_{failure.function}", source, inputs, failure=failure))
    return programs


# ---------------------------------------------------------------------------
# Running programs on the lanes.


@dataclass(frozen=True)
class Toolchain:
    compiler: str
    compile_flags: tuple[str, ...]
    link_flags: tuple[str, ...]


@dataclass
class LaneResult:
    lane: str
    status: int | None
    stdout: str
    stderr: str
    stage: str
    seconds: float = 0.0


def printed_compile_command(build_stdout: str) -> list[str] | None:
    """The `Compile: ...` command `chelis build --emit-c` prints, retargeted to `out/case`."""
    for line in build_stdout.splitlines():
        if line.startswith("Compile: "):
            argv = shlex.split(line[len("Compile: "):])
            if "-o" in argv:
                at = argv.index("-o")
                del argv[at:at + 2]
            return [*argv, "-o", "out/case"]
    return None


def std_version() -> str:
    manifest = tomllib.loads((REPO / "packages" / "chelis-std" / "reef.toml").read_text(encoding="utf-8"))
    return manifest["package"]["version"]


def publish_std(chelis: Path, work: Path) -> Path:
    """A fresh reef home with this checkout's `chelis-std` published into it."""
    home = work / "reef-home"
    package = work / "chelis-std"
    for path in (home, package):
        if path.exists():
            shutil.rmtree(path)
    home.mkdir(parents=True)
    shutil.copytree(REPO / "packages" / "chelis-std", package)
    env = dict(os.environ, CHELIS_STYLE_GATE_DISABLE="1", CHELIS_REEF_HOME=str(home))
    done = subprocess.run([str(chelis), "reef", "publish", str(package)], env=env, capture_output=True, text=True,
                          check=False)
    if done.returncode != 0:
        raise SystemExit(f"chelis reef publish {package} failed:\n{done.stdout}{done.stderr}")
    return home


class Runner:
    def __init__(self, chelis: Path, work: Path, timeout: int, reef_home: Path, std: str,
                 toolchain: Toolchain | None = None, keep_artifacts: bool = False, compiler_version: str | None = None) -> None:
        self.chelis = chelis
        self.work = work
        self.timeout = timeout
        self.reef_home = reef_home
        self.std_version = std
        self.toolchain = toolchain
        self.keep_artifacts = keep_artifacts
        if compiler_version is None:
            version = subprocess.run([str(chelis), "--version"], capture_output=True, text=True, check=True)
            compiler_version = version.stdout.split()[1]
        self.compiler_version = compiler_version
        # Each image's build outcome: its binary, or the lane result that stopped it.
        self.images: dict[str, tuple[Path | None, LaneResult | None]] = {}
        self.image_locks: dict[str, threading.Lock] = {}
        self.locks_guard = threading.Lock()

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update({"CHELIS_STYLE_GATE_DISABLE": "1", "OMP_NUM_THREADS": "1", "CHELIS_REEF_HOME": str(self.reef_home)})
        return env

    def app(self, app: Path, source: str) -> tuple[Path, str]:
        """A package at `app` depending on `chelis-std`, with `source` as its entry module."""
        if app.exists():
            shutil.rmtree(app)
        (app / "src").mkdir(parents=True)
        (app / "reef.toml").write_text(
            'schema = "1"\n\n[package]\nname = "decimal-oracle"\nversion = "0.1.0"\n'
            f'compiler = "={self.compiler_version}"\nmodule_prefix = "Demo"\n\n'
            f'[dependencies]\nchelis-std = {{ version = "{self.std_version}" }}\n', encoding="utf-8")
        (app / "src" / "main.ch").write_text(source, encoding="utf-8")
        return app, "src/main.ch"

    def run(self, argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, cwd=cwd, env=self.env(), capture_output=True, text=True, timeout=self.timeout,
                              check=False)

    def eval_lane(self, program: Program) -> LaneResult:
        app, main = self.app(self.work / "eval" / program.name, program.source)
        (app / CASES_FILE).write_text(program.inputs, encoding="utf-8")
        done = self.run([str(self.chelis), "eval", "--file", main], app)
        return LaneResult("eval", done.returncode, done.stdout, done.stderr, "eval")

    def image(self, program: Program) -> tuple[Path | None, LaneResult | None]:
        """The program's image built once, however many runs ask for it at once."""
        with self.locks_guard:
            lock = self.image_locks.setdefault(program.image, threading.Lock())
        with lock:
            if program.image not in self.images:
                try:
                    self.images[program.image] = self.build_image(program)
                except subprocess.TimeoutExpired as error:
                    stopped = LaneResult("c", None, "", f"timed out after {error.timeout} s", "timeout")
                    self.images[program.image] = (None, stopped)
            return self.images[program.image]

    def build_image(self, program: Program) -> tuple[Path | None, LaneResult | None]:
        app, main = self.app(self.work / "c" / program.image, program.source)
        build = self.run([str(self.chelis), "build", main, "--target", "c", "--emit-c", "--output", "out"], app)
        if build.returncode != 0:
            return None, LaneResult("c", build.returncode, build.stdout, build.stderr, "build")
        stem = Path(main).stem
        if self.toolchain is not None:
            argv = [self.toolchain.compiler, *self.toolchain.compile_flags, "-Iout", f"out/{stem}.c",
                    "out/libchelis_runtime.a", *self.toolchain.link_flags, "-o", "out/case"]
        else:
            argv = printed_compile_command(build.stdout)
            if argv is None:
                return None, LaneResult("c", None, build.stdout, "chelis build printed no Compile: command", "build")
        link = self.run(argv, app)
        if link.returncode != 0:
            return None, LaneResult("c", link.returncode, link.stdout, link.stderr, "link")
        if not self.keep_artifacts:
            # Each build carries its own runtime archive (about 30 MB); the
            # binary and the program source stay for the runs and reproduction.
            for entry in (app / "out").iterdir():
                if entry.name != "case":
                    shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        return app / "out" / "case", None

    def c_lane(self, program: Program) -> LaneResult:
        binary, stopped = self.image(program)
        if stopped is not None:
            return stopped
        run_dir = self.work / "c-runs" / program.name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / CASES_FILE).write_text(program.inputs, encoding="utf-8")
        done = self.run([str(binary)], run_dir)
        return LaneResult("c", done.returncode, done.stdout, done.stderr, "run")

    def lane(self, name: str, program: Program) -> LaneResult:
        return self.eval_lane(program) if name == "eval" else self.c_lane(program)


# ---------------------------------------------------------------------------
# Comparison.


def parse_bindings(stdout: str) -> tuple[dict[str, str], list[str]]:
    """Printed bindings by name, and every line that is not a `name = value` binding."""
    found: dict[str, str] = {}
    stray = []
    for line in stdout.splitlines():
        name, sep, value = line.partition(" = ")
        if sep and re.fullmatch(r"[a-z0-9_]+", name) and name not in found:
            found[name] = value
        else:
            stray.append(line)
    return found, stray


def split_elements(printed: str) -> list[str] | None:
    """The elements of a printed `List[string]` whose elements never contain `, `."""
    if not (printed.startswith("[") and printed.endswith("]")):
        return None
    inner = printed[1:-1]
    return inner.split(", ") if inner else []


def special_text(element: FloatBits) -> str | None:
    """The pinned spelling of a non-finite float (every NaN payload prints `NaN`), else None."""
    precision, exponent_bits = element.fmt
    fraction_bits = precision - 1
    if (element.bits >> fraction_bits) & (2**exponent_bits - 1) != 2**exponent_bits - 1:
        return None
    if element.bits & (2**fraction_bits - 1):
        return "NaN"
    return "-inf" if element.bits >> (exponent_bits + fraction_bits) else "inf"


def element_matches(expected: Element, printed: str) -> bool:
    if isinstance(expected, FloatBits):
        special = special_text(expected)
        if special is not None:
            return printed == special
        return ref.text_bits(printed, expected.fmt) == expected.bits
    return printed == expected


def compare_binding(binding: Binding, printed: str | None, limit: int = 5) -> list[str]:
    if printed is None:
        return [f"{binding.name}: no output line"]
    elements = split_elements(printed)
    expected = binding.expected
    if elements is None:
        return [f"{binding.name}: not a list: {printed[:300]}"]
    problems = []
    width = len(binding.cases[0].expected)
    for index, (want, got) in enumerate(zip(expected, elements)):
        if not element_matches(want, got):
            case = binding.cases[index // width]
            problems.append(f"{binding.name}[{index // width}] {case.category} {binding.row}{case.args!r}: "
                            f"element {index % width} expected {want}, printed {got}")
            if len(problems) >= limit:
                problems.append(f"{binding.name}: further mismatches not shown")
                break
    if len(elements) != len(expected):
        problems.append(f"{binding.name}: {len(expected)} elements expected, {len(elements)} printed")
    return problems


MESSAGE = re.compile(r"(?:error: )?([a-z0-9_]+): (domain|overflow): (.*)")


def failure_message(result: LaneResult) -> str | None:
    for line in result.stderr.splitlines():
        match = MESSAGE.fullmatch(line.strip())
        if match:
            return line.strip().removeprefix("error: ")
    return None


@dataclass
class Report:
    cases: int = 0
    elements: int = 0
    failures_checked: int = 0
    problems: list[str] = field(default_factory=list)
    classes: Counter = field(default_factory=Counter)

    def problem(self, kind: str, text: str) -> None:
        self.classes[kind] += 1
        self.problems.append(f"[{kind}] {text}")


def check_failure(failure: Failure, results: list[LaneResult], report: Report) -> None:
    messages = {}
    expr = f"\n    expression: {failure.expr[:400]}"
    for result in results:
        message = failure_message(result)
        if result.status == 0 or message is None:
            report.problem("failure-missing", f"{failure.name} [{result.lane}/{result.stage}]: expected "
                           f"`{failure.function}: {failure.error.kind}: ...`, got status {result.status}; stdout "
                           f"{result.stdout.strip()[:300]!r}; stderr {result.stderr.strip()[:300]!r}{expr}")
            continue
        messages[result.lane] = message
        function, kind, _ = MESSAGE.fullmatch(message).groups()
        if (function, kind) != (failure.function, failure.error.kind):
            report.problem("failure-kind", f"{failure.name} [{result.lane}]: expected `{failure.function}: "
                           f"{failure.error.kind}: ...`, got `{message[:300]}`{expr}")
        elif not failure.error.matches(message):
            report.problem("failure-detail", f"{failure.name} [{result.lane}]: expected `{failure.error.message[:300]}`, "
                           f"got `{message[:300]}`{expr}")
    if len(set(messages.values())) > 1:
        report.problem("lane-message", f"{failure.name}: lanes disagree on the message: {messages}")
    report.failures_checked += 1


def check_program(program: Program, results: list[LaneResult], report: Report) -> None:
    if program.failure is not None:
        check_failure(program.failure, results, report)
        return
    for result in results:
        if result.status != 0:
            report.problem("program", f"{program.name} [{result.lane}/{result.stage}]: status {result.status}; "
                           f"stderr {result.stderr.strip()[:2000]}")
            continue
        printed, stray = parse_bindings(result.stdout)
        expected_names = {binding.name for binding in program.bindings}
        stray += [f"{name} = {value[:200]}" for name, value in printed.items() if name not in expected_names]
        if stray:
            report.problem("stray-output", f"{program.name} [{result.lane}]: unexpected output lines: {stray[:10]}")
        for binding in program.bindings:
            for problem in compare_binding(binding, printed.get(binding.name)):
                report.problem("value", f"{program.name} [{result.lane}] {problem}")
    report.cases += sum(len(binding.cases) for binding in program.bindings)
    report.elements += sum(len(binding.expected) for binding in program.bindings)


def run_all(runner, programs: list[Program], lanes: Sequence[str], jobs: int, log: Callable[[str], None]) -> Report:
    report = Report()

    def run_one(program: Program) -> tuple[Program, list[LaneResult]]:
        results = []
        for lane in lanes:
            started = time.monotonic()
            try:
                result = runner.lane(lane, program)
            except subprocess.TimeoutExpired as error:
                result = LaneResult(lane, None, "", f"timed out after {error.timeout} s", "timeout")
            result.seconds = time.monotonic() - started
            results.append(result)
        return program, results

    done = 0
    lane_seconds: Counter = Counter()
    last = time.monotonic()
    # One run of every image goes first, so the C builds start together rather
    # than queueing behind the runs that wait for them.
    seen: set[str] = set()
    firsts = [p for p in programs if not (p.image in seen or seen.add(p.image))]
    ordered = firsts + [p for p in programs if p not in firsts]
    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        for program, results in pool.map(run_one, ordered):
            check_program(program, results, report)
            for result in results:
                lane_seconds[result.lane] += result.seconds
            done += 1
            if done == len(programs) or time.monotonic() - last > 60:
                last = time.monotonic()
                log(f"{done}/{len(programs)} programs, {len(report.problems)} problems")
    log("lane seconds: " + ", ".join(f"{lane} {lane_seconds[lane]:.0f}" for lane in lanes))
    return report


def summary(report: Report, lanes: Sequence[str], profile: str = "default") -> str:
    detail = (f"{report.cases} cases ({report.elements} elements) and {report.failures_checked} failure programs "
              f"on lanes {'+'.join(lanes)}; {profile} corpus")
    if report.problems:
        classes = ", ".join(f"{kind} {count}" for kind, count in sorted(report.classes.items()))
        return f"{FAIL_MARKER} ({len(report.problems)} problems: {classes}; {detail})"
    return f"{PASS_MARKER} ({detail})"


def resolve_chelis(flag: Path | None) -> Path:
    if flag is not None:
        return flag
    env = os.environ.get("CHELIS_BIN")
    if env:
        return Path(env)
    raise SystemExit("give the chelis binary under test with --chelis or CHELIS_BIN")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chelis", type=Path, help="the chelis binary under test (default: $CHELIS_BIN)")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--large", action="store_true", help="the manual gate's large seeded corpus")
    profile.add_argument("--canary", action="store_true", help="the small hand-picked corpus pull-request CI runs")
    parser.add_argument("--lanes", default="eval,c")
    parser.add_argument("--toolchain-json", help="C toolchain {compiler, compile_flags, link_flags}; default: the "
                        "command chelis build --emit-c prints")
    parser.add_argument("--reef-home", type=Path, help="a reef home with chelis-std published (default: publish this "
                        "checkout's packages/chelis-std into a fresh one)")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="parallel runs (default: the CPU count)")
    parser.add_argument("--chunk", type=int, default=100, help="cases per grid run")
    parser.add_argument("--failures-per-path", type=int, help="failure programs per category, function and kind")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per lane process")
    parser.add_argument("--work", type=Path, help="keep generated programs here instead of a temporary directory")
    parser.add_argument("--keep-artifacts", action="store_true", help="keep each C build's out/ directory")
    parser.add_argument("--list", action="store_true", help="print the corpus composition and exit")
    parser.add_argument("--only", help="diagnosis: run only programs whose name matches this regular expression")
    args = parser.parse_args(argv)

    lanes = args.lanes.split(",")
    if not lanes or any(lane not in BOTH for lane in lanes):
        parser.error("--lanes is a comma-separated subset of eval,c")
    profile = "large" if args.large else "canary" if args.canary else "default"
    corpus = build_canary_corpus() if args.canary else build_corpus(args.large)
    per_path = args.failures_per_path or {"large": 12, "canary": 0, "default": 3}[profile]
    programs = make_programs(corpus, args.chunk, per_path, combined=args.canary)
    if args.only:
        programs = [p for p in programs if re.search(args.only, p.name)]
    if args.list:
        for title, counts in corpus.composition().items():
            print(f"{title}:")
            for key, count in sorted(counts.items()):
                print(f"  {key}: {count}")
        print(f"{len(corpus.cases)} cases, {len(corpus.failures)} failures "
              f"({sum(1 for p in programs if p.failure)} sampled), {len(programs)} programs")
        return 0

    chelis = resolve_chelis(args.chelis)
    toolchain = None
    if args.toolchain_json:
        spec = json.loads(args.toolchain_json)
        toolchain = Toolchain(spec["compiler"], tuple(spec["compile_flags"]), tuple(spec["link_flags"]))
    def log(text: str) -> None:
        print(text, flush=True)

    log(f"{len(corpus.cases)} cases, {len(programs)} programs, lanes {'+'.join(lanes)}, binary {chelis}")
    with tempfile.TemporaryDirectory(prefix="decimal-oracle-") as scratch:
        work = args.work or Path(scratch)
        work.mkdir(parents=True, exist_ok=True)
        reef_home = args.reef_home or publish_std(chelis.resolve(), work.resolve())
        runner = Runner(chelis.resolve(), work.resolve(), args.timeout, reef_home.resolve(), std_version(), toolchain,
                        args.keep_artifacts)
        report = run_all(runner, programs, lanes, args.jobs, log)
    for problem in report.problems[:200]:
        print(problem)
    if len(report.problems) > 200:
        print(f"... {len(report.problems) - 200} more problems")
    print(summary(report, lanes, profile))
    return 1 if report.problems else 0


if __name__ == "__main__":
    sys.exit(main())
