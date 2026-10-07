#!/usr/bin/env python3
"""Vendor CORE-MATH kernels and generate the `chelis-crmath` amalgamation (chelis#2957).

`spec/design/correctly_rounded_math.md` section 3.3 owns the design. The eighteen
upstream kernel files under `crates/chelis-crmath/vendor/core-math/` are kept
byte-for-byte as upstream ships them; `VENDOR.toml` beside them records the
upstream commit, each file's SHA-256, and the file-scope identifiers of each
kernel, taken from Clang AST dumps when the kernels were imported. From those
inputs alone (no compiler needed) this script generates
`crates/chelis-crmath/csrc/crmath_amalgamation.c`, the one translation-unit text
every lane compiles:

1. every file-scope identifier and macro of a kernel gets the kernel's prefix
   (`chelis_cr_expf__`), so the eighteen kernels coexist in one unit;
2. the kernel's external entry (`cr_expf`) is declared `static` before its
   definition, so every definition has internal linkage;
3. a `static` entry `chelis_cr_<name>` calls the kernel and replaces any NaN
   result with [04-NUM-2]'s canonical quiet NaN;
4. the kernel text is reduced to the generated-C contract that built programs
   must satisfy, because `chelis build` emits these same bytes into every unit
   that calls a kernel (design section 4.2). `contract_clean` drops what the
   contract forbids and what Chelis never observes: the `errno` blocks, the
   floating-point exception raises and status-flag save and clear, `<fenv.h>`, the `FENV_ACCESS` pragma, the
   `noinline`/`cold` attributes, and the x86-64 SSE intrinsic arms (the
   portable arm beside each stays), and a `#define` between an `if` arm and its
   `else` moves to the top of its kernel. The dynamic rounding-mode switch keeps only
   its round-to-nearest case, which Chelis pins at every entry (design section
   6). Values are unchanged: every dropped arm computes the same bits as the
   arm that stays;
5. every kernel that rounds a reduced argument to an integer calls one
   `roundeven_finite` helper, `ROUNDEVEN_FINITE`, around `__builtin_rint`.
   `inline_roundeven` replaces upstream's definitions (the `__builtin_roundeven`
   macro and its fallback). `route_sin_roundeven` gives the helper to binary64
   `sin`, which calls the builtin directly, and routes that call through it.

The amalgamation opens with `#error` guards against fast math, finite-math-only,
and excess-precision evaluation.

Usage (from the repository root):

    .venv/bin/python scripts/vendor_core_math.py            # regenerate the amalgamation
    .venv/bin/python scripts/vendor_core_math.py --check    # fail on any drift
    .venv/bin/python scripts/vendor_core_math.py import --upstream DIR
        # copy kernels from a CORE-MATH checkout, re-derive identifiers with clang
    .venv/bin/python scripts/vendor_core_math.py fixtures --upstream DIR
        # regenerate the MPFR-derived test fixtures (needs gmpy2)
    .venv/bin/python scripts/vendor_core_math.py obligations [--check]
        # regenerate (or check) the profile obligation fixture (needs gmpy2)
    .venv/bin/python scripts/vendor_core_math.py resolve FILE
        # resolve the exhaustive gate's ambiguous inputs with MPFR (needs gmpy2)

Exit status: 0 success, 1 drift or verification failure, 2 usage or missing tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib
from fractions import Fraction
from dataclasses import dataclass
from pathlib import Path

try:  # optional: only the `fixtures` and `resolve` maintainer subcommands need MPFR
    import gmpy2
except ImportError:  # pragma: no cover - depends on the maintainer's environment
    gmpy2 = None

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "crates" / "chelis-crmath"
VENDOR = CRATE / "vendor" / "core-math"
MANIFEST = VENDOR / "VENDOR.toml"
AMALGAMATION = CRATE / "csrc" / "crmath_amalgamation.c"
FIXTURES = CRATE / "tests" / "fixtures"
CANARY_FIXTURE = FIXTURES / "canary.txt"
WORST_CASE_FIXTURE = FIXTURES / "binary64_worst_cases.txt"
UPSTREAM_URL = "https://gitlab.inria.fr/core-math/core-math"

# The [05-OP-46] transcendentals at both widths. `sqrt` needs no kernel: IEEE 754
# makes the hardware square root correctly rounded.
FUNCTIONS = ("exp", "log", "sin", "cos", "tan", "atan", "tanh", "erf", "erfc")


@dataclass(frozen=True)
class Kernel:
    """One upstream kernel: `name` is the C name (`expf`), `function` the math name."""

    name: str
    function: str
    width: int  # 32 or 64
    path: str  # relative to the upstream root and to VENDOR
    local_includes: tuple[str, ...] = ()

    @property
    def ctype(self) -> str:
        return "float" if self.width == 32 else "double"

    @property
    def prefix(self) -> str:
        return f"chelis_cr_{self.name}__"

    @property
    def upstream_entry(self) -> str:
        return f"cr_{self.name}"

    @property
    def entry(self) -> str:
        return f"chelis_cr_{self.name}"


def _kernels() -> tuple[Kernel, ...]:
    out = []
    for fn in FUNCTIONS:
        out.append(Kernel(f"{fn}f", fn, 32, f"src/binary32/{fn}/{fn}f.c"))
    for fn in FUNCTIONS:
        includes = ("dint.h",) if fn == "log" else ()
        out.append(Kernel(fn, fn, 64, f"src/binary64/{fn}/{fn}.c", includes))
    return tuple(out)


KERNELS = _kernels()
LICENSE_FILE = "LICENSE"


class VendorError(Exception):
    """A verification failure with a message for the maintainer."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def vendored_files(kernel: Kernel) -> list[str]:
    """Relative paths of a kernel's source and its local headers."""
    base = kernel.path.rsplit("/", 1)[0]
    return [kernel.path] + [f"{base}/{inc}" for inc in kernel.local_includes]


# --- C token rewriting -------------------------------------------------------

_TOKEN = re.compile(
    r"""
    (?P<comment>/\*.*?\*/|//[^\n]*)
  | (?P<string>"(?:\\.|[^"\\\n])*")
  | (?P<char>'(?:\\.|[^'\\\n])*')
  | (?P<number>\.?[0-9](?:[eEpP][+-]|[0-9A-Za-z_.])*)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE | re.DOTALL,
)
_INCLUDE_LINE = re.compile(r"^[ \t]*#[ \t]*include\b[^\n]*$", re.MULTILINE)
_LOCAL_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"[^\n]*$', re.MULTILINE)
_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def macro_names(text: str) -> set[str]:
    """Every macro a kernel text defines; they are file-scope names too."""
    return set(_DEFINE.findall(text))


def rename_identifiers(text: str, names: set[str], prefix: str) -> str:
    """Prefix every identifier token in `names`, leaving comments, literals, and
    `#include` lines untouched."""
    protected = [(m.start(), m.end()) for m in _INCLUDE_LINE.finditer(text)]
    out: list[str] = []
    pos = 0
    for m in _TOKEN.finditer(text):
        if m.lastgroup != "ident" or m.group() not in names:
            continue
        if any(a <= m.start() < b for a, b in protected):
            continue
        out.append(text[pos : m.start()])
        out.append(prefix + m.group())
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def inline_local_includes(text: str, kernel: Kernel, read) -> str:
    """Replace `#include "dint.h"`-style lines with the vendored header's text."""
    base = kernel.path.rsplit("/", 1)[0]

    def splice(m: re.Match[str]) -> str:
        header = m.group(1)
        if header not in kernel.local_includes:
            raise VendorError(f"{kernel.path}: undeclared local include {header!r}")
        body = read(f"{base}/{header}")
        return f"/* begin inlined {base}/{header} */\n{body}\n/* end inlined {base}/{header} */"

    return _LOCAL_INCLUDE.sub(splice, text)


# --- generated-C contract ------------------------------------------------------

_DIRECTIVE = re.compile(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|elif|else|endif)\b")

# Conditionals whose `#else` arm is the portable C one; the arm before it is the
# x86-64 SSE intrinsic or inline-assembly form, `errno` support (no `#else`), or
# `unsigned _BitInt(128)`, which the contract's C grammar does not parse and
# which is the same 128-bit unsigned arithmetic as the `unsigned __int128` arm.
_PORTABLE_ARM_CONDITIONALS = (
    "#if (defined(__clang__) && __clang_major__ >= 14) || (defined(__GNUC__) && __GNUC__ >= 14 && __BITINT_MAXWIDTH__ && __BITINT_MAXWIDTH__ >= 128)",
    "#if defined(__x86_64__)",
    "#ifdef __x86_64__",
    "#ifdef CORE_MATH_SUPPORT_ERRNO",
)

_ROUNDING_SWITCH = re.compile(
    r"([ \t]*)switch \(fegetround\(\)\) \{\n"
    r"[ \t]*case FE_TONEAREST:\n"
    r"(?P<body>(?:(?![ \t]*break;).*\n)+?)"
    r"[ \t]*break;\n"
    r"(?:[ \t]*case FE_(?:DOWNWARD|UPWARD|TOWARDZERO):\n(?:(?![ \t]*break;).*\n)*?[ \t]*break;\n)+"
    r"\1\}\n"
)
_ATTRIBUTE = re.compile(r"__attribute__\(\((?:cold|noinline)(?:,(?:cold|noinline))*\)\)[ \t]*")
_RAISE = re.compile(r"^[ \t]*feraiseexcept[ \t]*\([A-Z_]+\);[^\n]*\n", re.MULTILINE)
# binary64 erfc saves the underflow flag on entry and clears a spurious underflow
# before it returns; both only manage status flags, which Chelis never observes.
_FLAG_SAVE = re.compile(r"^[ \t]*int[ \t]+underflow[ \t]*=[ \t]*fetestexcept[ \t]*\(FE_UNDERFLOW\);[^\n]*\n", re.MULTILINE)
_FLAG_CLEAR = re.compile(
    r"^[ \t]*if \(underflow == 0 && [^\n]*fetestexcept \(FE_UNDERFLOW\)\)\n"
    r"[ \t]*feclearexcept \(FE_UNDERFLOW\);[^\n]*\n",
    re.MULTILINE,
)
_DROPPED_LINES = re.compile(
    r"^[ \t]*(?:#[ \t]*pragma[ \t]+STDC[ \t]+FENV_ACCESS[ \t]+ON|#[ \t]*include[ \t]*<fenv\.h>)[^\n]*\n",
    re.MULTILINE,
)
# What the reduced text may still include; `generated_header.rs` admits each.
CONTRACT_INCLUDES = {"<float.h>", "<inttypes.h>", "<stdint.h>", "<stdio.h>", "<string.h>"}
_FORBIDDEN_TOKENS = ("__attribute", "__attribute__", "__asm", "__asm__", "asm", "_Pragma", "__declspec", "fegetround",
                     "feraiseexcept", "fetestexcept", "feclearexcept", "errno", "_mm_setcsr")


def keep_portable_arms(text: str) -> str:
    """Replace every conditional in `_PORTABLE_ARM_CONDITIONALS` with its `#else`
    arm (or nothing when it has none)."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].rstrip() not in _PORTABLE_ARM_CONDITIONALS:
            out.append(lines[i])
            i += 1
            continue
        depth, j, else_at = 0, i + 1, None
        while True:
            if j >= len(lines):
                raise VendorError(f"unterminated conditional {lines[i]!r}")
            m = _DIRECTIVE.match(lines[j])
            kind = m.group(1) if m else None
            if kind in ("if", "ifdef", "ifndef"):
                depth += 1
            elif kind == "endif":
                if depth == 0:
                    break
                depth -= 1
            elif kind == "elif" and depth == 0:
                raise VendorError(f"unexpected #elif in {lines[i]!r}")
            elif kind == "else" and depth == 0:
                else_at = j
            j += 1
        if else_at is not None:
            out.extend(keep_portable_arms("\n".join(lines[else_at + 1 : j])).split("\n"))
        i = j + 1
    return "\n".join(out)


def hoist_defines_before_else(text: str, origin: str) -> str:
    """Move a `#define` that sits between an `if` arm and its `else` to the top of
    the kernel text. The C grammar the generated-C contract parses has no
    directive inside an `if`/`else` chain; the hoisted macro means the same
    because no token before its original line names it."""
    lines = text.split("\n")
    hoisted: list[str] = []
    out: list[str] = []
    for i, line in enumerate(lines):
        m = _DEFINE.match(line)
        if m:
            j = i + 1
            in_comment = False
            while j < len(lines):
                stripped = lines[j].strip()
                if in_comment:
                    in_comment = "*/" not in stripped
                elif stripped.startswith("/*"):
                    in_comment = "*/" not in stripped
                elif stripped and not stripped.startswith("//"):
                    break
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith("else"):
                name = m.group(1)
                if name in _code_tokens("\n".join(out)):
                    raise VendorError(f"{origin}: cannot hoist `{name}`, which is used before its definition")
                hoisted.append(line)
                continue
        out.append(line)
    if not hoisted:
        return text
    return "\n".join(hoisted + out)


def contract_clean(text: str, origin: str) -> str:
    """Reduce one kernel's text to the generated-C contract (module docstring, item 4)."""
    text = keep_portable_arms(text)
    text = hoist_defines_before_else(text, origin)
    text = _ROUNDING_SWITCH.sub(lambda m: _reindent(m.group("body"), m.group(1)), text)
    text = _RAISE.sub("", text)
    text = _FLAG_CLEAR.sub("", _FLAG_SAVE.sub("", text))
    text = _DROPPED_LINES.sub("", text)
    text = _ATTRIBUTE.sub("", text)
    code = _code_tokens(text)
    for token in _FORBIDDEN_TOKENS:
        if token in code:
            raise VendorError(f"{origin}: `{token}` survives the generated-C reduction")
    if any(token.startswith("FE_") for token in code):
        raise VendorError(f"{origin}: a floating-point environment macro survives")
    for m in _INCLUDE_LINE.finditer(text):
        target = re.sub(r"\s*//.*$", "", m.group()).split("include", 1)[1].strip()
        if target not in CONTRACT_INCLUDES:
            raise VendorError(f"{origin}: include {target} is outside the generated-C contract")
    return text


# Six kernels round a reduced argument to an integer, ties to even: binary32 `sin`,
# `cos`, and `tan`, and binary64 `exp` and `erfc` through an upstream
# `roundeven_finite` helper, and binary64 `sin` (`cr_sin_moderate`) through a direct
# `__builtin_roundeven` call. Upstream's helper is that builtin on GCC 10 and Clang 17
# and a `round`-based fallback elsewhere. On baseline x86-64, and on aarch64 with
# GCC 10, the builtin becomes a call to the C library's `roundeven`. musl and glibc
# before 2.25 do not have it. `chelis build` compiles every kernel into its compiler
# canary, so one such call stops every native build there. `inline_roundeven` and
# `route_sin_roundeven` give all six `ROUNDEVEN_FINITE` instead.
#
# `rint` rounds to an integer in the current rounding mode. Chelis pins
# round-to-nearest-even at every entry point (design section 6), so `rint` gives the
# value of `roundeven` for every finite input, the sign of zero included. AArch64
# compiles it to one `frintx`, and GCC inlines it on baseline x86-64. Clang on
# baseline x86-64 calls the C library's `rint`. Every C99 C library has it.
# The portable `copysign((|x| + 2^52) - 2^52, x)` gives the same values, but it makes
# most of these kernels about 20% slower on AArch64.
ROUNDEVEN_FINITE = """\
/* round x to nearest integer, breaking ties to even, in the round-to-nearest-even
   mode Chelis pins at every entry */
static double
roundeven_finite (double x)
{
  return __builtin_rint (x);
}
"""
_SIN_ROUNDEVEN_CALL = "__builtin_roundeven (invpi * ax)"
_FENV_PRAGMA = "#pragma STDC FENV_ACCESS ON\n"
_ROUNDEVEN_HELPER = re.compile(
    r"^/\* __builtin_roundeven was introduced in gcc 10:\n.*?^#endif\n", re.MULTILINE | re.DOTALL
)
_ROUNDEVEN_DEFINE = re.compile(r"#[ \t]*define[ \t]+roundeven_finite\(x\)[ \t]+__builtin_roundeven \(x\)")
# A `roundeven_finite` definition under any kernel prefix, however its signature is
# spaced and whatever its parameter list.
_ROUNDEVEN_DEFINITION = re.compile(r"(?<!\w)(\w*)roundeven_finite\s*\([^)]*\)\s*\{")
_ROUNDEVEN_MACRO = re.compile(r"^[ \t]*#[ \t]*define[ \t]+\w*roundeven_finite\b", re.MULTILINE)
# Integer rounding a kernel must not reach except through `ROUNDEVEN_FINITE`: the
# builtin and the C library function it lowers to.
_ROUNDEVEN_CALLS = {"__builtin_roundeven", "__builtin_roundevenf", "__builtin_roundevenl",
                    "roundeven", "roundevenf", "roundevenl"}


def inline_roundeven(text: str, origin: str) -> str:
    """`text` with every upstream `roundeven_finite` helper (the builtin macro and its
    fallback) replaced by `ROUNDEVEN_FINITE`."""
    for helper in _ROUNDEVEN_HELPER.findall(text):
        if not _ROUNDEVEN_DEFINE.search(helper) or not _ROUNDEVEN_DEFINITION.search(helper):
            raise VendorError(f"{origin}: an upstream roundeven_finite helper no longer has the shape this replaces")
    return _ROUNDEVEN_HELPER.sub(lambda _m: ROUNDEVEN_FINITE, text)


def route_sin_roundeven(text: str) -> str:
    """binary64 `sin`'s text with `ROUNDEVEN_FINITE` after its `FENV_ACCESS` pragma
    and its `__builtin_roundeven` call replaced by a call to that helper."""
    if text.count(_SIN_ROUNDEVEN_CALL) != 1 or text.count(_FENV_PRAGMA) != 1:
        raise VendorError("binary64 sin no longer has the one `__builtin_roundeven` call this routes")
    text = text.replace(_SIN_ROUNDEVEN_CALL, "roundeven_finite (invpi * ax)")
    return text.replace(_FENV_PRAGMA, f"{_FENV_PRAGMA}\n{ROUNDEVEN_FINITE}")


def require_inline_roundeven(text: str, origin: str) -> None:
    """Fail unless `text` rounds to an integer only through `ROUNDEVEN_FINITE`, under
    any kernel prefix: no `roundeven` builtin or C library function in its code, and
    every `roundeven_finite` definition is that helper."""
    found = sorted(_code_tokens(text) & _ROUNDEVEN_CALLS)
    if found:
        raise VendorError(f"{origin}: `{found[0]}` rounds to an integer outside the inline roundeven_finite")
    macro = _ROUNDEVEN_MACRO.search(text)
    if macro:
        raise VendorError(f"{origin}: `{macro.group().strip()}` defines roundeven_finite as a macro, not the inline helper")
    definitions: dict[str, int] = {}
    for m in _ROUNDEVEN_DEFINITION.finditer(text):
        definitions[m.group(1)] = definitions.get(m.group(1), 0) + 1
    for prefix, count in sorted(definitions.items()):
        if text.count(ROUNDEVEN_FINITE.replace("roundeven_finite", prefix + "roundeven_finite")) != count:
            raise VendorError(f"{origin}: a {prefix}roundeven_finite definition is not the inline helper")


def _reindent(body: str, indent: str) -> str:
    """The round-to-nearest case body, moved out of its `case` to `indent`."""
    lines = [line for line in body.split("\n") if line.strip()]
    shift = min(len(line) - len(line.lstrip()) for line in lines) - len(indent)
    kept = "\n".join(line[shift:] for line in lines)
    return f"{indent}/* round to nearest, the only mode Chelis runs in */\n{kept}\n"


def _code_tokens(text: str) -> set[str]:
    """Identifier tokens outside comments and literals."""
    return {m.group() for m in _TOKEN.finditer(text) if m.lastgroup == "ident"}


# --- manifest -----------------------------------------------------------------


@dataclass
class Manifest:
    commit: str
    files: dict[str, str]  # relative path -> sha256
    identifiers: dict[str, list[str]]  # kernel name -> sorted file-scope names
    worst_cases: dict[str, str]  # function -> sha256 of upstream binary64 .wc corpus


def load_manifest(path: Path = MANIFEST) -> Manifest:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    upstream = data["upstream"]
    files = {row["path"]: row["sha256"] for row in data["file"]}
    identifiers = {row["name"]: list(row["identifiers"]) for row in data["kernel"]}
    worst = {row["function"]: row["sha256"] for row in data.get("worst_cases", [])}
    return Manifest(upstream["commit"], files, identifiers, worst)


def render_manifest(m: Manifest) -> str:
    lines = [
        "# Generated by scripts/vendor_core_math.py import; do not edit by hand.",
        "# The kernel files are upstream CORE-MATH, unmodified. `identifiers` are the",
        "# file-scope names (declarations from Clang AST dumps on arm64 and x86_64,",
        "# plus macros) that the amalgamation prefixes per kernel.",
        "",
        "[upstream]",
        f'url = "{UPSTREAM_URL}"',
        f'commit = "{m.commit}"',
        'license = "MIT"',
        "",
    ]
    for path in sorted(m.files):
        lines += ["[[file]]", f'path = "{path}"', f'sha256 = "{m.files[path]}"', ""]
    for k in KERNELS:
        names = ", ".join(f'"{n}"' for n in m.identifiers[k.name])
        lines += ["[[kernel]]", f'name = "{k.name}"', f"identifiers = [{names}]", ""]
    for fn in sorted(m.worst_cases):
        lines += [
            "[[worst_cases]]",
            f'function = "{fn}"',
            f'path = "src/binary64/{fn}/{fn}.wc"',
            f'sha256 = "{m.worst_cases[fn]}"',
            "",
        ]
    return "\n".join(lines).rstrip("\n") + "\n"


# --- amalgamation -------------------------------------------------------------

HEADER = """\
/* Generated by scripts/vendor_core_math.py from crates/chelis-crmath/vendor/core-math
 * (CORE-MATH {commit}, {url}, MIT licence; each kernel below
 * keeps its upstream copyright and permission notice). Do not edit: regenerate with
 * `.venv/bin/python scripts/vendor_core_math.py`; `--check` fails on any drift.
 *
 * Correctly rounded exp, log, sin, cos, tan, atan, tanh, erf, and erfc at binary32 and binary64
 * ([05-OP-46], spec/design/correctly_rounded_math.md). Every definition is static;
 * the entries are chelis_cr_<name>, and each returns [04-NUM-2]'s canonical quiet NaN
 * for every NaN result. */

#if defined(__FAST_MATH__)
#error "chelis-crmath: compiled with fast math; the kernels require IEEE semantics"
#endif
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "chelis-crmath: compiled with finite-math-only; the kernels require IEEE semantics"
#endif

#include <float.h>
#include <stdint.h>
#include <string.h>

#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "chelis-crmath: FLT_EVAL_METHOD must be 0 (no excess-precision evaluation)"
#endif

static float chelis_cr_canonical_nanf(void) {{
  const uint32_t bits = 0x7fc00000u;
  float r;
  memcpy(&r, &bits, sizeof r);
  return r;
}}

static double chelis_cr_canonical_nan(void) {{
  const uint64_t bits = 0x7ff8000000000000ull;
  double r;
  memcpy(&r, &bits, sizeof r);
  return r;
}}
"""

ENTRY = """\
static {ctype} {entry}({ctype} x) {{
  {ctype} y = {inner}(x);
  return y != y ? {nan}() : y;
}}
"""


def kernel_text(kernel: Kernel, names: list[str], read) -> str:
    raw = read(kernel.path)
    rename = set(names)
    if kernel.path == "src/binary64/sin/sin.c":
        raw = route_sin_roundeven(raw)
    text = inline_roundeven(inline_local_includes(raw, kernel, read), kernel.path)
    if _ROUNDEVEN_DEFINITION.search(text):
        rename.add("roundeven_finite")
    text = contract_clean(text, kernel.path)
    require_inline_roundeven(text, kernel.path)
    if kernel.upstream_entry not in rename:
        raise VendorError(f"{kernel.name}: entry {kernel.upstream_entry} missing from identifiers")
    body = rename_identifiers(text, rename, kernel.prefix)
    inner = kernel.prefix + kernel.upstream_entry
    # The definition itself says `static`, not only the declaration before it,
    # so a reader of the unit that does not apply C's linkage carry-over (the
    # generated-C binder) sees internal linkage too.
    body, count = re.subn(
        rf"^(?=(?:float|double)\s+{re.escape(inner)}\s*\()", "static ", body, flags=re.MULTILINE
    )
    if count == 0:
        raise VendorError(f"{kernel.name}: no definition of {kernel.upstream_entry} found")
    nan = "chelis_cr_canonical_nanf" if kernel.width == 32 else "chelis_cr_canonical_nan"
    return "".join(
        [
            f"\n/* ==== kernel {kernel.name}: {kernel.path} ==== */\n\n",
            f"static {kernel.ctype} {inner}({kernel.ctype});\n\n",
            body.rstrip("\n") + "\n\n",
            ENTRY.format(ctype=kernel.ctype, entry=kernel.entry, inner=inner, nan=nan),
        ]
    )


def verify_vendored(manifest: Manifest, vendor: Path) -> None:
    expected = {p for k in KERNELS for p in vendored_files(k)} | {LICENSE_FILE}
    if set(manifest.files) != expected:
        raise VendorError(
            "VENDOR.toml file list differs from the kernel set: "
            f"missing {sorted(expected - set(manifest.files))}, "
            f"extra {sorted(set(manifest.files) - expected)}"
        )
    for rel, digest in sorted(manifest.files.items()):
        path = vendor / rel
        if not path.is_file():
            raise VendorError(f"vendored file missing: {path}")
        actual = sha256(path.read_bytes())
        if actual != digest:
            raise VendorError(f"{rel}: sha256 {actual} differs from VENDOR.toml {digest}")
    missing = [k.name for k in KERNELS if k.name not in manifest.identifiers]
    if missing:
        raise VendorError(f"VENDOR.toml has no identifiers for {missing}")


def generate(vendor: Path = VENDOR, manifest_path: Path = MANIFEST) -> str:
    manifest = load_manifest(manifest_path)
    verify_vendored(manifest, vendor)

    def read(rel: str) -> str:
        return (vendor / rel).read_text(encoding="utf-8")

    parts = [HEADER.format(commit=manifest.commit, url=UPSTREAM_URL)]
    for kernel in KERNELS:
        parts.append(kernel_text(kernel, manifest.identifiers[kernel.name], read))
    return "".join(parts)


# --- import (maintainer action; needs clang) ---------------------------------


def _ast_names(source: Path, extra_args: list[str]) -> set[str]:
    """File-scope names declared in `source` itself (not its system headers)."""
    cmd = ["clang", "-std=c11", "-Xclang", "-ast-dump=json", "-fsyntax-only", *extra_args, str(source)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise VendorError(f"clang failed on {source} {extra_args}:\n{proc.stderr}")
    tree = json.loads(proc.stdout)
    defined: set[str] = set()
    declared_only: set[str] = set()
    for node in tree.get("inner", []):
        loc = node.get("loc", {})
        if node.get("isImplicit") or "includedFrom" in loc or "includedFrom" in loc.get("spellingLoc", {}):
            continue
        name = node.get("name")
        kind = node.get("kind")
        if not name:
            continue
        if kind == "FunctionDecl":
            has_body = any(c.get("kind") == "CompoundStmt" for c in node.get("inner", []))
            (defined if has_body else declared_only).add(name)
            if has_body and node.get("storageClass") != "static" and not name.startswith("cr_"):
                raise VendorError(f"{source}: non-static helper {name} is not an entry")
        elif kind == "VarDecl":
            if node.get("storageClass") != "static":
                raise VendorError(f"{source}: file-scope object {name} lacks static linkage")
            defined.add(name)
        elif kind in ("TypedefDecl", "RecordDecl", "EnumDecl"):
            defined.add(name)
            if kind == "EnumDecl":
                defined |= {c["name"] for c in node.get("inner", []) if c.get("kind") == "EnumConstantDecl"}
    # A prototype without a definition names a C library function; keep it.
    return defined


AST_CONFIGS = (
    [],
    ["-target", "arm64-apple-macos11"],
    ["-target", "x86_64-apple-macos11"],
    ["-target", "x86_64-apple-macos11", "-mavx2", "-mfma"],
)


def import_upstream(upstream: Path, vendor: Path = VENDOR) -> Manifest:
    if shutil.which("clang") is None:
        raise VendorError("import needs clang on PATH for the AST dumps")
    commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    files: dict[str, str] = {}
    for rel in [p for k in KERNELS for p in vendored_files(k)] + [LICENSE_FILE]:
        src = upstream / rel
        dst = vendor / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files[rel] = sha256(dst.read_bytes())
    identifiers: dict[str, list[str]] = {}
    for kernel in KERNELS:
        text = inline_local_includes(
            (vendor / kernel.path).read_text(encoding="utf-8"),
            kernel,
            lambda rel: (vendor / rel).read_text(encoding="utf-8"),
        )
        names = macro_names(text)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / Path(kernel.path).name
            src.write_text(text, encoding="utf-8")
            for config in AST_CONFIGS:
                names |= _ast_names(src, config)
        identifiers[kernel.name] = sorted(names)
    worst = {}
    for fn in FUNCTIONS:
        wc = upstream / f"src/binary64/{fn}/{fn}.wc"
        worst[fn] = sha256(wc.read_bytes())
    manifest = Manifest(commit, files, identifiers, worst)
    (vendor / "VENDOR.toml").write_text(render_manifest(manifest), encoding="utf-8")
    return manifest


# --- MPFR fixtures (maintainer action; needs gmpy2) --------------------------


def _require_gmpy2():
    if gmpy2 is None:
        print(
            "error: this subcommand needs gmpy2 (MPFR). Install it into this checkout's venv:\n"
            "  uv pip install --python .venv/bin/python gmpy2",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return gmpy2


LN2 = 0.6931471805599453
CANONICAL_NAN = {32: 0x7FC00000, 64: 0x7FF8000000000000}


def bits_to_float(bits: int, width: int) -> float:
    if width == 32:
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def float_to_bits(value: float, width: int) -> int:
    if width == 32:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def mpfr_reference(gmpy2, function: str, bits: int, width: int) -> int:
    """The correctly rounded result bits under IEEE binary32/binary64 (MPFR with
    the width's precision, exponent range, and subnormalization)."""
    x = bits_to_float(bits, width)
    ctx = gmpy2.ieee(width)
    ctx.round = gmpy2.RoundToNearest
    with gmpy2.context(ctx):
        fn = {"exp": gmpy2.exp, "log": gmpy2.log, "sin": gmpy2.sin, "cos": gmpy2.cos,
              "tan": gmpy2.tan, "atan": gmpy2.atan, "tanh": gmpy2.tanh,
              "erf": gmpy2.erf, "erfc": gmpy2.erfc}[function]
        y = fn(gmpy2.mpfr(x))
        if gmpy2.is_nan(y):
            return CANONICAL_NAN[width]
        return float_to_bits(float(y), width)


def _f32(x: float) -> int:
    return float_to_bits(x, 32)


def _f64(x: float) -> int:
    return float_to_bits(x, 64)


def special_inputs(width: int) -> list[tuple[int, str]]:
    """[05-OP-46] special cases shared by every function, as (bits, note)."""
    if width == 32:
        return [
            (0x00000000, "+0"), (0x80000000, "-0"), (0x7F800000, "+inf"), (0xFF800000, "-inf"),
            (0x7FC00000, "canonical NaN"), (0xFFC00000, "negative quiet NaN"),
            (0x7FC12345, "quiet NaN with payload"), (0x7F800001, "signaling NaN"),
            (0xFFA00001, "negative signaling NaN with payload"),
            (0x00000001, "smallest subnormal"), (0x80000001, "-smallest subnormal"),
            (0x007FFFFF, "largest subnormal"), (0x00800000, "smallest normal"),
            (0x7F7FFFFF, "largest finite"), (0xFF7FFFFF, "-largest finite"),
            (_f32(1.0), "1"), (_f32(-1.0), "-1"), (_f32(0.5), "0.5"),
        ]
    return [
        (0x0000000000000000, "+0"), (0x8000000000000000, "-0"),
        (0x7FF0000000000000, "+inf"), (0xFFF0000000000000, "-inf"),
        (0x7FF8000000000000, "canonical NaN"), (0xFFF8000000000000, "negative quiet NaN"),
        (0x7FF8000000012345, "quiet NaN with payload"), (0x7FF0000000000001, "signaling NaN"),
        (0xFFF4000000000001, "negative signaling NaN with payload"),
        (0x0000000000000001, "smallest subnormal"), (0x8000000000000001, "-smallest subnormal"),
        (0x000FFFFFFFFFFFFF, "largest subnormal"), (0x0010000000000000, "smallest normal"),
        (0x7FEFFFFFFFFFFFFF, "largest finite"), (0xFFEFFFFFFFFFFFFF, "-largest finite"),
        (_f64(1.0), "1"), (_f64(-1.0), "-1"), (_f64(0.5), "0.5"),
    ]


def function_inputs(function: str, width: int) -> list[tuple[int, str]]:
    """Per-function special cases, thresholds, and the #2957 witnesses."""
    f = _f32 if width == 32 else _f64
    rows: list[tuple[int, str]] = []
    if function == "exp":
        if width == 32:
            rows += [
                (0x42B17217, "largest x with finite expf"), (0x42B17218, "smallest x overflowing expf"),
                (_f32(-149 * LN2), "x near -149 ln 2: result near the smallest subnormal"),
                (_f32(-150 * LN2), "x near -150 ln 2: result near half the smallest subnormal"),
                (_f32(-126 * LN2), "x near -126 ln 2: result near the smallest normal"),
                (_f32(-103.97), "exp(-103.97)"),
                (_f32(-104.0), "exp(-104): correctly rounded result is +0"),
                (_f32(-1.72889078), "#2952: macOS expf misrounds"),
            ]
            for x in (-1.3359944820404053, -0.46123430132865906, -1.219169020652771,
                      0.9197595119476318, -0.33510392904281616, 0.03105185739696026):
                rows.append((_f32(x), "#2952 normal_cdf input"))
                xf = bits_to_float(_f32(x), 32)
                half_sq = bits_to_float(_f32(-bits_to_float(_f32(xf * xf), 32) * 0.5), 32)
                rows.append((_f32(half_sq), "#2952 normal_cdf exp argument -x*x/2"))
            rows += [(_f32(-0.0480351411), "#2952 sigmoid exp(-x)"),
                     (_f32(1.8767333030700684), "#2959 gelu input negated"),
                     (_f32(0.05580474063754082), "#2959 tanh input negated")]
            softmax = [0.1, 1.7, -2.3, 0.33, 3.9, -0.77, 2.2, 1.05, -5.5, 0.6, 4.1, -1.9, 2.75, 0.01, -0.4, 1.3]
            top = bits_to_float(_f32(4.1), 32)
            for v in softmax:
                shifted = bits_to_float(_f32(bits_to_float(_f32(v), 32) - top), 32)
                rows.append((_f32(shifted), "#2971 softmax max-shifted exponent"))
        else:
            rows += [
                (0x40862E42FEFA39EF, "largest x with finite exp"), (0x40862E42FEFA39F0, "smallest x overflowing exp"),
                (_f64(-1074 * LN2), "x near -1074 ln 2: result near the smallest subnormal"),
                (_f64(-1075 * LN2), "x near -1075 ln 2: result near half the smallest subnormal"),
                (_f64(-1022 * LN2), "x near -1022 ln 2: result near the smallest normal"),
                (_f64(-746.0), "exp(-746): result +0"),
                (0xBFD93179431561A0, "#2952: macOS exp misrounds"),
            ]
    elif function == "log":
        rows += [(f(-1.0), "log of a negative: NaN"), (f(2.0), "log 2"), (f(10.0), "log 10")]
    elif function in ("sin", "cos", "tan"):
        rows += [(f(1e22), "large argument"), (f(3.141592653589793), "near pi"),
                 (f(1.5707963267948966), "near pi/2"), (f(-1e-3), "small negative")]
        if width == 32:
            rows += [(_f32(2.0**100), "2^100")]
        else:
            rows += [(0x7FE0000000000000, "2^1023")]
    elif function == "atan":
        rows += [(f(1e30), "large argument"), (f(-1e30), "large negative argument"), (f(1e-30), "tiny argument")]
    elif function == "tanh":
        rows += [(f(1e-3), "#2959: old lowering 620 ULP off"), (f(1e-5), "#2959: old lowering 14,932 ULP off"),
                 (f(-0.05580474063754082), "#2959 tanh witness"), (f(20.0), "saturates near 1"),
                 (f(-20.0), "saturates near -1"), (f(1e-30), "tiny argument")]
    elif function in ("erf", "erfc"):
        rows += [(f(1e-30), "tiny argument"), (f(-1e-30), "tiny negative argument"),
                 (f(0.5), "erf(0.5)"), (f(-2.0), "erfc near 2"), (f(3.0), "erfc tail"),
                 (f(6.0), "erf saturates near 1"), (f(-6.0), "erf saturates near -1")]
        # [05-OP-35] normal_cdf arguments -x/sqrt(2): the deep left tail where erfc
        # underflows, and the |x| = 64 bound of the spec/05 section 3.3 Phi graph.
        for x in (-5.0, -12.6, -36.5, -38.4, 45.25, -45.25):
            rows.append((f(-x * 0.7071067811865476), f"Phi argument at x = {x}"))
        if width == 32:
            rows += [(_f32(10.0546875), "erfcf result near the smallest subnormal"),
                     (_f32(10.1), "erfcf result +0")]
        else:
            rows += [(_f64(27.2), "erfc result subnormal"), (_f64(27.3), "erfc result +0")]
    return rows


# f32 inputs on which the correctly rounded f64 result, narrowed once to f32, differs
# from the correctly rounded f32 result (double rounding). Found by an exhaustive scan
# of all 2^32 inputs comparing the two kernels; the expected bits still come from
# MPFR. They let the canary show that it rejects the evaluator's former f32 route
# (compute at f64, narrow once). The scan found none for exp and tan.
FUNNEL_WITNESSES: dict[str, tuple[int, ...]] = {
    "log": (0x3C413D3A, 0x41178FEB, 0x4C5D65A5),
    "sin": (0xC6199998, 0x46199998),
    "cos": (0xDF18B878, 0xE115CB11),
    "atan": (0xBD8D6B23, 0x3D8D6B23),
}
FUNNEL_NOTE = "f64 funnel double-rounds"


def canary_inputs() -> list[tuple[str, int, int, str]]:
    """(function, width, input bits, note) for every canary row, deduplicated by input."""
    out = []
    for width in (32, 64):
        for function in FUNCTIONS:
            rows = special_inputs(width) + function_inputs(function, width)
            if width == 32:
                rows += [(bits, FUNNEL_NOTE) for bits in FUNNEL_WITNESSES.get(function, ())]
            seen = set()
            for bits, note in rows:
                if bits not in seen:
                    seen.add(bits)
                    out.append((function, width, bits, note))
    return out


def fixture_row(function: str, width: int, bits: int, expected: int, note: str) -> str:
    digits = 8 if width == 32 else 16
    return f"{function} f{width} {bits:0{digits}x} {expected:0{digits}x} # {note}"


def canary_rows(gmpy2) -> list[str]:
    return [
        fixture_row(function, width, bits, mpfr_reference(gmpy2, function, bits, width), note)
        for function, width, bits, note in canary_inputs()
    ]


WORST_CASE_SAMPLE = 128  # per function, deterministic


def worst_case_rows(gmpy2, upstream: Path, manifest: Manifest) -> list[str]:
    rows = []
    for function in FUNCTIONS:
        wc = upstream / f"src/binary64/{function}/{function}.wc"
        data = wc.read_bytes()
        if manifest.worst_cases.get(function) != sha256(data):
            raise VendorError(f"{wc}: does not match VENDOR.toml worst_cases sha256")
        inputs = [line.split()[0] for line in data.decode().splitlines() if line.strip() and not line.startswith("#")]
        rng = random.Random(f"chelis-crmath-{function}")
        picked = sorted(rng.sample(range(len(inputs)), min(WORST_CASE_SAMPLE, len(inputs))))
        for index in picked:
            x = float.fromhex(inputs[index])
            for value in (x, -x):
                bits = _f64(value)
                expected = mpfr_reference(gmpy2, function, bits, 64)
                rows.append(fixture_row(function, 64, bits, expected, f"{function}.wc line {index}"))
    return rows


# --- Profile obligations (maintainer action; needs gmpy2) ---------------------
#
# `fixtures/profile_obligations.txt` is the arithmetic part of the floating-point
# profile's obligation table; `canary.txt` and `binary64_worst_cases.txt` are its kernel
# part. `chelis_crmath::profile` reads all three: it generates the compiler canary that
# `chelis build` compiles and runs under the selected compiler and profile, and the
# crate's tests check the same rows in the Rust lanes. Each row names the obligation it
# witnesses (`profile::Obligation` maps each to the spec text it enforces).

OBLIGATION_FIXTURE = FIXTURES / "profile_obligations.txt"
F16_CONTEXT = dict(precision=11, emin=-23, emax=16)
BF16_CONTEXT = dict(precision=8, emin=-132, emax=128)
RESULT_WIDTH = {"eq": 1, "ne": 1, "lt": 1, "self_eq": 1, "gt_max": 1}


def _ctx(width: int):
    if width == 16:
        return gmpy2.context(round=gmpy2.RoundToNearest, subnormalize=True, **F16_CONTEXT)
    if width == 8:  # bfloat16
        return gmpy2.context(round=gmpy2.RoundToNearest, subnormalize=True, **BF16_CONTEXT)
    ctx = gmpy2.ieee(width)
    ctx.round = gmpy2.RoundToNearest
    return ctx


def _result_bits(y, width: int) -> int:
    """Bits of an MPFR value already rounded to `width` (8 is bfloat16)."""
    if gmpy2.is_nan(y):
        return {8: 0x7FC0, 16: 0x7E00, 32: 0x7FC00000, 64: 0x7FF8000000000000}[width]
    value = float(y)
    if width == 16:
        return struct.unpack("<H", struct.pack("<e", value))[0]
    if width == 8:
        return float_to_bits(value, 32) >> 16
    return float_to_bits(value, width)


def _round_to(width: int, value: float):
    """`value` (exact in binary64) rounded once to `width`, keeping its sign."""
    exact = gmpy2.mpfr(value, 53)  # outside the target context, so it is not rounded
    with gmpy2.context(_ctx(width)):
        return gmpy2.mul(exact, 1)


def obligation_reference(primitive: str, width: int, operands: list[int]) -> int:
    """The profile's result bits for one row: MPFR at the width's IEEE precision,
    exponent range, and subnormalization, each written operation rounded once, NaN
    results canonical."""
    xs = [bits_to_float(bits, width) for bits in operands]
    if primitive in RESULT_WIDTH:
        a = xs[0]
        b = xs[1] if len(xs) > 1 else None
        limit = bits_to_float(0x7F7FFFFF if width == 32 else 0x7FEFFFFFFFFFFFFF, width)
        return int({"eq": lambda: a == b, "ne": lambda: a != b, "lt": lambda: a < b,
                    "self_eq": lambda: a == a, "gt_max": lambda: a > limit}[primitive]())
    if primitive in ("narrow", "widen", "to_f16", "to_bf16"):
        target = TARGET_WIDTH[primitive]
        return _result_bits(_round_to(target, xs[0]) if xs[0] == xs[0] else gmpy2.nan(), target)
    m = [gmpy2.mpfr(x, 53) for x in xs]  # exact: every operand is a binary64 value
    with gmpy2.context(_ctx(width)):
        a = m[0]
        b = m[1] if len(m) > 1 else None
        c = m[2] if len(m) > 2 else None
        y = {
            "add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b,
            "div": lambda: a / b, "sqrt": lambda: gmpy2.sqrt(a),
            "fma": lambda: gmpy2.fma(a, b, c),
            "mul_add": lambda: (a * b) + c, "add_sub": lambda: (a + b) - a,
            "div_three": lambda: a / 3, "add_zero": lambda: a + 0,
            "sub_self": lambda: a - a, "mul_zero": lambda: a * 0,
        }[primitive]()
    return _result_bits(y, width)


def _is_tie(exact: Fraction, width: int) -> bool:
    """Whether `exact` lies exactly halfway between two adjacent `width` values."""
    precision, smallest = {8: (8, -133), 16: (11, -24), 32: (24, -149), 64: (53, -1074)}[width]
    if exact == 0:
        return False
    magnitude = abs(exact)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    half_ulp = Fraction(2) ** max(exponent - precision, smallest - 1)
    scaled = magnitude / half_ulp
    return scaled.denominator == 1 and scaled.numerator % 2 == 1


TARGET_WIDTH = {"narrow": 32, "widen": 64, "to_f16": 16, "to_bf16": 8}


def _exact(primitive: str, operands: list[Fraction]) -> Fraction:
    a, *rest = operands
    return {
        "add": lambda: a + rest[0], "sub": lambda: a - rest[0], "mul": lambda: a * rest[0],
        "div": lambda: a / rest[0], "fma": lambda: a * rest[0] + rest[1],
    }.get(primitive, lambda: a)()


def obligation_rows() -> list[str]:
    """Every arithmetic row as `primitive width expected operand... # obligation: note`.
    A note that starts with `tie:` claims an exact midpoint; it is checked here."""
    lines = []
    for primitive, width, operands, obligation, note in obligation_inputs():
        if note.startswith("tie:"):
            exact = _exact(primitive, [Fraction(bits_to_float(bits, width)) for bits in operands])
            if not _is_tie(exact, TARGET_WIDTH.get(primitive, width)):
                raise VendorError(f"{primitive} f{width} {operands}: `{note}` is not a midpoint")
        expected = obligation_reference(primitive, width, operands)
        result_width = RESULT_WIDTH.get(primitive, TARGET_WIDTH.get(primitive, width))
        digits = {1: 1, 8: 4, 16: 4, 32: 8, 64: 16}[result_width]
        operand_digits = 8 if width == 32 else 16
        text = " ".join(f"{bits:0{operand_digits}x}" for bits in operands)
        lines.append(f"{primitive} f{width} {expected:0{digits}x} {text} # {obligation}: {note}")
    return lines


OBLIGATION_HEADER = (
    "# Generated by `scripts/vendor_core_math.py obligations` from MPFR (gmpy2) at the\n"
    "# result width's IEEE precision, exponent range, and subnormalization, each written\n"
    "# operation rounded once to nearest even; NaN results are [04-NUM-2]'s canonical\n"
    "# quiet NaN. bf16 and f16 results are storage bits; comparison results are 0 or 1.\n"
    "# Columns: primitive operand-width expected-bits operand-bits... # obligation: note\n"
)


def _special(width: int) -> dict[str, int]:
    if width == 32:
        return {"qnan": 0x7FC00000, "pnan": 0x7FC12345, "nnan": 0xFFC00000, "snan": 0x7F800001,
                "nsnan": 0xFFA00001, "inf": 0x7F800000, "ninf": 0xFF800000, "zero": 0,
                "nzero": 0x80000000, "minsub": 1, "maxsub": 0x007FFFFF, "minnorm": 0x00800000,
                "max": 0x7F7FFFFF}
    return {"qnan": 0x7FF8000000000000, "pnan": 0x7FF8000000012345, "nnan": 0xFFF8000000000000,
            "snan": 0x7FF0000000000001, "nsnan": 0xFFF4000000000001, "inf": 0x7FF0000000000000,
            "ninf": 0xFFF0000000000000, "zero": 0, "nzero": 0x8000000000000000, "minsub": 1,
            "maxsub": 0x000FFFFFFFFFFFFF, "minnorm": 0x0010000000000000, "max": 0x7FEFFFFFFFFFFFFF}


def obligation_inputs() -> list[tuple[str, int, list[int], str, str]]:
    """(primitive, width, operand bits, obligation, note) for every arithmetic row."""
    rows: list[tuple[str, int, list[int], str, str]] = []
    for width in (32, 64):
        s = _special(width)
        p = 24 if width == 32 else 53
        f = _f32 if width == 32 else _f64

        def add(primitive, operands, obligation, note, width=width):
            rows.append((primitive, width, [o if isinstance(o, int) else f(o) for o in operands], obligation, note))

        one, two, three = 1.0, 2.0, 3.0
        u = 2.0 ** -(p - 1)  # ulp of 1
        for primitive in ("add", "sub", "mul", "div"):
            add(primitive, [s["pnan"], one], "canonical-nan", "quiet NaN with payload operand")
            add(primitive, [one, s["nnan"]], "canonical-nan", "negative quiet NaN operand")
            add(primitive, [s["snan"], two], "canonical-nan", "signaling NaN operand")
            add(primitive, [s["nsnan"], s["inf"]], "canonical-nan", "negative signaling NaN with payload and inf")
            add(primitive, [s["inf"], one], "ieee-special", "inf operand")
            add(primitive, [s["ninf"], two], "ieee-special", "-inf operand")
            add(primitive, [s["minsub"], one], "gradual-underflow", "smallest subnormal operand")
            add(primitive, [s["maxsub"], s["minsub"]], "gradual-underflow", "subnormal operands")
            add(primitive, [s["nzero"], s["nzero"]], "signed-zero", "-0 and -0")
            add(primitive, [s["nzero"], s["zero"]], "signed-zero", "-0 and +0")
            add(primitive, [s["zero"], s["nzero"]], "signed-zero", "+0 and -0")
            add(primitive, [s["max"], s["max"]], "ieee-special", "largest finite operands")
        add("add", [s["inf"], s["ninf"]], "ieee-special", "inf + -inf is invalid")
        add("sub", [s["inf"], s["inf"]], "ieee-special", "inf - inf is invalid")
        add("mul", [s["zero"], s["inf"]], "ieee-special", "0 * inf is invalid")
        add("mul", [s["ninf"], s["nzero"]], "ieee-special", "-inf * -0 is invalid")
        add("div", [s["inf"], s["ninf"]], "ieee-special", "inf / -inf is invalid")
        add("div", [s["zero"], s["nzero"]], "ieee-special", "0 / -0 is invalid")
        add("div", [one, s["zero"]], "ieee-special", "1 / +0 is +inf")
        add("div", [one, s["nzero"]], "ieee-special", "1 / -0 is -inf")
        add("div", [-one, s["inf"]], "signed-zero", "-1 / inf is -0")
        add("mul", [-one, s["zero"]], "signed-zero", "-1 * +0 is -0")
        add("sub", [one, one], "signed-zero", "x - x is +0")
        add("add", [s["minsub"], s["minsub"]], "gradual-underflow", "subnormal result")
        add("sub", [s["minnorm"], s["minsub"]], "gradual-underflow", "smallest normal - smallest subnormal is subnormal")
        add("mul", [2.0 ** -1000, 2.0 ** -60] if width == 64 else [2.0 ** -100, 2.0 ** -30],
            "gradual-underflow", "product underflows to a subnormal")
        add("mul", [s["minsub"], 0.5], "round-once", "tie: half the smallest subnormal is +0")
        add("mul", [3, 0.5], "round-once", "tie: 1.5 smallest subnormals is 2")
        add("div", [s["minnorm"], 2.0 ** 10], "gradual-underflow", "quotient is subnormal")
        add("div", [s["minsub"], 2.0], "round-once", "tie: half the smallest subnormal is +0")
        # Ties to even at the width: the exact result is a midpoint.
        add("add", [one, u / 2], "round-once", "tie: 1 + half an ulp stays 1")
        add("add", [one + u, u / 2], "round-once", "tie: rounds up to the even neighbour")
        add("sub", [one, u / 4], "round-once", "tie: just below a binade boundary")
        add("sub", [two, u * 1.5], "round-once", "tie: rounds to the even neighbour")
        # (1 + 2^-k)(1 + 2^-j) = 1 + 2^-k + 2^-j + 2^-(k+j), a midpoint when k + j = p.
        k = p // 2
        add("mul", [1.0 + 2.0 ** -k, 1.0 + 2.0 ** -(p - k)], "round-once", "tie: product rounds to even")
        add("mul", [1.0 + 2.0 ** -k + u, 1.0 + 2.0 ** -(p - k)], "round-once", "near a tie: product rounds up")
        add("div", [one, three], "round-once", "1 / 3")
        add("div", [two, three], "round-once", "2 / 3")
        add("div", [one, 10.0], "round-once", "1 / 10")
        # sqrt
        add("sqrt", [s["pnan"]], "canonical-nan", "quiet NaN with payload operand")
        add("sqrt", [s["nnan"]], "canonical-nan", "negative quiet NaN operand")
        add("sqrt", [s["snan"]], "canonical-nan", "signaling NaN operand")
        add("sqrt", [-one], "ieee-special", "sqrt of a negative is invalid")
        add("sqrt", [s["ninf"]], "ieee-special", "sqrt(-inf) is invalid")
        add("sqrt", [s["inf"]], "ieee-special", "sqrt(inf) is inf")
        add("sqrt", [s["nzero"]], "signed-zero", "sqrt(-0) is -0")
        add("sqrt", [s["zero"]], "signed-zero", "sqrt(+0) is +0")
        add("sqrt", [s["minsub"]], "gradual-underflow", "sqrt of the smallest subnormal")
        add("sqrt", [s["maxsub"]], "gradual-underflow", "sqrt of the largest subnormal")
        add("sqrt", [two], "round-once", "sqrt(2)")
        add("sqrt", [three], "round-once", "sqrt(3)")
        add("sqrt", [s["max"]], "round-once", "sqrt of the largest finite")
        # The explicit fused multiply-add ([05-OP-8]) rounds once.
        a = 1.0 + 2.0 ** -(p // 2 + 1)
        add("fma", [a, a, -(1.0 + 2.0 ** -(p // 2))], "round-once", "fused: keeps the low product bits")
        add("fma", [s["pnan"], one, one], "canonical-nan", "quiet NaN with payload operand")
        add("fma", [one, one, s["nnan"]], "canonical-nan", "negative quiet NaN addend")
        add("fma", [s["snan"], s["zero"], one], "canonical-nan", "signaling NaN operand")
        add("fma", [s["inf"], s["zero"], one], "ieee-special", "inf * 0 + 1 is invalid")
        add("fma", [s["inf"], one, s["ninf"]], "ieee-special", "inf * 1 - inf is invalid")
        add("fma", [s["max"], two, s["zero"]], "ieee-special", "overflow to inf")
        add("fma", [s["nzero"], one, s["nzero"]], "signed-zero", "-0 * 1 + -0 is -0")
        add("fma", [one, s["zero"], s["nzero"]], "signed-zero", "1 * 0 + -0 is +0")
        add("fma", [s["minsub"], 0.5, s["minsub"]], "gradual-underflow", "subnormal operands, tie to even")
        add("fma", [s["minnorm"], 0.5, s["zero"]], "gradual-underflow", "subnormal result")
        add("fma", [one, one, u / 2], "round-once", "tie: 1 + half an ulp stays 1")
        # Composite expressions: the shapes value-changing optimizations rewrite.
        b = 1.0 + 2.0 ** -(p // 2 + 1)
        add("mul_add", [b, b, -(1.0 + 2.0 ** -(p // 2))], "no-contraction",
            "a*b + c rounds the product: fused, the low bits survive")
        add("mul_add", [one + u, one - u / 2, -one], "no-contraction", "a*b + c, product rounded first")
        add("mul_add", [s["pnan"], one, one], "canonical-nan", "NaN payload through a*b + c")
        add("mul_add", [s["inf"], s["zero"], one], "ieee-special", "inf*0 + 1 is invalid")
        add("mul_add", [s["nzero"], one, s["nzero"]], "signed-zero", "-0*1 + -0 is -0")
        add("mul_add", [s["minsub"], 0.5, s["minsub"]], "gradual-underflow", "subnormal operands")
        add("add_sub", [2.0 ** p, one], "no-value-changing-optimization", "(a + b) - a is not b")
        add("add_sub", [one, u / 2], "no-value-changing-optimization", "(1 + half ulp) - 1 is 0")
        add("add_sub", [s["inf"], one], "ieee-special", "(inf + 1) - inf is invalid")
        add("add_sub", [s["max"], s["max"]], "ieee-special", "(max + max) - max is inf")
        add("div_three", [5.0], "no-value-changing-optimization", "5 / 3 is not 5 * (1/3)")
        add("div_three", [s["inf"]], "ieee-special", "inf / 3")
        add("div_three", [s["minsub"]], "gradual-underflow", "subnormal / 3")
        add("add_zero", [s["nzero"]], "signed-zero", "-0 + 0 is +0")
        add("add_zero", [s["pnan"]], "canonical-nan", "NaN payload + 0")
        add("sub_self", [s["inf"]], "ieee-special", "inf - inf is invalid")
        add("sub_self", [s["pnan"]], "canonical-nan", "NaN - NaN")
        add("sub_self", [-one], "signed-zero", "x - x is +0")
        add("mul_zero", [s["inf"]], "ieee-special", "inf * 0 is invalid")
        add("mul_zero", [-one], "signed-zero", "-1 * 0 is -0")
        add("mul_zero", [s["pnan"]], "canonical-nan", "NaN payload * 0")
        add("self_eq", [s["pnan"]], "ieee-special", "NaN == NaN is false")
        add("self_eq", [s["inf"]], "ieee-special", "inf == inf is true")
        add("gt_max", [s["inf"]], "ieee-special", "inf > largest finite")
        add("gt_max", [s["max"]], "ieee-special", "largest finite > largest finite is false")
        for primitive in ("eq", "ne", "lt"):
            add(primitive, [s["qnan"], s["qnan"]], "ieee-special", "NaN and NaN are unordered")
            add(primitive, [s["pnan"], one], "ieee-special", "NaN and 1 are unordered")
            add(primitive, [s["nzero"], s["zero"]], "signed-zero", "-0 and +0 compare equal")
            add(primitive, [s["inf"], s["max"]], "ieee-special", "inf and the largest finite")
            add(primitive, [s["minsub"], s["zero"]], "gradual-underflow", "a subnormal is not zero")
    # Conversions between widths ([04-NUM-2] finalization, [04-NUM-14]).
    s64, s32 = _special(64), _special(32)
    conversions = [
        ("narrow", 64, 0x7FFC000000012345, "canonical-nan", "NaN payload that survives narrowing"),
        ("narrow", 64, s64["nnan"], "canonical-nan", "negative NaN narrows to the canonical NaN"),
        ("narrow", 64, s64["inf"], "ieee-special", "inf"),
        ("narrow", 64, s64["max"], "ieee-special", "overflow to inf"),
        ("narrow", 64, s64["nzero"], "signed-zero", "-0"),
        ("narrow", 64, s64["minsub"], "gradual-underflow", "below half the f32 subnormal: +0"),
        ("narrow", 64, _f64(2.0 ** -149), "gradual-underflow", "the smallest f32 subnormal"),
        ("narrow", 64, _f64(2.0 ** -150), "round-once", "tie: half the smallest f32 subnormal is +0"),
        ("narrow", 64, _f64(1.0 + 2.0 ** -24), "round-once", "tie: 1 + 2^-24 is 1"),
        ("narrow", 64, _f64(1.0 + 2.0 ** -23 + 2.0 ** -24), "round-once", "tie: rounds up to even"),
        ("widen", 32, s32["pnan"], "canonical-nan", "NaN payload widens to the canonical NaN"),
        ("widen", 32, s32["nsnan"], "canonical-nan", "negative signaling NaN widens canonical"),
        ("widen", 32, s32["ninf"], "ieee-special", "-inf"),
        ("widen", 32, s32["nzero"], "signed-zero", "-0"),
        ("widen", 32, s32["minsub"], "gradual-underflow", "f32 subnormal widens exactly"),
    ]
    for target in ("to_f16", "to_bf16"):
        for width, s in ((32, s32), (64, s64)):
            f = _f32 if width == 32 else _f64
            conversions += [
                (target, width, s["pnan"], "canonical-nan", "NaN payload"),
                (target, width, s["nnan"], "canonical-nan", "negative NaN"),
                (target, width, s["ninf"], "ieee-special", "-inf"),
                (target, width, s["max"], "ieee-special", "overflow to inf"),
                (target, width, s["nzero"], "signed-zero", "-0"),
                (target, width, s["minsub"], "gradual-underflow", "far below the target's subnormals: +0"),
            ]
            if target == "to_f16":
                conversions += [
                    (target, width, f(65519.0), "round-once", "largest value rounding to 65504"),
                    (target, width, f(65520.0), "ieee-special", "tie at the f16 overflow threshold: inf"),
                    (target, width, f(2.0 ** -24), "gradual-underflow", "the smallest f16 subnormal"),
                    (target, width, f(2.0 ** -25), "round-once", "tie: half the smallest f16 subnormal is +0"),
                    (target, width, f(3 * 2.0 ** -25), "round-once", "tie: 1.5 smallest subnormals is 2"),
                    (target, width, f(2.0 ** -25 + 2.0 ** -40), "round-once", "just above the subnormal tie"),
                    # Bits far below the subnormal quantum still decide a near-tie.
                    (target, width, f(2.0 ** -25 * (1 + 2.0 ** -23)), "gradual-underflow",
                     "one f32 ulp above half the smallest subnormal: rounds up"),
                    (target, width, f(5 * 2.0 ** -25 + 2.0 ** -46), "gradual-underflow",
                     "one f32 ulp above the tie between 2 and 3 subnormal quanta: rounds up to 3"),
                    (target, width, f(2.0 ** -15 + 2.0 ** -25 + 2.0 ** -38), "gradual-underflow",
                     "just above a tie in a larger subnormal: rounds up"),
                    (target, width, f(2.0 ** -15 + 2.0 ** -25 - 2.0 ** -38), "gradual-underflow",
                     "just below a tie in a larger subnormal: rounds down"),
                    (target, width, f(1.0 + 2.0 ** -11), "round-once", "tie: 1 + 2^-11 is 1"),
                    (target, width, f(1.0 + 2.0 ** -10 + 2.0 ** -11), "round-once", "tie: rounds up to even"),
                    (target, width, f(1.0 + 2.0 ** -11 + 2.0 ** -23), "round-once", "just above a tie"),
                ]
            else:
                conversions += [
                    (target, width, f(2.0 ** -133), "gradual-underflow", "the smallest bf16 subnormal"),
                    (target, width, f(2.0 ** -134), "round-once", "tie: half the smallest bf16 subnormal is +0"),
                    (target, width, f(2.0 ** -134 + 2.0 ** -149), "gradual-underflow",
                     "one f32 ulp above half the smallest bf16 subnormal: rounds up"),
                    (target, width, f(1.0 + 2.0 ** -8), "round-once", "tie: 1 + 2^-8 is 1"),
                    (target, width, f(1.0 + 2.0 ** -7 + 2.0 ** -8), "round-once", "tie: rounds up to even"),
                    (target, width, f(1.0 + 2.0 ** -8 + 2.0 ** -23), "round-once", "just above a tie"),
                ]
        # #3041: within 2^-30 of a midpoint, closer than an f32 rounding step, so a
        # conversion that rounds to f32 first lands on the midpoint and ties wrongly.
        half_ulp = 2.0 ** (-11 if target == "to_f16" else -8)
        conversions += [
            (target, 64, _f64(1.0 + half_ulp + 2.0 ** -30), "round-once", "#3041: just above a tie, below f32 precision"),
            (target, 64, _f64(1.0 + 3 * half_ulp - 2.0 ** -30), "round-once", "#3041: just below a tie, below f32 precision"),
        ]
    for primitive, width, bits, obligation, note in conversions:
        rows.append((primitive, width, [bits], obligation, note))
    return rows


FIXTURE_HEADER = (
    "# Generated by `scripts/vendor_core_math.py fixtures` from MPFR (gmpy2) at the\n"
    "# width's IEEE precision, exponent range, and subnormalization, round to\n"
    "# nearest even; NaN results are [04-NUM-2]'s canonical quiet NaN.\n"
    "# Columns: function width input-bits expected-bits # note\n"
)


def write_fixtures(upstream: Path) -> None:
    gmpy2 = _require_gmpy2()
    manifest = load_manifest()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    CANARY_FIXTURE.write_text(FIXTURE_HEADER + "\n".join(canary_rows(gmpy2)) + "\n", encoding="utf-8")
    header = FIXTURE_HEADER + (
        f"# Sample: {WORST_CASE_SAMPLE} inputs per function from CORE-MATH {manifest.commit}'s\n"
        "# binary64 .wc corpora (seeded by function name), each with its negation.\n"
    )
    WORST_CASE_FIXTURE.write_text(
        header + "\n".join(worst_case_rows(gmpy2, upstream, manifest)) + "\n", encoding="utf-8"
    )
    write_obligations()


def obligation_text() -> str:
    _require_gmpy2()
    return OBLIGATION_HEADER + "\n".join(obligation_rows()) + "\n"


def write_obligations() -> None:
    OBLIGATION_FIXTURE.write_text(obligation_text(), encoding="utf-8")


def check_obligations() -> int:
    current = OBLIGATION_FIXTURE.read_text(encoding="utf-8") if OBLIGATION_FIXTURE.is_file() else None
    if current != obligation_text():
        print(
            f"error: {OBLIGATION_FIXTURE} is out of date with its generator; "
            "run `.venv/bin/python scripts/vendor_core_math.py obligations`",
            file=sys.stderr,
        )
        return 1
    print("chelis-crmath profile obligations are up to date")
    return 0


def resolve_ambiguous(path: Path) -> int:
    """Check each `function f32 input result` line of the exhaustive gate's ambiguous
    list against MPFR; print mismatches and return their count."""
    gmpy2 = _require_gmpy2()
    mismatches = 0
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        function, width_text, input_hex, result_hex = line.split()[:4]
        width = int(width_text.removeprefix("f"))
        expected = mpfr_reference(gmpy2, function, int(input_hex, 16), width)
        checked += 1
        if expected != int(result_hex, 16):
            mismatches += 1
            print(f"MISMATCH {function} f{width} input {input_hex}: kernel {result_hex}, MPFR {expected:x}")
    print(f"resolved {checked} ambiguous inputs against MPFR: {mismatches} mismatches")
    return mismatches


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the amalgamation is not up to date")
    sub = parser.add_subparsers(dest="command")
    imp = sub.add_parser("import", help="copy kernels from a CORE-MATH checkout and re-derive identifiers")
    imp.add_argument("--upstream", type=Path, required=True)
    fix = sub.add_parser("fixtures", help="regenerate the MPFR-derived test fixtures (needs gmpy2)")
    fix.add_argument("--upstream", type=Path, required=True)
    obl = sub.add_parser("obligations", help="regenerate the profile obligation fixture (needs gmpy2)")
    obl.add_argument("--check", action="store_true", dest="check_obligations", help="fail if it is out of date")
    res = sub.add_parser("resolve", help="check the exhaustive gate's ambiguous inputs against MPFR")
    res.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            import_upstream(args.upstream)
            AMALGAMATION.parent.mkdir(parents=True, exist_ok=True)
            AMALGAMATION.write_text(generate(), encoding="utf-8")
            return 0
        if args.command == "fixtures":
            write_fixtures(args.upstream)
            return 0
        if args.command == "obligations":
            if args.check_obligations:
                return check_obligations()
            write_obligations()
            return 0
        if args.command == "resolve":
            return 1 if resolve_ambiguous(args.file) else 0
        text = generate()
        if args.check:
            current = AMALGAMATION.read_text(encoding="utf-8") if AMALGAMATION.is_file() else None
            if current != text:
                print(
                    f"error: {AMALGAMATION} is out of date with the vendored kernels; "
                    "run `.venv/bin/python scripts/vendor_core_math.py`",
                    file=sys.stderr,
                )
                return 1
            print("chelis-crmath amalgamation is up to date")
            return 0
        AMALGAMATION.parent.mkdir(parents=True, exist_ok=True)
        AMALGAMATION.write_text(text, encoding="utf-8")
        return 0
    except VendorError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
