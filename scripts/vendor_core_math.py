#!/usr/bin/env python3
"""Vendor CORE-MATH kernels and generate the `chelis-crmath` amalgamation (chelis#2957).

`spec/design/correctly_rounded_math.md` section 3.3 owns the design. The fourteen
upstream kernel files under `crates/chelis-crmath/vendor/core-math/` are kept
byte-for-byte as upstream ships them; `VENDOR.toml` beside them records the
upstream commit, each file's SHA-256, and the file-scope identifiers of each
kernel, taken from Clang AST dumps when the kernels were imported. From those
inputs alone (no compiler needed) this script generates
`crates/chelis-crmath/csrc/crmath_amalgamation.c`, the one translation-unit text
every lane compiles:

1. every file-scope identifier and macro of a kernel gets the kernel's prefix
   (`chelis_cr_expf__`), so the fourteen kernels coexist in one unit;
2. the kernel's external entry (`cr_expf`) is declared `static` before its
   definition, so every definition has internal linkage;
3. a `static` entry `chelis_cr_<name>` calls the kernel and replaces any NaN
   result with [04-NUM-2]'s canonical quiet NaN;
4. the kernel text is reduced to the generated-C contract that built programs
   must satisfy, because `chelis build` emits these same bytes into every unit
   that calls a kernel (design section 4.2). `contract_clean` drops what the
   contract forbids and what Chelis never observes: the `errno` blocks, the
   floating-point exception raises, `<fenv.h>`, the `FENV_ACCESS` pragma, the
   `noinline`/`cold` attributes, the inline-assembly `roundeven` arms (the
   portable fallback beside them stays), and the x86-64 SSE intrinsic arms (the
   portable arm beside each stays). The dynamic rounding-mode switch keeps only
   its round-to-nearest case, which Chelis pins at every entry (design section
   6). Values are unchanged: every dropped arm computes the same bits as the
   arm that stays.

The amalgamation opens with `#error` guards against fast math, finite-math-only,
and excess-precision evaluation.

Usage (from the repository root):

    .venv/bin/python scripts/vendor_core_math.py            # regenerate the amalgamation
    .venv/bin/python scripts/vendor_core_math.py --check    # fail on any drift
    .venv/bin/python scripts/vendor_core_math.py import --upstream DIR
        # copy kernels from a CORE-MATH checkout, re-derive identifiers with clang
    .venv/bin/python scripts/vendor_core_math.py fixtures --upstream DIR
        # regenerate the MPFR-derived test fixtures (needs gmpy2)
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
FUNCTIONS = ("exp", "log", "sin", "cos", "tan", "atan", "tanh")


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
    "# if (defined(__GNUC__) || defined(__clang__)) && (defined(__AVX__) || defined(__SSE4_1__) || (__ARM_ARCH >= 8))",
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
_DROPPED_LINES = re.compile(
    r"^[ \t]*(?:#[ \t]*pragma[ \t]+STDC[ \t]+FENV_ACCESS[ \t]+ON|#[ \t]*include[ \t]*<fenv\.h>)[^\n]*\n",
    re.MULTILINE,
)
# What the reduced text may still include; `generated_header.rs` admits each.
CONTRACT_INCLUDES = {"<float.h>", "<inttypes.h>", "<stdint.h>", "<stdio.h>", "<string.h>"}
_FORBIDDEN_TOKENS = ("__attribute", "__attribute__", "__asm", "__asm__", "asm", "_Pragma", "__declspec", "fegetround",
                     "feraiseexcept", "errno", "_mm_setcsr")


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


def contract_clean(text: str, origin: str) -> str:
    """Reduce one kernel's text to the generated-C contract (module docstring, item 4)."""
    text = keep_portable_arms(text)
    text = _ROUNDING_SWITCH.sub(lambda m: _reindent(m.group("body"), m.group(1)), text)
    text = _RAISE.sub("", text)
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
 * Correctly rounded exp, log, sin, cos, tan, atan, and tanh at binary32 and binary64
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
    text = contract_clean(inline_local_includes(raw, kernel, read), kernel.path)
    rename = set(names)
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
              "tan": gmpy2.tan, "atan": gmpy2.atan, "tanh": gmpy2.tanh}[function]
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
