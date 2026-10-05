#!/usr/bin/env python3
"""Unit tests for `scripts/vendor_core_math.py` (chelis#2957).

The rewrite tests feed small C texts through the identifier prefixing and check what
must change (code tokens, macro bodies, include guards) and what must not (comments,
string and character literals, hexadecimal float literals, `#include` lines, longer
identifiers). The repository tests regenerate the checked-in amalgamation and require
it to be byte-identical, then break one input at a time (an amalgamation byte, a
vendored kernel byte, a manifest row) and require the drift to be reported. The
fixture tests check the MPFR-derived fixtures were generated from exactly the inputs
this script names, without needing MPFR.
"""

from __future__ import annotations

import contextlib
import io
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vendor_core_math as vcm  # noqa: E402


class RenameTests(unittest.TestCase):
    def rename(self, text: str, names: set[str]) -> str:
        return vcm.rename_identifiers(text, names, "p__")

    def test_code_tokens_are_prefixed(self):
        text = "static const double tb[4];\nfloat cr_expf(float x){ return tb[0] + x; }\n"
        out = self.rename(text, {"tb", "cr_expf"})
        self.assertEqual(out, "static const double p__tb[4];\nfloat p__cr_expf(float x){ return p__tb[0] + x; }\n")

    def test_comments_and_literals_are_untouched(self):
        text = '/* tb is a table */ // tb again\nconst char *s = "tb"; char c = \'t\'; int tb;\n'
        out = self.rename(text, {"tb"})
        self.assertIn("/* tb is a table */ // tb again", out)
        self.assertIn('"tb"', out)
        self.assertIn("int p__tb;", out)

    def test_hex_float_literals_are_untouched(self):
        # `p` and `e` inside a hexadecimal float are part of the number, not names.
        text = "double a = 0x1.62e42fefa39efp+9 * e + p;\n"
        out = self.rename(text, {"e", "p", "fp"})
        self.assertEqual(out, "double a = 0x1.62e42fefa39efp+9 * p__e + p__p;\n")

    def test_longer_identifiers_are_not_split(self):
        out = self.rename("int tb, tbx, xtb;\n", {"tb"})
        self.assertEqual(out, "int p__tb, tbx, xtb;\n")

    def test_include_lines_are_untouched_but_macros_are_renamed(self):
        text = '#include <x86intrin.h>\n#ifndef DINT_H\n#define DINT_H\n#define CH 0x1p-3\ndouble y = CH;\n'
        out = self.rename(text, {"x86intrin", "DINT_H", "CH"})
        self.assertIn("#include <x86intrin.h>", out)
        self.assertIn("#ifndef p__DINT_H\n#define p__DINT_H", out)
        self.assertIn("#define p__CH 0x1p-3\ndouble y = p__CH;", out)

    def test_macro_names_finds_every_define(self):
        text = "#define A 1\n  #  define B(x) x\n#undef C\n"
        self.assertEqual(vcm.macro_names(text), {"A", "B"})

    def test_undeclared_local_include_is_rejected(self):
        kernel = vcm.Kernel("expf", "exp", 32, "src/binary32/exp/expf.c")
        with self.assertRaisesRegex(vcm.VendorError, "undeclared local include"):
            vcm.inline_local_includes('#include "dint.h"\n', kernel, lambda rel: "")

    def test_declared_local_include_is_inlined(self):
        kernel = next(k for k in vcm.KERNELS if k.local_includes)
        out = vcm.inline_local_includes('#include "dint.h"\nx\n', kernel, lambda rel: f"<{rel}>")
        self.assertIn("<src/binary64/log/dint.h>", out)
        self.assertNotIn('#include "dint.h"', out)


class ContractCleanTests(unittest.TestCase):
    """The reduction to the generated-C contract that `chelis build` units satisfy."""

    def test_portable_arm_replaces_x86_and_errno_conditionals(self):
        text = (
            "a;\n#ifdef __x86_64__\n__m128d r;\n#if defined(__clang__)\nx;\n#else\ny;\n#endif\n"
            "#else\nportable;\n#endif\n#ifdef CORE_MATH_SUPPORT_ERRNO\nerrno = EDOM;\n#endif\nb;\n"
        )
        self.assertEqual(vcm.keep_portable_arms(text), "a;\nportable;\nb;\n")

    def test_rounding_switch_keeps_only_the_nearest_case(self):
        text = (
            "  switch (fegetround()) {\n  case FE_TONEAREST:\n    hi += lo ? md : hi & md;\n    break;\n"
            "  case FE_DOWNWARD:\n    hi += a;\n    break;\n  case FE_UPWARD:\n    hi += b;\n    break;\n  }\n"
        )
        out = vcm.contract_clean(text, "t")
        self.assertIn("  hi += lo ? md : hi & md;\n", out)
        self.assertNotIn("FE_", out)
        self.assertNotIn("fegetround", out)

    def test_attributes_pragmas_raises_and_fenv_are_dropped(self):
        text = (
            "#include <stdint.h>\n#include <fenv.h> // raise\n#pragma STDC FENV_ACCESS ON\n"
            "static __attribute__((cold,noinline)) double f(double x){\n  feraiseexcept(FE_INVALID);\n  return x;\n}\n"
        )
        out = vcm.contract_clean(text, "t")
        self.assertEqual(out, "#include <stdint.h>\nstatic double f(double x){\n  return x;\n}\n")

    def test_surviving_forbidden_construct_is_rejected(self):
        for text, what in [
            ('__asm__("frintn %d0, %d1":"=w"(ix):"w"(x));\n', "__asm"),
            ("static __attribute__((always_inline)) int f(void);\n", "__attribute"),
            ("int m = fegetround();\n", "fegetround"),
            ("int m = FE_UPWARD;\n", "environment macro"),
        ]:
            with self.subTest(what=what), self.assertRaisesRegex(vcm.VendorError, what):
                vcm.contract_clean(text, "t")

    def test_include_outside_the_contract_is_rejected(self):
        with self.assertRaisesRegex(vcm.VendorError, "x86intrin"):
            vcm.contract_clean("#include <x86intrin.h>\n", "t")

    def test_forbidden_words_in_comments_are_allowed(self):
        text = "return 0.0f/0.0f; // to raise FE_INVALID via __asm\n"
        self.assertEqual(vcm.contract_clean(text, "t"), text)

    def test_checked_in_amalgamation_satisfies_the_contract(self):
        text = vcm.AMALGAMATION.read_text(encoding="utf-8")
        code = vcm._code_tokens(text)
        for token in vcm._FORBIDDEN_TOKENS:
            self.assertNotIn(token, code)
        self.assertFalse(any(t.startswith("FE_") for t in code))
        includes = {re.sub(r"\s*//.*$", "", m).split("include", 1)[1].strip()
                    for m in re.findall(r"^[ \t]*#[ \t]*include\b[^\n]*$", text, re.MULTILINE)}
        self.assertLessEqual(includes, vcm.CONTRACT_INCLUDES)
        self.assertNotRegex(text, r"#[ \t]*pragma[ \t]+STDC")

    def test_checked_in_amalgamation_reaches_roundeven_only_through_its_guard(self):
        # GCC 9 and Clang 16 lack `__builtin_roundeven`; one bare call in any kernel
        # breaks every native build on them, because the compiler canary carries all.
        text = vcm.AMALGAMATION.read_text(encoding="utf-8")
        vcm.require_guarded_roundeven(text, "amalgamation")
        self.assertIn("chelis_cr_sin__roundeven_finite (invpi * ax)", text)

    def test_bare_roundeven_is_rejected_and_the_guarded_definition_is_not(self):
        guarded = (
            "/* __builtin_roundeven was introduced in gcc 10 */\n"
            "# define roundeven_finite(x) __builtin_roundeven (x)\n"
        )
        vcm.require_guarded_roundeven(guarded, "t")
        with self.assertRaisesRegex(vcm.VendorError, "outside a guarded roundeven_finite"):
            vcm.require_guarded_roundeven(guarded + "double k = __builtin_roundeven (y);\n", "t")


class RepositoryTests(unittest.TestCase):
    def test_checked_in_amalgamation_is_current(self):
        self.assertEqual(vcm.generate(), vcm.AMALGAMATION.read_text(encoding="utf-8"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vcm.main(["--check"]), 0)

    def test_manifest_round_trips(self):
        self.assertEqual(vcm.render_manifest(vcm.load_manifest()), vcm.MANIFEST.read_text(encoding="utf-8"))

    def test_kernel_matrix_is_the_op46_transcendentals_at_both_widths(self):
        pairs = {(k.function, k.width) for k in vcm.KERNELS}
        self.assertEqual(pairs, {(f, w) for f in ("exp", "log", "sin", "cos", "tan", "atan", "tanh") for w in (32, 64)})

    def test_amalgamation_opens_with_the_guards(self):
        text = vcm.AMALGAMATION.read_text(encoding="utf-8")
        first_kernel = text.index("/* ==== kernel")
        for guard in ("__FAST_MATH__", "__FINITE_MATH_ONLY__", "FLT_EVAL_METHOD != 0"):
            self.assertLess(text.index(guard), first_kernel, guard)

    def test_every_kernel_is_static_prefixed_and_canonicalizing(self):
        text = vcm.AMALGAMATION.read_text(encoding="utf-8")
        code = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.DOTALL)
        for k in vcm.KERNELS:
            inner = k.prefix + k.upstream_entry
            self.assertIn(f"static {k.ctype} {inner}({k.ctype});", text)
            self.assertIn(f"static {k.ctype} {k.entry}({k.ctype} x)", text)
            # No upstream entry name survives unprefixed in code.
            self.assertIsNone(re.search(rf"(?<![\w]){k.upstream_entry}\s*\(", code), k.name)
        self.assertEqual(text.count("return y != y ? chelis_cr_canonical_nan"), len(vcm.KERNELS))
        # Every function definition at file scope is static or carries a static prototype.
        for m in re.finditer(r"^(float|double) (\w+)\(", code, flags=re.MULTILINE):
            self.assertIn(f"static {m.group(1)} {m.group(2)}(", text, m.group(2))


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.vendor = self.tmp / "core-math"
        shutil.copytree(vcm.VENDOR, self.vendor)

    def test_edited_amalgamation_fails_check(self):
        copy = self.tmp / "crmath_amalgamation.c"
        copy.write_text(vcm.AMALGAMATION.read_text(encoding="utf-8").replace("0x7fc00000u", "0x7fc00001u"))
        with mock.patch.object(vcm, "AMALGAMATION", copy), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(vcm.main(["--check"]), 1)
        self.assertIn("out of date", err.getvalue())

    def test_missing_amalgamation_fails_check(self):
        with mock.patch.object(vcm, "AMALGAMATION", self.tmp / "absent.c"), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(vcm.main(["--check"]), 1)

    def test_edited_vendored_kernel_is_rejected(self):
        kernel = self.vendor / vcm.KERNELS[0].path
        kernel.write_bytes(kernel.read_bytes() + b"\n")
        with self.assertRaisesRegex(vcm.VendorError, "sha256"):
            vcm.generate(vendor=self.vendor, manifest_path=self.vendor / "VENDOR.toml")

    def test_missing_vendored_header_is_rejected(self):
        (self.vendor / "src/binary64/log/dint.h").unlink()
        with self.assertRaisesRegex(vcm.VendorError, "missing"):
            vcm.generate(vendor=self.vendor, manifest_path=self.vendor / "VENDOR.toml")

    def test_manifest_without_a_kernel_is_rejected(self):
        manifest = self.vendor / "VENDOR.toml"
        text = manifest.read_text(encoding="utf-8")
        start = text.index('[[kernel]]\nname = "tanh"')
        end = text.index("\n\n", start)
        manifest.write_text(text[:start] + text[end + 2 :], encoding="utf-8")
        with self.assertRaisesRegex(vcm.VendorError, "no identifiers"):
            vcm.generate(vendor=self.vendor, manifest_path=manifest)

    def test_identifier_list_without_the_entry_is_rejected(self):
        manifest = self.vendor / "VENDOR.toml"
        text = manifest.read_text(encoding="utf-8").replace(', "cr_expf"', "", 1)
        manifest.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(vcm.VendorError, "entry cr_expf missing"):
            vcm.generate(vendor=self.vendor, manifest_path=manifest)

    def test_dropped_identifier_changes_the_output(self):
        # A name missing from the manifest stays unprefixed, so --check would see it.
        manifest = self.vendor / "VENDOR.toml"
        text = manifest.read_text(encoding="utf-8").replace('"b32u32_u", ', "", 1)
        manifest.write_text(text, encoding="utf-8")
        regenerated = vcm.generate(vendor=self.vendor, manifest_path=manifest)
        self.assertNotEqual(regenerated, vcm.AMALGAMATION.read_text(encoding="utf-8"))


def fixture_rows(path: Path) -> list[tuple[str, int, int, int, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        data, _, note = line.partition("#")
        function, width, bits, expected = data.split()
        rows.append((function, int(width[1:]), int(bits, 16), int(expected, 16), note.strip()))
    return rows


class FixtureTests(unittest.TestCase):
    def test_canary_rows_are_exactly_the_named_inputs(self):
        rows = fixture_rows(vcm.CANARY_FIXTURE)
        self.assertEqual([(f, w, b, n) for f, w, b, _, n in rows], vcm.canary_inputs())

    def test_canary_nan_inputs_expect_the_canonical_nan(self):
        for function, width, bits, expected, note in fixture_rows(vcm.CANARY_FIXTURE):
            if "NaN" in note:
                self.assertEqual(expected, vcm.CANONICAL_NAN[width], f"{function} f{width} {note}")

    def test_canary_carries_the_issue_witnesses(self):
        rows = fixture_rows(vcm.CANARY_FIXTURE)
        # #2952: the correctly rounded expf(-1.72889078f) is 0x3e35bda0.
        self.assertIn(("exp", 32, vcm.float_to_bits(-1.72889078, 32), 0x3E35BDA0, "#2952: macOS expf misrounds"), rows)
        notes = " ".join(n for *_, n in rows)
        for issue in ("#2952", "#2959", "#2971", vcm.FUNNEL_NOTE):
            self.assertIn(issue, notes)

    def test_worst_case_sample_is_balanced(self):
        rows = fixture_rows(vcm.WORST_CASE_FIXTURE)
        for function in vcm.FUNCTIONS:
            own = [r for r in rows if r[0] == function]
            self.assertEqual(len(own), 2 * vcm.WORST_CASE_SAMPLE, function)
            self.assertTrue(all(w == 64 for _, w, *_ in own))

    def test_bit_conversions_round_trip(self):
        for width, bits in ((32, 0x3E35BDA0), (32, 0x80000001), (64, 0xBFD93179431561A0)):
            self.assertEqual(vcm.float_to_bits(vcm.bits_to_float(bits, width), width), bits)


if __name__ == "__main__":
    unittest.main()
