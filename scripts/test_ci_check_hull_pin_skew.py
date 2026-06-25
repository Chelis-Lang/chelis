"""Unit tests for `ci_check_hull_pin_skew.py`.

Run via: `python3 -m unittest scripts.test_ci_check_hull_pin_skew` from repo
root, or `python3 scripts/test_ci_check_hull_pin_skew.py`.

The check compares the live chelis binary version against the compiler pin in
the checked-out Hull repo's reef.toml. The tests cover the parsers, the
match/skew decision (including the load-bearing invariant that a skew still
returns nonzero -- it must NOT mask the real failure), and that the campaign
can only run when the versions are byte-equal.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "ci_check_hull_pin_skew", here / "ci_check_hull_pin_skew.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cps = _load_module()


REEF_TOML = """\
[package]
name = "hull"
version = "0.1.4"
compiler = "=0.7.27"
module_prefix = "Hull"
"""


class PinParserTests(unittest.TestCase):
    def test_parses_compiler_pin(self):
        self.assertEqual(cps.parse_hull_compiler_pin(REEF_TOML), "0.7.27")

    def test_parses_pin_with_extra_spacing(self):
        text = 'compiler   =   "=0.10.0"\n'
        self.assertEqual(cps.parse_hull_compiler_pin(text), "0.10.0")

    def test_missing_pin_raises(self):
        with self.assertRaises(ValueError):
            cps.parse_hull_compiler_pin('[package]\nname = "hull"\n')

    def test_a_dependency_compiler_key_is_not_mistaken_for_the_pin(self):
        # The pin regex anchors `compiler` at the start of a line (after
        # optional whitespace), so a value that merely contains the word
        # is not matched as a pin.
        text = 'name = "compiler-helper"\n'
        with self.assertRaises(ValueError):
            cps.parse_hull_compiler_pin(text)


class VersionParserTests(unittest.TestCase):
    def test_parses_chelis_version(self):
        self.assertEqual(cps.parse_chelis_version("chelis 0.10.0\n"), "0.10.0")

    def test_parses_version_with_trailing_text(self):
        self.assertEqual(
            cps.parse_chelis_version("chelis 0.9.0 (build abc123)"), "0.9.0"
        )

    def test_unparseable_version_raises(self):
        with self.assertRaises(ValueError):
            cps.parse_chelis_version("not a version line")


class QueryVersionTests(unittest.TestCase):
    def test_runs_binary_and_parses_stdout(self):
        run = mock.Mock(
            return_value=mock.Mock(returncode=0, stdout="chelis 0.10.0\n", stderr="")
        )
        self.assertEqual(cps.query_chelis_version("/bin/chelis", run=run), "0.10.0")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["/bin/chelis", "--version"])

    def test_nonzero_exit_raises(self):
        run = mock.Mock(
            return_value=mock.Mock(returncode=1, stdout="", stderr="boom")
        )
        with self.assertRaises(ValueError):
            cps.query_chelis_version("/bin/chelis", run=run)


class SkewDecisionTests(unittest.TestCase):
    def test_match_returns_zero(self):
        self.assertEqual(cps.check_skew("0.10.0", "0.10.0"), 0)

    def test_skew_returns_nonzero(self):
        # The load-bearing invariant: a version skew must NOT be turned green.
        # The diagnostic replaces an opaque crash but the nightly stays red.
        self.assertEqual(cps.check_skew("0.10.0", "0.8.0"), 1)

    def test_skew_message_names_both_versions_and_the_cascade_fix(self):
        with mock.patch.object(cps.sys, "stderr") as err:
            cps.check_skew("0.10.0", "0.8.0")
        printed = "".join(c.args[0] for c in err.write.call_args_list if c.args)
        # The operator must be able to act on it: both versions + the fix.
        self.assertIn("0.10.0", printed)
        self.assertIn("0.8.0", printed)
        self.assertIn("hull_commit", printed)
        self.assertIn("cascade", printed.lower())

    def test_exact_string_compare_not_semver_range(self):
        # `chelis reef build` requires byte-exact equality; 0.10.0 does NOT
        # satisfy a =0.1.0 pin even though 10 > 1 numerically-by-component.
        self.assertEqual(cps.check_skew("0.10.0", "0.1.0"), 1)


class MainTests(unittest.TestCase):
    def _write(self, tmp, name, text):
        p = Path(tmp) / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_main_match_exit_zero(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            reef = self._write(tmp, "reef.toml", 'compiler = "=0.10.0"\n')
            with mock.patch.object(cps, "query_chelis_version", return_value="0.10.0"):
                rc = cps.main(["--chelis-bin", "/bin/chelis", "--hull-reef-toml", reef])
        self.assertEqual(rc, 0)

    def test_main_skew_exit_one(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            reef = self._write(tmp, "reef.toml", 'compiler = "=0.8.0"\n')
            with mock.patch.object(cps, "query_chelis_version", return_value="0.10.0"):
                rc = cps.main(["--chelis-bin", "/bin/chelis", "--hull-reef-toml", reef])
        self.assertEqual(rc, 1)

    def test_main_unparseable_pin_exit_two(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            reef = self._write(tmp, "reef.toml", "[package]\nname = \"hull\"\n")
            with mock.patch.object(cps, "query_chelis_version", return_value="0.10.0"):
                rc = cps.main(["--chelis-bin", "/bin/chelis", "--hull-reef-toml", reef])
        self.assertEqual(rc, 2)

    def test_main_binary_version_failure_exit_two(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            reef = self._write(tmp, "reef.toml", 'compiler = "=0.10.0"\n')
            with mock.patch.object(
                cps, "query_chelis_version", side_effect=ValueError("no binary")
            ):
                rc = cps.main(["--chelis-bin", "/bin/chelis", "--hull-reef-toml", reef])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
