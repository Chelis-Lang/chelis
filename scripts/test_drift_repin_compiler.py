"""Unit tests for `drift_repin_compiler.py`.

Run via: `python3 -m unittest scripts.test_drift_repin_compiler` from repo
root, or `python3 scripts/test_drift_repin_compiler.py`.

The script is a canary-only helper that rewrites a shell's reef.toml
compiler pin to a target chelis version. The tests assert it rewrites the
pin, is idempotent, preserves the equality marker and trailing comments,
touches only the compiler line (not the package/dependency versions), and
fails loudly when no equality pin is present.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "drift_repin_compiler", here / "drift_repin_compiler.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


drc = _load_module()


class RepinTests(unittest.TestCase):
    def test_rewrites_older_pin(self):
        src = '[package]\nname = "hull"\ncompiler = "=0.7.26"\n'
        out = drc.repin(src, "0.7.27")
        self.assertIn('compiler = "=0.7.27"', out)
        self.assertNotIn("0.7.26", out)

    def test_idempotent_when_already_target(self):
        src = 'compiler = "=0.7.27"\n'
        self.assertEqual(drc.repin(src, "0.7.27"), src)

    def test_preserves_equality_marker_and_trailing_content(self):
        src = 'compiler = "=0.7.26"  # bump in lockstep\n'
        out = drc.repin(src, "0.8.0")
        self.assertEqual(out, 'compiler = "=0.8.0"  # bump in lockstep\n')

    def test_only_touches_compiler_line(self):
        src = (
            "[package]\n"
            'version = "0.7.26"\n'
            'compiler = "=0.7.26"\n'
            "[dependencies]\n"
            'nautilus = { version = "0.7.26" }\n'
        )
        out = drc.repin(src, "0.7.27")
        # Package version and dependency version must NOT be rewritten;
        # only the compiler equality pin changes.
        self.assertIn('version = "0.7.26"', out)
        self.assertIn('nautilus = { version = "0.7.26" }', out)
        self.assertIn('compiler = "=0.7.27"', out)

    def test_rejects_missing_pin(self):
        with self.assertRaisesRegex(ValueError, "no parseable"):
            drc.repin('[package]\nname = "x"\n', "0.7.27")

    def test_rejects_non_equality_pin(self):
        # reef pins are hard equalities; a range like ">=0.7.0" is not a
        # canary-rewritable pin and must be reported, not silently ignored.
        with self.assertRaisesRegex(ValueError, "no parseable"):
            drc.repin('compiler = ">=0.7.0"\n', "0.7.27")


class MainTests(unittest.TestCase):
    def test_writes_file_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "reef.toml"
            p.write_text('compiler = "=0.7.26"\n', encoding="utf-8")
            rc = drc.main(["drift_repin_compiler.py", str(p), "0.7.27"])
            self.assertEqual(rc, 0)
            self.assertIn('compiler = "=0.7.27"', p.read_text(encoding="utf-8"))

    def test_returns_one_on_missing_pin(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "reef.toml"
            p.write_text('name = "x"\n', encoding="utf-8")
            rc = drc.main(["drift_repin_compiler.py", str(p), "0.7.27"])
            self.assertEqual(rc, 1)

    def test_returns_two_on_bad_args(self):
        self.assertEqual(drc.main(["drift_repin_compiler.py"]), 2)

    def test_returns_one_on_missing_path(self):
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / "nope" / "reef.toml"
            rc = drc.main(["drift_repin_compiler.py", str(missing), "0.7.27"])
            self.assertEqual(rc, 1)


class DirTreeTests(unittest.TestCase):
    """A shell checkout is a tree of nested reef packages, not one manifest.

    These lock in that a directory target repins EVERY `reef.toml` under it
    (root + nested), skips pinless manifests without failing, and fails
    loudly when a run would repin nothing.
    """

    @staticmethod
    def _write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_repins_root_and_nested(self):
        with tempfile.TemporaryDirectory() as d:
            shell = Path(d)
            self._write(shell / "reef.toml", 'compiler = "=0.7.26"\n')
            self._write(
                shell / "examples" / "demo" / "reef.toml",
                'compiler = "=0.7.26"\n',
            )
            self._write(
                shell / "spike" / "probes" / "reef.toml",
                'compiler = "=0.7.26"  # probe\n',
            )
            rc = drc.main(["drift_repin_compiler.py", str(shell), "0.7.27"])
            self.assertEqual(rc, 0)
            for rel in (
                "reef.toml",
                "examples/demo/reef.toml",
                "spike/probes/reef.toml",
            ):
                text = (shell / rel).read_text(encoding="utf-8")
                self.assertIn('compiler = "=0.7.27"', text, rel)
                self.assertNotIn("0.7.26", text, rel)

    def test_repins_all_workflow_compiler_env_pins(self):
        with tempfile.TemporaryDirectory() as d:
            shell = Path(d)
            self._write(shell / "reef.toml", 'compiler = "=0.17.1"\n')
            ci = shell / ".github" / "workflows" / "ci.yml"
            release = shell / ".github" / "workflows" / "release.yml"
            self._write(
                ci,
                "env:\n"
                "  CHELIS_TAG: v0.17.1\n"
                "  CHELIS_VERSION: 0.17.1\n"
                "  SHOALS_VERSION: 0.24.1\n"
                "jobs:\n"
                "  nested:\n"
                "    env:\n"
                '      CHELIS_TAG: "v0.17.1"\n'
                "      CHELIS_VERSION: '0.17.1'\n",
            )
            self._write(
                release,
                "env:\n"
                "  CHELIS_TAG : 'v0.17.1' # release compiler\n"
                '  CHELIS_VERSION : "0.17.1" # release compiler\n',
            )

            rc = drc.main(["drift_repin_compiler.py", str(shell), "0.17.4"])

            self.assertEqual(rc, 0)
            for workflow in (ci, release):
                text = workflow.read_text(encoding="utf-8")
                self.assertNotIn("0.17.1", text)
                self.assertIn("0.17.4", text)
            ci_text = ci.read_text(encoding="utf-8")
            self.assertEqual(ci_text.count("CHELIS_TAG"), 2)
            self.assertEqual(ci_text.count("v0.17.4"), 2)
            self.assertEqual(ci_text.count("CHELIS_VERSION"), 2)
            self.assertIn("SHOALS_VERSION: 0.24.1", ci_text)

    def test_workflow_repin_is_idempotent_and_ignores_other_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            shell = Path(d)
            self._write(shell / "reef.toml", 'compiler = "=0.17.4"\n')
            workflow = shell / ".github" / "workflows" / "ci.yaml"
            unrelated = shell / "config.yml"
            source = (
                "env:\n"
                "  CHELIS_TAG: v0.17.4\n"
                "  CHELIS_VERSION: 0.17.4\n"
            )
            self._write(workflow, source)
            self._write(
                unrelated,
                "CHELIS_TAG: v0.17.1\nCHELIS_VERSION: 0.17.1\n",
            )

            self.assertEqual(
                drc.main(["drift_repin_compiler.py", str(shell), "0.17.4"]), 0
            )
            self.assertEqual(workflow.read_text(encoding="utf-8"), source)
            self.assertIn("0.17.1", unrelated.read_text(encoding="utf-8"))

    def test_skips_pinless_manifest_but_repins_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            shell = Path(d)
            self._write(shell / "reef.toml", 'compiler = "=0.7.26"\n')
            # A workspace-style / negative-fixture manifest with no pin must
            # be tolerated (skipped), not fail the whole tree.
            self._write(
                shell / "spike" / "callsite_bad" / "reef.toml",
                'name = "callsite_bad"\n',
            )
            repinned, skipped = drc.repin_tree(shell, "0.7.27")
            self.assertEqual((repinned, skipped), (1, 1))
            rc = drc.main(["drift_repin_compiler.py", str(shell), "0.7.27"])
            self.assertEqual(rc, 0)
            self.assertIn(
                "callsite_bad",
                (shell / "spike" / "callsite_bad" / "reef.toml").read_text(
                    encoding="utf-8"
                ),
            )

    def test_only_touches_compiler_line_in_nested(self):
        with tempfile.TemporaryDirectory() as d:
            shell = Path(d)
            self._write(shell / "reef.toml", 'compiler = "=0.7.26"\n')
            nested = shell / "examples" / "demo" / "reef.toml"
            self._write(
                nested,
                "[package]\n"
                'version = "0.7.26"\n'
                'compiler = "=0.7.26"\n'
                "[dependencies]\n"
                'nautilus = { version = "0.7.26" }\n',
            )
            drc.main(["drift_repin_compiler.py", str(shell), "0.7.27"])
            text = nested.read_text(encoding="utf-8")
            self.assertIn('version = "0.7.26"', text)
            self.assertIn('nautilus = { version = "0.7.26" }', text)
            self.assertIn('compiler = "=0.7.27"', text)

    def test_idempotent_on_second_run(self):
        with tempfile.TemporaryDirectory() as d:
            shell = Path(d)
            self._write(shell / "reef.toml", 'compiler = "=0.7.27"\n')
            self.assertEqual(
                drc.main(["drift_repin_compiler.py", str(shell), "0.7.27"]), 0
            )
            self.assertEqual(
                drc.main(["drift_repin_compiler.py", str(shell), "0.7.27"]), 0
            )

    def test_fails_loudly_when_no_reef_toml(self):
        with tempfile.TemporaryDirectory() as d:
            shell = Path(d)
            self._write(shell / "src" / "main.ch", "x\n")
            rc = drc.main(["drift_repin_compiler.py", str(shell), "0.7.27"])
            self.assertEqual(rc, 1)

    def test_fails_loudly_when_nothing_repinnable(self):
        with tempfile.TemporaryDirectory() as d:
            shell = Path(d)
            self._write(shell / "reef.toml", 'name = "workspace-only"\n')
            rc = drc.main(["drift_repin_compiler.py", str(shell), "0.7.27"])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
