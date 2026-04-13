"""Tests for prove.py.

Run with: python3 -m unittest proof/scripts/test_prove.py

No network or external tool calls. urllib and subprocess are mocked.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import prove  # noqa: E402


SAMPLE_FILE = """\
-- header comment
import Mathlib

theorem add_comm_nat (a b : Nat) : a + b = b + a := by
  sorry

theorem other (n : Nat) : n = n := by rfl
"""


def _make_args(**kw):
    """Build a minimal argparse.Namespace used by run_one_pass."""
    import argparse

    base = dict(
        file=Path("/tmp/none.lean"),
        theorem="t",
        passes=1,
        hint=None,
        model="labs-leanstral-2603",
        temperature=1.0,
        max_tokens=8000,
        timeout=240,
        dry_run=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


class TestParseArgs(unittest.TestCase):
    def test_parse_args_valid(self):
        ns = prove.parse_args(
            [
                "--file",
                "/tmp/x.lean",
                "--theorem",
                "foo",
                "--passes",
                "3",
                "--hint",
                "try induction",
                "--model",
                "m",
                "--temperature",
                "0.5",
                "--max-tokens",
                "1024",
                "--timeout",
                "30",
                "--dry-run",
            ]
        )
        self.assertEqual(ns.file, Path("/tmp/x.lean"))
        self.assertEqual(ns.theorem, "foo")
        self.assertEqual(ns.passes, 3)
        self.assertEqual(ns.hint, "try induction")
        self.assertEqual(ns.model, "m")
        self.assertAlmostEqual(ns.temperature, 0.5)
        self.assertEqual(ns.max_tokens, 1024)
        self.assertEqual(ns.timeout, 30)
        self.assertTrue(ns.dry_run)

    def test_parse_args_missing_file(self):
        with self.assertRaises(SystemExit):
            prove.parse_args(["--theorem", "foo"])

    def test_parse_args_invalid_passes(self):
        with self.assertRaises(SystemExit):
            prove.parse_args(["--file", "/x", "--theorem", "t", "--passes", "0"])
        with self.assertRaises(SystemExit):
            prove.parse_args(["--file", "/x", "--theorem", "t", "--passes", "-2"])


class TestFindProjectRoot(unittest.TestCase):
    def test_find_project_root_found(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "lakefile.toml").write_text("name = \"x\"\n")
            sub = root / "a" / "b"
            sub.mkdir(parents=True)
            lean_file = sub / "X.lean"
            lean_file.write_text("-- x\n")
            self.assertEqual(prove.find_project_root(lean_file).resolve(), root.resolve())

    def test_find_project_root_lakefile_lean(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "lakefile.lean").write_text("-- lake\n")
            sub = root / "nested"
            sub.mkdir()
            self.assertEqual(prove.find_project_root(sub).resolve(), root.resolve())

    def test_find_project_root_not_found(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                prove.find_project_root(Path(d))


class TestExtractContext(unittest.TestCase):
    def test_extract_context_basic(self):
        ctx = prove.extract_context(SAMPLE_FILE, "add_comm_nat")
        self.assertEqual(ctx.theorem_name, "add_comm_nat")
        lines = SAMPLE_FILE.splitlines()
        self.assertTrue(lines[ctx.theorem_line].startswith("theorem add_comm_nat"))
        self.assertIn("sorry", lines[ctx.sorry_line])
        self.assertEqual(ctx.indent, "  ")
        self.assertIn("add_comm_nat", ctx.context_snippet)
        self.assertIn("sorry", ctx.context_snippet)

    def test_extract_context_missing_theorem(self):
        with self.assertRaises(ValueError):
            prove.extract_context(SAMPLE_FILE, "nope")

    def test_extract_context_no_sorry(self):
        with self.assertRaises(ValueError):
            prove.extract_context(SAMPLE_FILE, "other")


class TestBuildPrompt(unittest.TestCase):
    def test_build_prompt_includes_theorem_and_hint(self):
        ctx = prove.extract_context(SAMPLE_FILE, "add_comm_nat")
        messages = prove.build_prompt(ctx, hint="use Nat.add_comm")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        user = messages[1]["content"]
        self.assertIn("add_comm_nat", user)
        self.assertIn("use Nat.add_comm", user)
        self.assertIn("```lean", user)

    def test_build_prompt_no_hint(self):
        ctx = prove.extract_context(SAMPLE_FILE, "add_comm_nat")
        messages = prove.build_prompt(ctx, hint=None)
        self.assertNotIn("Hint:", messages[1]["content"])


class TestExtractProof(unittest.TestCase):
    def test_extract_proof_fenced_lean(self):
        r = "sure thing:\n```lean\nby exact foo\n```\ndone"
        self.assertEqual(prove.extract_proof_from_response(r), "by exact foo")

    def test_extract_proof_fenced_lean4(self):
        r = "```lean4\nby\n  exact foo\n```"
        self.assertEqual(prove.extract_proof_from_response(r), "by\n  exact foo")

    def test_extract_proof_any_fence(self):
        r = "```\nby trivial\n```"
        self.assertEqual(prove.extract_proof_from_response(r), "by trivial")

    def test_extract_proof_unfenced(self):
        r = "  by exact foo  "
        self.assertEqual(prove.extract_proof_from_response(r), "by exact foo")

    def test_extract_proof_empty_response(self):
        self.assertEqual(prove.extract_proof_from_response(""), "")
        self.assertEqual(prove.extract_proof_from_response("   \n"), "")


class TestPatchFile(unittest.TestCase):
    def test_patch_file_replaces_sorry_preserving_indent(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "X.lean"
            p.write_text(SAMPLE_FILE)
            ctx = prove.extract_context(SAMPLE_FILE, "add_comm_nat")
            original = prove.patch_file(p, ctx, "exact Nat.add_comm a b")
            new = p.read_text()
            self.assertEqual(original, SAMPLE_FILE)
            self.assertIn("  exact Nat.add_comm a b", new)
            self.assertNotIn("  sorry", new)
            # Other theorem (rfl on same line) should still be intact.
            self.assertIn("theorem other", new)

    def test_patch_file_multiline_proof(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "X.lean"
            p.write_text(SAMPLE_FILE)
            ctx = prove.extract_context(SAMPLE_FILE, "add_comm_nat")
            prove.patch_file(p, ctx, "rw [Nat.add_comm]\nrfl")
            new = p.read_text()
            self.assertIn("  rw [Nat.add_comm]", new)
            self.assertIn("  rfl", new)

    def test_patch_file_restores_on_demand(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "X.lean"
            p.write_text(SAMPLE_FILE)
            ctx = prove.extract_context(SAMPLE_FILE, "add_comm_nat")
            original = prove.patch_file(p, ctx, "whatever")
            self.assertNotEqual(p.read_text(), original)
            prove.restore_file(p, original)
            self.assertEqual(p.read_text(), SAMPLE_FILE)


def _canned_response(content: str) -> bytes:
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    ).encode("utf-8")


class _FakeResp:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestCallLeanstral(unittest.TestCase):
    def test_call_leanstral_happy_path_mocked(self):
        fake = _FakeResp(_canned_response("```lean\nby trivial\n```"))
        with mock.patch("prove.urllib.request.urlopen", return_value=fake) as m:
            out = prove.call_leanstral(
                [{"role": "user", "content": "hi"}],
                model="m",
                temperature=1.0,
                max_tokens=10,
                timeout=5,
                api_key="fake-key",
            )
        self.assertIn("by trivial", out)
        req = m.call_args[0][0]
        self.assertEqual(req.get_full_url(), prove.API_URL)
        self.assertEqual(req.headers["Authorization"], "Bearer fake-key")

    def test_call_leanstral_http_error(self):
        err = urllib.error.HTTPError(
            prove.API_URL, 500, "Server Error", {}, io.BytesIO(b"boom")
        )
        with mock.patch("prove.urllib.request.urlopen", side_effect=err):
            with self.assertRaises(prove.LeanstralAPIError):
                prove.call_leanstral(
                    [{"role": "user", "content": "hi"}],
                    model="m",
                    temperature=1.0,
                    max_tokens=10,
                    timeout=5,
                    api_key="fake-key",
                )

    def test_call_leanstral_url_error(self):
        err = urllib.error.URLError("no dns")
        with mock.patch("prove.urllib.request.urlopen", side_effect=err):
            with self.assertRaises(prove.LeanstralAPIError):
                prove.call_leanstral(
                    [{"role": "user", "content": "hi"}],
                    model="m",
                    temperature=1.0,
                    max_tokens=10,
                    timeout=5,
                    api_key="fake-key",
                )

    def test_call_leanstral_missing_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(prove.LeanstralAPIError):
                prove.call_leanstral(
                    [], model="m", temperature=1.0, max_tokens=10, timeout=5
                )


class TestRunOnePass(unittest.TestCase):
    def _setup(self, d: str):
        root = Path(d)
        (root / "lakefile.toml").write_text('name = "x"\n')
        lean = root / "X.lean"
        lean.write_text(SAMPLE_FILE)
        ctx = prove.extract_context(SAMPLE_FILE, "add_comm_nat")
        args = _make_args(file=lean)
        return root, lean, ctx, args

    def test_run_one_pass_success_mocked(self):
        with tempfile.TemporaryDirectory() as d:
            root, lean, ctx, args = self._setup(d)
            with mock.patch(
                "prove.call_leanstral",
                return_value="```lean\nexact Nat.add_comm a b\n```",
            ), mock.patch("prove.run_lake_build", return_value=(0, "ok")):
                result = prove.run_one_pass(args, ctx, root)
        self.assertTrue(result.success)
        self.assertIn("Nat.add_comm", result.proof)

    def test_run_one_pass_lake_build_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root, lean, ctx, args = self._setup(d)
            with mock.patch(
                "prove.call_leanstral",
                return_value="```lean\nexact False.elim sorry\n```",
            ), mock.patch("prove.run_lake_build", return_value=(1, "nope")):
                result = prove.run_one_pass(args, ctx, root)
            after = lean.read_text()
        self.assertFalse(result.success)
        self.assertFalse(result.api_error)
        # File restored.
        self.assertEqual(after, SAMPLE_FILE)

    def test_run_one_pass_api_error(self):
        with tempfile.TemporaryDirectory() as d:
            root, lean, ctx, args = self._setup(d)
            with mock.patch(
                "prove.call_leanstral",
                side_effect=prove.LeanstralAPIError("HTTP 500"),
            ):
                result = prove.run_one_pass(args, ctx, root)
            after = lean.read_text()
        self.assertFalse(result.success)
        self.assertTrue(result.api_error)
        # File untouched because we never patched.
        self.assertEqual(after, SAMPLE_FILE)


class TestMain(unittest.TestCase):
    def _project(self, d: str) -> Path:
        root = Path(d)
        (root / "lakefile.toml").write_text('name = "x"\n')
        lean = root / "X.lean"
        lean.write_text(SAMPLE_FILE)
        return lean

    def test_main_missing_api_key_exit_3(self):
        with tempfile.TemporaryDirectory() as d:
            lean = self._project(d)
            argv = ["--file", str(lean), "--theorem", "add_comm_nat"]
            with mock.patch.dict("os.environ", {}, clear=True):
                rc = prove.main(argv)
        self.assertEqual(rc, 3)

    def test_main_all_passes_fail_exit_1(self):
        with tempfile.TemporaryDirectory() as d:
            lean = self._project(d)
            argv = [
                "--file",
                str(lean),
                "--theorem",
                "add_comm_nat",
                "--passes",
                "2",
            ]
            fail = prove.PassResult(success=False, error="lake exit 1")
            with mock.patch.dict(
                "os.environ", {"MISTRAL_API_KEY": "k"}, clear=True
            ), mock.patch("prove.run_one_pass", return_value=fail):
                rc = prove.main(argv)
        self.assertEqual(rc, 1)

    def test_main_second_pass_succeeds_exit_0(self):
        with tempfile.TemporaryDirectory() as d:
            lean = self._project(d)
            argv = [
                "--file",
                str(lean),
                "--theorem",
                "add_comm_nat",
                "--passes",
                "3",
            ]
            results = [
                prove.PassResult(success=False, error="bad"),
                prove.PassResult(success=True, proof="exact Nat.add_comm a b"),
            ]
            with mock.patch.dict(
                "os.environ", {"MISTRAL_API_KEY": "k"}, clear=True
            ), mock.patch("prove.run_one_pass", side_effect=results):
                rc = prove.main(argv)
        self.assertEqual(rc, 0)

    def test_main_api_errors_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            lean = self._project(d)
            argv = [
                "--file",
                str(lean),
                "--theorem",
                "add_comm_nat",
                "--passes",
                "2",
            ]
            api_fail = prove.PassResult(
                success=False, error="HTTP 500", api_error=True
            )
            with mock.patch.dict(
                "os.environ", {"MISTRAL_API_KEY": "k"}, clear=True
            ), mock.patch("prove.run_one_pass", return_value=api_fail):
                rc = prove.main(argv)
        self.assertEqual(rc, 2)

    def test_main_dry_run_no_api(self):
        with tempfile.TemporaryDirectory() as d:
            lean = self._project(d)
            argv = [
                "--file",
                str(lean),
                "--theorem",
                "add_comm_nat",
                "--dry-run",
            ]
            buf = io.StringIO()
            with mock.patch.dict("os.environ", {}, clear=True), mock.patch(
                "sys.stdout", buf
            ):
                rc = prove.main(argv)
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("DRY RUN", out)
        self.assertIn("add_comm_nat", out)

    def test_main_file_not_found_exit_3(self):
        rc = prove.main(
            ["--file", "/nonexistent/nope.lean", "--theorem", "t"]
        )
        self.assertEqual(rc, 3)


if __name__ == "__main__":
    unittest.main()
