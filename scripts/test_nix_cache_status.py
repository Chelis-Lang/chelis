"""Tests for the cache status report of the scheduled Nix workflow.

Run with `.venv/bin/python scripts/test_nix_cache_status.py`.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nix_cache_status  # noqa: E402
from nix_cache_status import Completed  # noqa: E402

PATHS = {
    "aarch64-darwin": [
        "/nix/store/aaaa-chelis-0.20.0",
        "/nix/store/bbbb-chelis-runtime-0.20.0",
        "/nix/store/gggg-chelis-unwrapped-0.20.0",
        "/nix/store/cccc-chelisup-0.20.0",
    ],
    "aarch64-linux": [
        "/nix/store/jjjj-chelis-0.20.0",
        "/nix/store/kkkk-chelis-runtime-0.20.0",
        "/nix/store/llll-chelis-unwrapped-0.20.0",
        "/nix/store/mmmm-chelisup-0.20.0",
    ],
    "x86_64-linux": [
        "/nix/store/dddd-chelis-0.20.0",
        "/nix/store/eeee-chelis-runtime-0.20.0",
        "/nix/store/hhhh-chelis-unwrapped-0.20.0",
        "/nix/store/ffff-chelisup-0.20.0",
    ],
}
ALL_PATHS = [path for outputs in PATHS.values() for path in outputs]
SHELL = "/nix/store/iiii-activated-devenv-shell"
SYSTEM = "aarch64-darwin"


class FakeNix:
    """Answer Nix and Devenv commands and record each call."""

    def __init__(
        self,
        evaluated: object = PATHS,
        missing: Sequence[str] = (),
        broken: Sequence[str] = (),
        eval_code: int = 0,
        system: str = SYSTEM,
        shell: object = SHELL,
        shell_code: int = 0,
        query_error: str = "error: unable to download narinfo",
    ) -> None:
        self.evaluated = evaluated
        self.missing = set(missing)
        self.broken = set(broken)
        self.eval_code = eval_code
        self.system = system
        self.shell = shell
        self.shell_code = shell_code
        self.query_error = query_error
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> Completed:
        argv = list(argv)
        self.calls.append(argv)
        if argv[:2] == ["nix", "eval"]:
            if self.eval_code:
                return Completed(self.eval_code, "", "error: evaluation failed")
            if "--expr" in argv:
                return Completed(0, self.system, "")
            return Completed(0, json.dumps(self.evaluated), "")
        if Path(argv[0]).name == "devenv":
            if self.shell_code:
                return Completed(self.shell_code, "", "error: shell evaluation failed")
            return Completed(0, json.dumps({"shell.outPath": self.shell}), "")
        if argv[:2] == ["nix", "path-info"]:
            path = argv[-1]
            if path in self.broken:
                return Completed(1, "", self.query_error)
            if path in self.missing:
                return Completed(
                    1,
                    "",
                    "don't know how to build these paths:\n"
                    f"  {path}\nerror: path '{path}' is not valid\n",
                )
            return Completed(0, f"{path}\n", "")
        raise AssertionError(f"unexpected command {argv!r}")

    def queried(self) -> list[str]:
        return [argv[-1] for argv in self.calls if argv[:2] == ["nix", "path-info"]]


def run_main(nix: FakeNix, output: Path | None = None) -> tuple[int, str, str]:
    argv = [] if output is None else ["--github-output", str(output)]
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = nix_cache_status.main(argv, run=nix)
    return code, stdout.getvalue(), stderr.getvalue()


class NixCacheStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "github-output"

    def test_cache_that_holds_every_path_is_complete(self) -> None:
        nix = FakeNix()
        code, stdout, _ = run_main(nix, self.output)
        self.assertEqual(code, 0)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "complete=true\n")
        self.assertEqual(sorted(nix.queried()), sorted([*ALL_PATHS, SHELL]))
        self.assertIn("holds 13 of 13 package and shell path(s)", stdout)

    def test_one_missing_path_makes_the_cache_incomplete(self) -> None:
        missing = PATHS["x86_64-linux"][0]
        code, stdout, _ = run_main(FakeNix(missing=[missing]), self.output)
        self.assertEqual(code, 0)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "complete=false\n")
        self.assertIn(f"does not hold {missing} (x86_64-linux)", stdout)
        self.assertIn("holds 12 of 13 package and shell path(s)", stdout)

    def test_each_query_names_the_published_cache(self) -> None:
        nix = FakeNix()
        run_main(nix)
        for argv in nix.calls:
            if argv[:2] == ["nix", "path-info"]:
                self.assertEqual(
                    argv[:4],
                    ["nix", "path-info", "--store", "https://chelis-lang.cachix.org"],
                )

    def test_evaluation_disables_import_from_derivation(self) -> None:
        nix = FakeNix()
        run_main(nix)
        [evaluation] = [argv for argv in nix.calls if ".#packages" in argv]
        self.assertIn("allow-import-from-derivation", evaluation)
        index = evaluation.index("allow-import-from-derivation")
        self.assertEqual(evaluation[index - 1 : index + 2], ["--option", "allow-import-from-derivation", "false"])
        self.assertIn(".#packages", evaluation)


    def test_without_an_output_file_the_report_still_prints(self) -> None:
        code, stdout, _ = run_main(FakeNix())
        self.assertEqual(code, 0)
        self.assertIn("holds 13 of 13", stdout)

    def test_unknown_query_failure_is_an_error_not_a_missing_path(self) -> None:
        broken = PATHS["aarch64-darwin"][1]
        code, _, stderr = run_main(FakeNix(broken=[broken]), self.output)
        self.assertEqual(code, 1)
        self.assertIn("unable to download narinfo", stderr)
        self.assertFalse(self.output.exists(), "a failed query must write no verdict")

    def test_failed_evaluation_is_an_error(self) -> None:
        code, _, stderr = run_main(FakeNix(eval_code=1), self.output)
        self.assertEqual(code, 1)
        self.assertIn("evaluation failed", stderr)
        self.assertFalse(self.output.exists())

    def test_evaluation_with_an_unknown_shape_is_an_error(self) -> None:
        shapes = {
            "no systems": {},
            "not a map": ["/nix/store/aaaa-chelis"],
            "short list": {"x86_64-linux": PATHS["x86_64-linux"][:2]},
            "not a store path": {"x86_64-linux": ["a", "b", "c"]},
        }
        for name, shape in shapes.items():
            with self.subTest(shape=name):
                code, _, _ = run_main(FakeNix(evaluated=shape), self.output)
                self.assertEqual(code, 1)
                self.assertFalse(self.output.exists())

    def test_missing_activated_shell_prevents_a_skip_with_cached_packages(self) -> None:
        nix = FakeNix(missing=[SHELL])
        code, stdout, _ = run_main(nix, self.output)
        self.assertEqual(code, 0)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "complete=false\n")
        self.assertIn(f"does not hold {SHELL} ({SYSTEM})", stdout)
        self.assertEqual(sorted(nix.queried()), sorted([*ALL_PATHS, SHELL]))

    def test_shell_query_failure_is_not_a_cache_miss(self) -> None:
        code, _, stderr = run_main(FakeNix(broken=[SHELL]), self.output)
        self.assertEqual(code, 1)
        self.assertIn("unable to download narinfo", stderr)
        self.assertFalse(self.output.exists())

    def test_invalid_signature_is_an_error_not_a_missing_shell(self) -> None:
        error = f"error: signature for path '{SHELL}' is not valid"
        nix = FakeNix(broken=[SHELL], query_error=error)
        code, _, stderr = run_main(nix, self.output)
        self.assertEqual(code, 1)
        self.assertIn(error, stderr)
        self.assertFalse(self.output.exists())

    def test_failed_shell_evaluation_writes_no_verdict(self) -> None:
        code, _, stderr = run_main(FakeNix(shell_code=1), self.output)
        self.assertEqual(code, 1)
        self.assertIn("shell evaluation failed", stderr)
        self.assertFalse(self.output.exists())

    def test_invalid_shell_evaluation_writes_no_verdict(self) -> None:
        for shell in (None, [], "", "/tmp/shell"):
            with self.subTest(shell=shell):
                code, _, _ = run_main(FakeNix(shell=shell), self.output)
                self.assertEqual(code, 1)
                self.assertFalse(self.output.exists())

    def test_unknown_native_system_writes_no_verdict(self) -> None:
        code, _, stderr = run_main(FakeNix(system="unsupported"), self.output)
        self.assertEqual(code, 1)
        self.assertIn("unsupported", stderr)
        self.assertFalse(self.output.exists())

    def test_each_native_report_checks_its_own_activated_shell(self) -> None:
        for system in PATHS:
            with self.subTest(system=system):
                nix = FakeNix(system=system)
                code, _, _ = run_main(nix)
                self.assertEqual(code, 0)
                [evaluation] = [
                    argv for argv in nix.calls if Path(argv[0]).name == "devenv"
                ]
                self.assertEqual(evaluation[evaluation.index("--system") + 1], system)
                index = evaluation.index("allow-import-from-derivation")
                self.assertEqual(evaluation[index + 1], "false")
                self.assertIn(SHELL, nix.queried())


if __name__ == "__main__":
    unittest.main()
