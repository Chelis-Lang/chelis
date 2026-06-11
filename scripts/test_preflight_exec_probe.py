"""Unit tests for `preflight_exec_probe.py`.

Run via: `python3 -m unittest scripts.test_preflight_exec_probe` from
repo root, or `python3 scripts/test_preflight_exec_probe.py`.

Four things are locked here:

  (a) the success path: a real (tiny, fast) compile + first-exec round
      trip exits 0 and prints `exec ok (N ms)`;
  (b) the wedge classification: a binary that outlives the timeout is
      classified as wedged (`exec_probe` returns None), and the
      end-to-end `main` path exits 1 with a warning that names the
      runbook (`docs/local_macos_environment.md`);
  (c) cleanup: the probe leaves no `chelis-preflight-exec-probe-*` temp
      dirs behind, so it is safe to run from anywhere;
  (d) CLI shape: the default timeout is 15s and non-positive timeouts
      are rejected by argparse.
"""

import importlib.util
import io
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "preflight_exec_probe", here / "preflight_exec_probe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _load_module()

# A payload that outlives any short test timeout: used to exercise the
# wedge classification without depending on an actually wedged machine.
SLEEPER_SOURCE = "#include <unistd.h>\nint main(void) { sleep(60); return 0; }\n"
BRIEF_SLEEPER_SOURCE = (
    "#include <unistd.h>\nint main(void) { usleep(200000); return 0; }\n"
)


def _probe_temp_dirs() -> set[str]:
    tmp_root = Path(tempfile.gettempdir())
    return {str(p) for p in tmp_root.glob(f"{probe.TEMP_DIR_PREFIX}*")}


class SuccessPathTests(unittest.TestCase):
    def test_main_success_prints_exec_ok_and_exits_zero(self):
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = probe.main([])
        self.assertEqual(rc, probe.EXIT_OK, f"stderr: {err.getvalue()!r}")
        self.assertRegex(out.getvalue().strip(), r"^exec ok \(\d+ ms\)$")
        self.assertEqual(err.getvalue(), "")

    def test_exec_probe_returns_elapsed_ms_for_trivial_binary(self):
        with tempfile.TemporaryDirectory() as td:
            binary = probe.compile_probe(Path(td))
            elapsed_ms = probe.exec_probe(binary, timeout_seconds=30.0)
        self.assertIsNotNone(elapsed_ms)
        self.assertGreaterEqual(elapsed_ms, 0.0)

    def test_unexecutable_probe_is_env_error_not_wedge(self):
        # A refused exec (noexec TMPDIR, stripped permissions) is an
        # environment problem: it must surface as the RuntimeError ->
        # exit-2 lane, never crash or masquerade as the exit-1 wedge.
        with tempfile.TemporaryDirectory() as td:
            binary = probe.compile_probe(Path(td))
            binary.chmod(0o644)
            with self.assertRaises(RuntimeError):
                probe.exec_probe(binary, timeout_seconds=5.0)


class WedgeClassificationTests(unittest.TestCase):
    def test_exec_probe_classifies_timeout_as_wedged(self):
        with tempfile.TemporaryDirectory() as td:
            binary = probe.compile_probe(Path(td), source=SLEEPER_SOURCE)
            # The sleeper outlives the 0.2s timeout deterministically,
            # so this exercises the wedge classification without
            # depending on an actually wedged machine.
            result = probe.exec_probe(binary, timeout_seconds=0.2)
        self.assertIsNone(
            result,
            "a binary that outlives the timeout must classify as wedged",
        )

    def test_main_wedged_exits_one_and_points_at_runbook(self):
        out = io.StringIO()
        err = io.StringIO()
        with mock.patch.object(probe, "PROBE_SOURCE", SLEEPER_SOURCE):
            with redirect_stdout(out), redirect_stderr(err):
                rc = probe.main(["--timeout", "0.2"])
        self.assertEqual(rc, probe.EXIT_WEDGED)
        self.assertEqual(out.getvalue(), "")
        message = err.getvalue()
        self.assertIn("appears wedged", message)
        self.assertIn(probe.RUNBOOK, message)

    def test_main_slow_admission_exits_three_and_names_356(self):
        # The silent degradation variant (chelis#356): the exec SUCCEEDS
        # but takes longer than --warn-ms. Must be distinguishable from
        # both healthy (0) and wedged (1) for automation.
        out = io.StringIO()
        err = io.StringIO()
        with mock.patch.object(probe, "PROBE_SOURCE", BRIEF_SLEEPER_SOURCE):
            with redirect_stdout(out), redirect_stderr(err):
                rc = probe.main(["--timeout", "30", "--warn-ms", "50"])
        self.assertEqual(rc, probe.EXIT_SLOW)
        self.assertEqual(out.getvalue(), "")
        message = err.getvalue()
        self.assertIn("admitting binaries slowly", message)
        self.assertIn("chelis#356", message)
        self.assertIn(probe.RUNBOOK, message)

    def test_fast_exec_under_default_threshold_is_ok_not_slow(self):
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = probe.main([])
        self.assertEqual(rc, probe.EXIT_OK)
        self.assertIn("exec ok", out.getvalue())

    def test_non_positive_warn_ms_is_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            probe.parse_args(["--warn-ms", "0"])
        self.assertEqual(ctx.exception.code, 2)

    def test_compile_failure_is_env_error_not_wedge(self):
        out = io.StringIO()
        err = io.StringIO()
        with mock.patch.object(probe, "PROBE_SOURCE", "this is not C\n"):
            with redirect_stdout(out), redirect_stderr(err):
                rc = probe.main([])
        self.assertEqual(rc, probe.EXIT_ENV)
        self.assertIn("cannot run probe", err.getvalue())


class CleanupTests(unittest.TestCase):
    def test_success_run_leaves_no_probe_temp_dirs(self):
        before = _probe_temp_dirs()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = probe.main([])
        self.assertEqual(rc, probe.EXIT_OK)
        leftover = _probe_temp_dirs() - before
        self.assertEqual(leftover, set(), f"probe leaked temp dirs: {leftover}")

    def test_wedged_run_leaves_no_probe_temp_dirs(self):
        before = _probe_temp_dirs()
        with mock.patch.object(probe, "PROBE_SOURCE", SLEEPER_SOURCE):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = probe.main(["--timeout", "0.2"])
        self.assertEqual(rc, probe.EXIT_WEDGED)
        leftover = _probe_temp_dirs() - before
        self.assertEqual(leftover, set(), f"probe leaked temp dirs: {leftover}")


class CliShapeTests(unittest.TestCase):
    def test_default_timeout_is_fifteen_seconds(self):
        args = probe.parse_args([])
        self.assertEqual(args.timeout, 15.0)
        self.assertEqual(probe.DEFAULT_TIMEOUT_SECONDS, 15.0)

    def test_timeout_is_configurable(self):
        args = probe.parse_args(["--timeout", "60"])
        self.assertEqual(args.timeout, 60.0)

    def test_non_positive_timeout_is_rejected(self):
        for bad in ("0", "-5"):
            with self.subTest(bad=bad):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as ctx:
                        probe.parse_args(["--timeout", bad])
                self.assertEqual(ctx.exception.code, 2)

    def test_runbook_constant_points_at_existing_doc(self):
        repo_root = Path(__file__).resolve().parent.parent
        runbook = repo_root / probe.RUNBOOK
        self.assertTrue(runbook.is_file(), f"missing runbook {runbook}")
        # The runbook must name the exact observed log signatures (the
        # acceptance criterion from issue #349).
        text = runbook.read_text()
        self.assertIn("Unable to initialize qtn_proc: 3", text)
        self.assertIn("dispatch_mig_server returned 268435459", text)
        self.assertIn("_dyld_start", text)


if __name__ == "__main__":
    unittest.main()
