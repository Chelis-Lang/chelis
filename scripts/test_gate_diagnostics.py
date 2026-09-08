"""Tests for uv routing and actionable local-gate failure diagnostics."""

from __future__ import annotations

import importlib.util
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "gate_under_diagnostic_tests", here / "gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load_module()


class ManagedRuntimeTests(unittest.TestCase):
    def test_uv_run_marker_is_managed(self):
        self.assertTrue(
            gate.is_managed_runtime(
                {"UV_RUN_RECURSION_DEPTH": "1"},
                Path("/uv/python"),
                Path("/uv"),
                Path("/uv"),
            )
        )

    def test_uv_created_venv_is_managed(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / ".venv"
            prefix.mkdir()
            (prefix / "pyvenv.cfg").write_text(
                "home = /uv/python\nuv = 0.9.28\n"
            )
            self.assertTrue(
                gate.is_managed_runtime(
                    {},
                    prefix / "bin/python",
                    prefix,
                    Path("/uv/python"),
                )
            )

    def test_active_devenv_interpreter_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            prefix = state / "venv"
            self.assertTrue(
                gate.is_managed_runtime(
                    {"DEVENV_STATE": str(state)},
                    prefix / "bin/python",
                    prefix,
                    Path("/nix/store/python"),
                )
            )

    def test_unmanaged_invocation_reexecs_through_uv(self):
        calls = []

        def fake_execvpe(program, argv, environment):
            calls.append((program, argv, environment))
            raise RuntimeError("exec intercepted")

        with self.assertRaisesRegex(RuntimeError, "exec intercepted"):
            gate.ensure_managed_runtime(
                ["--local"],
                environ={"PATH": "/usr/bin"},
                executable=Path("/usr/bin/python3"),
                prefix=Path("/System/Python"),
                base_prefix=Path("/System/Python"),
                find_uv=lambda _: "/opt/bin/uv",
                execvpe=fake_execvpe,
            )

        self.assertEqual(len(calls), 1)
        program, argv, environment = calls[0]
        self.assertEqual(program, "/opt/bin/uv")
        self.assertEqual(
            argv[:8],
            [
                "/opt/bin/uv",
                "run",
                "--managed-python",
                "--python",
                "3.11",
                "--no-project",
                "python",
                str(Path(gate.__file__).resolve()),
            ],
        )
        self.assertEqual(argv[8:], ["--local"])
        self.assertEqual(environment["PATH"], "/usr/bin")

    def test_reexec_drops_a_conflicting_python_preference(self):
        """uv rejects --python-preference beside the --managed-python the gate passes.

        Devenv exports UV_PYTHON_PREFERENCE=only-system, and any uv user may
        set it, which turned `python3 scripts/gate.py` into `error: the
        argument --managed-python cannot be used with --python-preference`
        (chelis#1421). The gate's own flag decides which interpreter runs it.
        """
        calls = []

        def fake_execvpe(program, argv, environment):
            calls.append((program, argv, environment))
            raise RuntimeError("exec intercepted")

        with self.assertRaisesRegex(RuntimeError, "exec intercepted"):
            gate.ensure_managed_runtime(
                ["--list"],
                environ={
                    "PATH": "/usr/bin",
                    "UV_PYTHON_PREFERENCE": "only-system",
                    "UV_PYTHON_DOWNLOADS": "never",
                    "UV_PROJECT_ENVIRONMENT": "/state/venv",
                },
                executable=Path("/usr/bin/python3"),
                prefix=Path("/System/Python"),
                base_prefix=Path("/System/Python"),
                find_uv=lambda _: "/opt/bin/uv",
                execvpe=fake_execvpe,
            )

        _, argv, environment = calls[0]
        self.assertIn("--managed-python", argv)
        self.assertNotIn("UV_PYTHON_PREFERENCE", environment)
        # Only the conflicting knob is dropped. A workstation that forbids
        # downloads keeps failing loudly instead of having the gate fetch an
        # interpreter behind that choice.
        self.assertEqual(environment["UV_PYTHON_DOWNLOADS"], "never")
        self.assertEqual(environment["UV_PROJECT_ENVIRONMENT"], "/state/venv")
        self.assertEqual(environment["PATH"], "/usr/bin")

    def test_reexec_leaves_the_callers_own_environment_untouched(self):
        """The strip applies to the child, not to the caller's mapping."""
        caller_environment = {
            "PATH": "/usr/bin",
            "UV_PYTHON_PREFERENCE": "only-system",
        }

        def fake_execvpe(program, argv, environment):
            raise RuntimeError("exec intercepted")

        with self.assertRaisesRegex(RuntimeError, "exec intercepted"):
            gate.ensure_managed_runtime(
                ["--list"],
                environ=caller_environment,
                executable=Path("/usr/bin/python3"),
                prefix=Path("/System/Python"),
                base_prefix=Path("/System/Python"),
                find_uv=lambda _: "/opt/bin/uv",
                execvpe=fake_execvpe,
            )

        self.assertEqual(caller_environment["UV_PYTHON_PREFERENCE"], "only-system")

    def test_missing_uv_prints_install_and_setup_guidance(self):
        error = io.StringIO()
        result = gate.ensure_managed_runtime(
            ["--local"],
            environ={},
            executable=Path("/usr/bin/python3"),
            prefix=Path("/System/Python"),
            base_prefix=Path("/System/Python"),
            find_uv=lambda _: None,
            error_stream=error,
        )
        self.assertEqual(result, 127)
        diagnostic = error.getvalue()
        self.assertIn("uv is required", diagnostic)
        self.assertIn("astral.sh/uv/install.sh", diagnostic)
        self.assertIn("uv --version", diagnostic)
        self.assertIn("uv python install 3.11", diagnostic)


class GateEnvironmentTests(unittest.TestCase):
    def test_valid_explicit_pyo3_python_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configured = Path(sys.executable)
            environment = gate.gate_environment(
                {"PYO3_PYTHON": str(configured)},
                executable=Path("/uv/default-python"),
                repo_root=root,
            )
            self.assertEqual(environment["PYO3_PYTHON"], str(configured))

    def test_relative_explicit_pyo3_python_resolves_from_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configured = root / "managed/python"
            configured.parent.mkdir()
            configured.symlink_to(Path(sys.executable))
            environment = gate.gate_environment(
                {"PYO3_PYTHON": "managed/python"},
                executable=Path("/uv/default-python"),
                repo_root=root,
            )
            self.assertEqual(environment["PYO3_PYTHON"], str(configured))

    def test_invalid_explicit_pyo3_python_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(
                ValueError, r"PYO3_PYTHON.*missing/python"
            ):
                gate.gate_environment(
                    {"PYO3_PYTHON": "missing/python"},
                    executable=Path("/uv/default-python"),
                    repo_root=root,
                )

    def test_existing_non_python_pyo3_path_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configured = root / "not-python"
            configured.write_text("not a Python interpreter\n")
            with self.assertRaisesRegex(
                ValueError,
                rf"PYO3_PYTHON.*{re.escape(str(configured))}.*"
                r"usable Python 3\.11",
            ):
                gate.gate_environment(
                    {"PYO3_PYTHON": str(configured)},
                    executable=Path(sys.executable),
                    repo_root=root,
                )

    def test_selected_uv_interpreter_is_set_for_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(sys.executable)
            environment = gate.gate_environment(
                {"PATH": "/usr/bin"},
                executable=executable,
                repo_root=Path(tmp),
            )
            self.assertEqual(environment["PYO3_PYTHON"], str(executable))
            self.assertEqual(
                environment["CARGO_TARGET_DIR"],
                str((Path(tmp) / "target").resolve()),
            )
            self.assertEqual(
                environment["CARGO_HUSKY_DONT_INSTALL_HOOKS"],
                "1",
            )

    def test_relative_cargo_target_is_normalized_inside_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment = gate.gate_environment(
                {"CARGO_TARGET_DIR": "target/agents/test"},
                executable=Path(sys.executable),
                repo_root=root,
            )
            self.assertEqual(
                environment["CARGO_TARGET_DIR"],
                str((root / "target/agents/test").resolve()),
            )

    def test_external_cargo_target_is_rejected_as_cross_worktree_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkout"
            root.mkdir()
            with self.assertRaisesRegex(
                ValueError,
                r"CARGO_TARGET_DIR.*outside the current worktree",
            ):
                gate.gate_environment(
                    {"CARGO_TARGET_DIR": str(Path(tmp) / "shared-target")},
                    executable=Path(sys.executable),
                    repo_root=root,
                )

    def test_python_stage_uses_selected_interpreter(self):
        command = gate.materialize_command(
            gate.CHECKPOINT_COMPILE_FAIL,
            Path("/uv/python3.11"),
        )
        self.assertEqual(
            command,
            [
                "/uv/python3.11",
                "scripts/check_checkpoint_compile_fail.py",
            ],
        )


class FailureDiagnosticTests(unittest.TestCase):
    def test_failure_streams_persists_and_replays_complete_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failure_root = root / "target/gate-failures"
            output = io.StringIO()
            error = io.StringIO()
            command = [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "for i in range(205):\n"
                    " print(f'line-{i:03}', flush=True)\n"
                    "print('stderr-marker', file=sys.stderr, flush=True)\n"
                    "raise SystemExit(7)\n"
                ),
            ]
            result = gate.run_commands(
                [command],
                stage_label="diagnostic-test",
                repo_root=root,
                failure_root=failure_root,
                environ={"PATH": os.environ.get("PATH", "")},
                executable=Path(sys.executable),
                output_stream=output,
                error_stream=error,
            )

            self.assertEqual(result, 7)
            streamed = output.getvalue()
            self.assertIn("line-000", streamed)
            self.assertIn("line-204", streamed)
            self.assertIn("stderr-marker", streamed)

            logs = list(failure_root.glob("*.log"))
            self.assertEqual(len(logs), 1)
            transcript = logs[0].read_text()
            self.assertIn("line-000", transcript)
            self.assertIn("line-204", transcript)
            self.assertIn("stderr-marker", transcript)

            diagnostic = error.getvalue()
            self.assertIn("diagnostic-test command 1/1", diagnostic)
            self.assertIn("exit code: 7", diagnostic)
            self.assertIn("duration:", diagnostic)
            self.assertIn("PYO3_PYTHON=", diagnostic)
            self.assertIn("rerun:", diagnostic)
            rerun = next(
                line
                for line in diagnostic.splitlines()
                if line.startswith("gate: rerun:")
            )
            self.assertIn("CARGO_HUSKY_DONT_INSTALL_HOOKS=1", rerun)
            self.assertIn(str(logs[0]), diagnostic)
            self.assertIn("final 200 lines", diagnostic)
            self.assertIn("6 earlier lines omitted", diagnostic)
            self.assertNotIn("line-000", diagnostic)
            self.assertIn("line-204", diagnostic)

    def test_signal_is_named(self):
        self.assertEqual(
            gate.describe_returncode(-15),
            "signal: SIGTERM (15)",
        )

    def test_success_keeps_no_persistent_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failure_root = root / "target/gate-failures"
            output = io.StringIO()
            result = gate.run_commands(
                [
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os; "
                            "print(os.environ['PYO3_PYTHON'])"
                        ),
                    ]
                ],
                stage_label="success-test",
                repo_root=root,
                failure_root=failure_root,
                environ={"PATH": os.environ.get("PATH", "")},
                executable=Path(sys.executable),
                output_stream=output,
                error_stream=io.StringIO(),
            )
            self.assertEqual(result, 0)
            self.assertIn(sys.executable, output.getvalue())
            self.assertFalse(failure_root.exists())

    def test_launch_failure_is_diagnostic_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failure_root = root / "target/gate-failures"
            error = io.StringIO()
            real_popen = gate.subprocess.Popen

            def fail_only_missing_command(command, *args, **kwargs):
                if command[0] == "missing-command":
                    raise FileNotFoundError("missing-command")
                return real_popen(command, *args, **kwargs)

            with mock.patch.object(
                gate.subprocess,
                "Popen",
                side_effect=fail_only_missing_command,
            ):
                result = gate.run_commands(
                    [["missing-command"]],
                    stage_label="launch-test",
                    repo_root=root,
                    failure_root=failure_root,
                    environ={"PATH": "/usr/bin"},
                    executable=Path(sys.executable),
                    output_stream=io.StringIO(),
                    error_stream=error,
                )
            self.assertEqual(result, 127)
            self.assertIn("launch error", error.getvalue())
            logs = list(failure_root.glob("*.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("missing-command", logs[0].read_text())


if __name__ == "__main__":
    unittest.main()
