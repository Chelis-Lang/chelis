"""Tests for uv routing and actionable local-gate failure diagnostics."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
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
                ["--validation"],
                environ={"PATH": "/usr/bin"},
                executable=Path("/usr/bin/python3"),
                prefix=Path("/System/Python"),
                base_prefix=Path("/System/Python"),
                repo_root=Path(tempfile.gettempdir()) / "gate-no-owned-venv",
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
        self.assertEqual(argv[8:], ["--validation"])
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
                repo_root=Path(tempfile.gettempdir()) / "gate-no-owned-venv",
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
                repo_root=Path(tempfile.gettempdir()) / "gate-no-owned-venv",
                find_uv=lambda _: "/opt/bin/uv",
                execvpe=fake_execvpe,
            )

        self.assertEqual(caller_environment["UV_PYTHON_PREFERENCE"], "only-system")

    def test_missing_uv_prints_install_and_setup_guidance(self):
        error = io.StringIO()
        result = gate.ensure_managed_runtime(
            ["--validation"],
            environ={},
            executable=Path("/usr/bin/python3"),
            prefix=Path("/System/Python"),
            base_prefix=Path("/System/Python"),
            repo_root=Path(tempfile.gettempdir()) / "gate-no-owned-venv",
            find_uv=lambda _: None,
            error_stream=error,
        )
        self.assertEqual(result, 127)
        diagnostic = error.getvalue()
        self.assertIn("uv is required", diagnostic)
        self.assertIn("astral.sh/uv/install.sh", diagnostic)
        self.assertIn("uv --version", diagnostic)
        self.assertIn("uv python install 3.11", diagnostic)


class OwnedInterpreterTests(unittest.TestCase):
    """chelis#2511: a venv another checkout put first on PATH is managed but
    not owned, so the gate must not hand it to every child command."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gate-owned-")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name).resolve()
        self.root = self.base / "worktree"
        self.root.mkdir()
        self.foreign = self.base / "primary" / ".venv"

    def fake_venv(self, prefix):
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin" / "python").write_text("", encoding="utf-8")
        (prefix / "pyvenv.cfg").write_text("uv = 0.9.28\n", encoding="utf-8")
        return prefix / "bin" / "python"

    def launch(self, environ, prefix):
        calls = []

        def fake_execvpe(program, argv, environment):
            calls.append((program, argv))
            self.reexec_environment = environment
            raise RuntimeError("exec intercepted")

        def no_uv(_):
            raise AssertionError("an owned interpreter must not route through uv")

        try:
            result = gate.ensure_managed_runtime(
                ["--validation"],
                environ=environ,
                executable=prefix / "bin" / "python",
                prefix=prefix,
                base_prefix=Path("/uv/python"),
                repo_root=self.root,
                find_uv=no_uv,
                execvpe=fake_execvpe,
            )
        except RuntimeError:
            return "exec", calls
        return result, calls

    def test_foreign_checkout_venv_reexecs_through_the_owned_venv(self):
        self.fake_venv(self.foreign)
        owned = self.fake_venv(self.root / ".venv")
        result, calls = self.launch({"PATH": f"{self.foreign}/bin"}, self.foreign)
        self.assertEqual(result, "exec")
        self.assertEqual(
            calls,
            [(str(owned), [str(owned), str(Path(gate.__file__).resolve()), "--validation"])],
        )
        self.assertEqual(self.reexec_environment[gate.OWNED_REEXEC_ENV], "1")

    def test_a_second_unowned_arrival_stops_instead_of_looping(self):
        """The re-exec target was chosen because it exists; if it still does
        not start as this checkout's environment, the gate must stop."""
        self.fake_venv(self.foreign)
        owned = self.fake_venv(self.root / ".venv")
        error = io.StringIO()
        result = gate.ensure_managed_runtime(
            ["--validation"],
            environ={gate.OWNED_REEXEC_ENV: "1"},
            executable=self.foreign / "bin" / "python",
            prefix=self.foreign,
            base_prefix=Path("/uv/python"),
            repo_root=self.root,
            find_uv=lambda _: self.fail("must not route through uv"),
            execvpe=lambda *_: self.fail("must not re-exec again"),
            error_stream=error,
        )
        self.assertEqual(result, gate.EXIT_ENVIRONMENT)
        self.assertIn(str(owned), error.getvalue())
        self.assertIn("uv venv --python 3.11", error.getvalue())

    def test_a_malformed_owned_venv_fails_loudly_when_really_executed(self):
        """Execute the bootstrap for real: a `.venv/bin/python` that starts
        the base interpreter, not the venv, must end in one clear failure."""
        checkout = self.base / "malformed"
        scripts = checkout / "scripts"
        scripts.mkdir(parents=True)
        here = Path(__file__).resolve().parent
        for name in ("gate.py", "unrepresentable_domain_oracle.py", "ci_setup_uv_python.py"):
            shutil.copy2(here / name, scripts / name)
        wrapper = checkout / ".venv" / "bin" / "python"
        wrapper.parent.mkdir(parents=True)
        base = Path(sys.base_prefix) / "bin" / f"python{sys.version_info[0]}"
        wrapper.write_text(f'#!/bin/sh\nexec "{base}" "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYO3_PYTHON", "DEVENV_STATE", gate.OWNED_REEXEC_ENV}
        }
        result = subprocess.run(
            [sys.executable, str(scripts / "gate.py"), "--list"],
            cwd=checkout, env=environment, capture_output=True, text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, gate.EXIT_ENVIRONMENT, result.stderr)
        self.assertIn("does not start as this checkout's own environment", result.stderr)

    def test_owned_venv_proceeds_without_any_reexec(self):
        self.fake_venv(self.root / ".venv")
        result, calls = self.launch({}, self.root / ".venv")
        self.assertIsNone(result)
        self.assertEqual(calls, [])

    def test_owned_venv_without_a_uv_marker_proceeds_rather_than_looping(self):
        """Re-exec into the owned venv must end there: routing it back
        through uv would re-exec into the owned venv again."""
        prefix = self.root / ".venv"
        self.fake_venv(prefix)
        (prefix / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
        self.assertEqual(self.launch({}, prefix), (None, []))

    def test_owned_devenv_state_venv_is_preferred_and_proceeds(self):
        state = self.root / ".devenv" / "profiles" / "ci" / "state"
        devenv = self.fake_venv(state / "venv")
        self.fake_venv(self.root / ".venv")
        environ = {"DEVENV_STATE": str(state)}
        self.assertEqual(self.launch(environ, state / "venv"), (None, []))
        self.fake_venv(self.foreign)
        result, calls = self.launch(environ, self.foreign)
        self.assertEqual(result, "exec")
        self.assertEqual(calls[0][0], str(devenv))

    def test_devenv_state_outside_the_checkout_is_not_owned(self):
        state = self.base / "primary" / ".devenv" / "state"
        self.fake_venv(state / "venv")
        owned = self.fake_venv(self.root / ".venv")
        result, calls = self.launch({"DEVENV_STATE": str(state)}, state / "venv")
        self.assertEqual(result, "exec")
        self.assertEqual(calls[0][0], str(owned))

    def test_explicit_pyo3_python_is_authoritative(self):
        self.fake_venv(self.foreign)
        self.fake_venv(self.root / ".venv")
        environ = {"PYO3_PYTHON": str(self.foreign / "bin" / "python")}
        self.assertEqual(self.launch(environ, self.foreign), (None, []))

    def test_foreign_venv_without_an_owned_interpreter_keeps_the_managed_runtime(self):
        self.fake_venv(self.foreign)
        self.assertEqual(self.launch({}, self.foreign), (None, []))

    def test_ownership_rule_agrees_with_the_census(self):
        """Both copies of the rule answer the same on real interpreters."""
        scripts = str(Path(__file__).resolve().parent)
        probe = (
            "import json, sys; from pathlib import Path; "
            f"sys.path.insert(0, {scripts!r}); "
            "from capacity_census_native_execution import _owned_interpreter; "
            "print(json.dumps([sys.prefix, "
            f"_owned_interpreter(Path({str(self.root)!r}))]))"
        )
        owned_state = self.root / ".devenv" / "profiles" / "ci" / "state"
        foreign_state = self.base / "primary" / ".devenv" / "state"
        cases = {
            "checkout venv": (self.root / ".venv", None),
            "checkout devenv": (owned_state / "venv", owned_state),
            "foreign devenv": (foreign_state / "venv", foreign_state),
            "foreign venv": (self.foreign, None),
        }
        verdicts = {}
        for name, (prefix, state) in cases.items():
            venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(prefix)
            environment = dict(os.environ)
            environment.pop("DEVENV_STATE", None)
            if state is not None:
                environment["DEVENV_STATE"] = str(state)
            reported, census = json.loads(subprocess.check_output(
                [str(prefix / "bin" / "python"), "-I", "-c", probe],
                cwd=self.root, env=environment, text=True,
            ))
            with self.subTest(case=name):
                self.assertEqual(
                    gate.owns_interpreter(environment, Path(reported), self.root),
                    census,
                )
            verdicts[name] = census
        self.assertEqual(
            verdicts,
            {
                "checkout venv": True,
                "checkout devenv": True,
                "foreign devenv": False,
                "foreign venv": False,
            },
        )


class GateEnvironmentTests(unittest.TestCase):
    def test_current_interpreter_replaces_stale_pyo3_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = gate.gate_environment(
                {"PYO3_ENVIRONMENT_SIGNATURE": "stale"},
                executable=Path(sys.executable), repo_root=Path(tmp),
            )
            self.assertTrue(environment["PYO3_ENVIRONMENT_SIGNATURE"].startswith("chelis-pyo3-v1-"))
            repeated = gate.gate_environment(environment, executable=Path(sys.executable), repo_root=Path(tmp))
            self.assertEqual(environment["PYO3_ENVIRONMENT_SIGNATURE"], repeated["PYO3_ENVIRONMENT_SIGNATURE"])

    def test_discovery_override_is_not_forwarded_to_gate_children(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "PYO3_NO_PYTHON"):
                gate.gate_environment(
                    {"PYO3_NO_PYTHON": "1"}, executable=Path(sys.executable), repo_root=Path(tmp),
                )

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
