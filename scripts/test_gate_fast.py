"""Unit tests for `gate.py --fast`, the preflight, the run summary, and the
advisory lease.

Run via `.venv/bin/python -m unittest scripts.test_gate_fast` from the repo
root, or through the CI `unittest discover -s scripts` step.

Everything here is cargo-free: child commands go through a `Popen` stub,
git is injected, the probe is a canned dict, and the lease and summary live
in a temporary directory. The only real subprocess is `gate_environment`'s
interpreter probe of `sys.executable`.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "gate_under_fast_tests", here / "gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load_module()
REPO_ROOT = Path(__file__).resolve().parent.parent

CANNED_MEMBERS = {
    "crates/chelis-cli": "chelis-cli",
    "crates/chelis-surf": "chelis-surf",
    "crates/chelis-std-bundle": "chelis-std-bundle",
}
CANNED_GIT_FACTS = {
    "head": "1111111111111111111111111111111111111111",
    "origin_main": "2222222222222222222222222222222222222222",
    "merge_base": "2222222222222222222222222222222222222222",
}


def _canned_probe(*_args, **_kwargs):
    return {"verdict": "ok", "exit_code": 0, "output": "exec ok (1 ms)"}


def _isolated_environ(tmp: str) -> dict:
    return {
        "PATH": os.environ.get("PATH", ""),
        gate.REPORT_DIR_ENV: tmp,
        gate.LEASE_DIR_ENV: tmp,
    }


def _parse_quietly(argv):
    with redirect_stderr(io.StringIO()):
        return gate.parse_args(argv)


class _Stub:
    """A `Popen` stand-in with no output and a scripted exit code."""

    def __init__(self, returncode: int = 0) -> None:
        self.stdout = io.StringIO("")
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def _popen_stub(returncodes: dict[int, int] | None = None, *, raise_on: int | None = None):
    """Intercept only the gate's own children (they carry `env`); the
    interpreter probe in `gate_environment` goes to the real Popen."""
    real_popen = gate.subprocess.Popen
    launched: list[list[str]] = []
    codes = returncodes or {}

    def fake_popen(command, *args, **kwargs):
        if "env" not in kwargs:
            return real_popen(command, *args, **kwargs)
        launched.append(list(command))
        if raise_on is not None and len(launched) == raise_on:
            raise KeyboardInterrupt()
        return _Stub(codes.get(len(launched), 0))

    return fake_popen, launched


def _summary_files(tmp: str) -> list[Path]:
    return sorted(Path(tmp).glob("*.json"))


class FastCommandListTests(unittest.TestCase):
    def test_static_prefix_is_regen_fmt_write_then_lint(self):
        commands = gate.fast_command_list([], std_changed=False)
        self.assertEqual(
            commands[:3],
            [gate.REGEN_TIER0_WRITE, gate.FMT_WRITE, gate.CHELIS_LINT_CHECK],
        )
        self.assertEqual(
            gate.render(gate.REGEN_TIER0_WRITE),
            "<managed-python> scripts/regen_all.py --tier 0",
        )
        self.assertEqual(gate.render(gate.FMT_WRITE), "cargo fmt --all")
        self.assertNotIn("--check", gate.FMT_WRITE)

    def test_one_clippy_per_changed_crate_precedes_the_tripwire_run(self):
        commands = gate.fast_command_list(
            ["chelis-cli", "chelis-surf"], std_changed=False
        )
        self.assertEqual(
            commands[3:5],
            [
                ["cargo", "clippy", "-p", "chelis-cli", "--tests", "--", "-D", "warnings"],
                ["cargo", "clippy", "-p", "chelis-surf", "--tests", "--", "-D", "warnings"],
            ],
        )
        self.assertEqual(commands[5], gate.FAST_TRIPWIRE_NEXTEST)
        self.assertEqual(len(commands), 6)

    def test_std_bundle_legs_appear_only_when_std_paths_changed(self):
        self.assertTrue(gate.std_paths_changed(["packages/chelis-std/src/x.ch"]))
        self.assertTrue(gate.std_paths_changed(["crates/chelis-std-bundle/build.rs"]))
        self.assertFalse(gate.std_paths_changed(["crates/chelis-cli/src/main.rs"]))
        self.assertFalse(gate.std_paths_changed([]))
        without = gate.fast_command_list([], std_changed=False)
        with_std = gate.fast_command_list(["chelis-std-bundle"], std_changed=True)
        self.assertEqual(without[-1], gate.FAST_TRIPWIRE_NEXTEST)
        self.assertNotIn(gate.REGEN_TIER1_WRITE, without)
        self.assertNotIn(gate.STD_BUNDLE_SELF_CONSISTENCY, without)
        # Writers first: tier 1 regenerates the bundle right after tier 0, so
        # lint, clippy, and the bundled-std tripwire see fresh embedded bytes;
        # the self-consistency test is a check and goes last.
        self.assertEqual(
            with_std,
            [
                gate.REGEN_TIER0_WRITE,
                gate.REGEN_TIER1_WRITE,
                gate.FMT_WRITE,
                gate.CHELIS_LINT_CHECK,
                ["cargo", "clippy", "-p", "chelis-std-bundle", "--tests", "--", "-D", "warnings"],
                gate.FAST_TRIPWIRE_NEXTEST,
                gate.STD_BUNDLE_SELF_CONSISTENCY,
            ],
        )
        self.assertEqual(
            gate.render(gate.REGEN_TIER1_WRITE),
            "<managed-python> scripts/regen_all.py --tier 1",
        )

    def test_fast_excludes_workspace_clippy_fmt_check_doctests_and_both_oracles(self):
        rendered = [
            gate.render(c)
            for c in gate.fast_command_list(["chelis-cli"], std_changed=True)
        ]
        for excluded in (
            gate.CLIPPY_WORKSPACE,
            gate.CLIPPY_SOLVER_FREE_FEATURES,
            gate.CLIPPY_NO_DEFAULT_FEATURES,
            gate.FMT_CHECK,
            gate.DOCTEST_TYPES,
            gate.DOCTEST_COMPILER_API,
            gate.DOCTEST_PIPELINE_CORE,
            gate.UNREPRESENTABLE_DOMAIN_ORACLE,
            gate.RUNTIME_REPRESENTATION_ORACLE,
            gate.CHELIS_STD_BUNDLE_CHECK,
            gate.NEXTEST_WORKSPACE,
            gate.BUILD_WORKSPACE,
        ):
            self.assertNotIn(gate.render(excluded), rendered)
        for command in rendered:
            self.assertNotIn("--workspace", command)

    def test_tripwire_command_names_existing_test_targets(self):
        command = gate.FAST_TRIPWIRE_NEXTEST
        self.assertEqual(command[:4], ["cargo", "nextest", "run", "--no-fail-fast"])
        package = None
        pairs = []
        for index, token in enumerate(command):
            if token == "-p":
                package = command[index + 1]
            elif token == "--test":
                self.assertIsNotNone(package)
                pairs.append((package, command[index + 1]))
        self.assertEqual(len(pairs), 12)
        for package, target in pairs:
            path = REPO_ROOT / "crates" / package / "tests" / f"{target}.rs"
            self.assertTrue(path.is_file(), f"missing tripwire target {path}")
        self.assertNotIn("--lib", command)
        # The std-bundle unit test rides separately so `--lib` does not
        # widen every package selection above.
        self.assertIn("--lib", gate.STD_BUNDLE_SELF_CONSISTENCY)
        self.assertEqual(gate.STD_BUNDLE_SELF_CONSISTENCY[3:5], ["-p", "chelis-std-bundle"])

    def test_fast_list_hands_over_no_oracle_binary(self):
        commands = gate.fast_command_list(["chelis-cli"], std_changed=True)
        self.assertIsNone(gate.oracle_binary_handoff(commands, "/t"))

    def test_fast_uses_the_managed_python_marker(self):
        for command in (gate.REGEN_TIER0_WRITE, gate.REGEN_TIER1_WRITE):
            self.assertEqual(command[0], gate.MANAGED_PYTHON)
        materialized = gate.materialize_command(
            gate.REGEN_TIER0_WRITE, Path("/py/bin/python")
        )
        self.assertEqual(materialized[0], "/py/bin/python")

    def test_fast_note_and_annotation_are_distinct_constants(self):
        self.assertEqual(gate.FAST_ANNOTATION, "fast + local + ci")
        self.assertNotEqual(gate.FAST_ANNOTATION, gate.LOCAL_ANNOTATION)
        self.assertTrue(gate.FAST_DYNAMIC_NOTE.startswith("# "))


class FastArgTests(unittest.TestCase):
    def test_fast_alone_parses(self):
        args = _parse_quietly(["--fast"])
        self.assertTrue(args.fast)
        self.assertFalse(args.local)

    def test_fast_excludes_stage_list_local_and_integration_selectors(self):
        for argv in (
            ["--fast", "lint-and-unit"],
            ["--fast", "--list"],
            ["--fast", "--local"],
            ["--fast", "integration", "--tests-only"],
            ["--fast", "integration", "--support-only"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                _parse_quietly(argv)

    def test_lease_flags_exclude_list_and_stages(self):
        for flag in (["--no-wait"], ["--no-lease"], ["--lease-timeout", "5"]):
            for other in (["--list"], ["lint-and-unit"], ["integration"]):
                with self.subTest(flag=flag, other=other), self.assertRaises(SystemExit):
                    _parse_quietly([*other, *flag])

    def test_lease_flag_pairs_are_exclusive(self):
        for argv in (
            ["--local", "--no-lease", "--no-wait"],
            ["--local", "--no-lease", "--lease-timeout", "5"],
            ["--local", "--no-wait", "--lease-timeout", "5"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                _parse_quietly(argv)

    def test_lease_timeout_must_be_positive(self):
        for value in ("0", "-3"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                _parse_quietly(["--local", "--lease-timeout", value])
        args = _parse_quietly(["--local", "--lease-timeout", "90"])
        self.assertEqual(args.lease_timeout, 90.0)

    def test_lease_flags_parse_with_local_fast_and_bare(self):
        self.assertTrue(_parse_quietly(["--local", "--no-wait"]).no_wait)
        self.assertTrue(_parse_quietly(["--fast", "--no-lease"]).no_lease)
        self.assertTrue(_parse_quietly(["--no-lease"]).no_lease)


class PreflightTests(unittest.TestCase):
    def _preflight(
        self,
        *,
        system="Darwin",
        probe_result=None,
        venv=True,
        environ_extra=None,
        probe_calls=None,
    ):
        report = gate.GateReport(mode="local", started_at="now")
        out, err = io.StringIO(), io.StringIO()
        calls = [] if probe_calls is None else probe_calls

        def probe(python, root):
            calls.append((python, root))
            return probe_result or _canned_probe()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            if venv:
                (root / ".venv" / "bin").mkdir(parents=True)
                (root / ".venv" / "bin" / "python").write_text("")
            environ = {"PATH": os.environ.get("PATH", "")}
            environ.update(environ_extra or {})
            with mock.patch.object(gate, "_git_facts", lambda: dict(CANNED_GIT_FACTS)):
                code, environment = gate.run_preflight(
                    mode="local",
                    report=report,
                    environ=environ,
                    executable=Path(sys.executable),
                    repo_root=root,
                    system=lambda: system,
                    probe=probe,
                    output_stream=out,
                    error_stream=err,
                )
        return code, environment, report, out.getvalue(), err.getvalue(), calls

    def test_host_system_defaults_to_platform_and_is_resolved_at_call_time(self):
        report = gate.GateReport(mode="local", started_at="now")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(gate, "_git_facts", lambda: dict(CANNED_GIT_FACTS)), \
                mock.patch.object(gate, "host_system", lambda: "Linux"), \
                mock.patch.object(gate, "run_probe", lambda *a, **k: self.fail("probe ran")):
            code, _env = gate.run_preflight(
                mode="local", report=report, environ={"PATH": os.environ.get("PATH", "")},
                executable=Path(sys.executable), repo_root=Path(tmp),
                output_stream=io.StringIO(), error_stream=io.StringIO(),
            )
        self.assertIsNone(code)
        self.assertEqual(report.preflight["probe"]["verdict"], "skipped")
        self.assertEqual(gate.host_system(), gate.platform.system())

    def test_host_facts_record_null_when_unavailable(self):
        def broken_loadavg():
            raise OSError("no load average here")

        with mock.patch.object(gate.os, "getloadavg", broken_loadavg):
            _code, _env, report, _out, _err, _calls = self._preflight(system="Linux")
        self.assertIsNone(report.preflight["host"]["load_average_1m"])
        self.assertEqual(report.preflight["host"]["system"], "Linux")

    def test_probe_exit_0_proceeds(self):
        code, environment, report, _out, err, calls = self._preflight()
        self.assertIsNone(code)
        self.assertIsNotNone(environment)
        self.assertIn("PYO3_PYTHON", environment)
        self.assertEqual(report.preflight["probe"]["verdict"], "ok")
        self.assertEqual(report.termination, "pass")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("warning", err)
        self.assertEqual(report.git["head"], CANNED_GIT_FACTS["head"])
        self.assertIn("platform", report.preflight["host"])

    def test_probe_exit_1_stops_with_exit_3_and_class(self):
        code, environment, report, _out, err, _calls = self._preflight(
            probe_result={
                "verdict": "wedged",
                "exit_code": 1,
                "output": "first exec timed out",
                "full_output": "first exec timed out\nsee runbook",
            }
        )
        self.assertEqual(code, gate.EXIT_PREFLIGHT_STOP)
        self.assertEqual(code, 3)
        self.assertIsNone(environment)
        self.assertEqual(report.termination, "preflight-stop")
        self.assertIn("preflight stop", err)
        self.assertIn(gate.PROBE_RUNBOOK, err)
        self.assertIn("first exec timed out", err)

    def test_probe_exit_2_and_3_warn_and_proceed(self):
        for verdict, exit_code in (("could-not-run", 2), ("slow", 3)):
            with self.subTest(verdict=verdict):
                code, _env, report, _out, err, _calls = self._preflight(
                    probe_result={"verdict": verdict, "exit_code": exit_code, "output": "x"}
                )
                self.assertIsNone(code)
                self.assertEqual(report.termination, "pass")
                self.assertIn("warning", err)
                self.assertEqual(report.preflight["probe"]["verdict"], verdict)
                self.assertEqual(report.preflight["probe"]["exit_code"], exit_code)

    def test_probe_timeout_and_launch_failure_are_could_not_run(self):
        def timing_out(*_args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="probe", timeout=kwargs.get("timeout", 0))

        result = gate.run_probe(Path(sys.executable), REPO_ROOT, probe_runner=timing_out)
        self.assertEqual(result["verdict"], "could-not-run")
        self.assertIsNone(result["exit_code"])
        self.assertIn("timed out", result["output"])

        def missing(*_args, **_kwargs):
            raise FileNotFoundError("no such interpreter")

        result = gate.run_probe(Path("/nope"), REPO_ROOT, probe_runner=missing)
        self.assertEqual(result["verdict"], "could-not-run")

    def test_probe_exit_codes_map_to_verdicts(self):
        for code, verdict in ((0, "ok"), (1, "wedged"), (2, "could-not-run"), (3, "slow"), (9, "could-not-run")):
            def runner(command, **_kwargs):
                return subprocess.CompletedProcess(command, code, stdout=f"line one code {code}\nmore", stderr="")

            with self.subTest(code=code):
                result = gate.run_probe(Path(sys.executable), REPO_ROOT, probe_runner=runner)
                self.assertEqual(result["verdict"], verdict)
                self.assertEqual(result["exit_code"], code)
                self.assertEqual(result["output"], f"line one code {code}")

    def test_probe_is_a_subprocess_of_the_bootstrap_free_script(self):
        seen = []

        def runner(command, **kwargs):
            seen.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, stdout="exec ok", stderr="")

        gate.run_probe(Path("/py"), REPO_ROOT, probe_runner=runner)
        self.assertEqual(seen[0][0], ["/py", "scripts/preflight_exec_probe.py"])
        self.assertEqual(seen[0][1]["timeout"], gate.PROBE_TIMEOUT_SECONDS)

    def test_probe_skipped_off_darwin(self):
        code, _env, report, _out, _err, calls = self._preflight(system="Linux")
        self.assertIsNone(code)
        self.assertEqual(calls, [])
        self.assertEqual(report.preflight["probe"], {"verdict": "skipped", "reason": "not darwin"})

    def test_missing_venv_warns_but_does_not_stop(self):
        code, environment, report, _out, err, _calls = self._preflight(venv=False)
        self.assertIsNone(code)
        self.assertIsNotNone(environment)
        self.assertFalse(report.preflight["venv_present"])
        self.assertIn(".venv/bin/python is missing", err)
        self.assertIn("uv venv --python 3.11", err)
        self.assertEqual(report.termination, "pass")

    def test_present_venv_is_recorded_without_a_warning(self):
        _code, _env, report, _out, err, _calls = self._preflight(venv=True)
        self.assertTrue(report.preflight["venv_present"])
        self.assertNotIn(".venv/bin/python is missing", err)

    def test_environment_failure_is_exit_2_before_any_git_or_probe_call(self):
        calls: list = []
        with mock.patch.object(
            gate, "_git_facts", lambda: self.fail("git must not run after an environment failure")
        ):
            code, environment, report, _out, err, probe_calls = self._preflight(
                environ_extra={"CARGO_TARGET_DIR": "/definitely/elsewhere"},
                probe_calls=calls,
            )
        self.assertEqual(code, gate.EXIT_ENVIRONMENT)
        self.assertEqual(code, 2)
        self.assertIsNone(environment)
        self.assertEqual(report.termination, "environment")
        self.assertEqual(probe_calls, [])
        self.assertIn("CARGO_TARGET_DIR", err)

    def test_git_facts_tolerate_a_missing_origin_main(self):
        def git_output(args):
            if args[0] == "rev-parse" and args[1] == "HEAD":
                return "abc123\n"
            raise subprocess.CalledProcessError(128, ["git", *args], stderr="unknown revision")

        with mock.patch.object(gate, "_git_output", git_output):
            facts = gate._git_facts()
        self.assertEqual(facts, {"head": "abc123", "origin_main": None, "merge_base": None})


class SummaryTests(unittest.TestCase):
    def _run_main(
        self, argv, *, diff="", status="", returncodes=None, raise_on=None,
        extra_patches=(), system="Darwin",
    ):
        fake_popen, launched = _popen_stub(returncodes, raise_on=raise_on)

        def fake_git_output(args):
            if args[0] == "diff":
                return diff
            if args[0] == "status":
                return status
            raise AssertionError(f"unexpected git invocation: {args}")

        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            patches = [
                mock.patch.object(gate.subprocess, "Popen", side_effect=fake_popen),
                mock.patch.object(gate, "_git_output", fake_git_output),
                mock.patch.object(gate, "_git_facts", lambda: dict(CANNED_GIT_FACTS)),
                mock.patch.object(gate, "run_probe", _canned_probe),
                mock.patch.object(gate, "host_system", lambda: system),
                mock.patch.object(gate, "workspace_member_packages", lambda: dict(CANNED_MEMBERS)),
                *extra_patches,
            ]
            for patch in patches:
                patch.start()
            try:
                with redirect_stdout(out), redirect_stderr(err):
                    rc = gate.main(argv, environ=_isolated_environ(tmp))
            finally:
                for patch in reversed(patches):
                    patch.stop()
            files = _summary_files(tmp)
            self.assertEqual(len(files), 1, f"expected one summary, found {files}")
            summary = json.loads(files[0].read_text(encoding="utf-8"))
            lease_paths = list(Path(tmp).glob("gate.lock*"))
        return rc, summary, launched, out.getvalue(), err.getvalue(), lease_paths

    def test_summary_written_on_pass(self):
        rc, summary, launched, out, _err, lease_paths = self._run_main(
            ["--fast"], diff="crates/chelis-cli/src/main.rs\n", status=" M crates/chelis-cli/src/main.rs\n"
        )
        self.assertEqual(rc, 0)
        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(summary["mode"], "fast")
        self.assertEqual(summary["termination"], "pass")
        self.assertEqual(summary["exit_code"], 0)
        expected = gate.fast_command_list(["chelis-cli"], std_changed=False)
        self.assertEqual(len(launched), len(expected))
        self.assertEqual(len(summary["stages"]), len(expected))
        for index, stage in enumerate(summary["stages"], start=1):
            self.assertEqual(stage["index"], index)
            self.assertGreaterEqual(stage["seconds"], 0)
            self.assertEqual(stage["returncode"], 0)
            self.assertIsNone(stage["launch_error"])
        self.assertIn("cargo clippy -p chelis-cli --tests -- -D warnings", summary["stages"][3]["command"])
        self.assertIsNone(summary["first_failing_stage"])
        self.assertEqual(summary["git"]["head"], CANNED_GIT_FACTS["head"])
        self.assertEqual(summary["git"]["selected_crates"], ["chelis-cli"])
        self.assertFalse(summary["git"]["std_changed"])
        self.assertTrue(summary["git"]["dirty"])
        self.assertEqual(summary["preflight"]["probe"]["verdict"], "ok")
        # --fast never takes the lease: no lock file, mode not-taken.
        self.assertEqual(summary["lease"]["mode"], "not-taken")
        self.assertEqual(lease_paths, [])
        self.assertIn("gate --fast: changed crates vs origin/main: chelis-cli", out)
        self.assertTrue(summary["report_path"].endswith("-fast.json"))
        self.assertGreaterEqual(summary["seconds"], 0)
        # `python` names the interpreter the children ran (the exported
        # PYO3_PYTHON), which here is the gate's own executable.
        self.assertEqual(summary["python"], sys.executable)
        self.assertEqual(summary["runner_python"], sys.executable)
        self.assertIsNone(summary["files_changed_note"])

    def test_linux_summary_records_the_probe_as_skipped(self):
        # The script-unit CI job runs these tests on Linux; the preflight must
        # not depend on the host it happens to run on.
        def must_not_run(*_a, **_k):
            self.fail("the exec probe must not run off Darwin")

        rc, summary, _launched, _out, err, _lease = self._run_main(
            ["--fast"], system="Linux",
            extra_patches=[mock.patch.object(gate, "run_probe", must_not_run)],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(summary["preflight"]["probe"], {"verdict": "skipped", "reason": "not darwin"})
        self.assertEqual(summary["preflight"]["host"]["system"], "Linux")
        self.assertIn("load_average_1m", summary["preflight"]["host"])
        self.assertNotIn("preflight stop", err)

    def test_std_path_change_appends_the_std_legs(self):
        rc, summary, launched, out, _err, _lease = self._run_main(
            ["--fast"], diff="packages/chelis-std/src/time.ch\n"
        )
        self.assertEqual(rc, 0)
        self.assertTrue(summary["git"]["std_changed"])
        self.assertEqual(summary["git"]["selected_crates"], [])
        rendered = [" ".join(c) for c in launched]
        self.assertTrue(rendered[1].endswith("scripts/regen_all.py --tier 1"), rendered[1])
        self.assertTrue(rendered[-1].startswith("cargo nextest run -p chelis-std-bundle --lib"))
        self.assertEqual(len(rendered), 6)
        self.assertIn("chelis-std paths changed", out)
        self.assertIn("no crate changes detected", out)
        self.assertIn("per-crate clippy", out)

    def test_summary_written_on_stage_failure(self):
        rc, summary, launched, _out, err, _lease = self._run_main(
            ["--fast"], returncodes={2: 101}
        )
        self.assertEqual(rc, 101)
        self.assertEqual(summary["termination"], "stage-failure")
        self.assertEqual(summary["exit_code"], 101)
        self.assertEqual(len(launched), 2)
        self.assertEqual(len(summary["stages"]), 2)
        self.assertEqual(summary["first_failing_stage"]["index"], 2)
        self.assertEqual(summary["first_failing_stage"]["returncode"], 101)
        self.assertEqual(summary["first_failing_stage"]["command"], "cargo fmt --all")
        self.assertIn("gate-failures", summary["first_failing_stage"]["transcript"])
        self.assertIn("fast command 2/", err)

    def test_signal_death_is_its_own_termination_class(self):
        rc, summary, _launched, _out, _err, _lease = self._run_main(
            ["--fast"], returncodes={1: -9}
        )
        self.assertEqual(rc, 137)
        self.assertEqual(summary["termination"], "signal")
        self.assertEqual(summary["exit_code"], 137)
        self.assertEqual(summary["first_failing_stage"]["returncode"], -9)

    def test_summary_written_on_preflight_stop(self):
        wedged = {"verdict": "wedged", "exit_code": 1, "output": "wedged", "full_output": "wedged"}
        rc, summary, launched, _out, err, _lease = self._run_main(
            ["--fast"],
            extra_patches=[mock.patch.object(gate, "run_probe", lambda *a, **k: dict(wedged))],
        )
        self.assertEqual(rc, 3)
        self.assertEqual(summary["termination"], "preflight-stop")
        self.assertEqual(summary["stages"], [])
        self.assertEqual(launched, [])
        self.assertEqual(summary["preflight"]["probe"]["verdict"], "wedged")
        self.assertIn(gate.PROBE_RUNBOOK, err)

    def test_summary_written_on_keyboard_interrupt(self):
        rc, summary, launched, out, err, lease_paths = self._run_main(["--local"], raise_on=2)
        self.assertEqual(rc, 130)
        self.assertEqual(summary["termination"], "user-cancel")
        self.assertEqual(summary["exit_code"], 130)
        self.assertEqual(len(launched), 2)
        self.assertEqual(len(summary["stages"]), 1)
        self.assertIn("cancelled", err)
        # The lease taken by --local is released in main's finally.
        self.assertEqual(summary["lease"]["mode"], "held")
        self.assertEqual([p.name for p in lease_paths], ["gate.lock"])
        self.assertIn("USER-CANCEL", out)

    def test_summary_written_on_git_failure(self):
        def failing_git_output(args):
            raise subprocess.CalledProcessError(128, ["git", *args], stderr="fatal: bad revision")

        rc, summary, launched, _out, err, _lease = self._run_main(
            ["--fast"],
            extra_patches=[mock.patch.object(gate, "_git_output", failing_git_output)],
        )
        self.assertEqual(rc, 128)
        self.assertEqual(summary["termination"], "environment")
        self.assertEqual(launched, [])
        self.assertIn("gate --fast: git failed", err)

    def test_local_summary_records_the_held_lease_and_releases_it(self):
        rc, summary, launched, _out, _err, lease_paths = self._run_main(
            ["--local"], diff="crates/chelis-surf/src/lib.rs\n"
        )
        self.assertEqual(rc, 0)
        self.assertEqual(summary["mode"], "local")
        self.assertEqual(summary["lease"]["mode"], "held")
        self.assertEqual(summary["lease"]["wait_seconds"], 0.0)
        self.assertIsNone(summary["lease"]["holder_seen"])
        self.assertEqual(len(launched), len(gate.local_command_list(["chelis-surf"])))
        # Lock file persists (it is just an inode to flock); the sidecar is gone.
        self.assertEqual([p.name for p in lease_paths], ["gate.lock"])
        self.assertEqual(summary["files_changed_by_run"], [])

    def test_stage_runs_get_a_summary_but_no_preflight_or_lease(self):
        rc, summary, launched, _out, _err, lease_paths = self._run_main(["lint-and-unit"])
        self.assertEqual(rc, 0)
        self.assertEqual(summary["mode"], "lint-and-unit")
        self.assertEqual(summary["preflight"], {})
        self.assertEqual(summary["lease"], {})
        self.assertEqual(lease_paths, [])
        self.assertEqual(len(launched), len(gate.STAGES["lint-and-unit"]))
        self.assertEqual(summary["git"]["head"], CANNED_GIT_FACTS["head"])

    def test_no_lease_is_recorded_as_bypassed(self):
        rc, summary, _launched, _out, _err, lease_paths = self._run_main(["--local", "--no-lease"])
        self.assertEqual(rc, 0)
        self.assertEqual(summary["lease"]["mode"], "bypassed")
        self.assertEqual(lease_paths, [])

    def test_files_changed_by_run_uses_content_hashes(self):
        snapshots = iter(
            [
                {"a.rs": "h1", "b.rs": "h2", "gone.rs": "h9"},
                {"a.rs": "h1-changed", "b.rs": "h2", "c.rs": "h3"},
            ]
        )
        with mock.patch.object(gate, "_porcelain_hashes", lambda *a, **k: next(snapshots)):
            rc, summary, _launched, out, _err, _lease = self._run_main(["--fast"])
        self.assertEqual(rc, 0)
        self.assertEqual(summary["files_changed_by_run"], ["a.rs", "c.rs", "gone.rs"])
        self.assertIn("changed 3 file(s) in place", out)
        self.assertIn("  a.rs", out)

    def test_git_failure_after_the_run_records_null_changed_files_not_an_error(self):
        snapshots = [{"a.rs": "h1"}]

        def hashes(*_a, **_k):
            if snapshots:
                return snapshots.pop()
            raise subprocess.CalledProcessError(128, ["git", "status"], stderr="fatal: index locked")

        with mock.patch.object(gate, "_porcelain_hashes", hashes):
            rc, summary, _launched, _out, err, _lease = self._run_main(["--fast"])
        self.assertEqual(rc, 0)
        self.assertEqual(summary["termination"], "pass")
        self.assertIsNone(summary["files_changed_by_run"])
        self.assertIn("index locked", summary["files_changed_note"])
        self.assertIn("gate --fast: warning: git status failed after the run", err)

    def test_porcelain_hashes_see_content_not_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.rs").write_text("fn a() {}\n")
            (root / "new").mkdir()
            (root / "new" / "c.rs").write_text("c\n")
            status = " M a.rs\n?? new/c.rs\n D deleted.rs\n"
            with mock.patch.object(gate, "_git_output", lambda args: status):
                before = gate._porcelain_hashes(repo_root=root)
                (root / "a.rs").write_text("fn a() {\n}\n")  # still ` M`
                after = gate._porcelain_hashes(repo_root=root)
        self.assertEqual(set(before), {"a.rs", "new/c.rs", "deleted.rs"})
        self.assertIsNone(before["deleted.rs"])
        self.assertNotEqual(before["a.rs"], after["a.rs"])
        self.assertEqual(before["new/c.rs"], after["new/c.rs"])
        self.assertEqual(gate.files_changed_between(before, after), ["a.rs"])

    def test_human_summary_line_names_stage_count_seconds_verdict_and_report(self):
        report = gate.GateReport(mode="fast", started_at="now")
        report.stages.append(gate.StageRecord(1, "cargo fmt --all", 0.5, 0))
        report.stages.append(gate.StageRecord(2, "x", 1.0, 0))
        report.files_changed_by_run = ["crates/x/src/a.rs"]
        line = gate.human_summary(report, REPO_ROOT / "target/gate-reports/r.json", REPO_ROOT)
        self.assertTrue(line.startswith("gate --fast: 2 stages, "), line)
        self.assertIn(" s, PASS", line)
        self.assertIn("changed: 1 file(s) (crates/x/src/a.rs)", line)
        self.assertTrue(line.endswith("report: target/gate-reports/r.json"), line)

        report.termination = "stage-failure"
        report.exit_code = 101
        report.stages.append(gate.StageRecord(3, "cargo clippy -p x --tests -- -D warnings", 2.0, 101))
        line = gate.human_summary(report, None, REPO_ROOT)
        self.assertIn("FAIL (stage 3: cargo clippy -p x --tests -- -D warnings; exit code: 101)", line)
        self.assertNotIn("report:", line)

        unknown = gate.GateReport(mode="fast", started_at="now")
        unknown.files_changed_by_run = None
        self.assertIn("changed: unknown (git status failed after the run)", gate.human_summary(unknown, None, REPO_ROOT))

        local = gate.GateReport(mode="local", started_at="now")
        local.termination = "lease-timeout"
        local.exit_code = 4
        line = gate.human_summary(local, None, REPO_ROOT)
        self.assertTrue(line.startswith("gate --local: 0 stages, "), line)
        self.assertIn("LEASE-TIMEOUT (exit 4)", line)
        self.assertNotIn("changed:", line)

    def test_report_directory_defaults_under_target_and_honors_the_override(self):
        self.assertEqual(
            gate.report_directory({}, Path("/w")), Path("/w/target/gate-reports")
        )
        self.assertEqual(
            gate.report_directory({gate.REPORT_DIR_ENV: "/elsewhere"}, Path("/w")),
            Path("/elsewhere"),
        )
        self.assertEqual(
            gate.report_directory({gate.REPORT_DIR_ENV: "rel/reports"}, Path("/w")),
            Path("/w/rel/reports"),
        )


class LeaseTests(unittest.TestCase):
    def _lease(self, path: Path, **overrides) -> "gate.GateLease":
        settings = dict(
            mode="local",
            worktree=Path("/wt"),
            head="abc",
            wait=False,
            timeout=None,
            output=io.StringIO(),
        )
        settings.update(overrides)
        return gate.GateLease(path, **settings)

    def test_lease_dir_resolution_order(self):
        self.assertEqual(gate.lease_dir({gate.LEASE_DIR_ENV: "/x"}), Path("/x"))
        self.assertEqual(gate.lease_dir({"XDG_CACHE_HOME": "/cache"}), Path("/cache/chelis"))
        self.assertEqual(gate.lease_dir({}), Path.home() / ".cache" / "chelis")

    def test_acquire_writes_sidecar_and_release_removes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / gate.LEASE_FILE_NAME
            lease = self._lease(path)
            lease.acquire()
            try:
                self.assertTrue(lease.held)
                sidecar = json.loads((Path(tmp) / "nested" / "gate.lock.json").read_text())
                self.assertEqual(sidecar["pid"], os.getpid())
                self.assertEqual(sidecar["mode"], "local")
                self.assertEqual(sidecar["worktree"], "/wt")
                self.assertEqual(sidecar["head"], "abc")
                self.assertEqual(sidecar["schema_version"], 1)
                self.assertTrue(sidecar["started_at"].endswith("Z"))
                self.assertEqual(gate.GateLease.current_holder(path)["pid"], os.getpid())
            finally:
                lease.release()
            self.assertFalse(lease.held)
            self.assertFalse((Path(tmp) / "nested" / "gate.lock.json").exists())
            self.assertTrue(path.exists())
            self.assertIsNone(gate.GateLease.peek(path))

    def test_second_fd_blocks_while_first_holds(self):
        # flock locks belong to the open file description, so two separate
        # opens in one process conflict; no second process is needed.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gate.LEASE_FILE_NAME
            first = self._lease(path)
            first.acquire()
            try:
                second = self._lease(path, mode="full")
                with self.assertRaises(gate.LeaseHeld) as raised:
                    second.acquire()
                self.assertEqual(raised.exception.holder["pid"], os.getpid())
                self.assertEqual(raised.exception.holder["mode"], "local")
                self.assertFalse(second.held)
                self.assertEqual(gate.GateLease.peek(path)["pid"], os.getpid())
                # The failed acquirer must not have disturbed the holder's sidecar.
                self.assertEqual(gate.GateLease.current_holder(path)["mode"], "local")
            finally:
                first.release()
            self._lease(path).acquire()

    def test_waiter_acquires_after_holder_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gate.LEASE_FILE_NAME
            holder = self._lease(path)
            holder.acquire()
            out = io.StringIO()
            sleeps: list[float] = []
            clock = iter(float(t) for t in range(0, 1000, 7))

            def fake_sleep(seconds):
                sleeps.append(seconds)
                if len(sleeps) == 2:
                    holder.release()

            waiter = self._lease(
                path, wait=True, output=out, sleep=fake_sleep, clock=lambda: next(clock),
                poll_seconds=10, heartbeat_seconds=60,
            )
            waiter.acquire()
            try:
                self.assertTrue(waiter.held)
                self.assertEqual(sleeps, [10, 10])
                self.assertGreater(waiter.wait_seconds, 0)
                self.assertEqual(waiter.holder_seen["pid"], os.getpid())
                text = out.getvalue()
                self.assertEqual(text.count("waiting for the gate lease"), 1)
                self.assertIn(f"pid {os.getpid()}", text)
                self.assertEqual(json.loads((path.with_name("gate.lock.json")).read_text())["mode"], "local")
            finally:
                waiter.release()

    def test_heartbeat_repeats_every_heartbeat_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gate.LEASE_FILE_NAME
            holder = self._lease(path)
            holder.acquire()
            out = io.StringIO()
            polls = 0
            ticks = iter(float(t) for t in range(0, 100000, 10))

            def fake_sleep(_seconds):
                nonlocal polls
                polls += 1
                if polls == 15:
                    holder.release()

            waiter = self._lease(
                path, wait=True, output=out, sleep=fake_sleep, clock=lambda: next(ticks),
                poll_seconds=10, heartbeat_seconds=60,
            )
            waiter.acquire()
            waiter.release()
        text = out.getvalue()
        self.assertEqual(text.count("waiting for the gate lease"), 1)
        self.assertGreaterEqual(text.count("still waiting"), 2)

    def test_lease_timeout_is_honoured_to_the_second_not_the_poll(self):
        # `--lease-timeout 3` must exit after 3 s of waiting, not after the
        # next 10 s poll boundary: the sleep is clamped to the remaining time.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gate.LEASE_FILE_NAME
            holder = self._lease(path)
            holder.acquire()
            try:
                now = [0.0]
                sleeps: list[float] = []

                def fake_sleep(seconds):
                    sleeps.append(seconds)
                    now[0] += seconds

                waiter = self._lease(
                    path, wait=True, timeout=3.0, sleep=fake_sleep,
                    clock=lambda: now[0], poll_seconds=10,
                )
                with self.assertRaises(gate.LeaseHeld) as raised:
                    waiter.acquire()
                self.assertEqual(sum(sleeps), 3.0, sleeps)
                self.assertEqual(sleeps, [3.0])
                self.assertEqual(raised.exception.waited, 3.0)
            finally:
                holder.release()

    def test_lease_timeout_longer_than_a_poll_still_polls_then_clamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gate.LEASE_FILE_NAME
            holder = self._lease(path)
            holder.acquire()
            try:
                now = [0.0]
                sleeps: list[float] = []

                def fake_sleep(seconds):
                    sleeps.append(seconds)
                    now[0] += seconds

                waiter = self._lease(
                    path, wait=True, timeout=25.0, sleep=fake_sleep,
                    clock=lambda: now[0], poll_seconds=10,
                )
                with self.assertRaises(gate.LeaseHeld):
                    waiter.acquire()
                self.assertEqual(sleeps, [10.0, 10.0, 5.0])
                self.assertEqual(now[0], 25.0)
            finally:
                holder.release()

    def test_peek_takes_a_shared_lock_never_an_exclusive_one(self):
        operations: list[int] = []
        real_flock = gate.fcntl.flock

        def recording_flock(fd, operation):
            operations.append(operation)
            return real_flock(fd, operation)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gate.LEASE_FILE_NAME
            path.write_text("")
            with mock.patch.object(gate.fcntl, "flock", recording_flock):
                self.assertIsNone(gate.GateLease.peek(path))
        self.assertEqual(
            operations,
            [gate.fcntl.LOCK_SH | gate.fcntl.LOCK_NB, gate.fcntl.LOCK_UN],
        )

    def test_acquire_retries_once_before_announcing_an_unknown_holder(self):
        # A holder between its flock and its sidecar write, or a --fast peek
        # holding LOCK_SH for microseconds, blocks the first attempt with no
        # readable sidecar. One short retry must settle it silently.
        attempts = iter([False, True])
        out = io.StringIO()
        sleeps: list[float] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gate.LEASE_FILE_NAME
            lease = self._lease(path, wait=True, output=out, sleep=sleeps.append)
            with mock.patch.object(
                gate.GateLease, "_try_flock", staticmethod(lambda fd, operation=gate.fcntl.LOCK_EX: next(attempts))
            ):
                lease.acquire()
            lease.release()
        self.assertEqual(sleeps, [gate.TRANSIENT_RETRY_SECONDS])
        self.assertEqual(out.getvalue(), "")

    def test_sidecar_write_failure_releases_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gate.LEASE_FILE_NAME
            # A directory where the sidecar must go makes the O_EXCL create fail.
            path.with_name("gate.lock.json").mkdir()
            lease = self._lease(path)
            with self.assertRaises(OSError):
                lease.acquire()
            self.assertFalse(lease.held)
            self.assertIsNone(lease._fd)
            # Nobody holds the flock afterwards.
            self.assertIsNone(gate.GateLease.peek(path))
            other = self._lease(path)
            with self.assertRaises(OSError):
                other.acquire()  # same directory obstacle, but no lock held
            self.assertIsNone(gate.GateLease.peek(path))

    def test_timeout_exits_with_lease_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gate.LEASE_FILE_NAME
            holder = self._lease(path)
            holder.acquire()
            try:
                ticks = iter([0.0, 4.0, 8.0, 12.0, 16.0])
                waiter = self._lease(
                    path, wait=True, timeout=10.0, sleep=lambda _s: None,
                    clock=lambda: next(ticks), poll_seconds=4,
                )
                with self.assertRaises(gate.LeaseHeld) as raised:
                    waiter.acquire()
                self.assertGreaterEqual(raised.exception.waited, 10.0)
                self.assertFalse(waiter.held)
            finally:
                holder.release()

    def _take(self, tmp: str, mode: str, **flags):
        args = argparse.Namespace(no_lease=False, no_wait=False, lease_timeout=None)
        for key, value in flags.items():
            setattr(args, key, value)
        report = gate.GateReport(mode=mode, started_at="now")
        report.git["head"] = "deadbeef"
        out, err = io.StringIO(), io.StringIO()
        code, lease = gate.take_lease(
            mode=mode, args=args, report=report, environ={gate.LEASE_DIR_ENV: tmp},
            repo_root=Path("/wt"), output_stream=out, error_stream=err,
        )
        return code, lease, report, out.getvalue(), err.getvalue()

    def test_no_wait_exits_4_with_holder_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            holder = self._lease(Path(tmp) / gate.LEASE_FILE_NAME, worktree=Path("/other"))
            holder.acquire()
            try:
                code, lease, report, _out, err = self._take(tmp, "local", no_wait=True)
            finally:
                holder.release()
        self.assertEqual(code, gate.EXIT_LEASE_TIMEOUT)
        self.assertEqual(code, 4)
        self.assertIsNone(lease)
        self.assertEqual(report.termination, "lease-timeout")
        self.assertEqual(report.lease["mode"], "timed-out")
        self.assertEqual(report.lease["holder_seen"]["worktree"], "/other")
        self.assertIn(f"pid {os.getpid()}", err)
        self.assertIn("/other", err)
        self.assertIn("--no-wait was given", err)
        self.assertIn("--no-lease", err)

    def test_timeout_flag_exits_4(self):
        with tempfile.TemporaryDirectory() as tmp:
            holder = self._lease(Path(tmp) / gate.LEASE_FILE_NAME)
            holder.acquire()
            try:
                with mock.patch.object(gate, "LEASE_POLL_SECONDS", 0.01):
                    code, _lease, report, _out, err = self._take(tmp, "full", lease_timeout=0.02)
            finally:
                holder.release()
        self.assertEqual(code, 4)
        self.assertEqual(report.lease["mode"], "timed-out")
        self.assertIn("--lease-timeout 0.02 elapsed", err)

    def test_no_lease_bypasses(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, lease, report, _out, _err = self._take(tmp, "local", no_lease=True)
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertIsNone(code)
        self.assertIsNone(lease)
        self.assertEqual(report.lease["mode"], "bypassed")

    def test_local_takes_and_records_the_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, lease, report, _out, _err = self._take(tmp, "local")
            try:
                self.assertIsNone(code)
                self.assertTrue(lease.held)
                self.assertEqual(report.lease["mode"], "held")
                self.assertEqual(report.lease["path"], str(Path(tmp) / "gate.lock"))
                holder = gate.GateLease.current_holder(Path(tmp) / "gate.lock")
                self.assertEqual(holder["head"], "deadbeef")
                self.assertEqual(holder["worktree"], "/wt")
            finally:
                lease.release()

    def test_stale_sidecar_without_lock_is_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gate.LEASE_FILE_NAME
            path.write_text("")
            path.with_name("gate.lock.json").write_text(
                json.dumps({"pid": 1, "worktree": "/dead", "mode": "local"})
            )
            self.assertIsNone(gate.GateLease.peek(path))
            lease = self._lease(path)
            lease.acquire()
            try:
                holder = gate.GateLease.current_holder(path)
                self.assertEqual(holder["pid"], os.getpid())
                self.assertEqual(holder["worktree"], "/wt")
            finally:
                lease.release()

    def test_partial_sidecar_reads_as_unknown_holder(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / gate.LEASE_FILE_NAME
            holder = self._lease(path)
            holder.acquire()
            try:
                path.with_name("gate.lock.json").write_text("{not json")
                self.assertEqual(gate.GateLease.peek(path), {})
                self.assertIn("unknown holder", gate.describe_holder(None))
                with self.assertRaises(gate.LeaseHeld) as raised:
                    self._lease(path).acquire()
                self.assertIsNone(raised.exception.holder)
            finally:
                holder.release()

    def test_fast_reports_holder_but_does_not_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            holder = self._lease(Path(tmp) / gate.LEASE_FILE_NAME, worktree=Path("/busy"))
            holder.acquire()
            try:
                code, lease, report, out, _err = self._take(tmp, "fast")
            finally:
                holder.release()
        self.assertIsNone(code)
        self.assertIsNone(lease)
        self.assertEqual(report.lease["mode"], "not-taken")
        self.assertEqual(report.lease["holder_seen"]["worktree"], "/busy")
        self.assertIn("a full gate is running", out)
        self.assertIn("/busy", out)

    def test_fast_with_no_holder_prints_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, lease, report, out, _err = self._take(tmp, "fast")
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertIsNone(code)
        self.assertIsNone(lease)
        self.assertEqual(report.lease["mode"], "not-taken")
        self.assertIsNone(report.lease["holder_seen"])
        self.assertEqual(out, "")

    def test_unwritable_lease_dir_proceeds_without_the_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not-a-dir"
            blocker.write_text("")
            code, lease, report, _out, err = self._take(str(blocker), "local")
        self.assertIsNone(code)
        self.assertIsNone(lease)
        self.assertEqual(report.lease["mode"], "bypassed")
        self.assertIn("proceeding without it", err)

    def test_stage_runs_never_open_the_lease_file(self):
        fake_popen, launched = _popen_stub()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(gate.subprocess, "Popen", side_effect=fake_popen), \
                mock.patch.object(gate, "_git_facts", lambda: dict(CANNED_GIT_FACTS)), \
                redirect_stdout(io.StringIO()):
            rc = gate.main(["integration", "--support-only"], environ=_isolated_environ(tmp))
            self.assertEqual(rc, 0)
            self.assertEqual(list(Path(tmp).glob("gate.lock*")), [])
            self.assertEqual(len(launched), len(gate.STAGES["integration"]) - 1)


class DocumentationLockTests(unittest.TestCase):
    def test_agents_md_does_not_transcribe_the_list(self):
        # AGENTS.md points at `gate.py --list` instead of copying its output;
        # a transcription drifts (the old one omitted six live rows).
        text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        offenders = [
            line for line in text.splitlines() if line.startswith("# cargo clippy --workspace")
        ]
        self.assertEqual(offenders, [])
        self.assertIn("python3 scripts/gate.py --fast", text)
        self.assertIn("python3 scripts/gate.py --list", text)


if __name__ == "__main__":
    unittest.main()
