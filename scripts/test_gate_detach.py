"""Unit tests for `gate.py --detach` / `--status` and the reaper protection.

Run via: `python3 -m unittest scripts.test_gate_detach` from the repo root.

A separate file from `test_gate_fast.py` (already 1,154 lines) so the detach
surface can grow without touching the fast-gate suite.

What is locked here:

  (a) argument shape: which combinations parse and which `p.error`. `--detach
      --fast` is rejected on purpose, because `--fast` fixes in place and a
      writer running unattended against a tree the agent is still editing is
      the collision chelis#1568 documents;
  (b) the spawn: exactly one `Popen`, with `--detach` stripped from the child
      argv, `start_new_session=True`, stdin closed, stderr merged, and the
      repo root as cwd;
  (c) the launcher is not a run: it writes NO run summary and takes NO lease,
      so `gate.py`'s "exactly one summary per run" invariant survives;
  (d) `--status`: the run's own exit code once it has one, 75 while it is
      alive, 1 when it died without writing a summary, 2 for a handle problem.
      Exit 4 moving from the shell into `--status` is covered explicitly,
      because that is the one behaviour change a caller can be surprised by;
  (e) that the child cannot be re-executed through uv, since a grandchild
      would write a summary keyed on a pid the handle does not know;
  (f) the reaper protection: a detached gate pid and its descendants are
      excluded from the orphan set, a recycled pid is not protected, and the
      two copies of the report-directory spelling agree.

Not executed here: no real gate run is started, no real lease outside a
temporary directory, and no process is signalled.
"""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path


def _load(name: str, filename: str):
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(name, here / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load("gate_under_detach_tests", "gate.py")
reap = _load("reap_under_detach_tests", "reap_orphans.py")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXED_NOW = datetime(2026, 9, 5, 3, 15, 0, 123456, tzinfo=timezone.utc)
STAMP = "20260905T031500.123456Z"


def _isolated_environ(tmp: str) -> dict:
    return {
        "PATH": os.environ.get("PATH", ""),
        gate.REPORT_DIR_ENV: tmp,
        gate.LEASE_DIR_ENV: tmp,
    }


def _parse_quietly(argv):
    with redirect_stderr(io.StringIO()):
        return gate.parse_args(argv)


def _summary_files(tmp: str) -> list[Path]:
    return sorted(Path(tmp).glob("*.json"))


class _PopenStub:
    """A `Popen` stand-in that records how it was called."""

    def __init__(self, pid: int = 41277):
        self.pid = pid
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        return self

    @property
    def last_kwargs(self) -> dict:
        return self.calls[-1][1]

    @property
    def last_argv(self) -> list[str]:
        return self.calls[-1][0]


def _spawn(environ, tmp, *, argv=None, popen=None, mode="validation", pid=41277):
    stub = _PopenStub(pid) if popen is None else popen
    payload = gate.spawn_detached(
        argv if argv is not None else ["--validation"],
        environ=environ,
        executable=Path("/managed/python"),
        directory=gate.detach_directory(environ, Path(tmp)),
        mode=mode,
        repo_root=Path(tmp),
        now=lambda: FIXED_NOW,
        popen=stub,
        git_facts=lambda: {"head": "a" * 40},
    )
    return payload, stub


class ArgumentShapeTests(unittest.TestCase):
    def test_detach_parses_with_local_and_alone(self):
        self.assertTrue(_parse_quietly(["--detach", "--validation"]).detach)
        self.assertTrue(_parse_quietly(["--detach"]).detach)

    def test_detach_forwards_every_lease_flag(self):
        for flag in (["--no-wait"], ["--no-lease"], ["--lease-timeout", "30"]):
            with self.subTest(flag=flag):
                args = _parse_quietly(["--detach", "--validation", *flag])
                self.assertTrue(args.detach)

    def test_status_defaults_to_latest_and_accepts_a_path(self):
        self.assertEqual(_parse_quietly(["--status"]).status, "latest")
        self.assertEqual(_parse_quietly(["--status", "/x/h.json"]).status, "/x/h.json")

    def test_status_consumes_its_optional_argument_as_a_handle_path(self):
        """`--status` takes an optional value, so a bare word after it is the
        handle, never a stage name. `<stage> --status` is the spelling that
        genuinely conflicts, and it is rejected."""
        args = _parse_quietly(["--status", "integration"])
        self.assertEqual(args.status, "integration")
        self.assertIsNone(args.stage)
        with self.assertRaises(SystemExit):
            _parse_quietly(["integration", "--status"])

    def test_rejected_combinations(self):
        for argv in (
            ["--detach", "--fast"],
            ["--detach", "--list"],
            ["--detach", "lint-and-unit"],
            ["--detach", "--status"],
            ["--status", "--fast"],
            ["--status", "--validation"],
            ["--status", "--list"],
            ["integration", "--status"],
            ["--status", "--no-wait"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    _parse_quietly(argv)

    def test_detach_with_fast_names_the_reason(self):
        stderr = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(stderr):
            gate.parse_args(["--detach", "--fast"])
        self.assertIn("writes to the worktree", stderr.getvalue())


class SpawnTests(unittest.TestCase):
    def test_exactly_one_spawn_with_the_expected_argv_and_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp:
            environ = _isolated_environ(tmp)
            payload, stub = _spawn(environ, tmp, argv=["--validation"])
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(stub.last_argv[0], "/managed/python")
        self.assertEqual(stub.last_argv[1], str(Path(gate.__file__).resolve()))
        self.assertEqual(stub.last_argv[2:], ["--validation"])
        kwargs = stub.last_kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(kwargs["cwd"], tmp)
        self.assertTrue(kwargs["close_fds"])
        self.assertEqual(payload["pid"], 41277)

    def test_detach_is_stripped_and_everything_else_passes_through(self):
        self.assertEqual(
            gate.detach_child_argv(
                ["--detach", "--validation", "--lease-timeout", "30", "--no-wait"]
            ),
            ["--validation", "--lease-timeout", "30", "--no-wait"],
        )

    def test_the_log_is_created_and_the_handle_names_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, _ = _spawn(_isolated_environ(tmp), tmp)
            log = Path(payload["log"])
            self.assertTrue(log.is_file())
            self.assertEqual(log.name, f"{STAMP}-{os.getpid()}.log")
            self.assertEqual(
                Path(payload["handle"]).name, f"{STAMP}-{os.getpid()}.json"
            )

    def test_handle_round_trips_and_newest_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, _ = _spawn(_isolated_environ(tmp), tmp)
            path = gate.write_handle(payload, Path(payload["handle"]))
            self.assertEqual(gate.read_handle(path), payload)
            directory = gate.detach_directory(_isolated_environ(tmp), Path(tmp))
            (directory / "20990101T000000.0Z-9.json").write_text(
                json.dumps({"pid": 5}), encoding="utf-8"
            )
            self.assertEqual(gate.newest_handle(directory).name,
                             "20990101T000000.0Z-9.json")


class LauncherTests(unittest.TestCase):
    def _run(self, tmp, argv=("--detach", "--validation"), popen=None):
        environ = _isolated_environ(tmp)
        out, err = io.StringIO(), io.StringIO()
        stub = _PopenStub() if popen is None else popen

        def spawn(child_argv, **kwargs):
            kwargs.pop("popen", None)
            return gate.spawn_detached(
                child_argv,
                popen=stub,
                now=lambda: FIXED_NOW,
                git_facts=lambda: {"head": "a" * 40},
                **kwargs,
            )

        code = gate.run_detach(
            _parse_quietly(list(argv)),
            argv=list(argv),
            environ=environ,
            executable=Path("/managed/python"),
            mode="validation",
            repo_root=Path(tmp),
            output_stream=out,
            error_stream=err,
            spawn=spawn,
        )
        return code, out.getvalue(), err.getvalue(), stub

    def test_launcher_returns_zero_and_prints_the_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _err, _stub = self._run(tmp)
            self.assertEqual(code, 0)
            self.assertIn("pid 41277", out)
            self.assertIn("gate: log:", out)
            self.assertIn("gate: handle:", out)
            self.assertIn("--status", out)

    def test_launcher_writes_no_run_summary_and_takes_no_lease(self):
        """The launcher is not a gate run. If it wrote a summary, `gate.py`'s
        one-summary-per-run invariant would break; if it took the lease, the
        child could never acquire it."""
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            self.assertEqual(_summary_files(tmp), [])
            self.assertFalse((Path(tmp) / gate.LEASE_FILE_NAME).exists())
            self.assertFalse(
                (Path(tmp) / (gate.LEASE_FILE_NAME + ".json")).exists()
            )

    def test_a_handle_that_cannot_be_written_still_names_the_live_child(self):
        """Red-team round 1 P3. The child is already running and detached; a
        bare "could not start" would be untrue and would leave it unfindable."""
        with tempfile.TemporaryDirectory() as tmp:
            environ = _isolated_environ(tmp)
            stub = _PopenStub()
            out, err = io.StringIO(), io.StringIO()

            def spawn(child_argv, **kwargs):
                payload = gate.spawn_detached(
                    child_argv,
                    popen=stub,
                    now=lambda: FIXED_NOW,
                    git_facts=lambda: {"head": "a" * 40},
                    **kwargs,
                )
                # Make the handle path unwritable by pointing it at a directory.
                Path(payload["handle"]).mkdir(parents=True, exist_ok=True)
                return payload

            code = gate.run_detach(
                _parse_quietly(["--detach", "--validation"]),
                argv=["--detach", "--validation"],
                environ=environ,
                executable=Path("/managed/python"),
                mode="validation",
                repo_root=Path(tmp),
                output_stream=out,
                error_stream=err,
                spawn=spawn,
            )
        message = err.getvalue()
        self.assertEqual(code, gate.EXIT_ENVIRONMENT)
        self.assertIn("pid 41277", message)
        self.assertIn("handle could not be written", message)
        self.assertNotIn("could not start a detached run", message)

    def test_a_failed_spawn_exits_environment_and_leaves_no_handle(self):
        def exploding(*_args, **_kwargs):
            raise OSError("no fork for you")

        with tempfile.TemporaryDirectory() as tmp:
            code, _out, err, _stub = self._run(tmp, popen=exploding)
            self.assertEqual(code, gate.EXIT_ENVIRONMENT)
            self.assertIn("no fork for you", err)
            directory = gate.detach_directory(_isolated_environ(tmp), Path(tmp))
            self.assertEqual(list(directory.glob("*.json")), [])


class StatusTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.environ = _isolated_environ(self.tmp)
        self.addCleanup(self._tmp.cleanup)
        payload, _ = _spawn(self.environ, self.tmp)
        self.handle_path = gate.write_handle(payload, Path(payload["handle"]))
        self.payload = payload

    def _summary(self, **fields):
        body = {
            "mode": "validation",
            "termination": "pass",
            "exit_code": 0,
            "seconds": 471.2,
        }
        body.update(fields)
        path = Path(self.tmp) / f"20260905T032000.0Z-{self.payload['pid']}-validation.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def _status(self, selector=None, alive=lambda _pid: False):
        args = _parse_quietly(
            ["--status"] if selector is None else ["--status", str(selector)]
        )
        out, err = io.StringIO(), io.StringIO()
        code = gate.run_status(
            args,
            environ=self.environ,
            repo_root=Path(self.tmp),
            output_stream=out,
            error_stream=err,
            state_of=lambda handle: gate.detached_state(handle, alive=alive),
        )
        return code, out.getvalue(), err.getvalue()

    def test_finished_pass_returns_zero(self):
        self._summary()
        code, out, _err = self._status(self.handle_path)
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)

    def test_finished_stage_failure_returns_one_and_names_the_transcript(self):
        self._summary(
            termination="stage-failure",
            exit_code=1,
            first_failing_stage={
                "index": 6,
                "command": "cargo clippy --workspace",
                "transcript": "target/gate-failures/x.log",
            },
        )
        code, out, _err = self._status(self.handle_path)
        self.assertEqual(code, 1)
        self.assertIn("stage 6", out)
        self.assertIn("target/gate-failures/x.log", out)

    def test_lease_timeout_surfaces_as_exit_four_with_its_explanation(self):
        """With `--detach` the launcher's exit code is a launch verdict, so
        exit 4 moves into `--status`. Say so where a caller will read it."""
        self._summary(termination="lease-timeout", exit_code=4)
        code, out, _err = self._status(self.handle_path)
        self.assertEqual(code, gate.EXIT_LEASE_TIMEOUT)
        self.assertIn("not a gate failure", out)

    def test_still_running_returns_seventy_five(self):
        code, out, _err = self._status(self.handle_path, alive=lambda _pid: True)
        self.assertEqual(code, gate.EXIT_STILL_RUNNING)
        self.assertIn("still running", out)

    def test_a_running_child_holding_the_lease_is_proven_by_the_lease(self):
        """The kernel-backed evidence, which is why a `--status` line can be
        trusted: a sidecar naming this pid over a lock that is really held."""
        lease_path = Path(self.payload["lease_path"])
        lease_path.write_text("", encoding="utf-8")
        lease_path.with_name(lease_path.name + ".json").write_text(
            json.dumps({"pid": self.payload["pid"], "worktree": self.tmp}),
            encoding="utf-8",
        )
        state = gate.detached_state(
            self.payload,
            alive=lambda _pid: False,
            peek=lambda _path: {"pid": self.payload["pid"]},
        )
        self.assertEqual(state["state"], "running")
        self.assertIn("lease", state["evidence"])

    def test_a_lease_held_by_a_different_pid_is_not_evidence(self):
        state = gate.detached_state(
            self.payload,
            alive=lambda _pid: False,
            peek=lambda _path: {"pid": self.payload["pid"] + 1},
        )
        self.assertEqual(state["state"], "died")

    def test_a_finished_summary_beats_a_live_pid(self):
        """Checked first so a recycled pid cannot make a finished run look
        like it is still going."""
        self._summary()
        state = gate.detached_state(self.payload, alive=lambda _pid: True)
        self.assertEqual(state["state"], "finished")

    def test_died_without_a_summary_returns_one(self):
        code, _out, err = self._status(self.handle_path, alive=lambda _pid: False)
        self.assertEqual(code, 1)
        self.assertIn("wrote no run summary", err)
        self.assertIn(self.payload["log"], err)

    def test_bare_status_resolves_the_newest_handle(self):
        self._summary()
        code, _out, _err = self._status(None)
        self.assertEqual(code, 0)

    def test_missing_truncated_and_non_object_handles_all_exit_two(self):
        bad = Path(self.tmp) / "bad.json"
        for content in (None, "{ truncated", '"a string"', "[]"):
            with self.subTest(content=content):
                if content is None:
                    target = Path(self.tmp) / "does-not-exist.json"
                else:
                    bad.write_text(content, encoding="utf-8")
                    target = bad
                code, _out, err = self._status(target)
                self.assertEqual(code, gate.EXIT_ENVIRONMENT)
                self.assertIn("cannot read the detach handle", err)

    def test_bare_status_with_no_handle_at_all_exits_two(self):
        for handle in Path(self.handle_path).parent.glob("*.json"):
            handle.unlink()
        code, _out, err = self._status(None)
        self.assertEqual(code, gate.EXIT_ENVIRONMENT)
        self.assertIn("no detached run handle", err)

    def test_a_stale_summary_with_the_same_pid_is_not_this_runs_verdict(self):
        """Red-team round 1 P2. A pid is not unique over time. An older run in
        the same report directory that happened to get this pid would be the
        only match for the whole window before this run finishes, so polling
        would return ITS verdict: a false PASS on the once-per-pull-request
        gate while the real run is still going."""
        stale = Path(self.tmp) / f"20260101T120000.0Z-{self.payload['pid']}-validation.json"
        stale.write_text(
            json.dumps(
                {"mode": "validation", "termination": "pass", "exit_code": 0,
                 "seconds": 471.2}
            ),
            encoding="utf-8",
        )
        code, out, _err = self._status(self.handle_path, alive=lambda _pid: True)
        self.assertEqual(code, gate.EXIT_STILL_RUNNING)
        self.assertNotIn("PASS", out)

    def test_a_summary_written_after_the_handle_is_this_runs_verdict(self):
        """The positive twin: the filter must not discard the real summary."""
        self._summary()
        code, out, _err = self._status(self.handle_path, alive=lambda _pid: True)
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)

    def test_the_earliest_summary_at_or_after_the_start_is_this_runs(self):
        """Red-team verification P3. Taking the newest match let a LATER run
        that reused the pid outrank this run's own summary: measured as PASS
        exit 0 when the real run exited 1. The earliest match at or after this
        run started is provably this run's, because no other process can hold
        this pid between this run's start and its exit, so an impostor's
        summary must end later. That closes both directions at once."""
        report = Path(self.tmp)
        pid = self.payload["pid"]
        stale = report / f"20260101T120000.0Z-{pid}-validation.json"
        ours = report / f"20260905T032000.0Z-{pid}-validation.json"
        later = report / f"20990101T120000.0Z-{pid}-validation.json"
        stale.write_text(json.dumps({"exit_code": 0, "termination": "pass"}), encoding="utf-8")
        ours.write_text(json.dumps({"exit_code": 1, "termination": "stage-failure"}), encoding="utf-8")
        later.write_text(json.dumps({"exit_code": 0, "termination": "pass"}), encoding="utf-8")
        self.assertEqual(
            gate.find_summary(
                report, pid, "validation", not_before=self.payload["started_at"]
            ),
            ours,
        )
        code, out, _err = self._status(self.handle_path, alive=lambda _pid: False)
        self.assertEqual(code, 1)
        self.assertIn("STAGE-FAILURE", out)

    def test_without_a_usable_floor_no_summary_can_be_proven_to_be_ours(self):
        """The filter must fail CLOSED. Degrading open would restore the exact
        false PASS it exists to prevent, while still looking like a guard."""
        report = Path(self.tmp)
        pid = self.payload["pid"]
        (report / f"20260101T120000.0Z-{pid}-validation.json").write_text(
            "{}", encoding="utf-8"
        )
        for floor in (None, "", "not a timestamp"):
            with self.subTest(floor=floor):
                self.assertIsNone(
                    gate.find_summary(report, pid, "validation", not_before=floor)
                )

    def test_a_handle_without_a_usable_started_at_is_rejected(self):
        """`read_handle` is where the floor is guaranteed, so a handle that
        cannot supply one never reaches the filter."""
        for started in (None, "", "yesterday", 12345):
            with self.subTest(started=started):
                payload = dict(self.payload)
                if started is None:
                    payload.pop("started_at")
                else:
                    payload["started_at"] = started
                path = Path(self.tmp) / "probe-handle.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    gate.read_handle(path)
                code, _out, err = self._status(path)
                self.assertEqual(code, gate.EXIT_ENVIRONMENT)
                self.assertIn("cannot read the detach handle", err)

    def test_an_unparseable_summary_name_cannot_be_proven_to_be_ours(self):
        """`write_summary` always produces a readable stamp, so a name this
        reader cannot parse is not something this gate wrote and cannot be
        proven to belong to this run."""
        odd = Path(self.tmp) / f"nostamp-{self.payload['pid']}-validation.json"
        odd.write_text("{}", encoding="utf-8")
        self.assertIsNone(
            gate.find_summary(
                Path(self.tmp),
                self.payload["pid"],
                "validation",
                not_before=self.payload["started_at"],
            )
        )

    def test_find_summary_ignores_another_pids_summary(self):
        self._summary()
        (Path(self.tmp) / "20260905T032000.0Z-99999-validation.json").write_text(
            json.dumps({"exit_code": 1}), encoding="utf-8"
        )
        found = gate.find_summary(
            Path(self.tmp),
            self.payload["pid"],
            "validation",
            not_before=self.payload["started_at"],
        )
        self.assertIsNotNone(found)
        self.assertIn(str(self.payload["pid"]), found.name)


class LivenessProbeTests(unittest.TestCase):
    """`_pid_alive` is the third and weakest liveness path, used only when a
    summary is absent and the lease does not name this pid. Every other test
    injects `alive`, so the real probe is exercised here."""

    def test_the_current_process_is_alive(self):
        self.assertTrue(gate._pid_alive(os.getpid()))

    def test_a_reaped_child_is_not_alive(self):
        child = subprocess.Popen(["/bin/sh", "-c", "exit 0"])
        child.wait()
        self.assertFalse(gate._pid_alive(child.pid))

    def test_a_process_we_may_not_signal_counts_as_alive(self):
        """`kill(pid, 0)` on another user's process raises PermissionError,
        which proves the process EXISTS. Reading that as dead would let
        `--status` report a live run as having died."""
        self.assertTrue(gate._pid_alive(1))

    def test_a_run_with_no_summary_and_no_lease_falls_through_to_the_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, _ = _spawn(_isolated_environ(tmp), tmp, pid=os.getpid())
            state = gate.detached_state(payload)
        self.assertEqual(state["state"], "running")
        self.assertEqual(state["evidence"], "the process is alive")


class UnreadableSummaryTests(unittest.TestCase):
    def test_a_summary_that_cannot_be_parsed_is_reported_not_guessed(self):
        """The run finished, but its verdict is unreadable. Inventing one
        would be worse than saying so."""
        with tempfile.TemporaryDirectory() as tmp:
            environ = _isolated_environ(tmp)
            payload, _ = _spawn(environ, tmp)
            handle = gate.write_handle(payload, Path(payload["handle"]))
            stamp = (
                gate._parse_iso(payload["started_at"])
            ).strftime("%Y%m%dT%H%M%S.%f") + "Z"
            (Path(tmp) / f"{stamp}-{payload['pid']}-validation.json").write_text(
                "{ truncated", encoding="utf-8"
            )
            state = gate.detached_state(payload)
            self.assertEqual(state["state"], "finished")
            self.assertIsNone(state["summary"])
            self.assertIn("could not read the run summary", state["error"])
            out, err = io.StringIO(), io.StringIO()
            code = gate.run_status(
                _parse_quietly(["--status", str(handle)]),
                environ=environ,
                repo_root=Path(tmp),
                output_stream=out,
                error_stream=err,
            )
        self.assertEqual(code, gate.EXIT_ENVIRONMENT)
        self.assertIn("could not read the run summary", err.getvalue())

    def test_a_log_outside_the_repo_root_prints_its_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            environ = _isolated_environ(tmp)
            stub = _PopenStub()
            out, err = io.StringIO(), io.StringIO()

            def spawn(child_argv, **kwargs):
                kwargs["repo_root"] = Path(tmp) / "elsewhere"
                kwargs["repo_root"].mkdir(exist_ok=True)
                return gate.spawn_detached(
                    child_argv,
                    popen=stub,
                    now=lambda: FIXED_NOW,
                    git_facts=lambda: {"head": "a" * 40},
                    **kwargs,
                )

            code = gate.run_detach(
                _parse_quietly(["--detach", "--validation"]),
                argv=["--detach", "--validation"],
                environ=environ,
                executable=Path("/managed/python"),
                mode="validation",
                repo_root=Path(tmp) / "unrelated",
                output_stream=out,
                error_stream=err,
                spawn=spawn,
            )
        self.assertEqual(code, 0)
        self.assertIn(tmp, out.getvalue())


class SummaryNamingTests(unittest.TestCase):
    """`find_summary` now discards a candidate whose stamp it cannot read, so
    if `write_summary`'s naming and `_summary_stamp`'s parsing ever drift
    apart, `--status` would silently report every run as still going. Lock the
    writer and the reader to each other rather than to a literal format.

    Readability only. This writes one summary, so it cannot exercise ordering;
    the earliest-match rule is proved by
    `test_the_earliest_summary_at_or_after_the_start_is_this_runs`."""

    def test_a_real_write_summary_name_is_readable_by_the_real_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = gate.GateReport(mode="validation", started_at="2026-09-05T03:00:00.000Z")
            report.exit_code = 0
            path = gate.write_summary(
                report, environ={gate.REPORT_DIR_ENV: tmp}, repo_root=Path(tmp)
            )
            stamp = gate._summary_stamp(path)
            self.assertIsNotNone(
                stamp, f"write_summary produced {path.name!r}, which _summary_stamp "
                "cannot read; --status would report every run as still running"
            )
            found = gate.find_summary(
                Path(tmp), os.getpid(), "validation", not_before="2026-09-05T03:00:00.000Z"
            )
            self.assertEqual(found, path)

    def test_iso_output_round_trips_through_the_floor_parser(self):
        """`_iso` emits milliseconds. A seconds-resolution parse of it fails
        silently and turns the whole filter into a no-op."""
        moment = datetime(2026, 9, 5, 3, 15, 0, 123456, tzinfo=timezone.utc)
        self.assertIsNotNone(gate._parse_iso(gate._iso(moment)))
        self.assertIsNotNone(gate._parse_iso("2026-09-05T03:15:00Z"))
        self.assertIsNone(gate._parse_iso("2026-09-05 03:15:00"))


class NoReExecTests(unittest.TestCase):
    def test_the_child_is_not_re_executed_through_uv(self):
        """`spawn_detached` passes the already-managed interpreter, so
        `ensure_managed_runtime` must be a no-op in the child. A surprise
        re-exec would fork a grandchild whose pid the handle does not record,
        and `find_summary` would never match its summary."""

        def find_uv():
            raise AssertionError("the child must not look for uv")

        def execvpe(*_):
            raise AssertionError("the child must not re-exec")

        with tempfile.TemporaryDirectory() as tmp:
            # The launcher already runs the checkout's own interpreter, so the
            # child it spawns with that interpreter owns it too.
            venv = Path(tmp) / ".venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").write_text("", encoding="utf-8")
            (venv / "pyvenv.cfg").write_text("uv = 0.4.0\n", encoding="utf-8")
            result = gate.ensure_managed_runtime(
                ["--validation"],
                environ={},
                executable=venv / "bin" / "python",
                prefix=venv,
                base_prefix=Path("/usr"),
                repo_root=Path(tmp),
                find_uv=find_uv,
                execvpe=execvpe,
            )
        self.assertIsNone(result)


class ReaperSafetyTests(unittest.TestCase):
    """A detached run has `ppid == 1` by design, so the obvious worry is that
    `reap_orphans.py --kill`, the hygiene step every agent runs before
    building, would reap it. Measured on a real detached `--validation` run: it
    cannot, for two independent reasons, and both are locked here because
    either one changing would create the hazard.

    First, the gate's interpreter is not a build tool and does not live under
    `target/`, so `match_repo_processes` never selects it. Second, its
    cargo/rustc children keep the live gate as their parent, so
    `classify_orphans` never calls them orphaned.

    The negative twin is locked too: once the gate itself dies, its children
    ARE orphans and the reaper still reaps them, which is the behaviour it
    exists for.
    """

    def _proc(self, pid, ppid, command):
        return reap.ProcInfo(pid=pid, ppid=ppid, etime="10:00", command=command)

    def _detached_snapshot(self, gate_alive: bool):
        gate_proc = self._proc(
            500, 1, f"/venv/bin/python3 {REPO_ROOT}/scripts/gate.py --validation"
        )
        children = [
            self._proc(501, 500, f"cargo check --manifest-path {REPO_ROOT}/Cargo.toml"),
            self._proc(502, 501, "rustc --edition 2021 src/lib.rs"),
        ]
        return ([gate_proc] if gate_alive else []) + children

    def test_python_is_not_a_build_tool_name(self):
        """Adding it would put every gate run, detached or not, into this
        script's SIGKILL set. This test is the tripwire on that."""
        for name in ("python", "python3", "python3.11"):
            self.assertNotIn(name, reap.BUILD_TOOL_NAMES)

    def test_a_live_detached_gate_process_is_not_matched(self):
        snapshot = self._detached_snapshot(gate_alive=True)
        matched = reap.match_repo_processes(
            snapshot, REPO_ROOT, cwd_lookup=lambda _pid: str(REPO_ROOT)
        )
        self.assertNotIn(500, {proc.pid for proc in matched})

    def test_a_live_detached_run_yields_no_orphans(self):
        snapshot = self._detached_snapshot(gate_alive=True)
        matched = reap.match_repo_processes(
            snapshot, REPO_ROOT, cwd_lookup=lambda _pid: str(REPO_ROOT)
        )
        self.assertEqual(reap.classify_orphans(matched, snapshot), set())

    def test_once_the_gate_dies_its_children_are_orphans_again(self):
        """The negative twin. Protection here would be a bug: a cargo whose
        owner is gone is exactly what the reaper is for."""
        snapshot = self._detached_snapshot(gate_alive=False)
        matched = reap.match_repo_processes(
            snapshot, REPO_ROOT, cwd_lookup=lambda _pid: str(REPO_ROOT)
        )
        self.assertEqual(reap.classify_orphans(matched, snapshot), {501, 502})


class RegressionTests(unittest.TestCase):
    def test_list_output_is_unaffected_by_the_new_flags(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = gate.main(["--list"], environ={})
        self.assertEqual(code, 0)
        self.assertNotIn("--detach", out.getvalue())
        self.assertNotIn("--status", out.getvalue())


if __name__ == "__main__":
    unittest.main()
