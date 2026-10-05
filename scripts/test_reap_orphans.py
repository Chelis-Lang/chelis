"""Unit tests for `reap_orphans.py`.

Run via: `python3 -m unittest scripts.test_reap_orphans` from repo root,
or `python3 scripts/test_reap_orphans.py`.

Everything is exercised on canned `ps` output and injected fakes: no
real process is signalled, no real `ps`/`lsof` is spawned. Locked here:

  (a) `ps` snapshot parsing (whitespace, malformed lines, commands with
      spaces);
  (b) repo scoping: build tools are matched by command line or working
      directory, repo `target/` test binaries are matched by command
      line, and unrelated processes (including an installed `chelis`
      running elsewhere — the issue #348 case) are not;
  (c) orphan classification: ppid 1, dead parent, transitive orphan
      children, and live-shell ownership;
  (d) the TERM-then-KILL escalation and grace period;
  (e) default dry-run: without `--kill`, `terminate` is never invoked.
"""

import importlib.util
import io
import os
import signal
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "reap_orphans", here / "reap_orphans.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


reap = _load_module()
REPO = str(reap.REPO_ROOT)


def _proc(pid, ppid, command, etime="01:23"):
    return reap.ProcInfo(pid=pid, ppid=ppid, etime=etime, command=command)


def _no_cwd(_pid):
    raise AssertionError("cwd_lookup must not be called for this case")


class ParsePsOutputTests(unittest.TestCase):
    def test_parses_pid_ppid_etime_and_spaced_command(self):
        text = (
            f"  101     1  1-02:03:04 {REPO}/target/debug/chelis check examples\n"
            f"  202   101       05:00 rustc --crate-name chelis_core --edition 2021\n"
        )
        procs = reap.parse_ps_output(text)
        self.assertEqual(len(procs), 2)
        self.assertEqual(procs[0].pid, 101)
        self.assertEqual(procs[0].ppid, 1)
        self.assertEqual(procs[0].etime, "1-02:03:04")
        self.assertEqual(
            procs[0].command, f"{REPO}/target/debug/chelis check examples"
        )
        self.assertEqual(
            procs[1].command, "rustc --crate-name chelis_core --edition 2021"
        )

    def test_skips_malformed_lines(self):
        text = (
            "\n"
            "not numbers here at all\n"
            "  303\n"
            "  303   1\n"
            f"  404     1       00:01 cargo nextest run -p chelis-cli\n"
        )
        procs = reap.parse_ps_output(text)
        self.assertEqual([p.pid for p in procs], [404])

    def test_basename_strips_path(self):
        proc = _proc(1, 1, "/Users/x/.cargo/bin/cargo-nextest nextest run")
        self.assertEqual(proc.basename, "cargo-nextest")
        self.assertEqual(_proc(2, 1, "").basename, "")


class MatchRepoProcessesTests(unittest.TestCase):
    def test_build_tool_with_repo_in_command_line_matches(self):
        procs = [_proc(10, 5, f"cargo build --manifest-path {REPO}/Cargo.toml")]
        matched = reap.match_repo_processes(procs, reap.REPO_ROOT, _no_cwd)
        self.assertEqual([p.pid for p in matched], [10])

    def test_target_dir_test_binary_matches_without_tool_basename(self):
        procs = [
            _proc(11, 5, f"{REPO}/target/debug/deps/rank_poly_tier3-abc123 --exact")
        ]
        matched = reap.match_repo_processes(procs, reap.REPO_ROOT, _no_cwd)
        self.assertEqual([p.pid for p in matched], [11])

    def test_sibling_checkout_path_does_not_match(self):
        # Review H1 regression: `<repo>-165` CONTAINS `<repo>` as a raw
        # substring; the boundary-aware matcher must not claim it. A
        # detached overnight build in a sibling checkout must never be
        # reaped by this checkout's hygiene step.
        # The cmdline no longer decides, so the cwd lookup legitimately
        # runs next; the sibling checkout's cwd settles it as foreign.
        procs = [
            _proc(70, 1, f"cargo build --manifest-path {REPO}-165/Cargo.toml"),
            _proc(71, 1, f"rustc --out-dir {REPO}-202/target/debug/deps lib.rs"),
        ]
        matched = reap.match_repo_processes(
            procs, reap.REPO_ROOT, lambda pid: f"{REPO}-165"
        )
        self.assertEqual(matched, [])

    def test_repo_path_at_end_of_command_matches(self):
        procs = [_proc(72, 5, f"cargo-nextest nextest list {REPO}")]
        matched = reap.match_repo_processes(procs, reap.REPO_ROOT, _no_cwd)
        self.assertEqual([p.pid for p in matched], [72])

    def test_target_path_in_arguments_only_does_not_match(self):
        # Review L1 regression: only a process whose EXECUTABLE lives in
        # target/ is a repo test binary; a tail or gcc that merely names
        # a target/ path in its arguments is not build activity.
        procs = [
            _proc(73, 1, f"tail -f {REPO}/target/nextest/ci/junit.xml"),
            _proc(74, 1, f"gcc -O2 {REPO}/target/out/prog.c -o /tmp/prog"),
        ]
        matched = reap.match_repo_processes(procs, reap.REPO_ROOT, _no_cwd)
        self.assertEqual(matched, [])

    def test_build_tool_with_repo_cwd_matches(self):
        # `cargo nextest run -p chelis-cli` from inside the repo never
        # mentions the repo path on its command line; the cwd lookup is
        # what scopes it.
        procs = [_proc(12, 5, "cargo nextest run -p chelis-cli")]
        matched = reap.match_repo_processes(
            procs, reap.REPO_ROOT, lambda pid: f"{REPO}/crates"
        )
        self.assertEqual([p.pid for p in matched], [12])

    def test_installed_chelis_running_elsewhere_does_not_match(self):
        # The issue #348 case: an installed chelis 0.7.20
        # spawned by an unrelated workload must not be listed or reaped.
        procs = [_proc(13, 5, "chelis eval --file season.ch")]
        matched = reap.match_repo_processes(
            procs, reap.REPO_ROOT, lambda pid: "/Users/x/other-project"
        )
        self.assertEqual(matched, [])

    def test_build_tool_with_unknown_cwd_does_not_match(self):
        procs = [_proc(14, 5, "rustc --crate-name serde")]
        matched = reap.match_repo_processes(procs, reap.REPO_ROOT, lambda pid: None)
        self.assertEqual(matched, [])

    def test_non_tool_process_mentioning_repo_does_not_match(self):
        # An editor or tail holding a repo path is not build activity.
        procs = [_proc(15, 5, f"vim {REPO}/README.md")]
        matched = reap.match_repo_processes(procs, reap.REPO_ROOT, _no_cwd)
        self.assertEqual(matched, [])

    def test_own_pid_is_excluded(self):
        procs = [
            _proc(os.getpid(), 5, f"cargo build --manifest-path {REPO}/Cargo.toml")
        ]
        matched = reap.match_repo_processes(procs, reap.REPO_ROOT, _no_cwd)
        self.assertEqual(matched, [])


class ClassifyOrphansTests(unittest.TestCase):
    def test_ppid_one_is_orphaned(self):
        cargo = _proc(20, 1, f"cargo nextest run --manifest-path {REPO}/Cargo.toml")
        snapshot = [_proc(1, 0, "/sbin/launchd"), cargo]
        self.assertEqual(reap.classify_orphans([cargo], snapshot), {20})

    def test_dead_parent_is_orphaned(self):
        cargo = _proc(21, 999, f"cargo build --manifest-path {REPO}/Cargo.toml")
        snapshot = [_proc(1, 0, "/sbin/launchd"), cargo]  # 999 is gone
        self.assertEqual(reap.classify_orphans([cargo], snapshot), {21})

    def test_child_of_orphaned_build_process_is_orphaned(self):
        cargo = _proc(22, 1, f"cargo build --manifest-path {REPO}/Cargo.toml")
        rustc = _proc(23, 22, f"rustc --crate-name chelis_core {REPO}/crates/x.rs")
        snapshot = [_proc(1, 0, "/sbin/launchd"), cargo, rustc]
        self.assertEqual(reap.classify_orphans([cargo, rustc], snapshot), {22, 23})

    def test_live_shell_ownership_is_not_orphaned(self):
        shell = _proc(30, 29, "-zsh")
        cargo = _proc(31, 30, f"cargo build --manifest-path {REPO}/Cargo.toml")
        rustc = _proc(32, 31, f"rustc --crate-name chelis_core {REPO}/crates/x.rs")
        snapshot = [_proc(1, 0, "/sbin/launchd"), shell, cargo, rustc]
        self.assertEqual(reap.classify_orphans([cargo, rustc], snapshot), set())

    def test_mixed_snapshot_only_flags_the_orphans(self):
        shell = _proc(40, 1, "-zsh")
        live = _proc(41, 40, f"cargo nextest run --manifest-path {REPO}/Cargo.toml")
        dead = _proc(42, 1, f"cargo nextest run -p chelis-cli {REPO}")
        snapshot = [_proc(1, 0, "/sbin/launchd"), shell, live, dead]
        self.assertEqual(reap.classify_orphans([live, dead], snapshot), {42})


class TerminateTests(unittest.TestCase):
    def test_term_then_kill_escalation(self):
        # pid 50 exits during the grace period; pid 51 survives and gets
        # SIGKILL. The probe is signal 0.
        calls = []
        sleeps = []
        gone_after_term = {50}

        def kill_fn(pid, sig):
            calls.append((pid, sig))
            if pid in gone_after_term and sig != signal.SIGTERM:
                raise ProcessLookupError

        killed = reap.terminate(
            [50, 51], grace_seconds=2.5, kill_fn=kill_fn, sleep_fn=sleeps.append
        )
        self.assertEqual(killed, [51])
        self.assertEqual(sleeps, [2.5])
        self.assertEqual(
            calls,
            [
                (50, signal.SIGTERM),
                (51, signal.SIGTERM),
                (50, 0),
                (51, 0),
                (51, signal.SIGKILL),
            ],
        )

    def test_identity_mismatch_skips_the_signal_entirely(self):
        # Review M1 regression (PID reuse): a pid whose live command no
        # longer matches the classification snapshot must receive NO
        # signal at all.
        calls = []
        killed = reap.terminate(
            [80],
            kill_fn=lambda pid, sig: calls.append((pid, sig)),
            sleep_fn=lambda s: None,
            expected={80: "cargo build --manifest-path /repo/Cargo.toml"},
            command_lookup=lambda pid: "/usr/bin/ssh important-host",
        )
        self.assertEqual(killed, [])
        self.assertEqual(calls, [])

    def test_identity_recheck_before_sigkill(self):
        # Identity holds at TERM time but the pid is reused during the
        # grace period: SIGKILL must be withheld.
        command = "cargo build --manifest-path /repo/Cargo.toml"
        lookups = iter([command, "/usr/bin/ssh important-host"])
        calls = []
        killed = reap.terminate(
            [81],
            grace_seconds=1.0,
            kill_fn=lambda pid, sig: calls.append((pid, sig)),
            sleep_fn=lambda s: None,
            expected={81: command},
            command_lookup=lambda pid: next(lookups),
        )
        self.assertEqual(killed, [])
        self.assertEqual(calls, [(81, signal.SIGTERM)])

    def test_matching_identity_proceeds_to_escalation(self):
        command = "cargo build --manifest-path /repo/Cargo.toml"
        calls = []
        killed = reap.terminate(
            [82],
            grace_seconds=1.0,
            kill_fn=lambda pid, sig: calls.append((pid, sig)),
            sleep_fn=lambda s: None,
            expected={82: command},
            command_lookup=lambda pid: command,
        )
        self.assertEqual(killed, [82])
        self.assertEqual(
            calls, [(82, signal.SIGTERM), (82, 0), (82, signal.SIGKILL)]
        )

    def test_already_dead_pids_skip_the_grace_sleep(self):
        def kill_fn(pid, sig):
            raise ProcessLookupError

        def sleep_fn(seconds):
            raise AssertionError("must not sleep when nothing was signalled")

        killed = reap.terminate([60], kill_fn=kill_fn, sleep_fn=sleep_fn)
        self.assertEqual(killed, [])

    def test_permission_error_is_reported_not_raised(self):
        def kill_fn(pid, sig):
            raise PermissionError

        stderr = io.StringIO()
        real_stderr = sys.stderr
        sys.stderr = stderr
        try:
            killed = reap.terminate(
                [70], kill_fn=kill_fn, sleep_fn=lambda s: None
            )
        finally:
            sys.stderr = real_stderr
        self.assertEqual(killed, [])
        self.assertIn("no permission", stderr.getvalue())


class ListingAndCliTests(unittest.TestCase):
    def test_format_listing_marks_orphans_first(self):
        owned = _proc(80, 79, f"cargo build --manifest-path {REPO}/Cargo.toml")
        orphan = _proc(81, 1, f"cargo nextest run -p chelis-cli {REPO}")
        text = reap.format_listing([owned, orphan], {81})
        lines = text.splitlines()
        self.assertIn("COMMAND", lines[0])
        self.assertTrue(lines[1].startswith("ORPHAN"))
        self.assertIn("81", lines[1])
        self.assertTrue(lines[2].startswith("owned"))

    def test_format_listing_empty(self):
        self.assertIn("no repo build/test processes", reap.format_listing([], set()))

    def test_default_is_dry_run(self):
        args = reap.parse_args([])
        self.assertFalse(args.kill)
        self.assertEqual(args.grace, reap.DEFAULT_GRACE_SECONDS)

    def test_kill_and_grace_flags_parse(self):
        args = reap.parse_args(["--kill", "--grace", "10"])
        self.assertTrue(args.kill)
        self.assertEqual(args.grace, 10.0)

    def test_main_dry_run_lists_but_never_terminates(self):
        shell = _proc(90, 1, "-zsh")
        live = _proc(91, 90, f"cargo build --manifest-path {REPO}/Cargo.toml")
        orphan = _proc(92, 1, f"cargo nextest run -p chelis-cli {REPO}")
        snapshot = [_proc(1, 0, "/sbin/launchd"), shell, live, orphan]
        original_snapshot = reap.ps_snapshot
        original_terminate = reap.terminate
        terminations = []
        reap.ps_snapshot = lambda: snapshot
        reap.terminate = lambda *a, **kw: terminations.append((a, kw)) or []
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = reap.main([])
        finally:
            reap.ps_snapshot = original_snapshot
            reap.terminate = original_terminate
        self.assertEqual(rc, 0)
        self.assertEqual(terminations, [])
        out = buf.getvalue()
        self.assertIn("ORPHAN", out)
        self.assertIn("92", out)
        self.assertIn("--kill", out)

    def test_main_kill_terminates_exactly_the_orphans(self):
        shell = _proc(95, 1, "-zsh")
        live = _proc(96, 95, f"cargo build --manifest-path {REPO}/Cargo.toml")
        orphan_a = _proc(97, 1, f"cargo nextest run -p chelis-cli {REPO}")
        orphan_b = _proc(98, 97, f"rustc --crate-name x {REPO}/crates/x.rs")
        snapshot = [_proc(1, 0, "/sbin/launchd"), shell, live, orphan_a, orphan_b]
        original_snapshot = reap.ps_snapshot
        original_terminate = reap.terminate
        terminations = []

        def fake_terminate(pids, grace_seconds, expected=None):
            terminations.append((pids, grace_seconds, expected))
            return []

        reap.ps_snapshot = lambda: snapshot
        reap.terminate = fake_terminate
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = reap.main(["--kill", "--grace", "3"])
        finally:
            reap.ps_snapshot = original_snapshot
            reap.terminate = original_terminate
        self.assertEqual(rc, 0)
        # main must thread the snapshot commands through so terminate can
        # re-verify pid identity before signaling (PID-reuse guard).
        self.assertEqual(
            terminations,
            [([97, 98], 3.0, {97: orphan_a.command, 98: orphan_b.command})],
        )

    def test_main_kill_with_no_orphans_is_a_no_op(self):
        shell = _proc(100, 1, "-zsh")
        live = _proc(101, 100, f"cargo build --manifest-path {REPO}/Cargo.toml")
        snapshot = [_proc(1, 0, "/sbin/launchd"), shell, live]
        original_snapshot = reap.ps_snapshot
        original_terminate = reap.terminate
        reap.ps_snapshot = lambda: snapshot
        reap.terminate = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("terminate must not run with no orphans")
        )
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = reap.main(["--kill"])
        finally:
            reap.ps_snapshot = original_snapshot
            reap.terminate = original_terminate
        self.assertEqual(rc, 0)
        self.assertIn("nothing to kill", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
