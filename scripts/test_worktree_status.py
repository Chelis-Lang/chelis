"""Unit tests for `worktree_status.py`.

Run via: `python3 -m unittest scripts.test_worktree_status` from the repo
root, or `.venv/bin/python -m unittest scripts.test_worktree_status -v`.

What is locked here, grouped by the property rather than by the function:

  (a) the verdict lattice and its precedence, BUSY > UNKNOWN > NOT CLEAN >
      FREE, including the two rules that give it meaning: positive evidence
      is never downgraded by a failure elsewhere, and absence of evidence
      never prints FREE;
  (b) every lease state the probe can meet: free, held by this worktree,
      held by another worktree, a holder that died leaving a sidecar behind,
      a sidecar that cannot be read, and a lease path that cannot be opened.
      These use a REAL `fcntl.flock` in a temporary directory, taken by a
      real `gate.GateLease`, because two open file descriptions in one
      process conflict and no second process is needed;
  (c) the dirty split: tracked modification, staged only, untracked only,
      unmerged, and a merely stale index, which is reported and deliberately
      does NOT make the tree dirty;
  (d) read-only discipline, twice over: every git argv is checked against the
      frozen allowlist, and one end-to-end test against a REAL git repository
      asserts that a whole directory tree, `.git` included, is byte-for-byte
      unchanged after a probe. That test is the reason this file exists: the
      collision being designed against was a reviewer who wrote into the
      worktree they were inspecting;
  (e) the parsers, against the awkward inputs git actually emits: NUL
      separators, renames that spend two fields, and paths with spaces and
      non-ASCII bytes.

Not executed here: nothing takes the real workstation lease, reads the real
`~/.cache/chelis`, or signals any process. `ps` output is canned text run
through `reap_orphans.parse_ps_output`.
"""

import fcntl
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


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "worktree_status", here / "worktree_status.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


status = _load_module()
gate = status.gate
reap = status.reap

FIXED_NOW = datetime(2026, 9, 5, 3, 4, 11, tzinfo=timezone.utc)
PROBED = "/wt/probed"
OTHER = "/wt/other"
HEAD_SHA = "58635052c132d8fa32d9d855539469af7c4b33ce"


def _now():
    return FIXED_NOW


def _rev_parse_output(
    worktree=PROBED, head=HEAD_SHA, branch="agent/x", linked=True, common=None
):
    """`--absolute-git-dir`, `--git-common-dir`, `--show-toplevel`, HEAD, branch.

    `common` defaults to the ABSOLUTE spelling, but git answers a bare
    relative `.git` for a primary checkout probed at its root, which is what
    `common=".git"` reproduces.
    """
    git_dir = "/repo/.git/worktrees/probed" if linked else f"{worktree}/.git"
    if common is None:
        common = "/repo/.git" if linked else f"{worktree}/.git"
    return "\n".join([git_dir, common, worktree, head, branch]) + "\n"


def _porcelain(*entries: str) -> str:
    """Build `--porcelain=v1 -z` output: every field is NUL-terminated."""
    return "".join(entry + "\0" for entry in entries)


class FakeGit:
    """A `query` stand-in. Records every argv so the read-only tests can
    inspect what would have been run."""

    def __init__(self, *, rev_parse=None, porcelain="", ls_files="", fail=None):
        self.rev_parse = _rev_parse_output() if rev_parse is None else rev_parse
        self.porcelain = porcelain
        self.ls_files = ls_files
        self.fail = fail or {}
        self.calls: list[list[str]] = []

    def __call__(self, args, *, worktree, **_kwargs):
        args = list(args)
        self.calls.append(args)
        subcommand = args[0]
        if subcommand in self.fail:
            raise status.GitQueryError(self.fail[subcommand])
        if subcommand == "rev-parse":
            return self.rev_parse
        if subcommand == "status":
            return self.porcelain
        if subcommand == "ls-files":
            return self.ls_files
        raise AssertionError(f"unexpected git subcommand {subcommand!r}")


def _proc(pid, ppid, command, etime="01:23"):
    return reap.ProcInfo(pid=pid, ppid=ppid, etime=etime, command=command)


def _no_processes():
    return []


def _no_cwd(_pid):
    return None


def _collect(**overrides):
    """Collect with every evidence source neutral unless overridden."""
    kwargs = {
        "environ": overrides.pop("environ", {status.gate.LEASE_DIR_ENV: "/nonexistent-lease-dir"}),
        "now": _now,
        "query": overrides.pop("query", FakeGit()),
        "snapshot": overrides.pop("snapshot", _no_processes),
        "cwd_lookup": overrides.pop("cwd_lookup", _no_cwd),
        "stat": overrides.pop("stat", lambda path: os.stat_result((0,) * 10)),
        "exists": overrides.pop("exists", lambda path: False),
    }
    worktree = overrides.pop("worktree", Path(PROBED))
    assert not overrides, f"unused overrides {sorted(overrides)}"
    return status.collect(worktree, **kwargs)


class _RealLease:
    """Hold a real flock on a real file, the way a running gate does."""

    def __init__(self, directory: Path, holder: dict | None, *, sidecar_text=None):
        self.path = directory / gate.LEASE_FILE_NAME
        self.sidecar = self.path.with_name(self.path.name + ".json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sidecar_text is not None:
            self.sidecar.write_text(sidecar_text, encoding="utf-8")
        elif holder is not None:
            self.sidecar.write_text(json.dumps(holder), encoding="utf-8")

    def close(self):
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)


def _holder(worktree=PROBED, pid=41277, mode="local"):
    return {
        "schema_version": 1,
        "pid": pid,
        "worktree": worktree,
        "head": HEAD_SHA,
        "mode": mode,
        "started_at": "2026-09-05T03:00:00Z",
    }


class VerdictLatticeTests(unittest.TestCase):
    def test_clean_and_idle_is_free(self):
        state = _collect()
        self.assertEqual(state["verdict"], status.VERDICT_FREE)
        self.assertEqual(state["exit_code"], status.EXIT_FREE)
        self.assertEqual(state["reasons"], [])

    def test_dirty_and_idle_is_not_clean(self):
        state = _collect(query=FakeGit(porcelain=_porcelain(" M src/a.rs")))
        self.assertEqual(state["verdict"], status.VERDICT_NOT_CLEAN)
        self.assertEqual(state["exit_code"], status.EXIT_NOT_CLEAN)

    def test_busy_beats_dirty(self):
        state = _collect(
            query=FakeGit(porcelain=_porcelain(" M src/a.rs")),
            snapshot=lambda: [_proc(10, 20, f"cargo build --manifest-path {PROBED}/Cargo.toml")],
        )
        self.assertEqual(state["verdict"], status.VERDICT_BUSY)
        self.assertEqual(state["exit_code"], status.EXIT_BUSY)

    def test_busy_beats_a_failed_evidence_source(self):
        """Positive evidence is never downgraded: a broken `ps` does not make
        a running gate process less real."""

        def broken_ps():
            raise RuntimeError("could not take a ps snapshot: boom")

        state = _collect(snapshot=broken_ps, exists=lambda path: path.name == "MERGE_HEAD")
        self.assertEqual(state["verdict"], status.VERDICT_BUSY)
        self.assertTrue(state["unknown"], "the ps failure is still recorded")

    def test_unknown_beats_dirty(self):
        def broken_ps():
            raise RuntimeError("no ps")

        state = _collect(
            query=FakeGit(porcelain=_porcelain(" M src/a.rs")), snapshot=broken_ps
        )
        self.assertEqual(state["verdict"], status.VERDICT_UNKNOWN)
        self.assertEqual(state["exit_code"], status.EXIT_UNKNOWN)

    def test_a_failed_source_never_prints_free(self):
        def broken_ps():
            raise RuntimeError("no ps")

        state = _collect(snapshot=broken_ps)
        self.assertNotEqual(state["verdict"], status.VERDICT_FREE)
        self.assertEqual(state["verdict"], status.VERDICT_UNKNOWN)

    def test_failed_git_is_unknown_not_free(self):
        state = _collect(query=FakeGit(fail={"rev-parse": "not a git repository"}))
        self.assertEqual(state["verdict"], status.VERDICT_UNKNOWN)
        self.assertIn("git", [item["source"] for item in state["unknown"]])

    def test_every_verdict_maps_to_its_documented_exit_code(self):
        self.assertEqual(
            status.EXIT_FOR_VERDICT,
            {
                "FREE": 0,
                "BUSY": 1,
                "UNKNOWN": 2,
                "NOT CLEAN": 3,
            },
        )


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.environ = {gate.LEASE_DIR_ENV: str(self.tmp)}
        self.addCleanup(self._tmp.cleanup)

    def test_free_when_no_lock_file_exists(self):
        state = _collect(environ=self.environ)
        self.assertFalse(state["lease"]["held"])
        self.assertEqual(state["verdict"], status.VERDICT_FREE)

    def test_held_by_this_worktree_is_busy(self):
        lease = _RealLease(self.tmp, _holder(worktree=PROBED))
        self.addCleanup(lease.close)
        state = _collect(environ=self.environ)
        self.assertTrue(state["lease"]["held"])
        self.assertTrue(state["lease"]["held_by_this_worktree"])
        self.assertEqual(state["verdict"], status.VERDICT_BUSY)
        self.assertIn("41277", status.render_human(state))

    def test_held_by_another_worktree_does_not_change_the_exit_code(self):
        """The exit code is scoped to the probed path. A foreign holder only
        predicts that a heavyweight run started here will queue."""
        lease = _RealLease(self.tmp, _holder(worktree=OTHER))
        self.addCleanup(lease.close)
        state = _collect(environ=self.environ)
        self.assertTrue(state["lease"]["held"])
        self.assertFalse(state["lease"]["held_by_this_worktree"])
        self.assertTrue(state["lease"]["held_elsewhere"])
        self.assertEqual(state["verdict"], status.VERDICT_FREE)
        self.assertIn("WARN", status.render_human(state))

    def test_holder_died_leaving_a_stale_sidecar(self):
        """No flock, but a sidecar names a holder. The kernel released the
        lock, so the file is descriptive and the worktree is free."""
        (self.tmp / gate.LEASE_FILE_NAME).write_text("", encoding="utf-8")
        (self.tmp / (gate.LEASE_FILE_NAME + ".json")).write_text(
            json.dumps(_holder(worktree=PROBED)), encoding="utf-8"
        )
        state = _collect(environ=self.environ)
        self.assertFalse(state["lease"]["held"])
        self.assertTrue(state["lease"]["stale_sidecar"])
        self.assertEqual(state["verdict"], status.VERDICT_FREE)
        self.assertIn("descriptive only", status.render_human(state))

    def test_unreadable_sidecar_is_unknown_not_busy_and_not_free(self):
        """Held, but by whom? It may be another worktree, so this cannot be
        BUSY; it may be this one, so it must not be FREE."""
        lease = _RealLease(self.tmp, None, sidecar_text="{ not json")
        self.addCleanup(lease.close)
        state = _collect(environ=self.environ)
        self.assertTrue(state["lease"]["held"])
        self.assertIsNone(state["lease"]["holder"])
        self.assertEqual(state["verdict"], status.VERDICT_UNKNOWN)
        self.assertIn(
            "gate-lease-identity", [item["source"] for item in state["unknown"]]
        )

    def test_sidecar_naming_no_worktree_is_unknown(self):
        lease = _RealLease(self.tmp, {"pid": 5, "mode": "local"})
        self.addCleanup(lease.close)
        state = _collect(environ=self.environ)
        self.assertEqual(state["verdict"], status.VERDICT_UNKNOWN)

    def test_unopenable_lease_path_is_unknown(self):
        path = self.tmp / gate.LEASE_FILE_NAME
        path.write_text("", encoding="utf-8")
        os.chmod(path, 0o000)
        self.addCleanup(os.chmod, path, 0o644)
        if os.access(path, os.R_OK):
            self.skipTest("running with rights that ignore the mode bits")
        state = _collect(environ=self.environ)
        self.assertEqual(state["verdict"], status.VERDICT_UNKNOWN)
        self.assertIn("gate-lease", [item["source"] for item in state["unknown"]])

    def test_a_busy_process_still_wins_over_an_unidentified_lease(self):
        lease = _RealLease(self.tmp, None, sidecar_text="{ not json")
        self.addCleanup(lease.close)
        state = _collect(
            environ=self.environ,
            snapshot=lambda: [_proc(11, 22, f"cargo nextest run {PROBED}/x")],
        )
        self.assertEqual(state["verdict"], status.VERDICT_BUSY)


class DirtyStateTests(unittest.TestCase):
    def test_tracked_modification(self):
        state = _collect(query=FakeGit(porcelain=_porcelain(" M src/a.rs")))
        self.assertEqual(state["dirty"]["modified"], ["src/a.rs"])
        self.assertFalse(state["dirty"]["clean"])

    def test_staged_only(self):
        state = _collect(query=FakeGit(porcelain=_porcelain("A  src/new.rs")))
        self.assertEqual(state["dirty"]["staged"], ["src/new.rs"])
        self.assertEqual(state["dirty"]["modified"], [])
        self.assertFalse(state["dirty"]["clean"])

    def test_untracked_only(self):
        state = _collect(query=FakeGit(porcelain=_porcelain("?? probe.txt")))
        self.assertEqual(state["dirty"]["untracked"], ["probe.txt"])
        self.assertEqual(state["dirty"]["modified"], [])
        self.assertEqual(state["verdict"], status.VERDICT_NOT_CLEAN)

    def test_unmerged(self):
        state = _collect(query=FakeGit(porcelain=_porcelain("UU src/conflict.rs")))
        self.assertEqual(state["dirty"]["unmerged"], ["src/conflict.rs"])
        self.assertFalse(state["dirty"]["clean"])

    def test_staged_and_modified_counts_in_both(self):
        state = _collect(query=FakeGit(porcelain=_porcelain("MM src/a.rs")))
        self.assertEqual(state["dirty"]["staged"], ["src/a.rs"])
        self.assertEqual(state["dirty"]["modified"], ["src/a.rs"])

    def test_index_staleness_failure_does_not_make_the_tree_unknown(self):
        """The dirty verdict does not depend on index freshness, so losing
        that source is recorded as degraded and must not withhold FREE."""
        state = _collect(query=FakeGit(fail={"ls-files": "boom"}))
        self.assertEqual(state["dirty"]["index_stale"], [])
        self.assertTrue(state["dirty"]["clean"])
        self.assertEqual(state["verdict"], status.VERDICT_FREE)
        self.assertEqual(state["unknown"], [])
        self.assertIn("index-freshness", [i["source"] for i in state["degraded"]])
        self.assertIn("verdict unaffected", status.render_human(state))


class PorcelainParserTests(unittest.TestCase):
    def test_plain_entries(self):
        parsed = status.parse_porcelain_z(_porcelain(" M a.rs", "?? b.rs"))
        self.assertEqual(parsed, [(" M", "a.rs"), ("??", "b.rs")])

    def test_rename_consumes_two_fields(self):
        """`R  new\\0old\\0` is one entry, and the origin path must not be
        mistaken for a second entry."""
        parsed = status.parse_porcelain_z(_porcelain("R  new.rs", "old.rs", " M c.rs"))
        self.assertEqual(parsed, [("R ", "new.rs"), (" M", "c.rs")])

    def test_paths_with_spaces_and_non_ascii_survive(self):
        parsed = status.parse_porcelain_z(_porcelain("?? d/uni-café and space.txt"))
        self.assertEqual(parsed, [("??", "d/uni-café and space.txt")])

    def test_empty_output_is_clean(self):
        self.assertEqual(status.parse_porcelain_z(""), [])


class LsFilesDebugParserTests(unittest.TestCase):
    SAMPLE = (
        "d/uni-café.txt\0"
        "  ctime: 100:5\n  mtime: 111:222\n  dev: 1\tino: 2\n"
        "  uid: 501\tgid: 0\n  size: 42\tflags: 0\n"
        "d/with space.txt\0"
        "  ctime: 100:5\n  mtime: 333:0\n  dev: 1\tino: 3\n"
        "  uid: 501\tgid: 0\n  size: 7\tflags: 0\n"
    )

    def test_parses_paths_and_stat_fields(self):
        parsed = status.parse_ls_files_debug(self.SAMPLE)
        self.assertEqual(
            parsed,
            {
                "d/uni-café.txt": {"mtime_ns": 111 * 10**9 + 222, "size": 42},
                "d/with space.txt": {"mtime_ns": 333 * 10**9, "size": 7},
            },
        )

    def test_empty_input(self):
        self.assertEqual(status.parse_ls_files_debug(""), {})


class ProcessMatchingTests(unittest.TestCase):
    def test_gate_process_matched_by_absolute_path(self):
        procs = [_proc(31, 1, f"/usr/bin/python3 {PROBED}/scripts/gate.py --local")]
        found = status.match_gate_processes(procs, Path(PROBED), _no_cwd)
        self.assertEqual([p.pid for p in found], [31])

    def test_gate_process_in_another_worktree_is_not_matched(self):
        procs = [_proc(31, 1, f"/usr/bin/python3 {OTHER}/scripts/gate.py --local")]
        found = status.match_gate_processes(procs, Path(PROBED), _no_cwd)
        self.assertEqual(found, [])

    def test_relative_gate_invocation_matched_by_cwd(self):
        procs = [_proc(32, 1, "python3 scripts/gate.py --local")]
        found = status.match_gate_processes(procs, Path(PROBED), lambda pid: PROBED)
        self.assertEqual([p.pid for p in found], [32])

    def test_relative_gate_invocation_elsewhere_is_not_matched(self):
        procs = [_proc(32, 1, "python3 scripts/gate.py --local")]
        found = status.match_gate_processes(procs, Path(PROBED), lambda pid: OTHER)
        self.assertEqual(found, [])

    def test_sibling_checkout_never_matches(self):
        """`<repo>-165` contains `<repo>`; the boundary matcher reused from
        reap_orphans is what stops it matching."""
        procs = [_proc(33, 1, f"python3 {PROBED}-165/scripts/gate.py --local")]
        found = status.match_gate_processes(procs, Path(PROBED), _no_cwd)
        self.assertEqual(found, [])

    def test_a_gate_process_makes_the_worktree_busy(self):
        state = _collect(
            snapshot=lambda: [_proc(31, 1, f"python3 {PROBED}/scripts/gate.py --local")]
        )
        self.assertEqual(state["verdict"], status.VERDICT_BUSY)
        self.assertEqual(
            [p["pid"] for p in state["processes"]["gate_processes"]], [31]
        )

    def test_build_processes_are_reported_and_deduplicated(self):
        """A process matched by the reaper is not listed twice as a gate
        process."""
        command = f"cargo nextest run --manifest-path {PROBED}/Cargo.toml"
        state = _collect(snapshot=lambda: [_proc(40, 1, command)])
        self.assertEqual([p["pid"] for p in state["processes"]["matched"]], [40])
        self.assertEqual(state["processes"]["gate_processes"], [])
        self.assertEqual(state["processes"]["orphaned"], 1)

    def test_ps_text_parses_through_the_reaper(self):
        procs = reap.parse_ps_output(
            f"  40     1       01:02 cargo nextest run {PROBED}/Cargo.toml\n"
        )
        self.assertEqual(procs[0].pid, 40)
        self.assertEqual(procs[0].ppid, 1)


class GitOperationTests(unittest.TestCase):
    def test_each_marker_makes_the_worktree_busy(self):
        for marker, _description in status.GIT_OPERATION_MARKERS:
            with self.subTest(marker=marker):
                state = _collect(exists=lambda path, m=marker: path.name == m)
                self.assertEqual(state["verdict"], status.VERDICT_BUSY)
                self.assertEqual(
                    [op["marker"] for op in state["git_operations"]], [marker]
                )

    def test_markers_are_read_from_the_resolved_git_dir(self):
        """A linked worktree keeps `.git` as a FILE, so the markers live under
        the path `rev-parse --absolute-git-dir` reports, not `<wt>/.git`."""
        seen: list[Path] = []

        def record(path):
            seen.append(path)
            return False

        _collect(exists=record)
        self.assertTrue(seen)
        for path in seen:
            self.assertEqual(str(path.parent), "/repo/.git/worktrees/probed")


class GitFactsAnchoringTests(unittest.TestCase):
    """`--git-common-dir` is reported relative to the directory git RAN in.

    Anchoring it to `--absolute-git-dir` instead builds `<root>/.git/.git`,
    which never equals the git dir, so every primary checkout would be
    labelled linked and the JSON would carry a path that does not exist.
    """

    def test_relative_common_dir_is_anchored_to_the_probed_worktree(self):
        facts = status.git_facts(
            Path(PROBED),
            query=FakeGit(
                rev_parse=_rev_parse_output(linked=False, common=".git")
            ),
        )
        self.assertEqual(facts["git_common_dir"], f"{PROBED}/.git")
        self.assertFalse(facts["linked"])

    def test_a_primary_checkout_is_not_labelled_linked(self):
        state = _collect(
            query=FakeGit(rev_parse=_rev_parse_output(linked=False, common=".git"))
        )
        self.assertIn("(main,", status.render_human(state))
        self.assertNotIn("(linked,", status.render_human(state))

    def test_a_linked_worktree_is_still_labelled_linked(self):
        state = _collect(query=FakeGit(rev_parse=_rev_parse_output(linked=True)))
        self.assertIn("(linked,", status.render_human(state))


class RootAnchoringTests(unittest.TestCase):
    """Everything below the git facts is scoped by path, so a caller who
    passes a subdirectory must be resolved to the repository root rather than
    trusted. Without this, probing `<root>/subdir` reports FREE while a gate
    runs at the root."""

    def test_a_subdirectory_is_resolved_to_the_repository_root(self):
        state = _collect(
            worktree=Path(f"{PROBED}/scripts"),
            query=FakeGit(rev_parse=_rev_parse_output(worktree=PROBED)),
        )
        self.assertEqual(state["worktree"], PROBED)
        self.assertEqual(state["probed_path"], f"{PROBED}/scripts")
        self.assertIn("resolved from the probed path", status.render_human(state))

    def test_a_gate_at_the_root_makes_a_probed_subdirectory_busy(self):
        state = _collect(
            worktree=Path(f"{PROBED}/scripts"),
            query=FakeGit(rev_parse=_rev_parse_output(worktree=PROBED)),
            snapshot=lambda: [
                _proc(31, 1, f"python3 {PROBED}/scripts/gate.py --local")
            ],
        )
        self.assertEqual(state["verdict"], status.VERDICT_BUSY)

    def test_the_lease_is_matched_against_the_root_not_the_probed_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            lease = _RealLease(Path(tmp), _holder(worktree=PROBED))
            try:
                state = _collect(
                    worktree=Path(f"{PROBED}/scripts"),
                    environ={gate.LEASE_DIR_ENV: tmp},
                    query=FakeGit(rev_parse=_rev_parse_output(worktree=PROBED)),
                )
            finally:
                lease.close()
        self.assertTrue(state["lease"]["held_by_this_worktree"])
        self.assertEqual(state["verdict"], status.VERDICT_BUSY)


class LastGateReportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.environ = {
            gate.REPORT_DIR_ENV: str(self.tmp),
            gate.LEASE_DIR_ENV: str(self.tmp / "lease"),
        }
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name, payload):
        (self.tmp / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_no_reports(self):
        state = _collect(environ=self.environ)
        self.assertIsNone(state["last_gate_report"])
        self.assertIn("no gate report", status.render_human(state))

    def test_newest_report_wins_and_is_labelled_history(self):
        self._write("20260905T010000.0Z-1-local.json", {"mode": "local", "exit_code": 1})
        self._write(
            "20260905T020000.0Z-2-fast.json",
            {
                "mode": "fast",
                "termination": "pass",
                "exit_code": 0,
                "seconds": 113.1,
                "ended_at": "2026-09-05T02:00:00Z",
            },
        )
        state = _collect(environ=self.environ)
        self.assertEqual(state["last_gate_report"]["mode"], "fast")
        block = status.render_human(state)
        self.assertIn("[history, not now]", block)

    def test_a_finished_report_never_makes_the_worktree_busy(self):
        """`gate.py` writes its summary from `main`'s `finally`, so a report
        is proof a run ENDED. It must never read as a run in flight."""
        self._write(
            "20260905T020000.0Z-2-local.json",
            {"mode": "local", "termination": "pass", "exit_code": 0},
        )
        state = _collect(environ=self.environ)
        self.assertEqual(state["verdict"], status.VERDICT_FREE)

    def test_corrupt_report_is_degraded_and_never_withholds_free(self):
        """A SIGKILLed gate can leave a truncated summary behind and nothing
        cleans it up, so history that cannot be read must not pin the verdict
        at UNKNOWN for as long as the file exists."""
        (self.tmp / "20260905T020000.0Z-2-local.json").write_text(
            "{ truncated", encoding="utf-8"
        )
        state = _collect(environ=self.environ)
        self.assertIsNone(state["last_gate_report"])
        self.assertEqual(state["unknown"], [])
        self.assertIn("gate-reports", [i["source"] for i in state["degraded"]])
        self.assertEqual(state["verdict"], status.VERDICT_FREE)


class ReadOnlyDisciplineTests(unittest.TestCase):
    def test_every_issued_git_argv_is_in_the_frozen_allowlist(self):
        fake = FakeGit(porcelain=_porcelain(" M a.rs"))
        _collect(query=fake)
        self.assertTrue(fake.calls)
        for argv in fake.calls:
            subcommand = next(arg for arg in argv if not arg.startswith("-"))
            self.assertIn(subcommand, status.READ_ONLY_GIT_SUBCOMMANDS)
            self.assertFalse(set(argv) & status.FORBIDDEN_GIT_TOKENS)

    def test_git_query_refuses_a_mutating_subcommand(self):
        def runner(*_args, **_kwargs):
            raise AssertionError("a mutating command must never be spawned")

        with self.assertRaises(status.GitQueryError):
            status.git_query(["checkout", "main"], worktree=Path("/x"), runner=runner)

    def test_git_query_refuses_a_forbidden_token_inside_an_allowed_subcommand(self):
        def runner(*_args, **_kwargs):
            raise AssertionError("must never be spawned")

        with self.assertRaises(status.GitQueryError):
            status.git_query(
                ["status", "--porcelain", "stash"], worktree=Path("/x"), runner=runner
            )

    def test_no_optional_locks_is_passed_on_every_call(self):
        seen: list[list[str]] = []

        def runner(argv, **_kwargs):
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 0, "out", "")

        status.git_query(["status", "--porcelain"], worktree=Path("/x"), runner=runner)
        self.assertIn(status.NO_OPTIONAL_LOCKS, seen[0])
        self.assertLess(seen[0].index(status.NO_OPTIONAL_LOCKS), seen[0].index("status"))

    def test_an_old_git_that_rejects_the_flag_is_retried_without_it(self):
        seen: list[list[str]] = []

        def runner(argv, **_kwargs):
            seen.append(argv)
            if status.NO_OPTIONAL_LOCKS in argv:
                return subprocess.CompletedProcess(
                    argv, 129, "", "error: unknown option `no-optional-locks'"
                )
            return subprocess.CompletedProcess(argv, 0, "recovered", "")

        out = status.git_query(
            ["status", "--porcelain"], worktree=Path("/x"), runner=runner
        )
        self.assertEqual(out, "recovered")
        self.assertEqual(len(seen), 2)
        self.assertNotIn(status.NO_OPTIONAL_LOCKS, seen[1])

    def test_a_genuine_failure_is_not_retried(self):
        seen: list[list[str]] = []

        def runner(argv, **_kwargs):
            seen.append(argv)
            return subprocess.CompletedProcess(argv, 128, "", "not a git repository")

        with self.assertRaises(status.GitQueryError):
            status.git_query(["rev-parse", "HEAD"], worktree=Path("/x"), runner=runner)
        self.assertEqual(len(seen), 1)


def _tree_snapshot(root: Path) -> dict:
    """Every path under `root` with its size and nanosecond mtime."""
    snapshot = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        snapshot[str(path.relative_to(root))] = (
            info.st_size,
            info.st_mtime_ns,
            info.st_mode,
        )
    return snapshot


@unittest.skipUnless(
    subprocess.run(
        ["git", "--version"], capture_output=True, check=False
    ).returncode
    == 0,
    "git is not runnable",
)
class RealRepositoryTests(unittest.TestCase):
    """End-to-end against a real repository, with real git.

    These are the tests that prove the two claims the module docstring makes
    about git's own behaviour, rather than restating them.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "repo"
        self.root.mkdir()
        self.addCleanup(self._tmp.cleanup)
        self._git("init", "-q", ".")
        self._git("config", "user.email", "t@example.invalid")
        self._git("config", "user.name", "Test")
        (self.root / "a.txt").write_text("hello\n", encoding="utf-8")
        (self.root / "b.txt").write_text("other\n", encoding="utf-8")
        self._git("add", "-A")
        self._git("commit", "-qm", "init")
        self.environ = {
            gate.LEASE_DIR_ENV: str(Path(self._tmp.name) / "lease"),
            gate.REPORT_DIR_ENV: str(Path(self._tmp.name) / "reports"),
        }

    def _git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def _probe(self):
        return status.collect(self.root, environ=self.environ, now=_now)

    def test_clean_repository_is_free(self):
        state = self._probe()
        self.assertEqual(state["verdict"], status.VERDICT_FREE)
        self.assertTrue(state["dirty"]["clean"])
        self.assertEqual(state["git"]["head"], self._git("rev-parse", "HEAD").strip())

    def test_a_real_modification_is_dirty(self):
        (self.root / "a.txt").write_text("CHANGED\n", encoding="utf-8")
        state = self._probe()
        self.assertEqual(state["dirty"]["modified"], ["a.txt"])
        self.assertEqual(state["verdict"], status.VERDICT_NOT_CLEAN)

    def test_an_identical_rewrite_is_stat_dirty_only_and_the_tree_stays_clean(self):
        """The measured finding the module docstring rests on: git's own
        status excludes a content-identical rewrite from the modified set, so
        no content re-hashing is needed here to tell the two apart."""
        os.utime(self.root / "a.txt", (0, 0))
        (self.root / "a.txt").write_text("hello\n", encoding="utf-8")
        os.utime(self.root / "a.txt", (1, 1))
        state = self._probe()
        self.assertEqual(state["dirty"]["modified"], [])
        self.assertTrue(state["dirty"]["clean"])
        self.assertEqual(state["verdict"], status.VERDICT_FREE)
        self.assertIn("a.txt", state["dirty"]["index_stale"])
        self.assertIn("stat-dirty only", status.render_human(state))

    def test_untracked_file_in_a_real_repository(self):
        (self.root / "probe.txt").write_text("reviewer was here\n", encoding="utf-8")
        state = self._probe()
        self.assertEqual(state["dirty"]["untracked"], ["probe.txt"])
        self.assertEqual(state["verdict"], status.VERDICT_NOT_CLEAN)

    def test_the_probe_writes_nothing_at_all(self):
        """The whole point. A reviewer's probe must not touch the checkout it
        is inspecting, and `.git` is part of the checkout: a plain
        `git status` rewrites the index when it has refreshed stat data worth
        saving, which is precisely the state a probe finds after somebody's
        formatter pass."""
        os.utime(self.root / "a.txt", (0, 0))
        (self.root / "a.txt").write_text("hello\n", encoding="utf-8")
        os.utime(self.root / "a.txt", (1, 1))
        before = _tree_snapshot(self.root)
        state = self._probe()
        after = _tree_snapshot(self.root)
        self.assertEqual(state["verdict"], status.VERDICT_FREE)
        self.assertEqual(before, after)
        self.assertIn("a.txt", state["dirty"]["index_stale"])

    def test_a_real_primary_checkout_is_not_labelled_linked(self):
        """The fakes could drift from what git actually prints, so the
        anchoring fix is also checked against real `rev-parse` output."""
        facts = status.git_facts(self.root)
        self.assertFalse(facts["linked"])
        self.assertTrue(Path(facts["git_common_dir"]).is_dir())
        self.assertEqual(
            Path(facts["git_common_dir"]).resolve(),
            (self.root / ".git").resolve(),
        )

    def test_a_real_linked_worktree_is_labelled_linked(self):
        linked = Path(self._tmp.name) / "linked"
        self._git("worktree", "add", "-q", "-b", "side", str(linked))
        self.addCleanup(
            lambda: subprocess.run(
                ["git", "-C", str(self.root), "worktree", "remove", "--force", str(linked)],
                check=False,
                capture_output=True,
            )
        )
        facts = status.git_facts(linked)
        self.assertTrue(facts["linked"])
        self.assertEqual(
            Path(facts["git_common_dir"]).resolve(), (self.root / ".git").resolve()
        )

    def test_a_real_subdirectory_resolves_to_the_repository_root(self):
        sub = self.root / "nested" / "deeper"
        sub.mkdir(parents=True)
        (sub / "c.txt").write_text("x\n", encoding="utf-8")
        state = status.collect(sub, environ=self.environ, now=_now)
        self.assertEqual(Path(state["worktree"]).resolve(), self.root.resolve())
        self.assertEqual(state["dirty"]["untracked"], ["nested/deeper/c.txt"])
        self.assertEqual(state["verdict"], status.VERDICT_NOT_CLEAN)

    def test_probing_a_directory_that_is_not_a_repository_is_unknown(self):
        outside = Path(self._tmp.name) / "not-a-repo"
        outside.mkdir()
        state = status.collect(outside, environ=self.environ, now=_now)
        self.assertEqual(state["verdict"], status.VERDICT_UNKNOWN)
        self.assertIn("git", [item["source"] for item in state["unknown"]])


class OutputTests(unittest.TestCase):
    def test_json_payload_shape(self):
        state = _collect()
        payload = json.loads(status.render_json(state))
        self.assertEqual(payload["schema_version"], status.SCHEMA_VERSION)
        self.assertEqual(payload["verdict"], status.VERDICT_FREE)
        self.assertEqual(payload["exit_code"], status.EXIT_FREE)
        self.assertEqual(payload["generated_at"], "2026-09-05T03:04:11Z")

    def test_json_and_human_agree_on_the_verdict(self):
        state = _collect(query=FakeGit(porcelain=_porcelain("?? probe.txt")))
        payload = json.loads(status.render_json(state))
        self.assertIn(f"VERDICT: {payload['verdict']}", status.render_human(state))

    def test_human_block_is_self_dating(self):
        """A brief that pastes a stale block should show its age, so the block
        carries the instant and the command that produced it."""
        block = status.render_human(_collect())
        self.assertIn("probed:   2026-09-05T03:04:11Z", block)
        self.assertIn("scripts/worktree_status.py", block)

    def test_main_returns_the_exit_code_and_prints_the_block(self):
        stream = io.StringIO()
        code = main_code = status.main(
            ["--path", PROBED],
            environ={gate.LEASE_DIR_ENV: "/nonexistent"},
            output_stream=stream,
            now=_now,
            query=FakeGit(porcelain=_porcelain(" M a.rs")),
            snapshot=_no_processes,
            cwd_lookup=_no_cwd,
            stat=lambda path: os.stat_result((0,) * 10),
            exists=lambda path: False,
        )
        self.assertEqual(code, status.EXIT_NOT_CLEAN)
        self.assertIn("VERDICT: NOT CLEAN", stream.getvalue())
        self.assertEqual(main_code, status.EXIT_NOT_CLEAN)

    def test_quiet_prints_only_the_verdict_line(self):
        stream = io.StringIO()
        status.main(
            ["--path", PROBED, "--quiet"],
            environ={gate.LEASE_DIR_ENV: "/nonexistent"},
            output_stream=stream,
            now=_now,
            query=FakeGit(),
            snapshot=_no_processes,
            cwd_lookup=_no_cwd,
            stat=lambda path: os.stat_result((0,) * 10),
            exists=lambda path: False,
        )
        self.assertEqual(stream.getvalue().strip(), "VERDICT: FREE")

    def test_json_and_quiet_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status.parse_args(["--json", "--quiet"])


class ImportDisciplineTests(unittest.TestCase):
    def test_importing_gate_does_not_re_execute_the_interpreter(self):
        """`gate.py` re-executes an unmanaged launcher through uv, but only
        under `__main__`. If that ever moved to module scope, importing it
        here would fork a uv process on every probe."""
        source = (Path(status.__file__).parent / "gate.py").read_text(encoding="utf-8")
        call = "ensure_managed_runtime(sys.argv[1:])"
        self.assertIn(call, source)
        guard = source.index('if __name__ == "__main__":')
        self.assertGreater(
            source.index(call),
            guard,
            "gate.ensure_managed_runtime must stay under the __main__ guard",
        )

    def test_the_shared_reaper_helpers_are_public(self):
        """`worktree_status` scopes its gate-process matching with these, so a
        rename back to private names must fail here rather than at run time."""
        self.assertTrue(callable(reap.command_mentions_path))
        self.assertTrue(callable(reap.path_is_under))
        self.assertTrue(reap.command_mentions_path("cargo build /a/b", "/a/b"))
        self.assertFalse(reap.command_mentions_path("cargo build /a/b-165", "/a/b"))


if __name__ == "__main__":
    unittest.main()
