"""Contract tests for the one-command OpenSpec submission.

The submission command must refuse before it acts. Every test that proves
a document change proceeds has the matching test proving a code change,
a symlink, or a validation finding stops it with nothing created.
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "openspec_submit.py"

SPEC = importlib.util.spec_from_file_location("openspec_submit", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load the submit module: {MODULE_PATH}")
submit_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = submit_module
SPEC.loader.exec_module(submit_module)

BLOB = "a" * 7
SPEC_DOC = "openspec/specs/tide/spec.md"
CHANGE_DOC = "openspec/changes/add-thing/proposal.md"


def raw(status: str, path: str, src: str = "100644", dst: str = "100644") -> str:
    return f":{src} {dst} {BLOB} {BLOB} {status}\0{path}\0"


class FakeGit:
    """A recording stand-in for every subprocess this command runs."""

    def __init__(
        self,
        *,
        diff: str = "",
        staged: str = "",
        untracked: str = "",
        unmerged: str = "",
        openspec_returncode: int = 0,
        failing: str = "",
        governance: tuple[str, ...] = (),
        auth_ok: bool = True,
        existing: list | None = None,
        pull_states: list[dict] | None = None,
        check_runs: list[dict] | None = None,
    ) -> None:
        self.diff = diff
        self.staged = staged
        self.untracked = untracked
        self.unmerged = unmerged
        self.openspec_returncode = openspec_returncode
        self.failing = failing
        self.governance = governance
        self.auth_ok = auth_ok
        self.existing = existing if existing is not None else []
        self.pull_states = pull_states
        self.check_runs = check_runs if check_runs is not None else []
        self.commands: list[list[str]] = []
        self.pull_views = 0

    DEFAULT_PULL = {
        "number": 42,
        "state": "MERGED",
        "url": "https://github.com/o/r/pull/42",
        "headRefOid": "c" * 40,
        "mergeCommit": {"oid": "d" * 40},
    }

    def __call__(self, command, **kwargs: object) -> SimpleNamespace:
        self.commands.append(list(command))
        joined = " ".join(command)
        if self.failing and self.failing in joined:
            return SimpleNamespace(returncode=1, stdout="", stderr="boom")
        if command[:3] == ["gh", "auth", "status"]:
            return SimpleNamespace(
                returncode=0 if self.auth_ok else 1,
                stdout="",
                stderr="" if self.auth_ok else "gh: not logged in",
            )
        if command[:3] == ["gh", "pr", "list"]:
            return SimpleNamespace(
                returncode=0, stdout=json.dumps(self.existing), stderr=""
            )
        if command[:3] == ["gh", "pr", "view"]:
            self.pull_views += 1
            if self.pull_states:
                index = min(self.pull_views - 1, len(self.pull_states) - 1)
                payload = self.pull_states[index]
            else:
                payload = self.DEFAULT_PULL
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "pr", "create"]:
            return SimpleNamespace(
                returncode=0,
                stdout="https://github.com/o/r/pull/42\n",
                stderr="",
            )
        if command[:2] == ["gh", "api"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"check_runs": self.check_runs}),
                stderr="",
            )
        if "--no-interactive" in command:
            return SimpleNamespace(
                returncode=self.openspec_returncode,
                stdout="finding: missing scenario",
                stderr="",
            )
        if "--verify" in command and ":" in command[-1]:
            revision, _, path = command[-1].partition(":")
            # A differing path gets a distinct object id on the head side.
            # Only the head side differs, so the base stays the reference.
            differs = path in self.governance and revision != "origin/main"
            suffix = "1" if differs else "0"
            return SimpleNamespace(returncode=0, stdout=f"{path}{suffix}\n", stderr="")
        if "rev-parse" in command and "HEAD" in command:
            return SimpleNamespace(returncode=0, stdout=f"{'c' * 40}\n", stderr="")
        if "rev-parse" in command:
            return SimpleNamespace(returncode=0, stdout=f"{REPO_ROOT}\n", stderr="")
        if "merge-base" in command:
            return SimpleNamespace(returncode=0, stdout=f"{'b' * 40}\n", stderr="")
        if "--unmerged" in command:
            return SimpleNamespace(returncode=0, stdout=self.unmerged, stderr="")
        if "ls-files" in command:
            return SimpleNamespace(returncode=0, stdout=self.untracked, stderr="")
        if "--cached" in command:
            return SimpleNamespace(returncode=0, stdout=self.staged, stderr="")
        if "diff" in command:
            return SimpleNamespace(returncode=0, stdout=self.diff, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def ran(self, needle: str) -> bool:
        return any(needle in " ".join(command) for command in self.commands)


class NamingTests(unittest.TestCase):
    def test_a_single_change_names_the_branch_after_itself(self) -> None:
        branch = submit_module.derive_slug([CHANGE_DOC])
        self.assertTrue(branch.startswith("openspec/add-thing-"))
        self.assertIn("add-thing", submit_module.derive_title([CHANGE_DOC], branch))

    def test_a_single_capability_names_the_branch_after_the_specification(
        self,
    ) -> None:
        branch = submit_module.derive_slug([SPEC_DOC])
        self.assertTrue(branch.startswith("openspec/spec-tide-"))
        self.assertIn("tide", submit_module.derive_title([SPEC_DOC], branch))

    def test_a_mixed_document_set_gets_a_neutral_name(self) -> None:
        paths = [SPEC_DOC, CHANGE_DOC, "openspec/changes/other/tasks.md"]
        branch = submit_module.derive_slug(paths)
        self.assertTrue(branch.startswith("openspec/documents-"))
        self.assertIn("3", submit_module.derive_title(paths, branch))

    def test_the_branch_name_is_stable_for_the_same_paths(self) -> None:
        self.assertEqual(
            submit_module.derive_slug([SPEC_DOC, CHANGE_DOC]),
            submit_module.derive_slug([CHANGE_DOC, SPEC_DOC]),
        )

    def test_different_paths_get_different_branches(self) -> None:
        self.assertNotEqual(
            submit_module.derive_slug([SPEC_DOC]),
            submit_module.derive_slug([CHANGE_DOC]),
        )

    def test_the_archive_directory_never_becomes_the_branch_name(self) -> None:
        branch = submit_module.derive_slug(
            ["openspec/changes/archive/add-thing/proposal.md"]
        )
        self.assertNotIn("archive", branch)


class UntrackedRecordTests(unittest.TestCase):
    def test_file_kinds_reach_the_classifier_with_their_real_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plain.md").write_text("x", encoding="utf-8")
            (root / "runme.md").write_text("x", encoding="utf-8")
            (root / "runme.md").chmod(0o755)
            (root / "link.md").symlink_to(root / "plain.md")
            (root / "tree").mkdir()
            stream = submit_module.untracked_records(
                root, ["plain.md", "runme.md", "link.md", "tree"]
            )
        self.assertIn(":000000 100644", stream)
        self.assertIn(":000000 100755", stream)
        self.assertIn(":000000 120000", stream)
        self.assertIn(":000000 040000", stream)

    def test_the_rendered_records_parse_as_real_diff_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "openspec"
            target.mkdir()
            (target / "project.md").write_text("x", encoding="utf-8")
            stream = submit_module.untracked_records(root, ["openspec/project.md"])
        records = submit_module.acceptance.parse_raw_diff(stream)
        self.assertEqual(records[0].status, "A")
        self.assertEqual(records[0].paths, ("openspec/project.md",))


class ValidationTests(unittest.TestCase):
    def test_a_clean_validation_passes(self) -> None:
        runner = FakeGit(openspec_returncode=0)
        with mock.patch.object(submit_module.shutil, "which", return_value="/bin/openspec"):
            submit_module.validate_openspec(REPO_ROOT, runner)

    def test_a_finding_requires_human_review(self) -> None:
        runner = FakeGit(openspec_returncode=1)
        with mock.patch.object(submit_module.shutil, "which", return_value="/bin/openspec"):
            with self.assertRaises(submit_module.ReviewRequired):
                submit_module.validate_openspec(REPO_ROOT, runner)

    def test_an_operational_exit_is_not_a_finding(self) -> None:
        runner = FakeGit(openspec_returncode=2)
        with mock.patch.object(submit_module.shutil, "which", return_value="/bin/openspec"):
            with self.assertRaises(submit_module.SubmitError):
                submit_module.validate_openspec(REPO_ROOT, runner)

    def test_a_missing_openspec_cli_is_operational(self) -> None:
        with mock.patch.object(submit_module.shutil, "which", return_value=None):
            with mock.patch.dict(submit_module.os.environ, {}, clear=False):
                submit_module.os.environ.pop("OPENSPEC_BIN", None)
                with self.assertRaises(submit_module.SubmitError):
                    submit_module.validate_openspec(REPO_ROOT, FakeGit())

    def test_a_timeout_is_operational(self) -> None:
        def raising(*_args: object, **_kwargs: object) -> SimpleNamespace:
            raise subprocess.TimeoutExpired("openspec", 600)

        with mock.patch.object(submit_module.shutil, "which", return_value="/bin/openspec"):
            with self.assertRaises(submit_module.SubmitError):
                submit_module.validate_openspec(REPO_ROOT, raising)


class EndToEndTests(unittest.TestCase):
    def run_main(self, argv: list[str], runner: FakeGit) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(submit_module.shutil, "which", return_value="/bin/tool"):
            with redirect_stdout(out), redirect_stderr(err):
                status = submit_module.main(argv, runner=runner)
        return status, out.getvalue(), err.getvalue()

    def test_a_document_change_produces_a_plan_and_creates_nothing_when_dry(
        self,
    ) -> None:
        runner = FakeGit(diff=raw("M", SPEC_DOC))
        status, out, err = self.run_main(["--dry-run", "--no-fetch"], runner)
        self.assertEqual(status, 0, err)
        self.assertIn("openspec/spec-tide-", out)
        self.assertIn("DRY RUN", out)
        self.assertFalse(runner.ran("git commit"))
        self.assertFalse(runner.ran("git push"))
        self.assertFalse(runner.ran("gh pr"))

    def test_a_full_submission_branches_commits_pushes_and_opens_a_request(
        self,
    ) -> None:
        runner = FakeGit(diff=raw("M", SPEC_DOC))
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 0, err)
        self.assertTrue(runner.ran("git checkout -b openspec/spec-tide-"))
        self.assertTrue(runner.ran(f"git add -- :(literal){SPEC_DOC}"))
        self.assertTrue(runner.ran("git commit --only -m docs(openspec)"))
        self.assertTrue(runner.ran("git push --set-upstream origin openspec/"))
        self.assertTrue(runner.ran("gh pr create --base main"))
        self.assertIn("opened", out)

    def test_the_submission_never_requests_auto_merge(self) -> None:
        """Auto-merge is a standing grant on a mutable branch.

        It survives later pushes by anyone with write access, so a verdict
        computed on one commit would authorize every commit after it. The
        command opens the pull request and stops.
        """
        runner = FakeGit(diff=raw("M", SPEC_DOC))
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 0, err)
        for forbidden in ("--auto", "gh pr merge", "--admin", "gh pr review"):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(runner.ran(forbidden))
        self.assertNotIn("auto-merge", out)

    def test_no_command_line_option_can_request_a_merge(self) -> None:
        for argv in (["--merge-method", "squash"], ["--auto"], ["--merge"]):
            with self.subTest(argv=argv):
                runner = FakeGit(diff=raw("M", SPEC_DOC))
                with self.assertRaises(SystemExit):
                    self.run_main(["--no-fetch", *argv], runner)

    def test_a_code_change_stops_before_anything_is_created(self) -> None:
        runner = FakeGit(diff=raw("M", "crates/chelis-types/src/lib.rs"))
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 1)
        self.assertIn("BLOCKED", out + err)
        self.assertIn("crates/chelis-types/src/lib.rs", out + err)
        self.assertFalse(runner.ran("git checkout"))
        self.assertFalse(runner.ran("gh pr"))

    def test_a_document_change_mixed_with_code_stops(self) -> None:
        runner = FakeGit(diff=raw("M", SPEC_DOC) + raw("M", "scripts/gate.py"))
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 1)
        self.assertIn("scripts/gate.py", out + err)
        self.assertFalse(runner.ran("gh pr"))

    def test_an_untracked_symlink_stops_the_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "openspec" / "specs" / "tide").mkdir(parents=True)
            real = root / "openspec" / "specs" / "tide" / "real.md"
            real.write_text("x", encoding="utf-8")
            (root / "openspec" / "specs" / "tide" / "spec.md").symlink_to(real)
            runner = FakeGit(untracked=f"{SPEC_DOC}\0")
            with mock.patch.object(
                submit_module, "repository_root", return_value=root
            ):
                status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 1)
        self.assertIn("file mode", out + err)
        self.assertFalse(runner.ran("gh pr"))

    def test_an_untracked_stray_outside_the_document_set_is_ignored(self) -> None:
        """Measured on the real tree: 37 untracked non-document paths.

        `result`, `.work/`, and build output are untracked and unignored
        there, so classifying them refuses every document submission. They
        cannot reach a commit: untracked files enter one only through an
        explicit `git add`, and this command adds exactly the classified
        paths as `:(literal)` pathspecs. Refusing on them proves nothing
        and shuts the front door.
        """
        runner = FakeGit(
            diff=raw("M", SPEC_DOC),
            untracked="result\0.work/notes.md\0target/debug/x\0",
        )
        status, out, err = self.run_main(["--dry-run", "--no-fetch"], runner)
        self.assertEqual(status, 0, err)
        self.assertNotIn("result", err)
        self.assertNotIn(".work/notes.md", out)
        self.assertNotIn("target/debug/x", out)

    def test_a_stray_is_ignored_without_reaching_the_commit(self) -> None:
        runner = FakeGit(diff=raw("M", SPEC_DOC), untracked="result\0")
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 0, err)
        for command in runner.commands:
            joined = " ".join(command)
            if joined.startswith(("git add", "git commit")):
                with self.subTest(command=joined):
                    self.assertNotIn("result", joined)

    def test_an_untracked_document_path_is_still_classified(self) -> None:
        """The mode check on untracked documents is what catches a symlink.

        Ignoring strays must not become ignoring untracked documents: a
        document path IS committed, so its filesystem mode still decides.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "openspec" / "specs" / "tide").mkdir(parents=True)
            real = root / "openspec" / "specs" / "tide" / "real.md"
            real.write_text("x", encoding="utf-8")
            (root / "openspec" / "specs" / "tide" / "spec.md").symlink_to(real)
            runner = FakeGit(untracked=f"{SPEC_DOC}\0result\0")
            with mock.patch.object(
                submit_module, "repository_root", return_value=root
            ):
                status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 1)
        self.assertIn("file mode", out + err)

    def test_a_tracked_change_outside_the_document_set_still_stops(self) -> None:
        """Only UNTRACKED strays are ignored. The index is a commit input."""
        for stream in (raw("M", "scripts/gate.py"), raw("A", "scripts/backdoor.py")):
            with self.subTest(stream=stream):
                runner = FakeGit(diff=stream, untracked="result\0")
                status, out, err = self.run_main(["--no-fetch"], runner)
                self.assertEqual(status, 1)
                self.assertFalse(runner.ran("gh pr"))

    def test_an_empty_worktree_stops_with_nothing_to_submit(self) -> None:
        runner = FakeGit()
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 1)
        self.assertIn("nothing to submit", out + err)

    def test_a_validation_finding_stops_before_the_branch_is_created(self) -> None:
        runner = FakeGit(diff=raw("M", SPEC_DOC), openspec_returncode=1)
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 1)
        self.assertIn("validation", (out + err).lower())
        self.assertFalse(runner.ran("git checkout"))

    def test_a_push_failure_is_operational(self) -> None:
        runner = FakeGit(diff=raw("M", SPEC_DOC), failing="git push")
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 2)
        self.assertIn("FAILED", out + err)

    def test_the_index_is_never_reset_or_staged_before_the_decision(self) -> None:
        runner = FakeGit(diff=raw("M", "scripts/gate.py"))
        self.run_main(["--no-fetch"], runner)
        self.assertFalse(runner.ran("git add"))
        self.assertFalse(runner.ran("git reset"))

    def test_a_staged_code_change_stops_the_submission(self) -> None:
        """The classified set must be the committed set.

        A file staged with one content and left unchanged on disk is
        invisible to a working-tree-only diff, and an ordinary `git commit`
        would carry it into the automatically merged pull request.
        """
        runner = FakeGit(
            diff=raw("M", SPEC_DOC), staged=raw("M", "scripts/gate.py")
        )
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 1)
        self.assertIn("scripts/gate.py", out + err)
        self.assertFalse(runner.ran("git checkout"))
        self.assertFalse(runner.ran("gh pr"))

    def test_a_staged_addition_that_is_gone_from_disk_stops_the_submission(
        self,
    ) -> None:
        runner = FakeGit(
            diff=raw("M", SPEC_DOC),
            staged=raw("A", "scripts/backdoor.py", src="000000"),
        )
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 1)
        self.assertIn("scripts/backdoor.py", out + err)

    def test_the_commit_carries_only_the_classified_paths(self) -> None:
        runner = FakeGit(diff=raw("M", SPEC_DOC))
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 0, err)
        commit = next(c for c in runner.commands if c[:2] == ["git", "commit"])
        self.assertIn("--only", commit)
        self.assertEqual(commit[-1], f":(literal){SPEC_DOC}")

    def test_paths_reach_git_as_filenames_rather_than_globs(self) -> None:
        """A file named `openspec/specs/*.md` is a valid path.

        Passed as an ordinary pathspec it would expand across the tree and
        commit files the classifier never saw.
        """
        glob_path = "openspec/specs/tide/*.md"
        runner = FakeGit(diff=raw("M", glob_path))
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 0, err)
        for name in ("add", "commit"):
            command = next(c for c in runner.commands if c[:2] == ["git", name])
            with self.subTest(command=name):
                self.assertIn(f":(literal){glob_path}", command)
                self.assertNotIn(glob_path, command)

    def test_the_comparison_starts_at_the_merge_base(self) -> None:
        runner = FakeGit(diff=raw("M", SPEC_DOC))
        self.run_main(["--no-fetch"], runner)
        self.assertTrue(runner.ran("git merge-base origin/main HEAD"))
        diffs = [c for c in runner.commands if "diff" in c]
        self.assertTrue(diffs)
        for command in diffs:
            with self.subTest(command=command):
                self.assertIn("b" * 40, command)
                self.assertNotIn("origin/main", command)

    def test_a_stale_governance_branch_stops_the_submission(self) -> None:
        """The local verdict must match the one CI would report.

        The hosted classifier refuses a head whose governance content
        differs from the base. If the command did not check the same
        thing, it would push a branch that CI immediately refuses.
        """
        runner = FakeGit(diff=raw("M", SPEC_DOC), governance=(".github",))
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 1)
        self.assertIn(".github", out + err)
        self.assertIn("rebase", (out + err).lower())
        self.assertFalse(runner.ran("git checkout -b"))
        self.assertFalse(runner.ran("gh pr"))

    def test_the_governance_check_compares_the_base_tip_to_the_head(self) -> None:
        runner = FakeGit(diff=raw("M", SPEC_DOC))
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 0, err)
        self.assertTrue(runner.ran("rev-parse --verify --quiet"))
        inspected = {
            command[-1].split(":", 1)[1]
            for command in runner.commands
            if "--verify" in command and ":" in command[-1]
        }
        self.assertEqual(inspected, set(submit_module.acceptance.GOVERNANCE_PATHS))

    def test_an_unresolved_merge_stops_the_submission(self) -> None:
        runner = FakeGit(
            diff=raw("M", SPEC_DOC),
            unmerged=f"100644 {'a' * 40} 1\topenspec/specs/tide/spec.md\0",
        )
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, 1)
        self.assertIn("unresolved merge", out + err)
        self.assertFalse(runner.ran("gh pr"))

    def test_the_pull_request_body_states_the_limits_of_the_evidence(self) -> None:
        verdict = submit_module.acceptance.classify_stream(raw("M", SPEC_DOC))
        body = submit_module.build_body([SPEC_DOC], verdict)
        self.assertIn("schema validity only", body)
        self.assertIn("does not", body)
        self.assertIn(SPEC_DOC, body)




class UserExperienceTests(unittest.TestCase):
    """The command reports stages and one final, unambiguous outcome.

    "A pull request was created" is not an outcome. The default run waits
    for the merge to happen or to fail, and says which.
    """

    def run_main(self, argv: list[str], runner: FakeGit) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(submit_module.shutil, "which", return_value="/bin/tool"):
            with mock.patch.object(submit_module.time, "sleep", lambda _s: None):
                with redirect_stdout(out), redirect_stderr(err):
                    status = submit_module.main(argv, runner=runner)
        return status, out.getvalue(), err.getvalue()

    def document_change(self, **kwargs: object) -> FakeGit:
        return FakeGit(diff=raw("M", SPEC_DOC), **kwargs)  # type: ignore[arg-type]

    # --- stage reporting -------------------------------------------------

    def test_every_stage_is_reported_in_order(self) -> None:
        status, out, err = self.run_main(["--no-fetch"], self.document_change())
        self.assertEqual(status, submit_module.ACCEPTED, err + out)
        positions = [
            out.index(f"  {name:<9}") for name in ("prepare", "validate", "submit", "wait")
        ]
        self.assertEqual(positions, sorted(positions))

    def test_the_last_line_is_the_outcome(self) -> None:
        _, out, _ = self.run_main(["--no-fetch"], self.document_change())
        self.assertTrue(out.strip().splitlines()[-1].strip().startswith("ACCEPTED"))

    # --- accepted vs queued ---------------------------------------------

    def test_a_merged_pull_request_reports_accepted_with_the_merge_commit(
        self,
    ) -> None:
        status, out, _ = self.run_main(["--no-fetch"], self.document_change())
        self.assertEqual(status, submit_module.ACCEPTED)
        self.assertIn("ACCEPTED", out)
        self.assertIn("d" * 7, out)
        self.assertNotIn("QUEUED", out)

    def test_no_wait_reports_queued_with_the_url_and_never_accepted(self) -> None:
        runner = self.document_change()
        status, out, _ = self.run_main(["--no-fetch", "--no-wait"], runner)
        self.assertEqual(status, submit_module.ACCEPTED)
        self.assertIn("QUEUED", out)
        self.assertIn("https://github.com/o/r/pull/42", out)
        self.assertNotIn("ACCEPTED", out)
        self.assertFalse(runner.ran("gh pr view"))

    def test_a_wait_that_expires_reports_queued_and_how_to_resume(self) -> None:
        open_pull = dict(FakeGit.DEFAULT_PULL, state="OPEN", mergeCommit=None)
        runner = self.document_change(pull_states=[open_pull])
        status, out, _ = self.run_main(
            ["--no-fetch", "--wait-timeout", "0"], runner
        )
        self.assertEqual(status, submit_module.TIMED_OUT)
        self.assertIn("QUEUED", out)
        self.assertIn("https://github.com/o/r/pull/42", out)
        self.assertIn("openspec-submit --watch", out)

    def test_watch_resumes_waiting_without_creating_anything(self) -> None:
        runner = self.document_change()
        status, out, _ = self.run_main(["--no-fetch", "--watch", "42"], runner)
        self.assertEqual(status, submit_module.ACCEPTED)
        self.assertIn("ACCEPTED", out)
        for forbidden in ("git checkout", "git commit", "git push", "gh pr create"):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(runner.ran(forbidden))

    # --- blocked outcomes, each named precisely --------------------------

    def test_a_validation_finding_is_reported_as_validation_not_approval(
        self,
    ) -> None:
        runner = self.document_change(openspec_returncode=1)
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, submit_module.BLOCKED)
        combined = out + err
        self.assertIn("BLOCKED", combined)
        self.assertIn("validation", combined.lower())
        self.assertNotIn("human review required", combined)
        self.assertNotIn("approval", combined.lower())
        self.assertFalse(runner.ran("git checkout -b"))

    def test_a_code_path_is_reported_as_a_boundary_refusal(self) -> None:
        runner = FakeGit(diff=raw("M", "crates/chelis-types/src/lib.rs"))
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, submit_module.BLOCKED)
        combined = out + err
        self.assertIn("crates/chelis-types/src/lib.rs", combined)
        self.assertNotIn("human review required", combined)
        self.assertIn("not an OpenSpec document", combined)

    def test_a_stale_branch_is_reported_with_the_rebase_remedy(self) -> None:
        runner = self.document_change(governance=(".github",))
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, submit_module.BLOCKED)
        combined = out + err
        self.assertIn(".github", combined)
        self.assertIn("git rebase origin/main", combined)

    def test_a_failed_required_check_is_named(self) -> None:
        open_pull = dict(FakeGit.DEFAULT_PULL, state="OPEN", mergeCommit=None)
        runner = self.document_change(
            pull_states=[open_pull],
            check_runs=[
                {
                    "name": "Integration Tests (Linux)",
                    "status": "completed",
                    "conclusion": "failure",
                }
            ],
        )
        status, out, _ = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, submit_module.BLOCKED)
        self.assertIn("BLOCKED", out)
        self.assertIn("Integration Tests (Linux)", out)

    def test_a_strict_validation_failure_on_the_pushed_commit_is_named(self) -> None:
        open_pull = dict(FakeGit.DEFAULT_PULL, state="OPEN", mergeCommit=None)
        runner = self.document_change(
            pull_states=[open_pull],
            check_runs=[
                {
                    "name": "OpenSpec Strict Validation",
                    "status": "completed",
                    "conclusion": "failure",
                }
            ],
        )
        status, out, _ = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, submit_module.BLOCKED)
        self.assertIn("OpenSpec Strict Validation", out)

    def test_a_head_that_changed_stops_the_wait(self) -> None:
        moved = dict(FakeGit.DEFAULT_PULL, state="OPEN", headRefOid="f" * 40, mergeCommit=None)
        runner = self.document_change(pull_states=[moved])
        status, out, _ = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, submit_module.BLOCKED)
        self.assertIn("BLOCKED", out)
        self.assertIn("head", out.lower())

    def test_a_pull_request_closed_without_merging_is_blocked(self) -> None:
        closed = dict(FakeGit.DEFAULT_PULL, state="CLOSED", mergeCommit=None)
        runner = self.document_change(pull_states=[closed])
        status, out, _ = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, submit_module.BLOCKED)
        self.assertIn("closed without merging", out)

    # --- preflight -------------------------------------------------------

    def test_missing_authentication_fails_before_any_write(self) -> None:
        runner = self.document_change(auth_ok=False)
        status, out, err = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, submit_module.FAILED)
        combined = out + err
        self.assertIn("gh auth login", combined)
        for forbidden in ("git checkout", "git commit", "git push", "gh pr create"):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(runner.ran(forbidden))

    def test_a_missing_tool_names_the_tool(self) -> None:
        runner = self.document_change()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(submit_module.shutil, "which", side_effect=lambda n: None if n == "gh" else "/bin/x"):
            with redirect_stdout(out), redirect_stderr(err):
                status = submit_module.main(["--no-fetch"], runner=runner)
        self.assertEqual(status, submit_module.FAILED)
        self.assertIn("gh", (out.getvalue() + err.getvalue()))
        self.assertFalse(runner.ran("git checkout"))

    def test_a_dry_run_needs_no_authentication(self) -> None:
        runner = self.document_change(auth_ok=False)
        status, out, err = self.run_main(["--dry-run", "--no-fetch"], runner)
        self.assertEqual(status, submit_module.ACCEPTED, err)
        self.assertIn("dry run", out.lower())

    # --- duplicate submissions ------------------------------------------

    def test_an_open_pull_request_for_the_same_content_is_not_duplicated(
        self,
    ) -> None:
        runner = self.document_change(
            existing=[
                {
                    "number": 42,
                    "state": "OPEN",
                    "url": "https://github.com/o/r/pull/42",
                    "headRefOid": "c" * 40,
                }
            ]
        )
        status, out, _ = self.run_main(["--no-fetch", "--no-wait"], runner)
        self.assertEqual(status, submit_module.ACCEPTED)
        self.assertIn("already", out.lower())
        self.assertFalse(runner.ran("gh pr create"))
        self.assertFalse(runner.ran("git push"))

    def test_an_already_merged_submission_reports_accepted_without_writing(
        self,
    ) -> None:
        runner = self.document_change(
            existing=[
                {
                    "number": 42,
                    "state": "MERGED",
                    "url": "https://github.com/o/r/pull/42",
                    "headRefOid": "c" * 40,
                }
            ]
        )
        status, out, _ = self.run_main(["--no-fetch"], runner)
        self.assertEqual(status, submit_module.ACCEPTED)
        self.assertIn("ACCEPTED", out)
        self.assertFalse(runner.ran("gh pr create"))

    # --- dry run honesty -------------------------------------------------

    def test_a_dry_run_states_the_remote_read_it_performed(self) -> None:
        runner = self.document_change()
        status, out, _ = self.run_main(["--dry-run"], runner)
        self.assertEqual(status, submit_module.ACCEPTED)
        self.assertTrue(runner.ran("git fetch"))
        self.assertIn("read-only", out.lower())

    def test_a_dry_run_writes_nothing_anywhere(self) -> None:
        runner = self.document_change()
        self.run_main(["--dry-run", "--no-fetch"], runner)
        for forbidden in (
            "git checkout",
            "git add",
            "git commit",
            "git push",
            "gh pr create",
            "gh pr merge",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(runner.ran(forbidden))

    def test_a_dry_run_lists_the_exact_paths_and_branch(self) -> None:
        runner = self.document_change()
        _, out, _ = self.run_main(["--dry-run", "--no-fetch"], runner)
        self.assertIn(SPEC_DOC, out)
        self.assertIn("openspec/spec-tide-", out)


if __name__ == "__main__":
    unittest.main()
