"""Contract tests for the OpenSpec automatic-acceptance boundary.

Every positive case has the matching negative case, because the failure
direction is the one that matters: a wrong "review" wastes a click, a
wrong "auto" merges unreviewed code.
"""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "openspec_acceptance.py"

SPEC = importlib.util.spec_from_file_location("openspec_acceptance", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load the acceptance module: {MODULE_PATH}")
acceptance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acceptance
SPEC.loader.exec_module(acceptance)

BLOB = "a" * 7
ZERO = "0" * 7
CHANGE_DOC_FOR_DELETE = "openspec/changes/add-thing/tasks.md"


def record(status: str, *paths: str, src: str = "100644", dst: str = "100644") -> str:
    """Build one `git diff --raw -z` record."""
    return f":{src} {dst} {BLOB} {BLOB} {status}\0" + "".join(f"{p}\0" for p in paths)


def added(path: str, **kwargs: str) -> str:
    return record("A", path, src="000000", **kwargs)


def modified(path: str, **kwargs: str) -> str:
    return record("M", path, **kwargs)


def deleted(path: str, **kwargs: str) -> str:
    return record("D", path, dst="000000", **kwargs)


def renamed(source: str, target: str) -> str:
    return f":100644 100644 {BLOB} {BLOB} R100\0{source}\0{target}\0"


class DocumentClassTests(unittest.TestCase):
    def test_openspec_documents_are_recognized(self) -> None:
        cases = {
            "openspec/specs/type-system/spec.md": "spec",
            "openspec/specs/tide/notes/detail.md": "spec",
            "openspec/changes/add-thing/proposal.md": "change",
            "openspec/changes/add-thing/.openspec.yaml": "change",
            "openspec/changes/add-thing/specs/tide/spec.md": "change",
            "openspec/changes/archive/old-thing/tasks.md": "change",
            "openspec/project.md": "project",
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(acceptance.document_class(path), expected)

    def test_each_path_safety_rule_is_pinned_on_its_own(self) -> None:
        """Each rule gets a case only it rejects.

        Grouping these in one list let two rules cover for each other, so
        deleting either left the suite green. One case per rule fixes that.
        """
        cases = {
            "parent-component": "openspec/specs/../../scripts/x.md",
            "current-component": "openspec/specs/./tide/spec.md",
            "empty-component": "openspec/specs//spec.md",
            "absolute": "/openspec/specs/tide/spec.md",
            "backslash": "openspec\\specs\\tide\\spec.md",
            "newline": "openspec/specs/tide/spec.md\nopenspec/x.md",
            "carriage-return": "openspec/specs/tide/spec\r.md",
            "null": "openspec/specs/tide/spec\x00.md",
            "delete-character": "openspec/specs/tide/spec\x7f.md",
            "leading-space": " openspec/specs/tide/spec.md",
            "trailing-space": "openspec/specs/tide/spec.md ",
        }
        for name, path in cases.items():
            with self.subTest(name=name):
                self.assertFalse(acceptance.is_safe_path(path))
                self.assertIsNone(acceptance.document_class(path))

    def test_non_documents_are_refused(self) -> None:
        cases = (
            "openspec/config.yaml",
            "openspec/specs/tide/fixture.json",
            "openspec/specs/tide/spec.md.bak",
            "openspec/changes/add-thing/run.py",
            "openspec/changes/add-thing/nested.yaml",
            "openspec/README.md.tar",
            "scripts/openspec_acceptance.py",
            "scripts/gate.py",
            ".github/workflows/ci.yml",
            "spec/04-type-system.md",
            "AGENTS.md",
            "crates/chelis-types/src/lib.rs",
            "openspec/../etc/passwd",
            "openspec/specs/../../secrets.md",
            "/openspec/specs/tide/spec.md",
            "openspec\\specs\\tide\\spec.md",
            "openspec/specs/tide/spec.md\n",
            "openspec/specs//spec.md",
            "",
        )
        for path in cases:
            with self.subTest(path=path):
                self.assertIsNone(acceptance.document_class(path))


class ParserTests(unittest.TestCase):
    def test_records_parse_with_their_modes_and_paths(self) -> None:
        stream = (
            modified("openspec/specs/tide/spec.md")
            + renamed(
                "openspec/changes/a/proposal.md",
                "openspec/changes/archive/a/proposal.md",
            )
            + deleted("openspec/changes/a/tasks.md")
        )
        records = acceptance.parse_raw_diff(stream)
        self.assertEqual([r.status for r in records], ["M", "R", "D"])
        self.assertEqual(len(records[1].paths), 2)
        self.assertEqual(records[2].dst_mode, "000000")

    def test_unreadable_records_raise_rather_than_being_skipped(self) -> None:
        cases = {
            "combined-merge": "::100644 100644 100644 aaaaaaa bbbbbbb MM\0f.md\0",
            "not-a-record": "openspec/specs/tide/spec.md\0",
            "missing-path": f":100644 100644 {BLOB} {BLOB} M\0",
            "missing-rename-target": (
                f":100644 100644 {BLOB} {BLOB} R100\0openspec/changes/a/x.md\0"
            ),
            "empty-path": f":100644 100644 {BLOB} {BLOB} M\0\0",
            "short-mode": f":10644 100644 {BLOB} {BLOB} M\0f.md\0",
            "lowercase-status": f":100644 100644 {BLOB} {BLOB} m\0f.md\0",
        }
        for name, stream in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(acceptance.BoundaryError):
                    acceptance.parse_raw_diff(stream)

    def test_a_parse_failure_becomes_a_review_verdict(self) -> None:
        verdict = acceptance.classify_stream("garbage\0")
        self.assertFalse(verdict.auto)
        self.assertTrue(verdict.reasons)


class AcceptanceTests(unittest.TestCase):
    def assert_auto(self, stream: str) -> None:
        verdict = acceptance.classify_stream(stream)
        self.assertTrue(verdict.auto, verdict.reasons)

    def assert_review(self, stream: str, needle: str = "") -> None:
        verdict = acceptance.classify_stream(stream)
        self.assertFalse(verdict.auto)
        self.assertTrue(verdict.reasons)
        if needle:
            self.assertIn(needle, verdict.reason)

    def test_normative_specification_edits_land_automatically(self) -> None:
        self.assert_auto(modified("openspec/specs/type-system/spec.md"))

    def test_a_new_capability_specification_lands_automatically(self) -> None:
        self.assert_auto(added("openspec/specs/new-capability/spec.md"))

    def test_a_complete_change_proposal_lands_automatically(self) -> None:
        self.assert_auto(
            added("openspec/changes/add-thing/.openspec.yaml")
            + added("openspec/changes/add-thing/proposal.md")
            + added("openspec/changes/add-thing/design.md")
            + added("openspec/changes/add-thing/tasks.md")
            + added("openspec/changes/add-thing/specs/tide/spec.md")
            + modified("openspec/specs/tide/spec.md")
        )

    def test_archiving_a_change_lands_automatically(self) -> None:
        self.assert_auto(
            renamed(
                "openspec/changes/add-thing/proposal.md",
                "openspec/changes/archive/add-thing/proposal.md",
            )
            + deleted("openspec/changes/add-thing/tasks.md")
        )

    def test_an_empty_change_set_requires_review(self) -> None:
        self.assert_review("", "empty")

    def test_a_code_change_requires_review(self) -> None:
        self.assert_review(
            modified("crates/chelis-types/src/lib.rs"),
            "outside the OpenSpec document boundary",
        )

    def test_a_document_change_mixed_with_code_requires_review(self) -> None:
        self.assert_review(
            modified("openspec/specs/tide/spec.md")
            + modified("crates/chelis-types/src/lib.rs"),
            "crates/chelis-types/src/lib.rs",
        )

    def test_editing_acceptance_policy_requires_review(self) -> None:
        for path in sorted(acceptance.POLICY_PATHS):
            with self.subTest(path=path):
                self.assert_review(modified(path), "acceptance policy itself")

    def test_editing_openspec_tool_configuration_requires_review(self) -> None:
        self.assert_review(modified("openspec/config.yaml"))

    def test_a_symlink_requires_review(self) -> None:
        self.assert_review(
            record("A", "openspec/specs/tide/spec.md", src="000000", dst="120000"),
            "file mode",
        )

    def test_turning_a_document_into_a_symlink_requires_review(self) -> None:
        self.assert_review(
            record("M", "openspec/specs/tide/spec.md", dst="120000"), "file mode"
        )

    def test_an_executable_document_requires_review(self) -> None:
        self.assert_review(
            record("A", "openspec/changes/a/proposal.md", src="000000", dst="100755"),
            "file mode",
        )

    def test_a_submodule_pointer_requires_review(self) -> None:
        self.assert_review(
            record("A", "openspec/changes/a/vendor", src="000000", dst="160000")
        )

    def test_a_type_change_requires_review(self) -> None:
        self.assert_review(
            record("T", "openspec/specs/tide/spec.md", dst="120000"),
            "outside the add, modify, delete, and rename set",
        )

    def test_unknown_and_conflicted_statuses_require_review(self) -> None:
        for status in ("U", "X", "B"):
            with self.subTest(status=status):
                self.assert_review(record(status, "openspec/specs/tide/spec.md"))

    def test_a_copy_record_requires_review(self) -> None:
        self.assert_review(
            f":100644 100644 {BLOB} {BLOB} C100\0"
            "openspec/specs/tide/spec.md\0openspec/specs/tide-copy/spec.md\0",
            "outside the add, modify, delete, and rename set",
        )

    def test_deleting_a_capability_specification_requires_review(self) -> None:
        self.assert_review(
            deleted("openspec/specs/tide/spec.md"),
            "deletes a normative OpenSpec document",
        )

    def test_deleting_the_project_document_requires_review(self) -> None:
        self.assert_review(deleted("openspec/project.md"))

    def test_moving_a_specification_out_of_the_spec_tree_requires_review(self) -> None:
        self.assert_review(
            renamed("openspec/specs/tide/spec.md", "openspec/changes/a/spec.md"),
            "moves a capability specification out of",
        )

    def test_renaming_a_specification_within_the_spec_tree_is_automatic(self) -> None:
        self.assert_auto(
            renamed("openspec/specs/tide/spec.md", "openspec/specs/tide-core/spec.md")
        )

    def test_moving_the_project_document_away_requires_review(self) -> None:
        """A move out of a normative slot is a deletion wearing a rename.

        Deleting `openspec/project.md` already routes to review. A rename
        to any other document path removes it just as completely, and both
        paths are documents, so nothing else in the rules objects.
        """
        self.assert_review(
            renamed("openspec/project.md", "openspec/changes/a/project.md"),
            "moves a normative OpenSpec document",
        )

    def test_renaming_the_project_document_in_place_is_not_a_thing(self) -> None:
        """The only accepted target for `project.md` is itself."""
        self.assert_auto(renamed("openspec/project.md", "openspec/project.md"))

    def test_a_rename_that_escapes_the_document_tree_requires_review(self) -> None:
        self.assert_review(
            renamed("openspec/changes/a/proposal.md", "scripts/proposal.md"),
            "scripts/proposal.md",
        )

    def test_a_rename_that_enters_from_outside_requires_review(self) -> None:
        self.assert_review(
            renamed("spec/04-type-system.md", "openspec/specs/tide/spec.md"),
            "spec/04-type-system.md",
        )

    def test_each_status_pins_both_of_its_file_modes(self) -> None:
        """Every status/mode pair, so no single mode check can be deleted."""
        document = "openspec/changes/add-thing/proposal.md"
        wrong = {
            "add-from-symlink": record("A", document, src="120000", dst="100644"),
            "add-to-symlink": record("A", document, src="000000", dst="120000"),
            "modify-from-symlink": record("M", document, src="120000"),
            "modify-to-symlink": record("M", document, dst="120000"),
            "delete-a-symlink": record("D", document, src="120000", dst="000000"),
            "delete-a-gitlink": record("D", document, src="160000", dst="000000"),
            "delete-leaves-content": record("D", document, dst="100644"),
            "rename-from-symlink": (
                f":120000 100644 {BLOB} {BLOB} R100\0{document}\0"
                "openspec/changes/add-thing/renamed.md\0"
            ),
            "rename-to-executable": (
                f":100644 100755 {BLOB} {BLOB} R100\0{document}\0"
                "openspec/changes/add-thing/renamed.md\0"
            ),
        }
        for name, stream in wrong.items():
            with self.subTest(name=name):
                self.assert_review(stream, "file mode")

    def test_a_deleted_code_file_is_not_called_a_normative_document(self) -> None:
        verdict = acceptance.classify_stream(deleted(".githooks/commit-msg"))
        self.assertFalse(verdict.auto)
        self.assertNotIn("normative OpenSpec document", verdict.reason)
        self.assertIn("outside the OpenSpec document boundary", verdict.reason)

    def test_a_deleted_change_document_is_not_called_normative(self) -> None:
        verdict = acceptance.classify_stream(deleted(CHANGE_DOC_FOR_DELETE))
        self.assertTrue(verdict.auto, verdict.reasons)

    def test_the_summary_is_bounded_while_the_reason_list_is_complete(self) -> None:
        stream = "".join(
            modified(f"crates/chelis-types/src/file{index}.rs") for index in range(40)
        )
        verdict = acceptance.classify_stream(stream)
        self.assertEqual(len(verdict.reasons), 40)
        self.assertIn("and 30 more", verdict.reason)
        self.assertLess(len(verdict.reason), 1500)

    def test_every_blocking_path_is_reported_once(self) -> None:
        verdict = acceptance.classify_stream(
            modified("scripts/gate.py") + modified("scripts/gate.py")
        )
        self.assertEqual(len(verdict.reasons), 1)


class RevisionTests(unittest.TestCase):
    def test_only_the_two_expected_revision_spellings_are_accepted(self) -> None:
        self.assertTrue(acceptance.valid_revision("origin/main"))
        self.assertTrue(acceptance.valid_revision("b" * 40))
        for value in ("main", "HEAD", "B" * 40, "b" * 39, "; rm -rf /", "--upload-pack"):
            with self.subTest(value=value):
                self.assertFalse(acceptance.valid_revision(value))

    def test_an_invalid_revision_never_reaches_git(self) -> None:
        def runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
            raise AssertionError("Git must not run for an invalid revision")

        with self.assertRaises(acceptance.BoundaryError):
            acceptance.read_raw_diff(REPO_ROOT, "main", None, runner)
        with self.assertRaises(acceptance.BoundaryError):
            acceptance.read_raw_diff(REPO_ROOT, "origin/main", "HEAD", runner)

    def test_a_git_failure_is_operational_not_a_verdict(self) -> None:
        def failing(*_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(returncode=128, stdout="", stderr="bad object")

        with self.assertRaises(acceptance.BoundaryError):
            acceptance.read_raw_diff(REPO_ROOT, "origin/main", None, failing)

        def raising(*_args: object, **_kwargs: object) -> SimpleNamespace:
            raise subprocess.TimeoutExpired("git", 120)

        with self.assertRaises(acceptance.BoundaryError):
            acceptance.read_raw_diff(REPO_ROOT, "origin/main", None, raising)

    def test_the_worktree_comparison_omits_a_head_revision(self) -> None:
        seen: dict[str, object] = {}

        def runner(command, **kwargs: object) -> SimpleNamespace:
            seen["command"] = command
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        acceptance.read_raw_diff(REPO_ROOT, "origin/main", None, runner)
        self.assertIn("origin/main", seen["command"])

    def test_a_two_revision_comparison_uses_the_merge_base(self) -> None:
        """Two-dot would report commits the pull request never made.

        Measured on this repository, 16 of 40 sampled pull requests had a
        `base.sha` behind the merge base, so a two-dot range refuses a
        document change for somebody else's code.
        """
        seen: dict[str, object] = {}

        def runner(command, **kwargs: object) -> SimpleNamespace:
            seen["command"] = command
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        head = "c" * 40
        acceptance.read_raw_diff(REPO_ROOT, "origin/main", head, runner)
        self.assertIn(f"origin/main...{head}", seen["command"])
        self.assertNotIn("origin/main", seen["command"])

    def test_the_revision_is_separated_from_any_path(self) -> None:
        seen: dict[str, object] = {}

        def runner(command, **kwargs: object) -> SimpleNamespace:
            seen["command"] = command
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        acceptance.read_raw_diff(REPO_ROOT, "origin/main", None, runner)
        self.assertEqual(seen["command"][-1], "--")


class GovernanceIdentityTests(unittest.TestCase):
    """Governance paths are compared by object identity, not by diff.

    A diff is taken against a merge base, so a head that simply predates a
    change to `.github/` shows no difference there while actually carrying
    an older workflow. Comparing tree objects between head and base answers
    the question that matters -- "is the governance content the same?" --
    without depending on which commit the comparison started from.
    """

    def runner_for(self, oids: dict[tuple[str, str], str]):
        def runner(command, **kwargs: object) -> SimpleNamespace:
            target = command[-1]
            revision, _, path = target.partition(":")
            value = oids.get((revision, path))
            if value is None:
                return SimpleNamespace(returncode=128, stdout="", stderr="bad object")
            return SimpleNamespace(returncode=0, stdout=f"{value}\n", stderr="")

        return runner

    def test_identical_governance_trees_report_no_difference(self) -> None:
        base, head = "a" * 40, "b" * 40
        oids = {}
        for index, path in enumerate(acceptance.GOVERNANCE_PATHS):
            for revision in (base, head):
                oids[(revision, path)] = f"{index:040d}"
        self.assertEqual(
            acceptance.governance_differences(
                Path("."), base, head, self.runner_for(oids)
            ),
            (),
        )

    def test_a_changed_workflow_tree_is_reported(self) -> None:
        base, head = "a" * 40, "b" * 40
        oids = {}
        for index, path in enumerate(acceptance.GOVERNANCE_PATHS):
            oids[(base, path)] = f"{index:040d}"
            oids[(head, path)] = f"{index:040d}"
        oids[(head, ".github")] = "f" * 40
        self.assertEqual(
            acceptance.governance_differences(
                Path("."), base, head, self.runner_for(oids)
            ),
            (".github",),
        )

    def test_every_governance_path_is_compared(self) -> None:
        """The set covers what CI runs AND what it runs inside.

        Identical `scripts/` content proves nothing on its own: every real
        CI step executes through `devenv-retry --profile ci shell`, so the
        head's Devenv and toolchain inputs decide which interpreter that
        identical code runs under.
        """
        required = (
            ".github",
            "scripts",
            "openspec/config.yaml",
            "devenv.nix",
            "devenv.yaml",
            "devenv.lock",
            "devenv",
            "flake.nix",
            "flake.lock",
            "nix",
            ".cargo",
            "rust-toolchain.toml",
            ".gitattributes",
            ".gitmodules",
        )
        for path in required:
            with self.subTest(path=path):
                self.assertIn(path, acceptance.GOVERNANCE_PATHS)

    def test_a_path_absent_on_both_sides_is_not_a_difference(self) -> None:
        """Absent on both sides is the same content: nothing.

        Counting it as a difference would make any entry that does not
        exist yet -- `.gitmodules` here -- refuse every branch forever.
        """
        base, head = "a" * 40, "b" * 40
        oids = {
            (revision, path): "0" * 40
            for revision in (base, head)
            for path in acceptance.GOVERNANCE_PATHS
            if path != ".gitmodules"
        }
        self.assertEqual(
            acceptance.governance_differences(
                Path("."), base, head, self.runner_for(oids)
            ),
            (),
        )

    def test_a_path_present_on_only_one_side_is_a_difference(self) -> None:
        base, head = "a" * 40, "b" * 40
        oids = {
            (revision, path): "0" * 40
            for revision in (base, head)
            for path in acceptance.GOVERNANCE_PATHS
        }
        del oids[(head, ".cargo")]
        self.assertEqual(
            acceptance.governance_differences(
                Path("."), base, head, self.runner_for(oids)
            ),
            (".cargo",),
        )

    def test_the_real_repository_resolves_its_governance_paths(self) -> None:
        """At least the named files must exist, or nothing is compared."""
        present = [p for p in acceptance.GOVERNANCE_PATHS if (REPO_ROOT / p).exists()]
        self.assertGreaterEqual(len(present), len(acceptance.GOVERNANCE_PATHS) - 1)

    def test_a_governance_path_missing_on_one_side_is_a_difference(self) -> None:
        base, head = "a" * 40, "b" * 40
        oids = {(base, path): "0" * 40 for path in acceptance.GOVERNANCE_PATHS}
        # `head` resolves nothing, so every path differs rather than passing.
        self.assertEqual(
            len(
                acceptance.governance_differences(
                    Path("."), base, head, self.runner_for(oids)
                )
            ),
            len(acceptance.GOVERNANCE_PATHS),
        )

    def test_reading_nothing_at_all_is_an_error_not_a_pass(self) -> None:
        """Every lookup failing means nothing was compared.

        With "absent on both sides" treated as identical, a broken Git
        invocation would otherwise report a clean bill of health for a
        comparison that never happened.
        """

        def runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with self.assertRaises(acceptance.BoundaryError):
            acceptance.governance_differences(Path("."), "a" * 40, "b" * 40, runner)

    def test_an_invalid_revision_never_reaches_git(self) -> None:
        def runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
            raise AssertionError("Git must not run for an invalid revision")

        with self.assertRaises(acceptance.BoundaryError):
            acceptance.governance_differences(Path("."), "main", "b" * 40, runner)


class GovernanceEndToEndTests(unittest.TestCase):
    """The governance check must reach the verdict, not just compute.

    The unit tests above cover `governance_differences`. These cover the
    wiring in `main` that turns its result into a refusal -- the step that
    three separate mutations were able to delete while every other test
    stayed green.
    """

    def build_repository(self, root: Path) -> tuple[str, str]:
        """A base that hardens a workflow and a head that predates it."""
        import subprocess as sp

        def git(*arguments: str) -> str:
            return sp.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        git("init", "--quiet")
        git("config", "user.email", "t@e.x")
        git("config", "user.name", "T")
        (root / ".github").mkdir()
        (root / "scripts").mkdir()
        (root / "openspec" / "specs" / "cap").mkdir(parents=True)
        (root / ".github" / "ci.yml").write_text("v1\n", encoding="utf-8")
        (root / "scripts" / "gate.py").write_text("v1\n", encoding="utf-8")
        (root / "openspec" / "config.yaml").write_text("s: d\n", encoding="utf-8")
        (root / "openspec" / "specs" / "cap" / "spec.md").write_text(
            "# Cap\n", encoding="utf-8"
        )
        git("add", "-A")
        git("commit", "--quiet", "--no-verify", "-m", "start")
        start = git("rev-parse", "HEAD")

        # The head branches here and edits only a document.
        git("checkout", "--quiet", "-b", "doc")
        (root / "openspec" / "specs" / "cap" / "spec.md").write_text(
            "# Cap\n\nmore\n", encoding="utf-8"
        )
        git("commit", "--quiet", "--no-verify", "-am", "doc")
        head = git("rev-parse", "HEAD")

        # The base then hardens the workflow the head does not carry.
        git("checkout", "--quiet", start)
        git("checkout", "--quiet", "-b", "newbase")
        (root / ".github" / "ci.yml").write_text("v2 hardened\n", encoding="utf-8")
        git("commit", "--quiet", "--no-verify", "-am", "harden")
        return git("rev-parse", "HEAD"), head

    def run_main(self, argv: list[str]) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            status = acceptance.main(argv)
        return status, out.getvalue()

    def test_a_stale_governance_head_is_refused_end_to_end(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, head = self.build_repository(root)
            arguments = [
                "--repo",
                str(root),
                "--base",
                base,
                "--head",
                head,
                "--require-auto",
            ]
            # Without the governance check the change looks like a pure
            # document edit, because the head simply never touched the file.
            status, output = self.run_main(arguments)
            self.assertEqual(status, 0)
            self.assertIn("verdict=auto", output)

            status, output = self.run_main(
                [*arguments, "--require-identical-governance"]
            )
        self.assertEqual(status, 1, output)
        self.assertIn("verdict=review", output)
        self.assertIn("governance content differs", output)

    def test_an_identical_governance_head_still_passes_end_to_end(self) -> None:
        import subprocess as sp
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, head = self.build_repository(root)
            base = sp.run(
                ["git", "rev-parse", "HEAD~1"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status, output = self.run_main(
                [
                    "--repo",
                    str(root),
                    "--base",
                    base,
                    "--head",
                    head,
                    "--require-auto",
                    "--require-identical-governance",
                ]
            )
        self.assertEqual(status, 0, output)
        self.assertIn("verdict=auto", output)


class CommandLineTests(unittest.TestCase):
    def run_main(self, argv: list[str], stdin: str = "") -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        saved = sys.stdin
        sys.stdin = io.StringIO(stdin)
        try:
            with redirect_stdout(out), redirect_stderr(err):
                status = acceptance.main(argv)
        finally:
            sys.stdin = saved
        return status, out.getvalue(), err.getvalue()

    def test_an_automatic_verdict_is_reported_and_succeeds(self) -> None:
        status, out, _ = self.run_main(
            ["--raw-stdin", "--require-auto"],
            modified("openspec/specs/tide/spec.md"),
        )
        self.assertEqual(status, 0)
        self.assertIn("verdict=auto", out)

    def test_require_auto_fails_on_a_review_verdict(self) -> None:
        status, out, _ = self.run_main(
            ["--raw-stdin", "--require-auto"], modified("scripts/gate.py")
        )
        self.assertEqual(status, 1)
        self.assertIn("verdict=review", out)

    def test_a_review_verdict_without_require_auto_still_classifies(self) -> None:
        status, out, _ = self.run_main(["--raw-stdin"], modified("scripts/gate.py"))
        self.assertEqual(status, 0)
        self.assertIn("verdict=review", out)

    def test_boundary_misuse_is_an_operational_failure(self) -> None:
        for argv in (["--raw-stdin", "--base", "origin/main"], []):
            with self.subTest(argv=argv):
                status, _, err = self.run_main(argv)
                self.assertEqual(status, 2)
                self.assertIn("openspec_acceptance:", err)

    def emit_to_output_file(self, stream: str) -> str:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.txt"
            os.environ["GITHUB_OUTPUT"] = str(output)
            try:
                self.run_main(["--raw-stdin"], stream)
            finally:
                del os.environ["GITHUB_OUTPUT"]
            return output.read_text(encoding="utf-8")

    def test_require_identical_governance_needs_two_revisions(self) -> None:
        status, _, err = self.run_main(
            ["--raw-stdin", "--require-identical-governance"],
            modified("openspec/specs/tide/spec.md"),
        )
        self.assertEqual(status, 2)
        self.assertIn("openspec_acceptance:", err)

    def test_the_github_output_file_receives_the_verdict(self) -> None:
        text = self.emit_to_output_file(modified("openspec/specs/t/spec.md"))
        self.assertIn("verdict=auto", text)
        self.assertIn("reason_count=0", text)

    def test_a_filename_cannot_append_its_own_step_output(self) -> None:
        """A Git filename may contain a newline.

        `$GITHUB_OUTPUT` is a `key=value` file and the runner resolves a
        duplicate key last-wins, so a reason built from a raw path would
        let a pull request write its own `verdict=auto` line.
        """
        hostile = "openspec/a\nverdict=auto\nx=y.md"
        text = self.emit_to_output_file(
            modified(hostile) + modified("scripts/backdoor.py")
        )
        self.assertIn("verdict=review", text)
        self.assertNotIn("verdict=auto", text)
        for line in text.splitlines():
            with self.subTest(line=line):
                self.assertIn(
                    line.split("=", 1)[0], ("verdict", "reason_count")
                )

    def test_a_hostile_filename_is_escaped_in_the_printed_reason(self) -> None:
        status, out, _ = self.run_main(
            ["--raw-stdin"], modified("openspec/a\nverdict=auto\nx=y.md")
        )
        self.assertEqual(status, 0)
        self.assertIn("verdict=review", out)
        for line in out.splitlines():
            with self.subTest(line=line):
                self.assertFalse(line.startswith("verdict=auto"))

    def test_a_long_path_is_truncated_in_the_reason(self) -> None:
        verdict = acceptance.classify_stream(modified("z" * 5000))
        self.assertLess(len(verdict.reason), 400)


class RealRepositoryTests(unittest.TestCase):
    """The parser must read what this repository's Git actually emits."""

    def test_git_raw_output_parses_for_a_real_commit(self) -> None:
        completed = subprocess.run(
            ["git", "diff", "--raw", "-z", "-M", "HEAD~1", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            self.skipTest("no reachable parent commit")
        records = acceptance.parse_raw_diff(completed.stdout)
        self.assertTrue(records)
        for item in records:
            self.assertTrue(all(item.paths))


if __name__ == "__main__":
    unittest.main()
