"""Unit tests for `check_openspec.py` (spec-first: written before the checker).

Run via: `python3 scripts/test_check_openspec.py` from repo root.

Owning requirements: `openspec/changes/adopt-openspec-governance/specs/`
(`spec-driven-change-governance` and `openspec-ci-governance`). What is
locked here:

  (a) argument parsing: the shared action's fixed
      `--self-test --merge-bound --base <base>` invocation, the local
      `--pre-archive` mode, and rejection of unknown/incoherent modes;
  (b) injected executable selection (`GIT_BIN` / `OPENSPEC_BIN`) with
      fail-closed behavior when neither injection nor PATH provides one;
  (c) exact OpenSpec 1.6.0 verification and unresolvable-history errors;
  (d) bounded GitHub event loading and the exact PR citation format
      `OpenSpec-Change: <change-id>` at line start;
  (e) path classification, one-record branch scope, hidden-path and
      inherited-lifecycle rejection;
  (f) exact maintenance-exemption manifests (never `spec/**`);
  (g) planning-before-implementation ordering over an ordered commit list;
  (h) built-in delta-spec schema enforcement (requirement blocks, scenario
      parity proxy: >= 2 scenarios per requirement);
  (i) task completion, archive naming, and replayed delta-to-baseline
      synchronization;
  (j) the self-test control inventory covers the twelve planted-negative
      controls named in the design, and `self_test()` reports no failures.

Every positive test has a planted-negative sibling (repo Negative Test
Parity rule).
"""

import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "check_openspec", here / "check_openspec.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


co = _load_module()

CHANGE_ID = "adopt-openspec-governance"
MARKER = f"openspec/changes/{CHANGE_ID}/.openspec.yaml"
PROPOSAL = f"openspec/changes/{CHANGE_ID}/proposal.md"
DELTA = f"openspec/changes/{CHANGE_ID}/specs/some-capability/spec.md"


class FakeRun:
    """Injectable subprocess.run replacement returning canned results."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        result = self.results.pop(0)
        return result


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestArguments(unittest.TestCase):
    def test_action_invocation_parses(self):
        ns = co.parse_arguments(
            ["--self-test", "--merge-bound", "--base", "origin/main"]
        )
        self.assertTrue(ns.self_test)
        self.assertTrue(ns.merge_bound)
        self.assertEqual(ns.base, "origin/main")

    def test_pre_archive_mode_parses(self):
        ns = co.parse_arguments(["--pre-archive", "--base", "origin/main"])
        self.assertTrue(ns.pre_archive)

    def test_merge_bound_requires_base(self):
        with self.assertRaises(SystemExit):
            co.parse_arguments(["--merge-bound"])

    def test_merge_bound_and_pre_archive_conflict(self):
        with self.assertRaises(SystemExit):
            co.parse_arguments(
                ["--merge-bound", "--pre-archive", "--base", "origin/main"]
            )

    def test_no_mode_rejected(self):
        with self.assertRaises(SystemExit):
            co.parse_arguments([])

    def test_unknown_argument_rejected(self):
        with self.assertRaises(SystemExit):
            co.parse_arguments(["--self-test", "--frobnicate"])


class TestExecutableSelection(unittest.TestCase):
    def _fake_binary(self, tmp, name):
        path = Path(tmp) / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_git_bin_injection_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = self._fake_binary(tmp, "fakegit")
            resolved = co.resolve_git({"GIT_BIN": str(fake)})
            self.assertEqual(str(fake), str(resolved))

    def test_missing_injected_git_fails_closed(self):
        errors = []
        try:
            co.resolve_git({"GIT_BIN": "/nonexistent/git-binary"})
        except co.CheckError:
            errors.append("raised")
        self.assertEqual(["raised"], errors)

    def test_openspec_bin_injection_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = self._fake_binary(tmp, "fakeopenspec")
            resolved = co.resolve_openspec({"OPENSPEC_BIN": str(fake)})
            self.assertEqual(str(fake), str(resolved))

    def test_missing_openspec_fails_closed(self):
        with self.assertRaises(co.CheckError):
            co.resolve_openspec({"OPENSPEC_BIN": "/nonexistent/openspec", "PATH": ""})


class TestOpenSpecVersion(unittest.TestCase):
    def test_exact_version_accepted(self):
        run = FakeRun([FakeCompleted(0, stdout="1.6.0\n")])
        self.assertEqual([], co.verify_openspec("/bin/openspec", runner=run))

    def test_wrong_version_rejected(self):
        run = FakeRun([FakeCompleted(0, stdout="1.5.0\n")])
        self.assertNotEqual([], co.verify_openspec("/bin/openspec", runner=run))

    def test_nonzero_exit_rejected(self):
        run = FakeRun([FakeCompleted(1, stdout="1.6.0\n")])
        self.assertNotEqual([], co.verify_openspec("/bin/openspec", runner=run))


class TestMergeBase(unittest.TestCase):
    def test_resolvable_base_returns_sha(self):
        sha = "a" * 40
        run = FakeRun([FakeCompleted(0, stdout=sha + "\n")])
        self.assertEqual(sha, co.resolve_merge_base("git", "origin/main", runner=run))

    def test_unresolvable_history_fails_closed(self):
        run = FakeRun([FakeCompleted(128, stdout="", stderr="fatal: no merge base")])
        with self.assertRaises(co.CheckError):
            co.resolve_merge_base("git", "origin/main", runner=run)


class TestEventLoading(unittest.TestCase):
    def test_valid_event_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "event.json"
            path.write_text(json.dumps({"action": "opened"}), encoding="utf-8")
            self.assertEqual({"action": "opened"}, co.load_event(path))

    def test_oversized_event_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "event.json"
            path.write_text(
                '{"pad": "' + "x" * (co.MAX_EVENT_BYTES + 16) + '"}',
                encoding="utf-8",
            )
            with self.assertRaises(co.CheckError):
                co.load_event(path)

    def test_symlinked_event_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "real.json"
            target.write_text("{}", encoding="utf-8")
            link = Path(tmp) / "event.json"
            os.symlink(target, link)
            with self.assertRaises(co.CheckError):
                co.load_event(link)

    def test_non_object_event_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "event.json"
            path.write_text("[1, 2]", encoding="utf-8")
            with self.assertRaises(co.CheckError):
                co.load_event(path)


class TestCitation(unittest.TestCase):
    def test_exact_citation_extracted(self):
        body = f"Implements the plan.\nOpenSpec-Change: {CHANGE_ID}\nMore text."
        self.assertEqual([CHANGE_ID], co.extract_citations(body))

    def test_missing_citation_is_empty(self):
        self.assertEqual([], co.extract_citations("no citation here"))

    def test_none_body_is_empty(self):
        self.assertEqual([], co.extract_citations(None))

    def test_body_claims_are_not_citations(self):
        body = "spec-exempt: true\nExempt: yes\nlabels: docs-only"
        self.assertEqual([], co.extract_citations(body))

    def test_indented_citation_not_matched(self):
        body = f"  OpenSpec-Change: {CHANGE_ID}"
        self.assertEqual([], co.extract_citations(body))

    def test_citation_match_accepted(self):
        self.assertEqual([], co.check_citation([CHANGE_ID], CHANGE_ID))

    def test_citation_mismatch_rejected(self):
        self.assertNotEqual([], co.check_citation(["other-change"], CHANGE_ID))

    def test_absent_citation_rejected(self):
        self.assertNotEqual([], co.check_citation([], CHANGE_ID))

    def test_multiple_distinct_citations_rejected(self):
        self.assertNotEqual(
            [], co.check_citation([CHANGE_ID, "other-change"], CHANGE_ID)
        )


class TestClassification(unittest.TestCase):
    def test_lifecycle_paths_classified(self):
        cls = co.classify_paths([MARKER, PROPOSAL, DELTA])
        self.assertEqual(frozenset([CHANGE_ID]), cls.active_lifecycles)
        self.assertEqual(frozenset(), cls.production)

    def test_archive_paths_classified(self):
        cls = co.classify_paths(
            [f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/proposal.md"]
        )
        self.assertEqual(frozenset([f"2026-07-24-{CHANGE_ID}"]), cls.archives)
        self.assertEqual(frozenset(), cls.active_lifecycles)

    def test_exemption_paths_classified(self):
        path = "openspec/exemptions/2026-07-24-fix-readme-typo.toml"
        cls = co.classify_paths([path])
        self.assertEqual(frozenset([path]), cls.exemptions)

    def test_spec_paths_are_not_production(self):
        cls = co.classify_paths(["spec/01-nomenclature.md", "crates/foo/src/lib.rs"])
        self.assertEqual(frozenset(["spec/01-nomenclature.md"]), cls.spec)
        self.assertEqual(frozenset(["crates/foo/src/lib.rs"]), cls.production)

    def test_planning_neutral_paths_are_not_production(self):
        cls = co.classify_paths([".gitignore"])
        self.assertEqual(frozenset(), cls.production)
        self.assertEqual(frozenset(), cls.spec)


class TestBranchScope(unittest.TestCase):
    def _cls(self, paths):
        return co.classify_paths(paths)

    def test_single_new_lifecycle_accepted(self):
        cls = self._cls([MARKER, PROPOSAL, "crates/foo/src/lib.rs"])
        self.assertEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="pre-archive"),
        )

    def test_two_lifecycles_rejected(self):
        cls = self._cls(
            [MARKER, "openspec/changes/other-change/.openspec.yaml"]
        )
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="pre-archive"),
        )

    def test_lifecycle_plus_exemption_rejected(self):
        cls = self._cls(
            [MARKER, "openspec/exemptions/2026-07-24-fix-typo.toml"]
        )
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="pre-archive"),
        )

    def test_inherited_lifecycle_mutation_rejected(self):
        cls = self._cls([f"openspec/changes/{CHANGE_ID}/tasks.md"])
        self.assertNotEqual(
            [],
            co.check_branch_scope(
                cls, inherited_ids=frozenset([CHANGE_ID]), mode="pre-archive"
            ),
        )

    def test_hidden_governance_path_rejected(self):
        cls = self._cls([f"openspec/changes/{CHANGE_ID}/.hidden/evidence.md"])
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="pre-archive"),
        )

    def test_marker_dotfile_is_not_hidden_evidence(self):
        cls = self._cls([MARKER, PROPOSAL])
        self.assertEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="pre-archive"),
        )

    def test_merge_bound_rejects_active_lifecycle(self):
        cls = self._cls([MARKER, PROPOSAL])
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="merge-bound"),
        )

    def test_merge_bound_accepts_single_archive(self):
        cls = self._cls(
            [
                f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/proposal.md",
                "crates/foo/src/lib.rs",
            ]
        )
        self.assertEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="merge-bound"),
        )

    def test_production_without_any_record_rejected(self):
        cls = self._cls(["crates/foo/src/lib.rs"])
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="merge-bound"),
        )


class TestExemption(unittest.TestCase):
    NAME = "2026-07-24-fix-readme-typo.toml"

    def _manifest(self, kind='"maintenance"', reason='"fix typo"', paths=None):
        if paths is None:
            paths = ["README.md"]
        rendered = ", ".join(f'"{p}"' for p in paths)
        return f"kind = {kind}\nreason = {reason}\npaths = [{rendered}]\n"

    def test_exact_manifest_accepted(self):
        errors = co.check_exemption_manifest(
            self.NAME, self._manifest(), frozenset(["README.md"])
        )
        self.assertEqual([], errors)

    def test_wrong_kind_rejected(self):
        errors = co.check_exemption_manifest(
            self.NAME, self._manifest(kind='"feature"'), frozenset(["README.md"])
        )
        self.assertNotEqual([], errors)

    def test_empty_reason_rejected(self):
        errors = co.check_exemption_manifest(
            self.NAME, self._manifest(reason='""'), frozenset(["README.md"])
        )
        self.assertNotEqual([], errors)

    def test_path_set_mismatch_rejected(self):
        errors = co.check_exemption_manifest(
            self.NAME,
            self._manifest(paths=["README.md"]),
            frozenset(["README.md", "docs/extra.md"]),
        )
        self.assertNotEqual([], errors)

    def test_spec_path_rejected(self):
        errors = co.check_exemption_manifest(
            self.NAME,
            self._manifest(paths=["spec/01-nomenclature.md"]),
            frozenset(["spec/01-nomenclature.md"]),
        )
        self.assertNotEqual([], errors)

    def test_governance_path_rejected(self):
        errors = co.check_exemption_manifest(
            self.NAME,
            self._manifest(paths=["openspec/config.yaml"]),
            frozenset(["openspec/config.yaml"]),
        )
        self.assertNotEqual([], errors)

    def test_invalid_filename_date_rejected(self):
        errors = co.check_exemption_manifest(
            "2026-13-40-fix-readme-typo.toml",
            self._manifest(),
            frozenset(["README.md"]),
        )
        self.assertNotEqual([], errors)

    def test_malformed_toml_rejected(self):
        errors = co.check_exemption_manifest(
            self.NAME, "kind = [unclosed", frozenset(["README.md"])
        )
        self.assertNotEqual([], errors)

    def test_unexpected_key_rejected(self):
        text = self._manifest() + 'extra = "value"\n'
        errors = co.check_exemption_manifest(
            self.NAME, text, frozenset(["README.md"])
        )
        self.assertNotEqual([], errors)


class TestPlanningOrder(unittest.TestCase):
    def test_planning_ancestor_accepted(self):
        commits = [
            ("a" * 40, frozenset([MARKER, PROPOSAL, DELTA, ".gitignore"])),
            ("b" * 40, frozenset(["crates/foo/src/lib.rs"])),
        ]
        self.assertEqual([], co.check_planning_order(commits, CHANGE_ID))

    def test_collapsed_commit_rejected(self):
        commits = [
            ("a" * 40, frozenset([MARKER, PROPOSAL, DELTA, "crates/foo/src/lib.rs"])),
        ]
        self.assertNotEqual([], co.check_planning_order(commits, CHANGE_ID))

    def test_implementation_before_planning_rejected(self):
        commits = [
            ("a" * 40, frozenset(["crates/foo/src/lib.rs"])),
            ("b" * 40, frozenset([MARKER, PROPOSAL, DELTA])),
        ]
        self.assertNotEqual([], co.check_planning_order(commits, CHANGE_ID))

    def test_missing_marker_rejected(self):
        commits = [
            ("a" * 40, frozenset([PROPOSAL, DELTA])),
            ("b" * 40, frozenset(["crates/foo/src/lib.rs"])),
        ]
        self.assertNotEqual([], co.check_planning_order(commits, CHANGE_ID))

    def test_planning_only_branch_accepted(self):
        commits = [("a" * 40, frozenset([MARKER, PROPOSAL, DELTA]))]
        self.assertEqual([], co.check_planning_order(commits, CHANGE_ID))


class TestDeltaShape(unittest.TestCase):
    GOOD = (
        "## ADDED Requirements\n\n"
        "### Requirement: Example holds\n"
        "The system SHALL hold.\n\n"
        "#### Scenario: It works\n"
        "- **WHEN** input is valid\n"
        "- **THEN** it SHALL pass\n\n"
        "#### Scenario: It fails\n"
        "- **WHEN** input is invalid\n"
        "- **THEN** it SHALL fail\n"
    )

    def test_well_formed_delta_accepted(self):
        self.assertEqual([], co.check_delta_shape(self.GOOD))

    def test_missing_requirement_rejected(self):
        self.assertNotEqual([], co.check_delta_shape("## ADDED Requirements\n"))

    def test_single_scenario_rejected(self):
        text = self.GOOD.split("#### Scenario: It fails")[0]
        self.assertNotEqual([], co.check_delta_shape(text))

    def test_unknown_operation_rejected(self):
        text = self.GOOD.replace("## ADDED Requirements", "## INVENTED Requirements")
        self.assertNotEqual([], co.check_delta_shape(text))


class TestTasks(unittest.TestCase):
    def test_complete_tasks_accepted(self):
        text = "## 1. Work\n\n- [x] 1.1 Do the thing.\n- [x] 1.2 Verify it.\n"
        self.assertEqual([], co.check_tasks(text))

    def test_unchecked_task_rejected(self):
        text = "- [x] 1.1 Do the thing.\n- [ ] 1.2 Verify it.\n"
        self.assertNotEqual([], co.check_tasks(text))

    def test_undescribed_task_rejected(self):
        text = "- [x] \n"
        self.assertNotEqual([], co.check_tasks(text))

    def test_empty_task_list_rejected(self):
        self.assertNotEqual([], co.check_tasks("no checkboxes at all\n"))


class TestArchiveName(unittest.TestCase):
    def test_valid_archive_name_accepted(self):
        self.assertEqual([], co.check_archive_name(f"2026-07-24-{CHANGE_ID}"))

    def test_invalid_date_rejected(self):
        self.assertNotEqual([], co.check_archive_name(f"2026-13-40-{CHANGE_ID}"))

    def test_missing_date_rejected(self):
        self.assertNotEqual([], co.check_archive_name(CHANGE_ID))

    def test_uppercase_id_rejected(self):
        self.assertNotEqual([], co.check_archive_name("2026-07-24-Adopt-Change"))


class TestSynchronization(unittest.TestCase):
    BLOCK = "### Requirement: Example holds\nBody.\n\n#### Scenario: A\n- x\n"

    def test_replayed_delta_matches_head(self):
        base = {"cap": {}}
        head = {"cap": {"Example holds": self.BLOCK}}
        deltas = {"cap": {"added": {"Example holds": self.BLOCK}}}
        self.assertEqual([], co.check_synchronization(base, head, deltas))

    def test_unsynchronized_baseline_rejected(self):
        base = {"cap": {}}
        head = {"cap": {"Example holds": self.BLOCK + "drifted\n"}}
        deltas = {"cap": {"added": {"Example holds": self.BLOCK}}}
        self.assertNotEqual([], co.check_synchronization(base, head, deltas))

    def test_added_requirement_already_present_rejected(self):
        base = {"cap": {"Example holds": self.BLOCK}}
        head = {"cap": {"Example holds": self.BLOCK}}
        deltas = {"cap": {"added": {"Example holds": self.BLOCK}}}
        self.assertNotEqual([], co.check_synchronization(base, head, deltas))

    def test_removed_requirement_missing_from_base_rejected(self):
        base = {"cap": {}}
        head = {"cap": {}}
        deltas = {"cap": {"removed": {"Example holds": self.BLOCK}}}
        self.assertNotEqual([], co.check_synchronization(base, head, deltas))


class TestSelfTest(unittest.TestCase):
    REQUIRED_CONTROLS = {
        "malformed-spec",
        "missing-negative-scenario",
        "artifact-drift",
        "planning-order-collapse",
        "citation-mismatch",
        "branch-scope-ambiguity",
        "invalid-exemption",
        "symlink-rejection",
        "unchecked-task",
        "active-merge-state",
        "malformed-archive",
        "unsynchronized-delta",
    }

    def test_control_inventory_is_complete(self):
        self.assertEqual(self.REQUIRED_CONTROLS, set(co.SELF_TEST_CONTROLS))

    def test_self_test_passes(self):
        self.assertEqual([], co.self_test())


if __name__ == "__main__":
    unittest.main()
