"""Unit tests for `check_openspec.py` (spec-first: written before the checker).

Run via: `python3 scripts/test_check_openspec.py` from repo root.

Owning requirements: `openspec/specs/` (`spec-driven-change-governance`
and `openspec-ci-governance`, synchronized from the archived
`adopt-openspec-governance` lifecycle). What is
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
  (j) the self-test control inventory covers every planted-negative
      control named in `SELF_TEST_CONTROLS`, and `self_test()` reports
      no failures;
  (k) red-team and review regression coverage: direct baseline
      `openspec/specs/` mutation without a same-diff archive, archived
      deltas missing requirement blocks or scenario parity at merge time,
      merge-commit content unattributable to any branch commit (evil
      merge), unrecognized `openspec/` paths, mutation of an archived
      lifecycle inherited from the comparison base (including the
      marker-dance laundering repro), incomplete archives, and
      `@`-prefixed paths forging commit boundaries all fail closed, while
      the fully synchronized archived control fixture stays green
      end-to-end. Fourth-pass regressions: the push-event lane skips only
      commit-ordering/attribution (squash merges collapse branch
      history) while every diff-shaped control still fails, double-space
      and ordered-list checkboxes cannot hide from task policing, active
      lifecycle identifiers must be lowercase kebab-case, and the
      repository root is git-resolved so subdirectory runs behave
      identically.

Every positive test has a planted-negative sibling (repo Negative Test
Parity rule).
"""

import contextlib
import importlib.util
import io
import json
import os
import stat
import subprocess
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

    def test_baseline_spec_paths_classified(self):
        path = "openspec/specs/some-capability/spec.md"
        cls = co.classify_paths([path])
        self.assertEqual(frozenset([path]), cls.baseline_specs)
        self.assertEqual(frozenset(), cls.unrecognized)
        self.assertEqual(frozenset(), cls.production)

    def test_non_spec_file_under_baseline_tree_unrecognized(self):
        path = "openspec/specs/some-capability/notes.md"
        cls = co.classify_paths([path])
        self.assertEqual(frozenset(), cls.baseline_specs)
        self.assertEqual(frozenset([path]), cls.unrecognized)

    def test_config_yaml_recognized(self):
        cls = co.classify_paths(["openspec/config.yaml"])
        self.assertEqual(frozenset(), cls.unrecognized)
        self.assertEqual(frozenset(["openspec/config.yaml"]), cls.config)

    def test_hidden_archive_path_flagged(self):
        path = f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/.hidden/e.md"
        cls = co.classify_paths([path])
        self.assertEqual(frozenset([path]), cls.hidden)

    def test_archive_marker_dotfile_is_not_hidden(self):
        path = f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/.openspec.yaml"
        cls = co.classify_paths([path])
        self.assertEqual(frozenset(), cls.hidden)

    def test_nested_marker_dotfile_is_hidden(self):
        # The marker is evidence only at the record root; a .openspec.yaml
        # buried a level deeper is a hidden path, not evidence.
        path = f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/specs/.openspec.yaml"
        cls = co.classify_paths([path])
        self.assertEqual(frozenset([path]), cls.hidden)

    def test_unknown_governance_path_unrecognized(self):
        cls = co.classify_paths(["openspec/notes/hack.txt"])
        self.assertEqual(frozenset(["openspec/notes/hack.txt"]), cls.unrecognized)

    def test_nested_exemption_path_unrecognized(self):
        path = "openspec/exemptions/nested/2026-07-24-fix.toml"
        cls = co.classify_paths([path])
        self.assertEqual(frozenset(), cls.exemptions)
        self.assertEqual(frozenset([path]), cls.unrecognized)


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

    def test_pre_archive_locally_archived_lifecycle_not_misreported(self):
        # A branch that archived locally before running pre-archive still
        # carries a record; the diagnostic must not claim it has none.
        cls = self._cls(
            [
                f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/proposal.md",
                "crates/foo/src/lib.rs",
            ]
        )
        errors = co.check_branch_scope(
            cls, inherited_ids=frozenset(), mode="pre-archive"
        )
        self.assertNotIn(
            "governed paths changed without a lifecycle or maintenance exemption",
            errors,
        )

    def test_lifecycle_plus_exemption_rejected(self):
        cls = self._cls(
            [MARKER, "openspec/exemptions/2026-07-24-fix-typo.toml"]
        )
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="pre-archive"),
        )

    def test_archived_lifecycle_plus_exemption_rejected_at_merge_bound(self):
        cls = self._cls(
            [
                f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/proposal.md",
                "openspec/exemptions/2026-07-24-fix-typo.toml",
            ]
        )
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="merge-bound"),
        )

    def test_inherited_lifecycle_mutation_rejected(self):
        cls = self._cls([f"openspec/changes/{CHANGE_ID}/tasks.md"])
        self.assertNotEqual(
            [],
            co.check_branch_scope(
                cls, inherited_ids=frozenset([CHANGE_ID]), mode="pre-archive"
            ),
        )

    def test_active_lifecycle_plus_archive_rejected_at_pre_archive(self):
        # One record per branch: an active lifecycle alongside an archived
        # one is two records, and must be caught at pre-archive (merge-bound
        # already rejects the unarchived active lifecycle separately).
        cls = self._cls(
            [
                MARKER,
                "openspec/changes/archive/2026-07-24-other-change/proposal.md",
            ]
        )
        errors = co.check_branch_scope(
            cls, inherited_ids=frozenset(), mode="pre-archive"
        )
        self.assertTrue(
            any("active lifecycle with an archived lifecycle" in e for e in errors)
        )

    def test_archival_endpoint_diff_is_not_a_mix(self):
        # Positive sibling: a normal create-then-archive nets to archive-only
        # in the endpoint diff (no active record), so it is not flagged.
        cls = self._cls(
            [
                f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/proposal.md",
                f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/.openspec.yaml",
            ]
        )
        errors = co.check_branch_scope(
            cls, inherited_ids=frozenset(), mode="merge-bound"
        )
        self.assertNotIn(
            "branch mixes an active lifecycle with an archived lifecycle", errors
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

    def test_unrecognized_governance_path_rejected(self):
        cls = self._cls(["openspec/notes/hack.txt"])
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="merge-bound"),
        )

    def test_baseline_mutation_without_archive_rejected_at_merge_bound(self):
        cls = self._cls(["openspec/specs/some-capability/spec.md"])
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="merge-bound"),
        )

    def test_baseline_change_with_archive_accepted_at_merge_bound(self):
        cls = self._cls(
            [
                f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/proposal.md",
                "openspec/specs/some-capability/spec.md",
            ]
        )
        self.assertEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="merge-bound"),
        )

    def test_baseline_mutation_rejected_at_pre_archive(self):
        cls = self._cls(
            [MARKER, PROPOSAL, "openspec/specs/some-capability/spec.md"]
        )
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="pre-archive"),
        )

    def test_config_change_without_lifecycle_rejected_at_merge_bound(self):
        cls = self._cls(["openspec/config.yaml"])
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="merge-bound"),
        )

    def test_config_change_with_archive_accepted_at_merge_bound(self):
        cls = self._cls(
            [
                f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/proposal.md",
                "openspec/config.yaml",
            ]
        )
        self.assertEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="merge-bound"),
        )

    def test_config_change_without_lifecycle_rejected_at_pre_archive(self):
        cls = self._cls(["openspec/config.yaml"])
        self.assertNotEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="pre-archive"),
        )

    def test_config_change_with_new_lifecycle_accepted_at_pre_archive(self):
        cls = self._cls([MARKER, PROPOSAL, "openspec/config.yaml"])
        self.assertEqual(
            [],
            co.check_branch_scope(cls, inherited_ids=frozenset(), mode="pre-archive"),
        )

    def test_hidden_archive_path_rejected(self):
        cls = self._cls(
            [f"openspec/changes/archive/2026-07-24-{CHANGE_ID}/.hidden/e.md"]
        )
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


class TestCommitCoverage(unittest.TestCase):
    def test_attributed_changes_accepted(self):
        commits = [
            ("a" * 40, frozenset([MARKER, PROPOSAL, DELTA])),
            ("b" * 40, frozenset(["crates/foo/src/lib.rs"])),
        ]
        changed = frozenset([MARKER, "crates/foo/src/lib.rs"])
        self.assertEqual([], co.check_commit_coverage(changed, commits))

    def test_merge_hidden_change_rejected(self):
        commits = [("a" * 40, frozenset([MARKER, PROPOSAL, DELTA]))]
        changed = frozenset([MARKER, "crates/foo/src/lib.rs"])
        self.assertNotEqual([], co.check_commit_coverage(changed, commits))

    def test_reverted_path_outside_endpoint_diff_accepted(self):
        commits = [("a" * 40, frozenset([MARKER, "crates/foo/src/lib.rs"]))]
        changed = frozenset([MARKER])
        self.assertEqual([], co.check_commit_coverage(changed, commits))

    def test_empty_commit_list_with_changes_rejected(self):
        self.assertNotEqual(
            [], co.check_commit_coverage(frozenset(["crates/foo/src/lib.rs"]), [])
        )


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

    def test_renamed_operation_rejected_with_guidance(self):
        text = self.GOOD.replace("## ADDED Requirements", "## RENAMED Requirements")
        errors = co.check_delta_shape(text)
        self.assertNotEqual([], errors)
        self.assertIn("REMOVED plus ADDED", errors[0])

    def test_mixed_case_renamed_gets_guidance(self):
        text = self.GOOD.replace("## ADDED Requirements", "## Renamed Requirements")
        errors = co.check_delta_shape(text)
        self.assertNotEqual([], errors)
        self.assertIn("REMOVED plus ADDED", errors[0])

    def test_double_space_heading_rejected(self):
        # A near-miss heading (extra interior space) must not fold its
        # blocks into the preceding section; both paths must reject it.
        text = self.GOOD + (
            "\n## REMOVED  Requirements\n\n### Requirement: Gone\nReason.\n"
        )
        errors = co.check_delta_shape(text)
        self.assertNotEqual([], errors)
        self.assertIn("malformed delta operation section", errors[0])
        with self.assertRaises(co.CheckError):
            co.parse_delta(text)

    def test_preamble_requirement_block_rejected(self):
        # A requirement block before the first operation heading is invisible
        # to replay and cannot count as coverage.
        text = (
            "### Requirement: Ghost\nBody.\n\n"
            "#### Scenario: a\n- x\n\n#### Scenario: b\n- y\n\n" + self.GOOD
        )
        errors = co.check_delta_shape(text)
        self.assertNotEqual([], errors)
        self.assertTrue(any("precedes the first delta operation" in e for e in errors))
        with self.assertRaises(co.CheckError):
            co.parse_delta(text)

    def test_duplicate_requirement_name_rejected(self):
        second = self.GOOD.replace("## ADDED Requirements\n\n", "")
        errors = co.check_delta_shape(self.GOOD + "\n" + second)
        self.assertNotEqual([], errors)
        self.assertTrue(any("duplicate requirement block" in e for e in errors))

    def test_removed_section_needs_no_scenarios(self):
        # A REMOVED section names the requirement and its reason; it has no
        # changed behavior to demonstrate, so the two-scenario parity rule
        # does not apply.
        text = (
            "## REMOVED Requirements\n\n"
            "### Requirement: Legacy path\n"
            "**Reason**: superseded by the new flow.\n"
        )
        self.assertEqual([], co.check_delta_shape(text))

    def test_added_section_still_needs_two_scenarios(self):
        # Negative parity for the REMOVED relaxation: ADDED and MODIFIED
        # sections keep the two-scenario requirement.
        text = self.GOOD.split("#### Scenario: It fails")[0]
        self.assertNotEqual([], co.check_delta_shape(text))

    def test_fenced_worked_example_accepted(self):
        # A delta that documents the delta format inside a code fence must
        # not have the fenced headings parsed as live structure. The real
        # requirement keeps its two real scenarios and the delta is clean.
        text = (
            "## ADDED Requirements\n\n"
            "### Requirement: Authoring guide\n"
            "Authors write deltas like this:\n\n"
            "```\n"
            "## ADDED Requirements\n\n"
            "### Requirement: Example\n"
            "Body.\n"
            "```\n\n"
            "#### Scenario: It works\n- **WHEN** x\n- **THEN** y\n\n"
            "#### Scenario: It fails\n- **WHEN** a\n- **THEN** b\n"
        )
        self.assertEqual([], co.check_delta_shape(text))

    def test_fenced_scenarios_do_not_satisfy_parity(self):
        # Negative parity: '#### Scenario:' lines hidden inside a code fence
        # are not real behavioral scenarios and must not satisfy the
        # two-scenario rule.
        text = (
            "## ADDED Requirements\n\n"
            "### Requirement: Sneaky\n"
            "No real scenarios.\n\n"
            "```\n#### Scenario: fake positive\n"
            "#### Scenario: fake negative\n```\n"
        )
        errors = co.check_delta_shape(text)
        self.assertNotEqual([], errors)
        self.assertTrue(any("at least two scenarios" in e for e in errors))

    def test_fenced_h2_requirements_heading_not_flagged(self):
        # A level-2 heading ending in 'requirements' that lives inside a
        # fence is not a near-miss operation heading.
        text = self.GOOD + "\n```\n## Reserved requirements\ncode\n```\n"
        self.assertEqual([], co.check_delta_shape(text))


class TestParseDelta(unittest.TestCase):
    def test_known_sections_parsed(self):
        parsed = co.parse_delta(TestDeltaShape.GOOD)
        self.assertIn("Example holds", parsed["added"])

    def test_renamed_section_rejected(self):
        text = "## RENAMED Requirements\n\n### Requirement: R\nBody.\n"
        with self.assertRaises(co.CheckError):
            co.parse_delta(text)

    def test_lowercase_operation_rejected(self):
        # parse_delta must agree with check_delta_shape: only the exact
        # uppercase operation headings are part of the delta schema.
        text = TestDeltaShape.GOOD.replace("## ADDED", "## added")
        with self.assertRaises(co.CheckError):
            co.parse_delta(text)

    def test_removed_section_parsed(self):
        text = (
            "## REMOVED Requirements\n\n"
            "### Requirement: Legacy path\n**Reason**: superseded.\n"
        )
        self.assertIn("Legacy path", co.parse_delta(text)["removed"])

    def test_fenced_requirement_is_not_a_real_block(self):
        # A '### Requirement:' inside a code fence is body text of the
        # enclosing requirement, not a second block; replay must not invent
        # a phantom requirement from a worked example.
        text = (
            "## ADDED Requirements\n\n"
            "### Requirement: Guide\n"
            "Example:\n\n```\n### Requirement: Phantom\nBody.\n```\n\n"
            "#### Scenario: a\n- x\n\n#### Scenario: b\n- y\n"
        )
        parsed = co.parse_delta(text)
        self.assertEqual(["Guide"], list(parsed["added"]))

    def test_parse_requirement_blocks_rejects_duplicate(self):
        with self.assertRaises(co.CheckError):
            co.parse_requirement_blocks(
                "### Requirement: R\nFirst.\n\n### Requirement: R\nSecond.\n"
            )

    def test_parse_requirement_blocks_unique_accepted(self):
        blocks = co.parse_requirement_blocks(
            "### Requirement: A\nOne.\n\n### Requirement: B\nTwo.\n"
        )
        self.assertEqual(["A", "B"], list(blocks))


class TestBaselineDeltaCoverage(unittest.TestCase):
    def test_covered_baseline_accepted(self):
        self.assertEqual(
            [],
            co.check_baseline_delta_coverage(
                frozenset({"openspec/specs/cap/spec.md"}), frozenset({"cap"})
            ),
        )

    def test_uncovered_baseline_rejected(self):
        errors = co.check_baseline_delta_coverage(
            frozenset({"openspec/specs/other-cap/spec.md"}), frozenset({"cap"})
        )
        self.assertNotEqual([], errors)
        self.assertIn("without a corresponding archived delta", errors[0])

    def test_no_baseline_change_accepted(self):
        self.assertEqual(
            [], co.check_baseline_delta_coverage(frozenset(), frozenset())
        )


class TestArchiveDeltas(unittest.TestCase):
    def _archive(self, tmp, delta_text):
        archive_dir = Path(tmp) / "2026-07-24-x-change"
        if delta_text is None:
            archive_dir.mkdir(parents=True)
        else:
            spec_dir = archive_dir / "specs" / "cap"
            spec_dir.mkdir(parents=True)
            (spec_dir / "spec.md").write_text(delta_text, encoding="utf-8")
        return archive_dir

    def test_well_formed_archive_delta_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            errors, deltas = co.check_archive_deltas(
                self._archive(tmp, TestDeltaShape.GOOD), "2026-07-24-x-change"
            )
            self.assertEqual([], errors)
            self.assertIn("cap", deltas)

    def test_archive_without_delta_specs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            errors, deltas = co.check_archive_deltas(
                self._archive(tmp, None), "2026-07-24-x-change"
            )
            self.assertNotEqual([], errors)
            self.assertEqual({}, deltas)

    def test_single_scenario_archive_delta_rejected(self):
        text = TestDeltaShape.GOOD.split("#### Scenario: It fails")[0]
        with tempfile.TemporaryDirectory() as tmp:
            errors, _ = co.check_archive_deltas(
                self._archive(tmp, text), "2026-07-24-x-change"
            )
            self.assertNotEqual([], errors)


class TestTasks(unittest.TestCase):
    def test_complete_tasks_accepted(self):
        text = "## 1. Work\n\n- [x] 1.1 Do the thing.\n- [x] 1.2 Verify it.\n"
        self.assertEqual([], co.check_tasks(text))

    def test_unchecked_task_rejected(self):
        text = "- [x] 1.1 Do the thing.\n- [ ] 1.2 Verify it.\n"
        errors = co.check_tasks(text)
        self.assertNotEqual([], errors)
        self.assertIn("task is not complete", errors[0])

    def test_undescribed_task_rejected(self):
        text = "- [x] \n"
        self.assertNotEqual([], co.check_tasks(text))

    def test_empty_task_list_rejected(self):
        self.assertNotEqual([], co.check_tasks("no checkboxes at all\n"))

    def test_capital_marker_rejected(self):
        # A GitHub-checked but noncanonical marker is rejected as a
        # marker-shape violation, not misreported as incomplete.
        errors = co.check_tasks("- [X] 1.1 Done.\n")
        self.assertNotEqual([], errors)
        self.assertIn("must be exactly '[x]'", errors[0])
        self.assertNotIn("not complete", errors[0])

    def test_nonstandard_marker_rejected(self):
        text = "- [x] 1.1 Done.\n- [~] 1.2 Deferred.\n"
        self.assertNotEqual([], co.check_tasks(text))

    def test_multichar_marker_rejected(self):
        text = "- [x] 1.1 Done.\n- [wip] 1.2 In progress.\n"
        self.assertNotEqual([], co.check_tasks(text))

    def test_double_space_bullet_unchecked_rejected(self):
        # GFM renders "-  [ ]" (two spaces) as a real checkbox, so an
        # unchecked task must not hide behind extra list-marker spacing.
        text = "- [x] 1.1 Done.\n-  [ ] 1.2 Hidden.\n"
        errors = co.check_tasks(text)
        self.assertNotEqual([], errors)
        self.assertIn("task is not complete", errors[0])

    def test_double_space_bullet_complete_accepted(self):
        self.assertEqual([], co.check_tasks("-  [x] 1.1 Done.\n"))

    def test_tab_bullet_unchecked_rejected(self):
        text = "- [x] 1.1 Done.\n-\t[ ] 1.2 Hidden.\n"
        self.assertNotEqual([], co.check_tasks(text))

    def test_ordered_item_unchecked_rejected(self):
        # GFM renders ordered-list task items ("1. [ ]") as checkboxes.
        text = "- [x] 1.1 Done.\n1. [ ] 1.2 Hidden.\n"
        self.assertNotEqual([], co.check_tasks(text))

    def test_ordered_item_complete_accepted(self):
        self.assertEqual([], co.check_tasks("1. [x] 1.1 Done.\n"))

    def test_ordered_link_item_is_not_a_checkbox(self):
        text = "- [x] 1.1 Done.\n1. [Evidence](https://example.invalid/run)\n"
        self.assertEqual([], co.check_tasks(text))

    def test_asterisk_bullet_unchecked_rejected(self):
        self.assertNotEqual([], co.check_tasks("* [ ] 1.1 Do the thing.\n"))

    def test_asterisk_bullet_complete_accepted(self):
        self.assertEqual([], co.check_tasks("* [x] 1.1 Done.\n"))

    def test_markdown_link_item_is_not_a_checkbox(self):
        text = "- [x] 1.1 Done.\n- [Evidence](https://example.invalid/run)\n"
        self.assertEqual([], co.check_tasks(text))


class TestArchiveNovelty(unittest.TestCase):
    def test_new_archive_accepted(self):
        self.assertEqual(
            [],
            co.check_archive_novelty(
                frozenset({"2026-07-26-new-change"}),
                frozenset({"2026-01-01-old-change"}),
            ),
        )

    def test_inherited_archive_mutation_rejected(self):
        errors = co.check_archive_novelty(
            frozenset({"2026-01-01-old-change"}),
            frozenset({"2026-01-01-old-change"}),
        )
        self.assertNotEqual([], errors)
        self.assertIn("inherited from the comparison base", errors[0])


class TestArchiveCompleteness(unittest.TestCase):
    def _populate(self, directory, names):
        for name in names:
            (directory / name).write_text("content\n", encoding="utf-8")

    def test_complete_archive_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._populate(directory, (".openspec.yaml", "proposal.md", "tasks.md"))
            self.assertEqual(
                [], co.check_archive_completeness(directory, "2026-07-24-x-change")
            )

    def test_missing_proposal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._populate(directory, (".openspec.yaml", "tasks.md"))
            errors = co.check_archive_completeness(directory, "2026-07-24-x-change")
            self.assertNotEqual([], errors)
            self.assertIn("proposal.md", errors[0])


class TestProposalCapabilities(unittest.TestCase):
    def test_new_and_modified_sections_parsed(self):
        text = (
            "### New Capabilities\n\n- `cap-a`: one.\n\n"
            "### Modified Capabilities\n\n- `cap-b`: two.\n"
        )
        self.assertEqual(
            frozenset({"cap-a", "cap-b"}), co._proposal_capabilities(text)
        )

    def test_none_placeholder_declares_nothing(self):
        text = "### Modified Capabilities\n\nNone.\n"
        self.assertEqual(frozenset(), co._proposal_capabilities(text))


class TestLifecycleName(unittest.TestCase):
    def test_kebab_case_accepted(self):
        self.assertEqual([], co.check_lifecycle_name(CHANGE_ID))

    def test_uppercase_rejected(self):
        self.assertNotEqual([], co.check_lifecycle_name("Adopt-Change"))

    def test_underscore_rejected(self):
        self.assertNotEqual([], co.check_lifecycle_name("evil_change"))

    def test_dot_prefix_rejected(self):
        self.assertNotEqual([], co.check_lifecycle_name(".evil"))


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
        "nonstandard-task-marker",
        "active-merge-state",
        "malformed-archive",
        "unsynchronized-delta",
        "unarchived-baseline-mutation",
        "merge-hidden-change",
        "unrecognized-governance-path",
        "unguarded-config-mutation",
        "inherited-exemption-mutation",
        "hidden-archive-path",
        "inherited-archive-mutation",
        "multichar-task-marker",
        "renamed-delta-operation",
        "incomplete-archive",
        "archived-exemption-mix",
        "spaced-task-checkbox",
        "ordered-task-checkbox",
        "malformed-lifecycle-id",
        "lowercase-delta-operation",
        "near-miss-delta-heading",
        "preamble-delta-block",
        "duplicate-delta-requirement",
        "uncovered-baseline-mutation",
        "fenced-scenario-bypass",
        "duplicate-baseline-requirement",
        "nested-hidden-marker",
        "active-archive-mix",
    }

    def test_control_inventory_is_complete(self):
        self.assertEqual(self.REQUIRED_CONTROLS, set(co.SELF_TEST_CONTROLS))

    def test_self_test_passes(self):
        self.assertEqual([], co.self_test())


class TestRepositoryFixtures(unittest.TestCase):
    """End-to-end merge-bound runs against throwaway git repositories.

    Regression fixtures for the 2026-07-24 red-team findings: each
    negative fixture mirrors a reproduction that previously exited 0.
    """

    LIFECYCLE = "x-change"
    ARCHIVE = "2026-07-24-x-change"
    OLD_ARCHIVE = "2026-01-01-old-change"
    PROPOSAL_TEXT = (
        "## Why\n\nFixture.\n\n"
        "## Capabilities\n\n"
        "### New Capabilities\n\n"
        "- `cap`: fixture capability.\n"
    )
    BASELINE = (
        "### Requirement: Example holds\n"
        "The system SHALL hold.\n\n"
        "#### Scenario: It works\n"
        "- **WHEN** input is valid\n"
        "- **THEN** it SHALL pass\n\n"
        "#### Scenario: It fails\n"
        "- **WHEN** input is invalid\n"
        "- **THEN** it SHALL fail\n"
    )

    def _git(self, repo, *args):
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.invalid",
                *args,
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )

    def _write(self, repo, relative, text):
        path = Path(repo) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _commit(self, repo, message):
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-q", "-m", message)

    def _init_repo(self, tmp, seed=None):
        repo = Path(tmp) / "repo"
        repo.mkdir()
        self._git(repo, "init", "-q", "-b", "main")
        self._write(repo, "README.md", "base\n")
        if seed is not None:
            seed(repo)
        self._commit(repo, "base")
        self._git(repo, "checkout", "-q", "-b", "feature")
        return repo

    def _seed_inherited_archive(self, repo):
        prefix = f"openspec/changes/archive/{self.OLD_ARCHIVE}"
        self._write(repo, f"{prefix}/.openspec.yaml", "schema: spec-driven\n")
        self._write(repo, f"{prefix}/proposal.md", self.PROPOSAL_TEXT)
        self._write(repo, f"{prefix}/tasks.md", "- [x] 1.1 Done.\n")
        self._write(repo, f"{prefix}/specs/cap/spec.md", TestDeltaShape.GOOD)
        self._write(repo, "openspec/specs/cap/spec.md", self.BASELINE)

    def _fake_openspec(self, tmp):
        path = Path(tmp) / "fake_openspec"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "if '--version' in sys.argv:\n"
            "    print('1.6.0')\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def _merge_bound(
        self, tmp, repo, chdir=None, extra_env=None, mode="--merge-bound"
    ):
        cwd = os.getcwd()
        stderr = io.StringIO()
        os.chdir(chdir or repo)
        environ = {
            "OPENSPEC_BIN": str(self._fake_openspec(tmp)),
            "PATH": os.environ.get("PATH", os.defpath),
        }
        environ.update(extra_env or {})
        try:
            with contextlib.redirect_stderr(stderr):
                code = co.main([mode, "--base", "main"], environ=environ)
        finally:
            os.chdir(cwd)
        return code, stderr.getvalue()

    def _squash_feature(self, repo):
        self._git(repo, "checkout", "-q", "main")
        self._git(repo, "checkout", "-q", "-b", "release")
        self._git(repo, "merge", "-q", "--squash", "feature")
        self._commit(repo, "squashed merge")

    def _plan(self, repo, delta_text, proposal=None, lifecycle=None):
        prefix = f"openspec/changes/{lifecycle or self.LIFECYCLE}"
        self._write(repo, f"{prefix}/.openspec.yaml", "schema: spec-driven\n")
        self._write(repo, f"{prefix}/proposal.md", proposal or self.PROPOSAL_TEXT)
        self._write(repo, f"{prefix}/specs/cap/spec.md", delta_text)
        self._commit(repo, "plan")

    def _archive_lifecycle(self, repo, baseline=None, keep_specs=True, keep_proposal=True):
        prefix = f"openspec/changes/{self.LIFECYCLE}"
        target = f"openspec/changes/archive/{self.ARCHIVE}"
        Path(repo, target).mkdir(parents=True, exist_ok=True)
        self._git(repo, "mv", prefix + "/.openspec.yaml", target)
        if keep_proposal:
            self._git(repo, "mv", prefix + "/proposal.md", target)
        else:
            self._git(repo, "rm", "-q", prefix + "/proposal.md")
        if keep_specs:
            self._git(repo, "mv", prefix + "/specs", target + "/specs")
        else:
            self._git(repo, "rm", "-q", "-r", prefix + "/specs")
        self._write(repo, f"{target}/tasks.md", "- [x] 1.1 Done.\n")
        if baseline is not None:
            self._write(repo, "openspec/specs/cap/spec.md", baseline)
        self._commit(repo, "archive")

    def test_synchronized_archived_control_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            self._archive_lifecycle(repo, baseline=self.BASELINE)
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(0, code, stderr)

    def test_direct_baseline_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._write(repo, "openspec/specs/cap/spec.md", self.BASELINE)
            self._commit(repo, "mutate baseline")
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn("without an archived lifecycle", stderr)
            self.assertIn("do not equal the replayed", stderr)

    def test_single_scenario_archived_delta_rejected(self):
        single_delta = TestDeltaShape.GOOD.split("#### Scenario: It fails")[0]
        single_baseline = self.BASELINE.split("#### Scenario: It fails")[0]
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, single_delta)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            self._archive_lifecycle(repo, baseline=single_baseline)
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn("needs at least two scenarios", stderr)

    def test_empty_archived_delta_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            self._archive_lifecycle(repo, baseline=None, keep_specs=False)
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn("no requirement delta specs", stderr)

    def test_evil_merge_hidden_production_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD)
            self._git(repo, "checkout", "-q", "main")
            self._git(repo, "checkout", "-q", "-b", "side")
            self._write(repo, "side.txt", "side\n")
            self._commit(repo, "side")
            self._git(repo, "checkout", "-q", "feature")
            self._git(repo, "merge", "-q", "--no-ff", "--no-commit", "side")
            self._write(repo, "crates/evil.rs", "fn main() {}\n")
            self._commit(repo, "evil merge")
            self._archive_lifecycle(repo, baseline=self.BASELINE)
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn("not attributable", stderr)
            self.assertIn("crates/evil.rs", stderr)

    def test_unrecognized_governance_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._write(repo, "openspec/notes/hack.txt", "hidden\n")
            self._commit(repo, "notes")
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn("unrecognized governance path", stderr)

    def test_archived_lifecycle_plus_exemption_rejected(self):
        # Review regression (third pass): the isolation requirement says a
        # branch never combines a lifecycle with an exemption, but the
        # pre-fix checker only rejected the active-lifecycle case, so an
        # archived lifecycle plus a valid exemption exited 0.
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            self._archive_lifecycle(repo, baseline=self.BASELINE)
            self._write(
                repo,
                "openspec/exemptions/2026-07-26-side-cleanup.toml",
                'kind = "maintenance"\nreason = "side cleanup"\n'
                'paths = ["crates/foo.rs"]\n',
            )
            self._commit(repo, "exempt")
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn(
                "archived lifecycle with a maintenance exemption", stderr
            )

    def test_new_exemption_accepted_at_merge_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._write(repo, "README.md", "improved\n")
            self._write(
                repo,
                "openspec/exemptions/2026-07-26-readme-touchup.toml",
                'kind = "maintenance"\nreason = "editorial"\n'
                'paths = ["README.md"]\n',
            )
            self._commit(repo, "maintenance")
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(0, code, stderr)

    def test_inherited_exemption_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            self._git(repo, "init", "-q", "-b", "main")
            self._write(repo, "README.md", "base\n")
            self._write(
                repo,
                "openspec/exemptions/2026-01-01-old-cleanup.toml",
                'kind = "maintenance"\nreason = "old cleanup"\npaths = []\n',
            )
            self._commit(repo, "base")
            self._git(repo, "checkout", "-q", "-b", "feature")
            self._write(
                repo,
                "openspec/exemptions/2026-01-01-old-cleanup.toml",
                'kind = "maintenance"\nreason = "rewritten"\npaths = []\n',
            )
            self._commit(repo, "tamper")
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn("inherited from the comparison base", stderr)

    def test_inherited_archive_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp, seed=self._seed_inherited_archive)
            self._write(
                repo,
                f"openspec/changes/archive/{self.OLD_ARCHIVE}/tasks.md",
                "- [x] 1.1 Rewritten.\n",
            )
            self._commit(repo, "tamper archive")
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn(
                "archived lifecycle inherited from the comparison base", stderr
            )

    def test_inherited_archive_laundering_rejected(self):
        # Marker-dance repro: plan a lifecycle named after the inherited
        # archive, rewrite that archive's delta into a no-op MODIFIED
        # replay, delete the active lifecycle, and ship production code
        # under the tampered archive. Exited 0 before check_archive_novelty.
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp, seed=self._seed_inherited_archive)
            self._plan(repo, TestDeltaShape.GOOD, lifecycle="old-change")
            self._write(
                repo,
                f"openspec/changes/archive/{self.OLD_ARCHIVE}/specs/cap/spec.md",
                "## MODIFIED Requirements\n\n" + self.BASELINE,
            )
            self._write(repo, "crates/evil.rs", "fn main() {}\n")
            self._commit(repo, "tamper and implement")
            self._git(repo, "rm", "-q", "-r", "openspec/changes/old-change")
            self._commit(repo, "hide the active lifecycle")
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn(
                "archived lifecycle inherited from the comparison base", stderr
            )

    def test_archive_missing_proposal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            self._archive_lifecycle(
                repo, baseline=self.BASELINE, keep_proposal=False
            )
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn("missing proposal.md", stderr)

    def test_archive_capability_drift_rejected(self):
        drifted = self.PROPOSAL_TEXT.replace("`cap`", "`other-cap`")
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD, proposal=drifted)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            self._archive_lifecycle(repo, baseline=self.BASELINE)
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn("disagree", stderr)

    def test_uncovered_baseline_capability_rejected(self):
        # Review regression: an archived change to one capability could
        # smuggle prose edits into an unrelated capability's baseline spec,
        # because requirement-block replay leaves non-requirement text
        # unchecked. Tie every changed baseline file to a delta.
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            self._archive_lifecycle(repo, baseline=self.BASELINE)
            self._write(
                repo,
                "openspec/specs/other-cap/spec.md",
                "# Other Cap\n\n## Purpose\n\nRewritten prose.\n\n"
                "## Requirements\n\n" + self.BASELINE,
            )
            self._commit(repo, "smuggle prose")
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn("without a corresponding archived delta", stderr)

    def test_at_prefixed_path_cannot_forge_commit_boundary(self):
        # The commit log uses a NUL sentinel, so a committed path that
        # begins with "@" must not desynchronize commit attribution.
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD)
            self._write(repo, "@sentinel.txt", "decoy\n")
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            self._archive_lifecycle(repo, baseline=self.BASELINE)
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(0, code, stderr)

    def test_squashed_push_event_accepted(self):
        # A provider squash merge collapses the planning ancestor into one
        # mainline commit; the post-merge push lane must validate every
        # diff-shaped control without re-litigating commit ordering.
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            self._archive_lifecycle(repo, baseline=self.BASELINE)
            self._squash_feature(repo)
            code, stderr = self._merge_bound(
                tmp, repo, extra_env={"GITHUB_EVENT_NAME": "push"}
            )
            self.assertEqual(0, code, stderr)

    def test_squashed_history_still_rejected_off_the_push_lane(self):
        # Negative parity: the push-lane carve-out must not leak into
        # local or pull-request validation, where ordering evidence over
        # real branch history is required.
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            self._archive_lifecycle(repo, baseline=self.BASELINE)
            self._squash_feature(repo)
            code, stderr = self._merge_bound(tmp, repo)
            self.assertEqual(1, code)
            self.assertIn("no planning commit", stderr)

    def test_push_event_still_enforces_diff_controls(self):
        # The push lane skips only ordering/attribution; a violated
        # diff-shaped control (direct baseline mutation) still fails.
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._write(repo, "openspec/specs/cap/spec.md", self.BASELINE)
            self._commit(repo, "mutate baseline")
            code, stderr = self._merge_bound(
                tmp, repo, extra_env={"GITHUB_EVENT_NAME": "push"}
            )
            self.assertEqual(1, code)
            self.assertIn("do not equal the replayed", stderr)

    def test_run_from_subdirectory_accepted(self):
        # The checker resolves the repository root from git, so running
        # from a subdirectory must behave identically to the root.
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            self._archive_lifecycle(repo, baseline=self.BASELINE)
            code, stderr = self._merge_bound(
                tmp, repo, chdir=Path(repo) / "crates"
            )
            self.assertEqual(0, code, stderr)

    def test_active_lifecycle_accepted_at_pre_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            code, stderr = self._merge_bound(tmp, repo, mode="--pre-archive")
            self.assertEqual(0, code, stderr)

    def test_uppercase_lifecycle_id_rejected_at_pre_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._plan(repo, TestDeltaShape.GOOD, lifecycle="Evil_Change")
            code, stderr = self._merge_bound(tmp, repo, mode="--pre-archive")
            self.assertEqual(1, code)
            self.assertIn("kebab-case", stderr)

    def test_wrong_openspec_version_fails_before_repo_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._init_repo(tmp)
            self._write(repo, "crates/foo.rs", "fn main() {}\n")
            self._commit(repo, "impl")
            fake = Path(tmp) / "fake_openspec_wrong"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "if '--version' in sys.argv:\n"
                "    print('1.7.0')\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            cwd = os.getcwd()
            stderr = io.StringIO()
            os.chdir(repo)
            try:
                with contextlib.redirect_stderr(stderr):
                    code = co.main(
                        ["--merge-bound", "--base", "main"],
                        environ={
                            "OPENSPEC_BIN": str(fake),
                            "PATH": os.environ.get("PATH", os.defpath),
                        },
                    )
            finally:
                os.chdir(cwd)
            self.assertEqual(1, code)
            text = stderr.getvalue()
            self.assertIn("version mismatch", text)
            self.assertNotIn("governed paths changed", text)


if __name__ == "__main__":
    unittest.main()
