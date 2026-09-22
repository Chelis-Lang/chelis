"""Executable contract for #1251: author, assemble, and publish release notes."""

from __future__ import annotations

import json
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import changelog as m


HISTORY = "# Changelog\n\nProject notes.\n\n## [0.1.0] - 2026-01-01\n\n### Fixed\n\n- Old fix.\n"
LEGACY_UNRELEASED = """# Changelog

Project notes.

## [Unreleased]

### Changed

- **First change.**
  Its continuation is retained.

- **BREAKING (checker/CLI): Second change.**
  ```text
  example
  ```

### Fixed

- **A fix.**

"""


class ChangelogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.write("CHANGELOG.md", HISTORY)
        self.write("Cargo.toml", '[workspace.package]\nversion = "0.2.0"\n')
        self.write("changelog.d/README.md", "# Fragments\n")
        for name in (
            "changelog.py",
            "ci_contract_paths.py",
            "ci_detect_docs_only.py",
        ):
            target = self.root / "scripts" / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(Path(__file__).with_name(name), target)
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Changelog Test")
        self.git("config", "core.hooksPath", "/dev/null")
        self.base = self.commit()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "fixture", "--allow-empty")
        return self.git("rev-parse", "HEAD")

    def cli(self, *args, success=True):
        result = subprocess.run(
            [sys.executable, "-B", str(self.root / "scripts/changelog.py"), *args],
            cwd=self.tmp.name, capture_output=True, text=True,
            env={key: value for key, value in os.environ.items()
                 if key not in ("GITHUB_EVENT_PATH", "GITHUB_STEP_SUMMARY")},
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def build(self, *extra, success=True):
        return self.cli("build", "--version", "0.2.0", "--date", "2026-09-08",
                        *extra, success=success)

    def policy(self, *extra, success=True):
        return self.cli("check-pr", "--base", self.base, "--head", "HEAD",
                        *extra, success=success)

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file() and ".git" not in p.parts}

    def invoke_main(self, args):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = m.main(args, root=self.root)
        return code, output.getvalue(), errors.getvalue()

    def test_exact_rendering_and_preview_is_read_only(self):
        self.write("changelog.d/z.fixed.md", "Ordinary fix.\n")
        self.write("changelog.d/b.changed.md", "New behavior.\n")
        self.write("changelog.d/a.fixed.breaking.md", "Migration needed.\n")
        self.write("changelog.d/c.added.md", "New command.\n")
        before = self.snapshot()
        result = self.build()
        self.assertEqual(result.stdout, "## [0.2.0] - 2026-09-08\n\n### Added\n\n"
                         "- New command.\n\n### Changed\n\n- New behavior.\n\n"
                         "### Fixed\n\n- **BREAKING:** Migration needed.\n\n- Ordinary fix.\n\n")
        self.assertIn("changelog.d/z.fixed.md", result.stderr)
        self.assertEqual(before, self.snapshot())

    def test_multiline_markdown_and_stable_filename_order(self):
        self.write("changelog.d/z.added.md", "Last.\n")
        self.write("changelog.d/a.added.md", "First.\n\nMore detail.\n\n- Nested\n\n"
                   "```text\n## [9.0.0] - 2026-01-01\n```\n")
        text = self.build().stdout
        self.assertLess(text.index("First."), text.index("Last."))
        self.assertIn("\n\n  More detail.\n\n  - Nested\n\n  ```text\n  ## [9.0.0]", text)
        self.build("--write")
        self.cli("check")

    def test_write_preserves_history_and_consumes_only_fragments(self):
        self.write("changelog.d/12.fixed.md", "A fix.\n")
        self.build("--write")
        text = (self.root / "CHANGELOG.md").read_text()
        self.assertTrue(text.startswith(HISTORY.split("## [0.1.0]")[0]))
        self.assertTrue(text.endswith(HISTORY[HISTORY.index("## [0.1.0]"):]))
        self.assertEqual([p.name for p in (self.root / "changelog.d").iterdir()], ["README.md"])
        self.assertIn("already exists", self.build("--write", success=False).stderr)

    def test_empty_check_succeeds_but_empty_build_fails(self):
        self.cli("check")
        self.assertIn("no fragments", self.build(success=False).stderr)

    def test_invalid_fragment_inputs_never_mutate(self):
        cases = [("note.md", "Entry."), ("note.removed.md", "Entry."),
                 ("note.fixed.md", " \n"), ("note.fixed.md", "- Outer bullet"),
                 ("note.fixed.md", "# Heading"), ("note.fixed.md", "```text\ncode\n```"),
                 ("nested/note.fixed.md", "Entry."), (".hidden", "Entry.")]
        for name, body in cases:
            with self.subTest(name=name, body=body):
                path = self.write("changelog.d/" + name, body)
                before = self.snapshot()
                self.build("--write", success=False)
                self.assertEqual(before, self.snapshot())
                path.unlink()
                if path.parent.name == "nested":
                    path.parent.rmdir()

    def test_invalid_utf8_is_a_policy_error(self):
        path = self.write("changelog.d/a.fixed.md", "")
        path.write_bytes(b"\xff")
        result = self.build(success=False)
        self.assertIn("UTF-8", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_symlinks_are_rejected_without_reading_targets(self):
        for name in ("a.fixed.md", "README.md"):
            with self.subTest(name=name):
                path = self.root / "changelog.d" / name
                path.unlink(missing_ok=True)
                path.symlink_to(self.root / "does_not_exist")
                self.assertIn("regular file", self.build(success=False).stderr)
                path.unlink()
        self.write("changelog.d/README.md", "Notes")

    def test_invalid_versions_dates_and_workspace_mismatch(self):
        self.write("changelog.d/a.fixed.md", "Fix.")
        for version, date in [("01.2.0", "2026-09-08"), ("0.2", "2026-09-08"),
                              ("0.2.0-01", "2026-09-08"), ("0.2.0", "2026-02-30"),
                              ("0.2.0", "20260908"), ("0.3.0", "2026-09-08")]:
            with self.subTest(version=version, date=date):
                before = self.snapshot()
                self.cli("build", "--version", version, "--date", date, "--write", success=False)
                self.assertEqual(before, self.snapshot())

    def test_prerelease_and_build_metadata(self):
        self.write("Cargo.toml", '[workspace.package]\nversion = "1.0.0-rc.1+build.2"\n')
        self.write("changelog.d/a.fixed.md", "Fix.")
        self.cli("build", "--version", "1.0.0-rc.1+build.2", "--date", "2026-09-08", "--write")
        self.cli("extract", "--version", "v1.0.0-rc.1+build.2", "--output", str(self.root / "notes.md"))

    def test_unreleased_and_duplicate_headers_reject(self):
        for extra in ("## [Unreleased]\n\n", "## [0.1.0] - 2026-01-01\n\n"):
            with self.subTest(extra=extra):
                self.write("CHANGELOG.md", extra + HISTORY)
                self.cli("check", success=False)

    def test_fenced_header_is_not_a_release(self):
        self.write("CHANGELOG.md", HISTORY + "\n~~~~text\n## [0.1.0] - 2026-01-01\n~~~~\n")
        self.cli("check")

    def test_unclosed_fragment_fence_never_swallows_history(self):
        self.write("changelog.d/a.fixed.md", "Fix.\n\n```text\nExample")
        before = self.snapshot()
        self.assertIn("unclosed code fence", self.build("--write", success=False).stderr)
        self.assertEqual(before, self.snapshot())

    def test_crlf_history_is_preserved_as_bytes(self):
        raw = HISTORY.replace("\n", "\r\n").encode()
        (self.root / "CHANGELOG.md").write_bytes(raw)
        self.write("changelog.d/a.fixed.md", "Fix.")
        self.build("--write")
        actual = (self.root / "CHANGELOG.md").read_bytes()
        self.assertTrue(actual.endswith(raw[raw.index(b"## [0.1.0]"):]))
        self.assertTrue(actual.startswith(raw[:raw.index(b"## [0.1.0]")]))

    def test_missing_and_nonregular_changelog_inputs_reject(self):
        for name in ("CHANGELOG.md", "Cargo.toml", "changelog.d/README.md"):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.unlink()
                self.build(success=False)
                path.symlink_to(self.root / "missing")
                self.build(success=False)
                path.unlink()
                path.write_bytes(original)

    def test_git_symlink_fragment_fails(self):
        (self.root / "changelog.d/a.fixed.md").symlink_to("README.md")
        self.commit()
        self.assertIn("regular file", self.policy(success=False).stdout)

    def test_summary_reports_failure_without_claiming_success(self):
        self.write("crates/compiler/src/lib.rs", "Change")
        self.commit()
        summary = self.root / "summary.md"
        event = self.write("event.json", json.dumps({"pull_request": {"labels": []}}))
        with mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}):
            code, output, errors = self.invoke_main(["check-pr", "--base", self.base, "--head", "HEAD", "--event-file", str(event)])
        self.assertEqual(code, 1, errors)
        self.assertIn("::error::missing fragment", output)
        self.assertIn("Changelog: FAIL", summary.read_text())
        self.assertIn("missing fragment", summary.read_text())
        self.assertNotIn("PASS", summary.read_text())

    def test_annotation_escapes_percent_and_newline_in_git_path(self):
        self.write("changelog.d/bad%\nname.md", "Entry.")
        self.commit()
        self.assertIn("bad%25%0Aname.md", self.policy(success=False).stdout)

    def test_failed_changelog_write_does_not_delete_fragments(self):
        self.write("changelog.d/a.fixed.md", "Fix.")
        before = self.snapshot()
        with mock.patch.object(m.os, "replace", side_effect=OSError("write refused")):
            code, _, errors = self.invoke_main(["build", "--version", "0.2.0", "--date", "2026-09-08", "--write"])
        self.assertEqual(code, 2)
        self.assertIn("write refused", errors)
        self.assertEqual(before, self.snapshot())

    def test_failed_deletion_keeps_published_note_and_reports_failure(self):
        self.write("changelog.d/a.fixed.md", "Fix.")
        with mock.patch.object(Path, "unlink", side_effect=PermissionError("delete refused")):
            code, _, errors = self.invoke_main(["build", "--version", "0.2.0", "--date", "2026-09-08", "--write"])
        self.assertEqual(code, 2)
        self.assertIn("delete refused", errors)
        self.assertIn("- Fix.", (self.root / "CHANGELOG.md").read_text())
        self.assertTrue((self.root / "changelog.d/a.fixed.md").exists())

    def test_extract_exact_section_and_does_not_touch_sources(self):
        self.write("changelog.d/a.changed.breaking.md", "New contract.\n\nMigration instructions.")
        expected = self.build().stdout
        self.build("--write")
        before = self.snapshot()
        output = self.root / "notes.md"
        self.cli("extract", "--version", "v0.2.0", "--output", str(output))
        self.assertEqual(output.read_text(), expected)
        output.unlink()
        self.assertEqual(before, self.snapshot())

    def test_extract_refuses_missing_empty_mismatched_or_pending_notes(self):
        self.cli("extract", "--version", "0.2.0", "--output", str(self.root / "out"), success=False)
        self.write("CHANGELOG.md", "# Changelog\n\n## [0.2.0] - 2026-09-08\n\n" + HISTORY[HISTORY.index("## [0.1.0]"):])
        self.cli("extract", "--version", "0.2.0", "--output", str(self.root / "out"), success=False)
        self.write("CHANGELOG.md", HISTORY)
        self.write("changelog.d/a.fixed.md", "Fix.")
        self.build("--write")
        self.cli("extract", "--version", "0.1.0", "--output", str(self.root / "out"), success=False)
        self.write("changelog.d/b.fixed.md", "Another fix.")
        self.cli("extract", "--version", "0.2.0", "--output", str(self.root / "out"), success=False)

    def test_extract_cannot_overwrite_its_sources(self):
        self.write("changelog.d/a.fixed.md", "Fix.")
        self.build("--write")
        before = self.snapshot()
        for output in ("CHANGELOG.md", "Cargo.toml", "changelog.d/README.md"):
            self.cli("extract", "--version", "0.2.0", "--output", str(self.root / output), success=False)
        self.assertEqual(before, self.snapshot())

    def test_docs_only_and_new_fragment_pass(self):
        self.write("docs/example.md", "Documentation")
        self.commit()
        self.policy()
        self.write("crates/compiler/src/lib.rs", "fn example() {}")
        self.write("changelog.d/a.fixed.md", "Fix.")
        self.commit()
        self.policy()

    def test_non_docs_and_normative_specs_need_fragment(self):
        paths = ["crates/compiler/src/lib.rs", "scripts/helper.py", "tests/example.rs",
                 ".github/workflows/ci.yml", "spec/00-context.md", "spec/registry/ops.md"]
        for path in paths:
            with self.subTest(path=path):
                self.git("reset", "--hard", self.base)
                self.write(path, "Change")
                self.commit()
                self.assertIn("missing fragment", self.policy(success=False).stdout)

    def test_label_only_suppresses_missing_fragment(self):
        event = self.write("event.json", json.dumps({"pull_request": {"labels": [{"name": "no-changelog"}]}}))
        self.write("crates/compiler/src/lib.rs", "Changed")
        self.commit()
        result = self.policy("--event-file", str(event))
        self.assertNotIn("missing fragment", result.stdout)
        self.write("CHANGELOG.md", HISTORY.replace("Old fix", "Historical rewrite"))
        self.commit()
        result = self.policy("--event-file", str(event), success=False)
        self.assertIn("direct CHANGELOG.md edit", result.stdout)

    def test_amended_fragment_counts_but_identity_rename_does_not(self):
        path = self.write("changelog.d/a.fixed.md", "Fix.")
        self.base = self.commit()
        path.rename(path.with_name("b.fixed.md"))
        self.write("crates/compiler/src/lib.rs", "Changed")
        self.commit()
        self.policy(success=False)
        self.write("changelog.d/b.fixed.md", "Corrected fix.")
        self.commit()
        self.policy()

    def test_breaking_flag_amendment_counts(self):
        path = self.write("changelog.d/a.fixed.md", "Fix.")
        self.base = self.commit()
        path.rename(path.with_name("a.fixed.breaking.md"))
        self.write("crates/compiler/src/lib.rs", "Changed")
        self.commit()
        self.policy()

    def test_invalid_fragment_fails_even_for_docs_only_change(self):
        self.write("changelog.d/a.fixed.md", "")
        self.commit()
        self.assertIn("empty", self.policy(success=False).stdout)

    def test_release_diff_is_reproducible_or_fails(self):
        self.write("Cargo.toml", '[workspace.package]\nversion = "0.1.0"\n')
        self.write("changelog.d/a.fixed.md", "Fix.")
        self.base = self.commit()
        self.write("Cargo.toml", '[workspace.package]\nversion = "0.2.0"\n')
        self.build("--write")
        self.commit()
        self.policy()
        self.write("CHANGELOG.md", (self.root / "CHANGELOG.md").read_text().replace("Old fix", "Rewritten"))
        self.commit()
        self.assertIn("direct CHANGELOG.md edit", self.policy(success=False).stdout)

    def test_fake_release_with_leftover_fragment_fails(self):
        self.write("Cargo.toml", '[workspace.package]\nversion = "0.1.0"\n')
        self.write("changelog.d/a.fixed.md", "Fix.")
        self.base = self.commit()
        self.write("Cargo.toml", '[workspace.package]\nversion = "0.2.0"\n')
        self.build("--write")
        self.write("changelog.d/b.fixed.md", "Left over.")
        self.commit()
        self.assertIn("direct CHANGELOG.md edit", self.policy(success=False).stdout)

    def test_stale_base_uses_merge_base_not_sibling_changes(self):
        self.git("checkout", "-qb", "sibling")
        self.write("crates/compiler/src/lib.rs", "Sibling change")
        sibling = self.commit()
        self.git("checkout", "--detach", self.base)
        self.write("docs/example.md", "Docs only")
        self.commit()
        self.base = sibling
        self.policy()

    def test_unchanged_changelog_does_not_block_fragment_authoring(self):
        self.write("CHANGELOG.md", "## [Unreleased]\n\n- Existing note.\n\n" + HISTORY)
        self.base = self.commit()
        self.write("docs/example.md", "Docs only")
        self.commit()
        self.policy()
        self.write("crates/compiler/src/lib.rs", "Changed")
        self.commit()
        self.assertIn("missing fragment", self.policy(success=False).stdout)
        self.write("changelog.d/a.fixed.md", "Fix.")
        self.commit()
        self.policy()
        self.cli("check", success=False)
        self.build(success=False)

    def test_existing_unreleased_notes_do_not_allow_direct_edits(self):
        self.write("CHANGELOG.md", "## [Unreleased]\n\n- Existing note.\n\n" + HISTORY)
        self.base = self.commit()
        self.write("CHANGELOG.md", "## [Unreleased]\n\n- Changed note.\n\n" + HISTORY)
        self.commit()
        event = self.write("event.json", json.dumps({"pull_request": {"labels": [{"name": "no-changelog"}]}}))
        result = self.policy("--event-file", str(event), success=False)
        self.assertIn("direct CHANGELOG.md edit", result.stdout)
        self.assertIn("move [Unreleased] notes", result.stdout)

    def test_migrate_unreleased_preview_and_write_are_lossless(self):
        self.write("CHANGELOG.md", LEGACY_UNRELEASED + HISTORY[HISTORY.index("## [0.1.0]"):])
        before = self.snapshot()
        result = self.cli("migrate-unreleased")
        self.assertIn("changelog.d/legacy-unreleased-001.changed.md", result.stdout)
        self.assertEqual(before, self.snapshot())
        self.cli("migrate-unreleased", "--write")
        self.assertEqual((self.root / "CHANGELOG.md").read_bytes(), HISTORY.encode())
        self.assertEqual((self.root / "changelog.d/legacy-unreleased-001.changed.md").read_text(),
                         "**First change.**\nIts continuation is retained.\n")
        self.assertEqual((self.root / "changelog.d/legacy-unreleased-002.changed.breaking.md").read_text(),
                         "**checker/CLI: Second change.**\n```text\nexample\n```\n")
        self.assertEqual((self.root / "changelog.d/legacy-unreleased-003.fixed.md").read_text(),
                         "**A fix.**\n")
        self.cli("check")

    def test_migration_pr_admission_requires_exact_conservation(self):
        self.write("CHANGELOG.md", LEGACY_UNRELEASED + HISTORY[HISTORY.index("## [0.1.0]"):])
        self.base = self.commit()
        self.cli("migrate-unreleased", "--write")
        self.commit()
        self.policy()
        self.write("changelog.d/legacy-unreleased-001.changed.md", "**Rewritten.**\n")
        self.commit()
        self.assertIn("direct CHANGELOG.md edit", self.policy(success=False).stdout)

    def test_migration_pr_rejects_omitted_or_reclassified_legacy_note(self):
        for action in ("omit", "reclassify"):
            with self.subTest(action=action):
                self.write("CHANGELOG.md", LEGACY_UNRELEASED + HISTORY[HISTORY.index("## [0.1.0]"):])
                self.base = self.commit()
                self.cli("migrate-unreleased", "--write")
                if action == "omit":
                    (self.root / "changelog.d/legacy-unreleased-001.changed.md").unlink()
                else:
                    path = self.root / "changelog.d/legacy-unreleased-001.changed.md"
                    path.rename(path.with_name("legacy-unreleased-001.fixed.md"))
                self.commit()
                self.assertIn("direct CHANGELOG.md edit", self.policy(success=False).stdout)

    def test_migration_pr_allows_docs_and_an_additional_new_fragment(self):
        self.write("CHANGELOG.md", LEGACY_UNRELEASED + HISTORY[HISTORY.index("## [0.1.0]"):])
        self.base = self.commit()
        self.cli("migrate-unreleased", "--write")
        self.write("changelog.d/follow-up.fixed.md", "A separate pending note.\n")
        self.write("changelog.d/README.md", "# Updated fragment instructions\n")
        self.commit()
        self.policy()

    def test_migration_does_not_count_old_notes_as_new_release_content(self):
        self.write("CHANGELOG.md", LEGACY_UNRELEASED + HISTORY[HISTORY.index("## [0.1.0]"):])
        self.base = self.commit()
        self.cli("migrate-unreleased", "--write")
        self.write("crates/compiler/src/lib.rs", "pub fn changed_behavior() {}\n")
        self.commit()
        self.assertIn("missing fragment", self.policy(success=False).stdout)
        self.write("changelog.d/duplicate.changed.md",
                   (self.root / "changelog.d/legacy-unreleased-001.changed.md").read_text())
        self.commit()
        self.assertIn("missing fragment", self.policy(success=False).stdout)
        event = self.write("event.json", json.dumps({"pull_request": {"labels": [{"name": "no-changelog"}]}}))
        self.policy("--event", str(event))
        self.write("changelog.d/new.changed.md", "A genuinely new behavior.\n")
        self.commit()
        self.policy()

    def test_migration_pr_rejects_lost_pending_fragment(self):
        self.write("CHANGELOG.md", LEGACY_UNRELEASED + HISTORY[HISTORY.index("## [0.1.0]"):])
        self.write("changelog.d/pending.fixed.md", "Pending note.\n")
        self.base = self.commit()
        self.cli("migrate-unreleased", "--write")
        (self.root / "changelog.d/pending.fixed.md").unlink()
        self.commit()
        self.assertIn("direct CHANGELOG.md edit", self.policy(success=False).stdout)

    def test_migration_pr_rejects_version_bump(self):
        self.write("CHANGELOG.md", LEGACY_UNRELEASED + HISTORY[HISTORY.index("## [0.1.0]"):])
        self.base = self.commit()
        self.cli("migrate-unreleased", "--write")
        self.write("Cargo.toml", '[workspace.package]\nversion = "0.3.0"\n')
        self.commit()
        self.assertIn("direct CHANGELOG.md edit", self.policy(success=False).stdout)

    def test_migration_pr_rejects_preamble_or_history_rewrite(self):
        for target, replacement in (("preamble", "Rewritten notes."),
                                    ("history", "Historical rewrite.")):
            with self.subTest(target=target):
                self.write("CHANGELOG.md", LEGACY_UNRELEASED + HISTORY[HISTORY.index("## [0.1.0]"):])
                self.base = self.commit()
                self.cli("migrate-unreleased", "--write")
                path = self.root / "CHANGELOG.md"
                old = "Project notes." if target == "preamble" else "Old fix."
                path.write_text(path.read_text().replace(old, replacement), encoding="utf-8")
                self.commit()
                self.assertIn("direct CHANGELOG.md edit", self.policy(success=False).stdout)

    def test_migration_rejects_malformed_legacy_markdown_and_collisions(self):
        cases = (
            "Free prose.\n\n### Changed\n\n- **Entry.**\n\n",
            "### Added\n\n- **Entry.**\n continuation\n\n",
            "### Security\n\n- **Entry.**\n\n",
            "### Changed\n\n- **Entry.**\n  ```text\n  unclosed\n",
        )
        for body in cases:
            with self.subTest(body=body):
                self.write("CHANGELOG.md", "# Changelog\n\n## [Unreleased]\n\n" + body + HISTORY[HISTORY.index("## [0.1.0]"):])
                before = self.snapshot()
                self.cli("migrate-unreleased", "--write", success=False)
                self.assertEqual(before, self.snapshot())
        self.write("CHANGELOG.md", LEGACY_UNRELEASED + HISTORY[HISTORY.index("## [0.1.0]"):])
        collision = self.write("changelog.d/legacy-unreleased-001.changed.md", "Foreign note.\n")
        self.cli("migrate-unreleased", success=False)
        self.cli("migrate-unreleased", "--write", success=False)
        collision.unlink()
        collision.symlink_to("README.md")
        self.cli("migrate-unreleased", success=False)
        self.cli("migrate-unreleased", "--write", success=False)

    def test_migration_requires_history_and_recognizes_only_canonical_breaking(self):
        cases = (
            "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- **Entry.**\n",
            "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- **BREAKING:** Entry.\n\n"
            + HISTORY[HISTORY.index("## [0.1.0]"):],
        )
        for text in cases:
            with self.subTest(text=text):
                self.write("CHANGELOG.md", text)
                before = self.snapshot()
                self.cli("migrate-unreleased", "--write", success=False)
                self.assertEqual(before, self.snapshot())

    def test_migration_precreated_fragment_must_have_exact_bytes_and_mode(self):
        self.write("CHANGELOG.md", LEGACY_UNRELEASED + HISTORY[HISTORY.index("## [0.1.0]"):])
        plan = m.migration(m.disk_tree(self.root))
        path = self.root / "changelog.d/legacy-unreleased-001.changed.md"
        path.write_bytes(plan.fragments["changelog.d/legacy-unreleased-001.changed.md"])
        path.chmod(0o644)
        self.cli("migrate-unreleased")
        path.chmod(0o755)
        self.cli("migrate-unreleased", success=False)

    def test_migration_rejects_non_top_or_multiple_unreleased_sections(self):
        cases = (
            HISTORY + "\n## [Unreleased]\n\n### Fixed\n\n- **Too late.**\n",
            "# Changelog\n\n## [Unreleased] - 2026-09-12\n\n### Fixed\n\n- **Dated.**\n\n"
            + HISTORY[HISTORY.index("## [0.1.0]"):],
            "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- **One.**\n\n"
            "## [Unreleased]\n\n### Fixed\n\n- **Two.**\n\n" + HISTORY[HISTORY.index("## [0.1.0]"):],
        )
        for text in cases:
            with self.subTest(text=text):
                self.write("CHANGELOG.md", text)
                before = self.snapshot()
                self.cli("migrate-unreleased", "--write", success=False)
                self.assertEqual(before, self.snapshot())

    def test_fenced_fake_unreleased_heading_does_not_enable_migration(self):
        self.write("CHANGELOG.md", HISTORY + "\n```text\n## [Unreleased]\n```\n")
        before = self.snapshot()
        self.cli("migrate-unreleased", "--write", success=False)
        self.assertEqual(before, self.snapshot())

    def test_migration_partial_fragment_write_is_recoverable(self):
        self.write("CHANGELOG.md", LEGACY_UNRELEASED + HISTORY[HISTORY.index("## [0.1.0]"):])
        original = m.write_new_file
        calls = 0

        def fail_second(path, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("fragment write refused")
            original(path, data)

        with mock.patch.object(m, "write_new_file", side_effect=fail_second):
            code, _, errors = self.invoke_main(["migrate-unreleased", "--write"])
        self.assertEqual(code, 2)
        self.assertIn("fragment write refused", errors)
        self.assertIn("[Unreleased]", (self.root / "CHANGELOG.md").read_text())
        self.assertTrue((self.root / "changelog.d/legacy-unreleased-001.changed.md").exists())
        self.cli("migrate-unreleased", "--write")
        self.assertNotIn("[Unreleased]", (self.root / "CHANGELOG.md").read_text())

    def test_advisory_escape_is_not_supported(self):
        self.write("crates/compiler/src/lib.rs", "Changed")
        self.commit()
        result = self.policy("--advisory", success=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments: --advisory", result.stderr)

    def test_bad_git_ref_and_event_are_operational_failures(self):
        result = self.cli("check-pr", "--base", "missing", "--head", "HEAD", success=False)
        self.assertEqual(result.returncode, 2)
        event = self.write("event.json", "invalid json")
        result = self.policy("--event-file", str(event), success=False)
        self.assertEqual(result.returncode, 2)


class WorkflowTests(unittest.TestCase):
    def test_test_fixture_does_not_emit_workflow_annotations(self):
        root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "-B", "-m", "unittest",
             "scripts.test_changelog.ChangelogTests.test_summary_reports_failure_without_claiming_success"],
            cwd=root, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "", "fixture output must not become real GitHub annotations")
        self.assertNotIn("::warning::", result.stderr)
        self.assertNotIn("::error::", result.stderr)

    def test_required_workflow_and_publishing_wiring(self):
        root = Path(__file__).resolve().parent.parent
        workflow = (root / ".github/workflows/changelog.yml").read_text()
        self.assertIn("name: Changelog\n", workflow)
        self.assertNotIn("--advisory", workflow)
        for event in ("opened", "synchronize", "reopened", "edited", "labeled", "unlabeled"):
            self.assertIn(event, workflow)
        self.assertNotIn("paths:", workflow)
        self.assertNotIn("continue-on-error", workflow)
        self.assertIn("contents: read", workflow)
        self.assertIn("scripts.test_changelog", workflow)
        publish = (root / ".github/workflows/release.yml").read_text().split("  publish-release:")[1]
        self.assertIn("scripts/changelog.py extract", publish)
        self.assertIn("body_path:", publish)
        self.assertIn("generate_release_notes: false", publish)
        self.assertIn("persist-credentials: false", publish)
        self.assertIn("--version", publish)
        self.assertIn("github.ref_name", publish)
        self.assertLess(publish.index("scripts/changelog.py extract"), publish.index("uses: softprops/action-gh-release"))


if __name__ == "__main__":
    unittest.main()
