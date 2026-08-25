#!/usr/bin/env python3
"""Unit tests for `eval_system_guard.py` (OpenSpec `add-eval-system-boundary`).

Positive fixtures assert that legitimate evaluator code (the real adapter,
comments/strings mentioning a banned form, ordinary `Path` construction that
never touches the operating system) produces zero hits. Negative fixtures
cover every accepted-Rust-spelling bypass named in the capability design:
`std::fs` and imported `fs` calls, `std::process::Command` and imported
`Command` calls, `std::path::Path::exists` and imported `Path::exists`
calls, and a bare method-form `.exists()` call.
"""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_system_guard as guard


def reasons(text: str) -> list[str]:
    return [hit.reason for hit in guard.classify_source(text)]


class ClassifySourcePositiveFixturesTests(unittest.TestCase):
    """Legitimate code must produce zero hits."""

    def test_boundary_wrapper_calling_the_adapter_is_accepted(self):
        source = """
        pub(crate) fn read_file(&mut self, path: &Path) -> Result<String, EvalSystemError> {
            self.check(EvalSystemOperation::ReadFile)?;
            self.adapter.read_file(path)
        }
        """
        self.assertEqual(reasons(source), [])

    def test_bare_path_new_without_exists_is_accepted(self):
        # `Path::new` alone constructs a value; it does not touch the
        # operating system. Evaluator dispatch legitimately calls this
        # before handing the path to the boundary (`runtime/eval.rs`).
        source = 'let path = Path::new(&path_str);\nself.system.read_file(path)?;'
        self.assertEqual(reasons(source), [])

    def test_doc_comment_mentioning_banned_forms_is_accepted(self):
        source = """
        //! This module must never call std::fs:: or std::process::Command
        //! or Path::new(x).exists() directly.
        /* std::process::Command::new and .exists( are also banned here. */
        fn noop() {}
        """
        self.assertEqual(reasons(source), [])

    def test_string_literal_mentioning_banned_forms_is_accepted(self):
        source = (
            'let message = "call std::fs::read_to_string or Command::new "'
            ".to_string();"
        )
        self.assertEqual(reasons(source), [])


class ClassifySourceDirectFormsTests(unittest.TestCase):
    """Fully qualified direct-access forms."""

    def test_direct_std_fs_call(self):
        hits = reasons("let text = std::fs::read_to_string(path)?;")
        self.assertTrue(any("std::fs::" in reason for reason in hits), hits)

    def test_direct_std_process_command(self):
        hits = reasons('let out = std::process::Command::new("echo").output()?;')
        self.assertTrue(any("process" in reason for reason in hits), hits)

    def test_direct_std_path_path_exists(self):
        hits = reasons("let ok = std::path::Path::new(&p).exists();")
        self.assertTrue(any("filesystem" in reason for reason in hits), hits)

    def test_method_form_exists_call(self):
        hits = reasons("let ok = candidate.exists();")
        self.assertTrue(any("method-form" in reason for reason in hits), hits)

    def test_method_form_try_exists_call(self):
        hits = reasons("let ok = candidate.try_exists()?;")
        self.assertTrue(any("method-form" in reason for reason in hits), hits)

    def test_path_filesystem_methods_are_rejected(self):
        source = """
        let _ = Path::new(path).read_dir();
        let _ = Path::new(path).metadata();
        let _ = Path::new(path).is_file();
        """
        hits = reasons(source)
        self.assertEqual(len(hits), 3, hits)
        self.assertTrue(all("filesystem" in reason for reason in hits), hits)

    def test_direct_std_path_path_try_exists(self):
        hits = reasons("let ok = std::path::Path::new(&p).try_exists()?;")
        self.assertTrue(any("filesystem" in reason for reason in hits), hits)


class ClassifySourceImportedAliasFormsTests(unittest.TestCase):
    """Imported and aliased forms -- the harder half of the guard."""

    def test_imported_fs_bare_module_call(self):
        source = "use std::fs;\nfn f(p: &Path) { let _ = fs::read_to_string(p); }"
        hits = reasons(source)
        self.assertTrue(
            any("filesystem-module alias" in reason for reason in hits), hits
        )

    def test_imported_fs_aliased_module_call(self):
        source = (
            "use std::fs as sneaky_fs;\n"
            "fn f(p: &Path) { let _ = sneaky_fs::read(p); }"
        )
        hits = reasons(source)
        self.assertTrue(
            any("filesystem-module alias" in reason for reason in hits), hits
        )

    def test_imported_fs_function_bare_call(self):
        source = (
            "use std::fs::read_to_string;\n"
            "fn f(p: &Path) { let _ = read_to_string(p); }"
        )
        hits = reasons(source)
        self.assertTrue(
            any("directly imported filesystem function" in reason for reason in hits),
            hits,
        )

    def test_imported_fs_function_group_and_alias(self):
        source = (
            "use std::fs::{read_to_string, write as write_out};\n"
            "fn f(p: &Path) { let _ = write_out(p, \"x\"); }"
        )
        hits = reasons(source)
        self.assertTrue(
            any("directly imported filesystem function" in reason for reason in hits),
            hits,
        )

    def test_imported_command_bare_call(self):
        source = (
            "use std::process::Command;\n"
            'fn f() { let _ = Command::new("echo").output(); }'
        )
        hits = reasons(source)
        # The import itself is sufficient evidence of a guarded process path.
        self.assertTrue(any("process" in reason for reason in hits), hits)

    def test_imported_command_aliased_call(self):
        source = (
            "use std::process::Command as Proc;\n"
            'fn f() { let _ = Proc::new("echo").output(); }'
        )
        hits = reasons(source)
        self.assertTrue(
            any(
                "imported process alias" in reason or "Command" in reason
                for reason in hits
            ),
            hits,
        )

    def test_imported_path_exists_aliased_call(self):
        # The genuinely hard case: an aliased `Path` where the later call
        # site carries none of the literal `std::path::Path` text at all.
        source = (
            "use std::path::Path as P;\n"
            "fn f(p: &str) { let _ = P::exists(P::new(p)); }"
        )
        hits = reasons(source)
        self.assertTrue(
            any("imported path alias" in reason for reason in hits), hits
        )

    def test_bare_process_module_import_then_command_construction(self):
        # A red-team-found gap: `use std::process::Command` is directly
        # caught (the `use` line carries the literal qualified text), but
        # `use std::process;` alone does not, and the later
        # `process::Command::new(...)` call site carries no `std::`
        # prefix either. This needs explicit process-MODULE alias
        # tracking (`_USE_PROCESS_MODULE`), distinct from the
        # `_USE_COMMAND` item-import tracking above.
        source = (
            "use std::process;\n"
            'fn f() { let _ = process::Command::new("echo").output(); }'
        )
        hits = reasons(source)
        self.assertTrue(
            any("process-module alias" in reason for reason in hits), hits
        )

    def test_aliased_process_module_import_then_command_construction(self):
        source = (
            "use std::process as sneaky_process;\n"
            'fn f() { let _ = sneaky_process::Command::new("echo").output(); }'
        )
        hits = reasons(source)
        self.assertTrue(
            any("process-module alias" in reason for reason in hits), hits
        )

    def test_braced_process_item_import_is_rejected(self):
        source = 'use std::process::{Command, Stdio};\nfn f() { Command::new("sh"); }'
        hits = reasons(source)
        self.assertTrue(any("std::process" in reason for reason in hits), hits)

    def test_nested_std_tree_fs_and_process_imports_are_rejected(self):
        source = (
            "use std::{fs, process};\n"
            'fn f(p: &Path) { let _ = fs::read(p); let _ = process::Command::new("x"); }'
        )
        hits = reasons(source)
        self.assertTrue(any("nested `std` filesystem" in reason for reason in hits), hits)
        self.assertTrue(any("nested `std` process" in reason for reason in hits), hits)

    def test_balanced_nested_std_tree_filesystem_import_is_rejected(self):
        source = (
            "use std::{collections::{HashMap, HashSet}, fs};\n"
            "fn f(p: &Path) { let _ = fs::read(p); }"
        )
        hits = reasons(source)
        self.assertTrue(any("nested `std` filesystem" in reason for reason in hits), hits)

    def test_nested_std_tree_path_import_is_rejected(self):
        source = "use std::{io, path::Path};\nfn f(p: &Path) { let _ = p.try_exists(); }"
        hits = reasons(source)
        self.assertTrue(any("nested `std` path" in reason for reason in hits), hits)
        self.assertTrue(any("method-form" in reason for reason in hits), hits)

    def test_imported_path_alias_try_exists_call_is_rejected(self):
        source = "use std::path::Path as P;\nfn f(p: &P) { let _ = P::try_exists(p); }"
        hits = reasons(source)
        self.assertTrue(any("imported path alias" in reason for reason in hits), hits)

    def test_braced_path_alias_import_is_rejected(self):
        source = "use std::path::{Path as P};\nfn f(p: &P) { let _ = P::metadata(p); }"
        hits = reasons(source)
        self.assertTrue(any("braced `std::path::Path`" in reason for reason in hits), hits)

    def test_multiline_split_std_fs_path_is_still_caught(self):
        # A second red-team-found gap: matching one line at a time let a
        # bypass evade every pattern just by inserting a newline between
        # `::`-separated path segments (`\s` in the patterns already
        # matches `\n`; splitting into lines before matching threw that
        # away). Rust does not care where whitespace falls inside a path.
        source = "let x = std\n    ::fs\n    ::read_to_string(p);\n"
        hits = reasons(source)
        self.assertTrue(any("std::fs::" in reason for reason in hits), hits)

    def test_multiline_split_method_form_exists_is_still_caught(self):
        source = "let ok = candidate.\n    exists();\n"
        hits = reasons(source)
        self.assertTrue(
            any("method-form" in reason for reason in hits), hits
        )

    def test_multiline_split_command_construction_is_still_caught(self):
        source = 'let c = std::process\n    ::Command::new("echo");\n'
        hits = reasons(source)
        self.assertTrue(any("process" in reason for reason in hits), hits)

    def test_multiline_split_aliased_use_is_still_tracked(self):
        # The `use` statement itself can also be split across lines; the
        # alias-collection pass must see it the same way the direct
        # patterns do (both run over the full stripped text, never one
        # line at a time).
        source = "use std::fs\n    as sneaky_fs;\nfn f() { sneaky_fs::read(p); }\n"
        hits = reasons(source)
        self.assertTrue(
            any("filesystem-module alias" in reason for reason in hits), hits
        )


class ParseTestOnlyModulesTests(unittest.TestCase):
    def test_cfg_test_gated_mod_is_test_only(self):
        text = "mod eval;\n#[cfg(test)]\nmod tests;\nmod transforms;\n"
        self.assertEqual(guard.parse_test_only_modules(text), frozenset({"tests"}))

    def test_multiple_cfg_test_gated_mods(self):
        text = (
            "mod system;\n"
            "mod system_adapter;\n"
            "#[cfg(test)]\nmod system_tests;\n"
            "#[cfg(test)]\nmod tests;\n"
            "mod transforms;\n"
        )
        self.assertEqual(
            guard.parse_test_only_modules(text),
            frozenset({"system_tests", "tests"}),
        )

    def test_cfg_test_only_applies_to_the_immediately_following_mod(self):
        text = "#[cfg(test)]\nmod tests;\nmod eval;\n"
        self.assertEqual(guard.parse_test_only_modules(text), frozenset({"tests"}))
        self.assertNotIn("eval", guard.parse_test_only_modules(text))

    def test_inline_cfg_test_mod_on_one_line_is_test_only(self):
        # A red-team-found gap: `#[cfg(test)] mod x;` on a single line was
        # not recognized by the two-line-only parser.
        text = "mod eval;\n#[cfg(test)] mod tests;\nmod transforms;\n"
        self.assertEqual(guard.parse_test_only_modules(text), frozenset({"tests"}))

    def test_cfg_test_gated_pub_mod_is_test_only(self):
        # A second red-team-found gap: a visibility modifier on the `mod`
        # line was not matched by the bare `mod` regex.
        text = "#[cfg(test)]\npub mod tests;\nmod eval;\n"
        self.assertEqual(guard.parse_test_only_modules(text), frozenset({"tests"}))
        text_crate_scoped = "#[cfg(test)]\npub(crate) mod tests;\nmod eval;\n"
        self.assertEqual(
            guard.parse_test_only_modules(text_crate_scoped), frozenset({"tests"})
        )

    def test_stacked_attribute_between_cfg_test_and_mod_still_applies(self):
        # A third red-team-found gap: Rust permits stacking attributes
        # above one item; an unrelated attribute between `#[cfg(test)]`
        # and `mod x;` should not cancel the pending gate.
        text = "#[cfg(test)]\n#[allow(dead_code)]\nmod tests;\nmod eval;\n"
        self.assertEqual(guard.parse_test_only_modules(text), frozenset({"tests"}))

    def test_unrelated_attribute_alone_does_not_mark_a_module_test_only(self):
        # An attribute that is NOT `#[cfg(test)]`, with no `#[cfg(test)]`
        # anywhere above it, must not mark the following `mod` as
        # test-only.
        text = "#[allow(dead_code)]\nmod eval;\n"
        self.assertEqual(guard.parse_test_only_modules(text), frozenset())


class ScanDirectoryTests(unittest.TestCase):
    """Integration coverage over a synthetic `runtime/` tree."""

    def _write_runtime_tree(self, root: Path, files: dict[str, str]) -> None:
        for name, content in files.items():
            (root / name).write_text(content, encoding="utf-8")

    def test_adapter_module_with_real_calls_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_runtime_tree(
                root,
                {
                    "mod.rs": "mod eval;\nmod system_adapter;\n",
                    "eval.rs": "fn f() { self.system.read_file(p)?; }\n",
                    "system_adapter.rs": (
                        "fn f(p: &Path) { std::fs::read_to_string(p).unwrap(); }\n"
                    ),
                },
            )
            violations = guard.scan_directory(root)
            self.assertEqual(violations, [])

    def test_test_only_module_with_real_calls_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_runtime_tree(
                root,
                {
                    "mod.rs": (
                        "mod eval;\nmod system_adapter;\n"
                        "#[cfg(test)]\nmod system_tests;\n"
                    ),
                    "eval.rs": "fn f() { self.system.read_file(p)?; }\n",
                    "system_adapter.rs": "fn f(p: &Path) { std::fs::read(p).unwrap(); }\n",
                    "system_tests.rs": (
                        "fn seed(p: &Path) { std::fs::write(p, \"x\").unwrap(); }\n"
                    ),
                },
            )
            violations = guard.scan_directory(root)
            self.assertEqual(violations, [])

    def test_bypass_in_production_module_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_runtime_tree(
                root,
                {
                    "mod.rs": "mod eval;\nmod system_adapter;\n",
                    "eval.rs": "fn f(p: &Path) { std::fs::read(p).unwrap(); }\n",
                    "system_adapter.rs": "fn f(p: &Path) { std::fs::read(p).unwrap(); }\n",
                },
            )
            violations = guard.scan_directory(root)
            self.assertEqual(len(violations), 1)
            self.assertEqual(violations[0].path.name, "eval.rs")

    def test_missing_mod_rs_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_runtime_tree(root, {"eval.rs": "fn f() {}\n"})
            with self.assertRaises(guard.SourceGuardError):
                guard.scan_directory(root)

    def test_missing_runtime_directory_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does-not-exist"
            with self.assertRaises(guard.SourceGuardError):
                guard.scan_directory(missing)


class MainEntryPointTests(unittest.TestCase):
    def test_main_passes_against_the_real_repository_tree(self):
        # The authoritative end-to-end check: run the guard against the
        # actual `crates/chelis-compiler-api/src/runtime/` tree in this
        # checkout.
        self.assertEqual(guard.main(), 0)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    count = suite.countTestCases()
    if count == 0:
        print("eval system guard tests: FAIL: no tests discovered", file=sys.stderr)
        return 1
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        return 1
    print(f"eval system guard tests: PASS ({count} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
