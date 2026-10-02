#!/usr/bin/env python3
"""Adversarial bypass and test-scope controls for `eval_system_guard.py`.

Positive fixtures assert that legitimate evaluator code (adapter delegation,
raw literals mentioning host names, non-path methods) produces zero hits.
Negative fixtures cover qualified and imported filesystem/process calls,
platform-specific filesystem modules, aliased `std` roots, filesystem
methods on explicitly constructed or typed `Path`/`PathBuf` receivers, and
direct host clock reads.
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

    def test_raw_literals_mask_quotes_comments_and_host_names(self):
        source = (
            'let doc = r##"quote " end # std::fs::read /* fake */'
            ' std::process::Command::new("echo")"##;\n'
            'let bytes = br#"// std::fs::write"#;\n'
        )
        self.assertEqual(reasons(source), [])

    def test_multiline_raw_literal_preserves_production_line_numbers(self):
        source = 'let doc = r#"quote "\nstd::fs::read(fake)\n"#;\nstd::fs::read(path);\n'
        hits = guard.classify_source(source)
        self.assertEqual([hit.line_number for hit in hits], [4])

    def test_unterminated_raw_literal_fails_closed(self):
        with self.assertRaises(guard.SourceGuardError):
            reasons('let _doc = r##"unterminated')

    def test_non_path_exists_method_is_accepted(self):
        source = (
            "struct Inventory;\n"
            "impl Inventory { fn exists(&self) -> bool { true } }\n"
            "fn f() { let inventory = Inventory; "
            "let _ = inventory.exists(); let _ = Inventory.exists(); }\n"
        )
        self.assertEqual(reasons(source), [])

    def test_std_alias_without_host_effect_is_accepted(self):
        source = "use std as host;\nfn f() { let _ = host::fmt::Error; }\n"
        self.assertEqual(reasons(source), [])

    def test_inline_test_fixture_is_not_production_but_next_item_is(self):
        source = '''
        #[cfg(test)]
        mod tests {
            fn fixture() { std::fs::write("fixture", b"x"); }
            mod nested { fn more() { std::process::Command::new("echo"); } }
        }
        fn production() { std::fs::read("secret"); }
        '''
        hits = reasons(source)
        self.assertEqual(len(hits), 1, hits)
        self.assertIn("std::fs::", hits[0])

    def test_unbalanced_inline_test_module_fails_closed(self):
        with self.assertRaises(guard.SourceGuardError):
            reasons('#[cfg(test)] mod tests { fn fixture() { std::fs::read("x"); }')


class ClassifySourceDirectFormsTests(unittest.TestCase):
    """Fully qualified direct-access forms."""

    def test_direct_std_fs_call(self):
        hits = reasons("let text = std::fs::read_to_string(path)?;")
        self.assertTrue(any("std::fs::" in reason for reason in hits), hits)

    def test_direct_host_clock_reads(self):
        for source in (
            "let now = std::time::SystemTime::now();",
            "let now = SystemTime::now();",
            "let now = std::time::Instant :: now();",
            "let origin = *ORIGIN.get_or_init(Instant::now);",
        ):
            hits = reasons(source)
            self.assertTrue(
                any("direct host clock read" in reason for reason in hits),
                (source, hits),
            )

    def test_pure_duration_use_is_accepted(self):
        source = """
        use std::time::Duration;
        // Instant::now() is read only by the adapter.
        fn whole_seconds(distance: Duration) -> u64 { distance.as_secs() }
        """
        self.assertEqual(reasons(source), [])

    def test_direct_std_process_command(self):
        hits = reasons('let out = std::process::Command::new("echo").output()?;')
        self.assertTrue(any("process" in reason for reason in hits), hits)

    def test_direct_std_path_path_exists(self):
        hits = reasons("let ok = std::path::Path::new(&p).exists();")
        self.assertTrue(any("filesystem" in reason for reason in hits), hits)

    def test_method_form_exists_call(self):
        hits = reasons("fn f(candidate: &Path) { let ok = candidate.exists(); }")
        self.assertTrue(any("method-form" in reason for reason in hits), hits)

    def test_method_form_try_exists_call(self):
        hits = reasons("fn f(candidate: &Path) { let ok = candidate.try_exists()?; }")
        self.assertTrue(any("method-form" in reason for reason in hits), hits)

    def test_pathbuf_alias_and_inferred_receiver_are_guarded(self):
        source = (
            "use std::path::PathBuf as P;\n"
            'fn f() { let candidate = P::from("path"); candidate.exists(); }\n'
        )
        hits = reasons(source)
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

    def test_raw_literal_embedded_quote_does_not_hide_real_host_read(self):
        source = (
            'fn leak() { let _doc = r#"quote " end"#; '
            'let path = Path::new("/etc/hosts"); '
            'let contents = std::fs::read(path).unwrap(); }\n'
        )
        hits = reasons(source)
        self.assertTrue(any("std::fs::" in reason for reason in hits), hits)

    def test_byte_raw_literal_before_host_read_does_not_hide_call(self):
        source = 'let _doc = br##"quote " end"##; std::fs::read(path);'
        hits = reasons(source)
        self.assertTrue(any("std::fs::" in reason for reason in hits), hits)

    def test_platform_specific_filesystem_modules_are_guarded(self):
        source = (
            '#[cfg(unix)] fn u() { std::os::unix::fs::symlink("a", "b"); }\n'
            '#[cfg(windows)] fn w() { std::os::windows::fs::symlink_file("a", "b"); }\n'
        )
        hits = reasons(source)
        self.assertEqual(
            sum("platform filesystem" in reason for reason in hits), 2, hits
        )


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

    def test_aliased_std_root_cannot_hide_filesystem_or_process_calls(self):
        source = (
            "use std as host;\n"
            'fn f() { host::fs::write("a", b"x"); '
            'host::process::Command::new("echo").output(); }\n'
        )
        hits = reasons(source)
        self.assertTrue(any("std::fs::" in reason for reason in hits), hits)
        self.assertTrue(any("std::process" in reason for reason in hits), hits)

    def test_aliased_platform_filesystem_module_is_guarded(self):
        source = (
            "use std::os::unix::fs as osfs;\n"
            'fn f() { osfs::symlink("a", "b"); }\n'
        )
        hits = reasons(source)
        self.assertTrue(any("platform filesystem" in reason for reason in hits), hits)
        self.assertTrue(
            any("filesystem-module alias" in reason for reason in hits), hits
        )

    def test_aliased_platform_filesystem_item_is_guarded(self):
        source = (
            "use std::os::windows::fs::symlink_file as link;\n"
            'fn f() { link("a", "b"); }\n'
        )
        hits = reasons(source)
        self.assertTrue(
            any("directly imported filesystem function" in reason for reason in hits),
            hits,
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
        source = "fn f(candidate: &Path) { let ok = candidate.\n    exists(); }\n"
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
