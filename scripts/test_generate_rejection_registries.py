#!/usr/bin/env python3
"""Tests for the [05-UNS-5] atom/issue registry generator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generate_rejection_registries import (
    MANIFEST_REL,
    OUTPUT_REL,
    ProductionWorkspace,
    RegistryError,
    compiler_source_closure_problems,
    derive_issue_numbers,
    discover_atoms,
    discover_issue_citations,
    discover_production_sources,
    load_issue_manifest,
    manifest_derivation_problems,
    parse_issue_citations,
    production_dep_info_files,
    production_target_roots_from_metadata,
    production_workspace_from_metadata,
    render_issue_manifest,
    render_registry,
)

ROOT = Path(__file__).resolve().parent.parent


class DiscoverAtoms(unittest.TestCase):
    def test_reads_only_normative_blockquote_atoms_and_sorts_them(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            spec = Path(raw)
            (spec / "05-risc-primitives.md").write_text(
                "cross reference [05-UNS-9]\n"
                "> **[05-UNS-2]** second\n"
                "> **[05-UNS-1]** first\n"
            )
            self.assertEqual(
                discover_atoms(spec), ["[05-UNS-1]", "[05-UNS-2]"]
            )

    def test_duplicate_normative_atom_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            spec = Path(raw)
            (spec / "04-type-system.md").write_text(
                "> **[04-NUM-1]** one\n> **[04-NUM-1]** duplicate\n"
            )
            with self.assertRaises(RegistryError):
                discover_atoms(spec)


class ProductionSources(unittest.TestCase):
    def test_cargo_workspace_includes_repository_members_outside_crates_and_excludes_nonproduction_targets(
        self,
    ) -> None:
        root = Path("/repo")
        metadata = {
            "workspace_members": ["example 0.1.0", "tree-sitter-chelis 0.1.0"],
            "packages": [
                {
                    "id": "example 0.1.0",
                    "manifest_path": "/repo/crates/example/Cargo.toml",
                    "targets": [
                        {
                            "kind": ["lib"],
                            "src_path": "/repo/crates/example/src/lib.rs",
                        },
                        {
                            "kind": ["bin"],
                            "src_path": "/repo/crates/example/src/main.rs",
                        },
                        {
                            "kind": ["test"],
                            "src_path": "/repo/crates/example/tests/integration.rs",
                        },
                        {
                            "kind": ["example"],
                            "src_path": "/repo/crates/example/examples/demo.rs",
                        },
                        {
                            "kind": ["bench"],
                            "src_path": "/repo/crates/example/benches/bench.rs",
                        },
                        {
                            "kind": ["custom-build"],
                            "src_path": "/repo/crates/example/build.rs",
                        },
                    ],
                },
                {
                    "id": "tree-sitter-chelis 0.1.0",
                    "manifest_path": "/repo/tree-sitter-chelis/Cargo.toml",
                    "targets": [
                        {
                            "kind": ["lib"],
                            "src_path": "/repo/tree-sitter-chelis/src/lib.rs",
                        }
                    ],
                },
            ],
        }
        workspace = production_workspace_from_metadata(root, metadata)
        self.assertEqual(
            workspace,
            ProductionWorkspace(
                package_ids=frozenset(
                    {"example 0.1.0", "tree-sitter-chelis 0.1.0"}
                ),
                target_roots=(
                    "crates/example/src/lib.rs",
                    "crates/example/src/main.rs",
                    "tree-sitter-chelis/src/lib.rs",
                ),
            ),
        )
        self.assertEqual(
            production_target_roots_from_metadata(root, metadata),
            [
                "crates/example/src/lib.rs",
                "crates/example/src/main.rs",
                "tree-sitter-chelis/src/lib.rs",
            ],
        )

    def test_dep_info_selection_uses_the_same_outside_crates_package_set(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            selected_artifact = root / "target/debug/libtree_sitter_chelis.rlib"
            selected_dep_info = root / "target/debug/tree_sitter_chelis.d"
            selected_dep_info.parent.mkdir(parents=True)
            selected_dep_info.write_text("target: tree-sitter-chelis/src/lib.rs\n")
            ignored_artifact = root / "target/debug/libexternal.rlib"
            ignored_dep_info = root / "target/debug/external.d"
            ignored_dep_info.write_text("target: external/src/lib.rs\n")
            workspace = ProductionWorkspace(
                package_ids=frozenset({"tree-sitter-chelis 0.1.0"}),
                target_roots=("tree-sitter-chelis/src/lib.rs",),
            )
            messages = [
                {
                    "reason": "compiler-artifact",
                    "package_id": "tree-sitter-chelis 0.1.0",
                    "profile": {"test": False},
                    "target": {"name": "tree-sitter-chelis", "kind": ["lib"]},
                    "filenames": [str(selected_artifact)],
                },
                {
                    "reason": "compiler-artifact",
                    "package_id": "external 0.1.0",
                    "profile": {"test": False},
                    "target": {"name": "external", "kind": ["lib"]},
                    "filenames": [str(ignored_artifact)],
                },
            ]
            self.assertEqual(
                production_dep_info_files(messages, workspace),
                {selected_dep_info},
            )

    def test_real_production_graph_includes_path_reached_source_outside_src(
        self,
    ) -> None:
        paths = {source.path for source in discover_production_sources(ROOT)}
        self.assertIn(Path("tests/support/c_lexical.rs"), paths)
        self.assertIn(Path("tree-sitter-chelis/bindings/rust/lib.rs"), paths)
        self.assertNotIn(Path("crates/chelis-types/src/unsupported.rs"), paths)

    def test_compiler_source_missing_from_parser_graph_fails_closure(self) -> None:
        self.assertEqual(
            compiler_source_closure_problems(
                {"crates/example/src/lib.rs"},
                {
                    "crates/example/src/lib.rs",
                    "tests/support/compiler_reached.rs",
                    "crates/chelis-types/src/unsupported.rs",
                },
            ),
            ["tests/support/compiler_reached.rs"],
        )


class IssueCitations(unittest.TestCase):
    def test_exact_decimal_literals_are_discovered_and_duplicate_sites_collapse(
        self,
    ) -> None:
        citations = parse_issue_citations(
            Path("crates/example/src/lib.rs"),
            """
            let _ = chelis_types::unimplemented_rejection!(1_277, "one");
            let _ = unimplemented_rejection!(879, "two");
            let _ = unimplemented_rejection!(879, "three");
            """,
        )
        self.assertEqual(
            [citation.number for citation in citations], [1277, 879, 879]
        )
        self.assertEqual(derive_issue_numbers(citations), [879, 1277])

    def test_all_valid_positive_unsuffixed_decimal_spellings_are_accepted(
        self,
    ) -> None:
        spellings = ("879", "000879", "8_79", "8__79", "879_", "0_879")
        for spelling in spellings:
            with self.subTest(spelling=spelling):
                citations = parse_issue_citations(
                    Path("crates/example/src/lib.rs"),
                    f'unimplemented_rejection!({spelling}, "production");',
                )
                self.assertEqual([citation.number for citation in citations], [879])

    def test_comments_and_string_literals_are_not_citations(self) -> None:
        citations = parse_issue_citations(
            Path("crates/example/src/lib.rs"),
            r'''
            // unimplemented_rejection!(600, "line comment")
            /* unimplemented_rejection!(689, "block comment") */
            let _ = "unimplemented_rejection!(729, \"string\")";
            let _ = r#"unimplemented_rejection!(759, "raw string")"#;
            let _ = b"unimplemented_rejection!(829, \"byte string\")";
            let _ = br#"unimplemented_rejection!(912, "raw byte string")"#;
            let _ = unimplemented_rejection!(879, "production");
            ''',
        )
        self.assertEqual([citation.number for citation in citations], [879])

    def test_citations_after_character_literals_are_discovered(self) -> None:
        prefixes = (
            "let ordinary = 'x';",
            r"""let quote = '\"';""",
            r"""let apostrophe = '\'';""",
            r"""let newline = '\n';""",
            r"""let unicode = '\u{22}';""",
            """let crab = '🦀';""",
            """let byte = b'x';""",
            r"""let byte_quote = b'\"';""",
            r"""let byte_apostrophe = b'\'';""",
            r"""let byte_hex = b'\x22';""",
            """let lifetime: &'static str = "still a string";""",
        )
        for prefix in prefixes:
            with self.subTest(prefix=prefix):
                citations = parse_issue_citations(
                    Path("crates/example/src/lib.rs"),
                    prefix + '\nunimplemented_rejection!(879, "production");\n',
                )
                self.assertEqual(
                    [citation.number for citation in citations],
                    [879],
                )

    def test_renamed_macro_import_or_reexport_is_rejected(self) -> None:
        invalid = (
            "use chelis_types::unimplemented_rejection as reject;\n",
            "pub use chelis_types::{unimplemented_rejection as reject};\n",
            "use chelis_types::unimplemented_rejection /* hidden */ as reject;\n",
        )
        for source in invalid:
            with self.subTest(source=source), self.assertRaisesRegex(
                RegistryError, "must retain its exact spelling"
            ):
                parse_issue_citations(Path("crates/example/src/lib.rs"), source)

    def test_dynamic_and_malformed_issue_arguments_are_rejected(self) -> None:
        invalid = (
            'unimplemented_rejection!(issue, "dynamic")',
            'unimplemented_rejection!("879", "string")',
            'unimplemented_rejection!(0, "zero")',
            'unimplemented_rejection!(0_, "zero")',
            'unimplemented_rejection!(0__0, "zero")',
            'unimplemented_rejection!(879u32, "suffix")',
            'unimplemented_rejection!(879 + 1, "expression")',
            'unimplemented_rejection!(0x36f, "hex")',
            'unimplemented_rejection!(879)',
            'unimplemented_rejection! { 879, "brace delimiter" }',
        )
        for source in invalid:
            with self.subTest(source=source), self.assertRaises(RegistryError):
                parse_issue_citations(Path("crates/example/src/lib.rs"), source)

    def test_source_discovery_rejects_a_dynamic_production_citation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text('unimplemented_rejection!(issue, "dynamic");\n')
            with self.assertRaises(RegistryError):
                discover_issue_citations(root)


class IssueManifest(unittest.TestCase):
    def write_manifest(self, root: Path, issues: list[dict]) -> Path:
        path = root / "issues.json"
        path.write_text(json.dumps({"schema": 1, "issues": issues}))
        return path

    def test_accepts_only_sorted_unique_open_issue_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = self.write_manifest(
                root,
                [
                    {"number": 705, "kind": "issue", "state": "open"},
                    {"number": 879, "kind": "issue", "state": "open"},
                ],
            )
            self.assertEqual(load_issue_manifest(path), [705, 879])

    def test_rejects_zero_duplicate_pr_closed_and_unsorted_rows(self) -> None:
        invalid = [
            [{"number": 0, "kind": "issue", "state": "open"}],
            [
                {"number": 705, "kind": "issue", "state": "open"},
                {"number": 705, "kind": "issue", "state": "open"},
            ],
            [{"number": 705, "kind": "pull_request", "state": "open"}],
            [{"number": 705, "kind": "issue", "state": "closed"}],
            [
                {"number": 879, "kind": "issue", "state": "open"},
                {"number": 705, "kind": "issue", "state": "open"},
            ],
        ]
        for index, rows in enumerate(invalid):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as raw:
                path = self.write_manifest(Path(raw), rows)
                with self.assertRaises(RegistryError):
                    load_issue_manifest(path)

    def test_missing_and_uncited_stale_rows_are_both_reported(self) -> None:
        self.assertEqual(
            manifest_derivation_problems(
                manifest_issues=[600, 879],
                source_issues=[600, 912],
            ),
            [
                "issue manifest is missing source-cited rows: [912]",
                "issue manifest contains uncited stale rows: [879]",
            ],
        )


class RenderRegistry(unittest.TestCase):
    def test_issue_manifest_render_is_stable(self) -> None:
        rendered = render_issue_manifest([705, 879])
        self.assertEqual(rendered, render_issue_manifest([705, 879]))
        self.assertEqual(
            json.loads(rendered),
            {
                "schema": 1,
                "issues": [
                    {"number": 705, "kind": "issue", "state": "open"},
                    {"number": 879, "kind": "issue", "state": "open"},
                ],
            },
        )

    def test_render_is_stable_and_keeps_atom_and_issue_types_separate(self) -> None:
        rendered = render_registry(["[04-NUM-1]", "[05-UNS-1]"], [705, 879])
        self.assertIn('"[04-NUM-1]"', rendered)
        self.assertIn('"[05-UNS-1]"', rendered)
        self.assertIn("705", rendered)
        self.assertIn("879", rendered)
        self.assertEqual(rendered, render_registry(["[04-NUM-1]", "[05-UNS-1]"], [705, 879]))

    def test_checked_in_registry_is_byte_identical_to_its_sources(self) -> None:
        issues = derive_issue_numbers(discover_issue_citations(ROOT))
        manifest = ROOT / MANIFEST_REL
        self.assertEqual(load_issue_manifest(manifest), issues)
        self.assertEqual(manifest.read_text(), render_issue_manifest(issues))
        expected = render_registry(
            discover_atoms(ROOT / "spec"),
            issues,
        )
        self.assertEqual((ROOT / OUTPUT_REL).read_text(), expected)


if __name__ == "__main__":
    unittest.main()
