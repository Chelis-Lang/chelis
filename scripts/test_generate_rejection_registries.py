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
    RegistryError,
    derive_issue_numbers,
    discover_atoms,
    discover_issue_citations,
    discover_production_sources,
    load_issue_manifest,
    manifest_derivation_problems,
    parse_issue_citations,
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
    def test_only_crate_src_trees_are_production_and_macro_owner_is_excluded(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            included = (
                root / "crates/example/src/lib.rs",
                root / "crates/example/src/nested/module.rs",
                root / "crates/other/src/main.rs",
            )
            excluded = (
                root / "crates/example/tests/integration.rs",
                root / "crates/example/examples/demo.rs",
                root / "crates/example/benches/bench.rs",
                root / "crates/example/build.rs",
                root / "crates/chelis-types/src/unsupported.rs",
                root / "scripts/not_production.rs",
            )
            for path in included + excluded:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("// fixture\n")

            self.assertEqual(discover_production_sources(root), list(included))


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
