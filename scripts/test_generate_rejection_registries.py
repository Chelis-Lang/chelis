#!/usr/bin/env python3
"""Tests for the [05-UNS-5] atom/issue registry generator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generate_rejection_registries import (
    AuthoritySite,
    MANIFEST_REL,
    OUTPUT_REL,
    RegistryError,
    discover_atoms,
    discover_issue_authorities,
    load_issue_manifest,
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


class IssueManifest(unittest.TestCase):
    def test_missing_crates_root_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(RegistryError, "missing crates source root"):
                discover_issue_authorities(Path(raw))

    def write_manifest(self, root: Path, issues: list[dict]) -> Path:
        path = root / "issues.json"
        path.write_text(json.dumps({"schema": 2, "issues": issues}))
        return path

    def test_accepts_only_sorted_unique_open_issue_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = self.write_manifest(
                root,
                [
                    {
                        "number": 705,
                        "kind": "issue",
                        "state": "open",
                        "sites": [{"path": "crates/a/src/lib.rs", "line": 1}],
                    },
                    {
                        "number": 879,
                        "kind": "issue",
                        "state": "open",
                        "sites": [{"path": "crates/b/src/lib.rs", "line": 2}],
                    },
                ],
            )
            self.assertEqual(load_issue_manifest(path), [705, 879])

    def test_rejects_zero_duplicate_pr_closed_and_unsorted_rows(self) -> None:
        invalid = [
            [{"number": 0, "kind": "issue", "state": "open", "sites": []}],
            [
                {"number": 705, "kind": "issue", "state": "open", "sites": [{"path": "crates/a/src/lib.rs", "line": 1}]},
                {"number": 705, "kind": "issue", "state": "open", "sites": [{"path": "crates/a/src/lib.rs", "line": 2}]},
            ],
            [{"number": 705, "kind": "pull_request", "state": "open", "sites": [{"path": "crates/a/src/lib.rs", "line": 1}]}],
            [{"number": 705, "kind": "issue", "state": "closed", "sites": [{"path": "crates/a/src/lib.rs", "line": 1}]}],
            [
                {"number": 879, "kind": "issue", "state": "open", "sites": [{"path": "crates/b/src/lib.rs", "line": 1}]},
                {"number": 705, "kind": "issue", "state": "open", "sites": [{"path": "crates/a/src/lib.rs", "line": 1}]},
            ],
            [{"number": 705, "kind": "issue", "state": "open", "sites": []}],
        ]
        for index, rows in enumerate(invalid):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as raw:
                path = self.write_manifest(Path(raw), rows)
                with self.assertRaises(RegistryError):
                    load_issue_manifest(path)

    def test_discovers_production_construction_sites_with_exact_lines(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                "fn first() {\n"
                "    let _ = unimplemented_rejection!(714, \"first\");\n"
                "}\n"
                "fn second() {\n"
                "    let _ = chelis_types::unimplemented_rejection!(714, \"second\");\n"
                "}\n"
            )
            test_source = root / "crates/example/tests/probe.rs"
            test_source.parent.mkdir(parents=True)
            test_source.write_text(
                "let _ = unimplemented_rejection!(999, \"test-only\");\n"
            )

            self.assertEqual(
                discover_issue_authorities(root),
                {
                    714: [
                        AuthoritySite("crates/example/src/lib.rs", 2),
                        AuthoritySite("crates/example/src/lib.rs", 5),
                    ]
                },
            )

    def test_discovery_ignores_comments_and_strings_but_allows_inner_comments(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                "// unimplemented_rejection!(600, \"comment\")\n"
                "const TEXT: &str = r#\"unimplemented_rejection!(689, \\\"raw\\\")\"#;\n"
                "/* nested /* unimplemented_rejection!(691, \"block\") */ done */\n"
                "fn live() {\n"
                "    let _ = unimplemented_rejection!(/* reviewed */ 714, \"live\");\n"
                "}\n"
            )
            self.assertEqual(
                discover_issue_authorities(root),
                {714: [AuthoritySite("crates/example/src/lib.rs", 5)]},
            )

    def test_comments_and_whitespace_around_macro_tokens_cannot_hide_a_site(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                "fn live() {\n"
                "    let _ = chelis_types::unimplemented_rejection /* gap */ !\n"
                "        /* gap */ (714, \"live\");\n"
                "}\n"
            )
            self.assertEqual(
                discover_issue_authorities(root),
                {714: [AuthoritySite("crates/example/src/lib.rs", 2)]},
            )

    def test_macro_alias_is_rejected_instead_of_silently_uninventoried(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                "use chelis_types::unimplemented_rejection as pending;\n"
                "fn live() { let _ = pending!(714, \"live\"); }\n"
            )
            with self.assertRaisesRegex(RegistryError, "noncanonical"):
                discover_issue_authorities(root)

    def test_build_script_is_a_production_authority_owner(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/build.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                "fn main() {\n"
                "    let _ = chelis_types::unimplemented_rejection!(714, \"live\");\n"
                "}\n"
            )
            self.assertEqual(
                discover_issue_authorities(root),
                {714: [AuthoritySite("crates/example/build.rs", 2)]},
            )

    def test_src_test_named_module_is_conservatively_inventoried(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/tests/probe.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                "let _ = chelis_types::unimplemented_rejection!(999, \"test-only\");\n"
            )
            self.assertEqual(
                discover_issue_authorities(root),
                {
                    999: [
                        AuthoritySite("crates/example/src/tests/probe.rs", 1),
                    ]
                },
            )

    def test_production_rust_include_is_rejected_as_an_uninventoried_edge(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text('include! /* gap */ ("../generated.rs");\n')
            included = root / "crates/example/generated.rs"
            included.write_text(
                "let _ = chelis_types::unimplemented_rejection!(999, \"hidden\");\n"
            )
            with self.assertRaisesRegex(RegistryError, "production include!"):
                discover_issue_authorities(root)

    def test_data_include_macros_do_not_create_rust_source_edges(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                'const TEXT: &str = include_str!("data.txt");\n'
                'const BYTES: &[u8] = include_bytes!("data.bin");\n'
            )
            self.assertEqual(discover_issue_authorities(root), {})

    def test_production_path_attribute_is_rejected_as_an_uninventoried_edge(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text('#[path /* gap */ = "../generated.rs"]\nmod generated;\n')
            included = root / "crates/example/generated.rs"
            included.write_text(
                "let _ = chelis_types::unimplemented_rejection!(999, \"hidden\");\n"
            )
            with self.assertRaisesRegex(RegistryError, "production path attribute"):
                discover_issue_authorities(root)

    def test_cfg_attr_path_is_also_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                '#[cfg_attr(feature = "generated", path = "../generated.rs")]\n'
                "mod generated;\n"
            )
            with self.assertRaisesRegex(RegistryError, "production path attribute"):
                discover_issue_authorities(root)

    def test_path_text_outside_an_attribute_is_not_a_source_edge(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                '// #[path = "comment.rs"]\n'
                'const TEXT: &str = r##"#[path = "raw.rs"]"##;\n'
                "let path = 1;\n"
            )
            self.assertEqual(discover_issue_authorities(root), {})

    def test_rendered_manifest_carries_sites_and_is_source_derived(self) -> None:
        rendered = render_issue_manifest(
            {
                714: [AuthoritySite("crates/backend/src/emit.rs", 17)],
                879: [AuthoritySite("crates/backend/src/host.rs", 23)],
            }
        )
        payload = json.loads(rendered)
        self.assertEqual(payload["schema"], 2)
        self.assertEqual(
            payload["issues"][0]["sites"],
            [{"path": "crates/backend/src/emit.rs", "line": 17}],
        )

    def test_checked_in_manifest_matches_production_construction_sites(self) -> None:
        self.assertEqual(
            (ROOT / MANIFEST_REL).read_text(),
            render_issue_manifest(discover_issue_authorities(ROOT)),
        )


class RenderRegistry(unittest.TestCase):
    def test_render_is_stable_and_keeps_atom_and_issue_types_separate(self) -> None:
        rendered = render_registry(["[04-NUM-1]", "[05-UNS-1]"], [705, 879])
        self.assertIn('"[04-NUM-1]"', rendered)
        self.assertIn('"[05-UNS-1]"', rendered)
        self.assertIn("705", rendered)
        self.assertIn("879", rendered)
        self.assertEqual(rendered, render_registry(["[04-NUM-1]", "[05-UNS-1]"], [705, 879]))

    def test_checked_in_registry_is_byte_identical_to_its_sources(self) -> None:
        authorities = discover_issue_authorities(ROOT)
        expected = render_registry(
            discover_atoms(ROOT / "spec"),
            sorted(authorities),
        )
        self.assertEqual((ROOT / OUTPUT_REL).read_text(), expected)


if __name__ == "__main__":
    unittest.main()
