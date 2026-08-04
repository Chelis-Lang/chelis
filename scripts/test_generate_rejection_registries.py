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
    discover_issue_authorities as discover_issue_authorities_from_manifest,
    load_issue_manifest,
    render_issue_manifest,
    render_registry,
)


ROOT = Path(__file__).resolve().parent.parent


def discover_issue_authorities(root: Path) -> dict[int, list[AuthoritySite]]:
    """Give compact synthetic crate fixtures ordinary workspace manifests."""
    workspace_manifest = root / "Cargo.toml"
    if not workspace_manifest.exists():
        crates = root / "crates"
        members = sorted(path for path in crates.glob("*") if path.is_dir())
        if members:
            rendered_members = ", ".join(
                json.dumps(path.relative_to(root).as_posix()) for path in members
            )
            workspace_manifest.write_text(
                f"[workspace]\nmembers = [{rendered_members}]\nresolver = \"3\"\n"
            )
            for member in members:
                manifest = member / "Cargo.toml"
                if not manifest.exists():
                    manifest.write_text(
                        "[package]\n"
                        f'name = "{member.name}"\n'
                        'version = "0.0.0"\n'
                        'edition = "2024"\n'
                    )
                manifest_text = manifest.read_text()
                if (
                    not (member / "src/lib.rs").is_file()
                    and not (member / "src/main.rs").is_file()
                    and "[lib]" not in manifest_text
                    and "[[bin]]" not in manifest_text
                ):
                    default_source = member / "src/lib.rs"
                    default_source.parent.mkdir(exist_ok=True)
                    default_source.write_text("pub fn fixture_target() {}\n")
    return discover_issue_authorities_from_manifest(root)


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
    def test_missing_workspace_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(RegistryError, "missing workspace manifest"):
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

    def test_rust_character_literals_cannot_mask_a_later_authority(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                """const QUOTE: char = '"';
const APOSTROPHE: char = '\\'';
let _ = unimplemented_rejection!(714, "live");
"""
            )
            self.assertEqual(
                discover_issue_authorities(root),
                {714: [AuthoritySite("crates/example/src/lib.rs", 3)]},
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

    def test_macro_owner_file_still_inventories_real_construction_sites(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/chelis-types/src/unsupported.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                "#[macro_export]\n"
                "macro_rules! unimplemented_rejection {\n"
                "    ($issue:literal) => {{ helper($issue) }};\n"
                "}\n"
                "fn live() {\n"
                "    let _ = crate::unimplemented_rejection!(999);\n"
                "}\n"
            )
            self.assertEqual(
                discover_issue_authorities(root),
                {
                    999: [
                        AuthoritySite(
                            "crates/chelis-types/src/unsupported.rs",
                            6,
                        )
                    ]
                },
            )

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

    def test_build_script_module_edge_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            build = root / "crates/example/build.rs"
            build.parent.mkdir(parents=True)
            build.write_text("mod helper;\nfn main() {}\n")
            helper = root / "crates/example/helper.rs"
            helper.write_text(
                "let _ = chelis_types::unimplemented_rejection!(999, \"hidden\");\n"
            )
            with self.assertRaisesRegex(RegistryError, "build-script module edge"):
                discover_issue_authorities(root)

    def test_build_script_raw_and_unicode_module_edges_are_rejected(self) -> None:
        for module in ("r#type", "δοκιμή"):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                build = root / "crates/example/build.rs"
                build.parent.mkdir(parents=True)
                build.write_text(f"mod {module};\nfn main() {{}}\n")
                with self.assertRaisesRegex(
                    RegistryError, "build-script module edge"
                ):
                    discover_issue_authorities(root)

    def test_source_tree_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            src = root / "crates/example/src"
            src.mkdir(parents=True)
            (src / "lib.rs").write_text("mod generated;\n")
            generated = root / "generated"
            generated.mkdir()
            (generated / "mod.rs").write_text(
                "let _ = chelis_types::unimplemented_rejection!(999, \"hidden\");\n"
            )
            (src / "generated").symlink_to(generated, target_is_directory=True)
            with self.assertRaisesRegex(RegistryError, "source symlink"):
                discover_issue_authorities(root)

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
                '#[cfg_attr(all(feature = "generated", any(unix, windows)), '
                'path = "../generated.rs")]\n'
                "mod generated;\n"
            )
            with self.assertRaisesRegex(RegistryError, "production path attribute"):
                discover_issue_authorities(root)

    def test_raw_builtin_path_spellings_are_rejected(self) -> None:
        spellings = (
            '#[r#path = "../generated.rs"]\nmod generated;\n',
            '#[cfg_attr(all(), r#path = "../generated.rs")]\nmod generated;\n',
            '#[r#cfg_attr(all(), path = "../generated.rs")]\nmod generated;\n',
        )
        for spelling in spellings:
            with self.subTest(spelling=spelling), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                source = root / "crates/example/src/lib.rs"
                source.parent.mkdir(parents=True)
                source.write_text(spelling)
                with self.assertRaisesRegex(
                    RegistryError, "production path attribute"
                ):
                    discover_issue_authorities(root)

    def test_path_text_outside_an_attribute_is_not_a_source_edge(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "crates/example/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                '// #[path = "comment.rs"]\n'
                'const TEXT: &str = r##"#[path = "raw.rs"]"##;\n'
                '#[some_tool(metadata(path = "not-a-module.rs"))]\n'
                "let path = 1;\n"
            )
            self.assertEqual(discover_issue_authorities(root), {})

    def test_custom_workspace_lib_bin_and_build_roots_are_inventoried(self) -> None:
        cases = (
            (
                "lib",
                "bindings/rust/lib.rs",
                '[lib]\npath = "bindings/rust/lib.rs"\n',
                "",
            ),
            (
                "bin",
                "tools/runner.rs",
                '[[bin]]\nname = "runner"\npath = "tools/runner.rs"\n',
                "",
            ),
            (
                "build",
                "support/custom_build.rs",
                "",
                'build = "support/custom_build.rs"\n',
            ),
        )
        for kind, source_rel, targets, package_extra in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                member = root / "tree-sitter-style"
                member.mkdir()
                (root / "Cargo.toml").write_text(
                    '[workspace]\nmembers = ["tree-sitter-style"]\nresolver = "3"\n'
                )
                (member / "Cargo.toml").write_text(
                    "[package]\n"
                    'name = "tree-sitter-style"\n'
                    'version = "0.0.0"\n'
                    'edition = "2024"\n'
                    f"{package_extra}\n"
                    f"{targets}"
                )
                source = member / source_rel
                source.parent.mkdir(parents=True)
                source.write_text(
                    "let _ = chelis_types::unimplemented_rejection!(999, \"live\");\n"
                )
                if kind == "build":
                    default_source = member / "src/lib.rs"
                    default_source.parent.mkdir()
                    default_source.write_text("pub fn fixture_target() {}\n")
                self.assertEqual(
                    discover_issue_authorities(root),
                    {
                        999: [
                            AuthoritySite(
                                f"tree-sitter-style/{source_rel}",
                                1,
                            )
                        ]
                    },
                )

    def test_root_package_is_an_implicit_workspace_member(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            member = root / "crates/member"
            member.mkdir(parents=True)
            (root / "Cargo.toml").write_text(
                "[package]\n"
                'name = "root-package"\n'
                'version = "0.0.0"\n'
                'edition = "2024"\n\n'
                "[workspace]\n"
                'members = ["crates/member"]\n'
                'resolver = "3"\n'
            )
            root_source = root / "src/lib.rs"
            root_source.parent.mkdir()
            root_source.write_text(
                'let _ = unimplemented_rejection!(997, "root member");\n'
            )
            (member / "Cargo.toml").write_text(
                "[package]\n"
                'name = "member"\n'
                'version = "0.0.0"\n'
                'edition = "2024"\n'
            )
            member_source = member / "src/lib.rs"
            member_source.parent.mkdir()
            member_source.write_text("pub fn member() {}\n")

            self.assertEqual(
                discover_issue_authorities(root),
                {997: [AuthoritySite("src/lib.rs", 1)]},
            )

    def test_in_tree_path_dependency_is_an_implicit_workspace_member(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            app = root / "crates/app"
            hidden = root / "support/hidden"
            app.mkdir(parents=True)
            hidden.mkdir(parents=True)
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["crates/app"]\nresolver = "3"\n'
            )
            (app / "Cargo.toml").write_text(
                "[package]\n"
                'name = "app"\n'
                'version = "0.0.0"\n'
                'edition = "2024"\n\n'
                "[dependencies]\n"
                'hidden = { path = "../../support/hidden" }\n'
            )
            app_source = app / "src/lib.rs"
            app_source.parent.mkdir()
            app_source.write_text("pub fn app() {}\n")
            (hidden / "Cargo.toml").write_text(
                "[package]\n"
                'name = "hidden"\n'
                'version = "0.0.0"\n'
                'edition = "2024"\n'
            )
            hidden_source = hidden / "src/lib.rs"
            hidden_source.parent.mkdir()
            hidden_source.write_text(
                'let _ = unimplemented_rejection!(998, "path member");\n'
            )

            self.assertEqual(
                discover_issue_authorities(root),
                {998: [AuthoritySite("support/hidden/src/lib.rs", 1)]},
            )

    def test_proc_macro_workspace_target_is_rejected_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            member = root / "crates/macros"
            member.mkdir(parents=True)
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["crates/macros"]\nresolver = "3"\n'
            )
            (member / "Cargo.toml").write_text(
                "[package]\n"
                'name = "macros"\n'
                'version = "0.0.0"\n'
                'edition = "2024"\n\n'
                "[lib]\nproc-macro = true\n"
            )
            source = member / "src/lib.rs"
            source.parent.mkdir()
            source.write_text(
                'const GENERATED: &str = "unimplemented_rejection!(999, hidden)";\n'
            )

            with self.assertRaisesRegex(RegistryError, "proc-macro"):
                discover_issue_authorities(root)

    def test_workspace_member_intermediate_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            real_parent = root / "real"
            member = real_parent / "member"
            member.mkdir(parents=True)
            (root / "linked").symlink_to(real_parent, target_is_directory=True)
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["linked/member"]\nresolver = "3"\n'
            )
            (member / "Cargo.toml").write_text(
                "[package]\n"
                'name = "member"\n'
                'version = "0.0.0"\n'
                'edition = "2024"\n'
            )
            source = member / "src/lib.rs"
            source.parent.mkdir()
            source.write_text("pub fn member() {}\n")

            with self.assertRaisesRegex(RegistryError, "workspace member symlink"):
                discover_issue_authorities(root)

    def test_workspace_exclude_is_not_over_inventoried(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            included = root / "crates/included"
            excluded = root / "crates/excluded"
            for member in (included, excluded):
                member.mkdir(parents=True)
                (member / "Cargo.toml").write_text(
                    "[package]\n"
                    f'name = "{member.name}"\n'
                    'version = "0.0.0"\n'
                    'edition = "2024"\n'
                )
                source = member / "src/lib.rs"
                source.parent.mkdir()
                source.write_text("pub fn target() {}\n")
            (root / "Cargo.toml").write_text(
                "[workspace]\n"
                'members = ["crates/*"]\n'
                'exclude = ["crates/excluded"]\n'
                'resolver = "3"\n'
            )
            (excluded / "src/lib.rs").write_text(
                'let _ = unimplemented_rejection!(996, "not a member");\n'
            )

            self.assertEqual(discover_issue_authorities(root), {})

    def test_repository_local_non_workspace_dependency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            app = root / "app"
            rogue = root / "rogue"
            for member in (app, rogue):
                member.mkdir()
                (member / "Cargo.toml").write_text(
                    "[package]\n"
                    f'name = "{member.name}"\n'
                    'version = "0.0.0"\n'
                    'edition = "2024"\n'
                )
                source = member / "src/lib.rs"
                source.parent.mkdir()
                source.write_text("pub fn target() {}\n")
            (root / "Cargo.toml").write_text(
                "[workspace]\n"
                'members = ["app"]\n'
                'exclude = ["rogue"]\n'
                'resolver = "3"\n'
            )
            (app / "Cargo.toml").write_text(
                "[package]\n"
                'name = "app"\n'
                'version = "0.0.0"\n'
                'edition = "2024"\n\n'
                "[dependencies]\n"
                'rogue = { path = "../rogue" }\n'
            )
            (rogue / "src/lib.rs").write_text(
                'let _ = unimplemented_rejection!(999999, "hidden");\n'
            )

            with self.assertRaisesRegex(
                RegistryError, "repository-local path dependency.*workspace member"
            ):
                discover_issue_authorities(root)

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
