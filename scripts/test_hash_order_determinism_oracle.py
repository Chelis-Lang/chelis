from __future__ import annotations

import importlib.util
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("hash_order_determinism_oracle.py")
SPEC = importlib.util.spec_from_file_location("hash_order_determinism_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)


class HashOrderTokenTripwireTests(unittest.TestCase):
    def assert_rejected(self, path: str, text: str) -> None:
        with self.assertRaises(ORACLE.HashOrderDeterminismFailure):
            ORACLE.validate_sources(
                {path: text}, allowed=(), approved_build_scripts=()
            )

    def compiled_macro_path_sources(
        self,
        *,
        source: str,
        cfg: str,
        target: str = "src/generated/escape.rs",
    ) -> dict[str, str]:
        raw_root = tempfile.TemporaryDirectory()
        self.addCleanup(raw_root.cleanup)
        root = Path(raw_root.name)
        target_path = root / target
        target_path.parent.mkdir(parents=True)
        (root / ".gitignore").write_text("src/generated/\n", encoding="utf-8")
        (root / "src").mkdir(exist_ok=True)
        (root / "src/lib.rs").write_text(source, encoding="utf-8")
        target_path.write_text(
            "pub type Escape = std::collections::HashMap<u8, u8>;\n",
            encoding="utf-8",
        )
        subprocess.run(("git", "init", "-q"), cwd=root, check=True)
        subprocess.run(("git", "add", ".gitignore", "src/lib.rs"), cwd=root, check=True)
        subprocess.run(
            (
                "rustc",
                "--crate-type",
                "lib",
                "--cfg",
                cfg,
                "src/lib.rs",
                "-o",
                str(root / "probe.rlib"),
            ),
            cwd=root,
            check=True,
            capture_output=True,
        )
        return ORACLE.tracked_rust_sources(repo_root=root)

    def test_rejects_library_target(self) -> None:
        self.assert_rejected("crates/demo/src/lib.rs", "type M = HashMap<String, i32>;")

    def test_rejects_non_default_feature_source(self) -> None:
        self.assert_rejected(
            "crates/demo/src/checkpoint_compile_probe.rs",
            "use std::collections::HashSet;",
        )

    def test_rejects_build_script(self) -> None:
        self.assert_rejected("crates/demo/build.rs", "use hashbrown::HashMap;")

    def test_rejects_test_target(self) -> None:
        self.assert_rejected("crates/demo/tests/order.rs", "FxHashMap::default();")

    def test_rejects_unlisted_disallowed_type_allowance(self) -> None:
        self.assert_rejected(
            "crates/demo/src/lib.rs",
            "#[allow(clippy::disallowed_types)]\nstruct Escape;",
        )

    def test_rejects_stale_allow_list_entry(self) -> None:
        allowed = (
            ORACLE.AllowedHit(
                "crates/demo/src/lib.rs",
                "type Missing",
                "HashMap",
                1,
                "test-only stale entry",
            ),
        )
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "stale allow-list entry"
        ):
            ORACLE.validate_sources(
                {"crates/demo/src/lib.rs": "fn clean() {}"},
                allowed,
                approved_build_scripts=(),
            )

    def test_allow_list_is_exact_about_token_spelling(self) -> None:
        allowed = (
            ORACLE.AllowedHit(
                "crates/demo/src/lib.rs",
                "fn allowed",
                "HashMap",
                1,
                "negative-control fixture",
            ),
        )
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "unlisted token 'HashSet'"
        ):
            ORACLE.validate_sources(
                {
                    "crates/demo/src/lib.rs": (
                        "fn allowed() {\n"
                        "    let _: Option<HashMap<u8, u8>> = None;\n"
                        "    let _: Option<HashSet<u8>> = None;\n"
                        "}\n"
                    )
                },
                allowed,
                approved_build_scripts=(),
            )

    def test_allow_list_is_exact_about_token_cardinality(self) -> None:
        allowed = (
            ORACLE.AllowedHit(
                "crates/demo/src/lib.rs",
                "fn allowed",
                "HashMap",
                1,
                "negative-control fixture",
            ),
        )
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "token count"
        ):
            ORACLE.validate_sources(
                {
                    "crates/demo/src/lib.rs": (
                        "fn allowed() {\n"
                        "    let _: Option<HashMap<u8, u8>> = None;\n"
                        "    let _: Option<HashMap<u16, u16>> = None;\n"
                        "}\n"
                    )
                },
                allowed,
                approved_build_scripts=(),
            )

    def test_rejects_unsupported_path_bearing_attribute(self) -> None:
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "unsupported path-bearing attribute"
        ):
            ORACLE.validate_sources(
                {
                    "crates/demo/src/lib.rs": (
                        '#[cfg_attr(feature = "escape", path = "ignored.rs")]\n'
                        "mod ignored;\n"
                    )
                },
                allowed=(),
                approved_build_scripts=(),
            )

    def test_ignored_path_module_is_compiled_and_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "src/generated").mkdir(parents=True)
            (root / ".gitignore").write_text("src/generated/\n", encoding="utf-8")
            (root / "src/lib.rs").write_text(
                "#[cfg(round3_escape)]\n"
                '#[path = "generated/escape.rs"]\n'
                "mod escape;\n",
                encoding="utf-8",
            )
            (root / "src/generated/escape.rs").write_text(
                "pub type Escape = std::collections::HashMap<u8, u8>;\n",
                encoding="utf-8",
            )
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            subprocess.run(
                ("git", "add", ".gitignore", "src/lib.rs"), cwd=root, check=True
            )
            subprocess.run(
                (
                    "rustc",
                    "--crate-type",
                    "lib",
                    "--cfg",
                    "round3_escape",
                    "src/lib.rs",
                    "-o",
                    str(root / "probe.rlib"),
                ),
                cwd=root,
                check=True,
                capture_output=True,
            )

            sources = ORACLE.tracked_rust_sources(repo_root=root)
            self.assertIn("src/generated/escape.rs", sources)
            with self.assertRaisesRegex(
                ORACLE.HashOrderDeterminismFailure, "unlisted token 'HashMap'"
            ):
                ORACLE.validate_sources(
                    sources, allowed=(), approved_build_scripts=()
                )

    def test_rejects_macro_generated_ignored_path_module(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "src/generated").mkdir(parents=True)
            (root / ".gitignore").write_text("src/generated/\n", encoding="utf-8")
            (root / "src/lib.rs").write_text(
                "macro_rules! load_path_module {\n"
                "    ($attribute:ident, $target:literal) => {\n"
                "        #[$attribute = $target]\n"
                "        mod escaped;\n"
                "    };\n"
                "}\n"
                "#[cfg(round4_escape)]\n"
                'load_path_module!(path, "generated/escape.rs");\n',
                encoding="utf-8",
            )
            (root / "src/generated/escape.rs").write_text(
                "pub type Escape = std::collections::HashMap<u8, u8>;\n",
                encoding="utf-8",
            )
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            subprocess.run(
                ("git", "add", ".gitignore", "src/lib.rs"), cwd=root, check=True
            )
            subprocess.run(
                (
                    "rustc",
                    "--crate-type",
                    "lib",
                    "--cfg",
                    "round4_escape",
                    "src/lib.rs",
                    "-o",
                    str(root / "probe.rlib"),
                ),
                cwd=root,
                check=True,
                capture_output=True,
            )

            with self.assertRaisesRegex(
                ORACLE.HashOrderDeterminismFailure,
                "macro-generated external-module attribute",
            ):
                ORACLE.tracked_rust_sources(repo_root=root)

    def test_scans_grouped_macro_path_attribute_target(self) -> None:
        sources = self.compiled_macro_path_sources(
            cfg="round5_group_escape",
            source=(
                "macro_rules! load_grouped_path_module {\n"
                "    ($attribute:tt) => {\n"
                "        #$attribute\n"
                "        mod escaped;\n"
                "    };\n"
                "}\n"
                "#[cfg(round5_group_escape)]\n"
                'load_grouped_path_module!([path = "generated/escape.rs"]);\n'
            ),
        )
        self.assertIn("src/generated/escape.rs", sources)
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "unlisted token 'HashMap'"
        ):
            ORACLE.validate_sources(sources, allowed=(), approved_build_scripts=())

    def test_scans_split_macro_path_attribute_target(self) -> None:
        sources = self.compiled_macro_path_sources(
            cfg="round5_split_escape",
            source=(
                "macro_rules! load_split_path_module {\n"
                "    ($pound:tt) => {\n"
                '        $pound[path = "generated/escape.rs"]\n'
                "        mod escaped;\n"
                "    };\n"
                "}\n"
                "#[cfg(round5_split_escape)]\n"
                "load_split_path_module!(#);\n"
            ),
        )
        self.assertIn("src/generated/escape.rs", sources)
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "unlisted token 'HashMap'"
        ):
            ORACLE.validate_sources(sources, allowed=(), approved_build_scripts=())

    def test_scans_escaped_rs_literal_target(self) -> None:
        sources = self.compiled_macro_path_sources(
            cfg="escaped_literal",
            source=(
                "macro_rules! load_grouped_path_module {\n"
                "    ($attribute:tt) => {\n"
                "        #$attribute\n"
                "        mod escaped;\n"
                "    };\n"
                "}\n"
                "#[cfg(escaped_literal)]\n"
                'load_grouped_path_module!([path = "generated/\\u{65}scape.rs"]);\n'
            ),
        )
        self.assertIn("src/generated/escape.rs", sources)

    def test_scans_raw_rs_literal_target(self) -> None:
        sources = self.compiled_macro_path_sources(
            cfg="raw_literal",
            source=(
                "macro_rules! load_grouped_path_module {\n"
                "    ($attribute:tt) => {\n"
                "        #$attribute\n"
                "        mod escaped;\n"
                "    };\n"
                "}\n"
                "#[cfg(raw_literal)]\n"
                'load_grouped_path_module!([path = r"generated/escape.rs"]);\n'
            ),
        )
        self.assertIn("src/generated/escape.rs", sources)

    def test_ordinary_missing_rs_literal_does_not_expand_source_set(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "src").mkdir()
            (root / "src/lib.rs").write_text(
                'const FIXTURE_NAME: &str = "not/a/module.rs";\n', encoding="utf-8"
            )
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            subprocess.run(("git", "add", "src/lib.rs"), cwd=root, check=True)

            self.assertEqual(
                ORACLE.tracked_rust_sources(repo_root=root),
                {"src/lib.rs": 'const FIXTURE_NAME: &str = "not/a/module.rs";\n'},
            )

    def test_allows_macro_attribute_matcher_when_transcriber_discards_it(self) -> None:
        ORACLE.validate_sources(
            {
                "crates/demo/src/lib.rs": (
                    "macro_rules! declare_names {\n"
                    "    ($(#[$meta:meta])* $name:ident) => {\n"
                    "        enum Names { $name }\n"
                    "    };\n"
                    "}\n"
                    "declare_names!(#[doc = \"kept at the call site\"] One);\n"
                )
            },
            allowed=(),
            approved_build_scripts=(),
        )

    def test_rejects_composed_generated_rust_from_a_build_script(self) -> None:
        build_path = "crates/demo/build.rs"
        generated_build = (
            "fn main() {\n"
            "    let name = [\"Hash\", \"Map\"].concat();\n"
            "    let output = std::path::PathBuf::from(std::env::var(\"OUT_DIR\").unwrap())\n"
            "        .join(\"generated.rs\");\n"
            "    std::fs::write(output, format!(\"type Escape = {name}<u8, u8>;\")).unwrap();\n"
            "}\n"
        )
        approved = (
            ORACLE.ApprovedBuildScript(
                build_path,
                hashlib.sha256(generated_build.encode("utf-8")).hexdigest(),
                "negative-control fixture",
            ),
        )
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "unlisted token 'include'"
        ):
            ORACLE.validate_sources(
                {
                    build_path: generated_build,
                    "crates/demo/src/lib.rs": (
                        'include!(concat!(env!("OUT_DIR"), "/generated.rs"));\n'
                    ),
                },
                allowed=(),
                approved_build_scripts=approved,
            )

    def test_rejects_aliased_include_of_generated_rust(self) -> None:
        build_path = "crates/demo/build.rs"
        generated_build = (
            "fn main() {\n"
            "    let name = [\"Hash\", \"Map\"].concat();\n"
            "    let output = std::path::PathBuf::from(std::env::var(\"OUT_DIR\").unwrap())\n"
            "        .join(\"generated.rs\");\n"
            "    std::fs::write(output, format!(\"type Escape = {name}<u8, u8>;\")).unwrap();\n"
            "}\n"
        )
        approved = (
            ORACLE.ApprovedBuildScript(
                build_path,
                hashlib.sha256(generated_build.encode("utf-8")).hexdigest(),
                "negative-control fixture",
            ),
        )
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "unlisted token 'include'"
        ):
            ORACLE.validate_sources(
                {
                    build_path: generated_build,
                    "crates/demo/src/lib.rs": (
                        "use std::include as include_generated;\n"
                        'include_generated!(concat!(env!("OUT_DIR"), "/generated.rs"));\n'
                    ),
                },
                allowed=(),
                approved_build_scripts=approved,
            )

    def test_rejects_a_changed_approved_build_script(self) -> None:
        build_path = "crates/demo/build.rs"
        approved_source = "fn main() {}\n"
        approved = (
            ORACLE.ApprovedBuildScript(
                build_path,
                hashlib.sha256(approved_source.encode("utf-8")).hexdigest(),
                "negative-control fixture",
            ),
        )
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "changed approved build script"
        ):
            ORACLE.validate_sources(
                {build_path: 'fn main() { std::fs::write("generated.rs", "").unwrap(); }'},
                allowed=(),
                approved_build_scripts=approved,
            )


class HashOrderOracleRunnerTests(unittest.TestCase):
    def test_scan_only_prints_the_tripwire_marker(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            ORACLE.validate_tripwire(
                source_loader=lambda: {}, allowed=(), approved_build_scripts=()
            )
        self.assertEqual(output.getvalue(), "HASH ORDER TOKEN TRIPWIRE: PASS\n")

    def test_runs_every_component_and_prints_one_pass_marker(self) -> None:
        seen: list[tuple[str, ...]] = []

        def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            seen.append(tuple(command))
            return subprocess.CompletedProcess(command, 0)

        output = io.StringIO()
        with redirect_stdout(output):
            ORACLE.validate(
                runner=runner,
                source_loader=lambda: {},
                allowed=(),
                approved_build_scripts=(),
            )
        self.assertEqual(seen, list(ORACLE.COMMANDS))
        self.assertEqual(output.getvalue(), "HASH ORDER DETERMINISM ORACLE: PASS\n")

    def test_stops_at_the_first_failed_component(self) -> None:
        commands = (("first",), ("second",), ("third",))
        seen: list[tuple[str, ...]] = []

        def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            seen.append(tuple(command))
            return subprocess.CompletedProcess(command, 7 if command == ("second",) else 0)

        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "exited 7: second"
        ):
            ORACLE.validate(
                runner=runner,
                commands=commands,
                source_loader=lambda: {},
                allowed=(),
                approved_build_scripts=(),
            )
        self.assertEqual(seen, [("first",), ("second",)])


if __name__ == "__main__":
    unittest.main()
