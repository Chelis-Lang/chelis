"""Tests for the ecosystem Chelis source-dependency classifier."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ci_detect_chelis_path_deps import (  # noqa: E402
    ChelisPathDependencyError,
    find_chelis_path_dependencies,
    main,
)


class ChelisPathDependencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.shell = self.workspace / "shell"
        self.chelis = self.workspace / "chelis"
        (self.chelis / "crates" / "chelis-types").mkdir(parents=True)
        self.shell.mkdir()

    def write_manifest(self, relative: str, text: str) -> Path:
        path = self.shell / relative / "Cargo.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_finds_a_root_chelis_path_dependency(self) -> None:
        manifest = self.write_manifest(
            ".",
            '[dependencies]\nchelis-types = { path = "../chelis/crates/chelis-types" }\n',
        )
        found = find_chelis_path_dependencies(self.shell, self.chelis)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].manifest, manifest.resolve())
        self.assertEqual(found[0].dependency_path, "../chelis/crates/chelis-types")

    def test_finds_a_nested_target_dependency(self) -> None:
        self.write_manifest(
            "consumer",
            "[target.'cfg(unix)'.dependencies]\n"
            'chelis-types = { path = "../../chelis/crates/chelis-types" }\n',
        )
        found = find_chelis_path_dependencies(self.shell, self.chelis)
        self.assertEqual(len(found), 1)

    def test_finds_a_chelis_workspace_member_outside_crates(self) -> None:
        (self.chelis / "tree-sitter-chelis").mkdir()
        self.write_manifest(
            ".",
            '[dependencies]\nparser = { path = "../chelis/tree-sitter-chelis" }\n',
        )
        found = find_chelis_path_dependencies(self.shell, self.chelis)
        self.assertEqual(len(found), 1)

    def test_finds_a_cargo_paths_override(self) -> None:
        (self.chelis / "tree-sitter-chelis").mkdir()
        config = self.shell / ".cargo" / "config.toml"
        config.parent.mkdir()
        config.write_text(
            'paths = ["../chelis/tree-sitter-chelis"]\n', encoding="utf-8"
        )
        found = find_chelis_path_dependencies(self.shell, self.chelis)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].manifest, config.resolve())

    def test_finds_a_cargo_config_patch(self) -> None:
        config = self.shell / ".cargo" / "config.toml"
        config.parent.mkdir()
        config.write_text(
            '[patch.crates-io]\nchelis-types = { path = "../chelis/crates/chelis-types" }\n',
            encoding="utf-8",
        )
        found = find_chelis_path_dependencies(self.shell, self.chelis)
        self.assertEqual(len(found), 1)

    def test_ignores_an_unrelated_cargo_paths_override(self) -> None:
        (self.workspace / "support").mkdir()
        config = self.shell / ".cargo" / "config.toml"
        config.parent.mkdir()
        config.write_text('paths = ["../support"]\n', encoding="utf-8")
        self.assertEqual(find_chelis_path_dependencies(self.shell, self.chelis), ())

    def test_rejects_an_invalid_cargo_config(self) -> None:
        config = self.shell / ".cargo" / "config.toml"
        config.parent.mkdir()
        config.write_text("paths = [\n", encoding="utf-8")
        with self.assertRaisesRegex(
            ChelisPathDependencyError, f"cannot parse {config.resolve()}"
        ):
            find_chelis_path_dependencies(self.shell, self.chelis)

    def test_ignores_an_unrelated_path_dependency(self) -> None:
        (self.workspace / "support").mkdir()
        self.write_manifest(
            ".",
            '[dependencies]\nsupport = { path = "../support" }\n',
        )
        self.assertEqual(find_chelis_path_dependencies(self.shell, self.chelis), ())

    def test_ignores_non_dependency_path_fields(self) -> None:
        self.write_manifest(
            ".",
            '[[bin]]\nname = "probe"\npath = "../chelis/crates/probe.rs"\n'
            '[workspace.metadata.fixture]\npath = "../chelis/crates/fixture"\n',
        )
        self.assertEqual(find_chelis_path_dependencies(self.shell, self.chelis), ())

    def test_ignores_target_and_git_manifests(self) -> None:
        for relative in ("target/generated", ".git/scratch"):
            self.write_manifest(
                relative,
                '[dependencies]\nchelis-types = { path = "../../../chelis/crates/chelis-types" }\n',
            )
        self.assertEqual(find_chelis_path_dependencies(self.shell, self.chelis), ())

    def test_rejects_invalid_root_toml_at_the_parse_boundary(self) -> None:
        manifest = self.write_manifest(".", "[dependencies\n")
        with self.assertRaisesRegex(
            ChelisPathDependencyError, f"cannot parse {manifest.resolve()}"
        ):
            find_chelis_path_dependencies(self.shell, self.chelis)

    def test_ignores_an_invalid_nested_fixture_manifest(self) -> None:
        self.write_manifest("tests/fixtures/broken", "[dependencies\n")
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            found = find_chelis_path_dependencies(self.shell, self.chelis)
        self.assertEqual(found, ())
        self.assertIn("ignored invalid nested Cargo manifest", errors.getvalue())

    def run_main(self, expectation: str, chelis_root: Path | None = None) -> int:
        output = io.StringIO()
        root = chelis_root if chelis_root is not None else self.chelis
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            return main([str(self.shell), str(root), "--expect", expectation])

    def test_present_expectation_rejects_an_absent_dependency(self) -> None:
        self.write_manifest(".", '[package]\nname = "consumer"\nversion = "0.1.0"\n')
        self.assertEqual(self.run_main("present"), 1)

    def test_absent_expectation_rejects_a_present_dependency(self) -> None:
        self.write_manifest(
            ".",
            '[dependencies]\nchelis-types = { path = "../chelis/crates/chelis-types" }\n',
        )
        self.assertEqual(self.run_main("absent"), 1)

    def test_absent_expectation_accepts_a_nonexistent_chelis_checkout(self) -> None:
        self.write_manifest(".", '[package]\nname = "consumer"\nversion = "0.1.0"\n')
        self.assertEqual(self.run_main("absent", self.workspace / "missing-chelis"), 0)

    def test_nonexistent_checkout_still_detects_its_lexical_path(self) -> None:
        self.write_manifest(
            ".",
            '[dependencies]\nchelis-types = { path = "../missing-chelis/crates/chelis-types" }\n',
        )
        self.assertEqual(self.run_main("absent", self.workspace / "missing-chelis"), 1)

    def test_each_matching_expectation_succeeds(self) -> None:
        self.write_manifest(".", '[package]\nname = "consumer"\nversion = "0.1.0"\n')
        self.assertEqual(self.run_main("absent"), 0)
        self.write_manifest(
            ".",
            '[dependencies]\nchelis-types = { path = "../chelis/crates/chelis-types" }\n',
        )
        self.assertEqual(self.run_main("present"), 0)


if __name__ == "__main__":
    unittest.main()
