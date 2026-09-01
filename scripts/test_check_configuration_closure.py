"""Tests for the configuration-closure gate.

Two of these are compile-backed, and they are the point of the file. The
mechanism this gate replaces was a reconstruction of rustc's compiled-source
set from source text; nine review rounds escaped it with a `#[path]` module, a
macro-synthesised attribute, a token-split attribute, a doc comment desugaring
to a `#[doc]` literal, a lifetime mis-lexed as a character literal, an ordinary
`mod` declaration, and a macro-emitted `mod` resolved at its invocation site.

Every one of those escapes needed an undeclared `cfg` to hide from the default
Clippy run, or was caught by plain Clippy anyway. So the two properties worth
proving executably are:

- an undeclared `cfg` name does not compile under the workspace lint config
  (`test_undeclared_cfg_is_a_compile_error`), and
- a source rustc really compiles appears in its dep-info even when git cannot
  see it (`test_dep_info_names_an_ignored_path_module_target`).

Together those bound the configuration space and prove the reconciliation sees
into it, without enumerating a single spelling.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("check_configuration_closure.py")
SPEC = importlib.util.spec_from_file_location("check_configuration_closure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLOSURE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLOSURE
SPEC.loader.exec_module(CLOSURE)

REPO_ROOT = CLOSURE.REPO_ROOT


def write_crate(root: Path, *, manifest: str, sources: dict[str, str]) -> None:
    (root / "Cargo.toml").write_text(manifest, encoding="utf-8")
    for name, text in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    toolchain = REPO_ROOT / "rust-toolchain.toml"
    if toolchain.is_file():
        shutil.copy(toolchain, root / "rust-toolchain.toml")


class CompileBackedControlTests(unittest.TestCase):
    """The two properties the whole design rests on, proved by rustc."""

    def test_undeclared_cfg_is_a_compile_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_crate(
                root,
                manifest=(
                    "[package]\n"
                    'name = "closure-probe"\n'
                    'version = "0.0.0"\n'
                    'edition = "2021"\n'
                    "\n"
                    "[lints.rust]\n"
                    'unexpected_cfgs = "deny"\n'
                ),
                sources={
                    "src/lib.rs": (
                        "#[cfg(round_escape)]\n"
                        "pub fn escaped() -> std::collections::HashMap<u8, u8> {\n"
                        "    std::collections::HashMap::new()\n"
                        "}\n"
                    )
                },
            )
            completed = subprocess.run(
                ("cargo", "check", "--quiet"),
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("unexpected `cfg` condition name", completed.stderr)

    def test_declared_feature_cfg_still_compiles(self) -> None:
        # The positive control: the deny must reject undeclared names only, not
        # ordinary feature gating.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_crate(
                root,
                manifest=(
                    "[package]\n"
                    'name = "closure-probe"\n'
                    'version = "0.0.0"\n'
                    'edition = "2021"\n'
                    "\n"
                    "[features]\n"
                    "declared = []\n"
                    "\n"
                    "[lints.rust]\n"
                    'unexpected_cfgs = "deny"\n'
                ),
                sources={
                    "src/lib.rs": '#[cfg(feature = "declared")]\npub fn gated() {}\n'
                },
            )
            completed = subprocess.run(
                ("cargo", "check", "--quiet"),
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_dep_info_names_an_ignored_path_module_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_crate(
                root,
                manifest=(
                    "[package]\n"
                    'name = "closure-probe"\n'
                    'version = "0.0.0"\n'
                    'edition = "2021"\n'
                ),
                sources={
                    "src/lib.rs": (
                        '#[path = "generated/escape.rs"]\nmod escape;\n'
                    ),
                    "src/generated/escape.rs": "pub fn generated() {}\n",
                    ".gitignore": "src/generated/\n",
                },
            )
            completed = subprocess.run(
                ("cargo", "check", "--quiet"),
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            compiled = CLOSURE.compiled_rust_sources((root / "target" / "debug",), root)
            self.assertIn("src/generated/escape.rs", compiled)


class DepInfoParsingTests(unittest.TestCase):
    def parse(self, text: str) -> set[str]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unit.d"
            path.write_text(text, encoding="utf-8")
            return CLOSURE.parse_dep_info(path)

    def test_reads_prerequisites(self) -> None:
        self.assertEqual(
            self.parse("target/deps/x.rmeta: src/lib.rs src/other.rs Cargo.toml\n"),
            {"src/lib.rs", "src/other.rs", "Cargo.toml"},
        )

    def test_unescapes_spaces_in_paths(self) -> None:
        self.assertEqual(
            self.parse("target/deps/x.rmeta: src/with\\ space.rs\n"),
            {"src/with space.rs"},
        )

    def test_skips_comments_and_phony_targets(self) -> None:
        self.assertEqual(
            self.parse(
                "# env-dep:CLIPPY_ARGS=\n"
                "target/deps/x.rmeta: src/lib.rs\n"
                "src/lib.rs:\n"
            ),
            {"src/lib.rs"},
        )


class DeclaredConfigurationSpaceTests(unittest.TestCase):
    def test_live_workspace_denies_unexpected_cfgs(self) -> None:
        CLOSURE.check_declared_configuration_space(REPO_ROOT)

    def test_rejects_a_member_that_does_not_inherit_the_lint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "member").mkdir()
            (root / "Cargo.toml").write_text(
                "[workspace]\n"
                'members = ["member"]\n'
                "\n"
                "[workspace.lints.rust]\n"
                'unexpected_cfgs = "deny"\n',
                encoding="utf-8",
            )
            (root / "member" / "Cargo.toml").write_text(
                '[package]\nname = "member"\nversion = "0.0.0"\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(
                CLOSURE.ConfigurationClosureFailure, "do not deny unexpected_cfgs"
            ):
                CLOSURE.check_declared_configuration_space(root)

    def test_rejects_a_workspace_that_only_warns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text(
                "[workspace]\n"
                "members = []\n"
                "\n"
                "[workspace.lints.rust]\n"
                'unexpected_cfgs = "warn"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CLOSURE.ConfigurationClosureFailure, 'unexpected_cfgs = "deny"'
            ):
                CLOSURE.check_declared_configuration_space(root)


class MatrixCoverageTests(unittest.TestCase):
    def test_live_matrix_covers_every_declared_feature(self) -> None:
        CLOSURE.check_matrix_covers_declared_features(REPO_ROOT)

    def test_live_matrix_names_only_declared_features(self) -> None:
        declared = {
            (package, feature)
            for package, features in CLOSURE.declared_features(REPO_ROOT).items()
            for feature in features
        }
        for run in CLOSURE.CLIPPY_MATRIX:
            for pair in run.explicit_features():
                self.assertIn(pair, declared, f"{run.label} names {pair}")

    def test_every_registered_run_is_issued_by_its_owner(self) -> None:
        for run in CLOSURE.CLIPPY_MATRIX:
            CLOSURE.check_owner_invokes(run, REPO_ROOT)

    def test_the_gate_owns_a_per_pull_request_run(self) -> None:
        cadences = {
            run.cadence for run in CLOSURE.CLIPPY_MATRIX if run.owner == "scripts/gate.py"
        }
        self.assertEqual(cadences, {CLOSURE.PER_PULL_REQUEST})

    def test_rejects_a_feature_no_registered_run_compiles(self) -> None:
        matrix = tuple(
            run for run in CLOSURE.CLIPPY_MATRIX if not run.covers_all_features
        )
        trimmed = tuple(
            CLOSURE.ClippyRun(
                label=run.label,
                command=tuple(
                    argument
                    for index, argument in enumerate(run.command)
                    if argument != "--features"
                    and (index == 0 or run.command[index - 1] != "--features")
                ),
                owner=run.owner,
                hosts=run.hosts,
                cadence=run.cadence,
            )
            for run in matrix
        )
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "compiled by no registered Clippy run"
        ):
            CLOSURE.check_matrix_covers_declared_features(REPO_ROOT, trimmed)

    def test_rejects_a_matrix_feature_no_member_declares(self) -> None:
        matrix = (
            CLOSURE.ClippyRun(
                label="invented",
                command=(
                    "cargo",
                    "clippy",
                    "--workspace",
                    "--all-targets",
                    "--features",
                    "chelis-types/no-such-feature",
                    "--",
                    "-D",
                    "warnings",
                ),
                owner="scripts/gate.py",
                hosts=("linux",),
                cadence=CLOSURE.PER_PULL_REQUEST,
            ),
        ) + CLOSURE.CLIPPY_MATRIX
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "no workspace member declares"
        ):
            CLOSURE.check_matrix_covers_declared_features(REPO_ROOT, matrix)

    def test_rejects_an_unqualified_feature(self) -> None:
        run = CLOSURE.ClippyRun(
            label="unqualified",
            command=("cargo", "clippy", "--features", "smt"),
            owner="scripts/gate.py",
            hosts=("linux",),
            cadence=CLOSURE.PER_PULL_REQUEST,
        )
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "unqualified feature"
        ):
            run.explicit_features()

    def test_rejects_a_run_its_owner_does_not_issue(self) -> None:
        run = CLOSURE.ClippyRun(
            label="unwired",
            command=("cargo", "clippy", "--workspace", "--never-passed"),
            owner="scripts/gate.py",
            hosts=("linux",),
            cadence=CLOSURE.PER_PULL_REQUEST,
        )
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "is not invoked by its owner"
        ):
            CLOSURE.check_owner_invokes(run, REPO_ROOT)


class SourceReconciliationTests(unittest.TestCase):
    def test_names_a_source_no_configuration_compiled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            (root / "src").mkdir()
            (root / "src" / "compiled.rs").write_text("", encoding="utf-8")
            (root / "src" / "orphan.rs").write_text("", encoding="utf-8")
            deps = root / "target" / "debug" / "deps"
            deps.mkdir(parents=True)
            (deps / "unit.d").write_text(
                "target/debug/deps/x.rmeta: src/compiled.rs\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                CLOSURE.ConfigurationClosureFailure, "src/orphan.rs"
            ):
                CLOSURE.check_every_source_is_compiled(
                    (root / "target" / "debug",), root, exceptions=(), nightly_only=()
                )

    def test_accepts_when_every_source_is_accounted_for(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            (root / "src").mkdir()
            (root / "src" / "compiled.rs").write_text("", encoding="utf-8")
            deps = root / "target" / "debug" / "deps"
            deps.mkdir(parents=True)
            (deps / "unit.d").write_text(
                "target/debug/deps/x.rmeta: src/compiled.rs\n", encoding="utf-8"
            )
            CLOSURE.check_every_source_is_compiled(
                (root / "target" / "debug",), root, exceptions=(), nightly_only=()
            )

    def test_rejects_an_empty_dep_info_set(self) -> None:
        # A matrix that never ran must not read as full coverage.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            with self.assertRaisesRegex(
                CLOSURE.ConfigurationClosureFailure, "no dep-info was found"
            ):
                CLOSURE.check_every_source_is_compiled(
                    (root / "target",), root, exceptions=(), nightly_only=()
                )

    def test_rejects_a_stale_exception_directory(self) -> None:
        exception = CLOSURE.UncompiledException(
            directory="crates/does-not-exist",
            reason="stale",
            owning_gate="scripts/gate.py",
        )
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "stale uncompiled-source exception"
        ):
            CLOSURE.check_every_source_is_compiled((), REPO_ROOT, (exception,))

    def test_rejects_an_exception_whose_gate_is_missing(self) -> None:
        exception = CLOSURE.UncompiledException(
            directory="crates/chelis-unord/tests/compile_fail",
            reason="fixtures",
            owning_gate="scripts/no_such_gate.py",
        )
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "missing owning gate"
        ):
            CLOSURE.check_every_source_is_compiled((), REPO_ROOT, (exception,))

    def test_live_nightly_only_inventory_is_well_formed(self) -> None:
        labels = {run.label: run for run in CLOSURE.CLIPPY_MATRIX}
        for source in CLOSURE.NIGHTLY_ONLY_SOURCES:
            self.assertTrue((REPO_ROOT / source.path).is_file(), source.path)
            self.assertIn(source.row, labels, source.path)
            self.assertEqual(labels[source.row].cadence, CLOSURE.NIGHTLY, source.path)

    def test_reports_a_nightly_only_entry_a_run_already_compiled(self) -> None:
        # The inventory prunes itself: an entry the per-pull-request matrix
        # covers is dead weight that would hide a later regression.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            (root / "src").mkdir()
            (root / "src" / "compiled.rs").write_text("", encoding="utf-8")
            deps = root / "target" / "debug" / "deps"
            deps.mkdir(parents=True)
            (deps / "unit.d").write_text(
                "target/debug/deps/x.rmeta: src/compiled.rs\n", encoding="utf-8"
            )
            nightly = (
                CLOSURE.NightlyOnlySource(
                    path="src/compiled.rs",
                    feature="pkg/feature",
                    row="all-features",
                ),
            )
            with self.assertRaisesRegex(
                CLOSURE.ConfigurationClosureFailure, "recorded as nightly-only"
            ):
                CLOSURE.check_every_source_is_compiled(
                    (root / "target" / "debug",),
                    root,
                    exceptions=(),
                    nightly_only=nightly,
                )

    def test_require_complete_drops_the_nightly_allowance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            (root / "src").mkdir()
            (root / "src" / "compiled.rs").write_text("", encoding="utf-8")
            (root / "src" / "nightly.rs").write_text("", encoding="utf-8")
            deps = root / "target" / "debug" / "deps"
            deps.mkdir(parents=True)
            (deps / "unit.d").write_text(
                "target/debug/deps/x.rmeta: src/compiled.rs\n", encoding="utf-8"
            )
            nightly = (
                CLOSURE.NightlyOnlySource(
                    path="src/nightly.rs",
                    feature="pkg/feature",
                    row="all-features",
                ),
            )
            # Permitted per pull request ...
            CLOSURE.check_every_source_is_compiled(
                (root / "target" / "debug",),
                root,
                exceptions=(),
                nightly_only=nightly,
            )
            # ... and not permitted on the run that compiles everything.
            with self.assertRaisesRegex(
                CLOSURE.ConfigurationClosureFailure, "src/nightly.rs"
            ):
                CLOSURE.check_every_source_is_compiled(
                    (root / "target" / "debug",),
                    root,
                    exceptions=(),
                    nightly_only=nightly,
                    require_complete=True,
                )

    def test_rejects_a_nightly_entry_naming_a_per_pull_request_row(self) -> None:
        nightly = (
            CLOSURE.NightlyOnlySource(
                path="crates/chelis-prove/src/z3_engine.rs",
                feature="chelis-prove/z3",
                row="default-features",
            ),
        )
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "not a nightly row"
        ):
            CLOSURE.check_every_source_is_compiled(
                (), REPO_ROOT, exceptions=(), nightly_only=nightly
            )

    def test_rejects_a_stale_nightly_entry(self) -> None:
        nightly = (
            CLOSURE.NightlyOnlySource(
                path="crates/chelis-prove/src/gone.rs",
                feature="chelis-prove/z3",
                row="all-features",
            ),
        )
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "stale nightly-only source"
        ):
            CLOSURE.check_every_source_is_compiled(
                (), REPO_ROOT, exceptions=(), nightly_only=nightly
            )

    def test_live_exceptions_are_standalone_cargo_projects(self) -> None:
        for exception in CLOSURE.UNCOMPILED_EXCEPTIONS:
            directory = REPO_ROOT / exception.directory
            self.assertTrue(directory.is_dir(), exception.directory)
            self.assertTrue(
                any(directory.rglob("Cargo.toml")),
                f"{exception.directory} must hold standalone Cargo projects",
            )


if __name__ == "__main__":
    unittest.main()
