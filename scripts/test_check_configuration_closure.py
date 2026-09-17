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

import yaml


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
    """Leg 2 must cover both feature states, and read features from cargo.

    Round 10 found the gap these lock: `chelis-cli`'s optional `chelis-prove`
    dependency is an implicit *default* feature, invisible to the manifest's
    `[features]` table, and every matrix row enabled it, so the 25
    `#[cfg(not(feature = "chelis-prove"))]` regions were linted by no row at
    any cadence.
    """

    def test_live_matrix_covers_every_declared_feature_both_ways(self) -> None:
        by_row = CLOSURE.check_matrix_covers_declared_features(REPO_ROOT)
        per_pull_request = CLOSURE.features_enabled_per_pull_request(by_row)
        self.assertIn(("chelis-prove", "clarabel"), per_pull_request)
        self.assertNotIn(("chelis-prove", "arb"), per_pull_request)
        self.assertNotIn(("chelis-prove", "z3"), per_pull_request)

    def test_declared_features_include_implicit_optional_dependencies(self) -> None:
        declared = CLOSURE.declared_features(REPO_ROOT)
        self.assertIn(
            "chelis-prove",
            declared["chelis-cli"],
            "an optional dependency creates a feature that `[features]` never lists",
        )

    def test_declared_features_agree_with_cargo(self) -> None:
        metadata = CLOSURE._cargo_metadata(REPO_ROOT, no_deps=True)
        cargo = {
            (package["name"], feature)
            for package in metadata["packages"]
            for feature in package.get("features", {})
            if feature != "default"
        }
        script = {
            (package, feature)
            for package, features in CLOSURE.declared_features(REPO_ROOT).items()
            for feature in features
        }
        self.assertEqual(script, cargo)

    def test_rejects_a_matrix_that_never_disables_a_default_feature(self) -> None:
        # The exact pre-repair matrix: additive rows only.
        additive = tuple(
            run for run in CLOSURE.CLIPPY_MATRIX if run.label != "no-default-features"
        )
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "chelis-cli/chelis-prove"
        ) as caught:
            CLOSURE.check_matrix_covers_declared_features(REPO_ROOT, additive)
        self.assertIn("cfg(not(feature", str(caught.exception))

    def test_rejects_a_matrix_that_never_enables_a_feature(self) -> None:
        off_only = tuple(
            run for run in CLOSURE.CLIPPY_MATRIX if run.label == "no-default-features"
        )
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "no registered Clippy run enables"
        ):
            CLOSURE.check_matrix_covers_declared_features(REPO_ROOT, off_only)

    def test_all_features_alone_does_not_satisfy_coverage(self) -> None:
        # `--all-features` enables everything at once, which is the worst case
        # for off-state coverage, not the best.
        all_only = tuple(
            run for run in CLOSURE.CLIPPY_MATRIX if run.label == "all-features"
        )
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "would not be linted"
        ):
            CLOSURE.check_matrix_covers_declared_features(REPO_ROOT, all_only)

    def test_cargo_feature_flags_are_extracted_from_each_command(self) -> None:
        flags = {run.label: run.cargo_feature_flags() for run in CLOSURE.CLIPPY_MATRIX}
        self.assertEqual(flags["default-features"], ())
        self.assertEqual(flags["no-default-features"], ("--no-default-features",))
        self.assertEqual(flags["all-features"], ("--all-features",))
        self.assertEqual(flags["cvc5-features"][0], "--features")
        for run in CLOSURE.CLIPPY_MATRIX:
            # `-D warnings` sits after `--`; it must never be read as a flag.
            self.assertNotIn("-D", run.cargo_feature_flags())

    def test_resolved_features_come_from_cargo_not_the_command(self) -> None:
        rows = {run.label: run for run in CLOSURE.CLIPPY_MATRIX}
        default = CLOSURE.resolved_features(rows["default-features"], REPO_ROOT)
        none = CLOSURE.resolved_features(rows["no-default-features"], REPO_ROOT)
        # Named by no command, but cargo enables it through `default`.
        self.assertIn(("chelis-cli", "chelis-prove"), default)
        self.assertNotIn(("chelis-cli", "chelis-prove"), none)
        # Named by no command either, but `smt` turns it on transitively.
        cvc5 = CLOSURE.resolved_features(rows["cvc5-features"], REPO_ROOT)
        self.assertIn(("chelis-prove", "cvc5-rs"), cvc5)

    def test_every_registered_run_is_issued_by_its_owner(self) -> None:
        for run in CLOSURE.CLIPPY_MATRIX:
            CLOSURE.check_owner_invokes(run, REPO_ROOT)

    def test_the_gate_owns_the_per_pull_request_rows(self) -> None:
        gate_rows = {
            run.label: run.cadence
            for run in CLOSURE.CLIPPY_MATRIX
            if run.owner == "scripts/gate.py"
        }
        self.assertEqual(set(gate_rows.values()), {CLOSURE.PER_PULL_REQUEST})
        self.assertIn("no-default-features", gate_rows)

    def test_macos_rows_are_nightly_and_linux_rows_remain_per_pr(self) -> None:
        owner = ".github/workflows/macos-nightly.yml"
        workflow = yaml.safe_load((REPO_ROOT / owner).read_text())
        job = workflow["jobs"]["macos-workspace-shard"]
        self.assertEqual(job["runs-on"], "macos-latest")
        commands = [step.get("run") for step in job["steps"]]
        mac_rows = [run for run in CLOSURE.CLIPPY_MATRIX if "macos" in run.hosts]
        self.assertEqual(len(mac_rows), 2)
        for run in mac_rows:
            self.assertEqual(run.hosts, ("macos",))
            self.assertEqual(run.owner, owner)
            self.assertEqual(run.cadence, CLOSURE.NIGHTLY)
            self.assertIn(" ".join(run.command), commands)
        gate_rows = [run for run in CLOSURE.CLIPPY_MATRIX if run.owner == "scripts/gate.py"]
        self.assertEqual(len(gate_rows), 3)
        for run in gate_rows:
            self.assertEqual(run.hosts, ("linux",))
            self.assertEqual(run.cadence, CLOSURE.PER_PULL_REQUEST)

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
    def test_wire_driver_has_one_exact_executable_owner(self) -> None:
        from capacity_census_wire_calls import DRIVER

        selected = tuple(
            entry for entry in CLOSURE.UNCOMPILED_EXCEPTIONS
            if entry.directory == Path(DRIVER).parent.as_posix()
        )
        self.assertEqual(len(selected), 1, "the standalone Clippy driver needs its owner")
        self.assertEqual(selected[0].sources, (DRIVER,))
        self.assertEqual(selected[0].owning_gate, "scripts/capacity_census_wire_calls.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            for name in (DRIVER, selected[0].owning_gate, "src/compiled.rs"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            deps = root / "target/debug/deps"
            deps.mkdir(parents=True)
            (deps / "unit.d").write_text(
                "target/debug/deps/x.rmeta: src/compiled.rs\n", encoding="utf-8"
            )

            def reconcile(owners=selected):
                CLOSURE.check_every_source_is_compiled(
                    (root / "target/debug",), root,
                    exceptions=owners, nightly_only=(), require_complete=True,
                )

            reconcile()
            with self.assertRaisesRegex(CLOSURE.ConfigurationClosureFailure, "wire_calls/driver.rs"):
                reconcile(())
            owner = root / selected[0].owning_gate
            owner.unlink()
            with self.assertRaisesRegex(CLOSURE.ConfigurationClosureFailure, "missing owning gate"):
                reconcile()
            owner.write_text("", encoding="utf-8")
            for relative in ("extra.rs", "nested/extra.rs"):
                extra = root / selected[0].directory / relative
                extra.parent.mkdir(parents=True, exist_ok=True)
                extra.write_text("", encoding="utf-8")
                with self.assertRaisesRegex(CLOSURE.ConfigurationClosureFailure, "exact fixture inventory"):
                    reconcile()
                extra.unlink()
            driver = root / DRIVER
            driver.unlink()
            with self.assertRaisesRegex(CLOSURE.ConfigurationClosureFailure, "exact fixture inventory"):
                reconcile()

    def test_cache_compile_fixtures_are_exact_owned_sources_not_a_directory_pass(self):
        from dataclasses import replace
        from capacity_census_cache_publication import COMPILE_CASES

        name = "crates/chelis-compiler-api/tests/fixtures/cache_publication"
        selected = tuple(x for x in CLOSURE.UNCOMPILED_EXCEPTIONS if x.directory == name)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].sources, tuple(c.fixture for c in COMPILE_CASES))
        self.assertEqual(selected[0].owning_gate, "scripts/capacity_census_cache_publication.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            for source in (*selected[0].sources, selected[0].owning_gate, "src/compiled.rs"):
                path = root / source
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            deps = root / "target/debug/deps"
            deps.mkdir(parents=True)
            (deps / "unit.d").write_text("target/debug/deps/x.rmeta: src/compiled.rs\n")
            args = ((root / "target/debug",), root)
            CLOSURE.check_every_source_is_compiled(*args, exceptions=selected, nightly_only=())
            with self.assertRaisesRegex(CLOSURE.ConfigurationClosureFailure, "fixture inventory"):
                CLOSURE.check_every_source_is_compiled(
                    *args,
                    exceptions=(replace(selected[0], sources=()),),
                    nightly_only=(),
                )
            extra = root / name / "nested/undriven.rs"
            extra.parent.mkdir()
            extra.write_text("")
            with self.assertRaisesRegex(CLOSURE.ConfigurationClosureFailure, "fixture inventory"):
                CLOSURE.check_every_source_is_compiled(*args, exceptions=selected, nightly_only=())
            extra.unlink()
            (root / selected[0].sources[0]).unlink()
            with self.assertRaisesRegex(CLOSURE.ConfigurationClosureFailure, "fixture inventory"):
                CLOSURE.check_every_source_is_compiled(*args, exceptions=selected, nightly_only=())

    def test_historical_producer_requires_real_dep_info_without_an_exception(self):
        source = "crates/chelis-compiler-api/tests/fixtures/cache_wire_v3/producer.rs"
        self.assertFalse(any(source.startswith(x.directory + "/") for x in CLOSURE.UNCOMPILED_EXCEPTIONS))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            path = root / source
            path.parent.mkdir(parents=True)
            path.write_text("")
            (root / "src").mkdir()
            (root / "src/compiled.rs").write_text("")
            deps = root / "target/debug/deps"
            deps.mkdir(parents=True)
            info = deps / "unit.d"
            info.write_text("target/debug/deps/x.rmeta: src/compiled.rs\n")
            args = ((root / "target/debug",), root)
            with self.assertRaisesRegex(CLOSURE.ConfigurationClosureFailure, "cache_wire_v3/producer.rs"):
                CLOSURE.check_every_source_is_compiled(*args, exceptions=(), nightly_only=())
            info.write_text("target/debug/deps/x.rmeta: src/compiled.rs " + source + "\n")
            CLOSURE.check_every_source_is_compiled(*args, exceptions=(), nightly_only=())

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
            CLOSURE.check_every_source_is_compiled(
                (), REPO_ROOT, (exception,), nightly_only=()
            )

    def test_rejects_an_exception_whose_gate_is_missing(self) -> None:
        exception = CLOSURE.UncompiledException(
            directory="crates/chelis-unord/tests/compile_fail",
            reason="fixtures",
            owning_gate="scripts/no_such_gate.py",
        )
        with self.assertRaisesRegex(
            CLOSURE.ConfigurationClosureFailure, "missing owning gate"
        ):
            CLOSURE.check_every_source_is_compiled(
                (), REPO_ROOT, (exception,), nightly_only=()
            )

    def test_live_nightly_only_inventory_is_well_formed(self) -> None:
        labels = {run.label: run for run in CLOSURE.CLIPPY_MATRIX}
        for source in CLOSURE.NIGHTLY_ONLY_SOURCES:
            self.assertTrue((REPO_ROOT / source.path).is_file(), source.path)
            self.assertIn(source.row, labels, source.path)
            self.assertEqual(labels[source.row].cadence, CLOSURE.NIGHTLY, source.path)

    def test_accumulated_dep_info_does_not_prune_a_nightly_only_entry(self) -> None:
        # A warm target includes dep-info from cargo invocations outside the
        # registered Clippy rows. That evidence can establish completeness,
        # but it cannot prove that the nightly-only inventory is stale.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            (root / "src").mkdir()
            (root / "src" / "nightly.rs").write_text("", encoding="utf-8")
            deps = root / "target" / "debug" / "deps"
            deps.mkdir(parents=True)
            (deps / "unit.d").write_text(
                "target/debug/deps/x.rmeta: src/nightly.rs\n", encoding="utf-8"
            )
            nightly = (
                CLOSURE.NightlyOnlySource(
                    path="src/nightly.rs",
                    feature="pkg/feature",
                    row="all-features",
                ),
            )
            CLOSURE.check_every_source_is_compiled(
                (root / "target" / "debug",),
                root,
                exceptions=(),
                nightly_only=nightly,
                resolved_features_by_row={
                    "all-features": frozenset({("pkg", "feature")})
                },
            )

    def test_reports_a_nightly_only_entry_a_registered_run_compiled(self) -> None:
        # Cargo's resolved registered-row features prune the inventory: an
        # entry that matrix covers is dead weight hiding a later regression.
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
                    resolved_features_by_row={
                        "all-features": frozenset({("pkg", "feature")}),
                        "default-features": frozenset({("pkg", "feature")}),
                    },
                )

    def test_rejects_a_nonexistent_nightly_feature_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            (root / "src").mkdir()
            (root / "src/nightly.rs").write_text("", encoding="utf-8")
            deps = root / "target/debug/deps"
            deps.mkdir(parents=True)
            (deps / "unit.d").write_text(
                "target/debug/deps/x.rmeta: src/nightly.rs\n", encoding="utf-8"
            )
            nightly = (
                CLOSURE.NightlyOnlySource(
                    path="src/nightly.rs",
                    feature="pkg/missing",
                    row="all-features",
                ),
            )
            with self.assertRaisesRegex(
                CLOSURE.ConfigurationClosureFailure,
                "pkg/missing.*not enabled.*all-features",
            ):
                CLOSURE.check_every_source_is_compiled(
                    (root / "target/debug",),
                    root,
                    exceptions=(),
                    nightly_only=nightly,
                    resolved_features_by_row={
                        "all-features": frozenset({("pkg", "feature")})
                    },
                )

    def test_rejects_a_feature_named_under_the_wrong_nightly_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            (root / "src").mkdir()
            (root / "src/nightly.rs").write_text("", encoding="utf-8")
            deps = root / "target/debug/deps"
            deps.mkdir(parents=True)
            (deps / "unit.d").write_text(
                "target/debug/deps/x.rmeta: src/nightly.rs\n", encoding="utf-8"
            )
            nightly = (
                CLOSURE.NightlyOnlySource(
                    path="src/nightly.rs",
                    feature="pkg/feature",
                    row="default-features-macos",
                ),
            )
            with self.assertRaisesRegex(
                CLOSURE.ConfigurationClosureFailure,
                "pkg/feature.*not enabled.*default-features-macos",
            ):
                CLOSURE.check_every_source_is_compiled(
                    (root / "target/debug",),
                    root,
                    exceptions=(),
                    nightly_only=nightly,
                    resolved_features_by_row={
                        "all-features": frozenset({("pkg", "feature")}),
                        "default-features-macos": frozenset(),
                    },
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
                resolved_features_by_row={
                    "all-features": frozenset({("pkg", "feature")})
                },
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
                    resolved_features_by_row={
                        "all-features": frozenset({("pkg", "feature")})
                    },
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

    def test_rustdoc_fixtures_have_exact_owning_gate_and_reconcile(self) -> None:
        fixture_directory = "scripts/fixtures/capacity_graph"
        selected = tuple(
            exception
            for exception in CLOSURE.UNCOMPILED_EXCEPTIONS
            if exception.directory == fixture_directory
        )
        self.assertEqual(
            len(selected), 1, "standalone rustdoc fixtures need their owning gate"
        )
        self.assertEqual(
            selected[0].owning_gate, "scripts/test_capacity_census_graph.py"
        )
        fixtures = {
            source
            for source in CLOSURE.repository_rust_sources()
            if source.startswith(fixture_directory + "/")
        }
        self.assertEqual(
            fixtures,
            {
                fixture_directory + "/graph.rs",
                fixture_directory + "/serde/src/lib.rs",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "--quiet"), cwd=root, check=True)
            for source in fixtures | {"src/compiled.rs", selected[0].owning_gate}:
                path = root / source
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            deps = root / "target/debug/deps"
            deps.mkdir(parents=True)
            (deps / "unit.d").write_text(
                "target/debug/deps/x.rmeta: src/compiled.rs\n", encoding="utf-8"
            )
            arguments = ((root / "target/debug",), root)
            CLOSURE.check_every_source_is_compiled(
                *arguments, exceptions=selected, nightly_only=()
            )
            with self.assertRaisesRegex(
                CLOSURE.ConfigurationClosureFailure,
                "capacity_graph/graph.rs.*capacity_graph/serde/src/lib.rs",
            ):
                CLOSURE.check_every_source_is_compiled(
                    *arguments, exceptions=(), nightly_only=()
                )
            gate = root / selected[0].owning_gate
            gate.unlink()
            with self.assertRaisesRegex(
                CLOSURE.ConfigurationClosureFailure, "missing owning gate"
            ):
                CLOSURE.check_every_source_is_compiled(
                    *arguments, exceptions=selected, nightly_only=()
                )
            gate.write_text("", encoding="utf-8")
            (root / "scripts/fixtures/neighbor.rs").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(
                CLOSURE.ConfigurationClosureFailure, "scripts/fixtures/neighbor.rs"
            ):
                CLOSURE.check_every_source_is_compiled(
                    *arguments, exceptions=selected, nightly_only=()
                )

    def test_live_exceptions_are_standalone_cargo_projects(self) -> None:
        for exception in CLOSURE.UNCOMPILED_EXCEPTIONS:
            directory = REPO_ROOT / exception.directory
            self.assertTrue(directory.is_dir(), exception.directory)
            if exception.sources:
                self.assertEqual(
                    {path.relative_to(REPO_ROOT).as_posix() for path in directory.rglob("*.rs")},
                    set(exception.sources),
                )
                continue
            self.assertTrue(
                any(directory.rglob("Cargo.toml")),
                f"{exception.directory} must hold standalone Cargo projects",
            )


if __name__ == "__main__":
    unittest.main()
