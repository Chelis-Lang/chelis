"""Unit tests for `bump_compiler_pins.py`.

Run via: `python3 -m unittest scripts.test_bump_compiler_pins` from repo root,
or `python3 scripts/test_bump_compiler_pins.py`.
"""

import importlib.util
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "bump_compiler_pins", here / "bump_compiler_pins.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register in sys.modules BEFORE exec so @dataclass can resolve the
    # module name when looking up annotations on Python 3.14+.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bump_mod = _load_module()


def _load_sibling(name: str):
    """Import a `scripts/<name>.py` gate script for its constants only."""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _lock_path_packages(lock_text: str) -> dict:
    """Map `{name: version}` for every PATH package in a `Cargo.lock`.

    Cargo records a registry package with a `source =` (and a checksum) and a
    local path package without one, so the absence of that key is the
    discriminator. Parsed rather than substring-matched: chelis#1233 review
    (@jeffreyksmithjr) noted that asking whether the workspace version appears
    ANYWHERE in the lock passes a partially stale file as soon as one package
    happens to be current - and the pipeline-artifacts lock records sixteen
    path packages, so "one of them is right" is very weak evidence.
    """
    packages = {}
    for block in lock_text.split("[[package]]")[1:]:
        block = block.split("\n[", 1)[0]
        if re.search(r"^source = ", block, re.MULTILINE):
            continue
        name = re.search(r'^name = "([^"]+)"', block, re.MULTILINE)
        version = re.search(r'^version = "([^"]+)"', block, re.MULTILINE)
        if name and version:
            packages[name.group(1)] = version.group(1)
    return packages


def _manifest_package_name(manifest: Path) -> str:
    """`[package] name` of a fixture crate - the one path package in its own
    lock that pins its own version rather than the workspace's."""
    m = re.search(r'^name = "([^"]+)"', manifest.read_text(), re.MULTILINE)
    assert m is not None, f"{manifest} has no [package] name"
    return m.group(1)


def _workspace_version() -> str:
    """Read `[workspace.package] version` out of the root Cargo.toml."""
    text = (bump_mod.REPO_ROOT / "Cargo.toml").read_text()
    section = text.split("[workspace.package]", 1)[1]
    section = section.split("\n[", 1)[0]
    m = re.search(r'^\s*version\s*=\s*"([^"]+)"', section, re.MULTILINE)
    assert m is not None, "root Cargo.toml has no [workspace.package] version"
    return m.group(1)


class BumpWorkspaceVersionTests(unittest.TestCase):
    """Regression: `bump_workspace_cargo_toml` used to drop the trailing
    newline because the regex `.*` does not match `\\n`. Bumping then
    smushed the next key onto the version line, producing a malformed
    `version = "0.3.2"license = "MIT"`. Lock that in.
    """

    def _bump(self, original: str, new_version: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "Cargo.toml"
            p.write_text(original)
            saved = bump_mod.WORKSPACE_CARGO_TOML
            bump_mod.WORKSPACE_CARGO_TOML = p
            try:
                bump_mod.bump_workspace_cargo_toml(new_version, dry_run=False)
                return p.read_text()
            finally:
                bump_mod.WORKSPACE_CARGO_TOML = saved

    def test_preserves_trailing_newline_on_version_line(self):
        original = (
            "[workspace.package]\n"
            'edition = "2024"\n'
            'version = "0.3.1"\n'
            'license = "MIT"\n'
        )
        bumped = self._bump(original, "0.3.2")
        self.assertIn('version = "0.3.2"\n', bumped)
        # The next key must remain on its own line.
        self.assertIn('"0.3.2"\nlicense', bumped)
        self.assertNotIn('"0.3.2"license', bumped)

    def test_only_rewrites_workspace_package_version(self):
        # A `version = "..."` outside `[workspace.package]` (e.g. inside
        # `[workspace.dependencies]`) must not be touched.
        original = (
            "[workspace.package]\n"
            'version = "0.3.1"\n'
            "[workspace.dependencies]\n"
            'serde = { version = "1.0" }\n'
        )
        bumped = self._bump(original, "0.3.2")
        self.assertIn('version = "0.3.2"\n', bumped)
        self.assertIn('serde = { version = "1.0" }\n', bumped)

    def test_no_op_when_version_already_matches(self):
        original = "[workspace.package]\nversion = \"0.3.2\"\n"
        bumped = self._bump(original, "0.3.2")
        self.assertEqual(bumped, original)


class BumpHullManifestPinTests(unittest.TestCase):
    """`chelis_version_pinned` in the Hull conformance manifest must ride the
    release change set: the v0.15.0 release left main's Hull Conformance gate
    red with STALE CORPUS because the pin was a post-release chore. Lock that
    the bump rewrites exactly the pin line and nothing else in the frozen
    artifact.
    """

    MANIFEST = (
        "{\n"
        '  "corpus_name": "hull-conformance-v1",\n'
        '  "chelis_version_pinned": "0.14.0",\n'
        '  "check_count": 120,\n'
        '  "generator_note": "chelis_version_pinned is frozen"\n'
        "}\n"
    )

    def _bump(self, original: str, new_version: str, dry_run: bool = False):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.json"
            p.write_text(original)
            saved = bump_mod.HULL_MANIFEST
            bump_mod.HULL_MANIFEST = p
            try:
                change = bump_mod.bump_hull_manifest_pin(new_version, dry_run=dry_run)
                return change, p.read_text()
            finally:
                bump_mod.HULL_MANIFEST = saved

    def test_rewrites_only_the_pin_line(self):
        change, bumped = self._bump(self.MANIFEST, "0.15.0")
        self.assertIsNotNone(change)
        self.assertEqual(change.before, "0.14.0")
        self.assertEqual(change.after, "0.15.0")
        self.assertEqual(bumped, self.MANIFEST.replace('"0.14.0"', '"0.15.0"'))
        # The value-mentioning key elsewhere in the file must be untouched.
        self.assertIn('"generator_note": "chelis_version_pinned is frozen"\n', bumped)

    def test_no_op_when_pin_already_matches(self):
        change, bumped = self._bump(self.MANIFEST, "0.14.0")
        self.assertIsNone(change)
        self.assertEqual(bumped, self.MANIFEST)

    def test_dry_run_reports_but_does_not_write(self):
        change, text = self._bump(self.MANIFEST, "0.15.0", dry_run=True)
        self.assertIsNotNone(change)
        self.assertEqual(change.after, "0.15.0")
        self.assertEqual(text, self.MANIFEST)

    def test_real_manifest_carries_the_pin_key(self):
        # The constant must point at a file that actually has the entry, so
        # the release-time rewrite target is real, not aspirational.
        self.assertTrue(bump_mod.HULL_MANIFEST.is_file(), bump_mod.HULL_MANIFEST)
        self.assertIn('"chelis_version_pinned"', bump_mod.HULL_MANIFEST.read_text())


class CleanUntrackedDistTests(unittest.TestCase):
    """`chelis reef build` drops a `dist/` next to every package root it
    builds. Only `packages/chelis-std/dist/` is a tracked artifact dir; the
    release fixture keeps no `dist/`, so regenerating its lock must leave
    the source tree clean. Lock that the cleanup removes the fixture's
    `dist/` but never the chelis-std one.
    """

    def test_removes_dist_for_non_chelis_std_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "release_pipe_stage"
            dist = pkg / "dist"
            dist.mkdir(parents=True)
            (dist / "artifact.chb").write_text("x")
            bump_mod._clean_untracked_dist(pkg)
            self.assertFalse(dist.exists(), "fixture dist/ must be removed")
            self.assertTrue(pkg.exists(), "package root must remain")

    def test_preserves_chelis_std_dist(self):
        # The chelis-std dist dir is tracked; cleanup must skip it even if
        # it exists. Point the module's CHELIS_STD_DIR at a temp dir so the
        # guard's identity check fires without touching the real tree.
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "chelis-std"
            dist = pkg / "dist"
            dist.mkdir(parents=True)
            (dist / "chelis-std-0.4.0.chb").write_text("x")
            saved = bump_mod.CHELIS_STD_DIR
            bump_mod.CHELIS_STD_DIR = pkg
            try:
                bump_mod._clean_untracked_dist(pkg)
            finally:
                bump_mod.CHELIS_STD_DIR = saved
            self.assertTrue(dist.exists(), "chelis-std dist/ must be preserved")

    def test_no_op_when_no_dist_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            # Must not raise when there is nothing to clean.
            bump_mod._clean_untracked_dist(pkg)
            self.assertTrue(pkg.exists())


class LockRegenerationWiringTests(unittest.TestCase):
    """Guard the constants the regeneration path depends on so a future
    edit cannot silently drop a category-4/5 target. These are the files
    the 0.9.0 release failure traced to.
    """

    def test_bundle_dist_and_regen_script_are_wired(self):
        self.assertTrue(
            str(bump_mod.BUNDLE_DIST).endswith("crates/chelis-std-bundle/dist"),
            bump_mod.BUNDLE_DIST,
        )
        self.assertTrue(
            str(bump_mod.REGEN_BUNDLE_SCRIPT).endswith(
                "scripts/regenerate_chelis_std_bundle.py"
            ),
            bump_mod.REGEN_BUNDLE_SCRIPT,
        )
        self.assertTrue(
            bump_mod.REGEN_BUNDLE_SCRIPT.is_file(),
            "the canonical bundle pipeline script must exist",
        )

    def test_pinned_lock_dirs_cover_fixture_and_chelis_std(self):
        names = {p.name for p in bump_mod.PINNED_REAL_LOCK_DIRS}
        self.assertIn("chelis-std", names)
        self.assertIn("release_pipe_stage", names)
        self.assertIn("nautilus_quantile_contract", names)
        self.assertIn("nautilus", names)
        # Every recorded lock dir must currently ship a reef.lock so the
        # regeneration target is real, not aspirational.
        for pkg_dir in bump_mod.PINNED_REAL_LOCK_DIRS:
            self.assertTrue(
                (pkg_dir / "reef.lock").is_file(),
                f"{pkg_dir} must ship a reef.lock to regenerate",
            )

    def test_followup_lock_builds_do_not_overwrite_the_final_std_artifacts(self):
        self.assertNotIn(
            bump_mod.CHELIS_STD_DIR,
            bump_mod.FOLLOWUP_LOCK_REBUILD_DIRS,
            "the canonical bundle generator already refreshes the std root lock; "
            "building it again would overwrite the final artifact bytes",
        )
        self.assertEqual(
            set(bump_mod.FOLLOWUP_LOCK_REBUILD_DIRS),
            set(bump_mod.PINNED_REAL_LOCK_DIRS) - {bump_mod.CHELIS_STD_DIR},
        )

    def test_pinned_toml_inventory_covers_executable_package_examples(self):
        relative = {
            path.relative_to(bump_mod.REPO_ROOT).as_posix()
            for path in bump_mod.PINNED_REAL_TOML_FILES
        }
        self.assertIn(
            "examples/nautilus_quantile_contract/reef.toml",
            relative,
        )
        self.assertIn(
            "examples/nautilus_quantile_contract/fixtures/nautilus/reef.toml",
            relative,
        )


class PinnedFixtureDataTests(unittest.TestCase):
    """Category 8: checked-in test-fixture data that hand-pins the compiler.

    A Rust fixture built from a string literal auto-syncs through category
    (1); an `include_str!`'d `.toml`/`.json` file on disk cannot. The
    `pipeline_parity` set went stale at 0.18.5 and was hand-repaired by the
    release operator, which is the recurrence this category exists to stop.
    """

    SCHEMA = (
        "{\n"
        '  "package": {\n'
        '    "name": "pipeline-parity",\n'
        '    "version": "1.2.3"\n'
        "  },\n"
        '  "compiler": "=0.18.5",\n'
        '  "modules": []\n'
        "}\n"
    )

    def _bump_json(self, original: str, new_version: str, dry_run: bool = False):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "expected_schema.json"
            path.write_text(original)
            change = bump_mod.bump_json_compiler_pin(path, new_version, dry_run=dry_run)
            return change, path.read_text()

    def test_json_pin_rewrite_moves_only_the_compiler_field(self):
        change, bumped = self._bump_json(self.SCHEMA, "0.18.6")
        self.assertIsNotNone(change)
        self.assertEqual(change.before, "=0.18.5")
        self.assertEqual(change.after, "=0.18.6")
        self.assertEqual(bumped, self.SCHEMA.replace('"=0.18.5"', '"=0.18.6"'))
        # The package's own semver must not be confused for the pin.
        self.assertIn('"version": "1.2.3"', bumped)

    def test_json_pin_rewrite_is_a_no_op_when_already_current(self):
        change, bumped = self._bump_json(self.SCHEMA, "0.18.5")
        self.assertIsNone(change)
        self.assertEqual(bumped, self.SCHEMA)

    def test_json_pin_dry_run_reports_but_does_not_write(self):
        change, text = self._bump_json(self.SCHEMA, "0.18.6", dry_run=True)
        self.assertIsNotNone(change)
        self.assertEqual(text, self.SCHEMA)

    def test_every_declared_fixture_file_exists_and_carries_a_pin(self):
        # An aspirational path would make the category silently cover
        # nothing, which is the state that produced the 0.18.5 repair.
        for path in bump_mod.PINNED_FIXTURE_TOML_FILES:
            self.assertTrue(path.is_file(), path)
            self.assertRegex(path.read_text(), r'(?m)^\s*compiler\s*=\s*"=')
        for path in bump_mod.PINNED_FIXTURE_JSON_FILES:
            self.assertTrue(path.is_file(), path)
            self.assertRegex(path.read_text(), r'(?m)^\s*"compiler"\s*:\s*"=')

    def test_fixture_pins_match_the_workspace_version(self):
        version = _workspace_version()
        for path in (
            bump_mod.PINNED_FIXTURE_TOML_FILES + bump_mod.PINNED_FIXTURE_JSON_FILES
        ):
            self.assertIn(f'"={version}"', path.read_text(), path)

    def test_nondeterministic_hash_capture_is_excluded(self):
        # `expected_hashes.txt` records values BASELINE.md declares
        # nondeterministic across machines and no longer asserted; bumping
        # it would imply an oracle this repository does not run.
        declared = set(
            bump_mod.PINNED_FIXTURE_TOML_FILES + bump_mod.PINNED_FIXTURE_JSON_FILES
        )
        self.assertNotIn(
            bump_mod.PIPELINE_PARITY_FIXTURES / "accepted/expected_hashes.txt",
            declared,
        )


class CompileFailFixtureLockTests(unittest.TestCase):
    """Category 7: the committed `Cargo.lock` beside an out-of-workspace
    compile-fail fixture pins the real crates at the workspace version, and
    the fixture's gate step compiles it with `cargo check --locked`. Left
    behind by a bump, `--locked` refuses to update it and the gate reports
    the fixture's *diagnostic* as missing — it reads as a compile-fail
    regression rather than a stale lock (chelis#1128, hit cutting 0.18.2).
    """

    GATE_SCRIPTS = (
        "check_checkpoint_compile_fail",
        "check_pipeline_core_compile_fail",
    )

    def test_inventory_covers_all_path_dependent_compile_fail_fixtures(self):
        discovered = set()
        for manifest in (bump_mod.REPO_ROOT / "crates").glob("*/tests/compile_fail/*/Cargo.toml"):
            data = tomllib.loads(manifest.read_text())
            if any(isinstance(dep, dict) and "path" in dep
                   for dep in data.get("dependencies", {}).values()):
                discovered.add(manifest)
        self.assertEqual(set(bump_mod.COMPILE_FAIL_FIXTURE_MANIFESTS), discovered)

    def test_inventory_covers_known_gated_fixtures(self):
        relative = {
            path.relative_to(bump_mod.REPO_ROOT).as_posix()
            for path in bump_mod.COMPILE_FAIL_FIXTURE_MANIFESTS
        }
        self.assertEqual(
            relative,
            {
                "crates/chelis-types/tests/compile_fail/checkpoint_raw_offset/Cargo.toml",
                "crates/chelis-compiler-api/tests/compile_fail/pipeline_artifacts/Cargo.toml",
                "crates/chelis-unord/tests/compile_fail/order_escape/Cargo.toml",
            },
        )

    def test_inventory_matches_the_gate_scripts_manifest_constants(self):
        # The parity lock the "keep in sync" comment asks for: a fixture that
        # moves must move in both places, or the bump silently stops
        # regenerating the lock its gate step is about to reject.
        gated = {_load_sibling(name).MANIFEST for name in self.GATE_SCRIPTS}
        for fixture in _load_sibling("check_hash_order_phase_b_compile_fail").FIXTURES:
            manifest = Path(fixture.command[fixture.command.index("--manifest-path") + 1])
            dependencies = tomllib.loads(manifest.read_text()).get("dependencies", {})
            if any(isinstance(dep, dict) and "path" in dep for dep in dependencies.values()):
                gated.add(manifest)
        self.assertEqual(set(bump_mod.COMPILE_FAIL_FIXTURE_MANIFESTS), gated)

    def test_each_manifest_ships_a_committed_lock(self):
        # The regeneration target must be real, not aspirational.
        for manifest in bump_mod.COMPILE_FAIL_FIXTURE_MANIFESTS:
            self.assertTrue(manifest.is_file(), manifest)
            self.assertTrue(manifest.with_name("Cargo.lock").is_file(), manifest)

    def test_each_lock_pins_every_path_package_to_the_workspace_version(self):
        # The tripwire for the class itself: this is the assertion that goes
        # red when a bump lands without the regeneration step, and it names
        # the lock instead of a phantom diagnostic regression.
        #
        # EVERY path package is checked, not just whether the version occurs
        # somewhere in the file (chelis#1233 review): a lock can be partially
        # stale, and the substring form passed as soon as any one package was
        # current.
        version = _workspace_version()
        for manifest in bump_mod.COMPILE_FAIL_FIXTURE_MANIFESTS:
            lock = manifest.with_name("Cargo.lock")
            fixture_crate = _manifest_package_name(manifest)
            pinned = {
                name: found
                for name, found in _lock_path_packages(lock.read_text()).items()
                # The fixture crate itself carries its own version (0.0.0), not
                # the workspace's; every other path package is a real crate.
                if name != fixture_crate
            }
            rel = lock.relative_to(bump_mod.REPO_ROOT)
            self.assertTrue(
                pinned, f"{rel} records no workspace path packages to check"
            )
            stale = sorted(
                f"{name} @ {found}" for name, found in pinned.items() if found != version
            )
            # `assertEqual` on the stale list, not `assertIn` on the file: the
            # haystack is a whole lockfile and dumping it buries the fix.
            self.assertEqual(
                stale,
                [],
                f"{rel} has path packages behind the workspace version "
                f"{version}: {stale}. Re-run "
                "`scripts/bump_compiler_pins.py <version>` so its gate step's "
                "`cargo check --locked` accepts the lock.",
            )

    def test_tripwire_rejects_a_partially_stale_lock(self):
        # The exact hole chelis#1233 review named: one current package used to
        # carry the whole file. Synthetic lock, so it holds regardless of what
        # the real fixtures happen to contain.
        version = _workspace_version()
        lock = (
            "version = 4\n\n"
            '[[package]]\nname = "chelis-current"\nversion = "' + version + '"\n\n'
            '[[package]]\nname = "chelis-stale"\nversion = "0.0.1-old"\n\n'
            '[[package]]\nname = "serde"\nversion = "1.0.229"\n'
            'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
            'checksum = "deadbeef"\n'
        )
        found = _lock_path_packages(lock)
        # The registry package is excluded by its `source` key; both path
        # packages are seen, and exactly one of them is stale.
        self.assertEqual(
            found, {"chelis-current": version, "chelis-stale": "0.0.1-old"}
        )
        self.assertIn(f'version = "{version}"', lock, "the old substring form passed")

    def test_regeneration_runs_cargo_update_once_per_manifest(self):
        with mock.patch.object(
            bump_mod.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as runner:
            bump_mod.regenerate_compile_fail_fixture_locks(dry_run=False)
        invoked = [call.args[0] for call in runner.call_args_list]
        self.assertEqual(
            len(invoked), len(bump_mod.COMPILE_FAIL_FIXTURE_MANIFESTS)
        )
        for argv, manifest in zip(invoked, bump_mod.COMPILE_FAIL_FIXTURE_MANIFESTS):
            self.assertEqual(argv[:3], ["cargo", "update", "--workspace"])
            self.assertEqual(argv[3:], ["--manifest-path", str(manifest)])

    def test_regeneration_never_passes_locked_or_regenerates_wholesale(self):
        # `--locked` would reproduce the very failure this step exists to
        # prevent; `generate-lockfile` would drag unrelated registry
        # dependencies forward inside a release change set.
        with mock.patch.object(
            bump_mod.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as runner:
            bump_mod.regenerate_compile_fail_fixture_locks(dry_run=False)
        for call in runner.call_args_list:
            argv = call.args[0]
            self.assertNotIn("--locked", argv)
            self.assertNotIn("generate-lockfile", argv)

    def test_dry_run_reports_without_invoking_cargo(self):
        with mock.patch.object(bump_mod.subprocess, "run") as runner:
            bump_mod.regenerate_compile_fail_fixture_locks(dry_run=True)
        runner.assert_not_called()

    def test_nonzero_cargo_exit_is_fatal(self):
        with mock.patch.object(
            bump_mod.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 101),
        ):
            with self.assertRaises(SystemExit) as caught:
                bump_mod.regenerate_compile_fail_fixture_locks(dry_run=False)
        self.assertIn("cargo update", str(caught.exception))

    def test_missing_manifest_is_fatal_before_cargo_runs(self):
        saved = bump_mod.COMPILE_FAIL_FIXTURE_MANIFESTS
        bump_mod.COMPILE_FAIL_FIXTURE_MANIFESTS = [
            bump_mod.REPO_ROOT / "crates/does-not-exist/Cargo.toml"
        ]
        try:
            with mock.patch.object(bump_mod.subprocess, "run") as runner:
                with self.assertRaises(SystemExit) as caught:
                    bump_mod.regenerate_compile_fail_fixture_locks(dry_run=False)
            runner.assert_not_called()
        finally:
            bump_mod.COMPILE_FAIL_FIXTURE_MANIFESTS = saved
        self.assertIn("does-not-exist", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
