"""Unit tests for `bump_compiler_pins.py`.

Run via: `python3 -m unittest scripts.test_bump_compiler_pins` from repo root,
or `python3 scripts/test_bump_compiler_pins.py`.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
