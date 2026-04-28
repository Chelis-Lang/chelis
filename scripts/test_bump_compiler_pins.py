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


if __name__ == "__main__":
    unittest.main()
