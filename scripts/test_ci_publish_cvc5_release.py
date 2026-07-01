"""Unit tests for scripts/ci_publish_cvc5_release.py.

Covers the idempotency contract (create only when absent; always upload with
--clobber) and the asset-file filter, without a live `gh`.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import ci_publish_cvc5_release as mod


class AssetFilesTests(unittest.TestCase):
    def test_selects_only_targz_and_sha256(self):
        with TemporaryDirectory() as td:
            d = Path(td)
            (d / "cvc5-linux-x86_64.tar.gz").write_bytes(b"x")
            (d / "cvc5-linux-x86_64.tar.gz.sha256").write_text("a\n")
            (d / "notes.txt").write_text("ignore\n")
            (d / "sub").mkdir()
            names = [p.name for p in mod.asset_files(d)]
            self.assertEqual(
                names,
                ["cvc5-linux-x86_64.tar.gz", "cvc5-linux-x86_64.tar.gz.sha256"],
            )


class BuildCommandsTests(unittest.TestCase):
    def _files(self, td: str) -> list[Path]:
        d = Path(td)
        f = d / "cvc5-linux-x86_64-cvc5sys0.3.1-v2.tar.gz"
        f.write_bytes(b"x")
        s = Path(str(f) + ".sha256")
        s.write_text("a\n")
        return [f, s]

    def test_creates_release_when_absent(self):
        with TemporaryDirectory() as td:
            cmds = mod.build_commands("cvc5-prebuilt-cvc5sys0.3.1", self._files(td), False)
            self.assertEqual(cmds[0][:3], ["gh", "release", "create"])
            self.assertIn("--prerelease", cmds[0])
            # Upload always last, with --clobber.
            self.assertEqual(cmds[-1][:3], ["gh", "release", "upload"])
            self.assertIn("--clobber", cmds[-1])

    def test_skips_create_when_present(self):
        with TemporaryDirectory() as td:
            cmds = mod.build_commands("cvc5-prebuilt-cvc5sys0.3.1", self._files(td), True)
            # Only the upload command; no create.
            self.assertEqual(len(cmds), 1)
            self.assertEqual(cmds[0][:3], ["gh", "release", "upload"])
            self.assertIn("--clobber", cmds[0])

    def test_tag_not_v_prefixed(self):
        # Guards the release.yml `v*` non-collision at the publish site.
        with TemporaryDirectory() as td:
            cmds = mod.build_commands("cvc5-prebuilt-cvc5sys0.3.1", self._files(td), False)
            for cmd in cmds:
                # The tag arg is index 3 for both create and upload.
                self.assertFalse(cmd[3].startswith("v"))

    def test_empty_files_raises(self):
        with self.assertRaises(RuntimeError):
            mod.build_commands("cvc5-prebuilt-cvc5sys0.3.1", [], False)


if __name__ == "__main__":
    unittest.main()
