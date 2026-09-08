"""Small executable positive/negative archive oracle, without a Chelis rebuild.

Run explicitly with `.venv/bin/python -m unittest scripts.ci_archive_execution`.
The workspace archive producer runs it with the same pinned nextest as CI.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import ci_test_archive as archive


class ExecutionTests(unittest.TestCase):
    def test_real_archives_preserve_selection_failure_and_feature_configuration(self):
        target = archive.ROOT / "target"
        target.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="archive-contract-", dir=target) as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "Cargo.toml").write_text('''[package]
name = "chelis-types"
version = "0.0.0"
edition = "2021"
[workspace]
[features]
generalize-sweep-oracle = []
''')
            (root / "src/lib.rs").write_text('''#[test] fn unit_pass() {}
#[cfg(feature = "generalize-sweep-oracle")]
#[test] fn feature_pass() {}
''')
            (root / "tests/stamp_to_typed.rs").write_text('''#[test] fn positive() {}
#[test] fn negative() { panic!("archive negative control"); }
#[test] #[ignore] fn ignored() {}
''')
            env = {**os.environ, "CARGO_TARGET_DIR": str(root / "target"),
                   "CARGO_HUSKY_DONT_INSTALL_HOOKS": "1"}

            def invoke(command):
                return subprocess.run(command, cwd=root, env=env, text=True,
                                      capture_output=True, check=False, timeout=120)

            def listed(command):
                result = invoke(command)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                return {(binary, name, case["ignored"], case["filter-match"]["status"])
                        for binary, suite in data["rust-suites"].items()
                        for name, case in suite["testcases"].items()}

            with patch.object(archive, "ROOT", root), patch.dict(os.environ, env):
                for configuration in archive.CONFIGURATIONS:
                    path = root / f"{configuration}.tar.zst"
                    self.assertEqual(archive.main(["create", "--configuration", configuration,
                                                   "--archive-file", str(path)]), 0)
                    archive.verify(path, configuration)
                    with self.assertRaises(ValueError):
                        archive.verify(path, "generalization" if configuration == "workspace" else "workspace")
                    base = ["cargo", "nextest", "list", "--workspace", "--message-format", "json"]
                    built = listed([*base, *archive.CONFIGURATIONS[configuration]])
                    reused = listed(archive.reuse_command(base, path))
                    self.assertEqual(built, reused)
                    self.assertEqual(any(row[1] == "feature_pass" for row in reused),
                                     configuration == "generalization")
                    selection = ["cargo", "nextest", "list", "-p", "chelis-types", "--test",
                                 "stamp_to_typed", "-E", "test(=positive)", "--message-format", "json"]
                    self.assertEqual(listed([*selection, *archive.CONFIGURATIONS[configuration]]),
                                     listed(archive.reuse_command(selection, path)))
                    for name, succeeds in (("positive", True), ("negative", False), ("absent", False)):
                        command = ["cargo", "nextest", "run", "-p", "chelis-types", "--test",
                                   "stamp_to_typed", "-E", f"test(={name})", "--no-fail-fast"]
                        result = invoke(archive.reuse_command(command, path))
                        self.assertEqual(result.returncode == 0, succeeds, result.stderr)
                    ignored = ["cargo", "nextest", "run", "-p", "chelis-types", "--test",
                               "stamp_to_typed", "--run-ignored", "only", "--test-threads", "1"]
                    result = invoke(archive.reuse_command(ignored, path))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("1 test run", result.stderr)
                    with path.open("ab") as stream:
                        stream.write(b"corruption")
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        archive.verify(path, configuration)


if __name__ == "__main__":
    unittest.main()
