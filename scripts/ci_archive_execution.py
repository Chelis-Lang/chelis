"""Small executable positive/negative archive oracle, without a Chelis rebuild.

Run explicitly with `.venv/bin/python -m unittest scripts.ci_archive_execution`.
The workspace archive producer runs it with the same pinned nextest as CI.
"""

import json
import contextlib
import io
import os
from pathlib import Path
import re
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts import ci_test_archive as archive


class ExecutionTests(unittest.TestCase):
    def test_failed_build_preserves_rust_diagnostics_without_a_manifest(self):
        target = archive.ROOT / "target"
        target.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="archive-error-", dir=target) as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "Cargo.toml").write_text('''[package]
name = "broken-runtime"
version = "0.0.0"
edition = "2021"
[workspace]
''')
            (root / "src/lib.rs").write_text('pub fn probe() { nonexistent_probe_function(); }\n')
            path = root / "broken.tar.zst"
            errors = io.StringIO()
            with patch.object(archive, "ROOT", root), \
                    patch.dict(os.environ, {"CARGO_TARGET_DIR": str(root / "target")}), \
                    contextlib.redirect_stderr(errors):
                self.assertEqual(archive.main(["create", "--configuration", "workspace",
                                               "--archive-file", str(path)]), 1)
            self.assertIn("E0425", errors.getvalue())
            self.assertIn("nonexistent_probe_function", errors.getvalue())
            self.assertIn("src/lib.rs", errors.getvalue())
            self.assertFalse(path.with_suffix(".zst.json").exists())

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
[lib]
name = "chelis_runtime"
crate-type = ["staticlib", "rlib"]
[workspace]
[features]
generalize-sweep-oracle = []
''')
            (root / "src/lib.rs").write_text('''#[test] fn unit_pass() {}
#[cfg(feature = "generalize-sweep-oracle")]
#[test] fn feature_pass() {}
''')
            (root / "src/main.rs").write_text('fn main() { print!("archive-cli"); }\n')
            (root / "tests/stamp_to_typed.rs").write_text('''#[test] fn positive() {}
#[test] fn negative() { panic!("archive negative control"); }
#[test] #[ignore] fn ignored() {}
#[test] fn runtime_present() {
    let deps = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/debug/deps");
    assert!(std::fs::read_dir(deps).unwrap().any(|entry| {
        let name = entry.unwrap().file_name();
        let name = name.to_string_lossy();
        name.starts_with("libchelis_runtime") && name.ends_with(".a")
    }), "runtime staticlib must come from the archive");
}
#[test] fn cli_present() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_chelis-types")).output().unwrap();
    assert!(output.status.success());
    assert_eq!(output.stdout, b"archive-cli");
}
''')
            env = {**os.environ, "CARGO_TARGET_DIR": str(root / "target"),
                   "CARGO_TERM_COLOR": "always",
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
                    selection = ["cargo", "nextest", "list", "-p", "chelis-types", "--test",
                                 "stamp_to_typed", "-E", "test(=positive)", "--message-format", "json"]
                    built_selection = listed([*selection, *archive.CONFIGURATIONS[configuration]])
                    # Nothing from compilation may supply a consumer's runtime artifacts.
                    shutil.rmtree(root / "target")
                    reused = listed(archive.reuse_command(base, path))
                    self.assertEqual(built, reused)
                    self.assertEqual(any(row[1] == "feature_pass" for row in reused),
                                     configuration == "generalization")
                    self.assertEqual(built_selection,
                                     listed(archive.reuse_command(selection, path)))
                    for name, succeeds in (("positive", True), ("runtime_present", True), ("cli_present", True),
                                           ("negative", False), ("absent", False)):
                        command = ["cargo", "nextest", "run", "-p", "chelis-types", "--test",
                                   "stamp_to_typed", "-E", f"test(={name})", "--no-fail-fast"]
                        result = invoke(archive.reuse_command(command, path))
                        self.assertEqual(result.returncode == 0, succeeds, result.stderr)
                    ignored = ["cargo", "nextest", "run", "-p", "chelis-types", "--test",
                               "stamp_to_typed", "--run-ignored", "only", "--test-threads", "1"]
                    result = invoke(archive.reuse_command(ignored, path))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("1 test run", re.sub(r"\x1b\[[0-9;]*m", "", result.stderr))
                    with path.open("ab") as stream:
                        stream.write(b"corruption")
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        archive.verify(path, configuration)


if __name__ == "__main__":
    unittest.main()
