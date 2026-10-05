"""Archive extraction must retain executable modes without escaping its cache."""
from pathlib import Path
import importlib.util
import stat
import tempfile
import unittest
import zipfile

RUNNER = Path(__file__).resolve().parents[1] / "crates/chelis-axis-core/proofs/verify_axis_verus.py"
spec = importlib.util.spec_from_file_location("axis_verus_runner", RUNNER)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
vm_spec = importlib.util.spec_from_file_location("axis_vermilion_runner", RUNNER.with_name("verify_axis_vermilion.py"))
assert vm_spec is not None and vm_spec.loader is not None
vermilion = importlib.util.module_from_spec(vm_spec)
vm_spec.loader.exec_module(vermilion)


class ArchiveTests(unittest.TestCase):
    def test_upstream_build_environment_uses_its_declared_targets(self):
        original = {"PATH": "/bin", "CARGO_TARGET_DIR": "/unrelated/shared-target", "KEEP": "value"}
        result = vermilion.build_environment(original)
        self.assertNotIn("CARGO_TARGET_DIR", result)
        self.assertEqual(result["KEEP"], "value")
        self.assertIn("CARGO_TARGET_DIR", original)

    def test_upstream_environment_without_override_retains_path(self):
        result = vermilion.build_environment({"PATH": "/bin"})
        self.assertTrue(result["PATH"].endswith(":/bin"))

    def test_executable_mode_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "tools.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                item = zipfile.ZipInfo("verus/cargo-verus")
                item.external_attr = (stat.S_IFREG | 0o755) << 16
                bundle.writestr(item, "binary")
            runner.extract_archive(archive, root / "out")
            self.assertEqual((root / "out/verus/cargo-verus").stat().st_mode & 0o777, 0o755)

    def test_parent_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "tools.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape", "bad")
            with self.assertRaises(ValueError):
                runner.extract_archive(archive, root / "out")
            self.assertFalse((root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
