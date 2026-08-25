"""Unit tests for `regenerate_chelis_std_bundle.py`."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "regenerate_chelis_std_bundle", here / "regenerate_chelis_std_bundle.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bundle = _load_module()


class CargoTargetDirectoryTests(unittest.TestCase):
    def test_default_target_is_under_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(bundle.cargo_target_dir(repository), repository / "target")

    def test_absolute_target_stays_absolute(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "repository"
            target = Path(temporary) / "agent-target"
            with patch.dict(os.environ, {"CARGO_TARGET_DIR": str(target)}, clear=True):
                self.assertEqual(bundle.cargo_target_dir(repository), target)

    def test_relative_target_resolves_from_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            with patch.dict(os.environ, {"CARGO_TARGET_DIR": "target/agent"}, clear=True):
                self.assertEqual(
                    bundle.cargo_target_dir(repository), repository / "target/agent"
                )


if __name__ == "__main__":
    unittest.main()
