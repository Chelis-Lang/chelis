from __future__ import annotations

import importlib.util
import os
import pathlib
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "scripts" / "grad_selector_identity_oracle.py"


class GradSelectorIdentityOracleTest(unittest.TestCase):
    def test_oracle_names_each_acceptance_surface_and_the_fallback_guard(self) -> None:
        spec = importlib.util.spec_from_file_location("grad_selector_identity_oracle", ORACLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        labels = [command[0] for command in module.COMMANDS]
        self.assertEqual(
            labels,
            [
                "surf selector identity",
                "compiler preparation propagation",
                "CLI selector diagnostics",
            ],
        )
        self.assertIn("(0..wrt.len())", ORACLE.read_text())

    def test_oracle_overrides_a_foreign_shared_target_with_its_owned_target(self) -> None:
        spec = importlib.util.spec_from_file_location("grad_selector_identity_oracle", ORACLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        foreign = ROOT.parent / "shared-cargo-target"
        with mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": str(foreign)}):
            environment = module.oracle_environment()

        self.assertEqual(
            pathlib.Path(environment["CARGO_TARGET_DIR"]),
            ROOT / "target" / "agents" / "1955-oracle",
        )
        self.assertNotEqual(
            pathlib.Path(environment["CARGO_TARGET_DIR"]).resolve(),
            foreign.resolve(),
        )


if __name__ == "__main__":
    unittest.main()
