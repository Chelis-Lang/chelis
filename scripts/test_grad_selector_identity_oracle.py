from __future__ import annotations

import importlib.util
import pathlib
import unittest


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


if __name__ == "__main__":
    unittest.main()
