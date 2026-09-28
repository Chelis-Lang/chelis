#!/usr/bin/env python3
"""Tests for the repository-owned Pyright/Pylance source graph."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_pyright_scope as scope


EXPECTED_INCLUDES = {
    ".github/scripts",
    "assets/mascot",
    "bindings/python",
    "crates/chelis-cli/tests/fixtures/pseudo_nautilus/parity",
    "docs/investigations/ci_diet_2026_09",
    "docs/investigations/probes",
    "py/src",
    "py/tests",
    "scripts",
    "tests",
}
REQUIRED_EXCLUDES = {
    "**/.devenv",
    "**/.git",
    "**/.venv",
    "**/__pycache__",
    "**/node_modules",
    "**/target",
}


class PyrightScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(scope.CONFIG_PATH.read_text(encoding="utf-8"))

    def test_checked_in_config_names_the_complete_intended_source_graph(self) -> None:
        self.assertEqual(set(self.config["include"]), EXPECTED_INCLUDES)
        self.assertTrue(REQUIRED_EXCLUDES.issubset(self.config["exclude"]))
        self.assertEqual(scope.validate_scope(self.config), [])

    def test_dropping_a_real_source_root_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["include"].remove("scripts")
        failures = scope.validate_scope(mutated)
        self.assertTrue(any("scripts/" in failure for failure in failures), failures)

    def test_dropping_a_generated_tree_exclusion_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["exclude"].remove("**/.devenv")
        failures = scope.validate_scope(mutated)
        self.assertTrue(any(".devenv" in failure for failure in failures), failures)

    def test_absolute_or_symlinked_include_roots_are_rejected(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["include"].append("/tmp")
        failures = scope.validate_scope(mutated)
        self.assertTrue(any("relative" in failure for failure in failures), failures)


if __name__ == "__main__":
    unittest.main()
