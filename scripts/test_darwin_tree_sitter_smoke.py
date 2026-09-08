#!/usr/bin/env python3
"""Tests for Darwin tree-sitter smoke diagnostic classification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from darwin_tree_sitter_smoke import target_mismatch_diagnostics


class DarwinTreeSitterDiagnosticsTests(unittest.TestCase):
    def test_target_mismatch_warning_is_rejected(self) -> None:
        output = (
            "warning: tree-sitter-chelis@0.18.6: Warning: supplying the --target "
            "arm64-apple-macosx != arm64-apple-darwin argument to a nix-wrapped "
            "compiler may not work correctly - cc-wrapper is currently not designed "
            "with multi-target compilers in mind.\n"
        )
        self.assertEqual(target_mismatch_diagnostics(output), [output.strip()])

    def test_unrelated_warning_is_not_hidden(self) -> None:
        self.assertEqual(
            target_mismatch_diagnostics("warning: an unrelated build warning\n"),
            [],
        )


if __name__ == "__main__":
    unittest.main()
