"""Tests for the fail-safe Nix graph-change detector."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "ci_detect_nix_graph_changes",
        here / "ci_detect_nix_graph_changes.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


detector = _load_module()


class NixGraphChangeDetectorTests(unittest.TestCase):
    def test_every_cargo_manifest_selects_regeneration(self) -> None:
        for path in ("Cargo.toml", "crates/chelis-cli/Cargo.toml"):
            with self.subTest(path=path):
                self.assertTrue(detector.requires_regeneration([path]))

    def test_lock_graph_and_pin_changes_select_regeneration(self) -> None:
        paths = (
            ".cargo/config",
            ".cargo/config.toml",
            "Cargo.lock",
            "Cargo.nix",
            "crate-hashes.json",
            "crate2nix.json",
            "flake.nix",
            "flake.lock",
            "devenv.yaml",
            "devenv.lock",
            "rust-toolchain.toml",
            "nix/workspace.nix",
            "nix/source.nix",
            "nix/crate2nix-regeneration.nix",
            "scripts/check_crate2nix_sync.py",
            "scripts/ci_detect_nix_graph_changes.py",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(detector.requires_regeneration([path]))

    def test_rust_source_change_does_not_select_regeneration(self) -> None:
        self.assertFalse(
            detector.requires_regeneration(["crates/chelis-cli/src/main.rs"])
        )

    def test_unrelated_nix_change_does_not_select_regeneration(self) -> None:
        self.assertFalse(detector.requires_regeneration(["nix/artifacts.nix"]))

    def test_any_graph_input_selects_regeneration_in_a_mixed_change(self) -> None:
        self.assertTrue(
            detector.requires_regeneration(
                ["crates/chelis-cli/src/main.rs", "crates/chelis-cli/Cargo.toml"]
            )
        )

    def test_empty_input_selects_regeneration(self) -> None:
        self.assertTrue(detector.requires_regeneration([]))


if __name__ == "__main__":
    unittest.main()
