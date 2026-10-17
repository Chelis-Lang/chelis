"""Tests for the native-system table of the lane-check proof-scope generator (chelis#3818)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lane_check_scope  # noqa: E402


class NativeArchitectureTests(unittest.TestCase):
    def test_each_supported_linux_system_accepts_its_native_host(self) -> None:
        for nix_system, machine in (("x86_64-linux", "x86_64"), ("aarch64-linux", "aarch64")):
            with self.subTest(nix_system=nix_system):
                self.assertEqual(
                    lane_check_scope.native_architecture(nix_system, "Linux", machine), machine
                )

    def test_the_table_names_exactly_the_linux_flake_systems(self) -> None:
        self.assertEqual(sorted(lane_check_scope.PROOF_SYSTEMS), ["aarch64-linux", "x86_64-linux"])

    def test_unsupported_or_foreign_hosts_are_rejected(self) -> None:
        cases = (
            ("aarch64-darwin", "Darwin", "arm64"),
            ("x86_64-darwin", "Darwin", "x86_64"),
            ("riscv64-linux", "Linux", "riscv64"),
            ("", "Linux", "x86_64"),
            ("aarch64-linux", "Linux", "x86_64"),
            ("x86_64-linux", "Linux", "aarch64"),
            ("aarch64-linux", "Linux", "arm64"),
            ("x86_64-linux", "Darwin", "x86_64"),
        )
        for nix_system, system, machine in cases:
            with self.subTest(nix_system=nix_system, system=system, machine=machine):
                with self.assertRaises(ValueError):
                    lane_check_scope.native_architecture(nix_system, system, machine)


class LinuxTripleTests(unittest.TestCase):
    def test_triples_that_name_the_architecture_and_linux_pass(self) -> None:
        for triple, architecture in (
            ("x86_64-unknown-linux-gnu", "x86_64"),
            ("x86_64-pc-linux-gnu", "x86_64"),
            ("aarch64-unknown-linux-gnu", "aarch64"),
        ):
            with self.subTest(triple=triple):
                self.assertTrue(lane_check_scope.is_linux_triple_for(triple, architecture))

    def test_other_architectures_and_systems_fail(self) -> None:
        for triple, architecture in (
            ("aarch64-unknown-linux-gnu", "x86_64"),
            ("x86_64-unknown-linux-gnu", "aarch64"),
            ("aarch64-apple-darwin", "aarch64"),
            ("aarch64_be-unknown-linux-gnu", "aarch64"),
            ("x86_64-unknown-linuxish-gnu", "x86_64"),
            ("", "x86_64"),
        ):
            with self.subTest(triple=triple):
                self.assertFalse(lane_check_scope.is_linux_triple_for(triple, architecture))


if __name__ == "__main__":
    unittest.main()
