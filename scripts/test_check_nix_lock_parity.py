"""Tests for the shared Nix input lock parity checker.

Run with `.venv/bin/python scripts/test_check_nix_lock_parity.py`.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_nix_lock_parity.py"
FIXTURES = REPO_ROOT / "scripts" / "fixtures" / "nix_lock_parity"
MATCHING_DEVENV = FIXTURES / "matching-devenv.lock"


def run_checker(flake_lock: Path, devenv_lock: Path = MATCHING_DEVENV) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--flake-lock",
            str(flake_lock),
            "--devenv-lock",
            str(devenv_lock),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class NixLockParityTests(unittest.TestCase):
    def test_matching_nixpkgs_and_rust_overlay_revisions_pass(self) -> None:
        completed = run_checker(FIXTURES / "matching-flake.lock")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("nixpkgs", completed.stdout)
        self.assertIn("rust-overlay", completed.stdout)
        self.assertIn("matching shared input revisions", completed.stdout)

    def test_mismatched_nixpkgs_revision_names_both_nodes_and_revisions(self) -> None:
        completed = run_checker(FIXTURES / "mismatched-nixpkgs-flake.lock")
        self.assertEqual(completed.returncode, 1)
        self.assertIn("flake.lock:nixpkgs", completed.stderr)
        self.assertIn("devenv.lock:nixpkgs-src", completed.stderr)
        self.assertIn("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", completed.stderr)
        self.assertIn("1111111111111111111111111111111111111111", completed.stderr)

    def test_mismatched_rust_overlay_revision_names_both_nodes_and_revisions(self) -> None:
        completed = run_checker(FIXTURES / "mismatched-rust-overlay-flake.lock")
        self.assertEqual(completed.returncode, 1)
        self.assertIn("flake.lock:rust-overlay", completed.stderr)
        self.assertIn("devenv.lock:rust-overlay", completed.stderr)
        self.assertIn("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", completed.stderr)
        self.assertIn("2222222222222222222222222222222222222222", completed.stderr)

    def test_missing_revision_fails_at_the_parse_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            invalid = Path(raw_dir) / "invalid.lock"
            invalid.write_text('{"nodes":{"nixpkgs":{"locked":{}}}}', encoding="utf-8")
            completed = run_checker(invalid)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("invalid lock file", completed.stderr)
        self.assertIn("nixpkgs", completed.stderr)
        self.assertIn("rev", completed.stderr)


if __name__ == "__main__":
    unittest.main()
