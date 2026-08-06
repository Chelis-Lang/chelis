#!/usr/bin/env python3
"""Tests for scripts/regenerate_cargo_nix.py and the committed-graph contract."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "regenerate_cargo_nix.py"

_spec = importlib.util.spec_from_file_location("regenerate_cargo_nix", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
regen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regen)

CANONICAL_COMMAND = "generate --no-default-features --features chelis-cli/smt"


class FakeRunner:
    """Simulate crate2nix by writing predetermined bytes to Cargo.nix."""

    def __init__(self, produces: bytes, returncode: int = 0) -> None:
        self.produces = produces
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, argv, cwd):  # type: ignore[no-untyped-def]
        self.calls.append(list(argv))
        if self.returncode == 0:
            (Path(cwd) / regen.CARGO_NIX).write_bytes(self.produces)

        class _Completed:
            pass

        completed = _Completed()
        completed.returncode = self.returncode
        return completed


def _seed(tmp: Path, cargo_nix: bytes | None) -> Path:
    (tmp / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    if cargo_nix is not None:
        (tmp / regen.CARGO_NIX).write_bytes(cargo_nix)
    return tmp


class RegenerateTests(unittest.TestCase):
    def test_regenerate_invokes_the_canonical_command(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = _seed(Path(directory), b"old\n")
            runner = FakeRunner(produces=b"new\n")
            regen.regenerate(root, runner)
        self.assertEqual(
            runner.calls, [["crate2nix", *regen.CANONICAL_ARGS]]
        )
        self.assertEqual(" ".join(regen.CANONICAL_ARGS), CANONICAL_COMMAND)

    def test_regenerate_raises_on_tool_failure(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = _seed(Path(directory), b"old\n")
            runner = FakeRunner(produces=b"", returncode=7)
            with self.assertRaises(SystemExit):
                regen.regenerate(root, runner)


class CheckTests(unittest.TestCase):
    def test_check_passes_when_committed_matches_fresh(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = _seed(Path(directory), b"graph-bytes\n")
            runner = FakeRunner(produces=b"graph-bytes\n")
            self.assertEqual(regen.check(root, runner), 0)
            # The tree is left unchanged.
            self.assertEqual(
                (root / regen.CARGO_NIX).read_bytes(), b"graph-bytes\n"
            )

    def test_check_fails_and_restores_on_drift(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = _seed(Path(directory), b"stale\n")
            runner = FakeRunner(produces=b"fresh\n")
            self.assertEqual(regen.check(root, runner), 1)
            # Drift is reported, but the committed bytes are restored.
            self.assertEqual((root / regen.CARGO_NIX).read_bytes(), b"stale\n")

    def test_check_fails_when_graph_is_absent(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = _seed(Path(directory), None)
            runner = FakeRunner(produces=b"fresh\n")
            self.assertEqual(regen.check(root, runner), 1)
            self.assertEqual(runner.calls, [])


class ContractParityTests(unittest.TestCase):
    """The canonical command must agree with the Nix and Devenv paths."""

    def test_packages_nix_documents_the_same_command(self) -> None:
        text = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        self.assertIn(CANONICAL_COMMAND, text)
        self.assertIn('import (root + "/Cargo.nix")', text)

    def test_devenv_command_delegates_to_this_worker(self) -> None:
        text = (REPO_ROOT / "devenv" / "commands.nix").read_text(encoding="utf-8")
        self.assertIn("regenerate_cargo_nix.py", text)
        self.assertIn('"regenerate-crate2nix"', text)
        self.assertIn("config.outputs.crate2nix", text)

    def test_worker_guidance_uses_the_devenv_command(self) -> None:
        text = (REPO_ROOT / "scripts" / "regenerate_cargo_nix.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("devenv shell -- regenerate-crate2nix", text)
        self.assertNotIn("nix run .#regenerate-crate2nix", text)

    def test_devenv_composes_ci_for_shared_tools(self) -> None:
        yaml = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        self.assertIn("ci/devenv/consumer", yaml)

    def test_committed_graph_carries_the_pinned_banner(self) -> None:
        cargo_nix = REPO_ROOT / regen.CARGO_NIX
        self.assertTrue(cargo_nix.is_file(), "Cargo.nix must be committed")
        head = cargo_nix.read_text(encoding="utf-8", errors="replace")[:512]
        self.assertIn("@generated by crate2nix 0.15.0", head)

    def test_flake_drops_the_crate2nix_input_and_ifd(self) -> None:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        # crate2nix now comes from the ci consumer module, not a flake input.
        self.assertNotIn("github:nix-community/crate2nix", flake)
        self.assertNotIn("allow-import-from-derivation", flake)


if __name__ == "__main__":
    unittest.main()
