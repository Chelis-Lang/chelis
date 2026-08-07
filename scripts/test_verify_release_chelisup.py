"""Tests for the portable chelisup release artifact verifier."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import verify_release_chelisup as verifier


class FakeRunner:
    def __init__(self, platform: str, version: str = "0.18.4") -> None:
        self.platform = platform
        self.version = version
        self.dynamic = False
        self.store_dependency = False

    def __call__(self, command: list[str]) -> str:
        if command[-1] == "--version":
            return f"chelisup {self.version}\n"
        if command[-1] == "--help":
            return "help\n"
        if command[0] == "file":
            architecture = (
                "ELF 64-bit LSB executable, x86-64"
                if self.platform == "linux-x86_64"
                else "Mach-O 64-bit executable arm64"
            )
            return f"artifact: {architecture}\n"
        if command[:2] == ["readelf", "-l"]:
            return "INTERP\n" if self.dynamic else "Program Headers:\n  LOAD\n"
        if command[:2] == ["readelf", "-d"]:
            return (
                "NEEDED lib.so\n" if self.dynamic else "There is no dynamic section.\n"
            )
        if command[:2] == ["otool", "-L"]:
            dependency = (
                "/nix/store/invalid/libiconv.2.dylib"
                if self.store_dependency
                else "/usr/lib/libSystem.B.dylib"
            )
            return f"artifact:\n\t{dependency} (compatibility version 1.0.0, current version 1.0.0)\n"
        raise AssertionError(f"unexpected command: {command}")


class VerifyReleaseChelisupTests(unittest.TestCase):
    def make_output(self, directory: Path, platform: str) -> Path:
        name = f"chelisup-{platform}"
        executable = directory / name
        executable.write_bytes(b"portable-chelisup\n")
        executable.chmod(0o755)
        digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        (directory / f"{name}.sha256").write_text(
            f"{digest}  {name}\n", encoding="utf-8"
        )
        return executable

    def test_linux_output_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_output(root, "linux-x86_64")
            verifier.verify_release_output(
                root,
                "linux-x86_64",
                "0.18.4",
                run_text=FakeRunner("linux-x86_64"),
            )

    def test_darwin_output_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_output(root, "darwin-arm64")
            verifier.verify_release_output(
                root,
                "darwin-arm64",
                "0.18.4",
                run_text=FakeRunner("darwin-arm64"),
            )

    def test_extra_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_output(root, "linux-x86_64")
            (root / "extra").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(verifier.VerificationError, "inventory"):
                verifier.verify_release_output(
                    root,
                    "linux-x86_64",
                    "0.18.4",
                    run_text=FakeRunner("linux-x86_64"),
                )

    def test_bad_checksum_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            executable = self.make_output(root, "linux-x86_64")
            executable.write_bytes(b"changed\n")
            executable.chmod(0o755)
            with self.assertRaisesRegex(verifier.VerificationError, "checksum"):
                verifier.verify_release_output(
                    root,
                    "linux-x86_64",
                    "0.18.4",
                    run_text=FakeRunner("linux-x86_64"),
                )

    def test_dynamic_linux_binary_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_output(root, "linux-x86_64")
            runner = FakeRunner("linux-x86_64")
            runner.dynamic = True
            with self.assertRaisesRegex(verifier.VerificationError, "static"):
                verifier.verify_release_output(
                    root,
                    "linux-x86_64",
                    "0.18.4",
                    run_text=runner,
                )

    def test_nix_store_darwin_dependency_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_output(root, "darwin-arm64")
            runner = FakeRunner("darwin-arm64")
            runner.store_dependency = True
            with self.assertRaisesRegex(verifier.VerificationError, "Nix store"):
                verifier.verify_release_output(
                    root,
                    "darwin-arm64",
                    "0.18.4",
                    run_text=runner,
                )

    def test_wrong_version_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_output(root, "linux-x86_64")
            with self.assertRaisesRegex(verifier.VerificationError, "version"):
                verifier.verify_release_output(
                    root,
                    "linux-x86_64",
                    "0.18.4",
                    run_text=FakeRunner("linux-x86_64", version="9.9.9"),
                )

    def test_unsupported_platform_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(verifier.VerificationError, "unsupported"):
                verifier.verify_release_output(
                    Path(raw),
                    "darwin-x86_64",
                    "0.18.4",
                    run_text=FakeRunner("darwin-arm64"),
                )

    def test_devenv_build_json_selects_release_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "build.json"
            path.write_text(
                json.dumps({"outputs.release-chelisup": "/nix/store/example"}),
                encoding="utf-8",
            )
            self.assertEqual(
                verifier.output_path_from_build_json(path),
                Path("/nix/store/example"),
            )

    def test_devenv_build_json_requires_release_key(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "build.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(verifier.VerificationError, "build result"):
                verifier.output_path_from_build_json(path)

    def test_stage_copies_exact_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            source = parent / "source"
            source.mkdir()
            self.make_output(source, "linux-x86_64")
            destination = parent / "staged"
            verifier.stage_release_output(
                source,
                destination,
                "linux-x86_64",
            )
            self.assertEqual(
                sorted(path.name for path in destination.iterdir()),
                ["chelisup-linux-x86_64", "chelisup-linux-x86_64.sha256"],
            )

    def test_stage_rejects_an_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            source = parent / "source"
            source.mkdir()
            self.make_output(source, "linux-x86_64")
            destination = parent / "staged"
            destination.mkdir()
            with self.assertRaisesRegex(verifier.VerificationError, "cannot stage"):
                verifier.stage_release_output(
                    source,
                    destination,
                    "linux-x86_64",
                )


if __name__ == "__main__":
    unittest.main()
