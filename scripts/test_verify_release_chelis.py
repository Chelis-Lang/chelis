"""Unit tests for the portable Linux chelis release verifier."""

from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import verify_release_chelis as mod


VERSION = "0.18.4"
PLATFORM = "linux-x86_64"
STEM = f"chelis-v{VERSION}-{PLATFORM}"


def _build_tarball(
    root: Path,
    *,
    platform: str = PLATFORM,
    drop: str | None = None,
    extra: str | None = None,
) -> Path:
    """Create a release tarball with the contracted tree in `root`."""
    stem = f"chelis-v{VERSION}-{platform}"
    files = {
        f"{stem}/bin/chelis": b"\x7fELF-fake",
        f"{stem}/lib/libchelis_runtime.a": b"!<arch>\n",
        f"{stem}/README.md": b"readme\n",
        f"{stem}/LICENSE": b"license\n",
    }
    files.update(
        {
            f"{stem}/include/{header}": b"// header\n"
            for header in mod.PUBLIC_RUNTIME_HEADERS
        }
    )
    if drop is not None:
        del files[drop]
    if extra is not None:
        files[extra] = b"extra\n"
    tarball = root / mod.asset_name(VERSION, platform)
    with tarfile.open(tarball, "w:gz") as archive:
        directories = sorted(
            {str(Path(name).parent) for name in files} | {stem}
        )
        for directory in directories:
            info = tarfile.TarInfo(directory)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return tarball


def _write_checksum(tarball: Path) -> Path:
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    checksum = tarball.with_name(tarball.name + ".sha256")
    checksum.write_text(f"{digest}  {tarball.name}\n", encoding="ascii")
    return checksum


def _make_valid_output(root: Path) -> None:
    tarball = _build_tarball(root)
    _write_checksum(tarball)


class VerifyReleaseOutputTests(unittest.TestCase):
    def test_valid_output_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _make_valid_output(root)
            mod.verify_release_output(root, VERSION, PLATFORM)

    def test_darwin_output_passes_under_its_platform(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tarball = _build_tarball(root, platform="darwin-arm64")
            _write_checksum(tarball)
            mod.verify_release_output(root, VERSION, "darwin-arm64")

    def test_an_unsupported_platform_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _make_valid_output(root)
            with self.assertRaisesRegex(mod.VerificationError, "unsupported"):
                mod.verify_release_output(root, VERSION, "windows-x86_64")

    def test_a_platform_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _make_valid_output(root)
            with self.assertRaisesRegex(mod.VerificationError, "inventory differs"):
                mod.verify_release_output(root, VERSION, "darwin-arm64")

    def test_extra_root_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _make_valid_output(root)
            (root / "stray.txt").write_text("stray\n", encoding="utf-8")
            with self.assertRaisesRegex(mod.VerificationError, "inventory differs"):
                mod.verify_release_output(root, VERSION, PLATFORM)

    def test_checksum_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tarball = _build_tarball(root)
            checksum = _write_checksum(tarball)
            checksum.write_text("0" * 64 + f"  {tarball.name}\n", encoding="ascii")
            with self.assertRaisesRegex(mod.VerificationError, "does not match"):
                mod.verify_release_output(root, VERSION, PLATFORM)

    def test_missing_member_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tarball = _build_tarball(root, drop=f"{STEM}/include/chelis_blas.h")
            _write_checksum(tarball)
            with self.assertRaisesRegex(mod.VerificationError, "chelis_blas.h"):
                mod.verify_release_output(root, VERSION, PLATFORM)

    def test_extra_member_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tarball = _build_tarball(root, extra=f"{STEM}/bin/chelisup")
            _write_checksum(tarball)
            with self.assertRaisesRegex(mod.VerificationError, "chelisup"):
                mod.verify_release_output(root, VERSION, PLATFORM)

    def test_asset_name_matches_the_installer_request(self) -> None:
        self.assertEqual(
            mod.asset_name(VERSION, "linux-x86_64"),
            f"chelis-v{VERSION}-linux-x86_64.tar.gz",
        )
        self.assertEqual(
            mod.asset_name(VERSION, "darwin-arm64"),
            f"chelis-v{VERSION}-darwin-arm64.tar.gz",
        )


class TagParityTests(unittest.TestCase):
    def test_tag_matching_the_workspace_version_passes(self) -> None:
        mod.verify_tag_parity(VERSION, "tag", f"v{VERSION}")

    def test_tag_differing_from_the_workspace_version_fails(self) -> None:
        with self.assertRaisesRegex(mod.VerificationError, "release tag differs"):
            mod.verify_tag_parity(VERSION, "tag", "v9.9.9")

    def test_branch_dispatch_skips_the_parity_check(self) -> None:
        mod.verify_tag_parity(VERSION, "branch", "some-branch")


class StageReleaseOutputTests(unittest.TestCase):
    def test_staging_copies_the_exact_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            _make_valid_output(source)
            destination = root / "stage"
            mod.stage_release_output(source, destination, VERSION, PLATFORM)
            mod.verify_release_output(destination, VERSION, PLATFORM)


if __name__ == "__main__":
    unittest.main()
