#!/usr/bin/env python3
"""Structural tests for the pinned Devenv/Kache documentation toolchain."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
KACHE_URL = "github:kunobi-ninja/kache/v0.16.0"
KACHE_REVISION = "a21d020142b1248537cd548ccde99a04c0a44820"
KACHE_SOURCE_HASH = "sha256-mrd4hlV0UXWLuo6GQXz44w1q0rrwzqvqlgcex9BHA4Q="
KACHE_CARGO_HASH = "sha256-VQJB5kGyXUjmfcKMeD2ggllbloeKXOpwjWOCVVsb0Rk="


class ManagedToolchainContractTests(unittest.TestCase):
    def test_kache_source_and_build_recipe_are_exact(self) -> None:
        module = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        for fragment in (
            'owner = "kunobi-ninja";',
            'repo = "kache";',
            f'rev = "{KACHE_REVISION}";',
            f'hash = "{KACHE_SOURCE_HASH}";',
            f'cargoHash = "{KACHE_CARGO_HASH}";',
            'version = "0.16.0";',
            'pkgs.runCommand "kache-0.16.0-build-source"',
            "cargoTestFlags",
        ):
            self.assertIn(fragment, module)

    def test_repository_kache_policy_is_configuration_invariant(self) -> None:
        config = (REPO_ROOT / ".kache.toml").read_text(encoding="utf-8")
        self.assertIn("ignore_env = true", config)
        self.assertIn("cache_executables = true", config)
        self.assertIn("heartbeat_secs = 30", config)

        module = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        for fragment in (
            "buildRustPackage",
            "kache-0.16.0-chelis-contract.patch",
            "kache-0.16.0-relocatable-macos-executables.patch",
            'RUSTC_WRAPPER = "${patchedKache}/bin/kache";',
            'CARGO_BUILD_RUSTC_WRAPPER = "${patchedKache}/bin/kache";',
            'CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER = "";',
            'KACHE_CONFIG = "${config.devenv.root}/.kache.toml";',
            'KACHE_SCHEMA_27_WRAPPER = "${legacyKache}/bin/kache";',
            "legacyKache",
            "patchedKache",
        ):
            self.assertIn(fragment, module)

    def test_managed_tools_and_darwin_native_compiler_are_explicit(self) -> None:
        module = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        for package in ("mdbook", "pyright"):
            self.assertIn(package, module)
        self.assertIn('CC_aarch64_apple_darwin = "${pkgs.stdenv.cc}/bin/cc";', module)
        self.assertIn('CXX_aarch64_apple_darwin = "${pkgs.stdenv.cc}/bin/c++";', module)
        self.assertIn('CRATE_CC_NO_DEFAULTS = "1";', module)

    def test_enter_shell_export_publish_is_atomic_and_python_owned(self) -> None:
        root = (REPO_ROOT / "devenv.nix").read_text(encoding="utf-8")
        module = (REPO_ROOT / "devenv/entry-shell.nix").read_text(encoding="utf-8")
        helper = (REPO_ROOT / "scripts/write_devenv_load_exports.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("./devenv/entry-shell.nix", root)
        self.assertIn("lib.mkForce", module)
        self.assertIn("write_devenv_load_exports.py", module)
        self.assertIn("os.replace", helper)
        self.assertIn("os.fsync", helper)

    def test_kache_patch_carries_both_behavioral_repairs(self) -> None:
        patch = (
            REPO_ROOT / "nix/patches/kache-0.16.0-chelis-contract.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("over typical", patch)
        self.assertIn("doctor_status", patch)
        self.assertIn('"Stale locks"', patch)
        self.assertIn("below_equal_and_above_typical", patch)

    def test_kache_executable_patch_strips_donor_paths_and_fails_closed(self) -> None:
        patch = (
            REPO_ROOT
            / "nix/patches/kache-0.16.0-relocatable-macos-executables.patch"
        ).read_text(encoding="utf-8")
        self.assertIn('Command::new("strip")', patch)
        self.assertIn('arg("-S")', patch)
        self.assertIn('Contents/Resources/Relocations', patch)
        self.assertIn('missing relocatable executable staging file', patch)
        self.assertIn('compiler-owned output untouched', patch)
        self.assertIn('artifact.store_name == executable_store_name', patch)
        self.assertIn('ArtifactKind::Other("extensionless")', patch)

    def test_kache_executable_patch_namespaces_the_new_cache_representation(self) -> None:
        patch = (
            REPO_ROOT
            / "nix/patches/kache-0.16.0-relocatable-macos-executables.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("-pub(crate) const CACHE_KEY_VERSION: u32 = 27;", patch)
        self.assertIn("+pub(crate) const CACHE_KEY_VERSION: u32 = 28;", patch)

    def test_kache_schema_fixture_is_oracle_only_and_pre_representation(self) -> None:
        module = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        fixture = module.split("legacyKache =", 1)[1].split("patchedKache =", 1)[0]
        self.assertIn("kache-0.16.0-chelis-contract.patch", fixture)
        self.assertNotIn("kache-0.16.0-relocatable-macos-executables.patch", fixture)
        self.assertIn("doCheck = false;", fixture)

    def test_hosted_and_local_mdbook_versions_match(self) -> None:
        workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("tool: mdbook@0.5.2", workflow)


if __name__ == "__main__":
    unittest.main()
