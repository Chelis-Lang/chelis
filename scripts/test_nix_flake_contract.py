"""Contract tests for the public Nix package and application surface."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NIX = shutil.which("nix")
REQUIRES_NIX = unittest.skipUnless(NIX, "Nix is not installed")
EXPECTED_SYSTEMS = ["aarch64-darwin", "x86_64-linux"]
EXPECTED_PACKAGES = ["chelis", "chelis-runtime", "chelisup", "default"]
EXPECTED_APPS = ["chelis", "chelisup", "default"]
EXPECTED_HEADERS = [
    "chelis_runtime.h",
    "chelis_runtime_views.h",
    "chelis_runtime_dtype.h",
    "chelis_blas.h",
    "chelis_simd.h",
    "chelis_math.h",
]
def nix_json(*args: str) -> object:
    completed = subprocess.run(
        [NIX, *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return json.loads(completed.stdout)


def nix_raw(*args: str) -> str:
    completed = subprocess.run(
        [NIX, *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return completed.stdout.strip()


def assert_automatic_crate2nix_contract(
    flake: str,
    packages: str,
    source: str,
    *,
    cargo_nix_exists: bool,
) -> None:
    required_flake = "allow-import-from-derivation = true;"
    if required_flake not in flake:
        raise AssertionError("the flake must enable import from derivation")

    required_packages = (
        'pkgs.callPackage (crate2nix + "/tools.nix")',
        "generatedCargoNix",
        "additionalCargoNixArgs",
        '"--no-default-features"',
        '"--features"',
        '"chelis-cli/smt"',
        'CARGO_NET_OFFLINE = "true";',
        "import generatedCargoNix",
    )
    missing_packages = [item for item in required_packages if item not in packages]
    if missing_packages:
        raise AssertionError(
            f"the automatic crate2nix contract is incomplete: {missing_packages!r}"
        )
    if 'import (root + "/Cargo.nix")' in packages:
        raise AssertionError("the package layer must not import a repository Cargo.nix")
    if cargo_nix_exists:
        raise AssertionError("the repository must not track Cargo.nix")

    required_source = ('"Cargo.lock"', '"tree-sitter-chelis"')
    missing_source = [item for item in required_source if item not in source]
    if missing_source:
        raise AssertionError(
            f"the crate2nix generator source is incomplete: {missing_source!r}"
        )


class NixFlakeContractTests(unittest.TestCase):

    @REQUIRES_NIX
    def test_supported_systems_are_exact(self) -> None:
        systems = nix_json("eval", "--json", ".#packages", "--apply", "builtins.attrNames")
        self.assertEqual(systems, EXPECTED_SYSTEMS)

    @REQUIRES_NIX
    def test_package_names_are_exact_on_each_system(self) -> None:
        for system in EXPECTED_SYSTEMS:
            with self.subTest(system=system):
                names = nix_json(
                    "eval",
                    "--json",
                    f".#packages.{system}",
                    "--apply",
                    "builtins.attrNames",
                )
                self.assertEqual(names, EXPECTED_PACKAGES)

    @REQUIRES_NIX
    def test_application_names_are_exact_on_each_system(self) -> None:
        for system in EXPECTED_SYSTEMS:
            with self.subTest(system=system):
                names = nix_json(
                    "eval",
                    "--json",
                    f".#apps.{system}",
                    "--apply",
                    "builtins.attrNames",
                )
                self.assertEqual(names, EXPECTED_APPS)

    @REQUIRES_NIX
    def test_default_package_and_application_are_exact_native_aliases(self) -> None:
        system = nix_raw(
            "eval",
            "--impure",
            "--raw",
            "--expr",
            "builtins.currentSystem",
        )
        self.assertIn(system, EXPECTED_SYSTEMS)
        default_package = nix_raw(
            "eval", "--raw", f".#packages.{system}.default.drvPath"
        )
        chelis_package = nix_raw(
            "eval", "--raw", f".#packages.{system}.chelis.drvPath"
        )
        self.assertEqual(default_package, chelis_package)

        default_program = nix_raw(
            "eval", "--raw", f".#apps.{system}.default.program"
        )
        chelis_program = nix_raw(
            "eval", "--raw", f".#apps.{system}.chelis.program"
        )
        self.assertEqual(default_program, chelis_program)

    @REQUIRES_NIX
    def test_nix_evaluation_fails_when_ifd_is_disabled(self) -> None:
        system = nix_raw(
            "eval",
            "--impure",
            "--raw",
            "--expr",
            "builtins.currentSystem",
        )
        completed = subprocess.run(
            [
                NIX,
                "eval",
                "--option",
                "allow-import-from-derivation",
                "false",
                "--raw",
                f".#checks.{system}.crate2nixGeneration.drvPath",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("allow-import-from-derivation", completed.stderr)
        self.assertIn("disabled", completed.stderr)

    @REQUIRES_NIX
    def test_contract_stubs_cover_package_shape_and_behavior(self) -> None:
        contracts = nix_json("eval", "--json", "--file", "nix/contracts.nix")
        shapes = contracts["packageShapes"]
        self.assertEqual(
            shapes["chelis"]["allowedProductExecutables"],
            ["bin/chelis"],
        )
        self.assertEqual(shapes["chelis"]["forbidden"], ["bin/chelisup"])
        self.assertEqual(
            shapes["chelis"]["inventory"],
            [
                "bin",
                "include",
                "lib",
                *shapes["chelis"]["required"],
            ],
        )
        self.assertEqual(shapes["chelis-runtime"]["allowedProductExecutables"], [])
        self.assertEqual(
            shapes["chelis-runtime"]["forbidden"],
            ["bin/chelis", "bin/chelisup"],
        )
        self.assertEqual(
            shapes["chelis-runtime"]["inventory"],
            [
                "include",
                "lib",
                *shapes["chelis-runtime"]["required"],
            ],
        )
        self.assertEqual(
            shapes["chelisup"]["required"],
            ["bin/chelisup", "libexec/chelisup"],
        )
        self.assertEqual(
            shapes["chelisup"]["allowedProductExecutables"],
            ["bin/chelisup", "libexec/chelisup"],
        )
        self.assertEqual(shapes["chelisup"]["forbidden"], ["bin/chelis"])
        self.assertEqual(
            shapes["chelisup"]["inventory"],
            ["bin", "libexec", "bin/chelisup", "libexec/chelisup"],
        )
        for header in EXPECTED_HEADERS:
            self.assertIn(f"include/{header}", shapes["chelis"]["required"])
            self.assertIn(f"include/{header}", shapes["chelis-runtime"]["required"])

    @REQUIRES_NIX
    def test_chelisup_install_roots_the_nix_closure_and_self_uninstall_removes_it(self) -> None:
        package = Path(
            nix_raw("build", "--no-link", "--print-out-paths", ".#chelisup")
        )
        manifest = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"))
        version = manifest["workspace"]["package"]["version"]

        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            toolchain = home / "toolchains" / version / "bin" / "chelis"
            toolchain.parent.mkdir(parents=True)
            toolchain.write_text("placeholder toolchain\n", encoding="utf-8")

            environment = os.environ.copy()
            environment["CHELIS_HOME"] = str(home)
            environment["PATH"] = ""
            installed = subprocess.run(
                [package / "bin" / "chelisup", "install", version],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(installed.returncode, 0, installed.stderr)

            gc_root = home / "nix-gcroots" / "chelisup"
            self.assertTrue(gc_root.is_symlink(), f"missing Nix GC root: {gc_root}")
            self.assertEqual(gc_root.resolve(), package.resolve())
            staging_root = home / "nix-gcroots" / "chelisup.next"
            partial_root = home / "nix-gcroots" / "chelisup.partial"
            self.assertFalse(staging_root.exists())
            self.assertFalse(staging_root.is_symlink())
            self.assertFalse(partial_root.exists())
            self.assertFalse(partial_root.is_symlink())

            real_binary = package / "libexec" / "chelisup"
            installed_binary = home / "bin" / "chelisup"
            installed_shim = home / "bin" / "chelis"
            package_wrapper = package / "bin" / "chelisup"
            self.assertEqual(installed_binary.read_bytes(), package_wrapper.read_bytes())
            self.assertNotEqual(installed_binary.read_bytes(), real_binary.read_bytes())
            self.assertEqual(installed_shim.read_bytes(), real_binary.read_bytes())

            reinstalled = subprocess.run(
                [installed_binary, "install", version],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(reinstalled.returncode, 0, reinstalled.stderr)
            self.assertEqual(installed_binary.read_bytes(), package_wrapper.read_bytes())

            copied_help = subprocess.run(
                [installed_binary, "--help"],
                cwd=REPO_ROOT,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(copied_help.returncode, 0, copied_help.stderr.decode())

            staging_root.symlink_to(package)
            partial_root.symlink_to(package)
            removed = subprocess.run(
                [installed_binary, "self", "uninstall"],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(removed.returncode, 0, removed.stderr)
            for root in (gc_root, staging_root, partial_root):
                self.assertFalse(root.exists())
                self.assertFalse(root.is_symlink())

    @REQUIRES_NIX
    def test_failed_self_uninstall_preserves_all_gc_roots(self) -> None:
        package = Path(
            nix_raw("build", "--no-link", "--print-out-paths", ".#chelisup")
        )

        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            bin_dir = home / "bin"
            bin_dir.mkdir()
            (bin_dir / "chelis").mkdir()
            roots = tuple(
                home / "nix-gcroots" / name
                for name in ("chelisup", "chelisup.next", "chelisup.partial")
            )
            roots[0].parent.mkdir()
            for root in roots:
                root.symlink_to(package)

            environment = os.environ.copy()
            environment["CHELIS_HOME"] = str(home)
            environment["PATH"] = ""
            failed = subprocess.run(
                [package / "bin" / "chelisup", "self", "uninstall"],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            for root in roots:
                self.assertTrue(root.is_symlink())
                self.assertEqual(root.resolve(), package.resolve())

    @REQUIRES_NIX
    def test_partial_chelisup_copy_keeps_the_partial_gc_root(self) -> None:
        package = Path(
            nix_raw("build", "--no-link", "--print-out-paths", ".#chelisup")
        )
        manifest = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"))
        version = manifest["workspace"]["package"]["version"]

        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            toolchain = home / "toolchains" / version / "bin" / "chelis"
            toolchain.parent.mkdir(parents=True)
            toolchain.write_text("placeholder toolchain\n", encoding="utf-8")

            bin_dir = home / "bin"
            bin_dir.mkdir()
            (bin_dir / "chelisup").mkdir()
            gc_root = home / "nix-gcroots" / "chelisup"
            gc_root.parent.mkdir(parents=True)
            prior_target = package / "bin" / "chelisup"
            gc_root.symlink_to(prior_target)

            environment = os.environ.copy()
            environment["CHELIS_HOME"] = str(home)
            environment["PATH"] = ""
            failed = subprocess.run(
                [package / "bin" / "chelisup", "install", version],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(gc_root.resolve(), prior_target.resolve())
            real_binary = package / "libexec" / "chelisup"
            self.assertEqual((bin_dir / "chelis").read_bytes(), real_binary.read_bytes())
            staging_root = home / "nix-gcroots" / "chelisup.next"
            partial_root = home / "nix-gcroots" / "chelisup.partial"
            self.assertFalse(staging_root.exists())
            self.assertFalse(staging_root.is_symlink())
            self.assertTrue(partial_root.is_symlink())
            self.assertEqual(partial_root.resolve(), package.resolve())

    @REQUIRES_NIX
    def test_stale_staging_root_recovery_preserves_required_closures(self) -> None:
        package = Path(
            nix_raw("build", "--no-link", "--print-out-paths", ".#chelisup")
        )
        real_binary = package / "libexec" / "chelisup"

        for has_partial_copy in (False, True):
            with self.subTest(has_partial_copy=has_partial_copy):
                with tempfile.TemporaryDirectory() as raw_home:
                    home = Path(raw_home)
                    roots = home / "nix-gcroots"
                    roots.mkdir()
                    stable_root = roots / "chelisup"
                    partial_root = roots / "chelisup.partial"
                    staging_root = roots / "chelisup.next"
                    stable_root.symlink_to(real_binary)
                    prior_partial_target = package / "bin" / "chelisup"
                    partial_root.symlink_to(prior_partial_target)
                    staging_root.symlink_to(package)

                    if has_partial_copy:
                        bin_dir = home / "bin"
                        bin_dir.mkdir()
                        (bin_dir / "chelis").write_bytes(real_binary.read_bytes())

                    environment = os.environ.copy()
                    environment["CHELIS_HOME"] = str(home)
                    environment["PATH"] = ""
                    failed = subprocess.run(
                        [package / "bin" / "chelisup", "install", "invalid-version"],
                        cwd=REPO_ROOT,
                        env=environment,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    self.assertNotEqual(failed.returncode, 0)
                    self.assertEqual(stable_root.resolve(), real_binary.resolve())
                    expected_partial = package if has_partial_copy else prior_partial_target
                    self.assertEqual(partial_root.resolve(), expected_partial.resolve())
                    self.assertFalse(staging_root.exists())
                    self.assertFalse(staging_root.is_symlink())

    @REQUIRES_NIX
    def test_stale_staging_recovery_promotes_the_staged_package(self) -> None:
        # The staged package must differ from the invoking wrapper's package:
        # a same-package staging root lets the later failure-path promotion
        # mask a broken recovery branch.
        package = Path(
            nix_raw("build", "--no-link", "--print-out-paths", ".#chelisup")
        )
        with tempfile.TemporaryDirectory() as raw_fixture:
            fixture = Path(raw_fixture) / "staged-package"
            (fixture / "libexec").mkdir(parents=True)
            (fixture / "libexec" / "chelisup").write_bytes(
                b"synthetic staged chelisup binary\n"
            )
            staged_package = Path(
                nix_raw(
                    "store",
                    "add",
                    "--name",
                    "chelisup-staged-fixture",
                    str(fixture),
                )
            )

        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            bin_dir = home / "bin"
            bin_dir.mkdir()
            (bin_dir / "chelis").write_bytes(
                (staged_package / "libexec" / "chelisup").read_bytes()
            )
            roots = home / "nix-gcroots"
            roots.mkdir()
            staging_root = roots / "chelisup.next"
            staging_root.symlink_to(staged_package)

            environment = os.environ.copy()
            environment["CHELIS_HOME"] = str(home)
            environment["PATH"] = ""
            failed = subprocess.run(
                [package / "bin" / "chelisup", "install", "invalid-version"],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            partial_root = roots / "chelisup.partial"
            self.assertTrue(
                partial_root.is_symlink(),
                "recovery must promote the partial root for the staged package",
            )
            self.assertEqual(partial_root.resolve(), staged_package.resolve())
            self.assertFalse(staging_root.exists())
            self.assertFalse(staging_root.is_symlink())

    @REQUIRES_NIX
    def test_failed_chelisup_install_preserves_the_existing_gc_root(self) -> None:
        package = Path(
            nix_raw("build", "--no-link", "--print-out-paths", ".#chelisup")
        )
        with tempfile.TemporaryDirectory() as raw_home:
            home = Path(raw_home)
            gc_root = home / "nix-gcroots" / "chelisup"
            partial_root = home / "nix-gcroots" / "chelisup.partial"
            gc_root.parent.mkdir(parents=True)
            prior_target = package / "libexec" / "chelisup"
            prior_partial_target = package / "bin" / "chelisup"
            gc_root.symlink_to(prior_target)
            partial_root.symlink_to(prior_partial_target)

            environment = os.environ.copy()
            environment["CHELIS_HOME"] = str(home)
            environment["PATH"] = ""
            failed = subprocess.run(
                [package / "bin" / "chelisup", "install", "invalid-version"],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(gc_root.resolve(), prior_target.resolve())
            self.assertEqual(partial_root.resolve(), prior_partial_target.resolve())
            staging_root = home / "nix-gcroots" / "chelisup.next"
            self.assertFalse(staging_root.exists())
            self.assertFalse(staging_root.is_symlink())


class NixSourceContractTests(unittest.TestCase):
    def test_package_shape_rejects_every_undeclared_path(self) -> None:
        checks = (REPO_ROOT / "nix" / "checks.nix").read_text(encoding="utf-8")
        self.assertIn("assert_exact_inventory", checks)
        self.assertIn('find "$package" -mindepth 1 -printf \'%P\\n\' | sort', checks)

    def test_static_cvc5_library_does_not_force_a_static_executable(self) -> None:
        cvc5 = (REPO_ROOT / "nix" / "cvc5.nix").read_text(encoding="utf-8")
        self.assertIn('"-DBUILD_SHARED_LIBS=OFF"', cvc5)
        self.assertIn('"-DSTATIC_BINARY=OFF"', cvc5)

    def test_flake_pins_crate2nix_as_a_non_flake_input(self) -> None:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        self.assertIn('url = "github:nix-community/crate2nix/0.15.0";', flake)
        self.assertRegex(
            flake,
            r"crate2nix\s*=\s*\{[^}]*flake\s*=\s*false;",
        )
        lock = json.loads((REPO_ROOT / "flake.lock").read_text(encoding="utf-8"))
        crate2nix = lock["nodes"]["crate2nix"]
        self.assertFalse(crate2nix["flake"])
        self.assertEqual(
            crate2nix["locked"]["rev"],
            "7c33e664668faecf7655fa53861d7a80c9e464a2",
        )
        self.assertEqual(crate2nix["original"]["ref"], "0.15.0")

    def test_rust_builds_use_an_automatic_crate2nix_graph(self) -> None:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        packages = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        source = (REPO_ROOT / "nix" / "source.nix").read_text(encoding="utf-8")
        assert_automatic_crate2nix_contract(
            flake,
            packages,
            source,
            cargo_nix_exists=(REPO_ROOT / "Cargo.nix").exists(),
        )
        self.assertIn('workspaceMembers."chelis-cli".build', packages)
        self.assertIn('workspaceMembers."chelisup".build', packages)
        self.assertNotIn("buildRustPackage", packages)


    def test_disabled_ifd_fails_the_automatic_graph_contract(self) -> None:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        packages = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        source = (REPO_ROOT / "nix" / "source.nix").read_text(encoding="utf-8")
        mutated = flake.replace("allow-import-from-derivation = true;", "")
        with self.assertRaisesRegex(AssertionError, "enable import from derivation"):
            assert_automatic_crate2nix_contract(
                mutated,
                packages,
                source,
                cargo_nix_exists=False,
            )

    def test_online_generator_fails_the_automatic_graph_contract(self) -> None:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        packages = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        source = (REPO_ROOT / "nix" / "source.nix").read_text(encoding="utf-8")
        mutated = packages.replace('      CARGO_NET_OFFLINE = "true";\n', "")
        with self.assertRaisesRegex(AssertionError, "contract is incomplete"):
            assert_automatic_crate2nix_contract(
                flake,
                mutated,
                source,
                cargo_nix_exists=False,
            )

    def test_missing_generator_source_fails_the_automatic_graph_contract(self) -> None:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        packages = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        source = (REPO_ROOT / "nix" / "source.nix").read_text(encoding="utf-8")
        mutated = source.replace('    "Cargo.lock"\n', "")
        with self.assertRaisesRegex(AssertionError, "generator source is incomplete"):
            assert_automatic_crate2nix_contract(
                flake,
                packages,
                mutated,
                cargo_nix_exists=False,
            )

    def test_cvc5_sys_override_uses_the_fixed_native_inputs(self) -> None:
        packages = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        self.assertIn('"cvc5-sys" = attrs:', packages)
        self.assertIn('CVC5_DIR = "${cvc5.dir}";', packages)
        self.assertIn("pkgs.llvmPackages.libclang", packages)
        self.assertIn("pkgs.pkg-config", packages)
        self.assertIn("LIBCLANG_PATH", packages)

    def test_chelisup_rust_crate_does_not_own_nix_gc_roots(self) -> None:
        crate_dir = REPO_ROOT / "crates" / "chelisup"
        rust_sources = sorted(crate_dir.rglob("*.rs"))
        self.assertTrue(rust_sources)
        for path in rust_sources:
            source = path.read_text(encoding="utf-8")
            self.assertNotRegex(source, r"\b[Nn]ix\b|nix-gcroots|nix_gc", str(path))

    def test_workspace_source_overrides_preserve_external_compile_assets(self) -> None:
        packages = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        expected_roots = {
            "chelis-cli": "chelis-source/crates/chelis-cli",
            "chelis-compiler-api": "chelis-source/crates/chelis-compiler-api",
            "chelis-cove": "chelis-source/crates/chelis-cove",
            "tree-sitter-chelis": "chelis-source/tree-sitter-chelis",
        }
        for crate, source_root in expected_roots.items():
            with self.subTest(crate=crate):
                self.assertIn(f'"{crate}" = attrs:', packages)
                self.assertIn(f'sourceRoot = "{source_root}";', packages)
        self.assertEqual(packages.count("src = crateSource;"), len(expected_roots))

    def test_repository_lint_has_no_generated_graph_exclusion(self) -> None:
        policy = (REPO_ROOT / "chelis-lint.toml").read_text(encoding="utf-8")
        self.assertNotIn('pattern = "Cargo.nix"', policy)

    def test_native_checks_inspect_the_automatic_crate2nix_graph(self) -> None:
        checks = (REPO_ROOT / "nix" / "checks.nix").read_text(encoding="utf-8")
        self.assertIn("crate2nixGeneration", checks)
        self.assertIn("built.generatedCargoNix", checks)
        self.assertIn("Cargo-generated.nix", checks)
        self.assertNotIn("crate2nixGraphSync", checks)
        self.assertNotIn("crate2nixRegeneration", checks)
        self.assertFalse((REPO_ROOT / "scripts/check_crate2nix_sync.py").exists())

    def test_version_contract_requires_exact_cli_output(self) -> None:
        checks = (REPO_ROOT / "nix" / "checks.nix").read_text(encoding="utf-8")
        self.assertIn(
            'if [ "$version_output" != ${escape "chelis ${built.version}"} ]; then',
            checks,
        )


if __name__ == "__main__":
    unittest.main()
