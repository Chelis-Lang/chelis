"""Contract tests for the public Nix package and application surface."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NIX = shutil.which("nix")
EXPECTED_SYSTEMS = ["aarch64-darwin", "x86_64-linux"]
EXPECTED_PACKAGES = ["chelis", "chelis-runtime", "chelisup", "default"]
EXPECTED_APPS = ["chelis", "chelisup", "default"]
EXPECTED_HEADERS = [
    "chelis_runtime.h",
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


@unittest.skipUnless(NIX, "Nix is not installed")
class NixFlakeContractTests(unittest.TestCase):
    def test_contract_records_exact_release_header_manifest(self) -> None:
        contracts = nix_json("eval", "--json", "--file", "nix/contracts.nix")
        self.assertEqual(contracts["publicRuntimeHeaders"], EXPECTED_HEADERS)

        release = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        staged_headers = set(
            re.findall(
                r"cp crates/chelis-runtime/include/(chelis_[^\s/]+\.h) "
                r'"\$staging/include/"',
                release,
            )
        )
        self.assertEqual(staged_headers, set(EXPECTED_HEADERS))
        self.assertIn('cp target/release/chelis "$staging/bin/"', release)
        self.assertIn(
            'cp target/release/libchelis_runtime.a "$staging/lib/"', release
        )
        self.assertEqual(
            contracts["runtimeConsumers"],
            {"aarch64-darwin": "Accelerate", "x86_64-linux": "OpenBLAS"},
        )
        self.assertEqual(
            contracts["compilerBehaviorChecks"],
            ["version", "help", "release-fixture", "smt"],
        )

    def test_supported_systems_are_exact(self) -> None:
        systems = nix_json("eval", "--json", ".#packages", "--apply", "builtins.attrNames")
        self.assertEqual(systems, EXPECTED_SYSTEMS)

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

    def test_default_package_and_application_are_exact_aliases(self) -> None:
        for system in EXPECTED_SYSTEMS:
            with self.subTest(system=system):
                default_package = nix_raw("eval", "--raw", f".#packages.{system}.default.drvPath")
                chelis_package = nix_raw("eval", "--raw", f".#packages.{system}.chelis.drvPath")
                self.assertEqual(default_package, chelis_package)

                default_program = nix_raw("eval", "--raw", f".#apps.{system}.default.program")
                chelis_program = nix_raw("eval", "--raw", f".#apps.{system}.chelis.program")
                self.assertEqual(default_program, chelis_program)

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

    def test_rust_builds_use_the_checked_in_crate2nix_graph(self) -> None:
        cargo_nix_path = REPO_ROOT / "Cargo.nix"
        self.assertTrue(cargo_nix_path.is_file())
        cargo_nix = cargo_nix_path.read_text(encoding="utf-8")
        self.assertIn("@generated by crate2nix 0.15.0", cargo_nix)
        self.assertRegex(
            cargo_nix,
            r"(?m)^# chelis-crate2nix-input-sha256: [0-9a-f]{64}$",
        )

        packages = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        self.assertIn('import (root + "/Cargo.nix")', packages)
        self.assertIn('workspaceMembers."chelis-cli".build', packages)
        self.assertIn('workspaceMembers."chelis-runtime".build', packages)
        self.assertIn('workspaceMembers."chelisup".build', packages)
        self.assertIn('features = [ "smt" ];', packages)
        self.assertNotIn("buildRustPackage", packages)
        handwritten_nix = "\n".join(
            path.read_text(encoding="utf-8")
            for path in [REPO_ROOT / "flake.nix", *(REPO_ROOT / "nix").glob("*.nix")]
        )
        self.assertNotIn("generatedCargoNix", handwritten_nix)
        self.assertNotIn("appliedCargoNix", handwritten_nix)
        self.assertNotIn("allow-import-from-derivation", handwritten_nix)

    def test_cvc5_sys_override_uses_the_fixed_native_inputs(self) -> None:
        packages = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        self.assertIn('"cvc5-sys" = attrs:', packages)
        self.assertIn('CVC5_DIR = "${cvc5.dir}";', packages)
        self.assertIn("pkgs.llvmPackages.libclang", packages)
        self.assertIn("pkgs.pkg-config", packages)
        self.assertIn("LIBCLANG_PATH", packages)

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

    def test_generated_graph_is_outside_the_editable_lint_corpus(self) -> None:
        policy = (REPO_ROOT / "chelis-lint.toml").read_text(encoding="utf-8")
        self.assertRegex(
            policy,
            r'(?s)pattern = "Cargo\.nix"\nclass = "generated"\ncross_ref = "§12\.2"',
        )

    def test_native_checks_regenerate_the_exact_crate2nix_graph(self) -> None:
        checks = (REPO_ROOT / "nix" / "checks.nix").read_text(encoding="utf-8")
        self.assertIn("scripts/check_crate2nix_sync.py", checks)
        self.assertIn("crate2nixGraphSync", checks)
        self.assertIn("crate2nixRegeneration", checks)
        self.assertIn("crate2nix generate", checks)
        self.assertIn("cmp Cargo.nix ${root}/Cargo.nix", checks)

    def test_version_contract_requires_exact_cli_output(self) -> None:
        checks = (REPO_ROOT / "nix" / "checks.nix").read_text(encoding="utf-8")
        self.assertIn(
            'if [ "$version_output" != ${escape "chelis ${built.version}"} ]; then',
            checks,
        )

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
            shapes["chelisup"]["allowedProductExecutables"],
            ["bin/chelisup"],
        )
        self.assertEqual(shapes["chelisup"]["forbidden"], ["bin/chelis"])
        self.assertEqual(
            shapes["chelisup"]["inventory"],
            ["bin", "bin/chelisup"],
        )
        for header in EXPECTED_HEADERS:
            self.assertIn(f"include/{header}", shapes["chelis"]["required"])
            self.assertIn(f"include/{header}", shapes["chelis-runtime"]["required"])


if __name__ == "__main__":
    unittest.main()
