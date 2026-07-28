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
