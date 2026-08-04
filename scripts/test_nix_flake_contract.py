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


def assert_shared_package_rule_contract(
    adapter: str,
    workspace: str,
    overrides: str,
    artifacts: str,
) -> None:
    adapter_required = (
        "import ./workspace.nix",
        "import ./artifacts.nix",
        "builtins.hasAttr name workspace.cargoGraph.workspaceMembers",
        "builtins.getAttr name workspace.cargoGraph.workspaceMembers",
        'throw "the crate2nix graph is missing required workspace member ${name}";',
        'requireWorkspaceMember "chelis-cli"',
        'requireWorkspaceMember "chelis-runtime"',
        'requireWorkspaceMember "chelisup"',
    )
    adapter_missing = [item for item in adapter_required if item not in adapter]
    if adapter_missing:
        raise AssertionError(
            f"the root package adapter must use the shared helpers: {adapter_missing!r}"
        )

    workspace_required = (
        "import ./crate-overrides.nix",
        'cargoNix = source + "/Cargo.nix";',
        "import cargoNix",
        "regenerationSource = import ./source.nix",
        '".cargo"',
        '"crate-hashes.json"',
        '"crate2nix.json"',
        '"scripts"',
        "import ./crate2nix-regeneration.nix",
        "toolchain",
    )
    workspace_missing = [item for item in workspace_required if item not in workspace]
    if workspace_missing:
        raise AssertionError(
            f"the shared workspace graph is incomplete: {workspace_missing!r}"
        )
    for member in ('"chelis-runtime"', '"chelisup"'):
        if member in workspace:
            raise AssertionError(
                "the shared workspace graph must not select product members"
            )

    override_required = (
        "cratePkgs.defaultCrateOverrides // chelisOverrides",
        '"chelis-cli" = attrs:',
        '"chelis-compiler-api" = attrs:',
        '"chelis-cove" = attrs:',
        '"cvc5-sys" = attrs:',
        '"tree-sitter-chelis" = attrs:',
        'CVC5_DIR = "${cvc5.dir}";',
        "requiredOverrideNames",
        "missingOverrides",
        "assert missingOverrides == [ ];",
    )
    override_missing = [item for item in override_required if item not in overrides]
    if override_missing:
        raise AssertionError(
            f"the shared crate overrides are incomplete: {override_missing!r}"
        )

    artifact_required = (
        "compilerCrate",
        "runtimeCrate",
        "chelisupCrate",
        'pkgs.runCommand "chelis-cli-${version}"',
        'pkgs.runCommand "chelis-runtime-${version}"',
        'pkgs.runCommand "chelisup-${version}"',
        "libexec/chelisup",
        "nix-gcroots",
    )
    artifact_missing = [item for item in artifact_required if item not in artifacts]
    if artifact_missing:
        raise AssertionError(
            f"the shared artifact assembly is incomplete: {artifact_missing!r}"
        )


def assert_checked_in_crate2nix_contract(
    flake: str,
    workspace: str,
    *,
    cargo_nix_exists: bool,
) -> None:
    if "allow-import-from-derivation" in flake:
        raise AssertionError("the flake must not enable import from derivation")

    required_workspace = (
        'cargoNix = source + "/Cargo.nix";',
        "import cargoNix",
        "rootFeatures = [ ];",
    )
    missing_workspace = [
        item for item in required_workspace if item not in workspace
    ]
    if missing_workspace:
        raise AssertionError(
            f"the checked-in crate2nix contract is incomplete: {missing_workspace!r}"
        )
    forbidden_workspace = ("generatedCargoNix", "appliedCargoNix")
    found = [item for item in forbidden_workspace if item in workspace]
    if found:
        raise AssertionError(
            f"package evaluation must not generate a Cargo graph: {found!r}"
        )
    if not cargo_nix_exists:
        raise AssertionError("the repository must track Cargo.nix")


def assert_regeneration_contract(regeneration: str) -> None:
    required = (
        'crate2nix + "/crate2nix/Cargo.nix"',
        'CARGO_NET_OFFLINE = "true";',
        "cargoSetupPostUnpackHook",
        '"--no-default-features"',
        '"--features"',
        '"chelis-cli/smt"',
        '"--output"',
        '"Cargo.nix"',
        "cp -R ${source} workspace",
        "cmp Cargo.nix ${source}/Cargo.nix",
    )
    if "${root}" in regeneration:
        raise AssertionError("the exact regeneration check must not copy the raw root")
    missing = [marker for marker in required if marker not in regeneration]
    if missing:
        raise AssertionError(
            f"the exact crate2nix regeneration contract is incomplete: {missing!r}"
        )


class NixFlakeContractTests(unittest.TestCase):
    @REQUIRES_NIX
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

    def test_root_packages_use_the_shared_workspace_and_artifact_rules(self) -> None:
        adapter = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        workspace = (REPO_ROOT / "nix" / "workspace.nix").read_text(
            encoding="utf-8"
        )
        overrides = (REPO_ROOT / "nix" / "crate-overrides.nix").read_text(
            encoding="utf-8"
        )
        artifacts = (REPO_ROOT / "nix" / "artifacts.nix").read_text(
            encoding="utf-8"
        )
        assert_shared_package_rule_contract(
            adapter,
            workspace,
            overrides,
            artifacts,
        )

    def test_rust_builds_use_one_checked_in_crate2nix_graph(self) -> None:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        workspace = (REPO_ROOT / "nix" / "workspace.nix").read_text(
            encoding="utf-8"
        )
        assert_checked_in_crate2nix_contract(
            flake,
            workspace,
            cargo_nix_exists=(REPO_ROOT / "Cargo.nix").is_file(),
        )
        cargo_nix = (REPO_ROOT / "Cargo.nix").read_text(encoding="utf-8")
        self.assertIn("@generated by crate2nix 0.15.0", cargo_nix)
        self.assertRegex(
            cargo_nix,
            r"(?m)^# chelis-crate2nix-input-sha256: [0-9a-f]{64}$",
        )
        self.assertNotIn("buildRustPackage", workspace)

    def test_empty_root_workspace_member_fallback_fails_the_package_rule_contract(
        self,
    ) -> None:
        adapter = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        workspace = (REPO_ROOT / "nix" / "workspace.nix").read_text(
            encoding="utf-8"
        )
        overrides = (REPO_ROOT / "nix" / "crate-overrides.nix").read_text(
            encoding="utf-8"
        )
        artifacts = (REPO_ROOT / "nix" / "artifacts.nix").read_text(
            encoding="utf-8"
        )
        mutated = adapter.replace(
            'throw "the crate2nix graph is missing required workspace member ${name}";',
            "{ };",
        )
        with self.assertRaisesRegex(AssertionError, "root package adapter"):
            assert_shared_package_rule_contract(
                mutated,
                workspace,
                overrides,
                artifacts,
            )

    def test_missing_shared_cvc5_override_fails_the_package_rule_contract(
        self,
    ) -> None:
        adapter = (REPO_ROOT / "nix" / "packages.nix").read_text(encoding="utf-8")
        workspace = (REPO_ROOT / "nix" / "workspace.nix").read_text(
            encoding="utf-8"
        )
        overrides = (REPO_ROOT / "nix" / "crate-overrides.nix").read_text(
            encoding="utf-8"
        )
        artifacts = (REPO_ROOT / "nix" / "artifacts.nix").read_text(
            encoding="utf-8"
        )
        mutated = overrides.replace('"cvc5-sys" = attrs:', '"cvc5-missing" = attrs:')
        with self.assertRaisesRegex(AssertionError, "crate overrides"):
            assert_shared_package_rule_contract(
                adapter,
                workspace,
                mutated,
                artifacts,
            )

    def test_ifd_marker_fails_the_checked_in_graph_contract(self) -> None:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        workspace = (REPO_ROOT / "nix" / "workspace.nix").read_text(
            encoding="utf-8"
        )
        mutated = flake.replace(
            "  inputs = {",
            "  nixConfig.allow-import-from-derivation = true;\n\n  inputs = {",
        )
        with self.assertRaisesRegex(AssertionError, "must not enable"):
            assert_checked_in_crate2nix_contract(
                mutated,
                workspace,
                cargo_nix_exists=True,
            )

    def test_missing_checked_in_graph_fails_the_contract(self) -> None:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        workspace = (REPO_ROOT / "nix" / "workspace.nix").read_text(
            encoding="utf-8"
        )
        with self.assertRaisesRegex(AssertionError, "must track Cargo.nix"):
            assert_checked_in_crate2nix_contract(
                flake,
                workspace,
                cargo_nix_exists=False,
            )

    def test_missing_tracked_graph_import_fails_the_contract(self) -> None:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        workspace = (REPO_ROOT / "nix" / "workspace.nix").read_text(
            encoding="utf-8"
        )
        mutated = workspace.replace("import cargoNix", "import missingGraph")
        with self.assertRaisesRegex(AssertionError, "contract is incomplete"):
            assert_checked_in_crate2nix_contract(
                flake,
                mutated,
                cargo_nix_exists=True,
            )

    @REQUIRES_NIX
    def test_package_evaluation_succeeds_when_ifd_is_disabled(self) -> None:
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
                f".#packages.{system}.chelis.drvPath",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(completed.stdout.strip().endswith(".drv"))

    def test_exact_regeneration_uses_offline_pinned_inputs(self) -> None:
        regeneration = (
            REPO_ROOT / "nix" / "crate2nix-regeneration.nix"
        ).read_text(encoding="utf-8")
        assert_regeneration_contract(regeneration)

        mutated = regeneration.replace('CARGO_NET_OFFLINE = "true";', "")
        with self.assertRaisesRegex(AssertionError, "regeneration contract"):
            assert_regeneration_contract(mutated)

        raw_root = regeneration.replace("${source}", "${root}")
        with self.assertRaisesRegex(AssertionError, "raw root"):
            assert_regeneration_contract(raw_root)

    def test_cvc5_sys_override_uses_the_fixed_native_inputs(self) -> None:
        overrides = (REPO_ROOT / "nix" / "crate-overrides.nix").read_text(
            encoding="utf-8"
        )
        self.assertIn('"cvc5-sys" = attrs:', overrides)
        self.assertIn('CVC5_DIR = "${cvc5.dir}";', overrides)
        self.assertIn("pkgs.llvmPackages.libclang", overrides)
        self.assertIn("pkgs.pkg-config", overrides)
        self.assertIn("LIBCLANG_PATH", overrides)

    def test_chelisup_rust_crate_does_not_own_nix_gc_roots(self) -> None:
        crate_dir = REPO_ROOT / "crates" / "chelisup"
        rust_sources = sorted(crate_dir.rglob("*.rs"))
        self.assertTrue(rust_sources)
        for path in rust_sources:
            source = path.read_text(encoding="utf-8")
            self.assertNotRegex(source, r"\b[Nn]ix\b|nix-gcroots|nix_gc", str(path))

    def test_workspace_source_overrides_preserve_external_compile_assets(self) -> None:
        overrides = (REPO_ROOT / "nix" / "crate-overrides.nix").read_text(
            encoding="utf-8"
        )
        expected_roots = {
            "chelis-cli": "chelis-source/crates/chelis-cli",
            "chelis-compiler-api": "chelis-source/crates/chelis-compiler-api",
            "chelis-cove": "chelis-source/crates/chelis-cove",
            "tree-sitter-chelis": "chelis-source/tree-sitter-chelis",
        }
        for crate, source_root in expected_roots.items():
            with self.subTest(crate=crate):
                self.assertIn(f'"{crate}" = attrs:', overrides)
                self.assertIn(f'sourceRoot = "{source_root}";', overrides)
        self.assertEqual(overrides.count("src = crateSource;"), len(expected_roots))

    def test_generated_graph_is_outside_the_editable_lint_corpus(self) -> None:
        policy = (REPO_ROOT / "chelis-lint.toml").read_text(encoding="utf-8")
        self.assertRegex(
            policy,
            r'(?s)pattern = "Cargo\.nix"\nclass = "generated"\ncross_ref = "§12\.2"',
        )

    def test_native_checks_reject_a_stale_checked_in_graph(self) -> None:
        checks = (REPO_ROOT / "nix" / "checks.nix").read_text(encoding="utf-8")
        self.assertIn("scripts/check_crate2nix_sync.py", checks)
        self.assertIn("crate2nixGraphSync", checks)
        self.assertNotIn("crate2nixGeneration", checks)
        self.assertNotIn("built.generatedCargoNix", checks)
        self.assertTrue((REPO_ROOT / "scripts/check_crate2nix_sync.py").is_file())

    def test_version_contract_requires_exact_cli_output(self) -> None:
        checks = (REPO_ROOT / "nix" / "checks.nix").read_text(encoding="utf-8")
        self.assertIn(
            'if [ "$version_output" != ${escape "chelis ${built.version}"} ]; then',
            checks,
        )

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


if __name__ == "__main__":
    unittest.main()
