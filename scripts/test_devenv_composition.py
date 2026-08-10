"""Tests for the composed Devenv configuration.

Run with `.venv/bin/python scripts/test_devenv_composition.py`.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMPORTS = (
    "./devenv/toolchains.nix",
    "./devenv/commands.nix",
    "./devenv/generated-files.nix",
    "./devenv/git-hooks.nix",
    "./devenv/smoke-tests.nix",
    "./devenv/package-outputs.nix",
    "./devenv/release-outputs.nix",
)
EXPECTED_COMMANDS = {
    "chelis-gate": "scripts/gate.py",
    "chelis-reap-orphans": "scripts/reap_orphans.py",
    "chelis-exec-preflight": "scripts/preflight_exec_probe.py",
    "chelis-z3-test": "scripts/z3_test.py",
    "chelis-hip-test": "scripts/hip_test.py",
}
EXPECTED_GENERATED_FILES = frozenset(
    {
        ".devenv/generated/compiler-probes/valid.c",
        ".devenv/generated/compiler-probes/warning.c",
        ".devenv/generated/compiler-probes/valid.cpp",
    }
)


@dataclass(frozen=True)
class DevenvComposition:
    imports: tuple[str, ...]


@dataclass(frozen=True)
class DevenvPackageOutputs:
    names: frozenset[str]
    reuses_flake_packages: bool


@dataclass(frozen=True)
class DevenvPython:
    version: tuple[int, int]
    uses_uv: bool
    manages_venv: bool
    pyo3_uses_venv: bool
    supplies_numpy: bool


@dataclass(frozen=True)
class DevenvCommands:
    names: frozenset[str]


@dataclass(frozen=True)
class DevenvGeneratedFiles:
    paths: frozenset[str]


@dataclass(frozen=True)
class DevenvCiProfiles:
    names: frozenset[str]
    exposes_cvc5: bool


def parse_root_composition(root_text: str, yaml_text: str) -> DevenvComposition:
    # Remote inputs (e.g. `ci/devenv/consumer`) are composed through
    # devenv.yaml. Local modules must stay in devenv.nix, so a `./`/`../` or
    # `.nix` entry under any yaml `imports:` block is rejected.
    for block in re.finditer(
        r"(?m)^imports:\s*$\n(?P<body>(?:[ \t]+-[ \t]+\S+\n?)+)", yaml_text
    ):
        local = [
            entry
            for entry in re.findall(
                r"(?m)^[ \t]+-[ \t]+(\S+)", block.group("body")
            )
            if entry.startswith(("./", "../")) or entry.endswith(".nix")
        ]
        if local:
            raise ValueError("devenv.yaml must not define local imports")

    match = re.fullmatch(
        (
            r"\s*\{ \.\.\. \}:\s*\{\s*imports\s*=\s*\[\s*"
            r"(?P<imports>.*?)\s*\];\s*\}\s*"
        ),
        root_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("devenv.nix must remain the local module composition root")

    parsed = tuple(match.group("imports").split())
    if any(re.fullmatch(r"\./devenv/[a-z-]+\.nix", item) is None for item in parsed):
        raise ValueError("devenv.nix must remain the local module composition root")
    if parsed != EXPECTED_IMPORTS:
        raise ValueError(f"devenv.nix must import the local modules: {parsed!r}")
    return DevenvComposition(imports=parsed)


def parse_package_outputs_module(text: str) -> DevenvPackageOutputs:
    if "builtins.getFlake (toString ../.)" in text:
        raise ValueError("the Devenv package flake source must exclude ignored paths")
    required = (
        'repoFlake = builtins.getFlake "git+file://${toString ../.}";',
        "system = pkgs.stdenv.hostPlatform.system;",
        "packageNames = (import ../nix/contracts.nix).packageNames;",
        "repoPackages = repoFlake.packages.${system};",
        "value = repoPackages.${name};",
        "outputs = builtins.listToAttrs (builtins.map mkOutput packageNames);",
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise ValueError(
            f"the Devenv package outputs must reuse the root flake packages: {missing!r}"
        )
    forbidden = (
        "languages.rust.import",
        "import ../nix/packages.nix",
        "pkgs.callPackage",
        "pkgs.runCommand",
    )
    found = [fragment for fragment in forbidden if fragment in text]
    if found:
        raise ValueError(
            f"the Devenv package outputs must not define another package graph: {found!r}"
        )
    return DevenvPackageOutputs(
        names=frozenset({"chelis", "chelis-runtime", "chelisup", "default"}),
        reuses_flake_packages=True,
    )


def parse_python_module(text: str) -> DevenvPython:
    required = (
        "package = pkgs.python311.withPackages",
        "ps.numpy",
        "venv.enable = true;",
        "uv.enable = true;",
        'PYO3_PYTHON = "${config.env.DEVENV_STATE}/venv/bin/python";',
        "CHELIS_PYTHON_WHEEL_LIBRARY_PATH",
        "pkgs.stdenv.cc.cc.lib",
        "pkgs.zlib",
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise ValueError(f"the Devenv Python contract is incomplete: {missing!r}")
    if re.search(r"(?m)^\s*enterShell\s*=", text):
        raise ValueError("the Python module must not define enterShell setup")
    return DevenvPython(
        version=(3, 11),
        uses_uv=True,
        manages_venv=True,
        pyo3_uses_venv=True,
        supplies_numpy=True,
    )


def parse_python_smoke_tests(text: str) -> None:
    required = (
        "pkg-config mdbook openspec",
        "mdbook --version",
        "import os, pathlib, sys, numpy",
        'numpy.__version__.split(".")[0] == "2"',
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise ValueError(f"the Devenv Python smoke contract is incomplete: {missing!r}")


def parse_ci_profiles(text: str) -> DevenvCiProfiles:
    required = (
        "cvc5 = import ../nix/cvc5.nix",
        "outputs.cvc5-dir = cvc5.dir;",
        "ci.module.env = {",
        'CARGO_PROFILE_DEV_DEBUG = "0";',
        'CARGO_PROFILE_TEST_DEBUG = "0";',
        "sanitizers = {",
        'extends = [ "ci" ];',
        'CHELIS_C_TEST_EXTRA_FLAGS = "-O1 -fsanitize=address,undefined -fno-omit-frame-pointer";',
        'ASAN_OPTIONS = "detect_leaks=1:halt_on_error=1";',
        'UBSAN_OPTIONS = "print_stacktrace=1:halt_on_error=1";',
        "smt = {",
        'CVC5_DIR = "${config.outputs.cvc5-dir}";',
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise ValueError(f"the Devenv CI profile contract is incomplete: {missing!r}")
    if text.count('extends = [ "ci" ];') != 2:
        raise ValueError("the SMT and sanitizer profiles must extend the CI profile")
    return DevenvCiProfiles(
        names=frozenset({"ci", "sanitizers", "smt"}),
        exposes_cvc5=True,
    )


def parse_openspec_composition(toolchains_text: str, yaml_text: str) -> None:
    # OpenSpec is provided by the ci consumer module (config.outputs.openspec),
    # not built in the Devenv shell. Its version is pinned once in ci and
    # verified at runtime by the smoke test's `openspec --version` check.
    if "config.outputs.openspec" not in toolchains_text:
        raise ValueError(
            "the Devenv shell must use OpenSpec from the ci consumer module"
        )
    if "openspecPinned" in toolchains_text:
        raise ValueError("the Devenv shell must not build its own OpenSpec")
    if "ci/devenv/consumer" not in yaml_text:
        raise ValueError("the Devenv shell must compose the ci consumer module")


def parse_commands_module(text: str) -> DevenvCommands:
    adapter_fragments = (
        "runPython =",
        'script = "${config.devenv.root}/''${relativePath}"',
        "os.execv(sys.executable, [sys.executable, script, *sys.argv[1:]])",
    )
    missing_adapter = [
        fragment for fragment in adapter_fragments if fragment not in text
    ]
    if missing_adapter:
        raise ValueError(
            f"the Devenv Python command adapter is incomplete: {missing_adapter!r}"
        )

    blocks = {
        name: body
        for _, name, body in re.findall(
            r'(?ms)^(\s+)"(chelis-[^"]+)" = \{\n(.*?)^\1\};$',
            text,
        )
    }
    if frozenset(blocks) != frozenset(EXPECTED_COMMANDS):
        raise ValueError(f"the Devenv command set is incomplete: {sorted(blocks)!r}")

    darwin_marker = "lib.optionalAttrs pkgs.stdenv.isDarwin"
    linux_marker = "lib.optionalAttrs pkgs.stdenv.isLinux"
    if darwin_marker not in text or linux_marker not in text:
        raise ValueError("platform-specific Devenv commands must use platform guards")
    if text.index('"chelis-exec-preflight"') < text.index(darwin_marker):
        raise ValueError("the exec preflight command must use the macOS guard")
    for name in ("chelis-z3-test", "chelis-hip-test"):
        if text.index(f'"{name}"') < text.index(linux_marker):
            raise ValueError(f"Devenv command {name} must use the Linux guard")

    for name, script_path in EXPECTED_COMMANDS.items():
        body = blocks[name]
        if "package = config.languages.python.package;" not in body:
            raise ValueError(f"Devenv command {name} must use the Python package")
        if f'exec = runPython "{script_path}";' not in body:
            raise ValueError(f"Devenv command {name} must invoke {script_path}")
    return DevenvCommands(names=frozenset(blocks))


def parse_generated_files_module(text: str) -> DevenvGeneratedFiles:
    paths = frozenset(
        re.findall(r"(?m)^\s{2}files\.\"([^\"]+)\"\.text = ''$", text)
    )
    if paths != EXPECTED_GENERATED_FILES:
        raise ValueError(
            f"the compiler probe file set is incomplete: {sorted(paths)!r}"
        )
    if "copyMode" in text:
        raise ValueError("compiler probe files must use read-only symlink mode")
    return DevenvGeneratedFiles(paths=paths)


def parse_contributor_docs(text: str) -> None:
    required = (
        "chelis-gate",
        "chelis-reap-orphans",
        "chelis-exec-preflight",
        "chelis-z3-test",
        "chelis-hip-test",
        "no-ai-authorship",
        "cargo-husky",
        "scripts/check_commit_message.py",
        ".devenv/state/venv",
        "PYO3_PYTHON",
        "OpenSpec 1.6.0",
        "openspec validate --all --strict --no-interactive",
        "devenv build outputs.chelis",
        "devenv build outputs.chelis-runtime",
        "devenv build outputs.chelisup",
        "devenv build --no-tui outputs.release-chelisup",
        "devenv --profile ci shell",
        "devenv --profile sanitizers shell",
        "devenv --profile smt shell",
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise ValueError(f"the Devenv contributor guide is incomplete: {missing!r}")


class DevenvCompositionTests(unittest.TestCase):
    def test_repository_composes_the_local_modules_from_the_root_file(self) -> None:
        root_text = (REPO_ROOT / "devenv.nix").read_text(encoding="utf-8")
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        self.assertEqual(
            parse_root_composition(root_text, yaml_text).imports,
            EXPECTED_IMPORTS,
        )

    def test_devenv_reuses_the_root_flake_package_outputs(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        parsed = parse_package_outputs_module(text)
        self.assertEqual(
            parsed.names,
            frozenset({"chelis", "chelis-runtime", "chelisup", "default"}),
        )
        self.assertTrue(parsed.reuses_flake_packages)

    def test_devenv_manages_python_and_the_pyo3_interpreter(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        parsed = parse_python_module(text)
        self.assertEqual(parsed.version, (3, 11))
        self.assertTrue(parsed.uses_uv)
        self.assertTrue(parsed.manages_venv)
        self.assertTrue(parsed.pyo3_uses_venv)
        self.assertTrue(parsed.supplies_numpy)

    def test_devenv_smoke_tests_cover_the_ci_python_tools(self) -> None:
        text = (REPO_ROOT / "devenv/smoke-tests.nix").read_text(encoding="utf-8")
        parse_python_smoke_tests(text)

    def test_missing_numpy_smoke_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/smoke-tests.nix").read_text(encoding="utf-8")
        mutated = text.replace("import os, pathlib, sys, numpy", "import os, pathlib, sys")
        with self.assertRaisesRegex(ValueError, "Python smoke contract is incomplete"):
            parse_python_smoke_tests(mutated)

    def test_devenv_composes_openspec_from_ci(self) -> None:
        toolchains = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        yaml = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        parse_openspec_composition(toolchains, yaml)

    def test_devenv_owns_ci_sanitizer_and_smt_profiles(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        profiles = parse_ci_profiles(text)
        self.assertEqual(profiles.names, frozenset({"ci", "sanitizers", "smt"}))
        self.assertTrue(profiles.exposes_cvc5)

    def test_missing_ci_debug_policy_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        mutated = text.replace('CARGO_PROFILE_DEV_DEBUG = "0";', "")
        with self.assertRaisesRegex(ValueError, "CI profile contract is incomplete"):
            parse_ci_profiles(mutated)

    def test_missing_sanitizer_policy_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        mutated = text.replace("-fsanitize=address,undefined", "-fno-sanitize=all")
        with self.assertRaisesRegex(ValueError, "CI profile contract is incomplete"):
            parse_ci_profiles(mutated)

    def test_missing_cvc5_profile_output_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        mutated = text.replace("outputs.cvc5-dir = cvc5.dir;", "")
        with self.assertRaisesRegex(ValueError, "CI profile contract is incomplete"):
            parse_ci_profiles(mutated)

    def test_devenv_exposes_the_python_command_facade(self) -> None:
        text = (REPO_ROOT / "devenv/commands.nix").read_text(encoding="utf-8")
        self.assertEqual(
            parse_commands_module(text).names,
            frozenset(EXPECTED_COMMANDS),
        )

    def test_devenv_declares_compiler_probe_files(self) -> None:
        text = (REPO_ROOT / "devenv/generated-files.nix").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            parse_generated_files_module(text).paths,
            EXPECTED_GENERATED_FILES,
        )

    def test_contributor_docs_explain_the_devenv_surface(self) -> None:
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        parse_contributor_docs(text)

    def test_missing_import_fails_at_the_parse_boundary(self) -> None:
        root_text = (REPO_ROOT / "devenv.nix").read_text(encoding="utf-8")
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        mutated = root_text.replace("    ./devenv/commands.nix\n", "")
        with self.assertRaisesRegex(ValueError, "must import the local modules"):
            parse_root_composition(mutated, yaml_text.replace("imports:", "other:"))

    def test_extra_root_setting_fails_at_the_parse_boundary(self) -> None:
        root_text = (REPO_ROOT / "devenv.nix").read_text(encoding="utf-8")
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        mutated = root_text.replace("\n}", "\n  packages = [ ];\n}")
        with self.assertRaisesRegex(ValueError, "composition root"):
            parse_root_composition(mutated, yaml_text.replace("imports:", "other:"))

    def test_yaml_imports_fail_at_the_parse_boundary(self) -> None:
        root_text = (REPO_ROOT / "devenv.nix").read_text(encoding="utf-8")
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must not define local imports"):
            parse_root_composition(root_text, f"{yaml_text}\nimports:\n  - ./other.nix\n")

    def test_non_package_flake_outputs_fail_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            "repoPackages = repoFlake.packages.${system};",
            "repoPackages = repoFlake.checks.${system};",
        )
        with self.assertRaisesRegex(ValueError, "must reuse the root flake packages"):
            parse_package_outputs_module(mutated)

    def test_unfiltered_devenv_flake_source_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            'builtins.getFlake "git+file://${toString ../.}"',
            "builtins.getFlake (toString ../.)",
        )
        with self.assertRaisesRegex(ValueError, "must exclude ignored paths"):
            parse_package_outputs_module(mutated)

    def test_independent_devenv_package_graph_fails_at_the_parse_boundary(
        self,
    ) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            "in\n{",
            "  duplicate = import ../nix/packages.nix;\nin\n{",
        )
        with self.assertRaisesRegex(ValueError, "must not define another package graph"):
            parse_package_outputs_module(mutated)

    def test_disabled_python_venv_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        mutated = text.replace("venv.enable = true;", "venv.enable = false;")
        with self.assertRaisesRegex(ValueError, "Python contract is incomplete"):
            parse_python_module(mutated)

    def test_missing_numpy_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        mutated = text.replace("ps.numpy", "ps.pytest")
        with self.assertRaisesRegex(ValueError, "Python contract is incomplete"):
            parse_python_module(mutated)

    def test_missing_python_wheel_loader_path_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        mutated = text.replace("CHELIS_PYTHON_WHEEL_LIBRARY_PATH", "OMITTED")
        with self.assertRaisesRegex(ValueError, "Python contract is incomplete"):
            parse_python_module(mutated)

    def test_missing_openspec_composition_fails_at_the_parse_boundary(self) -> None:
        toolchains = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        yaml = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "OpenSpec from the ci consumer"):
            parse_openspec_composition(
                toolchains.replace("config.outputs.openspec", "config.outputs.other"),
                yaml,
            )
        with self.assertRaisesRegex(ValueError, "compose the ci consumer"):
            parse_openspec_composition(
                toolchains, yaml.replace("ci/devenv/consumer", "ci/devenv/other")
            )

    def test_missing_command_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/commands.nix").read_text(encoding="utf-8")
        mutated = text.replace('"chelis-gate"', '"other-gate"', 1)
        with self.assertRaisesRegex(ValueError, "command set is incomplete"):
            parse_commands_module(mutated)

    def test_missing_generated_file_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/generated-files.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            '  files.".devenv/generated/compiler-probes/valid.c".text = \'\'',
            '  files."other.c".text = \'\'',
        )
        with self.assertRaisesRegex(ValueError, "file set is incomplete"):
            parse_generated_files_module(mutated)

    def test_missing_command_docs_fail_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        mutated = text.replace("chelis-gate", "missing-gate")
        with self.assertRaisesRegex(ValueError, "contributor guide is incomplete"):
            parse_contributor_docs(mutated)

    def test_missing_cargo_husky_docs_fail_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        mutated = text.replace("cargo-husky", "missing-hook-installer")
        with self.assertRaisesRegex(ValueError, "contributor guide is incomplete"):
            parse_contributor_docs(mutated)


if __name__ == "__main__":
    unittest.main()
