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
class DevenvPython:
    version: tuple[int, int]
    uses_uv: bool
    manages_venv: bool
    pyo3_uses_venv: bool


@dataclass(frozen=True)
class DevenvCommands:
    names: frozenset[str]


@dataclass(frozen=True)
class DevenvGeneratedFiles:
    paths: frozenset[str]


def parse_root_composition(root_text: str, yaml_text: str) -> DevenvComposition:
    if re.search(r"(?m)^\s*imports:\s*$", yaml_text):
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


def parse_python_module(text: str) -> DevenvPython:
    required = (
        "package = pkgs.python311;",
        "venv.enable = true;",
        "uv.enable = true;",
        'PYO3_PYTHON = "${config.env.DEVENV_STATE}/venv/bin/python";',
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
    )


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
        # The check no longer runs as a devenv-installed prek hook; the
        # contributor guide must name where it does run instead (chelis#1409).
        ".githooks/commit-msg",
        "cargo-husky",
        "scripts/check_commit_message.py",
        ".devenv/state/venv",
        "PYO3_PYTHON",
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

    def test_devenv_manages_python_and_the_pyo3_interpreter(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        parsed = parse_python_module(text)
        self.assertEqual(parsed.version, (3, 11))
        self.assertTrue(parsed.uses_uv)
        self.assertTrue(parsed.manages_venv)
        self.assertTrue(parsed.pyo3_uses_venv)

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

    def test_disabled_python_venv_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        mutated = text.replace("venv.enable = true;", "venv.enable = false;")
        with self.assertRaisesRegex(ValueError, "Python contract is incomplete"):
            parse_python_module(mutated)

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
