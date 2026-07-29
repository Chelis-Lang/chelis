"""Tests for the repository Devenv release pin.

Run with `.venv/bin/python scripts/test_devenv_version.py`.
"""

from __future__ import annotations

import json
import re
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_URL = "github:cachix/devenv/v2.2?dir=src/modules"
EXPECTED_REF = "v2.2"
EXPECTED_REVISION = "ffce215a42d09c6375c3d60dd9c4110438fc4d87"
EXPECTED_TEST_TASKS = (
    "chelis:toolchain-test",
    "chelis:python-test",
    "chelis:c-compiler-test",
    "chelis:cpp-compiler-test",
)


@dataclass(frozen=True)
class DevenvPin:
    url: str
    ref: str
    directory: str
    locked_directory: str
    revision: str


@dataclass(frozen=True)
class DevenvTestTasks:
    names: frozenset[str]


def parse_devenv_test_tasks(text: str) -> DevenvTestTasks:
    raw_names = re.findall(r'(?m)^\s{2}tasks\."([^"]+)" = \{$', text)
    names = frozenset(raw_names)
    if len(raw_names) != len(names):
        raise ValueError("devenv.nix must not define a duplicate test task")
    if names != frozenset(EXPECTED_TEST_TASKS):
        raise ValueError(f"devenv.nix must define the named test tasks: {names!r}")
    if text.count('after = [ "devenv:enterShell" ];') != len(names):
        raise ValueError("each named test task must run after devenv:enterShell")
    if text.count('before = [ "devenv:enterTest" ];') != len(names):
        raise ValueError("each named test task must run before devenv:enterTest")
    if re.search(r"(?m)^\s*enterTest\s*=", text):
        raise ValueError("devenv.nix must not define a monolithic enterTest script")
    if "processes." in text or "services." in text:
        raise ValueError("the Devenv smoke check must not define a service or process")
    return DevenvTestTasks(names=names)


def parse_devenv_url(text: str) -> str:
    lines = text.splitlines()
    try:
        inputs_index = lines.index("inputs:")
    except ValueError as error:
        raise ValueError("devenv.yaml does not define the inputs section") from error

    devenv_index = None
    for index, line in enumerate(lines[inputs_index + 1 :], start=inputs_index + 1):
        if line and not line.startswith(" "):
            break
        if line == "  devenv:":
            devenv_index = index
            break
    if devenv_index is None:
        raise ValueError("devenv.yaml does not define the devenv input")

    for line in lines[devenv_index + 1 :]:
        if line and not line.startswith("    "):
            break
        if line.startswith("    url: "):
            return line.removeprefix("    url: ")
    raise ValueError("the devenv input does not define a URL")


def parse_devenv_pin(yaml_text: str, lock_data: Any) -> DevenvPin:
    try:
        node = lock_data["nodes"]["devenv"]
        original = node["original"]
        locked = node["locked"]
        ref = original["ref"]
        directory = original["dir"]
        locked_directory = locked["dir"]
        revision = locked["rev"]
    except (KeyError, TypeError) as error:
        raise ValueError("devenv.lock does not contain a complete devenv pin") from error

    values = (ref, directory, locked_directory, revision)
    if not all(isinstance(value, str) for value in values):
        raise ValueError("the devenv lock fields must be strings")

    return DevenvPin(
        url=parse_devenv_url(yaml_text),
        ref=ref,
        directory=directory,
        locked_directory=locked_directory,
        revision=revision,
    )


def require_v22(pin: DevenvPin) -> None:
    expected = DevenvPin(
        url=EXPECTED_URL,
        ref=EXPECTED_REF,
        directory="src/modules",
        locked_directory="src/modules",
        revision=EXPECTED_REVISION,
    )
    if pin != expected:
        raise ValueError(f"the repository must pin Devenv {EXPECTED_REF}: {pin!r}")


class DevenvVersionTests(unittest.TestCase):
    def test_repository_pins_devenv_v22(self) -> None:
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        lock_data = json.loads((REPO_ROOT / "devenv.lock").read_text(encoding="utf-8"))
        require_v22(parse_devenv_pin(yaml_text, lock_data))

    def test_repository_uses_named_tasks_for_the_devenv_test_contract(self) -> None:
        config = (REPO_ROOT / "devenv.nix").read_text(encoding="utf-8")
        self.assertEqual(
            parse_devenv_test_tasks(config).names,
            frozenset(EXPECTED_TEST_TASKS),
        )

    def test_missing_named_test_task_fails_at_the_parse_boundary(self) -> None:
        config = (REPO_ROOT / "devenv.nix").read_text(encoding="utf-8")
        mutated = config.replace(
            'tasks."chelis:python-test" = {',
            'tasks."chelis:python-missing" = {',
        )
        with self.assertRaisesRegex(ValueError, "must define the named test tasks"):
            parse_devenv_test_tasks(mutated)

    def test_missing_enter_shell_dependency_fails_at_the_parse_boundary(self) -> None:
        config = (REPO_ROOT / "devenv.nix").read_text(encoding="utf-8")
        mutated = config.replace(
            'after = [ "devenv:enterShell" ];\n',
            "",
            1,
        )
        with self.assertRaisesRegex(ValueError, "must run after devenv:enterShell"):
            parse_devenv_test_tasks(mutated)

    def test_missing_devenv_input_fails_at_the_parse_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not define the devenv input"):
            parse_devenv_url("inputs:\n  nixpkgs:\n    url: github:NixOS/nixpkgs\n")

    def test_devenv_key_outside_inputs_fails_at_the_parse_boundary(self) -> None:
        text = (
            "inputs:\n"
            "  nixpkgs:\n"
            "    url: github:NixOS/nixpkgs\n"
            "profiles:\n"
            "  devenv:\n"
            f"    url: {EXPECTED_URL}\n"
        )
        with self.assertRaisesRegex(ValueError, "does not define the devenv input"):
            parse_devenv_url(text)

    def test_locked_module_directory_must_match_the_release_contract(self) -> None:
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        lock_data = json.loads((REPO_ROOT / "devenv.lock").read_text(encoding="utf-8"))
        lock_data["nodes"]["devenv"]["locked"]["dir"] = "other/modules"
        with self.assertRaisesRegex(ValueError, "must pin Devenv v2.2"):
            require_v22(parse_devenv_pin(yaml_text, lock_data))

    def test_old_devenv_revision_fails_the_release_contract(self) -> None:
        old_pin = DevenvPin(
            url=EXPECTED_URL,
            ref=EXPECTED_REF,
            directory="src/modules",
            locked_directory="src/modules",
            revision="9767a1f458fbd99b487dbb600a130917d4cd2b13",
        )
        with self.assertRaisesRegex(ValueError, "must pin Devenv v2.2"):
            require_v22(old_pin)


if __name__ == "__main__":
    unittest.main()
