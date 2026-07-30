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
GIT_HOOKS_MODULE = REPO_ROOT / "devenv/git-hooks.nix"
SMOKE_TEST_MODULE = REPO_ROOT / "devenv/smoke-tests.nix"
EXPECTED_URL = "github:cachix/devenv/v2.2?dir=src/modules"
EXPECTED_REF = "v2.2"
EXPECTED_REVISION = "ffce215a42d09c6375c3d60dd9c4110438fc4d87"
EXPECTED_TEST_TASKS = (
    "chelis:toolchain-test",
    "chelis:python-test",
    "chelis:c-compiler-test",
    "chelis:cpp-compiler-test",
)
EXPECTED_GIT_HOOKS_URL = "github:cachix/git-hooks.nix"
EXPECTED_ACTIVE_GIT_HOOKS = frozenset({"no-ai-authorship"})
EXPECTED_DISABLED_GIT_HOOKS = frozenset(
    {
        "actionlint",
        "check-added-large-files",
        "check-case-conflicts",
        "check-executables-have-shebangs",
        "check-json",
        "check-merge-conflicts",
        "check-python",
        "check-symlinks",
        "check-toml",
        "check-yaml",
        "detect-private-keys",
        "end-of-file-fixer",
        "fix-byte-order-marker",
        "forbid-new-submodules",
        "mixed-line-endings",
        "nixfmt",
        "rustfmt",
        "shellcheck",
        "trim-trailing-whitespace",
    }
)
GENERATED_GIT_HOOK_CONFIG = "/.pre-commit-config.yaml"


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


@dataclass(frozen=True)
class GitHookCatalog:
    disabled_names: frozenset[str]
    active_names: frozenset[str]


@dataclass(frozen=True)
class GitHooksPin:
    url: str
    revision: str


def parse_devenv_test_tasks(text: str) -> DevenvTestTasks:
    raw_names = re.findall(r'(?m)^\s{2}tasks\."([^"]+)" = \{$', text)
    names = frozenset(raw_names)
    if len(raw_names) != len(names):
        raise ValueError("the smoke-test module must not define a duplicate task")
    if names != frozenset(EXPECTED_TEST_TASKS):
        raise ValueError(f"the smoke-test module must define the named tasks: {names!r}")
    if text.count('after = [ "devenv:enterShell" ];') != len(names):
        raise ValueError("each named test task must run after devenv:enterShell")
    if text.count('before = [ "devenv:enterTest" ];') != len(names):
        raise ValueError("each named test task must run before devenv:enterTest")
    if re.search(r"(?m)^\s*enterTest\s*=", text):
        raise ValueError("the smoke-test module must not define enterTest")
    if "processes." in text or "services." in text:
        raise ValueError("the Devenv smoke check must not define a service or process")
    return DevenvTestTasks(names=names)


def parse_git_hook_catalog(text: str) -> GitHookCatalog:
    marker = "  git-hooks.hooks = {\n"
    try:
        start = text.index(marker) + len(marker)
        end = text.index("\n  };\n}", start)
    except ValueError as error:
        raise ValueError("the Git hook module must define the catalog") from error
    body = text[start:end]

    entries: list[tuple[str, str]] = re.findall(
        r"(?m)^    ([a-z0-9-]+)\.enable = (true|false);$",
        body,
    )
    for match in re.finditer(
        r"(?ms)^    ([a-z0-9-]+) = \{\n(.*?)^    \};$",
        body,
    ):
        name, block = match.groups()
        enable_values = re.findall(
            r"(?m)^      enable = (true|false);$",
            block,
        )
        if len(enable_values) != 1:
            raise ValueError(f"Git hook {name} must define one enable value")
        entries.append((name, enable_values[0]))

    names = [name for name, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("the Git hook module must not define a duplicate hook")
    expected_names = EXPECTED_DISABLED_GIT_HOOKS | EXPECTED_ACTIVE_GIT_HOOKS
    if frozenset(names) != expected_names:
        raise ValueError(f"the Git hook module must define the hook catalog: {names!r}")
    values = dict(entries)
    active_listed = [
        name for name in EXPECTED_DISABLED_GIT_HOOKS if values[name] != "false"
    ]
    if active_listed:
        raise ValueError(f"listed Git hooks must remain inactive: {active_listed!r}")
    inactive_custom = [
        name for name in EXPECTED_ACTIVE_GIT_HOOKS if values[name] != "true"
    ]
    if inactive_custom:
        raise ValueError(f"custom Git hooks must remain active: {inactive_custom!r}")

    nixfmt = re.search(
        (
            r'(?ms)^    nixfmt = \{\n.*?^      excludes = \[ '
            r'"\^Cargo\\\\\.nix\$" \];$.*?^    \};$'
        ),
        body,
    )
    if nixfmt is None:
        raise ValueError("the inactive nixfmt hook must exclude Cargo.nix")
    rustfmt = re.search(
        r"(?ms)^    rustfmt = \{\n.*?^      settings\.check = true;$.*?^    \};$",
        body,
    )
    if rustfmt is None:
        raise ValueError("the inactive rustfmt hook must use check mode")
    shellcheck = re.search(
        (
            r'(?ms)^    shellcheck = \{\n.*?^      files = "\^crates/'
            r'chelisup/bootstrap/chelisup\\\\\.sh\$";$.*?^    \};$'
        ),
        body,
    )
    if shellcheck is None:
        raise ValueError("the inactive shellcheck hook must select chelisup.sh")
    whitespace = re.search(
        (
            r'(?ms)^    trim-trailing-whitespace = \{\n.*?^      args = \[ '
            r'"--markdown-linebreak-ext=md" \];$.*?^    \};$'
        ),
        body,
    )
    if whitespace is None:
        raise ValueError("the whitespace hook must preserve Markdown line breaks")
    custom = re.search(
        (
            r"(?ms)^    no-ai-authorship = \{\n"
            r".*?^      enable = true;$"
            r".*?^      entry = \"\$\{config\.languages\.python\.package\}"
            r"/bin/python \$\{commitMessageChecker\}\";$"
            r".*?^      language = \"system\";$"
            r".*?^      pass_filenames = true;$"
            r".*?^      stages = \[ \"commit-msg\" \];$"
            r".*?^    \};$"
        ),
        body,
    )
    if custom is None:
        raise ValueError("the active authorship hook must use the commit-msg stage")
    return GitHookCatalog(
        disabled_names=EXPECTED_DISABLED_GIT_HOOKS,
        active_names=EXPECTED_ACTIVE_GIT_HOOKS,
    )


def parse_input_url(text: str, input_name: str) -> str:
    lines = text.splitlines()
    try:
        inputs_index = lines.index("inputs:")
    except ValueError as error:
        raise ValueError("devenv.yaml does not define the inputs section") from error

    input_index = None
    for index, line in enumerate(lines[inputs_index + 1 :], start=inputs_index + 1):
        if line and not line.startswith(" "):
            break
        if line == f"  {input_name}:":
            input_index = index
            break
    if input_index is None:
        raise ValueError(f"devenv.yaml does not define the {input_name} input")

    for line in lines[input_index + 1 :]:
        if line and not line.startswith("    "):
            break
        if line.startswith("    url: "):
            return line.removeprefix("    url: ")
    raise ValueError(f"the {input_name} input does not define a URL")


def parse_devenv_url(text: str) -> str:
    return parse_input_url(text, "devenv")


def parse_git_hooks_input(text: str) -> str:
    lines = text.splitlines()
    try:
        start = lines.index("  git-hooks:")
    except ValueError as error:
        raise ValueError("devenv.yaml does not define the git-hooks input") from error
    end = len(lines)
    for index, line in enumerate(lines[start + 1 :], start=start + 1):
        if line.startswith("  ") and not line.startswith("    "):
            end = index
            break
    expected = [
        "  git-hooks:",
        f"    url: {EXPECTED_GIT_HOOKS_URL}",
        "    inputs:",
        "      nixpkgs:",
        "        follows: nixpkgs",
    ]
    if lines[start:end] != expected:
        raise ValueError("the git-hooks input must follow nixpkgs")
    return parse_input_url(text, "git-hooks")


def parse_git_hooks_pin(yaml_text: str, lock_data: Any) -> GitHooksPin:
    url = parse_git_hooks_input(yaml_text)
    try:
        nodes = lock_data["nodes"]
        root = nodes[lock_data["root"]]
        node_name = root["inputs"]["git-hooks"]
        node = nodes[node_name]
        original = node["original"]
        locked = node["locked"]
        original_owner = original["owner"]
        original_repo = original["repo"]
        locked_owner = locked["owner"]
        locked_repo = locked["repo"]
        revision = locked["rev"]
        nixpkgs_input = node["inputs"]["nixpkgs"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "devenv.lock does not contain a complete git-hooks pin"
        ) from error

    identities = {
        (original_owner, original_repo),
        (locked_owner, locked_repo),
    }
    if identities != {("cachix", "git-hooks.nix")}:
        raise ValueError("the git-hooks input must use cachix/git-hooks.nix")
    if nixpkgs_input != ["nixpkgs"]:
        raise ValueError("the git-hooks input must follow nixpkgs")
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise ValueError("the git-hooks revision must be a full commit hash")
    return GitHooksPin(url=url, revision=revision)


def parse_generated_git_hook_ignore(text: str) -> str:
    matches = [line for line in text.splitlines() if line == GENERATED_GIT_HOOK_CONFIG]
    if matches != [GENERATED_GIT_HOOK_CONFIG]:
        raise ValueError(".gitignore must ignore the generated Git hook configuration")
    return matches[0]


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
        raise ValueError(
            "devenv.lock does not contain a complete devenv pin"
        ) from error

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
        config = SMOKE_TEST_MODULE.read_text(encoding="utf-8")
        self.assertEqual(
            parse_devenv_test_tasks(config).names,
            frozenset(EXPECTED_TEST_TASKS),
        )

    def test_repository_declares_the_git_hook_policy(self) -> None:
        config = GIT_HOOKS_MODULE.read_text(encoding="utf-8")
        catalog = parse_git_hook_catalog(config)
        self.assertEqual(catalog.disabled_names, EXPECTED_DISABLED_GIT_HOOKS)
        self.assertEqual(catalog.active_names, EXPECTED_ACTIVE_GIT_HOOKS)

    def test_repository_pins_the_git_hooks_input(self) -> None:
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        lock_data = json.loads((REPO_ROOT / "devenv.lock").read_text(encoding="utf-8"))
        pin = parse_git_hooks_pin(yaml_text, lock_data)
        self.assertEqual(pin.url, EXPECTED_GIT_HOOKS_URL)

    def test_repository_ignores_generated_git_hook_config(self) -> None:
        ignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertEqual(
            parse_generated_git_hook_ignore(ignore_text),
            GENERATED_GIT_HOOK_CONFIG,
        )

    def test_active_git_hook_fails_at_the_parse_boundary(self) -> None:
        config = GIT_HOOKS_MODULE.read_text(encoding="utf-8")
        mutated = config.replace(
            "    actionlint.enable = false;",
            "    actionlint.enable = true;",
        )
        with self.assertRaisesRegex(ValueError, "must remain inactive"):
            parse_git_hook_catalog(mutated)

    def test_missing_git_hook_fails_at_the_parse_boundary(self) -> None:
        config = GIT_HOOKS_MODULE.read_text(encoding="utf-8")
        mutated = config.replace("    actionlint.enable = false;\n", "")
        with self.assertRaisesRegex(ValueError, "must define the hook catalog"):
            parse_git_hook_catalog(mutated)

    def test_inactive_custom_hook_fails_at_the_parse_boundary(self) -> None:
        config = GIT_HOOKS_MODULE.read_text(encoding="utf-8")
        mutated = config.replace(
            "      enable = true;\n      name = \"Reject AI authorship markers\";",
            "      enable = false;\n      name = \"Reject AI authorship markers\";",
        )
        with self.assertRaisesRegex(ValueError, "must remain active"):
            parse_git_hook_catalog(mutated)

    def test_git_hooks_input_without_nixpkgs_follow_fails_at_parse_boundary(
        self,
    ) -> None:
        yaml_text = (
            "inputs:\n"
            "  git-hooks:\n"
            f"    url: {EXPECTED_GIT_HOOKS_URL}\n"
        )
        with self.assertRaisesRegex(ValueError, "must follow nixpkgs"):
            parse_git_hooks_input(yaml_text)

    def test_missing_git_hooks_lock_fails_at_the_parse_boundary(self) -> None:
        yaml_text = (
            "inputs:\n"
            "  git-hooks:\n"
            f"    url: {EXPECTED_GIT_HOOKS_URL}\n"
            "    inputs:\n"
            "      nixpkgs:\n"
            "        follows: nixpkgs\n"
        )
        lock_data = {"root": "root", "nodes": {"root": {"inputs": {}}}}
        with self.assertRaisesRegex(ValueError, "complete git-hooks pin"):
            parse_git_hooks_pin(yaml_text, lock_data)

    def test_missing_generated_git_hook_ignore_fails_at_parse_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "must ignore"):
            parse_generated_git_hook_ignore("/target\n")

    def test_missing_named_test_task_fails_at_the_parse_boundary(self) -> None:
        config = SMOKE_TEST_MODULE.read_text(encoding="utf-8")
        mutated = config.replace(
            'tasks."chelis:python-test" = {',
            'tasks."chelis:python-missing" = {',
        )
        with self.assertRaisesRegex(ValueError, "must define the named tasks"):
            parse_devenv_test_tasks(mutated)

    def test_missing_enter_shell_dependency_fails_at_the_parse_boundary(self) -> None:
        config = SMOKE_TEST_MODULE.read_text(encoding="utf-8")
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
