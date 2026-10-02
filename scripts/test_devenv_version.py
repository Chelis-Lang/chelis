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
EXPECTED_REVISION = "360b5eb1397291383d10845a63a0247981bd5598"
EXPECTED_URL = f"github:cachix/devenv/{EXPECTED_REVISION}?dir=src/modules"
EXPECTED_CLI_REQUIREMENT = "=2.2.3"
EXPECTED_TEST_TASKS = (
    "chelis:toolchain-test",
    "chelis:python-test",
    "chelis:c-compiler-test",
    "chelis:cpp-compiler-test",
    "chelis:kache-test",
    "chelis:pyright-test",
    "chelis:docs-test",
    "chelis:darwin-tree-sitter-test",
)
EXPECTED_TEST_TASK_PREREQUISITES = {
    "chelis:toolchain-test": frozenset(),
    "chelis:python-test": frozenset({"devenv:python:virtualenv"}),
    "chelis:c-compiler-test": frozenset({"devenv:files"}),
    "chelis:cpp-compiler-test": frozenset({"devenv:files"}),
    "chelis:kache-test": frozenset({"devenv:python:virtualenv"}),
    "chelis:pyright-test": frozenset({"devenv:python:virtualenv"}),
    "chelis:docs-test": frozenset(),
    "chelis:darwin-tree-sitter-test": frozenset({"devenv:python:virtualenv"}),
}
EXPECTED_GIT_HOOKS_URL = "github:cachix/git-hooks.nix"
# Shared inputs must pin an exact revision so `devenv update` cannot
# drift them away from the flake pins that check_nix_lock_parity.py
# compares. The revision values live in devenv.yaml and the lock files,
# not here; this contract locks the shape (no floating references).
SHARED_INPUT_URL_PREFIXES = {
    "nixpkgs": "github:cachix/devenv-nixpkgs/",
    "rust-overlay": "github:oxalica/rust-overlay/",
}
# The authorship check no longer runs as a devenv-installed prek hook: devenv
# copies the tracked `.githooks/commit-msg` into the shared hooks directory
# (chelis#1409). The install-task assertions in `parse_git_hook_catalog` are
# what now keep it from being silently dropped.
EXPECTED_ACTIVE_GIT_HOOKS: frozenset[str] = frozenset()
EXPECTED_DISABLED_GIT_HOOKS = frozenset(
    {
        "actionlint",
        "no-ai-authorship",
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
    original_revision: str
    directory: str
    locked_directory: str
    revision: str


@dataclass(frozen=True)
class DevenvCliVersionRequirement:
    constraint: str


@dataclass(frozen=True)
class DevenvTestTasks:
    names: frozenset[str]
    prerequisites: dict[str, frozenset[str]]


@dataclass(frozen=True)
class GitHookCatalog:
    disabled_names: frozenset[str]
    active_names: frozenset[str]


@dataclass(frozen=True)
class GitHooksPin:
    url: str
    revision: str


@dataclass(frozen=True)
class SharedInputPin:
    name: str
    url: str
    revision: str


def parse_devenv_test_tasks(text: str) -> DevenvTestTasks:
    task_blocks = re.findall(
        r'(?ms)^  tasks\."([^"]+)" = \{\n(.*?)^  \};$',
        text,
    )
    raw_names = [name for name, _ in task_blocks]
    names = frozenset(raw_names)
    if len(raw_names) != len(names):
        raise ValueError("the smoke-test module must not define a duplicate task")
    if names != frozenset(EXPECTED_TEST_TASKS):
        raise ValueError(f"the smoke-test module must define the named tasks: {names!r}")
    if 'after = [ "devenv:enterShell" ];' in text:
        raise ValueError("test tasks must not run during ordinary shell entry")
    if text.count('before = [ "devenv:enterTest" ];') != len(names):
        raise ValueError("each named test task must run before devenv:enterTest")
    if re.search(r"(?m)^\s*enterTest\s*=", text):
        raise ValueError("the smoke-test module must not define enterTest")
    if "processes." in text or "services." in text:
        raise ValueError("the Devenv smoke check must not define a service or process")

    prerequisites: dict[str, frozenset[str]] = {}
    for name, body in task_blocks:
        after_values = re.findall(r'(?m)^    after = \[([^]]*)\];$', body)
        if len(after_values) > 1:
            raise ValueError(f"test task {name} must define at most one after list")
        parsed = (
            frozenset(re.findall(r'"([^"]+)"', after_values[0]))
            if after_values
            else frozenset()
        )
        expected = EXPECTED_TEST_TASK_PREREQUISITES[name]
        if parsed != expected:
            raise ValueError(
                f"test task {name} must follow its generated prerequisites: "
                f"expected {sorted(expected)!r}, got {sorted(parsed)!r}"
            )
        prerequisites[name] = parsed

    return DevenvTestTasks(names=names, prerequisites=prerequisites)


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

    if "Cargo.nix" in body:
        raise ValueError("the Git hook catalog must not reference Cargo.nix")
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
    revived = re.search(
        r"(?ms)^    no-ai-authorship = \{\n.*?^      enable = true;$",
        body,
    )
    if revived is not None:
        raise ValueError(
            "no-ai-authorship must stay disabled here: installing it writes an "
            "absolute --config path into the shared .git/hooks (chelis#1409)"
        )
    install = re.search(
        r'(?ms)^  tasks\."chelis:install-git-hooks" = \{\n.*?^  \};$',
        text,
    )
    if install is None:
        raise ValueError(
            "the Git hook module must define the chelis:install-git-hooks task"
        )
    task = install.group(0)
    for fragment, why in (
        ('after = [ "devenv:enterShell" ];', "run on shell entry"),
        ("--path-format=absolute --git-common-dir", "resolve the shared hooks directory absolutely"),
        ('/.githooks"', "install from the tracked templates"),
        ("for hook in commit-msg pre-commit pre-push; do", "install every tracked hook"),
        ('cp "$template"', "copy the template rather than reference it"),
        ('mktemp "$hooks_dir/$hook.XXXXXX"', "stage under a unique name"),
        ('mv "$staged"', "install by atomic rename"),
        ("fi\n        # Unconditionally", "chmod outside the copy branch"),
        ("core.hooksPath", "warn when core.hooksPath would override the install"),
    ):
        if fragment not in task:
            raise ValueError(
                f"the commit-hook install task must {why}: missing {fragment!r}"
            )
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


def parse_cli_version_requirement(text: str) -> DevenvCliVersionRequirement:
    declarations = tuple(
        line for line in text.splitlines() if line.lstrip().startswith("require_version:")
    )
    expected = f'require_version: "{EXPECTED_CLI_REQUIREMENT}"'
    if declarations != (expected,):
        raise ValueError(
            "devenv.yaml must require the reviewed CLI range "
            f"{EXPECTED_CLI_REQUIREMENT}"
        )
    return DevenvCliVersionRequirement(constraint=EXPECTED_CLI_REQUIREMENT)


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


def parse_shared_input_pin(
    yaml_text: str, lock_data: Any, input_name: str
) -> SharedInputPin:
    url = parse_input_url(yaml_text, input_name)
    prefix = SHARED_INPUT_URL_PREFIXES[input_name]
    if not url.startswith(prefix):
        raise ValueError(f"the {input_name} input must use {prefix}<revision>")
    revision = url.removeprefix(prefix)
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(
            f"the {input_name} input must pin a full commit revision, "
            "not a floating reference"
        )
    try:
        node = lock_data["nodes"][input_name]
        original_revision = node["original"]["rev"]
        locked_revision = node["locked"]["rev"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"devenv.lock does not contain a complete {input_name} pin"
        ) from error
    if {original_revision, locked_revision} != {revision}:
        raise ValueError(
            f"the locked {input_name} revision must match the configured pin"
        )
    return SharedInputPin(name=input_name, url=url, revision=revision)


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
        original_revision = original["rev"]
        directory = original["dir"]
        locked_directory = locked["dir"]
        revision = locked["rev"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "devenv.lock does not contain a complete devenv pin"
        ) from error

    values = (original_revision, directory, locked_directory, revision)
    if not all(isinstance(value, str) for value in values):
        raise ValueError("the devenv lock fields must be strings")

    return DevenvPin(
        url=parse_devenv_url(yaml_text),
        original_revision=original_revision,
        directory=directory,
        locked_directory=locked_directory,
        revision=revision,
    )


def require_devenv_pin(pin: DevenvPin) -> None:
    expected = DevenvPin(
        url=EXPECTED_URL,
        original_revision=EXPECTED_REVISION,
        directory="src/modules",
        locked_directory="src/modules",
        revision=EXPECTED_REVISION,
    )
    if pin != expected:
        raise ValueError(f"the repository must pin the reviewed Devenv module: {pin!r}")


class DevenvVersionTests(unittest.TestCase):
    def test_repository_pins_the_reviewed_devenv_module(self) -> None:
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        lock_data = json.loads((REPO_ROOT / "devenv.lock").read_text(encoding="utf-8"))
        require_devenv_pin(parse_devenv_pin(yaml_text, lock_data))

    def test_repository_requires_the_reviewed_cli_range(self) -> None:
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        self.assertEqual(
            parse_cli_version_requirement(yaml_text).constraint,
            EXPECTED_CLI_REQUIREMENT,
        )

    def test_absent_cli_version_requirement_fails_at_the_parse_boundary(self) -> None:
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        mutated = yaml_text.replace(
            f'require_version: "{EXPECTED_CLI_REQUIREMENT}"\n',
            "",
        )
        with self.assertRaisesRegex(ValueError, "reviewed CLI range"):
            parse_cli_version_requirement(mutated)

    def test_false_cli_version_requirement_fails_at_the_parse_boundary(self) -> None:
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        mutated = yaml_text.replace(
            f'require_version: "{EXPECTED_CLI_REQUIREMENT}"',
            "require_version: false",
        )
        with self.assertRaisesRegex(ValueError, "reviewed CLI range"):
            parse_cli_version_requirement(mutated)

    def test_different_cli_range_fails_at_the_parse_boundary(self) -> None:
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        mutated = yaml_text.replace(EXPECTED_CLI_REQUIREMENT, ">=2.2.1, <=2.2.2", 1)
        with self.assertRaisesRegex(ValueError, re.escape(EXPECTED_CLI_REQUIREMENT)):
            parse_cli_version_requirement(mutated)

    def test_repository_uses_named_tasks_for_the_devenv_test_contract(self) -> None:
        config = SMOKE_TEST_MODULE.read_text(encoding="utf-8")
        tasks = parse_devenv_test_tasks(config)
        self.assertEqual(tasks.names, frozenset(EXPECTED_TEST_TASKS))
        self.assertEqual(tasks.prerequisites, EXPECTED_TEST_TASK_PREREQUISITES)

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

    def test_repository_pins_the_shared_inputs_by_revision(self) -> None:
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        lock_data = json.loads((REPO_ROOT / "devenv.lock").read_text(encoding="utf-8"))
        for input_name in SHARED_INPUT_URL_PREFIXES:
            with self.subTest(input=input_name):
                pin = parse_shared_input_pin(yaml_text, lock_data, input_name)
                self.assertEqual(pin.name, input_name)

    def test_floating_shared_input_reference_fails_the_pin_contract(self) -> None:
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        lock_data = json.loads((REPO_ROOT / "devenv.lock").read_text(encoding="utf-8"))
        mutated = re.sub(
            r"(url: github:cachix/devenv-nixpkgs/)[0-9a-f]{40}",
            r"\1rolling",
            yaml_text,
        )
        with self.assertRaisesRegex(ValueError, "full commit revision"):
            parse_shared_input_pin(mutated, lock_data, "nixpkgs")

    def test_stale_shared_input_lock_fails_the_pin_contract(self) -> None:
        yaml_text = (REPO_ROOT / "devenv.yaml").read_text(encoding="utf-8")
        lock_data = json.loads((REPO_ROOT / "devenv.lock").read_text(encoding="utf-8"))
        lock_data["nodes"]["rust-overlay"]["locked"]["rev"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "must match the configured pin"):
            parse_shared_input_pin(yaml_text, lock_data, "rust-overlay")

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

    def test_cargo_nix_hook_reference_fails_at_the_parse_boundary(self) -> None:
        config = GIT_HOOKS_MODULE.read_text(encoding="utf-8")
        mutated = config.replace(
            "    nixfmt.enable = false;",
            '    nixfmt = {\n      enable = false;\n      excludes = [ "Cargo.nix" ];\n    };',
        )
        with self.assertRaisesRegex(ValueError, "must not reference Cargo.nix"):
            parse_git_hook_catalog(mutated)

    def test_missing_git_hook_fails_at_the_parse_boundary(self) -> None:
        config = GIT_HOOKS_MODULE.read_text(encoding="utf-8")
        mutated = config.replace("    actionlint.enable = false;\n", "")
        with self.assertRaisesRegex(ValueError, "must define the hook catalog"):
            parse_git_hook_catalog(mutated)

    def test_reactivating_the_prek_authorship_hook_fails_at_the_parse_boundary(
        self,
    ) -> None:
        config = GIT_HOOKS_MODULE.read_text(encoding="utf-8")
        mutated = config.replace(
            '      enable = false;\n      name = "Reject AI authorship markers";',
            '      enable = true;\n      name = "Reject AI authorship markers";',
        )
        with self.assertRaisesRegex(ValueError, "must remain inactive"):
            parse_git_hook_catalog(mutated)

    def test_gutting_the_install_task_fails_at_the_parse_boundary(self) -> None:
        """Each fragment is one way the install could be silently defeated."""
        config = GIT_HOOKS_MODULE.read_text(encoding="utf-8")
        for fragment in (
            'after = [ "devenv:enterShell" ];',
            "--path-format=absolute --git-common-dir",
            '/.githooks"',
            "for hook in commit-msg pre-commit pre-push; do",
            'cp "$template"',
            'mktemp "$hooks_dir/$hook.XXXXXX"',
            'mv "$staged"',
        ):
            with self.subTest(fragment=fragment):
                mutated = config.replace(fragment, "")
                with self.assertRaises(ValueError):
                    parse_git_hook_catalog(mutated)

    def test_removing_the_install_task_fails_at_the_parse_boundary(self) -> None:
        config = GIT_HOOKS_MODULE.read_text(encoding="utf-8")
        mutated = config.replace(
            'tasks."chelis:install-git-hooks"', 'tasks."chelis:something-else"'
        )
        with self.assertRaisesRegex(ValueError, "install-git-hooks"):
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

    def test_shell_entry_dependency_fails_at_the_parse_boundary(self) -> None:
        config = SMOKE_TEST_MODULE.read_text(encoding="utf-8")
        mutated = config.replace(
            'before = [ "devenv:enterTest" ];',
            'after = [ "devenv:enterShell" ];\n    before = [ "devenv:enterTest" ];',
            1,
        )
        with self.assertRaisesRegex(ValueError, "must not run during ordinary shell entry"):
            parse_devenv_test_tasks(mutated)

    def test_missing_python_virtualenv_dependency_fails_at_the_parse_boundary(
        self,
    ) -> None:
        config = SMOKE_TEST_MODULE.read_text(encoding="utf-8")
        mutated = config.replace(
            '    after = [ "devenv:python:virtualenv" ];\n',
            "",
            1,
        )
        with self.assertRaisesRegex(ValueError, "generated prerequisites"):
            parse_devenv_test_tasks(mutated)

    def test_missing_generated_files_dependency_fails_at_the_parse_boundary(
        self,
    ) -> None:
        config = SMOKE_TEST_MODULE.read_text(encoding="utf-8")
        mutated = config.replace(
            '    after = [ "devenv:files" ];\n',
            "",
            1,
        )
        with self.assertRaisesRegex(ValueError, "generated prerequisites"):
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
        with self.assertRaises(ValueError):
            require_devenv_pin(parse_devenv_pin(yaml_text, lock_data))

    def test_old_devenv_revision_fails_the_release_contract(self) -> None:
        old_pin = DevenvPin(
            url=EXPECTED_URL,
            original_revision=EXPECTED_REVISION,
            directory="src/modules",
            locked_directory="src/modules",
            revision="9767a1f458fbd99b487dbb600a130917d4cd2b13",
        )
        with self.assertRaises(ValueError):
            require_devenv_pin(old_pin)


if __name__ == "__main__":
    unittest.main()
