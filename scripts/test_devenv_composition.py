"""Tests for the composed Devenv configuration.

Run with `.venv/bin/python scripts/test_devenv_composition.py`.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMPORTS = (
    "./devenv/toolchains.nix",
    "./devenv/commands.nix",
    "./devenv/generated-files.nix",
    "./devenv/git-hooks.nix",
    "./devenv/smoke-tests.nix",
    "./devenv/rust-workspace.nix",
    "./devenv/package-outputs.nix",
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
# These hashes lock parser-normalized Nix semantics. Whitespace and comments do not change the hashes.
EXPECTED_NIX_STRUCTURE_DIGESTS = {
    "devenv/toolchains.nix": (
        "0aff8afdcc1e9f95263b57291fb4cd56cefd1778a56939239066e0134956a87e"
    ),
    "devenv/commands.nix": (
        "0fda8df63e1c6f18125acc8a27798d4e3d52e15551f505074e4f90e6e2decceb"
    ),
    "devenv/generated-files.nix": (
        "50279230118ef689769333f8c4e1698975ad01fb305c08ef2e3a4aa86f18f11d"
    ),
    "devenv/git-hooks.nix": (
        "603b5dbe4feafec53b5da262804cbcab704cc70696f164c5994c26c05870d286"
    ),
    "devenv/smoke-tests.nix": (
        "ceb8a5753e0a13482e68093f6c2a96b259598e3b16f2ff8964a1d4b10b985136"
    ),
    "devenv/rust-workspace.nix": (
        "145ad9a99b1e6cea71a95da5c9d39acf62013db0cdce5d61206ce1d68e657a8b"
    ),
    "devenv/package-outputs.nix": (
        "2f74296bca5871c05ead84def1dc0da5d8af0cd620c9fdf7a1869e1a181c43f7"
    ),
}


@dataclass(frozen=True)
class DevenvComposition:
    imports: tuple[str, ...]


@dataclass(frozen=True)
class DevenvWorkspaceImport:
    argument_names: tuple[str, ...]
    uses_devenv_crate2nix: bool
    uses_devenv_toolchain: bool


@dataclass(frozen=True)
class DevenvPackageOutputs:
    names: frozenset[str]
    workspace_members: tuple[str, ...]
    shares_artifact_assembly: bool


@dataclass(frozen=True)
class DevenvPython:
    version: tuple[int, int]
    uses_uv: bool
    manages_venv: bool
    pyo3_uses_venv: bool


@dataclass(frozen=True)
class DevenvOpenSpec:
    version: str
    source_hash: str
    dependencies_hash: str


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


def _normalized_nix(text: str, relative_path: str) -> str:
    with tempfile.TemporaryDirectory(prefix="chelis_nix_structure_") as raw_root:
        root = Path(raw_root).resolve()
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        completed = subprocess.run(
            ["nix-instantiate", "--parse", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise ValueError(f"the Nix module does not parse: {completed.stderr.strip()}")
    return completed.stdout.replace(str(root), "<ROOT>").strip()


def _require_nix_structure(normalized: str, relative_path: str) -> None:
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    expected = EXPECTED_NIX_STRUCTURE_DIGESTS[relative_path]
    if digest != expected:
        raise ValueError(
            f"{relative_path} violates its structural contract: "
            f"expected {expected}, got {digest}"
        )


def audit_devenv_rust_module_policy(modules: Mapping[str, str]) -> None:
    expected_paths = {relative.removeprefix("./") for relative in EXPECTED_IMPORTS}
    if set(modules) != expected_paths:
        raise ValueError("the Rust module policy requires the exact Devenv module set")

    forbidden = (
        "disabledModules",
        "languages/rust.nix",
        "getFlake",
        "rootCrate",
        "mkForce",
    )
    findings: dict[str, list[str]] = {}
    for relative_path, text in modules.items():
        normalized = _normalized_nix(text, relative_path)
        found = [fragment for fragment in forbidden if fragment in normalized]
        if found:
            findings[relative_path] = found
        else:
            _require_nix_structure(normalized, relative_path)
    if findings:
        raise ValueError(f"the Devenv Rust module policy rejects: {findings!r}")


def parse_workspace_import_module(text: str) -> DevenvWorkspaceImport:
    required = (
        "options.chelis.rust.importWorkspace = lib.mkOption",
        "options.chelis.rust.workspaceGraph = lib.mkOption",
        "internal = true;",
        "type = lib.types.functionTo (lib.types.functionTo lib.types.attrs);",
        'name = "crate2nix";',
        'url = "github:nix-community/crate2nix";',
        'attribute = "chelis.rust.importWorkspace";',
        "config.lib.getInput",
        "config.languages.rust.toolchainPackage",
        "import ../nix/workspace.nix",
        "root = workspace;",
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise ValueError(
            f"the workspace import must use Devenv inputs and shared graph rules: {missing!r}"
        )

    signature = re.search(
        r"config\.chelis\.rust\.importWorkspace\s*=\s*workspace:\s*"
        r"\{\s*(?P<args>[^}]*)\s*\}:",
        text,
    )
    if signature is None:
        raise ValueError("the workspace import must declare its argument set")
    arguments = tuple(signature.group("args").split())
    if arguments != ("cvc5",):
        raise ValueError(
            f"the workspace import must accept exactly the cvc5 argument: {arguments!r}"
        )

    forbidden = (
        "rootCrate",
        "languages.rust.import",
        "disabledModules",
        "lib.mkForce",
        "languages/rust.nix",
    )
    found = [fragment for fragment in forbidden if fragment in text]
    if found:
        raise ValueError(
            f"the workspace import must not replace the Devenv Rust module: {found!r}"
        )

    normalized = _normalized_nix(text, "devenv/rust-workspace.nix")
    normalized_required = (
        'crate2nix = ((config).lib.getInput { attribute = "chelis.rust.importWorkspace"; '
        'name = "crate2nix"; url = "github:nix-community/crate2nix"; });',
        "importWorkspace = (workspace: ({ cvc5 }: (import <ROOT>/nix/workspace.nix "
        "{ inherit crate2nix cvc5 lib pkgs; root = workspace; "
        "toolchain = (config).languages.rust.toolchainPackage; })));",
        "workspaceGraph = ((lib).mkOption",
    )
    normalized_missing = [
        fragment for fragment in normalized_required if fragment not in normalized
    ]
    normalized_forbidden = (
        "getFlake",
        "generatedCargoNix",
        "rootCrate",
        "disabledModules",
        "mkForce",
        "languages/rust.nix",
    )
    normalized_found = [
        fragment for fragment in normalized_forbidden if fragment in normalized
    ]
    if normalized_missing or normalized_found:
        raise ValueError(
            "the workspace import violates its structural contract: "
            f"missing={normalized_missing!r}, forbidden={normalized_found!r}"
        )
    _require_nix_structure(normalized, "devenv/rust-workspace.nix")
    return DevenvWorkspaceImport(
        argument_names=arguments,
        uses_devenv_crate2nix=True,
        uses_devenv_toolchain=True,
    )


def parse_package_outputs_module(text: str) -> DevenvPackageOutputs:
    required = (
        "config.chelis.rust.importWorkspace ../. { inherit cvc5; };",
        "chelis.rust.workspaceGraph = graph;",
        "graph.cargoGraph.workspaceMembers",
        "builtins.hasAttr name graph.cargoGraph.workspaceMembers",
        "builtins.getAttr name graph.cargoGraph.workspaceMembers",
        'throw "the Devenv graph is missing required workspace member ${name}";',
        "import ../nix/artifacts.nix",
        'requireWorkspaceMember "chelis-cli"',
        'requireWorkspaceMember "chelis-runtime"',
        'requireWorkspaceMember "chelisup"',
        "outputs = rec {",
        "chelis = artifacts.chelis;",
        "chelis-runtime = artifacts.runtime;",
        "chelisup = artifacts.chelisup;",
        "default = chelis;",
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise ValueError(
            f"the Devenv package outputs must use one workspace graph: {missing!r}"
        )
    if text.count("config.chelis.rust.importWorkspace") != 1:
        raise ValueError("the Devenv package outputs must create exactly one graph")

    forbidden = (
        "builtins.getFlake",
        "repoFlake",
        "rootCrate",
        "languages.rust.import",
        "import ../nix/packages.nix",
        "pkgs.runCommand",
        "libexec/chelisup",
    )
    found = [fragment for fragment in forbidden if fragment in text]
    if found:
        raise ValueError(
            f"the Devenv package outputs must use shared package rules: {found!r}"
        )

    normalized = _normalized_nix(text, "devenv/package-outputs.nix")
    normalized_required = (
        "graph = ((config).chelis.rust.importWorkspace <ROOT> { inherit cvc5; });",
        "source = (import <ROOT>/nix/source.nix { inherit lib; root = <ROOT>; });",
        "artifacts = (import <ROOT>/nix/artifacts.nix { inherit chelisupCrate "
        "compilerCrate lib pkgs runtimeCrate; inherit (graph) source version; });",
        "requireWorkspaceMember = (name: (if ((builtins).hasAttr name "
        "(graph).cargoGraph.workspaceMembers) then ((builtins).getAttr name "
        "(graph).cargoGraph.workspaceMembers) else (throw (\"the Devenv graph is "
        "missing required workspace member \" + name))));",
        'compilerCrate = (((requireWorkspaceMember "chelis-cli")).build.override '
        '{ features = [ ("smt") ]; });',
        'runtimeCrate = (((requireWorkspaceMember "chelis-runtime")).build.override '
        "{ features = [ ]; });",
        'chelisupCrate = (((requireWorkspaceMember "chelisup")).build.override '
        "{ features = [ ]; });",
        "outputs = rec { chelis = (artifacts).chelis; chelis-runtime = "
        "(artifacts).runtime; chelisup = (artifacts).chelisup; default = chelis; };",
    )
    normalized_missing = [
        fragment for fragment in normalized_required if fragment not in normalized
    ]
    normalized_forbidden = (
        "getFlake",
        'getAttr "importWorkspace"',
        "rootCrate",
        "languages.rust.import",
        "nix/packages.nix",
        "runCommand",
        "libexec/chelisup",
    )
    normalized_found = [
        fragment for fragment in normalized_forbidden if fragment in normalized
    ]
    if normalized_missing or normalized_found:
        raise ValueError(
            "the Devenv package outputs violate their structural contract: "
            f"missing={normalized_missing!r}, forbidden={normalized_found!r}"
        )
    _require_nix_structure(normalized, "devenv/package-outputs.nix")
    return DevenvPackageOutputs(
        names=frozenset({"chelis", "chelis-runtime", "chelisup", "default"}),
        workspace_members=("chelis-cli", "chelis-runtime", "chelisup"),
        shares_artifact_assembly=True,
    )


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


def parse_openspec_package(text: str) -> DevenvOpenSpec:
    if "openspecPinned = pkgs.openspec.overrideAttrs" not in text:
        raise ValueError("the Devenv shell must define the pinned OpenSpec package")
    if re.search(r"(?m)^      openspecPinned$", text) is None:
        raise ValueError("the Devenv shell must expose the pinned OpenSpec package")

    versions = re.findall(r'(?m)^      version = "([^"]+)";$', text)
    if versions != ["1.6.0"]:
        raise ValueError(f"the Devenv shell must pin OpenSpec 1.6.0: {versions!r}")

    hashes = re.findall(
        r'(?m)^        hash = "(sha256-[A-Za-z0-9+/]{43}=)";$',
        text,
    )
    if len(hashes) != 2 or len(set(hashes)) != 2:
        raise ValueError("the OpenSpec source and dependency hashes must be fixed")
    return DevenvOpenSpec(
        version=versions[0],
        source_hash=hashes[0],
        dependencies_hash=hashes[1],
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
        "Devenv generates one crate2nix graph",
        "does not\nevaluate the root flake",
        "Dirty worktrees can give",
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

    def test_devenv_extends_rust_imports_for_the_virtual_workspace(self) -> None:
        text = (REPO_ROOT / "devenv/rust-workspace.nix").read_text(
            encoding="utf-8"
        )
        parsed = parse_workspace_import_module(text)
        self.assertEqual(parsed.argument_names, ("cvc5",))
        self.assertTrue(parsed.uses_devenv_crate2nix)
        self.assertTrue(parsed.uses_devenv_toolchain)

    def test_all_devenv_modules_preserve_the_builtin_rust_module(self) -> None:
        modules = {
            relative.removeprefix("./"): (
                REPO_ROOT / relative.removeprefix("./")
            ).read_text(encoding="utf-8")
            for relative in EXPECTED_IMPORTS
        }
        audit_devenv_rust_module_policy(modules)

    def test_devenv_builds_outputs_from_one_workspace_graph(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        parsed = parse_package_outputs_module(text)
        self.assertEqual(
            parsed.names,
            frozenset({"chelis", "chelis-runtime", "chelisup", "default"}),
        )
        self.assertEqual(
            parsed.workspace_members,
            ("chelis-cli", "chelis-runtime", "chelisup"),
        )
        self.assertTrue(parsed.shares_artifact_assembly)

    def test_devenv_manages_python_and_the_pyo3_interpreter(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        parsed = parse_python_module(text)
        self.assertEqual(parsed.version, (3, 11))
        self.assertTrue(parsed.uses_uv)
        self.assertTrue(parsed.manages_venv)
        self.assertTrue(parsed.pyo3_uses_venv)

    def test_devenv_exposes_openspec_1_6_0(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        self.assertEqual(parse_openspec_package(text).version, "1.6.0")

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

    def test_unknown_workspace_import_argument_fails_at_the_parse_boundary(
        self,
    ) -> None:
        text = (REPO_ROOT / "devenv/rust-workspace.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace("{ cvc5 }:", "{ cvc5, ... }:")
        with self.assertRaisesRegex(ValueError, "accept exactly"):
            parse_workspace_import_module(mutated)

    def test_root_crate_selection_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/rust-workspace.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            "root = workspace;",
            "root = workspace;\n      selected = cargoNix.rootCrate;",
        )
        with self.assertRaisesRegex(ValueError, "must not replace"):
            parse_workspace_import_module(mutated)

    def test_rust_module_replacement_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/rust-workspace.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            "options.chelis.rust.importWorkspace",
            "disabledModules = [ \"languages/rust.nix\" ];\n  "
            "options.chelis.rust.importWorkspace",
            1,
        )
        with self.assertRaisesRegex(ValueError, "must not replace"):
            parse_workspace_import_module(mutated)

    def test_rust_module_replacement_in_another_module_fails_the_policy(self) -> None:
        modules = {
            relative.removeprefix("./"): (
                REPO_ROOT / relative.removeprefix("./")
            ).read_text(encoding="utf-8")
            for relative in EXPECTED_IMPORTS
        }
        modules["devenv/toolchains.nix"] = modules["devenv/toolchains.nix"].replace(
            "in\n{",
            'in\n{\n  disabledModules = [ "languages/rust.nix" ];',
            1,
        )
        with self.assertRaisesRegex(ValueError, "Rust module policy"):
            audit_devenv_rust_module_policy(modules)

    def test_computed_root_flake_access_in_another_module_fails_policy(self) -> None:
        modules = {
            relative.removeprefix("./"): (
                REPO_ROOT / relative.removeprefix("./")
            ).read_text(encoding="utf-8")
            for relative in EXPECTED_IMPORTS
        }
        modules["devenv/toolchains.nix"] = modules["devenv/toolchains.nix"].replace(
            "in\n{",
            'in\n{\n  hidden = builtins.${"get" + "Flake"} (toString ../.);',
            1,
        )
        with self.assertRaisesRegex(ValueError, "structural contract"):
            audit_devenv_rust_module_policy(modules)

    def test_computed_disabled_modules_in_another_module_fails_policy(self) -> None:
        modules = {
            relative.removeprefix("./"): (
                REPO_ROOT / relative.removeprefix("./")
            ).read_text(encoding="utf-8")
            for relative in EXPECTED_IMPORTS
        }
        modules["devenv/toolchains.nix"] = modules["devenv/toolchains.nix"].replace(
            "in\n{",
            'in\n{\n  ${"disabled" + "Modules"} = [ ("languages" + "/rust.nix") ];',
            1,
        )
        with self.assertRaisesRegex(ValueError, "structural contract"):
            audit_devenv_rust_module_policy(modules)

    def test_root_flake_evaluation_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            "let\n",
            "let\n  repoFlake = builtins.getFlake (toString ../.);\n",
            1,
        )
        with self.assertRaisesRegex(ValueError, "shared package rules"):
            parse_package_outputs_module(mutated)

    def test_duplicate_devenv_workspace_graph_fails_at_the_parse_boundary(
        self,
    ) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            "graph = config.chelis.rust.importWorkspace ../. { inherit cvc5; };",
            "graph = config.chelis.rust.importWorkspace ../. { inherit cvc5; };\n"
            "  duplicateGraph = config.chelis.rust.importWorkspace ../. "
            "{ inherit cvc5; };",
        )
        with self.assertRaisesRegex(ValueError, "exactly one graph"):
            parse_package_outputs_module(mutated)

    def test_missing_workspace_member_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            'requireWorkspaceMember "chelis-runtime"',
            'requireWorkspaceMember "missing-runtime"',
        )
        with self.assertRaisesRegex(ValueError, "one workspace graph"):
            parse_package_outputs_module(mutated)

    def test_empty_workspace_member_fallback_fails_at_the_parse_boundary(
        self,
    ) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            'throw "the Devenv graph is missing required workspace member ${name}";',
            "{ };",
        )
        with self.assertRaisesRegex(ValueError, "one workspace graph"):
            parse_package_outputs_module(mutated)

    def test_indirect_root_flake_access_fails_the_structural_contract(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            "let\n",
            'let\n  flake = (builtins.getAttr "getFlake" builtins) '
            "(toString ../.);\n",
            1,
        )
        with self.assertRaisesRegex(ValueError, "structural contract"):
            parse_package_outputs_module(mutated)

    def test_indirect_duplicate_graph_fails_the_structural_contract(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        original = "graph = config.chelis.rust.importWorkspace ../. { inherit cvc5; };"
        mutated = text.replace(
            original,
            original
            + '\n  graphFactory = builtins.getAttr "importWorkspace" '
            + "config.chelis.rust;\n"
            + "  duplicateGraph = graphFactory ../. { inherit cvc5; };",
        )
        with self.assertRaisesRegex(ValueError, "structural contract"):
            parse_package_outputs_module(mutated)

    def test_computed_duplicate_graph_fails_the_structural_contract(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        original = "graph = config.chelis.rust.importWorkspace ../. { inherit cvc5; };"
        mutated = text.replace(
            original,
            original
            + "\n  graphFactory = builtins.getAttr "
            + '("import" + "Workspace") config.chelis.rust;\n'
            + "  duplicateGraph = graphFactory ../. { inherit cvc5; };",
        )
        with self.assertRaisesRegex(ValueError, "structural contract"):
            parse_package_outputs_module(mutated)

    def test_dead_missing_member_error_fails_the_structural_contract(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        error = 'throw "the Devenv graph is missing required workspace member ${name}";'
        mutated = text.replace(error, "{ };").replace(
            "let\n",
            f"let\n  deadRequiredError = name: {error}\n",
            1,
        )
        with self.assertRaisesRegex(ValueError, "structural contract"):
            parse_package_outputs_module(mutated)

    def test_unfiltered_source_fails_the_structural_contract(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        mutated = text.replace(
            "  source = import ../nix/source.nix {\n"
            "    inherit lib;\n"
            "    root = ../.;\n"
            "  };",
            "  source = ../.;",
        )
        with self.assertRaisesRegex(ValueError, "structural contract"):
            parse_package_outputs_module(mutated)

    def test_comment_only_graph_contract_fails_the_structural_contract(self) -> None:
        text = (REPO_ROOT / "devenv/package-outputs.nix").read_text(
            encoding="utf-8"
        )
        original = "graph = config.chelis.rust.importWorkspace ../. { inherit cvc5; };"
        mutated = text.replace(
            original,
            'graphFactory = builtins.getAttr "importWorkspace" config.chelis.rust;\n'
            "  graph = graphFactory ../. { inherit cvc5; };\n"
            f"  # {original}",
        )
        with self.assertRaisesRegex(ValueError, "structural contract"):
            parse_package_outputs_module(mutated)

    def test_disabled_python_venv_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        mutated = text.replace("venv.enable = true;", "venv.enable = false;")
        with self.assertRaisesRegex(ValueError, "Python contract is incomplete"):
            parse_python_module(mutated)

    def test_wrong_or_unexposed_openspec_fails_at_the_parse_boundary(self) -> None:
        text = (REPO_ROOT / "devenv/toolchains.nix").read_text(encoding="utf-8")
        fixtures = (
            text.replace('version = "1.6.0";', 'version = "1.4.1";'),
            text.replace("      openspecPinned\n", "      openspecMissing\n"),
        )
        for fixture in fixtures:
            with self.subTest():
                with self.assertRaisesRegex(ValueError, "OpenSpec"):
                    parse_openspec_package(fixture)

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
