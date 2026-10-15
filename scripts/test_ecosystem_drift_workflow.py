"""Regression checks for the scheduled ecosystem drift workflow."""

from pathlib import Path
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ecosystem-drift.yml"
RUNTIME_INCLUDE = REPO_ROOT / "crates" / "chelis-runtime" / "include"
NIX_CONTRACTS = REPO_ROOT / "nix" / "contracts.nix"
LOCAL_INCLUDE = re.compile(r'^\s*#include\s+"(chelis_[^"]+\.h)"', re.MULTILINE)
# The toolchain ships the headers of the runtime its chelis carries, taken from
# `chelis runtime export`, never from the source tree (spec/08-backends.md §2.1).
COPY_HEADER = re.compile(
    r'cp "\$runtime_export/(chelis_[^\s/"]+\.h)" '
    r'"\$staging/include/"'
)
REEF_BUILD_COMMAND = re.compile(
    r"^\s*(?:run:\s+|run\s+)?"
    r"(?P<command>(?:\./target/release/)?chelis reef build(?:\s.*)?)$"
)
SOURCE_PINS = {
    "nautilus": (
        "Chelis-Lang/nautilus@v0.7.45"
        "#563f2737c2988eaa05ca1e6ce4e941cf86c296b8"
    ),
    "coral": (
        "Chelis-Lang/coral@v0.7.42"
        "#bdb92de243c2c911c0c2a2b83bff29673186476b"
    ),
    "shoals": (
        "Chelis-Lang/shoals@v0.24.12"
        "#502fbac05ef1c2c4fc5e8d97611b8b9c192a9707"
    ),
    "school": (
        "Chelis-Lang/school@v0.1.13"
        "#7904a31ff6de4b9eec82507687097cc486f2baf7"
    ),
    "octant": (
        "Chelis-Lang/octant@v0.13.0"
        "#ebd2a22150647f6252f2bb5be02b5016ff069249"
    ),
    "c-earchin": (
        "Chelis-Lang/c-earchin@v0.3.3"
        "#70720831641ba325b217427d52cf47fcba4f0b27"
    ),
}
HELLO_SOURCE_SHA = "4d9796b15f00a075c3aabffd629fecfe3daac200"
OCTANT_CLI_SHA256 = (
    "74c46fbe5cbddf489295b9df7db308bc838c7c9c8669740197cbb1dfcddd48b0"
)
EXPECTED_REEF_DEPENDENCIES = {
    "nautilus": (),
    "coral": (SOURCE_PINS["nautilus"],),
    "shoals": (
        SOURCE_PINS["nautilus"],
        SOURCE_PINS["coral"],
    ),
    "school": (),
    "hull": (),
    "whale": (
        SOURCE_PINS["nautilus"],
        SOURCE_PINS["coral"],
        SOURCE_PINS["shoals"],
    ),
    "octant": (SOURCE_PINS["nautilus"],),
    "calcify": (
        SOURCE_PINS["nautilus"],
        SOURCE_PINS["coral"],
    ),
    "c-earchin": (),
    "hydronnx": (),
    "hello-chelis": (
        SOURCE_PINS["coral"],
        SOURCE_PINS["nautilus"],
        SOURCE_PINS["school"],
    ),
}
EXPECTED_TOOL_DEPENDENCIES = {
    "hello-chelis": (
        SOURCE_PINS["octant"],
        SOURCE_PINS["c-earchin"],
    ),
}


def public_runtime_headers() -> set[str]:
    contracts = NIX_CONTRACTS.read_text(encoding="utf-8")
    block = contracts.split("publicRuntimeHeaders = [", 1)[1].split("];", 1)[0]
    return set(re.findall(r'"(chelis_[^"]+\.h)"', block))


def local_dependency_closure(headers: set[str]) -> set[str]:
    closure = set(headers)
    pending = list(headers)
    while pending:
        header = pending.pop()
        source = (RUNTIME_INCLUDE / header).read_text(encoding="utf-8")
        for dependency in LOCAL_INCLUDE.findall(source):
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
    return closure


def matrix_reef_dependencies(workflow: str) -> dict[str, tuple[str, ...]]:
    dependencies: dict[str, tuple[str, ...]] = {}
    current_repo: str | None = None
    for line in workflow.splitlines():
        stripped = line.strip()
        if stripped.startswith("- repo:"):
            current_repo = stripped.split(":", 1)[1].strip()
        elif current_repo is not None and stripped.startswith("reef_deps:"):
            encoded = stripped.split(":", 1)[1].strip().strip('"')
            dependencies[current_repo] = tuple(encoded.split())
    return dependencies


def matrix_tool_dependencies(workflow: str) -> dict[str, tuple[str, ...]]:
    dependencies: dict[str, tuple[str, ...]] = {}
    current_repo: str | None = None
    for line in workflow.splitlines():
        stripped = line.strip()
        if stripped.startswith("- repo:"):
            current_repo = stripped.split(":", 1)[1].strip()
        elif current_repo is not None and stripped.startswith("tool_deps:"):
            encoded = stripped.split(":", 1)[1].strip().strip('"')
            dependencies[current_repo] = tuple(encoded.split())
    return dependencies


def matrix_scalar(workflow: str, key: str) -> dict[str, str]:
    values: dict[str, str] = {}
    current_repo: str | None = None
    for line in workflow.splitlines():
        stripped = line.strip()
        if stripped.startswith("- repo:"):
            current_repo = stripped.split(":", 1)[1].strip()
        elif current_repo is not None and stripped.startswith(f"{key}:"):
            values[current_repo] = stripped.split(":", 1)[1].strip().strip('"')
    return values


def reef_build_command_rows(workflow: str) -> list[tuple[int, str]]:
    lines = workflow.splitlines()
    commands: list[tuple[int, str]] = []
    for index in range(len(lines)):
        match = REEF_BUILD_COMMAND.fullmatch(lines[index])
        if match is not None:
            commands.append((index, match.group("command")))
    return commands


def named_step(workflow: str, name: str) -> str:
    marker = f"      - name: {name}\n"
    self_index = workflow.index(marker)
    next_index = workflow.find("\n      - name:", self_index + len(marker))
    if next_index == -1:
        return workflow[self_index:]
    return workflow[self_index:next_index]


def cearchin_recipe(workflow: str) -> str:
    block = workflow.split("          - repo: c-earchin\n", 1)[1].split(
        "          - repo:", 1
    )[0]
    return textwrap.dedent(block.split("            cargo_recipe: |\n", 1)[1])


def step_run(step: str) -> str:
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


class CEarchinHeadRecipeTests(unittest.TestCase):
    """Execute the workflow shell with process-boundary doubles, without Cargo."""

    def run_recipe(
        self, *, fail_matrix: bool = False, fail_refresh: bool = False,
        workflow: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools = root / "bin"
            tools.mkdir()
            (root / "tests/integration/consumer").mkdir(parents=True)
            log = root / "commands.jsonl"
            stub = tools / "stub"
            stub.write_text(
                f"#!{sys.executable}\n"
                "import json, os, pathlib, sys\n"
                "entry = dict(tool=pathlib.Path(sys.argv[0]).name, args=sys.argv[1:], "
                "cwd=os.getcwd(), compiler=os.environ.get('CHELIS_BIN'))\n"
                "with open(os.environ['COMMAND_LOG'], 'a') as stream: "
                "stream.write(json.dumps(entry) + '\\n')\n"
                "if entry['args'][:1] == ['scripts/check_examples.py']:\n"
                "    assert entry['compiler'] == os.environ['EXPECTED_COMPILER']\n"
                "    assert entry['args'] == ['scripts/check_examples.py', '--regenerate']\n"
                "    if os.environ.get('FAIL_MATRIX') == '1': sys.exit(19)\n"
                "if entry['args'][:2] == ['reef', 'update'] and os.environ.get('FAIL_REFRESH') == '1': sys.exit(18)\n"
                "if entry['tool'] == 'chelis' and any(a.endswith('.dp') for a in entry['args']): sys.exit(20)\n"
            )
            stub.chmod(0o755)
            for name in ("cargo", "chelis", "python3"):
                (tools / name).symlink_to(stub)
            expected = str(tools / "chelis")
            result = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", cearchin_recipe(workflow or WORKFLOW.read_text())],
                cwd=root, capture_output=True, text=True,
                env=dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}",
                         CHELIS_TOOLCHAIN=str(root), CHELIS_BIN="/stale/release/chelis",
                         EXPECTED_COMPILER=expected, COMMAND_LOG=str(log),
                         FAIL_MATRIX=str(int(fail_matrix)), FAIL_REFRESH=str(int(fail_refresh))),
            )
            entries = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            return result, entries

    def test_head_recipe_runs_native_matrix_and_both_reef_consumers(self) -> None:
        result, entries = self.run_recipe()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [(entry["tool"], entry["args"]) for entry in entries]
        for command in (
            ("cargo", ["build", "--workspace", "--all-targets", "--locked"]),
            ("cargo", ["test", "--workspace", "--locked"]),
            ("cargo", ["clippy", "--workspace", "--all-targets", "--locked", "--", "-D", "warnings"]),
            ("cargo", ["fmt", "--all", "--", "--check"]),
            ("python3", ["scripts/generate_vocabulary.py", "--check"]),
            ("chelis", ["check", "src/vocabulary.ch"]),
            ("chelis", ["check", "tests/integration/consumer/src/main.ch"]),
            ("chelis", ["test", "tests_neg/", "--expect", "neg"]),
            ("python3", ["scripts/check_examples.py", "--regenerate"]),
            ("chelis", ["reef", "build", "--no-auto-fetch", "tests/integration/consumer"]),
            ("chelis", ["reef", "build", "--no-auto-fetch", "."]),
        ):
            self.assertIn(command, commands)
        self.assertTrue(all(entry["compiler"].endswith("/bin/chelis") for entry in entries))
        refresh = [entry for entry in entries if entry["args"] == ["reef", "update", "chelis-std", "--offline"]]
        self.assertEqual(len(refresh), 2)
        self.assertTrue(refresh[1]["cwd"].endswith("/tests/integration/consumer"))
        self.assertLess(entries.index(refresh[-1]), commands.index(("cargo", ["build", "--workspace", "--all-targets", "--locked"])))

    def test_failed_native_matrix_fails_the_lane_before_packaging(self) -> None:
        result, entries = self.run_recipe(fail_matrix=True)
        self.assertEqual(result.returncode, 19, result.stderr)
        self.assertFalse(any(entry["args"][:2] == ["reef", "build"] for entry in entries))

    def test_failed_runtime_refresh_prevents_compiler_acceptance(self) -> None:
        result, entries = self.run_recipe(fail_refresh=True)
        self.assertEqual(result.returncode, 18, result.stderr)
        self.assertFalse(any(entry["tool"] == "cargo" for entry in entries))

    def test_path_alone_cannot_select_the_matrix_compiler(self) -> None:
        workflow = WORKFLOW.read_text()
        mutated = workflow.replace('export CHELIS_BIN="$CHELIS_TOOLCHAIN/bin/chelis"', "", 1)
        self.assertNotEqual(mutated, workflow)
        result, _ = self.run_recipe(workflow=mutated)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AssertionError", result.stderr)

    def test_staged_capabilities_require_boolean_smt_before_native_probe(self) -> None:
        workflow = WORKFLOW.read_text()
        run = step_run(named_step(workflow, "Verify staged HEAD proof support"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "chelis-head-toolchain/bin/chelis"
            binary.parent.mkdir(parents=True)
            binary.write_text(
                f"#!{sys.executable}\nimport os, sys\n"
                "assert sys.argv[1:] == ['prove', '--capabilities']\n"
                "print(os.environ['CAPABILITIES'])\n"
            )
            binary.chmod(0o755)
            tools = root / "tools"
            tools.mkdir()
            python = tools / "python3"
            python.write_text(
                f"#!{sys.executable}\nimport os, pathlib, sys\n"
                "if sys.argv[1:2] == ['-c']: os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
                "assert sys.argv[1:] == ['.github/scripts/verify_release_smt.py', '--tarball', 'chelis-head-toolchain.tar.gz']\n"
                "pathlib.Path('native-probe-ran').touch()\n"
                "sys.exit(int(os.environ.get('NATIVE_STATUS', '0')))\n"
            )
            python.chmod(0o755)
            for capabilities, accepted in (({"smt_available": True}, True), ({"smt_available": False}, False), ({}, False), ({"smt_available": "true"}, False)):
                with self.subTest(capabilities=capabilities):
                    receipt = root / "native-probe-ran"
                    receipt.unlink(missing_ok=True)
                    result = subprocess.run(
                        ["bash", "-c", run], cwd=root, capture_output=True, text=True,
                        env=dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", CAPABILITIES=json.dumps(capabilities), NATIVE_STATUS="0"),
                    )
                    self.assertEqual(result.returncode == 0, accepted, result.stderr)
                    self.assertEqual(receipt.exists(), accepted)
            result = subprocess.run(
                ["bash", "-c", run], cwd=root, capture_output=True, text=True,
                env=dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", CAPABILITIES='{"smt_available": true}', NATIVE_STATUS="7"),
            )
            self.assertEqual(result.returncode, 7, "native Surf provenance rejection must fail artifact readiness")

    def assert_smt_artifact_contract(self, workflow: str) -> None:
        build = named_step(workflow, "Build chelis-cli (release, smt)")
        command = shlex.split(build.split("        run: ", 1)[1])
        self.assertEqual(command[:4], ["python3", "scripts/ci_cvc5_build.py", "--", "cargo"])
        self.assertIn("smt", command[command.index("--features") + 1].split(","))
        self.assertIn("sealed-runtime", command[command.index("--features") + 1].split(","))
        dependencies = named_step(workflow, "Install SMT build dependencies (cvc5)")
        self.assertTrue({"g++", "cmake", "libclang-dev", "git", "libgmp-dev", "libmpfr-dev"}.issubset(shlex.split(dependencies.split("        run: ", 1)[1])))
        build_job = workflow.split("  build-chelis:\n", 1)[1].split("  drift:\n", 1)[0]
        self.assertIn("timeout-minutes: 75", build_job)
        self.assertLess(workflow.index("- name: Verify staged HEAD proof support"), workflow.index("- name: Upload HEAD toolchain artifact"))

    def test_uploaded_toolchain_requires_sealed_smt_build(self) -> None:
        self.assert_smt_artifact_contract(WORKFLOW.read_text())

    def test_solver_free_build_or_unverified_upload_is_rejected(self) -> None:
        workflow = WORKFLOW.read_text()
        readiness = named_step(workflow, "Verify staged HEAD proof support")
        for mutated in (
            workflow.replace("--features sealed-runtime,smt", "--features sealed-runtime", 1),
            workflow.replace(readiness, "", 1),
            workflow.replace("scripts/ci_cvc5_build.py -- cargo", "scripts/unbounded_retry.py -- cargo", 1),
        ):
            with self.subTest():
                self.assertNotEqual(mutated, workflow)
                with self.assertRaises((AssertionError, ValueError)):
                    self.assert_smt_artifact_contract(mutated)


class EcosystemDriftWorkflowTests(unittest.TestCase):
    def assert_runtime_header_manifest(self, workflow: str) -> None:
        public = public_runtime_headers()
        required = local_dependency_closure(public)
        self.assertEqual(
            public,
            required,
            "the public runtime-header contract omits a transitive dependency",
        )
        self.assertEqual(
            set(COPY_HEADER.findall(workflow)),
            required,
            "the ecosystem toolchain omits a contracted runtime header",
        )

    def assert_reef_builds_disable_autofetch(self, workflow: str) -> None:
        commands = reef_build_command_rows(workflow)
        self.assertGreaterEqual(len(commands), 5)
        missing = [
            f"line {index + 1}: {command}"
            for index, command in commands
            if "--no-auto-fetch" not in command.split()
        ]
        self.assertEqual(
            missing,
            [],
            "every workflow reef build must disable auto-fetch",
        )

    def assert_docker_failure_attribution(self, workflow: str) -> None:
        dependencies = named_step(
            workflow,
            "Build and install exact Docker dependency sources",
        )
        report = named_step(workflow, "File/update drift tracking issue")
        language = named_step(workflow, "Fail Docker language drift")
        infrastructure = named_step(workflow, "Fail Docker infrastructure")

        self.assertIn('if [ "$status" -eq 2 ]; then', dependencies)
        self.assertIn("failure_kind=setup", dependencies)
        self.assertIn("failure_kind=drift", dependencies)
        self.assertIn('exit "$status"', dependencies)
        drift_guard = (
            r"steps\.docker_dependencies\.outcome == 'failure'\s*&&\s*"
            r"steps\.docker_dependencies\.outputs\.failure_kind == 'drift'"
        )
        self.assertRegex(
            report,
            drift_guard,
            "only dependency source drift may open or update a drift issue",
        )
        self.assertRegex(language, drift_guard)
        self.assertEqual(
            report.count("steps.docker_dependencies.outcome == 'failure'"),
            2,
            "the report condition and status may each classify dependency drift once",
        )
        self.assertEqual(
            language.count("steps.docker_dependencies.outcome == 'failure'"),
            1,
            "language drift must have exactly one guarded dependency-failure path",
        )
        self.assertNotIn("failure_kind == 'setup'", report)
        self.assertNotIn("failure_kind == 'setup'", language)
        self.assertIn("steps.docker_gate.outcome == 'failure'", language)
        self.assertIn(
            "steps.docker_dependencies.outputs.failure_kind != 'drift'",
            infrastructure,
            "dependency setup failures must be attributed to infrastructure",
        )

    def test_head_toolchain_stages_the_public_runtime_header_closure(self) -> None:
        self.assert_runtime_header_manifest(WORKFLOW.read_text(encoding="utf-8"))

    def test_deleting_a_runtime_header_copy_is_rejected(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        mutated = workflow.replace(
            '          cp "$runtime_export/chelis_runtime_views.h" "$staging/include/"\n',
            "",
            1,
        )
        self.assertNotEqual(mutated, workflow)
        with self.assertRaisesRegex(
            AssertionError,
            "ecosystem toolchain omits a contracted runtime header",
        ):
            self.assert_runtime_header_manifest(mutated)

    def test_reef_canary_raises_the_whole_suite_timeout(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        reef_test_step = text.split(
            "- name: chelis test tests/ (${{ matrix.repo }} vs HEAD)", 1
        )[1].split("# cargo legs only", 1)[0]

        self.assertIn(
            "chelis test tests/ --timeout 600 --suite-timeout 900 --jobs auto",
            reef_test_step,
        )
        self.assertEqual(reef_test_step.count("--timeout 600"), 1)
        self.assertEqual(reef_test_step.count("--suite-timeout 900"), 1)

    def test_hello_head_canary_does_not_compare_release_generated_bytes(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        hello_step = text.split(
            "- name: hello-chelis check + test + C-backend (vs HEAD)", 1
        )[1].split("# ADVISORY conformance signal", 1)[0]

        self.assertIn("run chelis check", hello_step)
        self.assertIn("run timeout 900s chelis test", hello_step)
        self.assertIn("tests/test_c_backend.py", hello_step)
        self.assertNotIn("regen_deep.py", hello_step)

    def assert_current_artifact_matrix(self, workflow: str) -> None:
        self.assertEqual(
            matrix_reef_dependencies(workflow),
            EXPECTED_REEF_DEPENDENCIES,
            "the ecosystem canary must pin the reviewed current-format Reef graph",
        )
        self.assertEqual(
            matrix_tool_dependencies(workflow),
            EXPECTED_TOOL_DEPENDENCIES,
            "the Docker leg must pin its non-Reef tool-package releases",
        )
        self.assertEqual(
            matrix_scalar(workflow, "source_sha").get("hello-chelis"),
            HELLO_SOURCE_SHA,
            "hello-chelis must be checked out at the reviewed immutable source",
        )
        self.assertEqual(
            matrix_scalar(workflow, "octant_cli_sha256").get("hello-chelis"),
            OCTANT_CLI_SHA256,
            "the Docker tool binary must have an immutable content identity",
        )

    def test_dependency_matrix_pins_current_format_releases(self) -> None:
        self.assert_current_artifact_matrix(WORKFLOW.read_text(encoding="utf-8"))

    def test_predecessor_artifact_tags_are_rejected(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        mutations = (
            ("Chelis-Lang/nautilus@v0.7.45", "Chelis-Lang/nautilus@v0.7.34"),
            ("Chelis-Lang/coral@v0.7.42", "Chelis-Lang/coral@v0.7.32"),
            ("Chelis-Lang/shoals@v0.24.12", "Chelis-Lang/shoals@v0.20.1"),
            ("Chelis-Lang/school@v0.1.13", "Chelis-Lang/school@v0.1.12"),
            ("Chelis-Lang/octant@v0.13.0", "Chelis-Lang/octant@v0.10.1"),
            ("Chelis-Lang/c-earchin@v0.3.3", "Chelis-Lang/c-earchin@v0.3.2"),
        )
        for current, predecessor in mutations:
            with self.subTest(predecessor=predecessor):
                mutated = workflow.replace(current, predecessor, 1)
                self.assertNotEqual(mutated, workflow)
                with self.assertRaises(AssertionError):
                    self.assert_current_artifact_matrix(mutated)

    def test_wrong_source_shas_are_rejected(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        for name, source in SOURCE_PINS.items():
            expected_sha = source.rsplit("#", 1)[1]
            wrong_sha = ("0" if expected_sha[0] != "0" else "1") + expected_sha[1:]
            with self.subTest(name=name):
                mutated = workflow.replace(expected_sha, wrong_sha, 1)
                self.assertNotEqual(mutated, workflow)
                with self.assertRaises(AssertionError):
                    self.assert_current_artifact_matrix(mutated)

        wrong_hello = (
            ("0" if HELLO_SOURCE_SHA[0] != "0" else "1")
            + HELLO_SOURCE_SHA[1:]
        )
        mutated = workflow.replace(HELLO_SOURCE_SHA, wrong_hello, 1)
        self.assertNotEqual(mutated, workflow)
        with self.assertRaises(AssertionError):
            self.assert_current_artifact_matrix(mutated)

    def test_dependencies_are_rebuilt_and_verified_by_head(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn(
            "cp scripts/drift_prepare_dependencies.py",
            workflow,
        )
        self.assertIn(
            'python3 "$CHELIS_TOOLCHAIN/scripts/drift_prepare_dependencies.py"',
            workflow,
        )
        self.assertNotIn('chelis reef install --from-github "$dep"', workflow)
        self.assertNotIn("CORAL_VERSION=0.7.42", workflow)
        self.assertNotIn("NAUTILUS_VERSION=0.7.45", workflow)
        self.assertIn(
            "chelis reef install --from-monorepo "
            "/opt/chelis-head/dependencies",
            workflow,
        )
        self.assertNotIn("Build hello-chelis base image", workflow)
        self.assertNotIn("shell/docker/Dockerfile", workflow)
        self.assertNotIn("FROM hello-chelis:base", workflow)
        self.assertIn("FROM ubuntu:24.04", workflow)
        self.assertIn("COPY chelis /usr/local/bin/chelis", workflow)
        self.assertIn('cp -R shell "$ctx/workspace"', workflow)
        self.assertIn("COPY workspace /workspace", workflow)
        self.assertNotIn('-v "$PWD/shell:/workspace"', workflow)
        self.assertIn("id: docker_image", workflow)
        self.assertIn("continue-on-error: true", workflow)
        self.assertIn("steps.docker_image.outcome", workflow)
        self.assertIn(
            'test "$(git rev-parse HEAD)" = "${{ matrix.source_sha }}"',
            workflow,
        )

        dependency_step = workflow.index(
            "- name: Build and install exact Docker dependency sources"
        )
        image_step = workflow.index("- name: Assemble neutral HEAD Docker image")
        self.assertLess(dependency_step, image_step)

    def test_shell_builds_disable_dependency_autofetch(self) -> None:
        self.assert_reef_builds_disable_autofetch(
            WORKFLOW.read_text(encoding="utf-8")
        )

    def test_removing_no_auto_fetch_from_each_build_is_rejected(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        original_rows = reef_build_command_rows(workflow)
        self.assertGreaterEqual(len(original_rows), 5)
        for line_index, command in original_rows:
            with self.subTest(line=line_index + 1, command=command):
                mutated_lines = workflow.splitlines(keepends=True)
                mutated_lines[line_index] = mutated_lines[line_index].replace(
                    "--no-auto-fetch",
                    "",
                    1,
                )
                mutated = "".join(mutated_lines)
                self.assertNotEqual(mutated, workflow)
                with self.assertRaisesRegex(
                    AssertionError,
                    "every workflow reef build must disable auto-fetch",
                ):
                    self.assert_reef_builds_disable_autofetch(mutated)

    def test_docker_dependency_setup_is_not_reported_as_source_drift(self) -> None:
        self.assert_docker_failure_attribution(
            WORKFLOW.read_text(encoding="utf-8")
        )

    def test_relabeling_docker_dependency_failure_as_drift_is_rejected(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        language = named_step(workflow, "Fail Docker language drift")
        marker = (
            "          ((steps.docker_dependencies.outcome == 'failure' &&\n"
            "            steps.docker_dependencies.outputs.failure_kind == 'drift') ||\n"
        )
        mutated_language = language.replace(
            marker,
            "          (steps.docker_dependencies.outcome == 'failure' ||\n",
            1,
        )
        self.assertNotEqual(mutated_language, language)
        mutated = workflow.replace(language, mutated_language, 1)
        self.assertNotEqual(mutated, workflow)
        with self.assertRaisesRegex(
            AssertionError,
            "Regex didn't match",
        ):
            self.assert_docker_failure_attribution(mutated)

    def test_removing_dependency_failure_from_infrastructure_is_rejected(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        mutated = workflow.replace(
            "           steps.docker_dependencies.outputs.failure_kind != 'drift')",
            "           steps.docker_dependencies.outputs.failure_kind == 'drift')",
            1,
        )
        self.assertNotEqual(mutated, workflow)
        with self.assertRaisesRegex(
            AssertionError,
            "dependency setup failures must be attributed to infrastructure",
        ):
            self.assert_docker_failure_attribution(mutated)

    def test_adding_an_unguarded_dependency_drift_report_is_rejected(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        report = named_step(workflow, "File/update drift tracking issue")
        mutated_report = report.replace(
            "            matrix.type != 'docker' ||\n",
            "            matrix.type != 'docker' ||\n"
            "            steps.docker_dependencies.outcome == 'failure' ||\n",
            1,
        )
        self.assertNotEqual(mutated_report, report)
        mutated = workflow.replace(report, mutated_report, 1)
        with self.assertRaisesRegex(
            AssertionError,
            "report condition and status",
        ):
            self.assert_docker_failure_attribution(mutated)


if __name__ == "__main__":
    unittest.main()
