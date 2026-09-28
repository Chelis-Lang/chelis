"""Regression checks for the scheduled ecosystem drift workflow."""

from pathlib import Path
import re
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
