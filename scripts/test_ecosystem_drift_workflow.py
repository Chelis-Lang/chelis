"""Regression checks for the scheduled ecosystem drift workflow."""

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ecosystem-drift.yml"
RUNTIME_INCLUDE = REPO_ROOT / "crates" / "chelis-runtime" / "include"
NIX_CONTRACTS = REPO_ROOT / "nix" / "contracts.nix"
LOCAL_INCLUDE = re.compile(r'^\s*#include\s+"(chelis_[^"]+\.h)"', re.MULTILINE)
COPY_HEADER = re.compile(
    r"cp crates/chelis-runtime/include/(chelis_[^\s/]+\.h) "
    r'"\$staging/include/"'
)
EXPECTED_REEF_DEPENDENCIES = {
    "nautilus": (),
    "coral": ("Chelis-Lang/nautilus@v0.7.45",),
    "shoals": (
        "Chelis-Lang/nautilus@v0.7.45",
        "Chelis-Lang/coral@v0.7.42",
    ),
    "school": (),
    "hull": (),
    "whale": (
        "Chelis-Lang/nautilus@v0.7.45",
        "Chelis-Lang/coral@v0.7.42",
        "Chelis-Lang/shoals@v0.24.12",
    ),
    "octant": ("Chelis-Lang/nautilus@v0.7.45",),
    "calcify": (
        "Chelis-Lang/nautilus@v0.7.45",
        "Chelis-Lang/coral@v0.7.42",
    ),
    "c-earchin": (),
    "hydronnx": (),
    "hello-chelis": (
        "Chelis-Lang/coral@v0.7.42",
        "Chelis-Lang/nautilus@v0.7.45",
        "Chelis-Lang/school@v0.1.13",
    ),
}
EXPECTED_TOOL_DEPENDENCIES = {
    "hello-chelis": (
        "Chelis-Lang/octant@v0.13.0",
        "Chelis-Lang/c-earchin@v0.3.3",
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

    def test_head_toolchain_stages_the_public_runtime_header_closure(self) -> None:
        self.assert_runtime_header_manifest(WORKFLOW.read_text(encoding="utf-8"))

    def test_deleting_a_runtime_header_copy_is_rejected(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        mutated = workflow.replace(
            '          cp crates/chelis-runtime/include/chelis_runtime_views.h "$staging/include/"\n',
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
        self.assertIn(
            "mv /root/.chelis/reef /root/.chelis/reef-release-bootstrap",
            workflow,
        )

    def test_shell_builds_disable_dependency_autofetch(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        drift_job = workflow.split("\n  drift:\n", 1)[1]
        unsafe = [
            line
            for line in drift_job.splitlines()
            if line.strip() == "chelis reef build"
            or line.strip() == "run: chelis reef build"
        ]
        self.assertEqual(
            unsafe,
            [],
            "every shell build must use --no-auto-fetch after exact local seeding",
        )
        self.assertGreaterEqual(
            drift_job.count("chelis reef build --no-auto-fetch"),
            5,
        )


if __name__ == "__main__":
    unittest.main()
