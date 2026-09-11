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


if __name__ == "__main__":
    unittest.main()
