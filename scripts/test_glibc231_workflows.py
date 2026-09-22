"""Regression contract for the Debian Bullseye/glibc-2.31 CI lanes.

Run via: ``python3 scripts/test_glibc231_workflows.py`` from the repo root.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SMT_BUILD_DOC = REPO_ROOT / "docs/smt_build_setup.md"
PINNED_CONTAINER = (
    "python:3.11.13-bullseye@"
    "sha256:df0bd610af061603b29d63eb82027b64d5a3f15506e39270422b10b2a55079fc"
)
SNAPSHOT_FLAG = "--debian-bullseye-snapshot"
RAW_APT_COMMAND = re.compile(r"(?<![\w.-])(?:apt-get|apt|aptitude)(?=\s)")
PIP_INSTALL_COMMAND = re.compile(
    r"(?<![\w.-])(?:python(?:3(?:\.\d+)?)?\s+-m\s+)?"
    r"pip(?:3(?:\.\d+)?)?\s+install(?=\s|$)"
)
NO_PYPI_BOOTSTRAP_CLAIM = re.compile(
    r"\b(?:no|without)\b[^\n.;]{0,80}\bPyPI\b[^\n.;]{0,40}\bbootstrap\b",
    re.IGNORECASE,
)

GLIBC231_JOBS = (
    (REPO_ROOT / ".github/workflows/ci.yml", "smt-build-glibc231"),
    (REPO_ROOT / ".github/workflows/build-cvc5.yml", "build-linux-glibc231"),
    (REPO_ROOT / ".github/workflows/release.yml", "build-linux-x86_64-glibc231"),
)


def job_block(workflow: Path, job_name: str) -> str:
    text = workflow.read_text(encoding="utf-8")
    start = text.index(f"  {job_name}:\n")
    match = re.search(r"^  [A-Za-z0-9_-]+:\n", text[start + 1 :], re.MULTILINE)
    end = len(text) if match is None else start + 1 + match.start()
    return text[start:end]


def executable_job_text(block: str) -> str:
    """Return job text without YAML-only comment lines."""
    return "\n".join(
        line for line in block.splitlines() if not line.lstrip().startswith("#")
    )


def markdown_h2_section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^## {re.escape(heading)}\n(?P<body>.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing H2 section {heading!r} in {path}")
    return match.group("body")


class Glibc231WorkflowTests(unittest.TestCase):
    def test_every_lane_pins_the_same_python_bullseye_image(self):
        for workflow, job_name in GLIBC231_JOBS:
            with self.subTest(workflow=workflow.name, job=job_name):
                block = job_block(workflow, job_name)
                self.assertIn(f"container: {PINNED_CONTAINER}", block)

    def test_checkout_has_no_network_package_bootstrap_before_it(self):
        for workflow, job_name in GLIBC231_JOBS:
            with self.subTest(workflow=workflow.name, job=job_name):
                block = job_block(workflow, job_name)
                steps = block[block.index("    steps:\n") :]
                checkout = steps.index("uses: actions/checkout@v6")
                self.assertNotIn("apt-get", steps[:checkout])
                self.assertNotIn("run:", steps[:checkout])
                self.assertNotIn("Bootstrap checkout prerequisites", block)

    def test_every_apt_install_uses_the_pinned_snapshot(self):
        for workflow, job_name in GLIBC231_JOBS:
            with self.subTest(workflow=workflow.name, job=job_name):
                block = job_block(workflow, job_name)
                apt_invocations = [
                    line
                    for line in block.splitlines()
                    if "scripts/ci_apt_get.py" in line and not line.lstrip().startswith("#")
                ]
                self.assertEqual(len(apt_invocations), 1)
                self.assertIn(SNAPSHOT_FLAG, apt_invocations[0])

    def test_jobs_have_no_package_install_escape_hatches(self):
        for workflow, job_name in GLIBC231_JOBS:
            with self.subTest(workflow=workflow.name, job=job_name):
                block = job_block(workflow, job_name)
                executable = executable_job_text(block)
                self.assertNotRegex(executable, RAW_APT_COMMAND)
                self.assertNotRegex(executable, PIP_INSTALL_COMMAND)
                self.assertNotIn("python3-pip", executable)
                self.assertNotIn("python3-venv", executable)

    def test_release_docs_name_active_bullseye_snapshot_bootstrap(self):
        release_docs = markdown_h2_section(
            SMT_BUILD_DOC, "Release builds (chelis#422)"
        )

        for contract_term in ("Python 3.11", "Bullseye", "snapshot"):
            with self.subTest(contract_term=contract_term):
                self.assertRegex(
                    release_docs, re.compile(contract_term, re.IGNORECASE)
                )

    def test_release_docs_do_not_name_retired_container_or_pypi_dependency(self):
        release_docs = markdown_h2_section(
            SMT_BUILD_DOC, "Release builds (chelis#422)"
        )

        self.assertNotRegex(release_docs, re.compile(r"debian:11", re.IGNORECASE))
        self.assertNotRegex(
            release_docs,
            re.compile(r"(?<![A-Za-z0-9_])tomli(?![A-Za-z0-9_])", re.IGNORECASE),
        )

    def test_release_docs_explicitly_deny_separate_pypi_bootstrap(self):
        release_docs = markdown_h2_section(
            SMT_BUILD_DOC, "Release builds (chelis#422)"
        )

        self.assertRegex(release_docs, NO_PYPI_BOOTSTRAP_CLAIM)

    def test_escape_hatch_patterns_cover_realistic_command_spellings(self):
        for command in (
            "apt-get update && apt-get install -y jq",
            "sudo apt install -y jq",
            "aptitude install -y jq",
        ):
            with self.subTest(command=command):
                self.assertRegex(command, RAW_APT_COMMAND)

        for command in (
            "pip install tomli",
            "pip3 install tomli",
            "python -m pip install tomli",
            "python3 -m pip install tomli",
            "python3.11 -m pip install tomli",
            "uv pip install tomli",
        ):
            with self.subTest(command=command):
                self.assertRegex(command, PIP_INSTALL_COMMAND)


if __name__ == "__main__":
    unittest.main()
