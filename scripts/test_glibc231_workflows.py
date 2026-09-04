"""Regression contract for the Debian Bullseye/glibc-2.31 CI lanes.

Run via: ``python3 scripts/test_glibc231_workflows.py`` from the repo root.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_CONTAINER = (
    "python:3.11.13-bullseye@"
    "sha256:df0bd610af061603b29d63eb82027b64d5a3f15506e39270422b10b2a55079fc"
)
SNAPSHOT_FLAG = "--debian-bullseye-snapshot"

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

    def test_python_image_removes_the_unpinned_pypi_bootstrap(self):
        for workflow, job_name in GLIBC231_JOBS:
            with self.subTest(workflow=workflow.name, job=job_name):
                block = job_block(workflow, job_name)
                self.assertNotIn("pip3 install", block)
                self.assertNotIn("python3-pip", block)
                self.assertNotIn("python3-venv", block)


if __name__ == "__main__":
    unittest.main()
