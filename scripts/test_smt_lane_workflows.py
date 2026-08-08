"""Contract tests for the SMT lane cvc5 supply and toolchain."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
FULL_PROVE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "smt-full-prove.yml"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

SETUP_DEVENV_ACTION = (
    "Chelis-Lang/ci/actions/setup-devenv@73f017c4d3179dc313844e9d5f08d17a7879c824"
)
SMOKE_JOBS = {
    "smt-build": "x86_64-linux",
    "smt-build-darwin-arm64": "aarch64-darwin",
}
SMOKE_FORBIDDEN_MARKERS = (
    "dtolnay/rust-toolchain",
    "astral-sh/setup-uv",
    "scripts/ci_apt_get.py",
    "ci_cvc5_cache.py",
    "brew install",
)


def _job_block(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )
    return "" if match is None else match.group(0)


def smoke_lane_contract_errors(workflow: str) -> list[str]:
    """Return every SMT smoke-lane supply and toolchain contract error."""
    errors: list[str] = []
    for job_name, system in SMOKE_JOBS.items():
        block = _job_block(workflow, job_name)
        if not block:
            errors.append(f"ci.yml is missing {job_name}")
            continue
        required = (
            SETUP_DEVENV_ACTION,
            "actions/create-github-app-token@",
            "repositories: ci",
            f".#legacyPackages.{system}.cvc5-dir.drvPath",
            f"nix build .#legacyPackages.{system}.cvc5-dir --out-link .cvc5-dir",
            'printf \'CVC5_DIR=%s\\n\' "$(readlink -f .cvc5-dir)" >> "$GITHUB_ENV"',
            "actions/cache/restore@",
            "actions/cache/save@",
            "devenv shell --no-tui -- cargo build",
            "verify_release_smt.py",
        )
        for marker in required:
            if marker not in block:
                errors.append(f"{job_name} is missing {marker}")
        for marker in SMOKE_FORBIDDEN_MARKERS:
            if marker in block:
                errors.append(f"{job_name} uses forbidden marker {marker}")
    return errors


def full_prove_contract_errors(full_prove: str) -> list[str]:
    """Return every full-prove toolchain-mixing contract error."""
    errors: list[str] = []
    if "legacyPackages" in full_prove or "cvc5-dir" in full_prove:
        errors.append(
            "smt-full-prove must not export a Nix store CVC5_DIR while its "
            "cargo commands run on the host toolchain"
        )
    if "ci_cvc5_cache.py fetch" in full_prove:
        errors.append("smt-full-prove must not fetch a durable Release asset")
    for marker in ("ci_cvc5_cache.py key", "ci_cvc5_cache.py activate", "ci_cvc5_cache.py harvest"):
        if marker not in full_prove:
            errors.append(f"smt-full-prove is missing its harvest cycle: {marker}")
    return errors


def producer_retirement_errors(workflow_texts: dict[str, str]) -> list[str]:
    """Return every retired-producer reference error."""
    errors: list[str] = []
    if "build-cvc5.yml" in workflow_texts:
        errors.append("the retired producer workflow build-cvc5.yml exists")
    for name, text in workflow_texts.items():
        for marker in ("ci_publish_cvc5_release", "build-cvc5.yml"):
            if marker in text:
                errors.append(f"{name} references the retired producer: {marker}")
        if "ci_cvc5_cache.py fetch" in text:
            errors.append(f"{name} fetches a durable prebuilt cvc5 asset")
    return errors


def _workflow_texts() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
    }


class SmokeLaneContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    def test_smoke_lanes_use_the_pinned_supply(self) -> None:
        self.assertEqual(smoke_lane_contract_errors(self.workflow), [])

    def test_each_supply_marker_has_a_negative_mutation(self) -> None:
        for job_name, system in SMOKE_JOBS.items():
            block = _job_block(self.workflow, job_name)
            marker = f"nix build .#legacyPackages.{system}.cvc5-dir --out-link .cvc5-dir"
            with self.subTest(job=job_name):
                mutated = self.workflow.replace(
                    block, block.replace(marker, "removed-marker"), 1
                )
                errors = smoke_lane_contract_errors(mutated)
                self.assertIn(f"{job_name} is missing {marker}", errors)

    def test_a_host_toolchain_step_fails(self) -> None:
        block = _job_block(self.workflow, "smt-build")
        mutated = self.workflow.replace(
            block,
            block + "      - uses: dtolnay/rust-toolchain@stable\n",
            1,
        )
        errors = smoke_lane_contract_errors(mutated)
        self.assertIn(
            "smt-build uses forbidden marker dtolnay/rust-toolchain", errors
        )

    def test_a_harvested_prebuilt_fetch_fails(self) -> None:
        block = _job_block(self.workflow, "smt-build")
        mutated = self.workflow.replace(
            block,
            block + "      - run: python3 scripts/ci_cvc5_cache.py fetch\n",
            1,
        )
        errors = smoke_lane_contract_errors(mutated)
        self.assertIn("smt-build uses forbidden marker ci_cvc5_cache.py", errors)

    def test_required_context_name_is_frozen(self) -> None:
        block = _job_block(self.workflow, "smt-build")
        self.assertIn("name: SMT Feature Build (Linux)", block)


class FullProveContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.full_prove = FULL_PROVE_WORKFLOW.read_text(encoding="utf-8")

    def test_full_prove_contract(self) -> None:
        self.assertEqual(full_prove_contract_errors(self.full_prove), [])

    def test_nix_cvc5_with_host_cargo_fails(self) -> None:
        mutated = self.full_prove + "\n# nix build .#legacyPackages.x86_64-linux.cvc5-dir\n"
        errors = full_prove_contract_errors(mutated)
        self.assertIn(
            "smt-full-prove must not export a Nix store CVC5_DIR while its "
            "cargo commands run on the host toolchain",
            errors,
        )

    def test_durable_fetch_fails(self) -> None:
        mutated = self.full_prove + "\n# python3 scripts/ci_cvc5_cache.py fetch\n"
        errors = full_prove_contract_errors(mutated)
        self.assertIn("smt-full-prove must not fetch a durable Release asset", errors)

    def test_removing_the_harvest_cycle_fails(self) -> None:
        mutated = self.full_prove.replace("ci_cvc5_cache.py harvest", "removed", 1)
        errors = full_prove_contract_errors(mutated)
        self.assertIn(
            "smt-full-prove is missing its harvest cycle: ci_cvc5_cache.py harvest",
            errors,
        )


class ProducerRetirementTests(unittest.TestCase):
    def test_producer_is_retired(self) -> None:
        self.assertEqual(producer_retirement_errors(_workflow_texts()), [])

    def test_a_returning_producer_fails(self) -> None:
        texts = _workflow_texts()
        texts["build-cvc5.yml"] = "jobs: {}"
        errors = producer_retirement_errors(texts)
        self.assertIn("the retired producer workflow build-cvc5.yml exists", errors)

    def test_a_publish_reference_fails(self) -> None:
        texts = _workflow_texts()
        texts["ci.yml"] += "\n# scripts/ci_publish_cvc5_release.py\n"
        errors = producer_retirement_errors(texts)
        self.assertIn(
            "ci.yml references the retired producer: ci_publish_cvc5_release", errors
        )

    def test_publish_script_files_are_gone(self) -> None:
        self.assertFalse((REPO_ROOT / "scripts" / "ci_publish_cvc5_release.py").exists())
        self.assertFalse(
            (REPO_ROOT / "scripts" / "test_ci_publish_cvc5_release.py").exists()
        )
        self.assertFalse((WORKFLOWS_DIR / "build-cvc5.yml").exists())


if __name__ == "__main__":
    unittest.main()
