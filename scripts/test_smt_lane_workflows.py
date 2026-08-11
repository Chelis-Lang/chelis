"""Contract tests for the SMT lane cvc5 supply and toolchain."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
FULL_PROVE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "smt-full-prove.yml"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CVC5_RESTORE_ACTION = (
    REPO_ROOT / ".github" / "actions" / "cvc5-cache-restore" / "action.yml"
)
CVC5_SAVE_ACTION = REPO_ROOT / ".github" / "actions" / "cvc5-cache-save" / "action.yml"

SETUP_DEVENV_ACTION = (
    "Chelis-Lang/ci/actions/setup-devenv@9d4d4c59e46a0672b5e17a5644e99700821c87e6"
)
AUTH_DEVENV_ACTION = (
    "Chelis-Lang/ci/actions/authenticate-private-ci-input@"
    "9d4d4c59e46a0672b5e17a5644e99700821c87e6"
)
SMOKE_JOBS = {
    "smt-build": "x86_64-linux",
    "smt-build-darwin-arm64": "aarch64-darwin",
}
LANE_FORBIDDEN_MARKERS = (
    "dtolnay/rust-toolchain",
    "astral-sh/setup-uv",
    "scripts/ci_apt_get.py",
    "ci_cvc5_cache.py",
    "brew install",
    "nix build .#legacyPackages.",
    "printf 'CVC5_DIR=%s\\n'",
    "export LD_LIBRARY_PATH=",
    "repositories: ci",
    "access-tokens = github.com=",
)
SMT_DEVENV_PREFIX = "devenv-retry --profile smt shell --no-tui -- "
SMT_BARE_DEVENV_PREFIX = "devenv --profile smt shell --no-tui -- "
PORTABLE_DEVENV_SHELL = "devenv-ci bash --noprofile --norc -e -o pipefail {0}"
RETIRED_SCRIPTS = (
    "ci_publish_cvc5_release.py",
    "ci_cvc5_cache.py",
    "ci_cvc5_build.py",
)


def _job_block(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )
    return "" if match is None else match.group(0)


def _lane_errors(workflow: str, job_name: str, system: str) -> list[str]:
    block = _job_block(workflow, job_name)
    if not block:
        return [f"missing SMT lane job {job_name}"]
    errors: list[str] = []
    required = (
        SETUP_DEVENV_ACTION,
        AUTH_DEVENV_ACTION,
        "app-client-id: ${{ vars.CI_APP_ID }}",
        "app-private-key: ${{ secrets.CI_APP_PRIVATE_KEY }}",
        "uses: ./.github/actions/cvc5-cache-restore",
        "uses: ./.github/actions/cvc5-cache-save",
        f"system: {system}",
        "devenv-retry build --no-tui outputs.cvc5-dir",
        f"{SMT_DEVENV_PREFIX}cargo build",
        f"{SMT_DEVENV_PREFIX}python .github/scripts/verify_release_smt.py",
    )
    if job_name == "smt-build":
        required += (f"{SMT_DEVENV_PREFIX}cargo test",)
    for marker in required:
        if marker not in block:
            errors.append(f"{job_name} is missing {marker}")
    for marker in LANE_FORBIDDEN_MARKERS:
        if marker in block:
            errors.append(f"{job_name} uses forbidden marker {marker}")
    return errors


def smoke_lane_contract_errors(workflow: str) -> list[str]:
    """Return every SMT smoke-lane supply and toolchain contract error."""
    errors: list[str] = []
    for job_name, system in SMOKE_JOBS.items():
        errors.extend(_lane_errors(workflow, job_name, system))
    return errors


def full_prove_contract_errors(full_prove: str) -> list[str]:
    """Return every full-prove supply and toolchain contract error.

    The lane joined the smoke-lane contract with openspec
    converge-full-prove-on-devenv; its smt discharge proof is the smt test
    suite rather than the release verifier, so the verifier marker is
    replaced by the corpus markers.
    """
    errors = [
        error
        for error in _lane_errors(full_prove, "full-smt-prove", "x86_64-linux")
        if not error.endswith("verify_release_smt.py")
    ]
    for marker in (
        f"{SMT_DEVENV_PREFIX}cargo test -p chelis-prove --features smt",
        f"{SMT_DEVENV_PREFIX}chelis-z3-test --cargo-subcommand test -p chelis-prove --features z3",
        f'{SMT_DEVENV_PREFIX}chelis-z3-test --cargo-subcommand test -p chelis-prove --features "smt z3" --test cross_engine_oracle',
        f"{SMT_DEVENV_PREFIX}python scripts/generate_erf_proof.py --check-only",
    ):
        if marker not in full_prove:
            errors.append(f"full-smt-prove is missing {marker}")
    return errors


def cvc5_cache_action_errors(restore: str, save: str) -> list[str]:
    """Return every contract error in the shared cvc5 closure cache pair.

    The key derivation, archive import/export, and actions/cache calls
    moved into the composite pair, so their supply markers are locked
    here instead of per-workflow.
    """
    errors: list[str] = []
    for marker in (
        "cvc5-dir.drvPath",
        "uses: actions/cache/restore@v4",
        "cvc5-dir.outPath",
        "--no-check-sigs",
        "/tmp/chelis-cvc5-cache",
        "nix-cvc5-",
    ):
        if marker not in restore:
            errors.append(f"cvc5-cache-restore is missing {marker}")
    for marker in (
        "cvc5-dir.outPath",
        "uses: actions/cache/save@v4",
        "continue-on-error: true",
        "/tmp/chelis-cvc5-cache",
        "nix-cvc5-",
    ):
        if marker not in save:
            errors.append(f"cvc5-cache-save is missing {marker}")
    shell_marker = f"shell: {PORTABLE_DEVENV_SHELL}"
    if restore.count(shell_marker) != 2:
        errors.append("cvc5-cache-restore does not own each run shell")
    if save.count(shell_marker) != 1:
        errors.append("cvc5-cache-save does not own each run shell")
    return errors


def producer_retirement_errors(workflow_texts: dict[str, str]) -> list[str]:
    """Return every retired-machinery reference error."""
    errors: list[str] = []
    if "build-cvc5.yml" in workflow_texts:
        errors.append("the retired producer workflow build-cvc5.yml exists")
    for name, text in workflow_texts.items():
        for marker in RETIRED_SCRIPTS + ("build-cvc5.yml",):
            if marker in text:
                errors.append(f"{name} references retired machinery: {marker}")
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
            marker = "devenv-retry build --no-tui outputs.cvc5-dir"
            with self.subTest(job=job_name):
                mutated = self.workflow.replace(
                    block, block.replace(marker, "removed-marker"), 1
                )
                errors = smoke_lane_contract_errors(mutated)
                self.assertIn(f"{job_name} is missing {marker}", errors)

    def test_linux_lane_requires_the_selective_devenv_retry(self) -> None:
        block = _job_block(self.workflow, "smt-build")
        mutated = self.workflow.replace(
            block, block.replace(SMT_DEVENV_PREFIX, SMT_BARE_DEVENV_PREFIX, 1), 1
        )
        errors = smoke_lane_contract_errors(mutated)
        self.assertIn(f"smt-build is missing {SMT_DEVENV_PREFIX}cargo build", errors)

    def test_a_host_toolchain_step_fails(self) -> None:
        block = _job_block(self.workflow, "smt-build")
        mutated = self.workflow.replace(
            block,
            block + "      - uses: dtolnay/rust-toolchain@stable\n",
            1,
        )
        errors = smoke_lane_contract_errors(mutated)
        self.assertIn("smt-build uses forbidden marker dtolnay/rust-toolchain", errors)

    def test_a_harvested_prebuilt_reference_fails(self) -> None:
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

    def test_a_host_toolchain_step_fails(self) -> None:
        block = _job_block(self.full_prove, "full-smt-prove")
        mutated = self.full_prove.replace(
            block,
            block + "      - uses: dtolnay/rust-toolchain@stable\n",
            1,
        )
        errors = full_prove_contract_errors(mutated)
        self.assertIn(
            "full-smt-prove uses forbidden marker dtolnay/rust-toolchain", errors
        )

    def test_a_harvest_cycle_reference_fails(self) -> None:
        block = _job_block(self.full_prove, "full-smt-prove")
        mutated = self.full_prove.replace(
            block,
            block + "      - run: python3 scripts/ci_cvc5_cache.py harvest\n",
            1,
        )
        errors = full_prove_contract_errors(mutated)
        self.assertIn("full-smt-prove uses forbidden marker ci_cvc5_cache.py", errors)

    def test_removing_the_cvc5_supply_fails(self) -> None:
        marker = "devenv-retry build --no-tui outputs.cvc5-dir"
        mutated = self.full_prove.replace(marker, "removed-marker", 1)
        errors = full_prove_contract_errors(mutated)
        self.assertIn(f"full-smt-prove is missing {marker}", errors)

    def test_restoring_the_manual_z3_loader_path_fails(self) -> None:
        block = _job_block(self.full_prove, "full-smt-prove")
        mutated = self.full_prove.replace(
            block,
            block + '      - run: export LD_LIBRARY_PATH="$Z3_LIBRARY_PATH_OVERRIDE"\n',
            1,
        )
        errors = full_prove_contract_errors(mutated)
        self.assertIn(
            "full-smt-prove uses forbidden marker export LD_LIBRARY_PATH=",
            errors,
        )


class Cvc5CacheActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.restore = CVC5_RESTORE_ACTION.read_text(encoding="utf-8")
        self.save = CVC5_SAVE_ACTION.read_text(encoding="utf-8")

    def test_composite_pair_carries_the_supply_contract(self) -> None:
        self.assertEqual(cvc5_cache_action_errors(self.restore, self.save), [])

    def test_a_missing_key_derivation_fails(self) -> None:
        mutated = self.restore.replace("cvc5-dir.drvPath", "removed", 1)
        errors = cvc5_cache_action_errors(mutated, self.save)
        self.assertIn("cvc5-cache-restore is missing cvc5-dir.drvPath", errors)

    def test_a_failing_cache_save_must_stay_best_effort(self) -> None:
        mutated = self.save.replace("continue-on-error: true", "", 1)
        errors = cvc5_cache_action_errors(self.restore, mutated)
        self.assertIn("cvc5-cache-save is missing continue-on-error: true", errors)

    def test_a_host_shell_fails_the_cache_action_contract(self) -> None:
        portable = f"shell: {PORTABLE_DEVENV_SHELL}"
        mutated = self.restore.replace(portable, "shell: bash", 1)
        errors = cvc5_cache_action_errors(mutated, self.save)
        self.assertIn("cvc5-cache-restore does not own each run shell", errors)


class ProducerRetirementTests(unittest.TestCase):
    def test_retired_machinery_stays_retired(self) -> None:
        self.assertEqual(producer_retirement_errors(_workflow_texts()), [])

    def test_a_returning_producer_fails(self) -> None:
        texts = _workflow_texts()
        texts["build-cvc5.yml"] = "jobs: {}"
        errors = producer_retirement_errors(texts)
        self.assertIn("the retired producer workflow build-cvc5.yml exists", errors)

    def test_a_cache_script_reference_fails(self) -> None:
        texts = _workflow_texts()
        texts["ci.yml"] += "\n# scripts/ci_cvc5_cache.py\n"
        errors = producer_retirement_errors(texts)
        self.assertIn("ci.yml references retired machinery: ci_cvc5_cache.py", errors)

    def test_retired_script_files_are_gone(self) -> None:
        for name in RETIRED_SCRIPTS:
            with self.subTest(script=name):
                self.assertFalse((REPO_ROOT / "scripts" / name).exists())
                self.assertFalse((REPO_ROOT / "scripts" / f"test_{name}").exists())
        self.assertFalse((WORKFLOWS_DIR / "build-cvc5.yml").exists())


if __name__ == "__main__":
    unittest.main()
