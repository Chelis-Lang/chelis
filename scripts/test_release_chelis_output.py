"""Contract tests for the portable Linux chelis Devenv release output."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_DEVENV = REPO_ROOT / "devenv.nix"
RELEASE_MODULE = REPO_ROOT / "devenv" / "release-outputs.nix"
RELEASE_HELPER = REPO_ROOT / "nix" / "release-chelis.nix"
CONTRACTS = REPO_ROOT / "nix" / "contracts.nix"
CHELISUP_INSTALL = REPO_ROOT / "crates" / "chelisup" / "src" / "install.rs"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

SETUP_DEVENV_ACTION = (
    "Chelis-Lang/ci/actions/setup-devenv@73f017c4d3179dc313844e9d5f08d17a7879c824"
)
BUILD_JOB_REQUIRED_MARKERS = (
    SETUP_DEVENV_ACTION,
    "actions/create-github-app-token@",
    "repositories: ci",
    "access-tokens = github.com=$CI_TOKEN",
    "devenv build --no-tui --quiet outputs.release-chelis > release-chelis-build.json",
    "python scripts/verify_release_chelis.py",
    "--build-json release-chelis-build.json",
    "--stage-root release-chelis",
    "cp crates/chelisup/bootstrap/chelisup.sh chelisup.sh",
    "name: chelis-linux-x86_64",
    "${{ steps.build.outputs.path }}/chelis-v*-linux-x86_64.tar.gz",
    "${{ steps.build.outputs.path }}/chelis-v*-linux-x86_64.tar.gz.sha256",
    "if-no-files-found: error",
)
CONSUME_JOB_REQUIRED_MARKERS = (
    "needs: [build-chelis-release]",
    "container: ubuntu:24.04@sha256:",
    "libopenblas-dev",
    "verify_consumption_glibc.py",
    "verify_release_smt.py --tarball",
    "smoke_linux_openblas.py",
)
WORKFLOW_FORBIDDEN_MARKERS = (
    "debian:11",
    "glibc2.31",
    "glibc231",
)

MODULE_REQUIRED_MARKERS = (
    "outputs.release-chelis ",
    "../nix/release-chelis.nix",
    'pkgs.stdenv.hostPlatform.system == "x86_64-linux"',
)
HELPER_REQUIRED_MARKERS = (
    'root + "/Cargo.nix"',
    'workspaceMembers."chelis-cli"',
    'workspaceMembers."chelis-runtime"',
    'features = [ "smt" ]',
    "zstd.override { static = true; }",
    "libstdc++.a",
    "libzstd.a",
    "extraRustcOpts",
    "-Lnative=",
    "verify_release_smt.py",
    "T (malloc|free|calloc|realloc)",
    "patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2",
    "--remove-rpath",
    "readelf -d",
    "linuxReleaseGlibcFloor",
    "libgcc_s.so.1",
    "ld-linux-x86-64.so.2",
    "--library-path",
    "chelis-v${version}-linux-x86_64",
    "--version",
    "--help",
    "sha256sum",
    'find "$out"',
)
FORBIDDEN_MARKERS = (
    "builtins.getFlake",
    "buildRustPackage",
    "cargo build",
    "fromRustupToolchainFile",
    "pkgs.pkgsStatic",
    "pkgs.pkgsCross.musl64",
    "unknown-linux-musl",
)


def release_source_contract_errors(
    *, root_devenv: str, release_module: str, release_helper: str
) -> list[str]:
    """Return every static source contract error for the release output."""
    errors: list[str] = []

    if root_devenv.count("./devenv/release-outputs.nix") != 1:
        errors.append("devenv.nix must import release-outputs.nix exactly once")

    for marker in MODULE_REQUIRED_MARKERS:
        if marker not in release_module:
            errors.append(f"release module is missing {marker}")

    for marker in HELPER_REQUIRED_MARKERS:
        if marker not in release_helper:
            errors.append(f"release helper is missing {marker}")

    combined = release_module + "\n" + release_helper
    for marker in FORBIDDEN_MARKERS:
        if marker in combined:
            errors.append(f"release output uses forbidden path {marker}")

    if release_helper.count('workspaceMembers."chelis-cli"') != 1:
        errors.append("release helper must select chelis-cli exactly once")
    if release_helper.count('workspaceMembers."chelis-runtime"') != 1:
        errors.append("release helper must select chelis-runtime exactly once")

    return errors


def _contract_version(contracts: str, key: str) -> tuple[int, ...] | None:
    match = re.search(rf'{re.escape(key)} = "(\d+)\.(\d+)"', contracts)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def floor_contract_errors(contracts: str) -> list[str]:
    """Return every recorded-floor consistency error."""
    errors: list[str] = []
    floor = _contract_version(contracts, "linuxReleaseGlibcFloor")
    consumption = _contract_version(contracts, "linuxReleaseConsumptionGlibc")
    if floor is None:
        errors.append("contracts must record linuxReleaseGlibcFloor as MAJOR.MINOR")
    if consumption is None:
        errors.append("contracts must record linuxReleaseConsumptionGlibc as MAJOR.MINOR")
    if floor is not None and consumption is not None and floor > consumption:
        errors.append(
            "recorded glibc floor exceeds the off-Nix consumption environment"
        )
    return errors


class ReleaseChelisSourceContractTests(unittest.TestCase):
    def _current_errors(self) -> list[str]:
        return release_source_contract_errors(
            root_devenv=ROOT_DEVENV.read_text(encoding="utf-8"),
            release_module=RELEASE_MODULE.read_text(encoding="utf-8"),
            release_helper=RELEASE_HELPER.read_text(encoding="utf-8"),
        )

    def test_release_output_source_contract(self) -> None:
        self.assertEqual(self._current_errors(), [])

    def test_each_required_source_marker_has_a_negative_mutation(self) -> None:
        root_devenv = ROOT_DEVENV.read_text(encoding="utf-8")
        release_module = RELEASE_MODULE.read_text(encoding="utf-8")
        release_helper = RELEASE_HELPER.read_text(encoding="utf-8")
        for marker in MODULE_REQUIRED_MARKERS:
            with self.subTest(module_marker=marker):
                errors = release_source_contract_errors(
                    root_devenv=root_devenv,
                    release_module=release_module.replace(marker, "removed-marker"),
                    release_helper=release_helper,
                )
                self.assertIn(f"release module is missing {marker}", errors)
        for marker in HELPER_REQUIRED_MARKERS:
            with self.subTest(helper_marker=marker):
                errors = release_source_contract_errors(
                    root_devenv=root_devenv,
                    release_module=release_module,
                    release_helper=release_helper.replace(marker, "removed-marker"),
                )
                self.assertIn(f"release helper is missing {marker}", errors)

    def test_musl_runtime_target_fails(self) -> None:
        helper = RELEASE_HELPER.read_text(encoding="utf-8") + "\n# x86_64-unknown-linux-musl\n"
        errors = release_source_contract_errors(
            root_devenv=ROOT_DEVENV.read_text(encoding="utf-8"),
            release_module=RELEASE_MODULE.read_text(encoding="utf-8"),
            release_helper=helper,
        )
        self.assertIn("release output uses forbidden path unknown-linux-musl", errors)

    def test_flake_evaluation_fails(self) -> None:
        helper = RELEASE_HELPER.read_text(encoding="utf-8") + "\n# builtins.getFlake\n"
        errors = release_source_contract_errors(
            root_devenv=ROOT_DEVENV.read_text(encoding="utf-8"),
            release_module=RELEASE_MODULE.read_text(encoding="utf-8"),
            release_helper=helper,
        )
        self.assertIn("release output uses forbidden path builtins.getFlake", errors)

    def test_host_cargo_build_fails(self) -> None:
        helper = RELEASE_HELPER.read_text(encoding="utf-8") + "\n# cargo build\n"
        errors = release_source_contract_errors(
            root_devenv=ROOT_DEVENV.read_text(encoding="utf-8"),
            release_module=RELEASE_MODULE.read_text(encoding="utf-8"),
            release_helper=helper,
        )
        self.assertIn("release output uses forbidden path cargo build", errors)

    def test_asset_name_matches_the_installer_request(self) -> None:
        install = CHELISUP_INSTALL.read_text(encoding="utf-8")
        helper = RELEASE_HELPER.read_text(encoding="utf-8")
        self.assertIn('chelis-v{version}-{slug}.tar.gz', install)
        self.assertIn('("linux", "x86_64") => Ok("linux-x86_64")', install)
        self.assertIn('chelis-v${version}-linux-x86_64', helper)


def _job_block(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )
    return "" if match is None else match.group(0)


def _job_names(workflow: str) -> list[str]:
    return re.findall(r"(?m)^  ([A-Za-z0-9_-]+):\n", workflow)


def release_workflow_contract_errors(workflow: str) -> list[str]:
    """Return every workflow contract error for the Linux toolchain release."""
    errors: list[str] = []

    build_job = _job_block(workflow, "build-chelis-release")
    if not build_job:
        errors.append("release workflow is missing build-chelis-release")
    else:
        for marker in BUILD_JOB_REQUIRED_MARKERS:
            if marker not in build_job:
                errors.append(f"build-chelis-release is missing {marker}")

    consume_job = _job_block(workflow, "consume-chelis-release")
    if not consume_job:
        errors.append("release workflow is missing consume-chelis-release")
    else:
        for marker in CONSUME_JOB_REQUIRED_MARKERS:
            if marker not in consume_job:
                errors.append(f"consume-chelis-release is missing {marker}")

    for marker in WORKFLOW_FORBIDDEN_MARKERS:
        if marker in workflow:
            errors.append(f"release workflow uses forbidden marker {marker}")

    for job_name in _job_names(workflow):
        if job_name == "build-darwin-arm64":
            continue
        if "cargo build" in _job_block(workflow, job_name):
            errors.append(
                f"job {job_name} builds a published Linux artifact with Cargo"
            )

    publish = _job_block(workflow, "publish-release")
    needs_line = next(
        (line for line in publish.splitlines() if line.strip().startswith("needs:")),
        "",
    )
    for dependency in ("build-chelis-release", "consume-chelis-release"):
        if dependency not in needs_line:
            errors.append(f"publish-release must depend on {dependency}")
    if "release-assets/**/*.tar.gz" not in publish:
        errors.append("publish-release must include the release tarballs")

    return errors


class ReleaseChelisWorkflowContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    def test_release_workflow_uses_devenv_output(self) -> None:
        self.assertEqual(release_workflow_contract_errors(self.workflow), [])

    def test_each_build_marker_has_a_negative_mutation(self) -> None:
        job = _job_block(self.workflow, "build-chelis-release")
        for marker in BUILD_JOB_REQUIRED_MARKERS:
            with self.subTest(marker=marker):
                mutated = self.workflow.replace(
                    job, job.replace(marker, "removed-marker"), 1
                )
                errors = release_workflow_contract_errors(mutated)
                self.assertIn(f"build-chelis-release is missing {marker}", errors)

    def test_each_consume_marker_has_a_negative_mutation(self) -> None:
        job = _job_block(self.workflow, "consume-chelis-release")
        for marker in CONSUME_JOB_REQUIRED_MARKERS:
            with self.subTest(marker=marker):
                mutated = self.workflow.replace(
                    job, job.replace(marker, "removed-marker"), 1
                )
                errors = release_workflow_contract_errors(mutated)
                self.assertIn(f"consume-chelis-release is missing {marker}", errors)

    def test_distribution_container_build_fails(self) -> None:
        mutated = self.workflow + "\n# container: debian:11\n"
        errors = release_workflow_contract_errors(mutated)
        self.assertIn("release workflow uses forbidden marker debian:11", errors)

    def test_glibc231_asset_fails(self) -> None:
        mutated = self.workflow + "\n# chelis-v0.0.0-linux-x86_64-glibc2.31.tar.gz\n"
        errors = release_workflow_contract_errors(mutated)
        self.assertIn("release workflow uses forbidden marker glibc2.31", errors)

    def test_cargo_build_outside_darwin_fails(self) -> None:
        job = _job_block(self.workflow, "consume-chelis-release")
        mutated = self.workflow.replace(
            job,
            job + "      - name: Rogue cargo step\n"
            "        run: cargo build --release -p chelis-cli\n",
            1,
        )
        errors = release_workflow_contract_errors(mutated)
        self.assertIn(
            "job consume-chelis-release builds a published Linux artifact with Cargo",
            errors,
        )

    def test_darwin_job_keeps_its_cargo_evidence(self) -> None:
        darwin = _job_block(self.workflow, "build-darwin-arm64")
        self.assertIn("cargo build --release -p chelis-cli --features smt", darwin)
        self.assertIn("cargo build --release -p chelis-runtime", darwin)

    def test_missing_publish_dependency_fails(self) -> None:
        mutated = self.workflow.replace(
            "needs: [build-chelis-release, consume-chelis-release, ",
            "needs: [",
            1,
        )
        errors = release_workflow_contract_errors(mutated)
        self.assertIn("publish-release must depend on build-chelis-release", errors)
        self.assertIn("publish-release must depend on consume-chelis-release", errors)

    def test_ci_workflow_has_no_glibc231_lane(self) -> None:
        ci = CI_WORKFLOW.read_text(encoding="utf-8")
        self.assertNotIn("smt-build-glibc231", ci)
        self.assertNotIn("debian:11", ci)


class RecordedFloorContractTests(unittest.TestCase):
    def test_recorded_floor_contract(self) -> None:
        self.assertEqual(
            floor_contract_errors(CONTRACTS.read_text(encoding="utf-8")), []
        )

    def test_missing_floor_fails(self) -> None:
        contracts = CONTRACTS.read_text(encoding="utf-8").replace(
            "linuxReleaseGlibcFloor", "removedKey"
        )
        errors = floor_contract_errors(contracts)
        self.assertIn(
            "contracts must record linuxReleaseGlibcFloor as MAJOR.MINOR", errors
        )

    def test_missing_consumption_glibc_fails(self) -> None:
        contracts = CONTRACTS.read_text(encoding="utf-8").replace(
            "linuxReleaseConsumptionGlibc", "removedKey"
        )
        errors = floor_contract_errors(contracts)
        self.assertIn(
            "contracts must record linuxReleaseConsumptionGlibc as MAJOR.MINOR", errors
        )

    def test_floor_above_consumption_environment_fails(self) -> None:
        contracts = CONTRACTS.read_text(encoding="utf-8").replace(
            'linuxReleaseGlibcFloor = "2.39"', 'linuxReleaseGlibcFloor = "2.99"'
        )
        errors = floor_contract_errors(contracts)
        self.assertIn(
            "recorded glibc floor exceeds the off-Nix consumption environment", errors
        )

    def test_helper_reads_the_recorded_floor(self) -> None:
        helper = RELEASE_HELPER.read_text(encoding="utf-8")
        self.assertIn("contracts.linuxReleaseGlibcFloor", helper)


if __name__ == "__main__":
    unittest.main()
