"""Contract tests for the portable chelisup Devenv release output."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_DEVENV = REPO_ROOT / "devenv.nix"
RELEASE_MODULE = REPO_ROOT / "devenv" / "release-outputs.nix"
RELEASE_HELPER = REPO_ROOT / "nix" / "release-chelisup.nix"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
CHELISUP_INSTALL = REPO_ROOT / "crates" / "chelisup" / "src" / "install.rs"
CHELISUP_BOOTSTRAP = REPO_ROOT / "crates" / "chelisup" / "bootstrap" / "chelisup.sh"

SETUP_DEVENV_ACTION = (
    "Chelis-Lang/ci/actions/setup-devenv@73f017c4d3179dc313844e9d5f08d17a7879c824"
)
MODULE_REQUIRED_MARKERS = (
    "outputs.release-chelisup",
    "config.languages.rust.toolchainPackage",
    "../nix/release-chelisup.nix",
)
HELPER_REQUIRED_MARKERS = (
    'root + "/Cargo.nix"',
    'workspaceMembers."chelisup"',
    "pkgs.pkgsCross.musl64",
    "x86_64-unknown-linux-musl",
    "chelisup-linux-x86_64",
    "chelisup-darwin-arm64",
    "/usr/lib/libiconv.2.dylib",
    "readelf -l",
    "readelf -d",
    "otool -L",
    "--version",
    "--help",
    "sha256sum",
    'find "$out"',
)
WORKFLOW_REQUIRED_MARKERS = (
    "if: github.event_name == 'workflow_dispatch'",
    "permissions:\n      contents: read",
    "os: ubuntu-latest",
    "slug: linux-x86_64",
    "os: macos-latest",
    "slug: darwin-arm64",
    SETUP_DEVENV_ACTION,
    "actions/create-github-app-token@",
    "repositories: ci",
    "access-tokens = github.com=$CI_TOKEN",
    "devenv build --no-tui --quiet outputs.release-chelisup",
    "> release-chelisup-build.json",
    "python scripts/verify_release_chelisup.py",
    "--build-json release-chelisup-build.json",
    "--stage-root release-chelisup",
    "${{ steps.build.outputs.path }}/chelisup-${{ matrix.slug }}",
    "${{ steps.build.outputs.path }}/chelisup-${{ matrix.slug }}.sha256",
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

    forbidden = (
        "builtins.getFlake",
        "outputs.chelisup",
        "buildRustPackage",
        "cargo build",
        "fromRustupToolchainFile",
        "pkgs.pkgsStatic",
    )
    combined = release_module + "\n" + release_helper
    for marker in forbidden:
        if marker in combined:
            errors.append(f"release output uses forbidden path {marker}")

    if release_helper.count('workspaceMembers."chelisup"') != 1:
        errors.append("release helper must select chelisup exactly once")
    if "darwin-x86_64" in combined:
        errors.append("release output must not advertise darwin-x86_64")

    return errors


def _job_block(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
    )
    return "" if match is None else match.group(0)


def release_workflow_contract_errors(workflow: str) -> list[str]:
    """Return every workflow contract error for portable chelisup assets."""
    errors: list[str] = []
    job = _job_block(workflow, "build-chelisup-release")
    if not job:
        return ["release workflow is missing build-chelisup-release"]

    for marker in WORKFLOW_REQUIRED_MARKERS:
        if marker not in job:
            errors.append(f"release workflow matrix is missing {marker}")

    cargo_chelisup = re.compile(
        r"cargo\s+build[^\n]*?(?:-p|--package)\s+chelisup(?:\s|$)"
    )
    if cargo_chelisup.search(workflow):
        errors.append("release workflow must not build chelisup with Cargo")
    if "Stage chelisup asset" in workflow:
        errors.append("release workflow retains a legacy chelisup staging step")

    publish = _job_block(workflow, "publish-release")
    needs_line = next(
        (line for line in publish.splitlines() if line.strip().startswith("needs:")),
        "",
    )
    if "build-chelisup-release" not in needs_line:
        errors.append("publish-release must depend on build-chelisup-release")
    if "release-assets/**/chelisup-*" not in publish:
        errors.append("publish-release must include portable chelisup files")
    if "chelisup.sh" not in workflow:
        errors.append("release workflow must retain the bootstrap script")

    return errors


class ReleaseSourceContractTests(unittest.TestCase):
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

    def test_nix_package_reuse_fails(self) -> None:
        module = RELEASE_MODULE.read_text(encoding="utf-8") + "\n# outputs.chelisup\n"
        errors = release_source_contract_errors(
            root_devenv=ROOT_DEVENV.read_text(encoding="utf-8"),
            release_module=module,
            release_helper=RELEASE_HELPER.read_text(encoding="utf-8"),
        )
        self.assertIn("release output uses forbidden path outputs.chelisup", errors)

    def test_native_static_package_set_fails(self) -> None:
        helper = RELEASE_HELPER.read_text(encoding="utf-8").replace(
            "pkgs.pkgsCross.musl64", "pkgs.pkgsStatic"
        )
        errors = release_source_contract_errors(
            root_devenv=ROOT_DEVENV.read_text(encoding="utf-8"),
            release_module=RELEASE_MODULE.read_text(encoding="utf-8"),
            release_helper=helper,
        )
        self.assertIn("release output uses forbidden path pkgs.pkgsStatic", errors)

    def test_intel_macos_slug_fails(self) -> None:
        helper = RELEASE_HELPER.read_text(encoding="utf-8") + "\n# darwin-x86_64\n"
        errors = release_source_contract_errors(
            root_devenv=ROOT_DEVENV.read_text(encoding="utf-8"),
            release_module=RELEASE_MODULE.read_text(encoding="utf-8"),
            release_helper=helper,
        )
        self.assertIn("release output must not advertise darwin-x86_64", errors)

    def test_installer_and_bootstrap_advertise_only_published_platforms(self) -> None:
        surfaces = "\n".join(
            (
                CHELISUP_INSTALL.read_text(encoding="utf-8"),
                CHELISUP_BOOTSTRAP.read_text(encoding="utf-8"),
            )
        )
        self.assertIn("darwin-arm64", surfaces)
        self.assertIn("linux-x86_64", surfaces)
        self.assertNotIn("darwin-x86_64", surfaces)

    def test_bootstrap_rejects_intel_macos_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fake_bin = Path(raw)
            uname = fake_bin / "uname"
            uname.write_text(
                "#!/bin/sh\n"
                'case "$1" in\n'
                "  -s) printf '%s\\n' Darwin ;;\n"
                "  -m) printf '%s\\n' x86_64 ;;\n"
                "  *) exit 2 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            uname.chmod(0o755)
            for command in ("curl", "gh"):
                downloader = fake_bin / command
                downloader.write_text(
                    "#!/bin/sh\nprintf '%s\\n' unexpected-download >&2\nexit 99\n",
                    encoding="utf-8",
                )
                downloader.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            result = subprocess.run(
                ["sh", str(CHELISUP_BOOTSTRAP)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("unsupported host Darwin/x86_64", result.stderr)
            self.assertNotIn("unexpected-download", result.stderr)


class ReleaseWorkflowContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    def test_release_workflow_uses_devenv_output(self) -> None:
        self.assertEqual(release_workflow_contract_errors(self.workflow), [])

    def test_each_required_workflow_marker_has_a_negative_mutation(self) -> None:
        job = _job_block(self.workflow, "build-chelisup-release")
        for marker in WORKFLOW_REQUIRED_MARKERS:
            with self.subTest(marker=marker):
                mutated_job = job.replace(marker, "removed-marker")
                mutated = self.workflow.replace(job, mutated_job, 1)
                errors = release_workflow_contract_errors(mutated)
                self.assertIn(f"release workflow matrix is missing {marker}", errors)

    def test_direct_cargo_build_fails(self) -> None:
        mutated = self.workflow + "\n# cargo build --release -p chelisup\n"
        errors = release_workflow_contract_errors(mutated)
        self.assertIn("release workflow must not build chelisup with Cargo", errors)

    def test_missing_private_input_authentication_fails(self) -> None:
        job = _job_block(self.workflow, "build-chelisup-release")
        mutated = self.workflow.replace(
            job, job.replace("repositories: ci", "repositories: other", 1), 1
        )
        errors = release_workflow_contract_errors(mutated)
        self.assertIn("release workflow matrix is missing repositories: ci", errors)

    def test_missing_publish_dependency_fails(self) -> None:
        mutated = self.workflow.replace(
            ", build-chelisup-release]\n",
            "]\n",
            1,
        )
        errors = release_workflow_contract_errors(mutated)
        self.assertIn(
            "publish-release must depend on build-chelisup-release",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
