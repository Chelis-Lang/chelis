"""Tests for exact-source dependency preparation in ecosystem drift CI."""

from pathlib import Path
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    spec = importlib.util.spec_from_file_location(
        "drift_prepare_dependencies",
        here / "drift_prepare_dependencies.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


dpd = _load_module()

SOURCE_COMMITS = {
    "nautilus": "563f2737c2988eaa05ca1e6ce4e941cf86c296b8",
    "coral": "bdb92de243c2c911c0c2a2b83bff29673186476b",
    "shoals": "502fbac05ef1c2c4fc5e8d97611b8b9c192a9707",
}


class ManifestRewriteTests(unittest.TestCase):
    def test_rewrites_compiler_and_selected_direct_dependencies_only(self):
        source = (
            "[package]\n"
            'name = "whale"\n'
            'version = "0.1.15"\n'
            'compiler = "=0.18.5"\n\n'
            "[dependencies]\n"
            'chelis-std = { version = "0.4.0" }\n'
            'shoals = { version = "0.24.9" }\n'
        )
        rewritten = dpd.rewrite_manifest(
            source,
            "0.18.10",
            {"shoals": "0.24.12"},
        )
        self.assertIn('compiler = "=0.18.10"', rewritten)
        self.assertIn('shoals = { version = "0.24.12" }', rewritten)
        self.assertIn('chelis-std = { version = "0.4.0" }', rewritten)
        self.assertIn('version = "0.1.15"', rewritten)

    def test_rejects_an_unselected_ecosystem_dependency(self):
        source = (
            "[package]\n"
            'name = "whale"\n'
            'version = "0.1.15"\n'
            'compiler = "=0.18.5"\n\n'
            "[dependencies]\n"
            'shoals = { version = "0.24.9" }\n'
        )
        with self.assertRaisesRegex(ValueError, "shoals.*not selected"):
            dpd.rewrite_manifest(source, "0.18.10", {})

    def test_preserves_selected_path_dependency_without_inventing_a_version(self):
        source = (
            "[package]\n"
            'name = "consumer"\n'
            'version = "0.1.0"\n'
            'compiler = "=0.18.5"\n\n'
            "[dependencies]\n"
            'c-earchin = { path = "../../.." }\n'
        )
        rewritten = dpd.rewrite_manifest(
            source,
            "0.18.10",
            {"c-earchin": "0.3.3"},
        )
        self.assertIn('compiler = "=0.18.10"', rewritten)
        self.assertIn('c-earchin = { path = "../../.." }', rewritten)
        self.assertNotIn('version = "0.3.3"', rewritten)


class PreparationTests(unittest.TestCase):
    @staticmethod
    def _manifest(name: str, version: str, dependencies: str = "") -> str:
        return (
            "[package]\n"
            f'name = "{name}"\n'
            f'version = "{version}"\n'
            'compiler = "=0.18.5"\n\n'
            "[dependencies]\n"
            'chelis-std = { version = "0.4.0" }\n'
            f"{dependencies}"
        )

    def test_exact_commits_build_verify_install_and_repin_the_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shell = root / "shell"
            shell.mkdir()
            (shell / "reef.toml").write_text(
                self._manifest(
                    "whale",
                    "0.1.15",
                    'shoals = { version = "0.24.9" }\n',
                ),
                encoding="utf-8",
            )
            manifests = {
                "nautilus": self._manifest("nautilus", "0.7.45"),
                "coral": self._manifest(
                    "coral",
                    "0.7.42",
                    'nautilus = { version = "0.7.44" }\n',
                ),
                "shoals": self._manifest(
                    "shoals",
                    "0.24.12",
                    'nautilus = { version = "0.7.44" }\n'
                    'coral = { version = "0.7.41" }\n',
                ),
            }
            clones: list[tuple[str, str, Path]] = []
            commands: list[list[str]] = []

            def clone(repo: str, commit: str, destination: Path) -> None:
                clones.append((repo, commit, destination))
                destination.mkdir(parents=True)
                name = destination.name
                (destination / "reef.toml").write_text(
                    manifests[name],
                    encoding="utf-8",
                )
                (destination / "src").mkdir()
                (destination / "src" / "main.ch").write_text(
                    f"module {name.title()}.Main\n",
                    encoding="utf-8",
                )
                (destination / "dist").mkdir()
                version = dpd.read_package(destination / "reef.toml").version
                (destination / "dist" / f"{name}-{version}.chb").write_bytes(b"v5")
                (
                    destination / "dist" / f"{name}-{version}.tar.zst"
                ).write_bytes(b"archive")

            def head(destination: Path) -> str:
                return SOURCE_COMMITS[destination.name]

            def run(command: list[str]) -> None:
                commands.append(command)

            dpd.prepare_dependencies(
                workspace=root / "deps",
                shell=shell,
                compiler_version="0.18.10",
                encoded_specs=(
                    "Chelis-Lang/nautilus@v0.7.45"
                    f"#{SOURCE_COMMITS['nautilus']}",
                    "Chelis-Lang/coral@v0.7.42"
                    f"#{SOURCE_COMMITS['coral']}",
                    "Chelis-Lang/shoals@v0.24.12"
                    f"#{SOURCE_COMMITS['shoals']}",
                ),
                clone=clone,
                head=head,
                run=run,
            )

            self.assertEqual(
                [(repo, commit) for repo, commit, _ in clones],
                [
                    ("Chelis-Lang/nautilus", SOURCE_COMMITS["nautilus"]),
                    ("Chelis-Lang/coral", SOURCE_COMMITS["coral"]),
                    ("Chelis-Lang/shoals", SOURCE_COMMITS["shoals"]),
                ],
            )
            build_names = [
                Path(command[-1]).name
                for command in commands
                if command[1:4] == ["reef", "build", "--no-auto-fetch"]
            ]
            self.assertEqual(build_names, ["nautilus", "coral", "shoals"])
            migrated_names = [
                Path(command[-1]).parents[1].name
                for command in commands
                if command[1:6]
                == ["migrate", "surf", "--from", "0.18", "--inplace"]
            ]
            self.assertEqual(migrated_names, ["nautilus", "coral", "shoals"])
            self.assertEqual(
                sum(
                    command[1:3] == ["reef", "verify-artifact"]
                    for command in commands
                ),
                3,
            )
            self.assertEqual(
                sum(
                    command[1:4] == ["reef", "install", "--from-monorepo"]
                    for command in commands
                ),
                3,
            )
            shell_text = (shell / "reef.toml").read_text(encoding="utf-8")
            self.assertIn('compiler = "=0.18.10"', shell_text)
            self.assertIn('shoals = { version = "0.24.12" }', shell_text)

    def test_tag_and_manifest_version_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shell = root / "shell"
            shell.mkdir()
            (shell / "reef.toml").write_text(
                self._manifest("consumer", "0.1.0"),
                encoding="utf-8",
            )

            def clone(_repo: str, _commit: str, destination: Path) -> None:
                destination.mkdir(parents=True)
                (destination / "reef.toml").write_text(
                    self._manifest("nautilus", "0.7.44"),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(ValueError, "tag.*manifest version"):
                dpd.prepare_dependencies(
                    workspace=root / "deps",
                    shell=shell,
                    compiler_version="0.18.10",
                    encoded_specs=(
                        "Chelis-Lang/nautilus@v0.7.45"
                        f"#{SOURCE_COMMITS['nautilus']}",
                    ),
                    clone=clone,
                    head=lambda _destination: SOURCE_COMMITS["nautilus"],
                    run=lambda _command: None,
                )

    def test_authenticated_checkout_failure_is_classified_as_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shell = root / "shell"
            shell.mkdir()
            (shell / "reef.toml").write_text(
                self._manifest("consumer", "0.1.0"),
                encoding="utf-8",
            )

            def clone(_repo: str, _commit: str, _destination: Path) -> None:
                raise subprocess.CalledProcessError(
                    128,
                    ["gh", "repo", "clone"],
                )

            with self.assertRaisesRegex(
                dpd.DependencySetupError,
                "authenticated exact-source checkout failed",
            ):
                dpd.prepare_dependencies(
                    workspace=root / "deps",
                    shell=shell,
                    compiler_version="0.18.10",
                    encoded_specs=(
                        "Chelis-Lang/nautilus@v0.7.45"
                        f"#{SOURCE_COMMITS['nautilus']}",
                    ),
                    clone=clone,
                    head=lambda _destination: SOURCE_COMMITS["nautilus"],
                    run=lambda _command: None,
                )

    def test_moved_tag_with_same_manifest_version_is_rejected_by_head_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shell = root / "shell"
            shell.mkdir()
            (shell / "reef.toml").write_text(
                self._manifest("consumer", "0.1.0"),
                encoding="utf-8",
            )

            def clone(_repo: str, _commit: str, destination: Path) -> None:
                destination.mkdir(parents=True)
                (destination / "reef.toml").write_text(
                    self._manifest("nautilus", "0.7.45"),
                    encoding="utf-8",
                )

            moved_commit = "a" * 40
            with self.assertRaisesRegex(
                dpd.DependencySetupError,
                "checked out.*expected",
            ):
                dpd.prepare_dependencies(
                    workspace=root / "deps",
                    shell=shell,
                    compiler_version="0.18.10",
                    encoded_specs=(
                        "Chelis-Lang/nautilus@v0.7.45"
                        f"#{SOURCE_COMMITS['nautilus']}",
                    ),
                    clone=clone,
                    head=lambda _destination: moved_commit,
                    run=lambda _command: None,
                )

    def test_source_spec_requires_full_commit_sha(self):
        invalid = (
            "Chelis-Lang/nautilus@v0.7.45",
            "Chelis-Lang/nautilus@v0.7.45#563f273",
            "Chelis-Lang/nautilus@v0.7.45#" + "g" * 40,
        )
        for encoded in invalid:
            with self.subTest(encoded=encoded):
                with self.assertRaisesRegex(ValueError, "expected.*SHA"):
                    dpd.parse_dependency_spec(encoded)


class CloneContractTests(unittest.TestCase):
    def test_authenticated_full_clone_is_followed_by_exact_checkout(self):
        destination = Path("/tmp/nautilus")
        commit = SOURCE_COMMITS["nautilus"]
        with mock.patch.object(dpd.subprocess, "run") as run:
            dpd._clone_repo("Chelis-Lang/nautilus", commit, destination)

        self.assertEqual(
            run.call_args_list,
            [
                mock.call(
                    [
                        "gh",
                        "repo",
                        "clone",
                        "Chelis-Lang/nautilus",
                        str(destination),
                        "--",
                        "--no-checkout",
                    ],
                    check=True,
                ),
                mock.call(
                    [
                        "git",
                        "-C",
                        str(destination),
                        "checkout",
                        "--detach",
                        commit,
                    ],
                    check=True,
                ),
            ],
        )

    def test_clone_contract_has_no_plain_unauthenticated_fetch(self):
        destination = Path("/tmp/nautilus")
        with mock.patch.object(dpd.subprocess, "run") as run:
            dpd._clone_repo(
                "Chelis-Lang/nautilus",
                SOURCE_COMMITS["nautilus"],
                destination,
            )

        commands = [call.args[0] for call in run.call_args_list]
        self.assertFalse(
            any(command[0] == "git" and "fetch" in command for command in commands),
            "exact private sources must not use a plain unauthenticated git fetch",
        )


class MainTests(unittest.TestCase):
    ARGV = [
        "drift_prepare_dependencies.py",
        "--workspace",
        "/tmp/deps",
        "--shell",
        "/tmp/shell",
        "--compiler-version",
        "0.18.10",
        (
            "Chelis-Lang/nautilus@v0.7.45"
            f"#{SOURCE_COMMITS['nautilus']}"
        ),
    ]

    def test_setup_failure_uses_distinct_exit_status(self):
        stderr = io.StringIO()
        with (
            mock.patch.object(
                dpd,
                "prepare_dependencies",
                side_effect=dpd.DependencySetupError("auth failed"),
            ),
            mock.patch("sys.stderr", stderr),
        ):
            status = dpd.main(self.ARGV)

        self.assertEqual(status, dpd.DEPENDENCY_SETUP_FAILURE)
        self.assertIn("dependency setup failed: auth failed", stderr.getvalue())

    def test_source_drift_failure_keeps_general_failure_status(self):
        stderr = io.StringIO()
        with (
            mock.patch.object(
                dpd,
                "prepare_dependencies",
                side_effect=ValueError("source no longer builds"),
            ),
            mock.patch("sys.stderr", stderr),
        ):
            status = dpd.main(self.ARGV)

        self.assertEqual(status, 1)
        self.assertIn(
            "dependency preparation failed: source no longer builds",
            stderr.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
