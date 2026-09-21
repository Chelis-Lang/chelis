"""Lock the properties that make `.githooks/commit-msg` worktree-safe.

chelis#1409: the previous hook lived in `.git/hooks`, which every worktree of a
clone shares, and recorded an absolute `--config` path to whichever worktree
installed it. Commits from every other worktree failed, and all of them failed
once the named worktree was deleted. Exercise the installed hook against the
committing repository and its owned interpreter, including activated profiles.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / ".githooks" / "commit-msg"


class CommitHookContract(unittest.TestCase):
    def test_hook_is_tracked_and_executable(self) -> None:
        self.assertTrue(HOOK.is_file(), f"{HOOK} is missing")
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".githooks/commit-msg"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(tracked.returncode, 0, "the hook must be tracked, not local state")
        self.assertTrue(HOOK.stat().st_mode & 0o111, "the hook must be executable")

    def test_hook_fails_closed_when_truncated(self) -> None:
        """A truncated hook must not be a valid program that does nothing.

        git runs a zero-length or partial executable hook and accepts the
        commit when it exits 0, with no diagnostic. Opening a brace group on
        line 2 makes every later prefix an unterminated compound command, so
        sh rejects it loudly instead. Only fragments of the shebang line
        itself can still parse, and those cannot be made to fail.
        """

        raw = HOOK.read_bytes()
        message = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        message.write("docs: subject\n\nCo-Authored-By: Claude <a@b>\n")
        message.close()
        # Sample rather than sweep: 2000 subprocess spawns is too slow for the
        # default suite, and the property is structural.
        for cut in range(16, len(raw), 97):
            with tempfile.NamedTemporaryFile("wb", suffix=".sh", delete=False) as handle:
                handle.write(raw[:cut])
                fragment = handle.name
            result = subprocess.run(
                ["/bin/sh", fragment, message.name], capture_output=True, check=False
            )
            Path(fragment).unlink()
            silent_accept = result.returncode == 0 and not result.stdout and not result.stderr
            self.assertFalse(silent_accept, f"prefix of {cut} bytes accepted silently")
        Path(message.name).unlink()

    def test_hook_is_posix_sh_and_parses(self) -> None:
        self.assertTrue(HOOK.read_text().startswith("#!/bin/sh\n"))
        parsed = subprocess.run(["sh", "-n", str(HOOK)], capture_output=True, text=True, check=False)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)

    def test_hook_accepts_a_clean_message_and_rejects_an_authorship_marker(self) -> None:
        for text, should_pass in (
            ("docs: a clean subject\n", True),
            ("docs: subject\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n", False),
        ):
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
                handle.write(text)
                path = handle.name
            result = subprocess.run(
                [str(HOOK), path], cwd=REPO_ROOT, capture_output=True, text=True, check=False
            )
            Path(path).unlink()
            if should_pass:
                self.assertEqual(result.returncode, 0, result.stderr)
            else:
                self.assertNotEqual(result.returncode, 0, "authorship marker was accepted")


class CommitHookInterpreterTests(unittest.TestCase):
    """Both entrypoints must enforce the policy with the same interpreter ownership."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.repo = self.directory / "committing worktree"
        self.repo.mkdir()
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        git = shutil.which("git")
        self.assertIsNotNone(git, "hook regression tests require git")
        (self.bin / "git").symlink_to(git)
        self.env = {
            key: value for key, value in os.environ.items()
            if not key.startswith("GIT_")
            and key not in ("DEVENV_STATE", "VIRTUAL_ENV")
        }
        self.env.update(PATH=str(self.bin), UV_PYTHON_PREFERENCE="only-system")
        subprocess.run(
            ["git", "init", "--quiet", "--template=", str(self.repo)],
            env=self.env, check=True, capture_output=True,
        )
        (self.repo / "scripts").mkdir()
        shutil.copyfile(
            REPO_ROOT / "scripts/check_commit_message.py",
            self.repo / "scripts/check_commit_message.py",
        )

    def interpreter(self, path: Path, *, selected: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        command = (
            f'exec {shlex.quote(sys.executable)} "$@"'
            if selected else "exit 97"
        )
        path.write_text(f"#!/bin/sh\n{command}\n", encoding="utf-8")
        path.chmod(0o755)

    def assert_policy(self) -> None:
        message = self.repo / "COMMIT_EDITMSG"
        # Invoke the same external template from another repository, as the
        # common hooks directory does for each linked worktree.
        for hook in (HOOK, REPO_ROOT / ".cargo-husky/hooks/commit-msg"):
            for text, expected in (
                ("docs: a clean subject\n", 0),
                ("docs: subject\n\nCo-Authored-By: Claude <a@b>\n", 1),
            ):
                with self.subTest(hook=hook, text=text):
                    message.write_text(text, encoding="utf-8")
                    result = subprocess.run(
                        [str(hook), str(message)], cwd=self.repo, env=self.env,
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, expected, result.stderr)
                    if expected:
                        self.assertIn("prohibited AI authorship marker", result.stderr)

    def test_activated_profile_precedes_default_and_manual_environments(self) -> None:
        state = self.repo / ".devenv/profiles/ci/state"
        self.interpreter(state / "venv/bin/python")
        self.interpreter(self.repo / ".devenv/state/venv/bin/python", selected=False)
        self.interpreter(self.repo / ".venv/bin/python", selected=False)
        self.env["DEVENV_STATE"] = str(state)
        self.assert_policy()

    def test_default_devenv_environment_remains_supported(self) -> None:
        self.interpreter(self.repo / ".devenv/state/venv/bin/python")
        self.interpreter(self.repo / ".venv/bin/python", selected=False)
        self.assert_policy()

    def test_manual_environment_remains_supported(self) -> None:
        self.interpreter(self.repo / ".venv/bin/python")
        self.assert_policy()

    def test_foreign_profile_cannot_replace_the_manual_environment(self) -> None:
        state = self.directory / "other worktree/.devenv/profiles/ci/state"
        self.interpreter(state / "venv/bin/python", selected=False)
        self.interpreter(self.repo / ".venv/bin/python")
        self.env.update(DEVENV_STATE=str(state), VIRTUAL_ENV=str(state / "venv"))
        self.assert_policy()

    def test_profile_path_cannot_escape_the_worktree(self) -> None:
        state = self.directory / "foreign-state"
        self.interpreter(state / "venv/bin/python", selected=False)
        self.interpreter(self.repo / ".venv/bin/python")
        (self.repo / ".devenv").mkdir()
        link = self.repo / ".devenv/foreign"
        link.symlink_to(state, target_is_directory=True)
        for path in (link, self.repo / ".devenv/../../foreign-state"):
            with self.subTest(state=path):
                self.env["DEVENV_STATE"] = str(path)
                self.assert_policy()

    def test_uv_fallback_overrides_inherited_python_preference(self) -> None:
        # Model uv's conflicting-option rejection without downloading Python.
        uv = self.bin / "uv"
        uv.write_text(
            "#!/bin/sh\n"
            'if [ "${UV_PYTHON_PREFERENCE+x}" = x ]; then\n'
            "    printf '%s\\n' 'conflicting Python preference' >&2\n"
            "    exit 2\n"
            "fi\n"
            'if [ "$1" != run ] || [ "$2" != --managed-python ] ||\n'
            '   [ "$3" != --python ] || [ "$4" != 3.11 ] ||\n'
            '   [ "$5" != --no-project ] || [ "$6" != python ]; then\n'
            "    exit 2\n"
            "fi\n"
            "shift 6\n"
            f'exec {shlex.quote(sys.executable)} "$@"\n',
            encoding="utf-8",
        )
        uv.chmod(0o755)
        # A same-prefix sibling is still foreign and must reach uv, not run
        # the unrelated profile's interpreter.
        state = self.directory / "committing worktree-other/.devenv/profiles/ci/state"
        self.interpreter(state / "venv/bin/python", selected=False)
        self.env["DEVENV_STATE"] = str(state)
        self.assert_policy()


if __name__ == "__main__":
    unittest.main()
