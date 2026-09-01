"""Lock the properties that make `.githooks/commit-msg` worktree-safe.

chelis#1409: the previous hook lived in `.git/hooks`, which every worktree of a
clone shares, and recorded an absolute `--config` path to whichever worktree
installed it. Commits from every other worktree failed, and all of them failed
once the named worktree was deleted. These tests pin the three properties that
prevent a recurrence.
"""

from __future__ import annotations

import re
import subprocess
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

    def test_hook_names_no_absolute_path(self) -> None:
        """The defect was an absolute path to one worktree. Forbid the shape."""
        for line in HOOK.read_text().splitlines():
            code = line.split("#", 1)[0]
            self.assertNotRegex(
                code, r"(?<![\w.])/(Users|home|nix/store|tmp|var|opt)/",
                msg=f"absolute path in hook code: {line!r}",
            )

    def test_hook_resolves_the_repo_from_the_committing_worktree(self) -> None:
        body = HOOK.read_text()
        self.assertIn("git rev-parse --show-toplevel", body)
        self.assertIn("scripts/check_commit_message.py", body)

    def test_hook_fails_closed_when_truncated(self) -> None:
        """A truncated hook must not be a valid program that does nothing.

        git runs a zero-length or partial executable hook and accepts the
        commit when it exits 0, with no diagnostic. Opening a brace group on
        line 2 makes every later prefix an unterminated compound command, so
        sh rejects it loudly instead. Only fragments of the shebang line
        itself can still parse, and those cannot be made to fail.
        """
        lines = HOOK.read_text().splitlines()
        self.assertEqual(lines[0], "#!/bin/sh")
        self.assertEqual(lines[1], "{", "the brace group must open on line 2")
        self.assertEqual(lines[-1], "}", "the brace group must close on the last line")

        import tempfile

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

    def test_devenv_installs_this_hook_instead_of_its_own(self) -> None:
        """scripts/test_devenv_version.py pins the task shape; this pins the
        coupling, so the template and its installer cannot drift apart."""
        nix = (REPO_ROOT / "devenv" / "git-hooks.nix").read_text()
        self.assertIn(".githooks/commit-msg", nix)
        self.assertNotIn("enable = true;", nix)

    def test_hook_keeps_every_documented_interpreter_fallback(self) -> None:
        """Stripping one silently narrows where the check runs."""
        body = HOOK.read_text()
        for fragment in (
            ".devenv/state/venv/bin/python",
            ".venv/bin/python",
            "uv run --managed-python --python 3.11 --no-project python",
        ):
            self.assertIn(fragment, body)

    def test_hook_accepts_a_clean_message_and_rejects_an_authorship_marker(self) -> None:
        import tempfile

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


if __name__ == "__main__":
    unittest.main()
