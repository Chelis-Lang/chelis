"""Lock the regeneration hooks: `.githooks/pre-commit` and `.githooks/pre-push`.

Each behavioral test builds a temporary repository holding the conformance
asset generator and its inputs, installs both templates into the clone's
common hooks directory the way Devenv does, and drives them through real
`git commit`, sequencer, and `git push` invocations. Neither hook may write a
file, so several tests also assert that the worktree and index are untouched. The interpreter each hook selects is
a shim in the committing worktree, as in `test_commit_hook.py`, and the
environment carries no cargo target, so no test here can reach cargo.
"""

from __future__ import annotations

import io
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import regen_all  # noqa: E402
import regen_hooks  # noqa: E402

HOOKS = ("pre-commit", "pre-push")
ASSET = "crates/chelis-conformance/assets/canonical/agents-inheritance.md"
# The files a temporary repository needs for the conformance leg to run.
FIXTURE_PATHS = (
    "scripts/regen_hooks.py",
    "scripts/regen_all.py",
    "scripts/regenerate_conformance_assets.py",
    "AGENTS.md",
    "docs/CHELIS_SURFACE.md",
    "packages/chelis-std/SKILL.md",
    "agent-skills",
    "crates/chelis-conformance/assets",
)


class HookTemplateContract(unittest.TestCase):
    def test_hooks_are_tracked_executable_posix_sh(self) -> None:
        for hook in HOOKS:
            for path in (f".githooks/{hook}", f".cargo-husky/hooks/{hook}"):
                with self.subTest(path=path):
                    tracked = subprocess.run(
                        ["git", "ls-files", "--error-unmatch", path],
                        cwd=REPO_ROOT, capture_output=True, check=False,
                    )
                    self.assertEqual(tracked.returncode, 0, f"{path} must be tracked")
                    self.assertTrue((REPO_ROOT / path).stat().st_mode & 0o111)
                    self.assertTrue((REPO_ROOT / path).read_text().startswith("#!/bin/sh\n{\n"))
                    parsed = subprocess.run(
                        ["sh", "-n", str(REPO_ROOT / path)],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(parsed.returncode, 0, parsed.stderr)

    def test_cargo_husky_twins_match_the_tested_templates(self) -> None:
        """cargo-husky installs its own copy, so it must be the tested bytes."""
        for hook in HOOKS:
            with self.subTest(hook=hook):
                self.assertEqual(
                    (REPO_ROOT / ".cargo-husky/hooks" / hook).read_bytes(),
                    (REPO_ROOT / ".githooks" / hook).read_bytes(),
                )

    def test_hooks_fail_closed_when_truncated(self) -> None:
        """A truncated hook must not be a valid program that does nothing.

        Run outside any repository, the full hook reports git's failure, so a
        prefix that exits 0 with no output is a silent accept. Only fragments
        of the shebang line can still parse, and those cannot be made to fail.
        """
        with tempfile.TemporaryDirectory() as scratch:
            environment = dict(os.environ, GIT_CEILING_DIRECTORIES=scratch)
            for hook in HOOKS:
                raw = (REPO_ROOT / ".githooks" / hook).read_bytes()
                for cut in range(12, len(raw) + 1, 61):
                    fragment = Path(scratch) / f"{hook}-{cut}"
                    fragment.write_bytes(raw[:cut])
                    result = subprocess.run(
                        ["/bin/sh", str(fragment)], cwd=scratch, env=environment,
                        stdin=subprocess.DEVNULL, capture_output=True, check=False,
                    )
                    silent = result.returncode == 0 and not result.stdout and not result.stderr
                    self.assertFalse(silent, f"{hook}: prefix of {cut} bytes accepted silently")


def commit_leg_table() -> str:
    """The `legs='...'` table of the pre-commit template."""
    text = (REPO_ROOT / ".githooks/pre-commit").read_text(encoding="utf-8")
    start = text.index("\nlegs='") + len("\nlegs='")
    return text[start:text.index("'\n", start)]


class LegInputDeclarations(unittest.TestCase):
    def test_commit_leg_rows_match_the_regen_all_manifest(self) -> None:
        """The pre-commit table and regen_all.py must not drift apart."""
        manifest = {leg.name: leg for leg in regen_all.regen_legs("PY")}
        rows = regen_hooks.parse_commit_legs(commit_leg_table())
        self.assertEqual(
            [row.name for row in rows],
            ["conformance-assets", "reviewed-unsupported-wording", "opaque-corpus"],
        )
        for row in rows:
            with self.subTest(leg=row.name):
                leg = manifest[row.name]
                self.assertEqual(row.inputs, leg.inputs)
                self.assertEqual(row.outputs, leg.writes)
                self.assertEqual(("PY", *row.fix), leg.write_argv)
                self.assertTrue((REPO_ROOT / row.fix[0]).is_file())

    def test_every_hook_leg_declares_existing_inputs(self) -> None:
        """A misspelled input would make a hook skip its leg without a word."""
        for leg in regen_hooks.pre_push_legs("python"):
            with self.subTest(leg=leg.name):
                self.assertTrue(leg.inputs, "a leg the hooks run must declare inputs")
                for spec in leg.inputs:
                    path = REPO_ROOT / spec
                    self.assertTrue(
                        path.is_dir() if spec.endswith("/") else os.path.lexists(path),
                        f"{leg.name} declares a missing input {spec}",
                    )


class PrePushReadOnly(unittest.TestCase):
    """A hook may run only checks that never write into the worktree."""

    def test_pre_push_selects_only_read_only_tier_zero_legs(self) -> None:
        legs = regen_hooks.pre_push_legs("PY")
        self.assertEqual(
            [leg.name for leg in legs],
            ["conformance-assets", "reviewed-unsupported-wording", "opaque-corpus"],
        )
        self.assertTrue(all(leg.tier == 0 and leg.needs == "python" for leg in legs))

    def test_python_pre_push_checks_leave_the_tree_byte_identical(self) -> None:
        for leg in regen_hooks.pre_push_legs(sys.executable):
            if leg.needs != "python":
                continue
            with self.subTest(leg=leg.name), tempfile.TemporaryDirectory() as scratch:
                copy = Path(scratch)
                for spec in (*leg.inputs, *leg.writes):
                    source, target = REPO_ROOT / spec, copy / spec
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source.is_symlink():
                        target.symlink_to(os.readlink(source))
                    elif source.is_dir():
                        shutil.copytree(source, target, symlinks=True)
                    else:
                        shutil.copy2(source, target)
                before = regen_hooks.tree_state(copy)
                stamps = {p: p.lstat().st_mtime_ns for p in copy.rglob("*")}
                result = regen_all.run_leg(
                    leg, check=True, repo_root=copy, python=sys.executable,
                    runner=regen_hooks.quiet_runner,
                    environ=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                    out=io.StringIO(), position="1/1",
                )
                self.assertEqual(result.status, "ok", result.detail)
                self.assertEqual(regen_hooks.tree_state(copy), before)
                self.assertEqual(
                    {p: p.lstat().st_mtime_ns for p in copy.rglob("*")}, stamps,
                    "a check rewrote a file, even if with the same bytes",
                )


class StdlibOnlyLegs(unittest.TestCase):
    def test_every_hook_script_imports_without_site_packages(self) -> None:
        """`uv run --no-project` supplies no packages, so each script the
        hooks run, and everything it imports, must be stdlib or in-tree."""
        scripts = {Path("scripts/regen_hooks.py")}
        for row in regen_hooks.parse_commit_legs(commit_leg_table()):
            scripts.add(Path(row.fix[0]))
            if row.check is not None:
                scripts.add(Path(row.check[0]))
        for leg in regen_hooks.pre_push_legs("PY"):
            for argv in (leg.check_argv, leg.write_argv):
                if argv is not None:
                    scripts.add(Path(argv[1]))
        for script in sorted(scripts):
            with self.subTest(script=str(script)):
                result = subprocess.run(
                    [sys.executable, "-I", "-S", "-c",
                     "import sys; sys.path[:0] = sys.argv[1:3]; "
                     "__import__(sys.argv[3])",
                     str(REPO_ROOT / script.parent), str(REPO_ROOT / "scripts"),
                     script.stem],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class RegenHookRepository(unittest.TestCase):
    """Real commits and pushes through the installed templates."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.repo = self.directory / "main worktree"
        self.repo.mkdir()
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        git = shutil.which("git")
        self.assertIsNotNone(git, "hook regression tests require git")
        (self.bin / "git").symlink_to(git)
        # A system interpreter on PATH must never run: hooks use only the
        # worktree's uv-managed Python or uv itself.
        self.decoy_log = self.directory / "decoy-ran"
        for name in ("python3", "python"):
            decoy = self.bin / name
            decoy.write_text(
                f"#!/bin/sh\necho DECOY {name} >> {shlex.quote(str(self.decoy_log))}\n"
                "echo 'decoy interpreter ran' >&2\nexit 99\n",
                encoding="utf-8",
            )
            decoy.chmod(0o755)
        self.addCleanup(self.assert_no_decoy_or_leftover)
        # Any cargo invocation is logged and fails; only the warm-target test
        # expects one, and then only the read-only lock probe.
        self.cargo_log = self.directory / "cargo-ran"
        cargo = self.bin / "cargo"
        cargo.write_text(
            f'#!/bin/sh\necho "$*" >> {shlex.quote(str(self.cargo_log))}\nexit 101\n',
            encoding="utf-8",
        )
        cargo.chmod(0o755)
        self.tmp = self.directory / "tmp"
        self.tmp.mkdir()
        self.env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("GIT_", "CARGO", "CHELIS_"))
            and key not in ("DEVENV_STATE", "VIRTUAL_ENV")
        }
        self.env.update(
            PATH=str(self.bin),
            UV_PYTHON_PREFERENCE="only-system",
            GIT_AUTHOR_NAME="Hook Test",
            GIT_AUTHOR_EMAIL="hook@example.invalid",
            GIT_COMMITTER_NAME="Hook Test",
            GIT_COMMITTER_EMAIL="hook@example.invalid",
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            TMPDIR=str(self.tmp),
        )
        self.git("init", "--quiet", "--template=", "--initial-branch=main", str(self.repo), cwd=self.directory)
        self.quiesce_maintenance(self.repo)
        for relative in FIXTURE_PATHS:
            source = REPO_ROOT / relative
            target = self.repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copyfile(source, target)
        for surface in (".claude/skills", ".codex/skills"):
            (self.repo / surface).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / surface).symlink_to("../agent-skills", target_is_directory=True)
        (self.repo / "README.md").write_text("fixture\n", encoding="utf-8")
        (self.repo / ".gitignore").write_text(".venv/\n__pycache__/\n", encoding="utf-8")
        self.interpreter(self.repo / ".venv/bin/python")
        self.git("add", "--all")
        self.git("commit", "--quiet", "--no-verify", "-m", "fixture")
        self.install_hooks()
        self.remote = self.directory / "remote.git"
        self.git("init", "--quiet", "--bare", "--template=", str(self.remote), cwd=self.directory)
        self.quiesce_maintenance(self.remote)
        self.git("remote", "add", "origin", str(self.remote))
        self.first_push = self.run_git("push", "--quiet", "origin", "main")
        self.assertEqual(self.first_push.returncode, 0, self.first_push.stderr)

    # --- helpers ---------------------------------------------------------

    def assert_no_decoy_or_leftover(self) -> None:
        self.assertFalse(self.decoy_log.exists(), "a PATH python ran")
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), [],
                         "a temporary copy was left behind")

    def run_git(self, *args: str, cwd: Path | None = None, env=None):
        return subprocess.run(
            ["git", *args], cwd=cwd or self.repo, env=env or self.env,
            capture_output=True, text=True, check=False,
        )

    def git(self, *args: str, cwd: Path | None = None) -> str:
        result = self.run_git(*args, cwd=cwd)
        self.assertEqual(result.returncode, 0, f"git {args}: {result.stderr}")
        return result.stdout

    def interpreter(self, path: Path, *, selected: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        command = f'exec {shlex.quote(sys.executable)} "$@"' if selected else "exit 97"
        path.write_text(f"#!/bin/sh\n{command}\n", encoding="utf-8")
        path.chmod(0o755)

    def install_hooks(self) -> None:
        common = Path(self.git("rev-parse", "--path-format=absolute", "--git-common-dir").strip())
        (common / "hooks").mkdir(exist_ok=True)
        for hook in HOOKS:
            installed = common / "hooks" / hook
            shutil.copyfile(REPO_ROOT / ".githooks" / hook, installed)
            installed.chmod(0o755)

    def edit_agents(self, root: Path | None = None, text: str = "edited\n") -> None:
        agents = (root or self.repo) / "AGENTS.md"
        agents.write_text(agents.read_text(encoding="utf-8") + text, encoding="utf-8")

    def committed(self, path: str, root: Path | None = None) -> str:
        return self.git("show", f"HEAD:{path}", cwd=root)

    def push(self, *extra: str, env=None):
        return self.run_git("push", "origin", *extra, "main", env=env)

    def remote_head(self) -> str:
        return self.git("rev-parse", "main", cwd=self.remote).strip()

    def regenerate(self, root: Path | None = None) -> None:
        subprocess.run(
            [sys.executable, "scripts/regenerate_conformance_assets.py"],
            cwd=root or self.repo, env=self.env, capture_output=True, check=True,
        )

    def index_and_tree(self, root: Path | None = None) -> tuple[str, bytes]:
        index = self.git("ls-files", "--stage", cwd=root)
        tree = b"".join(
            path.read_bytes() for path in sorted((root or self.repo).rglob("*"))
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
        )
        return index, tree

    # --- pre-commit ------------------------------------------------------

    def test_pre_commit_accepts_consistent_staged_content(self) -> None:
        self.edit_agents()
        self.regenerate()
        self.git("add", "AGENTS.md", ASSET)
        result = self.run_git("commit", "-m", "docs: edit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(self.committed(ASSET), self.committed("AGENTS.md"))

    def test_pre_commit_rejects_an_inconsistent_pair_and_writes_nothing(self) -> None:
        head = self.git("rev-parse", "HEAD")
        self.edit_agents()
        self.git("add", "AGENTS.md")
        before = self.index_and_tree()
        result = self.run_git("commit", "-m", "docs: edit")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(f"pre-commit: AGENTS.md and {ASSET} are inconsistent", result.stderr)
        self.assertIn("scripts/regenerate_conformance_assets.py\n", result.stderr)
        self.assertIn(f"then: git add -- {ASSET}", result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD"), head)
        self.assertEqual(self.index_and_tree(), before, "pre-commit must not write")

    def test_pre_commit_judges_the_index_not_the_worktree(self) -> None:
        """Partial staging: the staged pair decides, whatever the worktree holds."""
        self.edit_agents(text="first\n")
        self.regenerate()
        self.git("add", "AGENTS.md", ASSET)
        self.edit_agents(text="second, unstaged\n")  # worktree pair now stale
        result = self.run_git("commit", "-m", "docs: first")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.committed(ASSET), self.committed("AGENTS.md"))
        # The inverse: a consistent worktree does not excuse a stale index.
        self.git("add", "AGENTS.md")
        self.regenerate()
        result = self.run_git("commit", "-m", "docs: second")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("are inconsistent", result.stderr)

    def test_pre_commit_never_touches_a_hand_edited_derived_file(self) -> None:
        self.edit_agents()
        self.regenerate()
        self.git("add", "AGENTS.md", ASSET)
        (self.repo / ASSET).write_text("hand edit in progress\n", encoding="utf-8")
        result = self.run_git("commit", "-m", "docs: edit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.repo / ASSET).read_text(encoding="utf-8"), "hand edit in progress\n")
        self.assertEqual(self.git("status", "--porcelain"), f" M {ASSET}\n")

    def test_pre_commit_judges_pathspec_and_all_commits_by_what_they_commit(self) -> None:
        self.edit_agents()
        self.regenerate()
        # `git commit <paths>` commits through a temporary index.
        only_source = self.run_git("commit", "-m", "docs: edit", "AGENTS.md")
        self.assertEqual(only_source.returncode, 1, only_source.stderr)
        both = self.run_git("commit", "-m", "docs: edit", "AGENTS.md", ASSET)
        self.assertEqual(both.returncode, 0, both.stderr)
        self.assertEqual(self.committed(ASSET), self.committed("AGENTS.md"))
        self.assertEqual(self.git("status", "--porcelain"), "")
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "")
        # `git commit -a` stages tracked edits into the index it hands the hook.
        self.edit_agents(text="again\n")
        stale = self.run_git("commit", "-a", "-m", "docs: again")
        self.assertEqual(stale.returncode, 1, stale.stderr)
        self.regenerate()
        fresh = self.run_git("commit", "-a", "-m", "docs: again")
        self.assertEqual(fresh.returncode, 0, fresh.stderr)
        self.assertEqual(self.committed(ASSET), self.committed("AGENTS.md"))

    def test_pre_commit_does_nothing_when_no_leg_path_is_staged(self) -> None:
        (self.repo / ASSET).write_text("unstaged hand edit\n", encoding="utf-8")
        (self.repo / "README.md").write_text("unrelated\n", encoding="utf-8")
        self.git("add", "README.md")
        result = self.run_git("commit", "-m", "docs: readme")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_pre_commit_checks_a_staged_derived_file_on_its_own(self) -> None:
        """An edited, renamed or deleted output selects its leg as well."""
        moved = ASSET.replace("agents-inheritance.md", "renamed.md")
        cases = (
            ("edit", lambda: (self.repo / ASSET).write_text("hand edit\n", encoding="utf-8"),
             ("commit", "-am", "chore: edit asset")),
            ("rename", lambda: self.git("mv", ASSET, moved),
             ("commit", "-m", "chore: move asset")),
            ("delete", lambda: self.git("rm", "--quiet", ASSET),
             ("commit", "-m", "chore: drop asset")),
        )
        for name, change, command in cases:
            with self.subTest(change=name):
                change()
                result = self.run_git(*command)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("are inconsistent", result.stderr)
                self.assertIn(ASSET, result.stderr)
                self.git("reset", "--quiet", "--hard")

    def test_amend_is_checked(self) -> None:
        self.edit_agents()
        self.git("add", "AGENTS.md")
        result = self.run_git("commit", "--amend", "-m", "docs: amended")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("are inconsistent", result.stderr)

    def conflicting_branches(self) -> None:
        """`side` and `main` both edit AGENTS.md (each with its asset)."""
        self.git("checkout", "--quiet", "-b", "side")
        self.edit_agents(text="side\n")
        self.regenerate()
        self.git("commit", "--quiet", "-am", "docs: side")
        self.git("checkout", "--quiet", "main")
        self.edit_agents(text="main\n")
        self.regenerate()
        self.git("commit", "--quiet", "-am", "docs: main")

    def resolve_agents_only(self) -> None:
        agents = self.git("show", "main:AGENTS.md") + "resolved\n"
        (self.repo / "AGENTS.md").write_text(agents, encoding="utf-8")
        (self.repo / ASSET).write_text(self.git("show", f"main:{ASSET}"), encoding="utf-8")
        self.git("add", "AGENTS.md", ASSET)

    def test_conflicted_cherry_pick_stays_resumable(self) -> None:
        self.conflicting_branches()
        self.assertNotEqual(self.run_git("cherry-pick", "side").returncode, 0)
        self.resolve_agents_only()
        env = dict(self.env, GIT_EDITOR=":")
        blocked = self.run_git("cherry-pick", "--continue", env=env)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn("are inconsistent", blocked.stderr)
        self.assertTrue((self.git_dir() / "CHERRY_PICK_HEAD").exists())
        self.regenerate()
        self.git("add", ASSET)
        done = self.run_git("cherry-pick", "--continue", env=env)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.committed(ASSET), self.committed("AGENTS.md"))

    def test_conflicted_merge_commit_stays_resumable(self) -> None:
        self.conflicting_branches()
        self.assertNotEqual(self.run_git("merge", "side").returncode, 0)
        self.resolve_agents_only()
        blocked = self.run_git("commit", "--no-edit")
        self.assertEqual(blocked.returncode, 1, blocked.stderr)
        self.assertTrue((self.git_dir() / "MERGE_HEAD").exists())
        self.regenerate()
        self.git("add", ASSET)
        done = self.run_git("commit", "--no-edit")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(len(self.git("log", "-1", "--format=%P").split()), 2)

    def test_rebase_continue_runs_no_pre_commit(self) -> None:
        """git's sequencer commits a resolved rebase step without the hook."""
        self.conflicting_branches()
        self.git("checkout", "--quiet", "side")
        self.assertNotEqual(self.run_git("rebase", "main").returncode, 0)
        self.resolve_agents_only()
        done = self.run_git("rebase", "--continue", env=dict(self.env, GIT_EDITOR=":"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("pre-commit", done.stderr)
        # The inconsistent step landed; pre-push is the net for this path.
        self.assertNotEqual(self.committed(ASSET), self.committed("AGENTS.md"))
        self.assertNotEqual(self.run_git("push", "origin", "side").returncode, 0)

    def git_dir(self) -> Path:
        return Path(self.git("rev-parse", "--path-format=absolute", "--git-dir").strip())

    def test_branch_without_the_script_is_left_alone(self) -> None:
        self.git("rm", "--quiet", "scripts/regen_hooks.py")
        self.git("commit", "--quiet", "--no-verify", "-m", "chore: predate hooks")
        self.edit_agents()
        self.git("add", "AGENTS.md")
        result = self.run_git("commit", "-m", "docs: edit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        pushed = self.push()
        self.assertEqual(pushed.returncode, 0, pushed.stderr)

    def test_no_managed_python_passes_an_irrelevant_commit(self) -> None:
        shutil.rmtree(self.repo / ".venv")
        (self.repo / "README.md").write_text("unrelated\n", encoding="utf-8")
        self.git("add", "README.md")
        result = self.run_git("commit", "-m", "docs: readme")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        pushed = self.push()
        self.assertEqual(pushed.returncode, 0, pushed.stderr)
        self.assertIn("pre-push hook: no managed Python interpreter", pushed.stderr)
        self.assertIn("CI checks them", pushed.stderr)

    def test_uv_runs_the_hook_when_no_venv_exists(self) -> None:
        shutil.rmtree(self.repo / ".venv")
        uv_log = self.directory / "uv-ran"
        uv = self.bin / "uv"
        uv.write_text(
            "#!/bin/sh\n"
            'if [ "$1" != run ] || [ "$2" != --managed-python ] ||\n'
            '   [ "$3" != --python ] || [ "$4" != 3.11 ] ||\n'
            '   [ "$5" != --no-project ] || [ "$6" != --isolated ] ||\n'
            '   [ "$7" != python ]; then\n'
            "    exit 2\n"
            "fi\n"
            "shift 7\n"
            f"echo ran >> {shlex.quote(str(uv_log))}\n"
            f'exec {shlex.quote(sys.executable)} "$@"\n',
            encoding="utf-8",
        )
        uv.chmod(0o755)
        self.edit_agents()
        self.git("add", "AGENTS.md")
        stale = self.run_git("commit", "-m", "docs: edit")
        self.assertNotEqual(stale.returncode, 0)
        self.assertIn("are inconsistent", stale.stderr)
        self.assertTrue(uv_log.exists(), "uv must supply the interpreter")

    def test_uv_runs_the_hook_in_a_fresh_linked_worktree(self) -> None:
        """A new agent worktree has no .venv; uv must serve it, and the
        primary checkout's interpreter must not."""
        linked = self.directory / "fresh agent worktree"
        self.git("worktree", "add", "--quiet", "-b", "agent", str(linked))
        self.interpreter(self.repo / ".venv/bin/python", selected=False)
        uv_log = self.directory / "uv-ran"
        uv = self.bin / "uv"
        uv.write_text(
            "#!/bin/sh\n"
            'if [ "$1" != run ] || [ "$2" != --managed-python ] ||\n'
            '   [ "$3" != --python ] || [ "$4" != 3.11 ] ||\n'
            '   [ "$5" != --no-project ] || [ "$6" != --isolated ] ||\n'
            '   [ "$7" != python ]; then\n'
            "    exit 2\n"
            "fi\n"
            "shift 7\n"
            f'echo "$PWD" >> {shlex.quote(str(uv_log))}\n'
            f'exec {shlex.quote(sys.executable)} "$@"\n',
            encoding="utf-8",
        )
        uv.chmod(0o755)
        self.edit_agents(linked)
        self.git("add", "AGENTS.md", cwd=linked)
        stale = self.run_git("commit", "-m", "docs: edit", cwd=linked)
        self.assertEqual(stale.returncode, 1, stale.stderr)
        self.assertIn("are inconsistent", stale.stderr)
        self.assertIn(f"fix: {regen_hooks.UV_FALLBACK} scripts/", stale.stderr)
        self.regenerate(linked)
        self.git("add", ASSET, cwd=linked)
        fresh = self.run_git("commit", "-m", "docs: edit", cwd=linked)
        self.assertEqual(fresh.returncode, 0, fresh.stderr)
        self.assertEqual(
            uv_log.read_text(encoding="utf-8").splitlines(), [str(linked)] * 2
        )

    def test_no_managed_python_fails_closed_with_the_fix(self) -> None:
        shutil.rmtree(self.repo / ".venv")
        self.edit_agents()
        self.regenerate()
        self.git("add", "AGENTS.md", ASSET)
        result = self.run_git("commit", "-m", "docs: edit")
        # git reports any failing hook as exit 1; the hook itself exits 2.
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("pre-commit hook: no managed Python interpreter", result.stderr)
        self.assertIn("uv venv --python 3.11", result.stderr)

    def quiesce_maintenance(self, repository: Path) -> None:
        # Commit and receive-pack start a detached `git maintenance run --auto`
        # that can still be writing under the repository when the scratch tree
        # is removed. Repository config reaches receive-pack, which clears
        # GIT_CONFIG_* from its environment.
        for key, value in (("maintenance.auto", "false"), ("gc.auto", "0"), ("receive.autogc", "false")):
            self.git("-C", str(repository), "config", key, value, cwd=self.directory)

    def test_concurrent_commits_in_two_worktrees_do_not_interfere(self) -> None:
        linked = self.directory / "linked worktree"
        self.git("worktree", "add", "--quiet", "-b", "feature", str(linked))
        self.interpreter(linked / ".venv/bin/python")
        self.edit_agents(text="main side\n")
        self.regenerate()
        self.git("add", "AGENTS.md", ASSET)
        self.edit_agents(linked, text="linked side\n")
        self.git("add", "AGENTS.md", cwd=linked)  # stale on purpose
        for _ in range(5):
            commits = [
                subprocess.Popen(
                    ["git", "commit", "--quiet", "-m", "docs: race"],
                    cwd=root, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True,
                )
                for root in (self.repo, linked)
            ]
            (main_out, main_err), (linked_out, linked_err) = (c.communicate() for c in commits)
            self.assertEqual(commits[0].returncode, 0, main_err)
            self.assertEqual(commits[1].returncode, 1, linked_err)
            self.assertIn("are inconsistent", linked_err)
            self.edit_agents(text="main again\n")
            self.regenerate()
            self.git("add", "AGENTS.md", ASSET)

    def replace_generator(self, body: str) -> None:
        generator = self.repo / "scripts/regenerate_conformance_assets.py"
        generator.write_text(body, encoding="utf-8")
        self.git("add", "scripts/regenerate_conformance_assets.py")

    def test_temporary_copy_is_removed_on_pass_fail_and_error(self) -> None:
        self.edit_agents()
        self.git("add", "AGENTS.md")
        self.assertEqual(self.run_git("commit", "-m", "docs: stale").returncode, 1)
        self.assertEqual(list(self.tmp.iterdir()), [], "fail path")
        self.regenerate()
        self.git("add", ASSET)
        self.assertEqual(self.run_git("commit", "-m", "docs: fresh").returncode, 0)
        self.assertEqual(list(self.tmp.iterdir()), [], "pass path")
        self.replace_generator("raise SystemExit('generator broke')\n")
        error = self.run_git("commit", "-m", "chore: broken generator")
        self.assertNotEqual(error.returncode, 0)
        self.assertIn("could not run on the staged content", error.stderr)
        self.assertEqual(list(self.tmp.iterdir()), [], "error path")

    def test_temporary_copy_is_removed_when_interrupted(self) -> None:
        marker = self.directory / "generator-started"
        self.replace_generator(
            "import os, pathlib, time\n"
            "pathlib.Path(os.environ['GENERATOR_STARTED']).touch()\n"
            "time.sleep(60)\n"
        )
        hook = Path(self.git("rev-parse", "--path-format=absolute",
                             "--git-common-dir").strip()) / "hooks/pre-commit"
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum.name):
                marker.unlink(missing_ok=True)
                process = subprocess.Popen(
                    [str(hook)], cwd=self.repo,
                    env=dict(self.env, GENERATOR_STARTED=str(marker)),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                deadline = time.monotonic() + 30
                while not marker.exists():
                    self.assertLess(time.monotonic(), deadline, "generator never started")
                    if process.poll() is not None:
                        self.fail(f"hook exited early: {process.communicate()}")
                    time.sleep(0.05)
                self.assertTrue(list(self.tmp.iterdir()), "the copy exists mid-run")
                process.send_signal(signum)
                _, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 128 + signum, stderr)
                self.assertEqual(list(self.tmp.iterdir()), [], f"{signum.name} left a copy")

    # --- pre-push --------------------------------------------------------

    def test_pre_push_passes_fresh_artifacts(self) -> None:
        self.edit_agents()
        self.regenerate()
        self.git("add", "AGENTS.md", ASSET)
        self.git("commit", "--quiet", "-m", "docs: edit")
        result = self.push()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("conformance-assets", result.stderr)
        self.assertEqual(self.remote_head(), self.git("rev-parse", "HEAD").strip())

    def test_pre_push_rejects_stale_artifacts_with_the_write_command(self) -> None:
        before = self.remote_head()
        self.edit_agents()
        self.git("add", "AGENTS.md")
        self.git("commit", "--quiet", "--no-verify", "-m", "docs: edit")
        result = self.push()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stale derived artifacts (conformance-assets)", result.stderr)
        self.assertIn("scripts/regenerate_conformance_assets.py\n", result.stderr)
        self.assertEqual(self.remote_head(), before)
        self.assertEqual(self.git("status", "--porcelain"), "", "pre-push must not write")

    def test_pre_push_does_not_check_a_ref_other_than_head(self) -> None:
        self.edit_agents()
        self.git("add", "AGENTS.md")
        self.git("commit", "--quiet", "--no-verify", "-m", "docs: edit")
        self.git("checkout", "--quiet", "-b", "elsewhere", "HEAD~1")
        result = self.push()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not checking conformance-assets for refs/heads/main", result.stderr)

    def test_pre_push_skips_a_leg_whose_paths_are_dirty(self) -> None:
        self.edit_agents()
        self.git("add", "AGENTS.md")
        self.git("commit", "--quiet", "--no-verify", "-m", "docs: edit")
        self.edit_agents(text="in progress\n")
        result = self.push()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not checking conformance-assets: its paths have uncommitted", result.stderr)

    def test_pre_push_never_invokes_cargo_even_when_every_leg_is_selected(self) -> None:
        """A URL remote has no tracking refs, so every path counts."""
        source = self.repo / "crates/demo/src/lib.rs"
        source.parent.mkdir(parents=True)
        source.write_text('fn f() { unimplemented_rejection!("x", 1); }\n', encoding="utf-8")
        (self.repo / "spec").mkdir()
        (self.repo / "spec/05-risc-primitives.md").write_text("# spec\n", encoding="utf-8")
        self.git("add", str(source), "spec")
        self.git("commit", "--quiet", "-m", "feat: cite")
        result = self.run_git("push", self.remote.as_uri(), "main:refs/heads/url-push")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("conformance-assets", result.stderr)
        self.assertNotIn("rejection-registry", result.stderr)
        self.assertFalse(self.cargo_log.exists(), "a hook invoked cargo")
        self.assertNotIn("rejection-registry", self.first_push.stderr)

    def test_pre_push_checks_a_pushed_annotated_tag(self) -> None:
        self.edit_agents()
        self.git("add", "AGENTS.md")
        self.git("commit", "--quiet", "--no-verify", "-m", "docs: edit")
        self.git("tag", "-a", "-m", "release", "v-stale")
        result = self.run_git("push", "origin", "v-stale")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("stale derived artifacts (conformance-assets)", result.stderr)

    def test_pre_push_leaves_the_index_file_alone(self) -> None:
        self.edit_agents()
        self.regenerate()
        self.git("add", "AGENTS.md", ASSET)
        self.git("commit", "--quiet", "-m", "docs: edit")
        # A newer mtime with the same bytes makes a status refresh rewrite
        # the index unless optional locks are off.
        stamp = time.time() + 5
        os.utime(self.repo / ASSET, (stamp, stamp))
        index = self.git_dir() / "index"
        before = (index.read_bytes(), index.stat().st_mtime_ns)
        result = self.push()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("conformance-assets", result.stderr)
        self.assertEqual((index.read_bytes(), index.stat().st_mtime_ns), before)

    def test_pre_commit_reads_skip_worktree_paths_in_a_sparse_checkout(self) -> None:
        sparse = self.directory / "sparse worktree"
        self.git("worktree", "add", "--quiet", "--no-checkout", "-b", "sparse", str(sparse))
        self.git("sparse-checkout", "set", "--cone", "scripts", "agent-skills", cwd=sparse)
        self.git("checkout", "--quiet", "sparse", cwd=sparse)
        self.assertFalse((sparse / ASSET).exists(), "the asset must be outside the cone")
        self.interpreter(sparse / ".venv/bin/python")
        self.edit_agents(sparse)
        self.git("add", "AGENTS.md", cwd=sparse)
        result = self.run_git("commit", "-m", "docs: edit", cwd=sparse)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(f"AGENTS.md and {ASSET} are inconsistent", result.stderr)

    # --- worktree safety -------------------------------------------------

    def test_hooks_use_the_committing_worktree_and_its_interpreter(self) -> None:
        linked = self.directory / "linked worktree"
        self.git("worktree", "add", "--quiet", "-b", "feature", str(linked))
        self.interpreter(linked / ".venv/bin/python")
        # The primary worktree's interpreter, and a profile activated there,
        # must never run for a commit made in the linked worktree.
        self.interpreter(self.repo / ".venv/bin/python", selected=False)
        foreign_state = self.repo / ".devenv/profiles/ci/state"
        self.interpreter(foreign_state / "venv/bin/python", selected=False)
        env = dict(self.env, DEVENV_STATE=str(foreign_state))
        self.edit_agents(linked)
        self.git("add", "AGENTS.md", cwd=linked)
        stale = self.run_git("commit", "-m", "docs: edit", cwd=linked, env=env)
        self.assertEqual(stale.returncode, 1, stale.stderr)
        self.assertIn("are inconsistent", stale.stderr)
        self.regenerate(linked)
        self.git("add", ASSET, cwd=linked)
        result = self.run_git("commit", "-m", "docs: edit", cwd=linked, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        pushed = self.run_git("push", "origin", "feature", cwd=linked, env=env)
        self.assertEqual(pushed.returncode, 0, pushed.stderr)
        self.assertIn("conformance-assets", pushed.stderr)


if __name__ == "__main__":
    unittest.main()
