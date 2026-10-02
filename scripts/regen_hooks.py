#!/usr/bin/env python3
"""Keep committed derived artifacts current at commit and push time.

The tracked `.githooks/pre-commit` and `.githooks/pre-push` templates (and
their `.cargo-husky/hooks/` twins) run this script with the committing
worktree's managed Python:

    .venv/bin/python scripts/regen_hooks.py pre-commit
    .venv/bin/python scripts/regen_hooks.py pre-push <remote> <url> < refs

Both hooks select legs from the `scripts/regen_all.py` manifest by each leg's
declared `inputs`, so that manifest stays the one place that says what
regenerates what.

pre-commit
    Check only; it writes nothing in the worktree or the index. When the
    staged change touches the inputs of a tier-0 leg that needs only Python,
    it extracts the staged version of that leg's inputs and outputs into a
    private temporary directory, runs the generator there, and fails, exit 1,
    when that changes any output: the commit would carry a derived file that
    disagrees with its source. The message names the paths, the write command
    and the `git add` that follows it. It judges the index git hands it, so
    partial staging, `git commit <paths>` and `git commit -a` are judged by
    what they commit. With no relevant input staged it prints nothing. It
    never runs cargo.

pre-push
    Check only; it writes nothing that survives it. For each pushed ref it
    collects the paths the push changes (against the remote's old value when
    this clone has it, otherwise against the merge base with the remote's
    default branch, otherwise every path), then runs the `--check` form of
    every tier-0 or tier-1 leg whose inputs or outputs those paths touch. The
    checks read the working tree, so a leg is checked only when the pushed
    commit is this worktree's HEAD and the leg's paths are clean; otherwise
    the hook prints one line saying the check was skipped and why. A leg that
    needs cargo is skipped the same way when the cargo target is cold, which
    is when the binary that leg builds is absent. A stale leg fails the push
    and the hook prints the exact write command. CI stays the authority.

The only bypass is git's own `--no-verify`. Exit codes: 0 pass or nothing to
do, 1 inconsistent or stale, 2 a git or generator command could not run.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import regen_all

REPO_ROOT = Path(__file__).resolve().parents[1]

EXIT_PASS = 0
EXIT_REFUSED = 1
EXIT_ERROR = 2

# A cargo leg is warm when the binary it builds already exists under the
# cargo target directory; otherwise its first check is a full build.
WARM_BINARIES: dict[str, str] = {
    "rejection-registry": "debug/rejection_source_inventory",
    "std-bundle": "debug/chelis",
}


# The rejection registry's check costs about 27 s even on a warm target, and
# its `crates/` input covers nearly every push. Only a Rust diff line naming
# the citation macro can change the issue manifest, so under `crates/` a path
# selects that leg only when its pushed diff adds or removes such a line. A
# module-graph edit that drops a citing file without touching it is left to
# CI. The atom half of the registry (`spec/`) is not narrowed.
CONTENT_FILTERS: dict[str, tuple[str, str]] = {
    "rejection-registry": ("crates/", "unimplemented_rejection!"),
}


class GitError(RuntimeError):
    """A git command the hook depends on failed."""


def git(repo: Path, *args: str, environ: dict[str, str]) -> str:
    completed = subprocess.run(
        ["git", "--literal-pathspecs", *args],
        cwd=repo,
        env=environ,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} exited {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def git_paths(repo: Path, command: str, *args: str, environ: dict[str, str]) -> list[str]:
    """The paths a `git <command> -z ...` listing prints."""
    listing = git(repo, command, "-z", *args, environ=environ)
    return [path for path in listing.split("\0") if path]


def matches(path: str, specs: tuple[str, ...]) -> bool:
    """Whether `path` is one of `specs` or lies under a `dir/` entry."""
    return any(
        path.startswith(spec) if spec.endswith("/") else path == spec
        for spec in specs
    )


def touched(
    legs: list[regen_all.RegenLeg], paths: list[str], *, outputs: bool
) -> list[regen_all.RegenLeg]:
    """The legs whose inputs (and, with `outputs`, declared writes) `paths` touch."""
    selected = []
    for leg in legs:
        specs = leg.inputs + (leg.writes if outputs else ())
        if any(matches(path, specs) for path in paths):
            selected.append(leg)
    return selected


def quiet_runner(argv, *, cwd, env, check):
    """Run a leg command with its output held back unless it fails."""
    completed = subprocess.run(
        argv, cwd=cwd, env=env, check=check, capture_output=True, text=True
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


# --- pre-commit --------------------------------------------------------------


def snapshot_index(
    repo: Path, specs: list[str], destination: Path, environ: dict[str, str]
) -> None:
    """Write the index's version of every path under `specs` below `destination`.

    The index is whichever one git handed the hook through GIT_INDEX_FILE, so
    `git commit <paths>` and `git commit -a` are judged by what they commit.
    """
    paths = git_paths(repo, "ls-files", "--cached", "--", *specs, environ=environ)
    if not paths:
        return
    completed = subprocess.run(
        ["git", "checkout-index", "-z", "--stdin", f"--prefix={destination}/"],
        cwd=repo,
        env=environ,
        input="\0".join(paths) + "\0",
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise GitError(f"git checkout-index exited {completed.returncode}: "
                       f"{completed.stderr.strip()}")


def tree_state(root: Path) -> dict[str, bytes]:
    """Every file below `root` by relative path; a symlink by its target."""
    state: dict[str, bytes] = {}
    for directory, subdirectories, files in os.walk(root):
        for name in files + subdirectories:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                state[relative] = b"symlink:" + os.readlink(path).encode()
            elif path.is_file():
                state[relative] = path.read_bytes()
    return state


def pre_commit(repo: Path, python: str, environ: dict[str, str], out) -> int:
    """Fail when a staged source and its staged derived output disagree.

    Writes nothing in the worktree or the index. Each selected generator runs
    in write mode inside a private snapshot of the staged inputs and outputs;
    any output it changes there is one the commit carries stale.
    """
    legs = [
        leg
        for leg in regen_all.regen_legs(python)
        if leg.tier == 0 and leg.needs == "python"
    ]
    staged = git_paths(
        repo, "diff", "--cached", "--name-only", "--no-renames", environ=environ
    )
    selected = touched(legs, staged, outputs=False)
    if not selected:
        return EXIT_PASS

    inconsistent = []
    with tempfile.TemporaryDirectory(prefix="chelis-pre-commit-") as scratch:
        for leg in selected:
            assert leg.write_argv is not None
            snapshot = Path(scratch) / leg.name
            snapshot.mkdir()
            snapshot_index(repo, [*leg.inputs, *leg.writes], snapshot, environ)
            before = tree_state(snapshot)
            completed = regen_all.launch(
                quiet_runner, list(leg.write_argv), cwd=snapshot,
                env=dict(environ, PYTHONDONTWRITEBYTECODE="1"),
            )
            if completed.returncode != 0:
                reason = getattr(completed, "launch_error", None) or (
                    f"exit {completed.returncode}"
                )
                print(f"pre-commit: {leg.name} could not run on the staged "
                      f"content ({reason})", file=out)
                return EXIT_ERROR
            after = tree_state(snapshot)
            differing = sorted(
                path for path in before.keys() | after.keys()
                if before.get(path) != after.get(path)
            )
            if differing:
                sources = [path for path in staged if matches(path, leg.inputs)]
                inconsistent.append((leg, sources, differing))

    if not inconsistent:
        return EXIT_PASS
    for leg, sources, differing in inconsistent:
        print(
            f"pre-commit: {', '.join(sources)} and {', '.join(differing)} are "
            "inconsistent in the staged content.",
            file=out,
        )
        print(f"  fix: {shlex.join(leg.write_argv)}", file=out)
        print(f"  then: git add -- {shlex.join(differing)}", file=out)
    print(
        "The derived files are generated from their sources; an edit made by hand "
        "to a derived file belongs in its source instead. --no-verify bypasses "
        "this check.",
        file=out,
    )
    return EXIT_REFUSED


# --- pre-push ----------------------------------------------------------------


@dataclass(frozen=True)
class RefUpdate:
    local_ref: str
    local_sha: str
    remote_ref: str
    remote_sha: str


def parse_ref_updates(text: str) -> list[RefUpdate]:
    updates = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise GitError(f"unexpected pre-push input line: {line!r}")
        updates.append(RefUpdate(*fields))
    return updates


def is_null(sha: str) -> bool:
    return set(sha) == {"0"}


def has_commit(repo: Path, sha: str, environ: dict[str, str]) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repo, env=environ, capture_output=True, check=False,
    ).returncode == 0


def push_base(repo: Path, remote: str, update: RefUpdate, environ) -> str | None:
    """The commit the pushed tree is compared with, or None for every path."""
    if not is_null(update.remote_sha) and has_commit(repo, update.remote_sha, environ):
        return update.remote_sha
    for candidate in (f"refs/remotes/{remote}/HEAD", f"refs/remotes/{remote}/main"):
        if not has_commit(repo, candidate, environ):
            continue
        completed = subprocess.run(
            ["git", "merge-base", candidate, update.local_sha],
            cwd=repo, env=environ, capture_output=True, text=True, check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.strip()
    return None


def pushed_legs(
    repo: Path, remote: str, update: RefUpdate, legs: list, environ
) -> list[regen_all.RegenLeg]:
    """The legs whose inputs or outputs the pushed range touches."""
    base = push_base(repo, remote, update, environ)
    if base is None:
        paths = git_paths(repo, "ls-tree", "-r", "--name-only", update.local_sha,
                          environ=environ)
        return touched(legs, paths, outputs=True)
    paths = git_paths(repo, "diff", "--name-only", "--no-renames", base,
                      update.local_sha, environ=environ)
    selected = []
    for leg in legs:
        candidates = paths
        if leg.name in CONTENT_FILTERS:
            prefix, pattern = CONTENT_FILTERS[leg.name]
            citing = set(git_paths(
                repo, "diff", "--name-only", "--no-renames", "-G", pattern, base,
                update.local_sha, "--", prefix, environ=environ,
            ))
            candidates = [
                path for path in paths
                if not path.startswith(prefix) or path in citing
                or matches(path, leg.writes)
            ]
        selected.extend(touched([leg], candidates, outputs=True))
    return selected


def cargo_target(repo: Path, environ: dict[str, str]) -> Path:
    target = Path(environ.get("CARGO_TARGET_DIR", repo / "target"))
    return target if target.is_absolute() else repo / target


def check_command(leg: regen_all.RegenLeg, python: str) -> str:
    if leg.check_argv is not None:
        return shlex.join(leg.check_argv)
    # The opaque corpus has no --check of its own; regen_all supplies it.
    return shlex.join((python, "scripts/regen_all.py", "--tier", str(leg.tier), "--check"))


def pre_push(
    repo: Path,
    python: str,
    remote: str,
    ref_text: str,
    environ: dict[str, str],
    out,
) -> int:
    legs = [leg for leg in regen_all.regen_legs(python) if leg.tier in (0, 1)]
    head = git(repo, "rev-parse", "HEAD", environ=environ).strip()
    to_check: list[regen_all.RegenLeg] = []
    for update in parse_ref_updates(ref_text):
        if is_null(update.local_sha):
            continue
        selected = pushed_legs(repo, remote, update, legs, environ)
        if not selected:
            continue
        names = ", ".join(leg.name for leg in selected)
        if update.local_sha != head:
            print(
                f"pre-push: not checking {names} for {update.local_ref}: it is not "
                "this worktree's HEAD, and the checks read the working tree; CI "
                "checks it.",
                file=out,
            )
            continue
        for leg in selected:
            if leg in to_check:
                continue
            dirty = git_paths(
                repo, "status", "--porcelain", "--untracked-files=all",
                "--no-renames", "--", *leg.inputs, *leg.writes, environ=environ,
            )
            if dirty:
                print(
                    f"pre-push: not checking {leg.name}: its paths have uncommitted "
                    "changes, so the working tree is not the pushed commit; on a "
                    f"clean tree run {check_command(leg, python)}",
                    file=out,
                )
                continue
            warm = WARM_BINARIES.get(leg.name)
            if warm is not None and not (cargo_target(repo, environ) / warm).is_file():
                print(
                    f"pre-push: skipping {leg.name}: the cargo target is cold "
                    f"({cargo_target(repo, environ) / warm} is absent); check it with "
                    f"{check_command(leg, python)}",
                    file=out,
                )
                continue
            to_check.append(leg)

    if not to_check:
        return EXIT_PASS
    # Later tiers consume earlier outputs, so check in manifest order.
    to_check.sort(key=legs.index)

    child_environ = regen_all.scrub_environment(
        environ, regen_all.leg_env_keys(regen_all.regen_legs(python))
    )
    results = []
    for index, leg in enumerate(to_check, start=1):
        results.append(
            regen_all.run_leg(
                leg,
                check=True,
                repo_root=repo,
                python=python,
                runner=quiet_runner,
                environ=child_environ,
                out=out,
                position=f"{index}/{len(to_check)}",
            )
        )
    stale = [result.leg for result in results if result.status != "ok"]
    if not stale:
        return EXIT_PASS
    print(
        f"pre-push: stale derived artifacts ({', '.join(leg.name for leg in stale)}). "
        "Regenerate, commit the result, and push again:",
        file=out,
    )
    for leg in stale:
        assert leg.write_argv is not None
        print(f"  {shlex.join(leg.write_argv)}", file=out)
    return EXIT_REFUSED


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("pre-commit", "pre-push"):
        print("usage: regen_hooks.py pre-commit | pre-push <remote> <url>",
              file=sys.stderr)
        return EXIT_ERROR
    environ = dict(os.environ)
    try:
        if argv[0] == "pre-commit":
            return pre_commit(REPO_ROOT, sys.executable, environ, sys.stderr)
        remote = argv[1] if len(argv) > 1 else "origin"
        return pre_push(
            REPO_ROOT, sys.executable, remote, sys.stdin.read(), environ, sys.stderr
        )
    except GitError as error:
        print(f"{argv[0]}: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
