"""Patch series of the pinned Charon and Aeneas builds.

The flake builds Charon, charon-ml and Aeneas from their pinned upstream sources
(flake.lock) plus the series in verification/toolchain/<series>/: every SERIES file
lists its patches in application order, and each patch is applied with `patch -p1`
in the series' patch root of the upstream source. verification/toolchain.json
records, per integration repository, the upstream revision, the integration commit
the series reproduces and its pristine base commit (the upstream tree committed as
the first commit of the integration repository).

    python3 verification/toolchain/series.py check
        Proves the committed series reproduce the recorded integration commits: the
        upstream source locked in flake.lock (fetched by `nix flake prefetch`, hash
        checked against the lock) is copied, every series is applied with the pinned
        GNU patch (nixpkgs from flake.lock), and the result is compared file by file
        (bytes, executable bit, symlink targets) with the integration commit. A patch
        applied with an offset or fuzz, a leftover file, or any difference fails.
        The only files allowed beyond the commit are upstream files that the
        integration repository never tracked (absent from its base and its commit),
        which must be byte-identical to upstream.

    python3 verification/toolchain/series.py refresh [--charon-rev REV] [--aeneas-rev REV]
        Regenerates every series from the first-parent history base..REV of the
        integration repositories (default REV: main): one patch per first-parent step,
        a merge contributing `git diff M^1 M`, split across the series by path. Each
        step may only touch the paths of its repository's series. Patches that left
        the series are deleted. Updates the integration commits in toolchain.json,
        then runs `check`.

The integration repositories are toolchain.json's `integration.repository` paths
(.work/toolchain/{charon-src,aeneas-src}); override with CHARON_SRC / AENEAS_SRC.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TOOLCHAIN_JSON = ROOT / "verification" / "toolchain.json"
SERIES_ROOT = ROOT / "verification" / "toolchain"
FLAKE_LOCK = ROOT / "flake.lock"
# Integration repositories (toolchain.json `integration.repository`, relative to the
# checkout) can be overridden with these environment variables.
SOURCE_VARIABLES = {"charon": "CHARON_SRC", "aeneas": "AENEAS_SRC"}
# Lines that must not occur in a generated diff: binary content, symlinks and
# submodules are outside what the series (and `patch`) reproduce faithfully.
REJECTED_DIFF_LINE = re.compile(
    r"^(Binary files |GIT binary patch|(new|deleted) file mode 1[26]0000|(old|new) mode 1[26]0000)"
)
# `patch` names each file it patches; any other output (offsets, fuzz, reversed or
# already applied hunks, rejects, garbage) means the series did not apply exactly.
PATCH_PROGRESS = re.compile(r"^patching file ")


class SeriesError(RuntimeError):
    pass


def git(repository: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(repository), *args], capture_output=True, text=not binary,
    )
    if result.returncode != 0:
        stderr = result.stderr if isinstance(result.stderr, str) else result.stderr.decode()
        raise SeriesError(f"git {' '.join(args)} failed in {repository}: {stderr.strip()}")
    return result.stdout


def load_toolchain() -> dict:
    return json.loads(TOOLCHAIN_JSON.read_text())


def integration_source(name: str, entry: dict) -> Path:
    variable = SOURCE_VARIABLES[name]
    path = Path(os.environ.get(variable, ROOT / entry["integration"]["repository"])).resolve()
    if not (path / ".git").exists():
        raise SeriesError(f"integration repository for {name} not found: {path} (set {variable})")
    return path


def lock_node(name: str) -> dict:
    """The flake.lock node of the upstream source: `aeneas` is a direct input,
    `charon` the input of the aeneas flake."""
    lock = json.loads(FLAKE_LOCK.read_text())
    nodes = lock["nodes"]
    aeneas = nodes[lock["nodes"][lock["root"]]["inputs"]["aeneas"]]
    key = {"aeneas": nodes[lock["root"]]["inputs"]["aeneas"], "charon": aeneas["inputs"]["charon"]}[name]
    return nodes[key]["locked"]


def upstream_source(name: str, revision: str) -> Path:
    locked = lock_node(name)
    if locked["rev"] != revision:
        raise SeriesError(
            f"toolchain.json pins {name} upstream {revision}, flake.lock locks {locked['rev']}"
        )
    reference = f"github:{locked['owner']}/{locked['repo']}/{locked['rev']}"
    fetched = json.loads(subprocess.run(
        ["nix", "flake", "prefetch", "--json", reference], check=True, capture_output=True, text=True,
    ).stdout)
    if fetched["hash"] != locked["narHash"]:
        raise SeriesError(f"{reference}: fetched {fetched['hash']}, flake.lock has {locked['narHash']}")
    return Path(fetched["storePath"])


def gnu_patch() -> Path:
    """GNU patch of the flake's nixpkgs: the program the Nix builds apply the series with."""
    out = subprocess.run(
        ["nix", "build", "--no-link", "--print-out-paths", "--inputs-from", str(ROOT), "nixpkgs#gnupatch"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    return Path(out[0]) / "bin" / "patch"


def series_names(directory: Path) -> list[str]:
    series = directory / "SERIES"
    if not series.is_file():
        raise SeriesError(f"missing {series}")
    names = [
        line.strip() for line in series.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    present = sorted(path.name for path in directory.iterdir() if path.name != "SERIES")
    if sorted(names) != present or len(set(names)) != len(names):
        raise SeriesError(
            f"{series} must list every file of {directory} exactly once: "
            f"unlisted {sorted(set(present) - set(names))}, missing {sorted(set(names) - set(present))}"
        )
    return names


# --- refresh -----------------------------------------------------------------

def slug(subject: str) -> str:
    words = re.sub(r"[^a-z0-9]+", "-", subject.lower()).strip("-")
    if len(words) > 60:
        words = words[:60].rsplit("-", 1)[0]
    return words or "change"


def location(series: dict) -> str:
    return "the root" if series["patch_root"] == "." else series["patch_root"] + "/"


def covers(paths: list[str], changed: str) -> bool:
    return any(path == "." or changed == path or changed.startswith(path + "/") for path in paths)


def step_diff(repository: Path, parent: str, commit: str, series: dict) -> bytes:
    args = ["diff", "--no-color", "--no-ext-diff", "--no-renames", "--no-textconv",
            "--src-prefix=a/", "--dst-prefix=b/"]
    root = series["patch_root"]
    if root != ".":
        args.append(f"--relative={root}/")
    args += [parent, commit, "--", *series["paths"]]
    return git(repository, *args, binary=True)


def refresh_repository(name: str, entry: dict, revision: str) -> tuple[str, str]:
    repository = integration_source(name, entry)
    base = git(repository, "rev-parse", "--verify", entry["integration"]["base"] + "^{commit}").strip()
    head = git(repository, "rev-parse", "--verify", revision + "^{commit}").strip()
    steps = git(repository, "rev-list", "--first-parent", "--reverse", f"{base}..{head}").split()
    first_parents = git(repository, "rev-list", "--first-parent", head).split()
    if base not in first_parents:
        raise SeriesError(f"{name}: base {base[:10]} is not on the first-parent history of {head[:10]}")
    patches: dict[str, list[tuple[str, bytes]]] = {series["name"]: [] for series in entry["series"]}
    for index, commit in enumerate(steps, start=1):
        parent = git(repository, "rev-parse", f"{commit}^1").strip()
        changed = git(repository, "diff", "--no-renames", "--name-only", parent, commit).split("\n")
        stray = [path for path in changed if path and not any(covers(s["paths"], path) for s in entry["series"])]
        if stray:
            raise SeriesError(f"{name} {commit[:10]} changes paths outside every series: {stray[:5]}")
        subject = git(repository, "log", "-1", "--format=%s", commit).strip()
        message = git(repository, "log", "-1", "--format=%B", commit).rstrip("\n")
        parents = git(repository, "log", "-1", "--format=%P", commit).split()
        for series in entry["series"]:
            diff = step_diff(repository, parent, commit, series)
            if not diff:
                continue
            for line in diff.decode(errors="replace").splitlines():
                if REJECTED_DIFF_LINE.match(line):
                    raise SeriesError(f"{name} {commit[:10]}: unsupported diff content: {line}")
            # The header names only the step, never the series head, so a refresh after
            # new integration steps leaves the existing patch files unchanged.
            header = [
                f"{name} integration step {index} after {base[:10]}: commit {commit}",
                f"first parent {parent}" + (f", merged {' '.join(parents[1:])}" if len(parents) > 1 else ""),
                f"series {series['name']}: patch -p1 in {location(series)} of the upstream source",
                "",
                *message.splitlines(),
            ]
            text = ("".join(f"# {line}".rstrip() + "\n" for line in header) + "\n").encode() + diff
            patches[series["name"]].append((f"{index:04d}-{slug(subject)}.patch", text))
    for series in entry["series"]:
        directory = SERIES_ROOT / series["name"]
        directory.mkdir(parents=True, exist_ok=True)
        wanted = dict(patches[series["name"]])
        for path in directory.iterdir():
            if path.name != "SERIES" and path.name not in wanted:
                path.unlink()
        for filename, text in wanted.items():
            target = directory / filename
            if not target.exists() or target.read_bytes() != text:
                target.write_bytes(text)
        (directory / "SERIES").write_text(
            "# Generated by verification/toolchain/series.py refresh; do not edit by hand.\n"
            f"# {name} integration {base[:10]}..{head[:10]}: one patch per first-parent step that\n"
            f"# changes {' '.join(series['paths'])}, applied in this order with `patch -p1` in\n"
            f"# {location(series)} of the upstream source {entry['revision'][:10]}.\n"
            + "".join(f"{filename}\n" for filename in wanted)
        )
        print(f"{series['name']}: {len(wanted)} patches ({name} {base[:10]}..{head[:10]})")
    return base, head


# --- check -------------------------------------------------------------------

def tree_entry(path: Path) -> tuple:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        return ("link", os.readlink(path))
    if stat.S_ISREG(info.st_mode):
        return ("file", bool(info.st_mode & stat.S_IXUSR), hashlib.sha256(path.read_bytes()).hexdigest())
    return ("other", stat.S_IFMT(info.st_mode))


def snapshot(directory: Path) -> dict[str, tuple]:
    """Every file, symlink or special file below `directory` (directories are implied)."""
    entries = {}
    for current, dirnames, filenames in os.walk(directory):
        for name in [*dirnames, *filenames]:
            path = Path(current) / name
            if not path.is_dir() or path.is_symlink():
                entries[str(path.relative_to(directory))] = tree_entry(path)
    return entries


def export_commit(repository: Path, commit: str, destination: Path) -> None:
    destination.mkdir(parents=True)
    archive = git(repository, "archive", "--format=tar", commit, binary=True)
    subprocess.run(["tar", "-x", "-C", str(destination)], input=archive, check=True)


def apply_series(patch: Path, directory: Path, series_dir: Path) -> int:
    names = series_names(series_dir)
    for filename in names:
        result = subprocess.run(
            [str(patch), "-p1", "-F0", "--no-backup-if-mismatch", "--forward", "--batch",
             "-i", str(series_dir / filename)],
            cwd=directory, capture_output=True, text=True,
        )
        output = result.stdout + result.stderr
        unexpected = [line for line in output.splitlines() if line and not PATCH_PROGRESS.match(line)]
        if result.returncode != 0 or unexpected:
            raise SeriesError(f"{series_dir.name}/{filename} does not apply exactly:\n{output}")
    return len(names)


def check_repository(name: str, entry: dict, patch: Path) -> None:
    repository = integration_source(name, entry)
    commit = entry["integration"]["commit"]
    base = entry["integration"]["base"]
    upstream = upstream_source(name, entry["revision"])
    with tempfile.TemporaryDirectory(prefix=f"series-{name}-") as scratch:
        work = Path(scratch)
        applied, expected, pristine = work / "applied", work / "expected", work / "pristine"
        shutil.copytree(upstream, applied, symlinks=True)
        for current, dirnames, filenames in os.walk(applied):
            for item in [current, *(os.path.join(current, f) for f in filenames)]:
                if not os.path.islink(item):
                    os.chmod(item, os.lstat(item).st_mode | stat.S_IWUSR)
        counts = []
        for series in entry["series"]:
            count = apply_series(patch, applied / series["patch_root"], SERIES_ROOT / series["name"])
            counts.append(f"{series['name']} {count}")
        export_commit(repository, commit, expected)
        export_commit(repository, base, pristine)
        got, want, before = snapshot(applied), snapshot(expected), snapshot(pristine)
        # Upstream files the integration repository never tracked (absent from its
        # base and its commit) stay as upstream has them.
        upstream_only = {
            path for path in set(got) - set(want)
            if path not in before and os.path.lexists(upstream / path) and tree_entry(upstream / path) == got[path]
        }
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want) - upstream_only)
        changed = sorted(path for path in set(got) & set(want) if got[path] != want[path])
        if missing or extra or changed:
            raise SeriesError(
                f"{name}: series applied to upstream {entry['revision'][:10]} differ from integration "
                f"{commit[:10]}: missing {missing[:10]}, extra {extra[:10]}, changed {changed[:10]}"
            )
    kept = f"; {len(upstream_only)} upstream files untracked by the integration repository kept" if upstream_only else ""
    print(
        f"{name}: upstream {entry['revision'][:10]} + series ({', '.join(counts)}) = integration "
        f"{commit[:10]} ({len(want)} tree entries identical{kept})"
    )


def check(toolchain: dict) -> None:
    patch = gnu_patch()
    for name in SOURCE_VARIABLES:
        check_repository(name, toolchain[name], patch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="check the committed series against toolchain.json")
    refresh = commands.add_parser("refresh", help="regenerate the series from the integration repositories")
    refresh.add_argument("--charon-rev", default="main")
    refresh.add_argument("--aeneas-rev", default="main")
    args = parser.parse_args()
    toolchain = load_toolchain()
    if args.command == "refresh":
        for name, revision in (("charon", args.charon_rev), ("aeneas", args.aeneas_rev)):
            base, head = refresh_repository(name, toolchain[name], revision)
            toolchain[name]["integration"].update(base=base, commit=head)
        TOOLCHAIN_JSON.write_text(json.dumps(toolchain, indent=2) + "\n")
    check(toolchain)


if __name__ == "__main__":
    try:
        main()
    except (SeriesError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
