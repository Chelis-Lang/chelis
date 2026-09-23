"""Build a writable Aeneas Lean library from the pinned .#aeneas-source output."""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True,
        help="The .#aeneas-source output containing backends/lean",
    )
    parser.add_argument(
        "--destination", type=Path, required=True,
        help="A new writable library directory; existing paths are refused",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve() / "backends" / "lean"
    destination = args.destination.expanduser().absolute()
    for filename in (
        "lean-toolchain", "lake-manifest.json", "lakefile.lean",
        "Aeneas.lean", "AeneasMeta.lean",
    ):
        if not (source / filename).is_file():
            parser.error(f"required source file is missing: {source / filename}")
    for directory in ("Aeneas", "AeneasMeta"):
        if not (source / directory).is_dir():
            parser.error(f"required source directory is missing: {source / directory}")
    if destination.exists() or destination.is_symlink():
        parser.error(f"destination already exists: {destination}")

    # Preserve the pinned toolchain and manifest, never a developer's Lake state.
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".lake"))
    for directory, _, filenames in os.walk(destination):
        paths = [Path(directory), *(Path(directory) / name for name in filenames)]
        for path in paths:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)

    environment = os.environ.copy()
    environment.pop("CI", None)
    for command in (["lake", "exe", "cache", "get"], ["lake", "build"]):
        print("+ " + shlex.join(command), flush=True)
        subprocess.run(command, cwd=destination, env=environment, check=True)
    print(f"Built Aeneas Lean library: {destination}")
    print(f"Use --aeneas-lean-dir {shlex.quote(str(destination))}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
