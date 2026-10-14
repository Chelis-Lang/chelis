#!/usr/bin/env python3
"""Push the runtime closure of the Chelis Nix packages to Cachix.

`nix-packages.yml` runs this script on a run on a release tag, after the
complete native flake check set passed on that system. `release.yml` dispatches
that run after it publishes the release. The script gives the output paths
of `packages.<system>.chelis`, `chelis-runtime`, and `chelisup` to
`cachix push`, and Cachix pushes their runtime closures. With the default
"upstream caches" setting of the cache, Cachix skips each path that
cache.nixos.org serves, for example zstd. Thus the cache receives the Chelis
outputs and the dependencies that cache.nixos.org does not serve.

The Cachix CLI comes from the flake's pinned Nixpkgs, and it reads its token
from `CACHIX_AUTH_TOKEN`. The workflow fills that variable from the
`CACHIX_DEPLOY` repository secret. `flake.nix` names the same cache in
`nixConfig.extra-substituters`. `docs/contributor_setup.md` says which Nix
users download these outputs instead of compiling them.

Exit codes: 0 pushed (or, with `--dry-run`, listed), 1 a Nix or Cachix command
failed, 2 the token is missing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_NAME = "chelis-lang"
TOKEN_VARIABLE = "CACHIX_AUTH_TOKEN"
PACKAGES = ("chelis", "chelis-runtime", "chelisup")

Runner = Callable[[Sequence[str], str | None], str]


class PublishError(RuntimeError):
    """A Nix or Cachix command failed, or its output had an unknown shape."""


def run_command(argv: Sequence[str], stdin: str | None = None) -> str:
    completed = subprocess.run(
        list(argv),
        cwd=REPO_ROOT,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PublishError(
            f"`{' '.join(argv)}` failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def current_system(run: Runner) -> str:
    system = run(
        ["nix", "eval", "--impure", "--raw", "--expr", "builtins.currentSystem"], None
    ).strip()
    if not system:
        raise PublishError("`nix eval builtins.currentSystem` printed no system")
    return system


def package_outputs(system: str, run: Runner) -> list[str]:
    installables = [f".#packages.{system}.{name}" for name in PACKAGES]
    printed = run(
        ["nix", "build", "--no-link", "--print-out-paths", *installables], None
    )
    outputs = sorted({line.strip() for line in printed.splitlines() if line.strip()})
    if not outputs:
        raise PublishError("`nix build` printed no output paths")
    return outputs


def push(paths: Sequence[str], run: Runner) -> None:
    run(
        ["nix", "run", "--inputs-from", ".", "nixpkgs#cachix", "--", "push", CACHE_NAME],
        "".join(f"{path}\n" for path in paths),
    )


def main(
    argv: Sequence[str] | None = None,
    run: Runner = run_command,
    environ: Mapping[str, str] = os.environ,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the package output paths and push nothing",
    )
    args = parser.parse_args(argv)
    if not args.dry_run and not environ.get(TOKEN_VARIABLE):
        print(
            f"publish_nix_cache: {TOKEN_VARIABLE} is not set, so nothing was pushed",
            file=sys.stderr,
        )
        return 2
    try:
        system = current_system(run)
        paths = package_outputs(system, run)
        if args.dry_run:
            for path in paths:
                print(path)
            return 0
        push(paths, run)
    except PublishError as error:
        print(f"publish_nix_cache: {error}", file=sys.stderr)
        return 1
    print(
        f"publish_nix_cache: pushed the runtime closure of {len(paths)} {system} "
        f"output(s) to {CACHE_NAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
