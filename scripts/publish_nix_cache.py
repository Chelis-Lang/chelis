#!/usr/bin/env python3
"""Push the Chelis Nix packages and their build inputs to Cachix.

`nix-packages.yml` runs this script on `main` and on release tags, after the
complete native flake check set passed on that system. `release.yml` dispatches
that run on each release tag, and a maintainer can dispatch it on `main`.

The script gives three groups of paths to `cachix push`:

- The outputs of `packages.<system>.chelis`, `chelis-runtime`, and `chelisup`.
- The cvc5 tree that the `cvc5-sys` crate links, which the flake exposes as
  `legacyPackages.<system>.cvc5-dir`.
- Each Rust crate build in the build closure of the three packages.
  `buildRustCrate` names each crate derivation `rust_<crate>-<version>`.

Cachix also pushes the runtime closure of each path. With the default
"upstream caches" setting of the cache, Cachix skips each path that
cache.nixos.org serves. The script does not select the Rust toolchain, because
a build downloads it from static.rust-lang.org in about the same time.

A build of a later revision then downloads the unchanged cvc5 tree and crates,
and it compiles only the crates that the change affects.

The Cachix CLI comes from the flake's pinned Nixpkgs, and it reads its token
from `CACHIX_AUTH_TOKEN`. The workflow fills that variable from the
`CACHIX_DEPLOY` repository secret. `flake.nix` names the same cache in
`nixConfig.extra-substituters`. `docs/contributor_setup.md` says which Nix
users download these paths instead of compiling them.

Exit codes: 0 pushed (or, with `--dry-run`, listed), 1 a Nix or Cachix command
failed or the build closure held no crate build, 2 the token is missing.
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
# The prebuilt cvc5 tree that the cvc5-sys crate links.
CVC5_TREE = "cvc5-dir"
# buildRustCrate names each crate derivation `rust_<crate>-<version>`.
CRATE_PREFIX = "rust_"

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


def package_installables(system: str) -> list[str]:
    return [f".#packages.{system}.{name}" for name in PACKAGES]


def built_outputs(system: str, run: Runner) -> list[str]:
    """The output paths of the three packages and the cvc5 tree."""
    printed = run(
        [
            "nix",
            "build",
            "--no-link",
            "--print-out-paths",
            *package_installables(system),
            f".#legacyPackages.{system}.{CVC5_TREE}",
        ],
        None,
    )
    outputs = sorted({line.strip() for line in printed.splitlines() if line.strip()})
    if not outputs:
        raise PublishError("`nix build` printed no output paths")
    return outputs


def crate_builds(system: str, run: Runner) -> list[str]:
    """The Rust crate builds in the build closure of the three packages."""
    derivations = run(
        ["nix", "path-info", "--derivation", *package_installables(system)], None
    ).split()
    if not derivations:
        raise PublishError("`nix path-info --derivation` printed no derivations")
    # --include-outputs adds the outputs of the closure that exist locally.
    closure = run(
        ["nix-store", "--query", "--requisites", "--include-outputs", *derivations],
        None,
    ).split()
    crates = sorted(
        {
            path
            for path in closure
            if not path.endswith(".drv")
            and Path(path).name.partition("-")[2].startswith(CRATE_PREFIX)
        }
    )
    if not crates:
        raise PublishError("the build closure of the packages holds no crate build")
    return crates


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
        help="print the paths to push and push nothing",
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
        paths = sorted({*built_outputs(system, run), *crate_builds(system, run)})
        if args.dry_run:
            for path in paths:
                print(path)
            return 0
        push(paths, run)
    except PublishError as error:
        print(f"publish_nix_cache: {error}", file=sys.stderr)
        return 1
    print(
        f"publish_nix_cache: pushed {len(paths)} {system} path(s) and their runtime "
        f"closures to {CACHE_NAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
