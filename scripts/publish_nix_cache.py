#!/usr/bin/env python3
"""Push the Chelis Nix packages and their build inputs to Cachix.

`nix-packages.yml` runs this script on `main` and on release tags, after the
complete native flake check set passed on that system. `release.yml` dispatches
that run on each release tag. A daily schedule runs it on `main` when the cache
does not hold the packages, and a maintainer can dispatch it on `main`.

The script gives four groups of paths to `cachix push`:

- The outputs of `packages.<system>.chelis`, `chelis-runtime`,
  `chelis-unwrapped`, and `chelisup`.
- The cvc5 tree that the `cvc5-sys` crate links, which the flake exposes as
  `legacyPackages.<system>.cvc5-dir`.
- Each Rust crate build in the build closure of the four packages.
  `buildRustCrate` names each crate derivation `rust_<crate>-<version>`.
- The Devenv shell roots under `$DEVENV_DOTFILE/gc/` and the `$DEVENV_PROFILE`
  profile. `scripts/ci_devenv.py` keeps `activated-shell`, the shell that
  activation built, and `activated-shell-env`, the shell environment that
  Devenv built from it. At publish time `shell` holds the shell of the
  `devenv test` smoke step. Their runtime closures hold the Rust toolchain,
  Kache, the other shell packages, and the small generated shell files.

Cachix also pushes the runtime closure of each path. With the default
"upstream caches" setting of the cache, Cachix skips each path that
cache.nixos.org serves.

A build of a later revision then downloads the unchanged cvc5 tree and crates.
It compiles only the crates whose sources or dependencies changed. A later run
on an unchanged Devenv shell downloads the activated shell and its environment,
and it builds no derivation during activation.

The Cachix CLI comes from the flake's pinned Nixpkgs, and it reads its token
from `CACHIX_AUTH_TOKEN`. The workflow fills that variable from the
`CACHIX_DEPLOY` repository secret. `flake.nix` names the same cache in
`nixConfig.extra-substituters`. `docs/contributor_setup.md` says which Nix
users download these paths instead of compiling them.

Exit codes: 0 pushed (or, with `--dry-run`, listed), 1 a Nix or Cachix command
failed, the build closure held no crate build, or a Devenv shell root was
missing, 2 the token is missing.
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
PACKAGES = ("chelis", "chelis-runtime", "chelis-unwrapped", "chelisup")
# The prebuilt cvc5 tree that the cvc5-sys crate links.
CVC5_TREE = "cvc5-dir"
# buildRustCrate names each crate derivation `rust_<crate>-<version>`.
CRATE_PREFIX = "rust_"
STORE_PREFIX = "/nix/store/"
# The roots under `$DEVENV_DOTFILE/gc/`. `devenv test` replaces `shell`.
# `scripts/ci_devenv.py` keeps the two activated roots.
DEVENV_ROOTS = ("shell", "activated-shell", "activated-shell-env")

Runner = Callable[[Sequence[str], str | None], str]
Resolver = Callable[[str], str]


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
    """The output paths of the four packages and the cvc5 tree."""
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
    """The Rust crate builds in the build closure of the four packages."""
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


def devenv_shell_roots(
    environ: Mapping[str, str], resolve: Resolver = os.path.realpath
) -> list[str]:
    """The store paths of the Devenv shell roots and the profile."""
    dotfile = environ.get("DEVENV_DOTFILE")
    profile = environ.get("DEVENV_PROFILE")
    if not dotfile or not profile:
        raise PublishError(
            "DEVENV_DOTFILE and DEVENV_PROFILE are not set, so no Devenv shell is active"
        )
    links = [*(str(Path(dotfile) / "gc" / name) for name in DEVENV_ROOTS), profile]
    roots = [resolve(link) for link in links]
    for root in roots:
        if not root.startswith(STORE_PREFIX):
            raise PublishError(f"a Devenv shell root is not a store path: {root}")
    return roots


def push(paths: Sequence[str], run: Runner) -> None:
    run(
        ["nix", "run", "--inputs-from", ".", "nixpkgs#cachix", "--", "push", CACHE_NAME],
        "".join(f"{path}\n" for path in paths),
    )


def main(
    argv: Sequence[str] | None = None,
    run: Runner = run_command,
    environ: Mapping[str, str] = os.environ,
    resolve: Resolver = os.path.realpath,
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
        shell_roots = devenv_shell_roots(environ, resolve)
        system = current_system(run)
        paths = sorted(
            {
                *built_outputs(system, run),
                *crate_builds(system, run),
                *shell_roots,
            }
        )
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
