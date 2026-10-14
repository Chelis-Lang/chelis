#!/usr/bin/env python3
"""Push the locally built runtime closure of the Chelis Nix packages to Cachix.

`nix-packages.yml` runs this script on a published release, after the complete
native flake check set passed on that system. The script resolves the runtime
closure of `packages.<system>.chelis`, `chelis-runtime`, and `chelisup`, and
pushes every path in it that this machine built. Nix records a locally built
path as ultimate. A path that a substituter supplied, for example zstd from
cache.nixos.org, is not ultimate, so the cache receives Chelis builds only.

The Cachix CLI comes from the flake's pinned Nixpkgs, and it reads its token
from `CACHIX_AUTH_TOKEN`. `flake.nix` names the same cache in
`nixConfig.extra-substituters`, so `nix run` and `nix build` of a released
revision download these outputs instead of compiling them.

Exit codes: 0 pushed (or, with `--dry-run`, listed), 1 a Nix or Cachix command
failed, 2 the token is missing or the closure holds no locally built path.
"""

from __future__ import annotations

import argparse
import json
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


def closure_infos(outputs: Sequence[str], run: Runner) -> Mapping[str, Mapping[str, object]]:
    printed = run(
        ["nix", "path-info", "--recursive", "--json", "--stdin"],
        "".join(f"{path}\n" for path in outputs),
    )
    try:
        decoded = json.loads(printed)
    except json.JSONDecodeError as error:
        raise PublishError(f"`nix path-info --json` printed invalid JSON: {error}") from error
    # Nix 2.19 and later print an object keyed by store path. Earlier
    # versions print a list of objects with a "path" key.
    if isinstance(decoded, dict):
        entries = decoded.items()
    elif isinstance(decoded, list):
        entries = ((item.get("path"), item) for item in decoded if isinstance(item, dict))
    else:
        raise PublishError("`nix path-info --json` printed neither an object nor a list")
    infos: dict[str, Mapping[str, object]] = {}
    for path, info in entries:
        if not isinstance(path, str) or not isinstance(info, dict):
            raise PublishError(f"`nix path-info --json` printed an unknown entry for {path!r}")
        infos[path] = info
    return infos


def locally_built(infos: Mapping[str, Mapping[str, object]]) -> list[str]:
    return sorted(path for path, info in infos.items() if info.get("ultimate") is True)


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
        help="print the locally built paths and push nothing",
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
        paths = locally_built(closure_infos(package_outputs(system, run), run))
        if not paths:
            print(
                f"publish_nix_cache: no locally built path in the {system} package closure",
                file=sys.stderr,
            )
            return 2
        if args.dry_run:
            for path in paths:
                print(path)
            return 0
        push(paths, run)
    except PublishError as error:
        print(f"publish_nix_cache: {error}", file=sys.stderr)
        return 1
    print(f"publish_nix_cache: pushed {len(paths)} {system} path(s) to {CACHE_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
