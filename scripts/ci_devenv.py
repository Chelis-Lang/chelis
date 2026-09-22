#!/usr/bin/env python3
"""Activate the checkout's CI profile once, then hand it to later Actions steps.

Run this helper with the portable setup-devenv action's Python. That action
provides bootstrap tools only; the project shell supplies Cargo, C, Python,
PyO3 and Kache. Evaluate once, initialize only shell prerequisites, and require
a fresh completion receipt before publishing any environment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_MARKER = "CHELIS_CI_PROJECT_ENVIRONMENT_CAPTURED"
# Step-local files and runner identity must never become persistent job state.
EPHEMERAL = {"_", "PWD", "OLDPWD", "SHLVL", "TMP", "TEMP", "TMPDIR", "CI"}


def capture(destination: Path) -> None:
    """Record only an activated project interpreter, never a store fallback."""
    interpreter = Path(os.environ["PYO3_PYTHON"])
    environment = Path(os.environ["DEVENV_STATE"]) / "venv"
    if (interpreter != environment / "bin/python"
            or Path(sys.executable) != interpreter
            or Path(sys.prefix) != environment):
        raise RuntimeError("project Python, activated venv and PYO3_PYTHON disagree")
    values = dict(os.environ, VIRTUAL_ENV=str(environment))
    destination.write_text(json.dumps(values), encoding="utf-8")
    print(CAPTURE_MARKER, flush=True)


def hosted_kache_config(policy: Path, destination: Path) -> None:
    """Keep the repository compiler-cache policy without its mesh remote.

    GitHub-hosted runners are not Tunnet members, and the policy's `ignore_env`
    intentionally ignores `KACHE_LOCAL_ONLY`; a separate local-only config
    selected through `KACHE_CONFIG` is the documented offline mode."""
    lines = policy.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines or lines[0].strip() != "[cache]" or any(
        line.partition("=")[0].strip() == "local_only" for line in lines
    ):
        raise ValueError("repository Kache policy must open with [cache] and not set local_only")
    destination.write_text("".join([lines[0], "local_only = true\n", *lines[1:]]), encoding="utf-8")


def activate(
    root: Path, devenv: str, github_env: Path, github_path: Path, *,
    profile: str = "ci", kache_config: Path | None = None,
) -> int:
    """Commit the shell environment only after a successful captured entry."""
    with tempfile.TemporaryDirectory(prefix="chelis-ci-entry-") as temporary:
        destination = Path(temporary) / "environment.json"
        evaluated = subprocess.run(
            [devenv, "--no-tui", "--profile", profile, "print-dev-env"],
            cwd=root, text=True, capture_output=True, check=False,
        )
        print(evaluated.stderr, end="", file=sys.stderr, flush=True)
        if evaluated.returncode:
            return 128 - evaluated.returncode if evaluated.returncode < 0 else evaluated.returncode
        # Match the receipt-based entry used by Sand Dollar's devenv-exec.sh.
        # Implicit `devenv shell` also runs downstream enterTest tasks and can
        # return zero after failed initialization; neither is shell activation.
        initialization = r'''
set -euo pipefail
: "${DEVENV_DOTFILE:?Devenv did not provide its shell state directory}"
rm -f "$DEVENV_DOTFILE/load-exports"
"$1" --no-tui --profile "$4" tasks run devenv:enterShell --mode before >&2
if [ ! -f "$DEVENV_DOTFILE/load-exports" ]; then
  echo "Project shell initialization did not complete." >&2
  exit 1
fi
. "$DEVENV_DOTFILE/load-exports"
exec python "$2" capture "$3"
'''
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c",
             evaluated.stdout + "\n" + initialization, "project-entry",
             devenv, str(Path(__file__).resolve()), str(destination), profile],
            cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout, end="", flush=True)
        if result.returncode or not destination.is_file():
            print(f"project Devenv activation failed: {result.returncode}", file=sys.stderr)
            return (128 - result.returncode if result.returncode < 0 else result.returncode) or 1
        values = json.loads(destination.read_text(encoding="utf-8"))
        if kache_config is not None:
            values["KACHE_CONFIG"] = str(kache_config)
        # GitHub's native Node actions must keep the host loader's libraries.
        # Only repository command subprocesses consume the Nix runtime path.
        values["CHELIS_CI_LIBRARY_PATH"] = values.pop("LD_LIBRARY_PATH", "")
        paths = values["PATH"].split(os.pathsep)
        if any(not path or "\n" in path or "\r" in path for path in paths):
            raise ValueError("project PATH cannot be represented in GITHUB_PATH")
        lines = []
        for key, value in sorted(values.items()):
            if key in EPHEMERAL or key.startswith(("GITHUB_", "RUNNER_")):
                continue
            if not key.isidentifier() or not isinstance(value, str) or "\0" in value:
                raise ValueError(f"invalid project environment variable: {key!r}")
            delimiter = "CHELIS_ENV_" + uuid.uuid4().hex
            while delimiter in value.splitlines():
                delimiter = "CHELIS_ENV_" + uuid.uuid4().hex
            lines.append(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")
        # The runner resolves custom shells through GITHUB_PATH, not GITHUB_ENV.
        # It prepends entries in reverse order; retain the activated venv first.
        path_payload = "".join(f"{path}\n" for path in reversed(paths))
        with github_env.open("a", encoding="utf-8") as stream, github_path.open("a", encoding="utf-8") as path_stream:
            stream.write("".join(lines))
            path_stream.write(path_payload)
    return 0


def run_step(script: Path) -> None:
    """Run an Actions command file in the activated project runtime."""
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = environment.pop("CHELIS_CI_LIBRARY_PATH")
    os.execvpe(
        "bash", ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", str(script)],
        environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="?", choices=("activate", "capture", "run"), default="activate")
    parser.add_argument("destination", nargs="?", type=Path)
    parser.add_argument("--profile", choices=("ci", "ci-smt"), default="ci")
    parser.add_argument(
        "--hosted", action="store_true",
        help="realize the GitHub-hosted twin, which public binary caches serve",
    )
    args = parser.parse_args()
    if args.operation == "capture":
        if args.destination is None:
            parser.error("capture requires a destination")
        capture(args.destination)
        return 0
    if args.operation == "run":
        if args.destination is None:
            parser.error("run requires an Actions command file")
        run_step(args.destination)
    # Both hosted lanes share one twin: the hosted SMT lane links the prebuilt
    # cvc5 asset from the workflow instead of realizing the Nix solver closure.
    kache_config = None
    if args.hosted:
        kache_config = Path(os.environ["RUNNER_TEMP"]) / "chelis-kache-hosted.toml"
        hosted_kache_config(ROOT / ".kache.toml", kache_config)
    return activate(
        ROOT, os.environ["CHELIS_DEVENV_BIN"],
        Path(os.environ["GITHUB_ENV"]), Path(os.environ["GITHUB_PATH"]),
        profile="ci-hosted" if args.hosted else args.profile,
        kache_config=kache_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
