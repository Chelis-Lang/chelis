#!/usr/bin/env python3
"""Check that the static Linux chelis and chelisup look up hosts without NSS plugins.

A statically linked glibc still reads /etc/nsswitch.conf. For a service it does not
carry, such as systemd's myhostname, resolve, or mymachines, it dlopens the host's
libnss_<service>.so.2. That plugin links the host's libc.so.6, a second C library a
static process cannot host, and the first host lookup ended with SIGFPE on Fedora 44
and Arch Linux. `chelisup::nss::use_builtin_services` restricts both binaries to the
built-in files and dns services.

This check runs each binary's first GitHub API request inside pinned distribution
images whose hosts line names such a plugin, with no network and an API base under
the reserved .invalid domain. The request cannot succeed. It must fail as an ordinary
error that names the host, which shows the lookup ran and returned. A lookup that
ends the process with a signal fails the check, and so does an image whose hosts
line no longer names a plugin the image carries.

Usage: python3 scripts/verify_static_nss.py --chelis PATH --chelisup PATH
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import signal
import subprocess
import sys

# Images whose hosts lines name systemd's NSS plugins: Fedora's reaches
# myhostname after files, and Arch's starts with mymachines.
IMAGES = (
    "fedora:44@sha256:43b29f65a41eb9c35e1cd5323e3bdf3b655c2357a9f4f1ff2f9c2798e5045d80",
    "archlinux:base-20260927.0.600689@sha256:b21322c663be387c0ed9cbc7bbbfe18e41633ad4e7b7c77cfad45f128be20040",
)
PROBE_HOST = "chelis-nss-probe.invalid"
BUILTIN_SERVICES = frozenset({"files", "dns"})
LIBRARY_DIRS = ("/usr/lib64", "/usr/lib", "/lib64", "/lib", "/usr/lib/x86_64-linux-gnu")
ENVIRONMENT = {
    "HOME": "/tmp/nss-probe",
    "CHELIS_HOME": "/tmp/nss-probe/.chelis",
    # Both binaries resolve a token before their first request.
    "GITHUB_TOKEN": "nss-probe",
    "CHELISUP_GITHUB_BASE_API": f"http://{PROBE_HOST}",
    "CHELIS_REEF_GITHUB_BASE_API": f"http://{PROBE_HOST}",
}
COMMANDS = {
    "chelisup": ("install", "0.1.0"),
    "chelis": ("reef", "install", "--from-github", "chelis-lang/nss-probe@v0.1.0"),
}
MARKER = "--- chelis nss libraries ---"
TIMEOUT_SECONDS = 300

Docker = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class CheckFailed(Exception):
    """One image or binary did not show a lookup that returns."""


def docker(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *argv],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )


def plugin_services(nsswitch: str) -> list[str]:
    """The services the hosts line names that glibc does not build in, in order."""
    for raw in nsswitch.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("hosts:"):
            return [
                word
                for word in line.removeprefix("hosts:").split()
                if not word.startswith("[") and word not in BUILTIN_SERVICES
            ]
    return []


def witness(image: str, run: Docker) -> str:
    """The NSS plugin a host lookup in `image` would load."""
    listing = " ".join(f"{directory}/libnss_*.so.2" for directory in LIBRARY_DIRS)
    script = f"cat /etc/nsswitch.conf; echo '{MARKER}'; ls {listing} 2>/dev/null; true"
    result = run(["run", "--rm", "--network", "none", image, "sh", "-c", script])
    config, marker, libraries = result.stdout.partition(MARKER)
    if result.returncode != 0 or not marker:
        raise CheckFailed(
            f"could not read its NSS configuration: {result.stderr.strip()}"
        )
    carried = {Path(path).name for path in libraries.split()}
    for service in plugin_services(config):
        if f"libnss_{service}.so.2" in carried:
            return service
    raise CheckFailed(
        "its hosts line names no NSS plugin the image carries, so it cannot show the crash"
    )


def judge(returncode: int, output: str) -> str:
    """Describe a lookup that returned as an ordinary error; raise otherwise."""
    tail = output.strip().splitlines()[-1:] or ["no output"]
    if returncode > 128 and returncode - 128 in signal.valid_signals():
        raise CheckFailed(
            f"ended with {signal.Signals(returncode - 128).name}: {tail[0]}"
        )
    if returncode == 0:
        raise CheckFailed(f"succeeded, but no request to {PROBE_HOST} can: {tail[0]}")
    if returncode != 1:
        raise CheckFailed(f"exited {returncode}: {tail[0]}")
    named = [line for line in output.splitlines() if PROBE_HOST in line]
    if not named:
        raise CheckFailed(f"failed before its host lookup: {tail[0]}")
    return named[0].strip()


def probe(image: str, name: str, binary: Path, run: Docker) -> str:
    argv = [
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        f"{binary.resolve()}:/probe/{name}:ro",
    ]
    for key, value in ENVIRONMENT.items():
        argv += ["-e", f"{key}={value}"]
    result = run([*argv, image, f"/probe/{name}", *COMMANDS[name]])
    return judge(result.returncode, result.stdout + result.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--chelis", type=Path, required=True)
    parser.add_argument("--chelisup", type=Path, required=True)
    args = parser.parse_args(argv)
    binaries = {"chelisup": args.chelisup, "chelis": args.chelis}
    failures: list[str] = []
    for image in IMAGES:
        label = image.split("@", 1)[0]
        try:
            pulled = docker(["pull", "--quiet", image])
            if pulled.returncode != 0:
                raise CheckFailed(f"could not pull it: {pulled.stderr.strip()}")
            service = witness(image, docker)
        except (CheckFailed, subprocess.TimeoutExpired) as error:
            failures.append(f"{label}: {error}")
            continue
        for name, binary in binaries.items():
            try:
                detail = probe(image, name, binary, docker)
            except (CheckFailed, subprocess.TimeoutExpired) as error:
                failures.append(f"{label} {name}: {error}")
                continue
            print(
                f"{label} {name}: hosts line names {service}; lookup failed normally: {detail}",
                flush=True,
            )
    for failure in failures:
        print(f"::error::{failure}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
