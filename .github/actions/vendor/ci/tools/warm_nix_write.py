#!/usr/bin/env python3
"""Push the job's new store paths to the private Nix cache and read one back.

`devenv test` runs this module as the root task `ci:warm-nix-cache`. On a
GitHub-hosted runner, on a pull request run, or on a developer machine the
module exits successfully without any cache traffic. On the warm self-hosted
runner with a main-branch job it:

1. builds one unique probe derivation through the host Nix daemon,
2. lists the store paths that the daemon registered since the job started,
3. pushes the probe and those paths with the pinned niks3 client over Tunnet,
4. copies the probe from the private cache into a fresh file store, which
   requires a signature by the reviewed public key,
5. compares the restored NAR hash and signature with the original one.

The daemon queries every substituter before it builds the probe, and it
caches that miss for an hour. The readback therefore runs as the job user
against a file store, which holds no negative cache.

Tunnet membership is the only access requirement. The pinned niks3 CLI still
requires a nonempty token argument, so it receives a public, nonsecret value
that the mesh server ignores. No job or cache credential is read or forwarded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

NIX_BIN = Path("/run/current-system/sw/bin")
# The cache serves its API on the private mesh name only. The warm host reaches
# that name through its own mesh enrollment.
SERVER_URL = "https://cache.mesh.cproof.ai"
# niks3 1.10.1 cmdutil.ResolveTokenSource rejects missing/empty tokens, even
# though the client library supports anonymous access. This public protocol
# value satisfies the CLI, not authentication; only Tunnet admits the request.
# Explicit --auth-token also prevents fallback to a token in the host's XDG config.
# https://github.com/Mic92/niks3/blob/v1.10.1/cmdutil/cmdutil.go
ANONYMOUS_TOKEN = "tunnet-anonymous"
INVENTORY = Path("tools/cproof_cache/seed-inventory.json")
SIGNING_KEY = "sand-dollar-niks3"
PUBLIC_KEY = "sand-dollar-niks3:NIdyQjpputnL1N5Utmm+4ve1/ugXmLLB6RSN1xnzBtI="
PROBE_NAME = "chelis-ci-warm-nix-write"
MAX_PUSH_PATHS = 400
# A path registered this long before the job process started still counts as
# the job's own work. Process start times and registration times are seconds.
JOB_START_SLACK_SECONDS = 60
COMMAND_TIMEOUT_SECONDS = 900
WORKER_PROCESS = "Runner.Worker"
STORE_PATH = re.compile(
    r"/nix/store/[0-9abcdfghijklmnpqrsvwxyz]{32}-[A-Za-z0-9.+_?=-]{1,211}"
)
IDENTITY = re.compile(r"[0-9]{1,20}")
NAR_HASH = re.compile(r"sha256-[A-Za-z0-9+/]{43}=")
SHA256 = re.compile(r"[0-9a-f]{64}")
SECRET_SHAPES = re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}|[A-Za-z0-9+/_-]{40,}")
FORWARDED_ENVIRONMENT = ("SSL_CERT_FILE", "NIX_SSL_CERT_FILE")


class WriteError(ValueError):
    """A fixed diagnostic without credential or endpoint detail."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise WriteError("warm-nix-cache:" + code)


@dataclass(frozen=True, slots=True)
class Niks3Package:
    version: str
    url: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PathRecord:
    path: str
    registered_at: int
    nar_hash: str
    nar_size: int
    signatures: tuple[str, ...]

    @property
    def derivation(self) -> bool:
        return self.path.endswith(".drv")


def warm_runner(environment: dict[str, str]) -> bool:
    """True only for a main-branch job on the warm self-hosted runner."""
    return (
        environment.get("GITHUB_ACTIONS") == "true"
        and environment.get("RUNNER_ENVIRONMENT") == "self-hosted"
        and environment.get("GITHUB_REF") == "refs/heads/main"
    )


def load_package(document: str) -> Niks3Package:
    """The pinned niks3 client from the seed inventory."""
    value = json.loads(document)
    require(isinstance(value, dict), "inventory-shape")
    package = value.get("niks3")
    require(isinstance(package, dict), "inventory-shape")
    version, url, sha256 = (
        package.get("version"),
        package.get("url"),
        package.get("sha256"),
    )
    require(
        isinstance(version, str)
        and isinstance(url, str)
        and isinstance(sha256, str)
        and SHA256.fullmatch(sha256) is not None
        and url.startswith("https://github.com/Mic92/niks3/releases/download/"),
        "inventory-package",
    )
    return Niks3Package(version, url, sha256)


def parse_path_info(document: str) -> tuple[PathRecord, ...]:
    """Parse `nix path-info --json` into bounded records.

    Nix prints one object keyed by store path. A later format wraps that
    object in a `paths` member beside a `version` member.
    """
    value = json.loads(document)
    require(isinstance(value, dict), "path-info-shape")
    if isinstance(value.get("paths"), dict):
        value = value["paths"]
    records: list[PathRecord] = []
    for path, info in value.items():
        require(
            isinstance(path, str)
            and STORE_PATH.fullmatch(path) is not None
            and isinstance(info, dict),
            "path-info-shape",
        )
        # A binary cache or file store records no registration time.
        registered = info.get("registrationTime")
        if registered is None:
            registered = 0
        nar_hash = info.get("narHash")
        nar_size = info.get("narSize")
        signatures = info.get("signatures", [])
        require(
            isinstance(registered, int)
            and isinstance(nar_hash, str)
            and NAR_HASH.fullmatch(nar_hash) is not None
            and isinstance(nar_size, int)
            and isinstance(signatures, list)
            and all(isinstance(item, str) for item in signatures),
            "path-info-shape",
        )
        records.append(
            PathRecord(path, registered, nar_hash, nar_size, tuple(signatures))
        )
    return tuple(sorted(records, key=lambda record: record.path))


def select_new_paths(
    records: tuple[PathRecord, ...], since: float, limit: int
) -> tuple[tuple[str, ...], int]:
    """The job's own outputs: paths registered since `since`, without derivations.

    Returns the selected paths, newest first, and the count left out by the
    limit.
    """
    fresh = sorted(
        (
            record
            for record in records
            if record.registered_at >= since and not record.derivation
        ),
        key=lambda record: (-record.registered_at, record.path),
    )
    selected = tuple(record.path for record in fresh[:limit])
    return selected, max(len(fresh) - limit, 0)


def probe_expression(identity: str) -> str:
    """A unique derivation whose output names this run and nothing else."""
    require(re.fullmatch(r"[0-9]{1,20}/[0-9]{1,4}", identity) is not None, "identity")
    stamp = identity.replace("/", "-")
    return (
        "derivation { "
        f'name = "{PROBE_NAME}-{stamp}"; '
        "system = builtins.currentSystem; "
        'builder = "/bin/sh"; '
        f'args = [ "-c" "echo {PROBE_NAME}/{identity} > $out" ]; '
        "}"
    )


def parse_store_path_line(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    require(bool(lines) and STORE_PATH.fullmatch(lines[-1]) is not None, "store-path")
    return lines[-1]


def process_start_time(pid: int, proc: Path = Path("/proc")) -> tuple[str, int, float]:
    """The command name, parent, and start time (epoch seconds) of one process."""
    stat = (proc / str(pid) / "stat").read_text(encoding="utf-8")
    boot = (proc / "stat").read_text(encoding="utf-8")
    open_index, close_index = stat.index("("), stat.rindex(")")
    command = stat[open_index + 1 : close_index]
    fields = stat[close_index + 2 :].split()
    parent = int(fields[1])
    start_ticks = int(fields[19])
    boot_time = next(
        int(line.split()[1]) for line in boot.splitlines() if line.startswith("btime ")
    )
    ticks = os.sysconf("SC_CLK_TCK")
    return command, parent, boot_time + start_ticks / ticks


def job_started_at(pid: int, proc: Path = Path("/proc")) -> float:
    """The start time of the runner worker process that owns this job."""
    current = pid
    for _ in range(64):
        try:
            command, parent, started = process_start_time(current, proc)
        except (OSError, ValueError, IndexError, StopIteration):
            break
        if command == WORKER_PROCESS:
            return started
        if parent <= 1:
            break
        current = parent
    raise WriteError("warm-nix-cache:job-start-unknown")


def signature_keys(signatures: tuple[str, ...]) -> tuple[str, ...]:
    """The key names of the signatures on one path."""
    return tuple(sorted({item.split(":", 1)[0] for item in signatures if ":" in item}))


def redact(text: str) -> str:
    return SECRET_SHAPES.sub("<redacted>", text)


def say(message: str) -> None:
    print(f"warm-nix-cache: {message}", flush=True)


def summary_lines(stream: str, limit: int = 12) -> list[str]:
    lines: list[str] = []
    for raw in stream.replace("\r", "\n").splitlines():
        line = redact(raw.strip())
        if not line:
            continue
        lines.append(line[:160])
        if len(lines) == limit:
            break
    return lines


def run(
    argv: list[str],
    environment: dict[str, str],
    cwd: Path,
    code: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise WriteError("warm-nix-cache:" + code + "-timeout") from None
    except OSError:
        raise WriteError("warm-nix-cache:" + code + "-unavailable") from None
    if check and result.returncode != 0:
        for line in summary_lines(result.stderr, limit=20):
            say("  " + line)
        raise WriteError("warm-nix-cache:" + code)
    return result


def nix_environment(caller: dict[str, str]) -> dict[str, str]:
    environment = {
        "PATH": f"{NIX_BIN}:{caller['PATH']}",
        "HOME": caller["HOME"],
        "LANG": "C.UTF-8",
    }
    for name in FORWARDED_ENVIRONMENT:
        if name in caller:
            environment[name] = caller[name]
    return environment


def path_records(
    environment: dict[str, str], cwd: Path, *arguments: str
) -> tuple[PathRecord, ...]:
    result = run(
        [str(NIX_BIN / "nix"), "path-info", "--json", *arguments],
        environment,
        cwd,
        "path-info",
    )
    return parse_path_info(result.stdout or "{}")


def stage_niks3(package: Niks3Package, environment: dict[str, str], root: Path) -> Path:
    """Fetch the hash-verified client through Nix and unpack its binary."""
    result = run(
        [
            str(NIX_BIN / "nix-prefetch-url"),
            "--type",
            "sha256",
            "--print-path",
            package.url,
            package.sha256,
        ],
        environment,
        root,
        "niks3-fetch",
    )
    archive = Path(parse_store_path_line(result.stdout))
    require(
        hashlib.sha256(archive.read_bytes()).hexdigest() == package.sha256,
        "niks3-digest",
    )
    binaries = root / "niks3"
    binaries.mkdir(mode=0o700)
    run(
        [
            "tar",
            "--extract",
            "--gzip",
            "--file",
            str(archive),
            "--directory",
            str(binaries),
            "niks3",
        ],
        environment,
        root,
        "niks3-extract",
    )
    binary = binaries / "niks3"
    require(binary.is_file() and not binary.is_symlink(), "niks3-binary")
    binary.chmod(0o700)
    return binary


def publish(evidence: dict[str, object], caller: dict[str, str]) -> None:
    document = json.dumps(evidence, indent=2, sort_keys=True)
    print(document, flush=True)
    # Devenv shows a successful task's standard output only on request, so the
    # job log carries the same document on standard error.
    print(document, file=sys.stderr, flush=True)
    summary = caller.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(
                "## Warm runner Nix cache write\n\n```json\n" + document + "\n```\n"
            )


def write(caller: dict[str, str]) -> dict[str, object]:
    started_at = time.time()
    for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "RUNNER_TEMP", "PATH", "HOME"):
        require(bool(caller.get(name)), "caller-environment")
    for name in ("nix", "nix-build", "nix-store", "nix-prefetch-url"):
        require(os.access(NIX_BIN / name, os.X_OK), "nix-missing")
    require(
        IDENTITY.fullmatch(caller["GITHUB_RUN_ID"]) is not None
        and re.fullmatch(r"[0-9]{1,4}", caller["GITHUB_RUN_ATTEMPT"]) is not None,
        "identity",
    )
    identity = caller["GITHUB_RUN_ID"] + "/" + caller["GITHUB_RUN_ATTEMPT"]
    package = load_package((Path.cwd() / INVENTORY).read_text(encoding="utf-8"))
    job_start = job_started_at(os.getpid())
    root = Path(tempfile.mkdtemp(prefix="warm-nix-write-", dir=caller["RUNNER_TEMP"]))
    environment = nix_environment(caller)

    say(f"staging the niks3 client {package.version}")
    niks3 = stage_niks3(package, environment, root)

    say("building the probe derivation through the host daemon")
    built = run(
        [str(NIX_BIN / "nix-build"), "--no-out-link", "-E", probe_expression(identity)],
        environment,
        root,
        "probe-build",
    )
    probe = parse_store_path_line(built.stdout)
    (before,) = path_records(environment, root, probe)
    require(before.path == probe, "probe-record")

    say("selecting the paths registered since the job started")
    records = path_records(environment, root, "--all")
    since = job_start - JOB_START_SLACK_SECONDS
    selected, omitted = select_new_paths(
        tuple(record for record in records if record.path != probe),
        since,
        MAX_PUSH_PATHS,
    )
    pushed_bytes = sum(
        record.nar_size for record in records if record.path in set(selected)
    )
    say(f"{len(selected)} new paths, {omitted} beyond the limit")

    say("pushing the probe and the new paths")
    push = run(
        [
            str(niks3),
            "push",
            "--server-url",
            SERVER_URL,
            "--auth-token",
            ANONYMOUS_TOKEN,
            probe,
            *selected,
        ],
        environment,
        root,
        "push",
    )
    push_output = summary_lines(push.stdout + "\n" + push.stderr)
    for line in push_output:
        say("  " + line)

    say("copying the probe from the private cache into a fresh file store")
    readback = root / "readback"
    readback.mkdir(mode=0o700)
    store = f"file://{readback}"
    copy = run(
        [
            str(NIX_BIN / "nix"),
            "--extra-experimental-features",
            "nix-command",
            "copy",
            "--from",
            SERVER_URL,
            "--to",
            store,
            "--option",
            "trusted-public-keys",
            PUBLIC_KEY,
            "--option",
            "require-sigs",
            "true",
            probe,
        ],
        environment,
        root,
        "probe-readback",
    )
    readback_output = summary_lines(copy.stderr, limit=4)
    for line in readback_output:
        say("  " + line)
    (after,) = path_records(environment, root, "--store", store, probe)
    require(after.nar_hash == before.nar_hash, "readback-altered")
    keys = signature_keys(after.signatures)
    require(SIGNING_KEY in keys, "readback-unsigned")

    shutil.rmtree(root, ignore_errors=True)
    return {
        "schema": "chelis-ci-warm-nix-write/v1",
        "run_id": caller["GITHUB_RUN_ID"],
        "run_attempt": caller["GITHUB_RUN_ATTEMPT"],
        "server_url": SERVER_URL,
        "niks3_version": package.version,
        "probe_path": probe,
        "probe_nar_hash": before.nar_hash,
        "job_started_at": int(job_start),
        "pushed_paths": len(selected) + 1,
        "pushed_nar_bytes": pushed_bytes + before.nar_size,
        "omitted_paths": omitted,
        "push_output": push_output,
        "readback_store": "file",
        "readback_output": readback_output,
        "readback_nar_hash_identical": True,
        "readback_signature_keys": list(keys),
        "seconds": round(time.time() - started_at, 1),
    }


def main() -> int:
    caller = dict(os.environ)
    if not warm_runner(caller):
        say("skipped: not a main-branch job on the warm self-hosted runner")
        return 0
    try:
        publish(write(caller), caller)
    except WriteError as error:
        # Devenv shows a failed task's standard output, so the code goes there too.
        say("failed: " + str(error))
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        # The job log must name an unexpected failure before the traceback.
        say("failed: " + redact(f"{type(error).__name__}: {error}")[:300])
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
