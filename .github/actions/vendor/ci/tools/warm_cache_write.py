#!/usr/bin/env python3
"""Push and pull one Rust artifact through the warm runner's Kache remote.

`devenv test` runs this module as the root task `ci:warm-cache`. On a
GitHub-hosted runner or a developer machine the module exits successfully
without any cache traffic. On the warm self-hosted runner it:

1. compiles a small fixture crate through the real Kache binary into an
   isolated store below `RUNNER_TEMP`,
2. pushes the resulting store entries to the Tunnet-only S3 gateway,
3. lists the new remote objects anonymously through that gateway, and
4. pulls the entries back into an emptied store.

Each run supplies its own Kache configuration. Kache 0.12.0 requires signing
credentials, so public, nonsecret placeholders satisfy its client-side SigV4
requirement; the gateway requires only network membership, not those values.
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
from datetime import UTC, datetime
from pathlib import Path

KACHE = "/run/current-system/sw/bin/kache"
AWS = "/run/current-system/sw/bin/aws"
BUCKET = "sand-dollar-kache-production-010928226848"
ENDPOINT = "https://kache.mesh.cproof.ai"
REGION = "us-east-2"
FIXTURE = Path(".github/fixtures/warm-cache-write")
CRATE = "warm_cache_write"
FIXTURE_SOURCE = Path("src/lib.rs")
FIXTURE_IDENTITY_MARKER = '"chelis-ci-warm-cache-write"'
STORE_PARENT = "kache"
INDEX_NAME = "index.db"
COMMAND_TIMEOUT_SECONDS = 300
# Objects written more than this long before the task started belong to
# earlier runs of the same fixture crate.
FRESHNESS_SLACK_SECONDS = 120
CACHE_KEY = re.compile(r"[0-9a-f]{64}")
# Kache prints a carriage-return counter and then one summary per direction.
TRANSFER = re.compile(
    r"(?:^|[\r\n])\s*(Uploaded|Downloaded):\s+(\d+)/(\d+)"
    r"(?:\s+\((\d+) failed\))?\s*(?=\r|\n|$)"
)
SECRET_SHAPES = re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}|[A-Za-z0-9+/_-]{40,}")
PROGRESS = re.compile(r"(?:Uploading|Downloading):\s+\d+/\d+")
# Kache 0.12.0 does not expose OpenDAL's skip_signature option, and its pinned
# reqsign-core 3.1.0 rejects missing credentials before invoking the signer.
# These PUBLIC protocol placeholders are not AWS credentials or gateway auth.
# Kache's static provider wins before any profile/broker credential lookup.
PUBLIC_SIGNING_VALUE = "tunnet-anonymous"


class WriteError(ValueError):
    """A fixed diagnostic without credential or endpoint detail."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise WriteError("warm-cache:" + code)


@dataclass(frozen=True, slots=True)
class Transfer:
    direction: str
    completed: int
    total: int
    failed: int


@dataclass(frozen=True, slots=True)
class RemoteObject:
    key: str
    size: int
    modified: float


def warm_runner(environment: dict[str, str]) -> bool:
    """Self-hosted jobs must exercise the cache, even if provisioning is broken."""
    return (
        environment.get("GITHUB_ACTIONS") == "true"
        and environment.get("RUNNER_ENVIRONMENT") == "self-hosted"
    )


def parse_transfer(stream: str, direction: str) -> Transfer | None:
    """Return the last summary for one direction, or None when Kache printed none."""
    found = [
        Transfer(
            match.group(1),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4) or 0),
        )
        for match in TRANSFER.finditer(stream)
        if match.group(1) == direction
    ]
    return found[-1] if found else None


def stamp_fixture(source: Path, destination: Path, identity: str) -> Path:
    """Copy the fixture crate and write the run identity into its source."""
    shutil.copytree(source, destination)
    target = destination / FIXTURE_SOURCE
    text = target.read_text(encoding="utf-8")
    require(text.count(FIXTURE_IDENTITY_MARKER) == 1, "fixture-marker")
    target.write_text(
        text.replace(FIXTURE_IDENTITY_MARKER, json.dumps(identity)), encoding="utf-8"
    )
    return destination


def cache_keys(store_root: Path) -> tuple[str, ...]:
    entries = store_root / "store"
    require(entries.is_dir(), "store-absent")
    keys = tuple(
        sorted(
            item.name
            for item in entries.iterdir()
            if item.is_dir() and CACHE_KEY.fullmatch(item.name)
        )
    )
    require(bool(keys), "store-empty")
    return keys


def remote_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    """The remote object names that Kache 0.12.0 uses for these store entries."""
    return tuple(
        f"v3/{kind}/{CRATE}/{key}{suffix}"
        for key in keys
        for kind, suffix in (("manifests", ".json"), ("packs", ".tar.zst"))
    )


def parse_listing(document: str) -> tuple[RemoteObject, ...]:
    """Parse `aws s3api list-objects-v2` JSON into bounded records."""
    value = json.loads(document)
    require(isinstance(value, dict), "listing-shape")
    contents = value.get("Contents", [])
    require(isinstance(contents, list), "listing-shape")
    objects: list[RemoteObject] = []
    for item in contents:
        require(isinstance(item, dict), "listing-shape")
        key, size, modified = (
            item.get("Key"),
            item.get("Size"),
            item.get("LastModified"),
        )
        require(
            isinstance(key, str)
            and isinstance(size, int)
            and isinstance(modified, str),
            "listing-shape",
        )
        stamp = datetime.fromisoformat(modified.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        objects.append(RemoteObject(key, size, stamp.timestamp()))
    return tuple(objects)


def new_objects(
    objects: tuple[RemoteObject, ...], started_at: float
) -> tuple[RemoteObject, ...]:
    """Objects written by this run, judged by their modification time."""
    threshold = started_at - FRESHNESS_SLACK_SECONDS
    return tuple(
        item for item in objects if item.modified >= threshold and item.size > 0
    )


def redact(text: str) -> str:
    return SECRET_SHAPES.sub("<redacted>", text)


def say(message: str) -> None:
    print(f"warm-cache: {message}", flush=True)


def run(
    argv: list[str], environment: dict[str, str], cwd: Path, code: str
) -> tuple[str, str]:
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
        raise WriteError("warm-cache:" + code + "-timeout") from None
    except OSError:
        raise WriteError("warm-cache:" + code + "-unavailable") from None
    if result.returncode != 0:
        for line in redact(result.stderr).splitlines()[-20:]:
            say("  " + line[:200])
        raise WriteError("warm-cache:" + code)
    return result.stdout, result.stderr


def summary_lines(stream: str, limit: int = 12) -> list[str]:
    """Kache's own bounded, redacted output without the in-flight counters."""
    lines: list[str] = []
    for raw in stream.replace("\r", "\n").splitlines():
        line = redact(raw.strip())
        if not line or PROGRESS.fullmatch(line):
            continue
        lines.append(line[:160])
        if len(lines) == limit:
            break
    return lines


def report_transfers(stream: str) -> None:
    for line in summary_lines(stream):
        say("  " + line)


def entry_digests(store_root: Path, keys: tuple[str, ...]) -> dict[str, str]:
    """SHA-256 over each entry's relative file names, sizes, and bytes."""
    digests: dict[str, str] = {}
    for key in keys:
        directory = store_root / "store" / key
        files = sorted(item for item in directory.rglob("*") if item.is_file())
        require(bool(files), "store-empty")
        summary = hashlib.sha256()
        for item in files:
            payload = item.read_bytes()
            summary.update(item.relative_to(directory).as_posix().encode("utf-8"))
            summary.update(b"\0" + str(len(payload)).encode("ascii") + b"\0")
            summary.update(payload)
        digests[key] = summary.hexdigest()
    return digests


def tool(name: str, path: str) -> str:
    located = shutil.which(name, path=path)
    require(located is not None, f"{name}-missing")
    assert located is not None
    return located


def build_environment(
    caller: dict[str, str], work: Path, xdg_cache: Path, rustc: str
) -> dict[str, str]:
    config = work / "kache.toml"
    config.write_text(
        "[cache]\n"
        f"local_store = {json.dumps(str(xdg_cache / STORE_PARENT))}\n"
        "ignore_env = true\n"
        "local_only = false\n"
        "remote_readonly = false\n\n"
        "[cache.remote]\n"
        'type = "s3"\n'
        f'bucket = "{BUCKET}"\n'
        f'region = "{REGION}"\n'
        f'endpoint = "{ENDPOINT}"\n'
        'prefix = ""\n',
        encoding="utf-8",
    )
    environment = {
        "PATH": caller["PATH"],
        "HOME": caller["HOME"],
        "LANG": "C.UTF-8",
        "RUSTC": rustc,
        "RUSTC_WRAPPER": KACHE,
        "CARGO_HOME": str(work / "cargo"),
        "CARGO_TARGET_DIR": str(work / "target"),
        "CARGO_TERM_COLOR": "never",
        "KACHE_CONFIG": str(config),
        "KACHE_PROGRESS": "verbose",
        "KACHE_S3_ACCESS_KEY": PUBLIC_SIGNING_VALUE,
        "KACHE_S3_SECRET_KEY": PUBLIC_SIGNING_VALUE,
        "XDG_CACHE_HOME": str(xdg_cache),
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_PAGER": "",
    }
    if "SSL_CERT_FILE" in caller:
        environment["SSL_CERT_FILE"] = caller["SSL_CERT_FILE"]
    return environment


def isolated_store(root: Path, xdg_cache: Path) -> Path:
    """The run's own store, proven by the index that Kache creates beside it."""
    store_root = xdg_cache / STORE_PARENT
    require((store_root / INDEX_NAME).is_file(), "store-not-isolated")
    resolved = store_root.resolve()
    require(root.resolve() in resolved.parents, "store-not-isolated")
    return resolved


def list_remote(
    prefix: str, environment: dict[str, str], cwd: Path
) -> tuple[RemoteObject, ...]:
    stdout, _ = run(
        [
            AWS,
            "s3api",
            "list-objects-v2",
            "--endpoint-url",
            ENDPOINT,
            "--no-sign-request",
            "--region",
            REGION,
            "--bucket",
            BUCKET,
            "--prefix",
            prefix,
            "--output",
            "json",
        ],
        environment,
        cwd,
        "listing",
    )
    return parse_listing(stdout or "{}")


def publish(evidence: dict[str, object], caller: dict[str, str]) -> None:
    document = json.dumps(evidence, indent=2, sort_keys=True)
    print(document, flush=True)
    summary = caller.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write(
                "## Warm runner Kache write\n\n```json\n" + document + "\n```\n"
            )


def write(caller: dict[str, str]) -> dict[str, object]:
    started_at = time.time()
    for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "RUNNER_TEMP", "PATH", "HOME"):
        require(bool(caller.get(name)), "caller-environment")
    require(os.access(KACHE, os.X_OK), "kache-missing")
    require(os.access(AWS, os.X_OK), "aws-missing")
    cargo = tool("cargo", caller["PATH"])
    rustc = tool("rustc", caller["PATH"])
    fixture = Path.cwd() / FIXTURE
    require((fixture / "Cargo.lock").is_file(), "fixture-missing")
    identity = "/".join(
        (
            "chelis-ci-warm-cache-write",
            caller["GITHUB_RUN_ID"],
            caller["GITHUB_RUN_ATTEMPT"],
        )
    )
    root = Path(tempfile.mkdtemp(prefix="warm-cache-write-", dir=caller["RUNNER_TEMP"]))
    work = root / "work"
    xdg_cache = root / "cache"
    work.mkdir(mode=0o700)
    xdg_cache.mkdir(mode=0o700)
    environment = build_environment(caller, work, xdg_cache, rustc)
    source = stamp_fixture(fixture, work / "fixture", identity)

    say("compiling the fixture crate through Kache with isolated mesh configuration")
    run([cargo, "build", "--offline", "--locked"], environment, source, "compile")
    store_root = isolated_store(root, xdg_cache)
    keys = cache_keys(store_root)
    say(f"{len(keys)} local store entries")

    digests = entry_digests(store_root, keys)

    say("pushing the store entries")
    push_out, push_err = run([KACHE, "sync", "--push"], environment, source, "push")
    pushed = push_out + "\n" + push_err
    report_transfers(pushed)
    uploaded = parse_transfer(pushed, "Uploaded")
    require(uploaded is None or uploaded.failed == 0, "push-failed")

    say("listing the remote objects anonymously through the mesh gateway")
    manifests = new_objects(
        list_remote(f"v3/manifests/{CRATE}/", environment, source), started_at
    )
    packs = new_objects(
        list_remote(f"v3/packs/{CRATE}/", environment, source), started_at
    )
    require(bool(manifests) and bool(packs), "remote-objects-absent")
    expected = set(remote_keys(keys))
    present = {item.key for item in manifests + packs}
    require(expected.issubset(present), "remote-objects-incomplete")
    say(f"{len(manifests)} new manifests, {len(packs)} new packs")

    say("pulling the entries back into an emptied store")
    subprocess.run(
        [KACHE, "daemon", "stop"],
        cwd=source,
        env=environment,
        capture_output=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )
    shutil.rmtree(store_root)
    pull_out, pull_err = run(
        [KACHE, "sync", "--pull", "--workspace"], environment, source, "pull"
    )
    pulled = pull_out + "\n" + pull_err
    report_transfers(pulled)
    downloaded = parse_transfer(pulled, "Downloaded")
    require(downloaded is None or downloaded.failed == 0, "pull-failed")
    # The readback proof is the restored store itself: every pushed key is
    # present again with byte-identical entry contents.
    restored_root = isolated_store(root, xdg_cache)
    restored = cache_keys(restored_root)
    require(set(keys).issubset(restored), "readback-incomplete")
    require(entry_digests(restored_root, keys) == digests, "readback-altered")

    subprocess.run(
        [KACHE, "daemon", "stop"],
        cwd=source,
        env=environment,
        capture_output=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )
    shutil.rmtree(root, ignore_errors=True)
    return {
        "schema": "chelis-ci-warm-cache-write/v1",
        "run_id": caller["GITHUB_RUN_ID"],
        "run_attempt": caller["GITHUB_RUN_ATTEMPT"],
        "crate": CRATE,
        "cache_keys": list(keys),
        "uploaded": None if uploaded is None else uploaded.total,
        "upload_reported": uploaded is not None,
        "remote_manifests": [[item.key, item.size] for item in manifests],
        "remote_packs": [[item.key, item.size] for item in packs],
        "expected_keys_present": expected.issubset(present),
        "downloaded": None if downloaded is None else downloaded.total,
        "download_reported": downloaded is not None,
        "restored_keys": len(restored),
        "restored_bytes_identical": True,
        "push_output": summary_lines(pushed),
        "pull_output": summary_lines(pulled),
        "seconds": round(time.time() - started_at, 1),
    }


def main() -> int:
    caller = dict(os.environ)
    if not warm_runner(caller):
        say("skipped: not the warm self-hosted runner")
        return 0
    try:
        publish(write(caller), caller)
    except WriteError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
