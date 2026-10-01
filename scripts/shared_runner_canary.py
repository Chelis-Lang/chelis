"""Manual member-cache probe. Environment checks are not host authentication."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

BIN = Path("/run/current-system/sw/bin")
NIX_URL = "https://cache.mesh.cproof.ai"
KACHE_URL = "https://kache.mesh.cproof.ai"
BUCKET = "sand-dollar-kache-production-010928226848"
SIGNING_KEY = "sand-dollar-niks3"
PUBLIC_KEY = "sand-dollar-niks3:NIdyQjpputnL1N5Utmm+4ve1/ugXmLLB6RSN1xnzBtI="
STORE_PATH = re.compile(
    r"/nix/store/[0-9abcdfghijklmnpqrsvwxyz]{32}-[A-Za-z0-9+._-]{1,160}"
)


class ProbeError(ValueError):
    """A fixed diagnostic without credential values or subprocess output."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def _pairs(items):
    value = {}
    for key, item in items:
        require(key not in value, "The JSON contains a duplicate field.")
        value[key] = item
    return value


def _constant(_value):
    raise ProbeError("The JSON contains a nonfinite number.")


def document(text: str):
    require(len(text) <= 5 * 1024 * 1024, "The JSON response exceeds the size limit.")
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, RecursionError):
        raise ProbeError("The command returned invalid JSON.") from None


@dataclass(frozen=True)
class Job:
    run_id: str
    attempt: str
    sha: str
    temp: Path

    @classmethod
    def parse(cls, env: dict[str, str]) -> Job:
        expected = {
            "GITHUB_ACTIONS": "true",
            "RUNNER_ENVIRONMENT": "self-hosted",
            "GITHUB_REPOSITORY": "Chelis-Lang/chelis",
            "GITHUB_REPOSITORY_ID": "1201277270",
            "GITHUB_REPOSITORY_OWNER_ID": "275816231",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_WORKFLOW_REF": "Chelis-Lang/chelis/.github/workflows/shared-runner-canary.yml@refs/heads/main",
        }
        require(
            all(env.get(key) == value for key, value in expected.items()),
            "The job is not the reviewed manual canary.",
        )
        sha = env.get("GITHUB_SHA", "")
        require(
            re.fullmatch(r"[0-9a-f]{40}", sha) is not None and sha != "0" * 40,
            "The source revision is invalid.",
        )
        require(
            env.get("GITHUB_WORKFLOW_SHA") == sha,
            "The workflow revision differs from the source revision.",
        )
        run_id, attempt = (
            env.get("GITHUB_RUN_ID", ""),
            env.get("GITHUB_RUN_ATTEMPT", ""),
        )
        require(
            re.fullmatch(r"[1-9][0-9]{0,19}", run_id) is not None,
            "The run identifier is invalid.",
        )
        require(
            re.fullmatch(r"[1-9][0-9]{0,3}", attempt) is not None,
            "The run attempt is invalid.",
        )
        temp = Path(env.get("RUNNER_TEMP", ""))
        require(
            temp.is_absolute() and ".." not in temp.parts,
            "The runner temporary path is invalid.",
        )
        return cls(run_id, attempt, sha, temp)


@dataclass(frozen=True)
class NarRecord:
    nar_hash: str
    signatures: tuple[str, ...]

    @classmethod
    def parse(cls, text: str, expected: str) -> NarRecord:
        require(
            STORE_PATH.fullmatch(expected) is not None, "The Nix store path is invalid."
        )
        value = document(text)
        if isinstance(value, dict) and set(value) == {expected}:
            record = value[expected]
        elif (
            isinstance(value, list)
            and len(value) == 1
            and isinstance(value[0], dict)
            and value[0].get("path") == expected
        ):
            record = value[0]
        else:
            raise ProbeError("The Nix path record does not identify the probe.")
        require(isinstance(record, dict), "The Nix path record is invalid.")
        nar_hash = record.get("narHash")
        require(
            isinstance(nar_hash, str) and nar_hash.startswith("sha256-"),
            "The NAR hash is invalid.",
        )
        try:
            raw = base64.b64decode(nar_hash[7:], validate=True)
        except ValueError:
            raise ProbeError("The NAR hash is invalid.") from None
        require(
            len(raw) == 32 and base64.b64encode(raw).decode() == nar_hash[7:],
            "The NAR hash is invalid.",
        )
        signatures = record.get("signatures", [])
        require(
            isinstance(signatures, list)
            and all(isinstance(item, str) for item in signatures),
            "The NAR signatures are invalid.",
        )
        return cls(nar_hash, tuple(signatures))


def require_readback(before: NarRecord, after: NarRecord) -> None:
    require(before.nar_hash == after.nar_hash, "The Nix readback hash differs.")
    # Metadata is not signature proof. nix_probe requires Nix verification first.
    require(
        any(item.startswith(SIGNING_KEY + ":") for item in after.signatures),
        "The Nix readback lacks the reviewed signature.",
    )


@dataclass(frozen=True)
class Host:
    instance_id: str
    system: str

    @classmethod
    def parse(cls, instance: str, system: str) -> Host:
        require(
            instance == "i-0f07f7850a4551b01", "The canary is on another EC2 instance."
        )
        require(
            re.fullmatch(
                r"/nix/store/[0-9abcdfghijklmnpqrsvwxyz]{32}-nixos-system-chelis-ci-warm-amazon-[A-Za-z0-9.+_-]+",
                system,
            )
            is not None,
            "The host system identity is invalid.",
        )
        return cls(instance, system)


def host_identity() -> Host:
    instance = Path("/sys/devices/virtual/dmi/id/board_asset_tag").read_text().strip()
    system = Path("/run/current-system").resolve(strict=True)
    require(
        system == Path("/nix/var/nix/profiles/system").resolve(strict=True),
        "The active system differs from the boot profile.",
    )
    return Host.parse(instance, str(system))


def child_environment(work: Path) -> dict[str, str]:
    return {
        "PATH": str(BIN),
        "HOME": str(work / "home"),
        "LANG": "C.UTF-8",
        "XDG_CONFIG_HOME": str(work / "config"),
        "XDG_CACHE_HOME": str(work / "xdg-cache"),
        "XDG_RUNTIME_DIR": str(work / "runtime"),
        "NIX_REMOTE": "daemon",
        "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "CARGO_HOME": str(work / "cargo"),
        "CARGO_TARGET_DIR": str(work / "target"),
        "CARGO_INCREMENTAL": "0",
        "CARGO_TERM_COLOR": "never",
        "RUSTC": str(BIN / "rustc"),
        "RUSTC_WRAPPER": str(BIN / "kache"),
        "KACHE_CACHE_DIR": str(work / "cache"),
        "KACHE_CONFIG": str(work / "kache.toml"),
        "KACHE_EVENT_ROOT": str(work / "project"),
        "KACHE_DAEMON_IDLE_TIMEOUT": "0",
        "KACHE_S3_ENDPOINT": KACHE_URL,
        "KACHE_S3_BUCKET": BUCKET,
        "KACHE_S3_REGION": "us-east-2",
        "KACHE_S3_PREFIX": "",
        "KACHE_S3_ACCESS_KEY": "tunnet-member",
        "KACHE_S3_SECRET_KEY": "tunnet-member",
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_REQUEST_CHECKSUM_CALCULATION": "when_required",
        "AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required",
    }


def run(
    argv: list[str],
    env: dict[str, str],
    cwd: Path,
    operation: str,
    *,
    timeout: int = 180,
) -> str:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ProbeError(f"The {operation} command did not finish.") from None
    require(result.returncode == 0, f"The {operation} command failed.")
    require(
        len(result.stdout) <= 5 * 1024 * 1024,
        "The command output exceeds the size limit.",
    )
    return result.stdout


def probe_expression(identity: str, python: str) -> str:
    require(
        re.fullmatch(r"[0-9]+-[0-9]+-[a-z0-9]+", identity) is not None,
        "The probe identity is invalid.",
    )
    require(
        re.fullmatch(STORE_PATH.pattern + r"/bin/python3(?:\.[0-9]+)?", python)
        is not None,
        "The probe builder is not store-owned Python.",
    )
    code = (
        "import os\nwith open(os.environ['out'], 'wb') as output:\n    output.write("
        + repr(identity.encode())
        + ")\n"
    )
    return (
        'builtins.derivation { name = "chelis-shared-runner-' + identity + '"; '
        'system = "x86_64-linux"; builder = builtins.storePath '
        + json.dumps(python)
        + "; "
        'args = [ "-I" "-S" "-c" ' + json.dumps(code) + " ]; }"
    )


def nix_probe(identity: str, env: dict[str, str], work: Path) -> dict:
    expression = probe_expression(
        identity, str(Path(sys.executable).resolve(strict=True))
    )
    path = run(
        [str(BIN / "nix-build"), "--no-out-link", "-E", expression],
        env,
        work,
        "Nix build",
    ).strip()
    require(
        STORE_PATH.fullmatch(path) is not None,
        "The Nix build returned an invalid probe path.",
    )
    before = NarRecord.parse(
        run([str(BIN / "nix"), "path-info", "--json", path], env, work, "Nix metadata"),
        path,
    )
    run(
        [str(BIN / "niks3"), "push", "--server-url", NIX_URL, "--tunnet", path],
        env,
        work,
        "Nix upload",
    )
    destination = work / "readback"
    destination.mkdir(mode=0o700)
    store = destination.as_uri()
    run(
        [
            str(BIN / "nix"),
            "copy",
            "--from",
            NIX_URL,
            "--to",
            store,
            "--option",
            "require-sigs",
            "true",
            "--option",
            "trusted-public-keys",
            PUBLIC_KEY,
            path,
        ],
        env,
        work,
        "Nix readback",
    )
    run(
        [
            str(BIN / "nix"),
            "store",
            "verify",
            "--store",
            store,
            "--sigs-needed",
            "1",
            "--option",
            "trusted-public-keys",
            PUBLIC_KEY,
            "--option",
            "substituters",
            "",
            "--option",
            "secret-key-files",
            "",
            path,
        ],
        env,
        work,
        "Nix signature verification",
    )
    after = NarRecord.parse(
        run(
            [str(BIN / "nix"), "path-info", "--store", store, "--json", path],
            env,
            work,
            "Nix readback metadata",
        ),
        path,
    )
    require_readback(before, after)
    return {
        "endpoint": NIX_URL,
        "path": path,
        "nar_hash": after.nar_hash,
        "without_bearer": True,
        "readback_hash_identical": True,
        "signature_checked_by_nix": True,
        "signing_key": SIGNING_KEY,
    }


def entry_digests(cache: Path) -> dict[str, str]:
    store = cache / "store"
    require(
        store.is_dir() and not store.is_symlink(), "The isolated cache store is absent."
    )
    entries = sorted(store.iterdir())
    require(0 < len(entries) <= 32, "The isolated cache entry count is invalid.")
    result = {}
    for entry in entries:
        require(
            re.fullmatch(r"[0-9a-f]{64}", entry.name) is not None
            and entry.is_dir()
            and not entry.is_symlink(),
            "The cache entry path is invalid.",
        )
        files = sorted(entry.rglob("*"))
        require(len(files) <= 256, "The cache entry exceeds the file limit.")
        digest = hashlib.sha256()
        count = 0
        for item in files:
            require(not item.is_symlink(), "The cache entry contains a symlink.")
            if item.is_dir():
                continue
            require(
                item.is_file() and item.stat().st_size <= 100 * 1024 * 1024,
                "The cache artifact is invalid.",
            )
            name = item.relative_to(entry).as_posix().encode()
            data = item.read_bytes()
            digest.update(len(name).to_bytes(8, "big") + name)
            digest.update(len(data).to_bytes(8, "big") + data)
            count += len(data)
        require(count > 0, "The cache entry is empty.")
        result[entry.name] = digest.hexdigest()
    return result


def local_hits(text: str, crate: str) -> int:
    value = document(text)
    require(
        isinstance(value, dict) and isinstance(value.get("summary"), dict),
        "The Kache report is invalid.",
    )
    summary = value["summary"]
    hits = summary.get("local_hits")
    require(
        type(hits) is int
        and hits > 0
        and all(
            type(summary.get(key)) is int and summary[key] == 0
            for key in ("errors", "store_failures")
        ),
        "The Kache report has no successful local hit.",
    )
    events = value.get("traceEvents")
    require(
        isinstance(events, list)
        and any(
            isinstance(item, dict) and item.get("name") == "hit: " + crate
            for item in events
        ),
        "The Kache report has no hit for the probe crate.",
    )
    return hits


class Daemon:
    """One probe-owned foreground daemon. Cleanup retains its process handle."""

    def __init__(self, env: dict[str, str], project: Path, work: Path):
        self.env, self.project, self.work = env, project, work
        self.process = None
        self.log = None

    def start(self) -> None:
        self.log = (self.work / "daemon.log").open("ab")
        self.process = subprocess.Popen(
            [str(BIN / "kache-daemon")],
            env=self.env,
            cwd=self.project,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        time.sleep(2)
        require(self.process.poll() is None, "The probe daemon did not start.")

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            run(
                [str(BIN / "kache"), "daemon", "stop"],
                self.env,
                self.project,
                "daemon stop",
                timeout=10,
            )
            self.process.wait(timeout=10)
            require(
                self.process.returncode == 0, "The probe daemon did not stop cleanly."
            )
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process = None
            self.log.close()
            self.log = None


def kache_config(*, local_only: bool) -> str:
    return (
        "[cache]\nignore_env = true\nlocal_only = "
        + str(local_only).lower()
        + '\n\n[cache.remote]\ntype = "s3"\nbucket = "'
        + BUCKET
        + '"\nregion = "us-east-2"\nprefix = ""\nendpoint = "'
        + KACHE_URL
        + '"\n'
    )


def kache_probe(identity: str, env: dict[str, str], work: Path) -> dict:
    crate = "shared_runner_" + identity.replace("-", "_")
    project, cache = work / "project", work / "cache"
    (project / "src").mkdir(parents=True, mode=0o700)
    (project / "Cargo.toml").write_text(
        f'[package]\nname = "{crate}"\nversion = "0.1.0"\nedition = "2021"\n[workspace]\n'
    )
    (project / "src/lib.rs").write_text(
        "pub fn probe() -> &'static str { \"" + identity + '" }\n'
    )
    (work / "kache.toml").write_text(kache_config(local_only=False))
    daemon = Daemon(env, project, work)
    try:
        daemon.start()
        run(
            [str(BIN / "cargo"), "build", "--offline", "--lib"],
            env,
            project,
            "first Rust build",
        )
        require((cache / "index.db").is_file(), "The cache index is not isolated.")
        before = entry_digests(cache)
        run([str(BIN / "kache"), "sync", "--push"], env, project, "Kache upload")
        daemon.stop()
        shutil.rmtree(cache)
        shutil.rmtree(work / "target")
        cache.mkdir(mode=0o700)
        run(
            [str(BIN / "kache"), "sync", "--pull", "--workspace"],
            env,
            project,
            "Kache readback",
        )
        require(entry_digests(cache) == before, "The restored cache bytes differ.")
        # Local-only compilation proves reuse of the pulled bytes, not a remote fallback.
        (work / "kache.toml").write_text(kache_config(local_only=True))
        daemon = Daemon(env, project, work)
        daemon.start()
        run(
            [str(BIN / "cargo"), "build", "--offline", "--lib"],
            env,
            project,
            "second Rust build",
        )
        hits = None
        for _attempt in range(50):
            try:
                report = run(
                    [
                        str(BIN / "kache"),
                        "report",
                        "--format",
                        "json",
                        "--root",
                        str(project),
                    ],
                    env,
                    project,
                    "Kache report",
                    timeout=5,
                )
                hits = local_hits(report, crate)
                break
            except ProbeError:
                time.sleep(0.1)
        require(hits is not None, "The Kache probe did not produce a local cache hit.")
        daemon.stop()
        return {
            "endpoint": KACHE_URL,
            "bucket": BUCKET,
            "crate": crate,
            "entry_digests": before,
            "without_aws_credentials": True,
            "workspace_only_pull": True,
            "restored_bytes_identical": True,
            "local_hits": hits,
            "remote_cleanup": "version-lifecycle",
        }
    finally:
        daemon.stop()


def probe(job: Job) -> dict:
    require(
        platform.system() == "Linux" and platform.machine() == "x86_64",
        "The canary requires the Linux x64 host.",
    )
    require(
        job.temp.is_dir() and job.temp.stat().st_uid == os.geteuid(),
        "The runner temporary directory has the wrong owner.",
    )
    for tool in (
        "nix",
        "nix-build",
        "niks3",
        "cargo",
        "rustc",
        "kache",
        "kache-daemon",
    ):
        require(os.access(BIN / tool, os.X_OK), "A reviewed host tool is absent.")
    host = host_identity()
    identity = f"{job.run_id}-{job.attempt}-{uuid4().hex}"
    with tempfile.TemporaryDirectory(
        prefix="shared-runner-", dir=job.temp
    ) as directory:
        work = Path(directory)
        for name in ("home", "config", "xdg-cache", "runtime", "cargo", "cache"):
            (work / name).mkdir(mode=0o700)
        env = child_environment(work)
        nix = nix_probe(identity, env, work)
        kache = kache_probe(identity, env, work)
        return {
            "schema": "chelis-shared-runner-canary/v1",
            "result": "pass",
            "access_model": "tunnet-membership/v1",
            "observed_at": datetime.now(UTC).isoformat(),
            "repository": "Chelis-Lang/chelis",
            "run_id": job.run_id,
            "run_attempt": job.attempt,
            "source_sha": job.sha,
            "host": {"instance_id": host.instance_id, "system": host.system},
            "probe_id": identity,
            "nix": nix,
            "kache": kache,
        }


def main() -> int:
    os.umask(0o077)
    try:
        require(len(sys.argv) == 1, "The canary accepts no arguments.")
        job = Job.parse(dict(os.environ))
        receipt = probe(job)
        encoded = json.dumps(receipt, indent=2, sort_keys=True)
        with (job.temp / "shared-runner-canary.json").open(
            "x", encoding="utf-8"
        ) as output:
            output.write(encoded + "\n")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with Path(summary).open("a", encoding="utf-8") as output:
                output.write(
                    "## Shared runner canary\n\n```json\n" + encoded + "\n```\n"
                )
        print(encoded)
        return 0
    except (ProbeError, OSError, subprocess.SubprocessError) as error:
        print(
            str(error)
            if isinstance(error, ProbeError)
            else "The canary failed without a safe diagnostic.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
