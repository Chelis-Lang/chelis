"""Install verified staged assets, then execute the shipped C-callable ABI.

This is an installation/packaging canary, not AD correctness, Hull agreement,
School adoption or full replacement acceptance. Source identity is supplied by
the build workflow; hashes bind bytes, not their source-level authenticity.
Candidate archives retain their dev-SHA identity while an explicitly recorded
payload-identical root rename exercises chelisup's concrete-version seam.
No installer behavior, real toolchain home or global default is changed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tomllib

if __package__:
    from .verify_runtime_package import verify_runtime_package
else:
    from verify_runtime_package import verify_runtime_package


ROOT = Path(__file__).resolve().parents[1]
HEADERS = tuple(f"include/{name}.h" for name in (
    "chelis_runtime", "chelis_runtime_views", "chelis_runtime_dtype",
    "chelis_blas", "chelis_simd", "chelis_math",
))
REQUIRED_FILES = ("bin/chelis", "lib/libchelis_runtime.a", *HEADERS)
REQUIRED_PROCESSES = ("install", "which", "version", "runtime-export",
                      "unavailable-root", "build", "link", "execute-0",
                      "execute-1", "execute-2", "link-last-coordinate",
                      "reject-last-coordinate", "link-input", "reject-input")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_sidecar(path: Path) -> str:
    expected = digest(path)
    text = Path(str(path) + ".sha256").read_text(encoding="utf-8")
    if text not in (f"{expected}  {path.name}\n", f"{expected} *{path.name}\n"):
        raise ValueError(f"checksum/filename mismatch: {path}")
    return expected


# The installer slug and the release build `chelisup install` downloads on each
# canary host. On Linux that is the static build.
PLATFORMS = {("Linux", "x86_64"): ("linux-x86_64", "linux-x86_64-static"),
             ("Darwin", "arm64"): ("darwin-arm64", "darwin-arm64")}


def archive_name(version: str, label: str, source: str, build: str) -> str:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("expected concrete X.Y.Z version")
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("expected full source commit")
    if build not in {known for _, known in PLATFORMS.values()}:
        raise ValueError("unsupported installed-canary build")
    if label not in (f"v{version}", f"dev-{source[:8]}"):
        raise ValueError("build label does not match version/source")
    return f"chelis-{label}-{build}.tar.gz"


def archive_inventory(path: Path) -> dict[str, str]:
    """Read without extraction; reject ambiguous or escaping archive members."""
    inventory: dict[str, str] = {}
    seen: set[str] = set()
    roots: set[str] = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not name.parts:
                raise ValueError("unsafe archive path")
            normalized = str(name)
            if normalized in seen or not (member.isfile() or member.isdir()):
                raise ValueError("duplicate/link/special archive member")
            seen.add(normalized)
            roots.add(name.parts[0])
            if member.isfile():
                if len(name.parts) < 2:
                    raise ValueError("archive file outside staging root")
                stream = archive.extractfile(member)
                assert stream is not None
                with stream:
                    inventory[str(PurePosixPath(*name.parts[1:]))] = (
                        hashlib.file_digest(stream, "sha256").hexdigest())
    if len(roots) != 1:
        raise ValueError("expected exactly one archive staging root")
    return inventory


def require_runtime_inventory(inventory: dict[str, str]) -> None:
    if missing := set(REQUIRED_FILES) - inventory.keys():
        raise ValueError(f"archive missing required runtime files: {sorted(missing)}")


def verify_installed(root: Path, inventory: dict[str, str]) -> None:
    require_runtime_inventory(inventory)
    for relative, expected in inventory.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError(f"installed bytes differ: {relative}")


def require_staged_runtime(output: Path, inventory: dict[str, str]) -> None:
    """Require `chelis build` to stage the shipped runtime from a sealed build.

    The staged runtime is the one the installed chelis carries, and the tarball
    must ship those same bytes (spec/08-backends.md §2.1). Its receipt must
    bind all six staged public headers as well as the archive to those bytes.
    A release binary must also be sealed: a development build would stage here
    too, because this runner holds the checkout it was built from, so only the
    staging receipt's mode tells them apart.
    """
    archive = inventory["lib/libchelis_runtime.a"]
    if digest(output / "libchelis_runtime.a") != archive:
        raise ValueError("generated build used a different runtime archive")
    for header in HEADERS:
        if digest(output / Path(header).name) != inventory[header]:
            raise ValueError(f"embedded/shipped header mismatch: {header}")
    try:
        receipt = json.loads((output / "chelis_runtime.receipt.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable staging receipt: {error}") from None
    expected = {"schema": "chelis-runtime-staging/1", "archive": "libchelis_runtime.a",
                "archive_sha256": archive, "mode": "sealed"}
    observed = {key: receipt.get(key) for key in expected} if isinstance(receipt, dict) else None
    if observed != expected:
        raise ValueError(f"staging receipt is not the shipped sealed runtime: {observed}")
    expected_headers = {Path(header).name: inventory[header] for header in HEADERS}
    if receipt.get("headers") != expected_headers:
        raise ValueError("staging receipt headers do not match the shipped and staged headers")



def installer_archive(source: Path, destination: Path, *, candidate: bool,
                      version: str, build: str) -> None:
    """chelisup requires a chelis-v* archive root, even for local candidates.

    Only a dev archive gets a new container. Published release bytes are copied
    unchanged. In both cases installation must match the ORIGINAL member census.
    """
    inventory = archive_inventory(source)
    if candidate:
        with tarfile.open(source, "r:gz") as incoming, tarfile.open(destination, "w:gz") as outgoing:
            for original in incoming:
                member = copy.copy(original)
                suffix = PurePosixPath(*PurePosixPath(member.name).parts[1:])
                member.name = str(PurePosixPath(f"chelis-v{version}-{build}") / suffix)
                member.pax_headers = {key: value for key, value in member.pax_headers.items()
                                      if key != "path"}
                stream = incoming.extractfile(original) if member.isfile() else None
                try:
                    outgoing.addfile(member, stream)
                finally:
                    if stream is not None:
                        stream.close()
    else:
        shutil.copyfile(source, destination)
        if digest(source) != digest(destination):
            raise ValueError("release archive changed while copying")
    if archive_inventory(destination) != inventory:
        raise ValueError("installer mapping changed member payloads")


class Runner:
    """Retain real process outcomes, including failed launch and timeout."""

    def __init__(self, root: Path, env: dict[str, str], records: list[dict]):
        self.root, self.env, self.records = root, env, records

    def run(self, name: str, command: list[object], *, timeout: float = 30,
            expected_failure: str | None = None) -> subprocess.CompletedProcess:
        argv = [str(part) for part in command]
        record = {"name": name, "argv": argv, "cwd": str(self.root),
                  "timeout_seconds": timeout, "status": "failed",
                  "stdout": f"{name}.stdout", "stderr": f"{name}.stderr"}
        self.records.append(record)
        print(f"+ {name}: {' '.join(argv)}", flush=True)
        with (self.root / record["stdout"]).open("wb") as stdout, (
            self.root / record["stderr"]).open("wb") as stderr:
            try:
                process = subprocess.Popen(argv, cwd=self.root, env=self.env,
                                           stdout=stdout, stderr=stderr,
                                           start_new_session=True)
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    record["termination"] = "timeout"
                    raise ValueError(f"{name}: timeout") from None
            except OSError as exc:
                record["termination"] = "launch-error"
                raise ValueError(f"{name}: {exc}") from exc
        record["returncode"] = code
        out = (self.root / record["stdout"]).read_text(encoding="utf-8")
        err = (self.root / record["stderr"]).read_text(encoding="utf-8")
        if expected_failure is None:
            accepted = code == 0
        else:
            accepted = code == 1 and expected_failure in err
        if not accepted:
            raise ValueError(f"{name}: unexpected exit {code}; see retained logs")
        record["status"] = "passed"
        return subprocess.CompletedProcess(argv, code, out, err)


def fixture_module():
    path = ROOT / ".github/scripts/smoke_macos_manifested_callable.py"
    spec = importlib.util.spec_from_file_location("manifested_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corrupt_driver(driver: str, target: str) -> str:
    """Inject a caller-side fault after execution, not a source-language rule."""
    if target not in ("last-coordinate", "input"):
        raise ValueError("unknown native control")
    marker = "    chelis_read_view output_view = chelis_tensor_read_view(outputs[0]);\n"
    if driver.count(marker) != 1:
        raise ValueError("native control lost its exact insertion point")
    tensor, index = ("outputs[0]", 7) if target == "last-coordinate" else ("a", 0)
    injection = f"""    chelis_tensor_write *fault_guard = chelis_tensor_begin_write({tensor});
    chelis_write_view fault_view = chelis_tensor_write_view(fault_guard);
    uint32_t fault_bits = 0;
    memcpy((char *)fault_view.data + {index} * sizeof(fault_bits), &fault_bits, sizeof(fault_bits));
    chelis_tensor_end_write(fault_guard);
"""
    return driver.replace(marker, injection + marker)


LINK_REQUIREMENTS = "Link requirements (after module archive): "
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


def reported_link_flags(build_stdout: str) -> list[str]:
    """The flags `chelis build --emit-c` tells a library consumer to link after the
    runtime archive. The canary links exactly as that guidance says, so on glibc
    before 2.34 it gets `-lpthread -ldl` alongside `-lm`. The line may open with
    variable assignments, the SDK a macOS build gives its tools; the canary links
    with the host's `cc`, whose `/usr/bin` shim selects that same SDK."""
    lines = [line for line in build_stdout.splitlines() if line.startswith(LINK_REQUIREMENTS)]
    if len(lines) != 1:
        raise ValueError("build did not report one link-requirements line")
    words = shlex.split(lines[0][len(LINK_REQUIREMENTS):])
    while words and ASSIGNMENT.match(words[0]):
        words.pop(0)
    if len(words) < 2 or not words[1].endswith("libchelis_runtime.a"):
        raise ValueError("link requirements do not name the runtime archive")
    return words[2:]


def execute(args: argparse.Namespace, report: dict) -> None:
    root = args.evidence
    slug, build = PLATFORMS.get((platform.system(), platform.machine()),
                                ("unsupported", "unsupported"))
    name = archive_name(args.version, args.build_label, args.source_sha, build)
    archive = args.artifacts / name
    installer = args.installer_assets / f"chelisup-{slug}"
    archive_hash, installer_hash = verify_sidecar(archive), verify_sidecar(installer)
    inventory = archive_inventory(archive)
    require_runtime_inventory(inventory)
    report.update(platform=slug, version=args.version, source_sha=args.source_sha,
                  build_label=args.build_label, archive={"path": str(archive),
                  "sha256": archive_hash}, installer={"path": str(installer),
                  "sha256": installer_hash}, archive_files=inventory)
    # Copy verified assets into this run's new directory, never a shared store.
    local_assets = root / "assets"
    local_assets.mkdir()
    mapped = local_assets / f"chelis-v{args.version}-{build}.tar.gz"
    candidate = args.build_label.startswith("dev-")
    installer_archive(archive, mapped, candidate=candidate, version=args.version, build=build)
    report["installer_mapping"] = {"from": archive.name, "to": mapped.name,
                                   "kind": "root-rename-only" if candidate else "byte-identical",
                                   "original_sha256": archive_hash,
                                   "mapped_sha256": digest(mapped)}
    local_installer = root / "chelisup"
    shutil.copyfile(installer, local_installer)
    local_installer.chmod(0o755)
    if digest(local_installer) != installer_hash:
        raise ValueError("installer changed while copying")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CHELIS_", "CHELISUP_"))}
    env.update(CHELIS_HOME=str(root / "home"),
               CHELISUP_RELEASE_BASE=str(local_assets))
    runner = Runner(root, env, report["processes"])
    runner.run("install", [local_installer, "install", args.version], timeout=60)
    installed = root / "home/toolchains" / args.version
    verify_installed(installed, inventory)
    shim = root / "home/bin/chelis"
    for path in (shim, root / "home/bin/chelisup"):
        if digest(path) != installer_hash:
            raise ValueError("installed shim is not the supplied installer")
    (root / "chelis-toolchain").write_text(args.version + "\n")
    selected = runner.run("which", [local_installer, "which"])
    if Path(selected.stdout.strip()).resolve() != (installed / "bin/chelis").resolve():
        raise ValueError("shim resolved an unexpected toolchain")
    version = runner.run("version", [shim, "--version"])
    if version.stdout.strip() != f"chelis {args.version}":
        raise ValueError("installed binary has an unexpected version")
    exported = root / "runtime-export"
    runner.run("runtime-export", [shim, "runtime", "export", exported])
    report["installed_export_sha256"] = verify_runtime_package(exported, installed)
    if report["installed_export_sha256"] != inventory["lib/libchelis_runtime.a"]:
        raise ValueError("installed compiler export differs from the shipped archive")
    fixture = fixture_module()
    negative = root / "self_bound.ch"
    negative.write_text(fixture.SELF_BOUND_PROGRAM)
    runner.run("unavailable-root", [shim, "build", negative, "--output", root / "negative"],
               expected_failure="[05-UNS-1] unavailable root `a`")
    if (root / "negative").exists():
        raise ValueError("negative left a partial output artifact")
    source = root / "manifested_callable.ch"
    source.write_text(fixture.ADD_PROGRAM)
    output = root / "generated"
    built = runner.run("build", [shim, "build", "--emit-c", source, "--output", output])
    link_flags = reported_link_flags(built.stdout)
    require_staged_runtime(output, inventory)
    # Quoted includes search the source directory first. Move only generated C
    # into a header-free directory, then explicitly use shipped include/lib.
    consumer = root / "consumer"
    consumer.mkdir()
    shutil.copyfile(output / "manifested_callable.c", consumer / "callable.c")
    (consumer / "driver.c").write_text(fixture.DRIVER_C)
    compiler = shutil.which("cc")
    if compiler is None:
        raise ValueError("host C compiler unavailable")
    runner.run("link", [compiler, "-O2", "-I", installed / "include",
                        consumer / "callable.c", consumer / "driver.c",
                        installed / "lib/libchelis_runtime.a", *link_flags,
                        "-o", consumer / "canary"])
    for call in range(3):
        runner.run(f"execute-{call}", [consumer / "canary"])
    for target, diagnostic in (("last-coordinate", "element 7: got"),
                               ("input", "changed a borrowed input")):
        driver = consumer / f"{target}.c"
        binary = consumer / target
        driver.write_text(corrupt_driver(fixture.DRIVER_C, target))
        runner.run(f"link-{target}", [compiler, "-O2", "-I", installed / "include",
                    consumer / "callable.c", driver,
                    installed / "lib/libchelis_runtime.a", *link_flags, "-o", binary])
        runner.run(f"reject-{target}", [binary], expected_failure=diagnostic)
    verify_installed(installed, inventory)
    if verify_sidecar(archive) != archive_hash or verify_sidecar(installer) != installer_hash:
        raise ValueError("input artifacts changed during execution")
    report["evidence_files"] = {str(path.relative_to(root)): digest(path)
                                for path in sorted(root.rglob("*"))
                                if path.is_file() and "home" not in path.relative_to(root).parts
                                and "assets" not in path.relative_to(root).parts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--installer-assets", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--build-label")
    parser.add_argument("--version")
    parser.add_argument("--evidence", type=Path, required=True,
                        help="new directory; existing evidence is never overwritten")
    args = parser.parse_args()
    args.version = args.version or tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    args.build_label = args.build_label or (os.environ["GITHUB_REF_NAME"]
        if os.environ.get("GITHUB_REF_TYPE") == "tag" else f"dev-{args.source_sha[:8]}")
    args.evidence = args.evidence.resolve()
    args.artifacts = args.artifacts.resolve()
    args.installer_assets = args.installer_assets.resolve()
    args.evidence.mkdir(parents=True, exist_ok=False)
    report = {"schema": "chelis-installed-callable-smoke-v1", "status": "failed",
              "replacement_acceptance": False, "processes": [],
              "required_processes": list(REQUIRED_PROCESSES),
              "workflow_run": os.environ.get("GITHUB_RUN_ID"),
              "workflow_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")}
    try:
        if os.environ.get("GITHUB_SHA", args.source_sha) != args.source_sha:
            raise ValueError("workflow/source identity mismatch")
        execute(args, report)
        report["status"] = "passed"
        return 0
    except (ValueError, OSError, tarfile.TarError) as exc:
        report["error"] = str(exc)
        print(f"installed-artifact canary failed: {exc}", file=sys.stderr)
        return 1
    finally:
        completed = {record["name"] for record in report["processes"]}
        report["not_run"] = [name for name in REQUIRED_PROCESSES if name not in completed]
        (args.evidence / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
