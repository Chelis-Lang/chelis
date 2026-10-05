#!/usr/bin/env python3
"""Check that a Chelis release installs and runs on this host.

`run` checks one host against one release's assets and writes `report.json`
and every process log into a new evidence directory. A published release is
checked the way users get it. An unpublished candidate from a `release.yml`
run, whose archives carry a `dev-<sha8>` label, is installed from a copy under
its release name, and the two online steps skip.

- assets: the sha256 sidecars of this host's archive and installer match.
- bootstrap: the install guide's `gh release download ... chelisup.sh | sh`,
  which takes chelisup from the latest release (online runs with `gh` on PATH).
- install-online: `chelisup install X.Y.Z` through the GitHub API.
- install-offline: `chelisup install X.Y.Z` from the verified local assets.
- first-program: every command of the release's own First Program page, as
  written, against the program that page defines.
- eval: the evaluator's numbers for a softmax probe.
- native: a scalar program built with `chelis build` and the host's C
  compiler, then run.
- smt: the release's cvc5 discharge gate against the installed compiler.
- canary: the release's installed-artifact canary, which installs, builds,
  links, and runs the shipped C callable.
- linkage: the installed binaries' loader and libraries; a static Linux build
  names neither.

The gates come from the source that produced the release, so a run checks what
the release claimed when it shipped. The evidence directory keeps `report.json`,
`logs/`, and the canary's report and logs in `canary/`; `work/` holds the
installed toolchains and other scratch files, which uploads leave out.
`summarize` renders every report under a directory as one Markdown table.
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        "release_e2e.py needs Python 3.11 or newer (the canary uses tomllib)"
    )

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import tarfile
import time

from installed_artifact_canary import digest, installer_archive, verify_sidecar

REPO = "Chelis-Lang/chelis"
SCHEMA = "chelis-release-e2e/v1"
TAG = re.compile(r"v([0-9]+\.[0-9]+\.[0-9]+)")
SOURCE_SHA = re.compile(r"[0-9a-f]{40}")
# The installer slug and the toolchain builds chelisup installs on each host,
# in preference order: a Linux release without a static build installs its
# glibc-2.31 build.
PLATFORMS = {
    ("Linux", "x86_64"): (
        "linux-x86_64",
        ("linux-x86_64-static", "linux-x86_64-glibc2.31"),
    ),
    ("Darwin", "arm64"): ("darwin-arm64", ("darwin-arm64",)),
}
FIRST_PROGRAM_PAGE = Path("docs/book/src/first-program.md")
FENCE = re.compile(r"```([A-Za-z0-9_-]*)")
EVAL_PROBE = (
    "def relu_then_softmax[n](x: tensor[n, f32]) -> tensor[n, f32]"
    " = x |> relu |> softmax(0)\n"
    "result = [-1.0, 0.0, 1.0] |> to_tensor |> relu_then_softmax\n"
)
# softmax(relu([-1, 0, 1])) = [1, 1, e] / (2 + e)
EVAL_RESULT = (0.21194, 0.21194, 0.57612)
NATIVE_PROBE = "def half_sum(a: f32, b: f32) -> f32 = ((a + b) * 0.5)\nresult = half_sum(1.5, 2.25)\n"
NATIVE_RESULT = re.compile(r"result\s*=\s*1\.875\b")
FLOAT = re.compile(r"-?[0-9]+\.[0-9]+(?:[eE][-+]?[0-9]+)?")
MACOS_SYSTEM_PREFIXES = ("/usr/lib/", "/System/Library/")
LINUX_LOADER = "/lib64/ld-linux-x86-64.so.2"
STEPS = (
    "assets",
    "bootstrap",
    "install-online",
    "install-offline",
    "first-program",
    "eval",
    "native",
    "smt",
    "canary",
    "linkage",
)
FAILING = frozenset({"failed", "blocked"})


class StepFailed(Exception):
    """The step did not meet its contract."""


class StepSkipped(Exception):
    """The step does not apply to this host or release."""


@dataclass
class Step:
    name: str
    status: str = "not-run"
    detail: str = ""
    seconds: float = 0.0
    processes: list[dict[str, object]] = field(default_factory=list)


def tail(text: str, lines: int = 6) -> str:
    kept = [line.strip() for line in text.splitlines() if line.strip()][-lines:]
    joined = " | ".join(kept)
    return joined if len(joined) <= 600 else "..." + joined[-600:]


def probe_result(text: str) -> tuple[float, ...] | None:
    values = [float(match) for match in FLOAT.findall(text)]
    for start in range(len(values) - len(EVAL_RESULT) + 1):
        window = values[start : start + len(EVAL_RESULT)]
        if all(
            abs(seen - want) < 1e-3
            for seen, want in zip(window, EVAL_RESULT, strict=True)
        ):
            return tuple(window)
    return None


def walkthrough(markdown: str) -> tuple[str | None, list[str]]:
    """The first `chelis-surf` block and every `sh` command line, in order."""
    program: str | None = None
    commands: list[str] = []
    language: str | None = None
    block: list[str] = []
    for line in markdown.splitlines():
        fence = FENCE.fullmatch(line.strip())
        if language is None:
            if fence is not None:
                language, block = fence.group(1), []
            continue
        if line.strip() != "```":
            block.append(line)
            continue
        if language == "chelis-surf" and program is None:
            program = "\n".join(block) + "\n"
        elif language == "sh":
            pending = ""
            for entry in block:
                text = entry.strip()
                if text.endswith("\\"):
                    pending += text[:-1].rstrip() + " "
                elif pending or (text and not text.startswith("#")):
                    commands.append(pending + text)
                    pending = ""
        language = None
    return program, commands


def overall(steps: Iterable[Step]) -> str:
    """A run passes only when no step failed or was blocked by a failure."""
    return "failed" if any(step.status in FAILING for step in steps) else "passed"


@dataclass
class ElfLinkage:
    interpreter: str | None
    needed: list[str]
    rpath: list[str]


def elf_dynamic(path: Path) -> ElfLinkage:
    """Read PT_INTERP, DT_NEEDED, and DT_RPATH/DT_RUNPATH of an x86-64 ELF."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1:
            raise StepFailed(f"{path.name} is not a 64-bit little-endian ELF")
        (phoff,) = struct.unpack_from("<Q", header, 0x20)
        phentsize, phnum = struct.unpack_from("<HH", header, 0x36)
        stream.seek(phoff)
        table = stream.read(phentsize * phnum)
        loads: list[tuple[int, int, int]] = []
        interpreter = None
        dynamic = None
        for index in range(phnum):
            kind, _, offset, vaddr, _, filesz, _, _ = struct.unpack_from(
                "<IIQQQQQQ", table, index * phentsize
            )
            if kind == 1:
                loads.append((vaddr, offset, filesz))
            elif kind == 2:
                dynamic = (offset, filesz)
            elif kind == 3:
                stream.seek(offset)
                interpreter = stream.read(filesz).rstrip(b"\0").decode()
        entries: list[tuple[int, int]] = []
        if dynamic is not None:
            stream.seek(dynamic[0])
            raw = stream.read(dynamic[1])
            for offset in range(0, len(raw) - 15, 16):
                tag, value = struct.unpack_from("<qQ", raw, offset)
                if tag == 0:
                    break
                entries.append((tag, value))
        strtab = next((value for tag, value in entries if tag == 5), None)

        def string(index: int) -> str:
            if strtab is None:
                raise StepFailed(f"{path.name} has no dynamic string table")
            for vaddr, offset, size in loads:
                if vaddr <= strtab < vaddr + size:
                    stream.seek(offset + strtab - vaddr + index)
                    return stream.read(4096).split(b"\0", 1)[0].decode()
            raise StepFailed(f"{path.name} string table is outside its segments")

        return ElfLinkage(
            interpreter=interpreter,
            needed=[string(value) for tag, value in entries if tag == 1],
            rpath=[string(value) for tag, value in entries if tag in (15, 29)],
        )


def elf_verdict(name: str, linkage: ElfLinkage, *, static: bool) -> str:
    """A static build names no loader or library; a dynamic one, the system loader."""
    if static:
        if linkage.interpreter is not None or linkage.needed:
            raise StepFailed(
                f"{name} is not static: interpreter {linkage.interpreter}, needs {linkage.needed}"
            )
        return f"{name}: static, no program interpreter or shared library"
    if linkage.interpreter != LINUX_LOADER:
        raise StepFailed(f"{name} loads through {linkage.interpreter}")
    stored = [entry for entry in linkage.rpath if "/nix/store" in entry]
    if stored:
        raise StepFailed(f"{name} has store run paths {stored}")
    return f"{name}: {LINUX_LOADER}, needs {', '.join(linkage.needed)}"


def macho_verdict(name: str, otool_libraries: str) -> str:
    """A macOS build loads only system libraries, given its `otool -L` output."""
    libraries = [
        line.strip().split(" (", 1)[0]
        for line in otool_libraries.splitlines()[1:]
        if line.strip()
    ]
    outside = [
        library
        for library in libraries
        if not library.startswith(MACOS_SYSTEM_PREFIXES)
    ]
    if outside:
        raise StepFailed(f"{name} loads {', '.join(outside)}")
    return f"{name}: {len(libraries)} system libraries"


def glibc_version() -> str | None:
    try:
        return os.confstr("CS_GNU_LIBC_VERSION")
    except (OSError, ValueError):
        return None


def host_facts() -> dict[str, object]:
    system, machine = platform.system(), platform.machine()
    facts: dict[str, object] = {
        "system": system,
        "machine": machine,
        "python": platform.python_version(),
    }
    if system == "Darwin":
        facts["os"] = f"macOS {platform.mac_ver()[0]}"
    else:
        release = Path("/etc/os-release")
        pretty = None
        if release.is_file():
            for line in release.read_text(encoding="utf-8").splitlines():
                if line.startswith("PRETTY_NAME="):
                    pretty = line.split("=", 1)[1].strip().strip('"')
        facts["os"] = pretty or f"{system} {platform.release()}"
        facts["libc"] = glibc_version()
        facts["nixos"] = Path("/etc/NIXOS").exists()
        facts["nix_ld"] = Path("/run/current-system/sw/share/nix-ld").exists()
    compiler = shutil.which("cc")
    facts["cc"] = compiler
    if compiler is not None:
        probe = subprocess.run(
            [compiler, "--version"], capture_output=True, text=True, check=False
        )
        banner = (probe.stdout or probe.stderr).strip().splitlines()
        facts["cc_version"] = banner[0] if banner else None
    facts["gh"] = shutil.which("gh") is not None
    return facts


class Proof:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.evidence: Path = args.evidence
        self.logs = self.evidence / "logs"
        self.work = self.evidence / "work"
        self.counter = 0
        self.steps: dict[str, Step] = {}
        self.current = Step("setup")
        self.version = args.version
        self.slug, self.builds = PLATFORMS.get(
            (platform.system(), platform.machine()), (None, ())
        )
        self.build: str | None = None
        self.candidate = args.archive_label != args.tag
        self.install_assets: Path = args.assets
        self.glibc_host = platform.system() != "Linux" or glibc_version() is not None
        self.offline_home: Path | None = None

    def env(self, **extra: str | Path) -> dict[str, str]:
        """The caller's environment without Chelis selection or CI identity."""
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("CHELIS_", "CHELISUP_")) and key != "GITHUB_SHA"
        }
        environment.update({key: str(value) for key, value in extra.items()})
        return environment

    def run(
        self,
        label: str,
        argv: Sequence[object],
        *,
        env: dict[str, str],
        cwd: Path | None = None,
        timeout: float = 300,
        stdin: bytes | None = None,
        check: bool = True,
    ) -> tuple[int, str, str]:
        self.counter += 1
        stem = f"{self.counter:02d}-{self.current.name}-{label}"
        command = [str(part) for part in argv]
        record: dict[str, object] = {
            "label": label,
            "argv": command,
            "cwd": str(cwd or self.work),
            "log": stem,
        }
        self.current.processes.append(record)
        print(f"+ {self.current.name}/{label}: {' '.join(command)}", flush=True)
        try:
            completed = subprocess.run(
                command,
                cwd=cwd or self.work,
                env=env,
                input=stdin,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            record["termination"] = "timeout"
            self._write(stem, error.stdout or b"", error.stderr or b"")
            raise StepFailed(f"{label} timed out after {timeout:.0f}s") from None
        except OSError as error:
            record["termination"] = "launch-error"
            raise StepFailed(f"{label} could not start {command[0]}: {error}") from None
        stdout, stderr = self._write(stem, completed.stdout, completed.stderr)
        record["returncode"] = completed.returncode
        if check and completed.returncode != 0:
            raise StepFailed(
                f"{label} exited {completed.returncode}: {tail(stderr or stdout)}"
            )
        return completed.returncode, stdout, stderr

    def _write(self, stem: str, stdout: bytes, stderr: bytes) -> tuple[str, str]:
        (self.logs / f"{stem}.out").write_bytes(stdout)
        (self.logs / f"{stem}.err").write_bytes(stderr)
        return stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")

    def archive(self, build: str) -> Path:
        return self.args.assets / f"chelis-{self.args.archive_label}-{build}.tar.gz"

    def copy_installer(self, directory: Path) -> Path:
        source = self.args.assets / f"chelisup-{self.slug}"
        expected = verify_sidecar(source)
        directory.mkdir()
        installer = directory / "chelisup"
        shutil.copyfile(source, installer)
        installer.chmod(0o755)
        if digest(installer) != expected:
            raise StepFailed("the installer changed while copying")
        return installer

    def require_version(self, home: Path) -> str:
        shim = home / "bin" / "chelis"
        _, stdout, _ = self.run(
            "version", [shim, "--version"], env=self.env(CHELIS_HOME=home)
        )
        if stdout.strip() != f"chelis {self.version}":
            raise StepFailed(f"the installed shim reports {stdout.strip()!r}")
        return f"`chelis --version` prints {stdout.strip()!r}"

    def require_offline_home(self) -> Path:
        if self.offline_home is None:
            raise StepFailed("install-offline did not complete")
        return self.offline_home

    def require_glibc_host(self) -> None:
        if not self.glibc_host:
            raise StepSkipped(
                "the runtime archive is a glibc library and this host's C library is not glibc"
            )

    def assets(self) -> str:
        if self.slug is None:
            raise StepFailed(
                f"no release build exists for {platform.system()}/{platform.machine()}"
            )
        self.build = next(
            (build for build in self.builds if self.archive(build).is_file()), None
        )
        if self.build is None:
            listed = ", ".join(self.archive(build).name for build in self.builds)
            raise StepFailed(f"the release has none of {listed}")
        archive = self.archive(self.build)
        names = [archive, self.args.assets / f"chelisup-{self.slug}"]
        for path in names:
            verify_sidecar(path)
        if not (self.args.assets / "chelisup.sh").is_file():
            raise StepFailed("the release has no chelisup.sh")
        mapping = ""
        if self.candidate:
            # chelisup installs only `chelis-vX.Y.Z-<build>` archives, so a
            # candidate installs from a copy under that name, payload unchanged.
            self.install_assets = self.work / "release-assets"
            self.install_assets.mkdir()
            mapped = self.install_assets / f"chelis-v{self.version}-{self.build}.tar.gz"
            installer_archive(
                archive, mapped, candidate=True, version=self.version, build=self.build
            )
            mapping = f"; installs {archive.name} as {mapped.name}"
        return (
            f"{self.build}: sidecars match "
            + ", ".join(path.name for path in names)
            + mapping
        )

    def bootstrap(self) -> str:
        if not self.args.online:
            raise StepSkipped("offline run")
        if self.candidate:
            raise StepSkipped("unpublished candidate")
        gh = shutil.which("gh")
        if gh is None:
            raise StepSkipped("gh is not on PATH")
        home = self.work / "online-home"
        env = self.env(CHELIS_HOME=home)
        _, script, _ = self.run(
            "download-chelisup.sh",
            [
                gh,
                "release",
                "download",
                "--repo",
                REPO,
                "--pattern",
                "chelisup.sh",
                "--output",
                "-",
            ],
            env=env,
            timeout=120,
        )
        self.run("sh", ["sh"], env=env, stdin=script.encode(), timeout=300)
        installed = home / "bin" / "chelisup"
        if not installed.is_file() or not os.access(installed, os.X_OK):
            raise StepFailed("the bootstrap did not install an executable chelisup")
        _, stdout, _ = self.run("chelisup-version", [installed, "--version"], env=env)
        return f"the latest release's chelisup.sh installed {stdout.strip()}"

    def install_online(self) -> str:
        if not self.args.online:
            raise StepSkipped("offline run")
        if self.candidate:
            raise StepSkipped("unpublished candidate")
        home = self.work / "online-home"
        installer = home / "bin" / "chelisup"
        if not installer.is_file():
            installer = self.copy_installer(self.work / "online-installer")
        _, stdout, _ = self.run(
            "install",
            [installer, "install", self.version],
            env=self.env(CHELIS_HOME=home),
            timeout=900,
        )
        installed = next(
            (
                line
                for line in stdout.splitlines()
                if line.startswith("installed chelis")
            ),
            "chelisup printed no install line",
        )
        return f"from the GitHub API: {installed}; {self.require_version(home)}"

    def install_offline(self) -> str:
        home = self.work / "offline-home"
        installer = self.copy_installer(self.work / "offline-installer")
        self.run(
            "install",
            [installer, "install", self.version],
            env=self.env(CHELIS_HOME=home, CHELISUP_RELEASE_BASE=self.install_assets),
            timeout=600,
        )
        detail = "from local assets; " + self.require_version(home)
        self.offline_home = home
        return detail

    def first_program(self) -> str:
        home = self.require_offline_home()
        self.require_glibc_host()
        page = self.args.gates / FIRST_PROGRAM_PAGE
        program, commands = walkthrough(page.read_text(encoding="utf-8"))
        if program is None or not commands:
            raise StepFailed(f"{FIRST_PROGRAM_PAGE} defines no program or commands")
        work = self.work / "first-program"
        work.mkdir()
        (work / "app.ch").write_text(program, encoding="utf-8")
        env = self.env(
            CHELIS_HOME=home, PATH=f"{home / 'bin'}{os.pathsep}{os.environ['PATH']}"
        )
        for index, command in enumerate(commands, start=1):
            self.run(
                f"doc-{index}", ["sh", "-c", command], env=env, cwd=work, timeout=600
            )
        return f"{len(commands)} documented commands succeeded: " + "; ".join(commands)

    def eval(self) -> str:
        home = self.require_offline_home()
        work = self.work / "eval"
        work.mkdir()
        (work / "probe.ch").write_text(EVAL_PROBE, encoding="utf-8")
        shim = home / "bin" / "chelis"
        _, evaluated, _ = self.run(
            "eval",
            [shim, "eval", "--file", "probe.ch"],
            env=self.env(CHELIS_HOME=home),
            cwd=work,
        )
        result = probe_result(evaluated)
        if result is None:
            raise StepFailed(f"eval did not print {EVAL_RESULT}: {tail(evaluated)}")
        return f"softmax(relu([-1, 0, 1])) = {list(result)}"

    def native(self) -> str:
        home = self.require_offline_home()
        self.require_glibc_host()
        work = self.work / "native"
        work.mkdir()
        (work / "app.ch").write_text(NATIVE_PROBE, encoding="utf-8")
        shim = home / "bin" / "chelis"
        env = self.env(CHELIS_HOME=home)
        self.run("fmt", [shim, "fmt", "--inplace", "app.ch"], env=env, cwd=work)
        _, built, _ = self.run(
            "build",
            [shim, "build", "app.ch", "--output", "out/"],
            env=env,
            cwd=work,
            timeout=600,
        )
        program = work / "out" / "app"
        how = "`chelis build` linked out/app"
        if not program.is_file():
            # 0.18.x releases print the compile command instead of running it.
            printed = [
                line.removeprefix("Compile: ")
                for line in built.splitlines()
                if line.startswith("Compile: ")
            ]
            if not printed:
                raise StepFailed(
                    f"build wrote no out/app and no command: {tail(built)}"
                )
            self.run(
                "compile", ["sh", "-c", printed[0]], env=env, cwd=work, timeout=600
            )
            how = f"ran the printed `{printed[0]}`"
        _, output, _ = self.run("out-app", [program], env=env, cwd=work)
        if NATIVE_RESULT.search(output) is None:
            raise StepFailed(f"out/app printed {tail(output)!r}, not result = 1.875")
        return f"{how}; out/app prints {output.strip()!r}"

    def smt(self) -> str:
        home = self.require_offline_home()
        chelis = home / "toolchains" / self.version / "bin" / "chelis"
        gate = self.args.gates / ".github" / "scripts" / "verify_release_smt.py"
        _, stdout, stderr = self.run(
            "verify_release_smt",
            [sys.executable, gate, chelis],
            env=self.env(),
            timeout=600,
        )
        return tail(stdout or stderr, 1)

    def canary(self) -> str:
        self.require_glibc_host()
        if shutil.which("cc") is None:
            raise StepFailed("no `cc` on PATH; the canary links generated C")
        gate = self.args.gates / "scripts" / "installed_artifact_canary.py"
        scratch = self.work / "canary"
        code, _, stderr = self.run(
            "installed_artifact_canary",
            [sys.executable, gate, "--artifacts", self.args.assets]
            + ["--installer-assets", self.args.assets]
            + ["--source-sha", self.args.source_sha]
            + ["--build-label", self.args.archive_label]
            + ["--version", self.version, "--evidence", scratch],
            env=self.env(),
            timeout=900,
            check=False,
        )
        # Keep the canary's report and process logs; its installed home stays in work/.
        if scratch.is_dir():
            kept = self.evidence / "canary"
            kept.mkdir()
            for path in scratch.iterdir():
                if path.is_file() and (
                    path.name == "report.json" or path.suffix in (".stdout", ".stderr")
                ):
                    shutil.copyfile(path, kept / path.name)
        report_path = scratch / "report.json"
        if not report_path.is_file():
            raise StepFailed(
                f"the canary wrote no report (exit {code}): {tail(stderr)}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        names = [str(process.get("name")) for process in report.get("processes", [])]
        if code != 0 or report.get("status") != "passed":
            raise StepFailed(
                f"canary {report.get('status')}: {report.get('error')}; not run: {report.get('not_run')}"
            )
        return f"{len(names)} processes passed: {', '.join(names)}"

    def linkage(self) -> str:
        home = self.require_offline_home()
        binaries = {
            "chelis": home / "toolchains" / self.version / "bin" / "chelis",
            "chelisup": home / "bin" / "chelisup",
        }
        if platform.system() == "Darwin":
            return "; ".join(
                self.macho_linkage(name, path) for name, path in binaries.items()
            )
        return "; ".join(
            self.elf_linkage(name, path) for name, path in binaries.items()
        )

    def macho_linkage(self, name: str, path: Path) -> str:
        _, stdout, _ = self.run(f"otool-{name}", ["otool", "-L", path], env=self.env())
        verdict = macho_verdict(name, stdout)
        _, commands, _ = self.run(
            f"otool-l-{name}", ["otool", "-l", path], env=self.env()
        )
        minos = re.search(r"cmd LC_BUILD_VERSION.*?minos ([0-9.]+)", commands, re.S)
        floor = minos.group(1) if minos else "unknown"
        return f"{verdict}, minimum macOS {floor}"

    def elf_linkage(self, name: str, path: Path) -> str:
        linkage = elf_dynamic(path)
        self.current.processes.append(
            {"label": f"elf-{name}", "facts": asdict(linkage)}
        )
        static = self.build is not None and self.build.endswith("-static")
        return elf_verdict(name, linkage, static=static)


def check_host(args: argparse.Namespace) -> int:
    match = TAG.fullmatch(args.tag)
    if match is None or SOURCE_SHA.fullmatch(args.source_sha) is None:
        raise SystemExit("--tag must be vX.Y.Z and --source-sha a full commit")
    args.version = match.group(1)
    args.archive_label = args.archive_label or args.tag
    args.assets = args.assets.resolve()
    args.gates = args.gates.resolve()
    args.evidence = args.evidence.resolve()
    args.evidence.mkdir(parents=True, exist_ok=False)
    (args.evidence / "logs").mkdir()
    (args.evidence / "work").mkdir()
    # Shim selection reads the nearest chelis-toolchain file; pin this run's
    # version so a checkout or caller project cannot select another one.
    (args.evidence / "chelis-toolchain").write_text(
        f"{args.version}\n", encoding="utf-8"
    )
    proof = Proof(args)
    facts = host_facts()
    actions: list[tuple[str, Callable[[], str], tuple[str, ...]]] = [
        ("assets", proof.assets, ()),
        ("bootstrap", proof.bootstrap, ("assets",)),
        ("install-online", proof.install_online, ("assets",)),
        ("install-offline", proof.install_offline, ("assets",)),
        ("first-program", proof.first_program, ("install-offline",)),
        ("eval", proof.eval, ("install-offline",)),
        ("native", proof.native, ("install-offline",)),
        ("smt", proof.smt, ("install-offline",)),
        ("canary", proof.canary, ("assets",)),
        ("linkage", proof.linkage, ("install-offline",)),
    ]
    report: dict[str, object] = {
        "schema": SCHEMA,
        "label": args.label or f"{facts['os']} ({facts['machine']})",
        "tag": args.tag,
        "archive_label": args.archive_label,
        "version": args.version,
        "source_sha": args.source_sha,
        "online": args.online,
        "host": facts,
        "status": "failed",
    }
    try:
        for name, action, requires in actions:
            step = Step(name)
            proof.steps[name] = step
            proof.current = step
            started = time.monotonic()
            unmet = [need for need in requires if proof.steps[need].status != "passed"]
            if unmet:
                step.status, step.detail = "blocked", f"needs {', '.join(unmet)}"
            else:
                try:
                    step.detail = action()
                    step.status = "passed"
                except StepSkipped as skipped:
                    step.status, step.detail = "skipped", str(skipped)
                except (StepFailed, OSError, tarfile.TarError, ValueError) as error:
                    step.status, step.detail = "failed", str(error)
            step.seconds = round(time.monotonic() - started, 1)
            print(f"{name}: {step.status}: {step.detail}", flush=True)
        report["status"] = overall(proof.steps.values())
        return 0 if report["status"] == "passed" else 1
    finally:
        report["build"] = proof.build
        report["steps"] = [asdict(step) for step in proof.steps.values()]
        (args.evidence / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(render([report]))


def cell(step: dict[str, object] | None) -> str:
    if step is None:
        return "not run"
    status = str(step["status"])
    return {"passed": "pass", "failed": "**FAIL**", "blocked": "**blocked**"}.get(
        status, status
    )


def render(reports: list[dict[str, object]]) -> str:
    header = ["Host", "OS", "libc", "Build", *STEPS, "Result"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    notes: list[str] = []
    for report in reports:
        host = report.get("host", {})
        assert isinstance(host, dict)
        steps = {str(step["name"]): step for step in report.get("steps", [])}
        row = [
            str(report.get("label")),
            str(host.get("os")),
            str(host.get("libc") or "-"),
            str(report.get("build") or "-"),
            *(cell(steps.get(name)) for name in STEPS),
            "pass" if report.get("status") == "passed" else "**FAIL**",
        ]
        lines.append("| " + " | ".join(row) + " |")
        for name in STEPS:
            step = steps.get(name)
            if step and step["status"] in FAILING:
                notes.append(f"- {report.get('label')} / {name}: {step['detail']}")
    if notes:
        lines += ["", *notes]
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> int:
    reports = []
    for path in sorted(args.root.rglob("report.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") == SCHEMA:
            reports.append(data)
    print(render(reports) if reports else "No release E2E reports were found.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    proof = commands.add_parser("run", help="check one release on this host")
    proof.add_argument("--tag", required=True)
    proof.add_argument("--source-sha", required=True)
    proof.add_argument("--assets", type=Path, required=True)
    proof.add_argument("--gates", type=Path, required=True)
    proof.add_argument("--evidence", type=Path, required=True)
    proof.add_argument("--label")
    proof.add_argument(
        "--archive-label",
        help="label in the archive names (default: --tag); dev-<sha8> marks an unpublished candidate",
    )
    proof.add_argument(
        "--online",
        action="store_true",
        help="also check the bootstrap and GitHub install",
    )
    proof.set_defaults(handler=check_host)
    table = commands.add_parser("summarize", help="render every report as one table")
    table.add_argument("root", type=Path)
    table.set_defaults(handler=summarize)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
