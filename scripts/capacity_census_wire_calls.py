"""Typed compiler evidence for the compiled compiler-api serde boundary.

This module discovers obligations; it does not admit codec owners. Calls are
selected by the defining serde traits and their elaborated bounds, not encoder
names. Every local MIR body is scanned, including private functions and inferred
opaque returns; generic local calls are instantiated from compiler substitutions.
Unknown/indirect publication paths within this boundary fail closed. Foreign
nongeneric wrappers with no serde obligation in their signature are not followed
into foreign MIR; handwritten encoders and new formats likewise still owe C6's
new-surface-kind extension. This is the Serialize/Serializer boundary, not a
Deserialize-input proof. Only the supplied compiled configuration is covered.

The driver uses the exact pinned rustc-private ABI and requires matching
rustc-dev. It is compiled directly, never as a normal Cargo example.
"""

import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tomllib

COMPILER = "88d9e12ae178fab0fb5cc050a94da85685d449ea"
DRIVER = "crates/chelis-compiler-api/examples/wire_calls/driver.rs"


def record_process(prefix: Path, command, result) -> dict:
    """Retain exact fresh process evidence beside the owning verifier target."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("stdout", "stderr"):
        content = getattr(result, name)
        content = content.encode() if isinstance(content, str) else content
        path = prefix.with_suffix("." + name + ".log")
        path.write_bytes(content)
        files[name] = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
    receipt = {"command": list(command), "returncode": result.returncode, **files}
    prefix.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _run(command, *, cwd, env=None, log_prefix=None):
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "RUSTC_BOOTSTRAP": "1", **(env or {})},
        check=False,
    )
    if log_prefix is not None:
        record_process(log_prefix, command, result)
    if result.returncode:
        diagnostics = []
        for line in result.stdout.splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and item.get("reason") == "compiler-message":
                rendered = item.get("message", {}).get("rendered")
                if isinstance(rendered, str):
                    diagnostics.append(rendered)
        details = result.stderr + "\n".join(diagnostics)
        raise ValueError(
            f"compiler command failed ({result.returncode}): {details[-12000:]}"
        )
    return result.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_library_environment(
    library: Path, *, platform: str = sys.platform, environ: dict | None = None
) -> dict[str, str]:
    """Bind a direct rustc-private driver to its pinned sysroot libraries."""
    library = library.resolve()
    if not library.is_dir():
        raise ValueError("missing pinned sysroot library directory")
    if platform == "darwin":
        name = "DYLD_FALLBACK_LIBRARY_PATH"
    elif platform.startswith("linux"):
        name = "LD_LIBRARY_PATH"
    else:
        raise ValueError("unsupported platform for pinned compiler driver runtime")
    current = (os.environ if environ is None else environ).get(name)
    return {name: str(library) + (os.pathsep + current if current else "")}


def _driver_policy(root: Path) -> tuple[list[str], list[dict]]:
    """Read the owning gate and Cargo policy; unsupported policy syntax rejects."""
    gate = ast.parse((root / "scripts/gate.py").read_text())
    rows = [
        node.value
        for node in gate.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "CLIPPY_WORKSPACE"
    ]
    if len(rows) != 1:
        raise ValueError("missing unique workspace Clippy policy")
    command = ast.literal_eval(rows[0])
    if (
        not isinstance(command, list)
        or command[:2] != ["cargo", "clippy"]
        or command.count("--") != 1
    ):
        raise ValueError("unsupported workspace Clippy policy")
    flags = command[command.index("--") + 1 :]
    if not all(isinstance(flag, str) for flag in flags) or flags != ["-D", "warnings"]:
        raise ValueError(
            "changed workspace Clippy flags need explicit driver policy support"
        )
    manifest = tomllib.loads((root / "Cargo.toml").read_text())
    lints = manifest["workspace"]["lints"]
    ordered = []
    for tool, entries in lints.items():
        if tool not in ("rust", "clippy"):
            raise ValueError("unsupported workspace lint tool")
        for name, value in entries.items():
            level, priority = (
                (value, 0)
                if isinstance(value, str)
                else (value["level"], value.get("priority", 0))
            )
            if (
                level not in ("allow", "warn", "deny", "forbid")
                or type(priority) is not int
            ):
                raise ValueError("unsupported workspace lint policy")
            if isinstance(value, dict) and set(value) - {"level", "priority"}:
                raise ValueError(
                    "workspace lint options require explicit driver support"
                )
            ordered.append(
                (
                    priority,
                    name,
                    f"--{level}=" + ("clippy::" if tool == "clippy" else "") + name,
                )
            )
    # No custom cfg or Cargo features belong to this standalone binary. Enable
    # rustc's closed cfg-name check so unexpected_cfgs has actual coverage.
    flags = [entry[2] for entry in sorted(ordered)] + ["--check-cfg=cfg()", *flags]
    inputs = [
        {"path": str(root / path), "sha256": _sha256(root / path)}
        for path in (
            "Cargo.toml",
            "clippy.toml",
            "rust-toolchain.toml",
            "scripts/gate.py",
        )
    ]
    return flags, inputs


def _driver_dep_inputs(path: Path, root: Path) -> list[dict]:
    # rustc's make-style prerequisites escape spaces/backslashes. Only the
    # compiler-emitted rules with prerequisites count; target-only rules and
    # env-dep comments are not Rust inputs. Unsupported/missing paths fail.
    paths, checksums = set(), {}
    for line in path.read_text().replace("\\\n", "").splitlines():
        if line.startswith("# checksum:"):
            match = re.fullmatch(
                r"# checksum:sha256=([0-9a-f]{64}) file_len:(\d+) (.+)", line
            )
            if match is None:
                raise ValueError("unsupported compiler source checksum")
            names = shlex.split(match[3])
            if len(names) != 1:
                raise ValueError("ambiguous compiler checksum input")
            source = (root / names[0]).resolve()
            checksums[source] = (match[1], int(match[2]))
        if line.startswith("#") or ": " not in line:
            continue
        paths.update((root / p).resolve() for p in shlex.split(line.split(": ", 1)[1]))
    if (root / DRIVER).resolve() not in paths:
        raise ValueError("compiler dep-info omitted the driver source")
    if paths != checksums.keys():
        raise ValueError("compiler dep-info omitted source checksums")
    if any((_sha256(p), p.stat().st_size) != checksums[p] for p in paths):
        raise ValueError("source changed since compiler dependency evidence")
    return [{"path": str(p), "sha256": _sha256(p)} for p in sorted(paths)]


def _current_driver_receipt(driver: Path) -> dict:
    receipt = json.loads(driver.with_suffix(".json").read_text())
    artifacts = [
        *receipt["inputs"],
        *receipt["policy_inputs"],
        receipt["dep_info"],
        receipt["compiler"],
    ]
    if any(_sha256(Path(item["path"])) != item["sha256"] for item in artifacts):
        raise ValueError("driver build dependency changed")
    if _sha256(driver) != receipt["binary_sha256"]:
        raise ValueError("driver build executable changed")
    library = Path(receipt["runtime_library_dir"]).resolve()
    compiler = Path(receipt["compiler"]["path"]).resolve()
    if library != compiler.parent.parent / "lib" or not library.is_dir():
        raise ValueError("driver pinned sysroot library changed")
    return receipt


def _driver_runtime_environment(driver: Path) -> dict[str, str]:
    receipt = _current_driver_receipt(driver)
    return _runtime_library_environment(Path(receipt["runtime_library_dir"]))


def build_driver(root: Path, directory: Path) -> Path:
    root, directory = root.resolve(), directory.resolve()
    version = _run(["rustc", "-vV"], cwd=root)
    if f"commit-hash: {COMPILER}\n" not in version:
        raise ValueError("wire invocation driver requires the exact pinned compiler")
    sysroot = Path(_run(["rustc", "--print", "sysroot"], cwd=root).strip())
    compiler = sysroot / "bin/clippy-driver"
    if _run([str(compiler), "-vV"], cwd=root) != version:
        raise ValueError("wire invocation driver requires matching pinned Clippy")
    flags, policy_inputs = _driver_policy(root)
    compiler_identity = {
        "path": str(compiler),
        "sha256": _sha256(compiler),
        "version": version,
    }
    runtime_library = (sysroot / "lib").resolve()
    _runtime_library_environment(runtime_library)
    directory.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(
        json.dumps(
            {
                "source": _sha256(root / DRIVER),
                "compiler": compiler_identity,
                "runtime_library_dir": str(runtime_library),
                "policy": policy_inputs,
                "flags": flags,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output = directory / ("wire-calls-driver-" + identity[:20])
    receipt = output.with_suffix(".json")
    dep_info = output.with_suffix(".d")
    command = [
        str(compiler),
        "--edition=2024",
        "-Cprefer-dynamic",
        "-Crpath",
        "-Lnative=" + str(sysroot / "lib"),
        "-Clink-arg=-Wl,-rpath," + str(sysroot / "lib"),
        *flags,
        "-Zchecksum-hash-algorithm=sha256",
        "--emit=dep-info=" + str(dep_info) + ",link",
        str(root / DRIVER),
        "-o",
        str(output),
    ]
    environment = {
        **_runtime_library_environment(runtime_library),
        "WIRE_DRIVER_COMPILER": COMPILER,
        "CLIPPY_CONF_DIR": str(root),
        "CLIPPY_ARGS": "",
        "CARGO_PRIMARY_PACKAGE": "1",
    }
    if output.is_file() and receipt.is_file():
        try:
            cached = _current_driver_receipt(output)
            if (
                cached["identity"] == identity
                and cached["command"] == command
                and cached["environment"] == environment
            ):
                return output
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Invalid cache rebuilds under policy; it never admits evidence.
    receipt.unlink(missing_ok=True)
    _run(command, cwd=root, env=environment)
    receipt.write_text(
        json.dumps(
            {
                "identity": identity,
                "command": command,
                "environment": environment,
                "compiler": compiler_identity,
                "runtime_library_dir": str(runtime_library),
                "policy_inputs": policy_inputs,
                "inputs": _driver_dep_inputs(dep_info, root),
                "dep_info": {"path": str(dep_info), "sha256": _sha256(dep_info)},
                "binary_sha256": _sha256(output),
            },
            sort_keys=True,
        )
    )
    _current_driver_receipt(output)
    return output


@dataclass(frozen=True)
class InvocationEvidence:
    identity: str
    raw: dict

    @property
    def errors(self):
        return tuple(self.raw["errors"])

    @property
    def dynamic_returns(self):
        return tuple(self.raw["dynamic_returns"])

    def payload_spellings(self):
        return {p["text"] for c in self.raw["calls"] for p in c["payloads"]}


def read_evidence(raw: dict) -> InvocationEvidence:
    required = {
        "format",
        "compiler",
        "crate",
        "serialize_trait",
        "serializer_trait",
        "schema_trait",
        "decoder_traits",
        "dynamic_carriers",
        "bodies",
        "instances",
        "calls",
        "codec_calls",
        "schema_calls",
        "dynamic_returns",
        "errors",
        "rustc_command",
        "inputs",
    }
    compiler_json = isinstance(raw, dict) and raw.get("format") == 2
    if compiler_json:
        required |= {"scope", "deserialize_trait", "conversion_traits", "reader_calls"}
    if not isinstance(raw, dict) or set(raw) != required or raw["format"] not in (1, 2):
        raise ValueError("incomplete compiler invocation evidence")
    if compiler_json and (
        raw["scope"] != "compiler-json"
        or raw["deserialize_trait"].get("crate") != "serde_core"
        or raw["deserialize_trait"].get("path") != "::de::Deserialize"
        or not isinstance(raw["conversion_traits"], list)
        or not isinstance(raw["reader_calls"], list)
    ):
        raise ValueError("invalid compiler JSON reader evidence")
    if raw["compiler"] != COMPILER:
        raise ValueError("unrecognized compiler evidence")
    for key in (
        "bodies",
        "calls",
        "codec_calls",
        "schema_calls",
        "dynamic_returns",
        "dynamic_carriers",
        "errors",
    ):
        if not isinstance(raw[key], list):
            raise ValueError(f"invalid compiler {key} evidence")
    if type(raw["instances"]) is not int or raw["instances"] < 0:
        raise ValueError("invalid compiler instance count")
    for key, path in (
        ("serialize_trait", "::ser::Serialize"),
        ("serializer_trait", "::ser::Serializer"),
    ):
        if raw[key]["crate"] != "serde_core" or raw[key]["path"] != path:
            raise ValueError("wrong defining serde trait identity")
    # This parser checks shape; collect_library below binds the compiler process,
    # invocation and dependency artifacts. A parsed dictionary is never authority.
    identity = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
    return InvocationEvidence(identity, raw)


def analyze_fixture(
    driver: Path, directory: Path, source: str, externs: dict, *, cfg=(), scope=None, log_prefix=None
):
    """Execute actual rustc fixtures using caller-supplied coherent artifacts."""
    with tempfile.TemporaryDirectory(prefix="fixture-", dir=directory) as tmp:
        tmp = Path(tmp)
        input_path, output_path = tmp / "lib.rs", tmp / "report.json"
        if scope not in (None, "compiler-json"):
            raise ValueError("unknown compiler fixture scope")
        prefix = "" if scope else "extern crate serde; extern crate serde_json; extern crate bincode;\n"
        input_path.write_text(prefix + source)
        command = [
            str(driver),
            "--edition=2024",
            "--crate-type=lib",
            "--crate-name=chelis_python" if scope else "--crate-name=wire_fixture",
            "-Zmir-opt-level=0",
            "-Zsrc-hash-algorithm=sha256",
            "-Copt-level=0",
            str(input_path),
        ]
        dependencies = set()
        for name, path in externs.items():
            path = Path(path).resolve()
            command += ["--extern", name + "=" + str(path)]
            dependencies.add(path.parent)
        command += ["-Ldependency=" + str(path) for path in sorted(dependencies)]
        command += [arg for name in cfg for arg in ("--cfg", name)]
        _run(
            command,
            cwd=directory,
            env={
                **_driver_runtime_environment(driver),
                "WIRE_CALL_REPORT": str(output_path),
                "WIRE_CALL_SCOPE": scope or "",
                "PYO3_PYTHON": sys.executable,
                "VIRTUAL_ENV": sys.prefix,
            },
            log_prefix=log_prefix,
        )
        if not output_path.is_file():
            raise ValueError("compiler did not write invocation evidence")
        return read_evidence(json.loads(output_path.read_text()))


def fixture_externs(root: Path, target: Path, directory: Path) -> dict:
    """Build only the tiny dependency fixture when no live artifact set is supplied.

    Ordinary unittest discovery has no environment prerequisite and never skips
    these controls. The normal live runner can pass its already-built externs to
    avoid even this small Cargo invocation.
    """
    versions = {}
    for package in tomllib.loads((root / "Cargo.lock").read_text())["package"]:
        if package["name"] in ("serde", "serde_json", "bincode", "schemars"):
            if package["name"] in versions:
                raise ValueError("ambiguous locked fixture dependency")
            versions[package["name"]] = package["version"]
    if set(versions) != {"serde", "serde_json", "bincode", "schemars"}:
        raise ValueError("missing locked fixture dependencies")
    manifest = directory / "Cargo.toml"
    manifest.write_text(
        '[package]\nname="wire_calls_fixture"\nversion="0.0.0"\nedition="2024"\n'
        '[workspace]\n[lib]\npath="dependencies.rs"\n[dependencies]\n'
        f'serde={{version="={versions["serde"]}",features=["derive","rc"]}}\n'
        f'serde_json={{version="={versions["serde_json"]}",features=["raw_value"]}}\n'
        f'bincode="={versions["bincode"]}"\n'
        f'schemars="={versions["schemars"]}"\n'
    )
    (directory / "dependencies.rs").write_text(
        "extern crate serde; extern crate serde_json; extern crate bincode;\n"
    )
    stream = _run(
        [
            "cargo",
            "build",
            "--offline",
            "--lib",
            "--message-format=json",
            "--manifest-path",
            str(manifest),
        ],
        cwd=root,
        env={"CARGO_TARGET_DIR": str(target), "CARGO_BUILD_JOBS": "1"},
    )
    result = {}
    for line in stream.splitlines():
        artifact = json.loads(line)
        if (
            artifact.get("reason") == "compiler-artifact"
            and artifact["target"]["name"] in versions
        ):
            paths = [p for p in artifact["filenames"] if p.endswith(".rlib")]
            if len(paths) != 1 or artifact["target"]["name"] in result:
                raise ValueError("ambiguous compiler fixture artifacts")
            result[artifact["target"]["name"]] = paths[0]
    if set(result) != set(versions):
        raise ValueError("incomplete compiler fixture artifacts")
    return result


def _artifact_id(path: Path, root: Path) -> str:
    metadata = _run(["rustc", "-Zls=root", str(path)], cwd=root)
    found = re.findall(r" stable_crate_id StableCrateId\((\d+)\)", metadata)
    if len(found) != 1:
        raise ValueError("missing or ambiguous artifact stable crate identity")
    return f"{int(found[0]):016x}"


def _locked_registry_package(root: Path, name: str, package_id: str) -> bool:
    registry = "registry+https://github.com/rust-lang/crates.io-index"
    packages = [
        p
        for p in tomllib.loads((root / "Cargo.lock").read_text())["package"]
        if p["name"] == name
    ]
    return (
        len(packages) == 1
        and packages[0].get("source") == registry
        and package_id == f"{registry}#{name}@{packages[0]['version']}"
    )


def collect_library(root: Path, target: Path, driver: Path, *, rustc_args=(), scope=None) -> dict:
    """Requires the project build slot. Fresh output forces the target callback.

    Return evidence plus actual Cargo/artifact provenance. No final ownership
    receipt is constructed here; the orchestration layer must discharge every
    call, codec implementation and dynamic-return obligation.
    """
    root, target = root.resolve(), target.resolve()
    if scope not in (None, "compiler-json"):
        raise ValueError("unknown compiled boundary scope")
    if not target.is_relative_to(root / "target"):
        raise ValueError("invocation target must belong to this worktree")
    # The driver must execute for the selected crate, bypassing compiler caches.
    # Keep its Cargo artifacts separate from the ordinary cache wrapper's
    # read-only outputs. This retained namespace uses the same source/config;
    # all provenance below comes from this invocation's own Cargo stream.
    invocation_target = target / ("compiler-json-invocations/cargo" if scope else "wire-invocations/cargo")
    if driver.resolve() != build_driver(root, driver.parent).resolve():
        raise ValueError("driver does not match the current source and compiler")
    binary_hash = hashlib.sha256(driver.read_bytes()).hexdigest()
    driver_source_hash = hashlib.sha256((root / DRIVER).read_bytes()).hexdigest()
    driver_build = _current_driver_receipt(driver)
    with tempfile.TemporaryDirectory(
        prefix="wire-invocations-", dir=root / "target"
    ) as tmp:
        tmp = Path(tmp)
        output = tmp / "calls.json"
        command = [
            "cargo",
            "rustc",
            "--locked",
            "--lib",
            "-p",
            "chelis-python" if scope else "chelis-compiler-api",
            "--message-format=json",
            "--",
            "-Zmir-opt-level=0",
            "-Zsrc-hash-algorithm=sha256",
            "-Copt-level=0",
            "-o",
            str(tmp / "fresh.rmeta"),
            *rustc_args,
        ]
        stream = _run(
            command,
            cwd=root,
            env={
                "CARGO_TARGET_DIR": str(invocation_target),
                "CARGO_BUILD_JOBS": "1",
                "PYO3_PYTHON": sys.executable,
                "VIRTUAL_ENV": sys.prefix,
                "RUSTC_WRAPPER": str(driver),
                "WIRE_CALL_CRATE": "chelis_python" if scope else "chelis_compiler_api",
                "WIRE_CALL_SCOPE": scope or "",
                "WIRE_CALL_REPORT": str(output),
            },
            log_prefix=invocation_target.parent / "compiler-json-cargo" if scope else None,
        )
        if hashlib.sha256(driver.read_bytes()).hexdigest() != binary_hash:
            raise ValueError("compiler driver executable changed during collection")
        if (
            hashlib.sha256((root / DRIVER).read_bytes()).hexdigest()
            != driver_source_hash
        ):
            raise ValueError("compiler driver source changed during collection")
        if _current_driver_receipt(driver) != driver_build:
            raise ValueError(
                "compiler driver build provenance changed during collection"
            )
        if not output.is_file():
            raise ValueError("Cargo produced no fresh compiler invocation evidence")
        evidence = read_evidence(json.loads(output.read_text()))
        if evidence.raw["format"] != (2 if scope else 1):
            raise ValueError("compiler evidence does not match the requested scope")
        for source in evidence.raw["inputs"]:
            if source["path"] is not None:
                path = root / source["path"]
                if (
                    source["hash"]
                    != "sha256=" + hashlib.sha256(path.read_bytes()).hexdigest()
                ):
                    raise ValueError(f"compiler source changed or mismatched: {path}")
        provenance = []
        definitions = [
            evidence.raw["serialize_trait"],
            evidence.raw["serializer_trait"],
            *evidence.raw["dynamic_carriers"],
            *evidence.raw["decoder_traits"],
        ]
        if scope:
            definitions += [evidence.raw["deserialize_trait"], *evidence.raw["conversion_traits"]]
            definitions += [call["callee"] for call in evidence.raw["calls"] + evidence.raw["reader_calls"]]
        if evidence.raw["schema_trait"] is not None:
            definitions.append(evidence.raw["schema_trait"])
        definitions.extend(
            call["callee"]
            for call in evidence.raw["calls"]
            if call["callee"]["crate"] == "bincode"
        )
        for name in sorted({d["crate"] for d in definitions}):
            artifacts = [
                a
                for line in stream.splitlines()
                if (a := json.loads(line)).get("reason") == "compiler-artifact"
                and a["target"]["name"] == name
            ]
            if len(artifacts) != 1 or not _locked_registry_package(
                root, name, artifacts[0]["package_id"]
            ):
                raise ValueError(f"unresolved defining {name} Cargo origin")
            paths = [Path(p) for p in artifacts[0]["filenames"] if p.endswith(".rlib")]
            if len(paths) != 1:
                raise ValueError(f"missing defining {name} artifact")
            stable_id = _artifact_id(paths[0], root)
            if any(
                d["stable_crate_id"] != stable_id
                for d in definitions
                if d["crate"] == name
            ):
                raise ValueError(f"compiler/Cargo {name} identity mismatch")
            provenance.append(
                {
                    "package": artifacts[0]["package_id"],
                    "artifact": str(paths[0]),
                    "sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                    "stable_crate_id": stable_id,
                }
            )
        collected = {
            "evidence": evidence.raw,
            "identity": evidence.identity,
            "command": command,
            "provenance": provenance,
            "target_directory": str(invocation_target),
            "driver_source_sha256": driver_source_hash,
            "driver_build": driver_build,
            "binary_sha256": binary_hash,
        }
        if scope:
            # Construction controls compile against the dependency artifacts
            # emitted by this same Cargo invocation, never a filesystem glob.
            externs = []
            for name in ("chelis_compiler_api", "pyo3", "serde_json"):
                artifacts = [item for line in stream.splitlines()
                             if (item := json.loads(line)).get("reason") == "compiler-artifact"
                             and item["target"]["name"] == name]
                if len(artifacts) != 1:
                    raise ValueError(f"missing exact construction dependency {name}")
                paths = [Path(path) for path in artifacts[0]["filenames"] if path.endswith(".rlib")]
                if len(paths) != 1 or not paths[0].is_relative_to(invocation_target):
                    raise ValueError(f"missing owned construction artifact {name}")
                externs.append({"name": name, "artifact": str(paths[0]), "sha256": _sha256(paths[0])})
            collected["fixture_externs"] = externs
            collected["process"] = json.loads((invocation_target.parent / "compiler-json-cargo.json").read_text())
        return collected
