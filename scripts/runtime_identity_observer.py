#!/usr/bin/env python3
"""Effectful observer engine, embedded in chelis-runtime-identity-build.

Protocol 1 receipts are output-addressed, never selected by directory order. A
receipt contains a unit, exact dependency receipt edges, inventory/content and
compiler-output bindings. Only traversal of the runtime closure enforces errors.
Cargo's controller publishes exact artifact events separately from compiler
receipts; no compiler occupies a Cargo job while waiting for another compilation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


class ObservationError(RuntimeError):
    pass


def digest(data: bytes) -> str:
    # Receipt integrity only. All compatibility policy is in the Rust core.
    return hashlib.sha256(data).hexdigest()


def load(path):
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if isinstance(value, dict) and "outputs" in value and value.get("protocol") == 1:
        expected = value.get("receipt_digest")
        actual = digest(json.dumps({k: v for k, v in value.items() if k != "receipt_digest"}, sort_keys=True).encode())
        if expected != actual:
            raise ObservationError(f"corrupt observation receipt {path}")
    return value


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict) and "outputs" in value and value.get("protocol") == 1:
        value = {k: v for k, v in value.items() if k != "receipt_digest"}
        value["receipt_digest"] = digest(json.dumps(value, sort_keys=True).encode())
    fd, temporary = tempfile.mkstemp(prefix=".identity-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def state() -> Path:
    if os.environ.get("CHELIS_IDENTITY_PROTOCOL") != "1":
        raise ObservationError("managed observation protocol 1 is required")
    return Path(os.environ["CHELIS_IDENTITY_STATE"])


def key(path) -> str:
    return digest(os.fsencode(str(Path(path).absolute())))


def binding(path) -> Path:
    return state() / "bindings" / (key(path) + ".json")


def receipt_for(path) -> Path:
    try:
        receipt = Path(load(binding(path))["observation"])
        if not receipt.is_file():
            raise FileNotFoundError(receipt)
        return receipt
    except FileNotFoundError as error:
        raise ObservationError(f"CHELIS_IDENTITY_MISSING_OBSERVATION: no exact cached compiler receipt for {path}; rebuild in a clean managed target") from error


def helper(operation, value):
    proc = subprocess.run([os.environ["CHELIS_IDENTITY_HELPER"], operation], input=json.dumps(value), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise ObservationError(f"{operation}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def probe(command):
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, close_fds=False)
    if proc.returncode:
        raise ObservationError(f"probe failed: {shlex.join(command)}: {proc.stderr}")
    return proc.stdout


def transparent_wrapper(wrapper):
    if not wrapper:
        return None
    executable = shutil.which(wrapper)
    if executable:
        with open(executable, "rb") as stream:
            magic = stream.read(4)
        native = magic == b"\x7fELF" or magic in {b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe"}
        if native and probe([executable, "--version"]).startswith("kache "):
            return executable
    raise ObservationError("CHELIS_IDENTITY_UNOBSERVABLE_WRAPPER: only the transparent Kache compiler cache is supported; arbitrary compiler wrappers can change unobserved arguments or inputs")


def workspace_compiler_wrapper(wrapper):
    """Admit Cargo's Clippy workspace compiler without hiding its identity."""
    if not wrapper:
        return None
    executable = shutil.which(wrapper)
    if executable and Path(executable).name == "clippy-driver":
        with open(executable, "rb") as stream:
            magic = stream.read(4)
        native = magic == b"\x7fELF" or magic in {
            b"\xcf\xfa\xed\xfe",
            b"\xfe\xed\xfa\xcf",
            b"\xca\xfe\xba\xbe",
        }
        if native:
            version = probe([executable, "--version"]).strip()
            if version.startswith("clippy "):
                return {
                    "path": executable,
                    "identity": version
                    + "\nsha256:"
                    + digest(Path(executable).read_bytes()),
                }
    raise ObservationError(
        "CHELIS_IDENTITY_UNOBSERVABLE_WRAPPER: only Cargo's native "
        "clippy-driver workspace compiler is supported"
    )


def workspace_compiler_applies():
    """Mirror Cargo's RUSTC_WORKSPACE_WRAPPER membership boundary."""
    if not os.environ.get("CHELIS_IDENTITY_WORKSPACE_WRAPPER"):
        return False
    metadata_path = os.environ.get("CHELIS_IDENTITY_METADATA")
    manifest_dir = os.environ.get("CARGO_MANIFEST_DIR")
    if not metadata_path or not manifest_dir:
        return False
    metadata = load(metadata_path)
    manifest = Path(manifest_dir).absolute() / "Cargo.toml"
    matches = [
        package
        for package in metadata["packages"]
        if Path(package["manifest_path"]).absolute() == manifest
    ]
    if len(matches) != 1:
        raise ObservationError(f"no unique Cargo package for {manifest.parent}")
    return matches[0]["id"] in set(metadata["workspace_members"])


def rustc_command(real_rustc, args):
    """Compose the original cache/workspace wrappers in Cargo's order."""
    inner = transparent_wrapper(os.environ.get("CHELIS_IDENTITY_INNER_WRAPPER"))
    workspace = (
        workspace_compiler_wrapper(
            os.environ.get("CHELIS_IDENTITY_WORKSPACE_WRAPPER")
        )
        if workspace_compiler_applies()
        else None
    )
    command = ([inner] if inner else []) + (
        [workspace["path"]] if workspace else []
    ) + [real_rustc, *args]
    return command, workspace


_UNSPECIFIED_WORKSPACE_WRAPPER = object()


def values(args, flag):
    result = []
    for index, arg in enumerate(args):
        if arg == flag:
            if index + 1 == len(args):
                raise ObservationError(f"missing value for {flag}")
            result.append(args[index + 1])
        elif arg.startswith(flag + "="):
            result.append(arg[len(flag) + 1:])
        elif flag in ("-C", "-L", "-l") and arg.startswith(flag) and len(arg) > 2:
            result.append(arg[2:])
    return result


def one(args, flag, default=None):
    found = values(args, flag)
    return found[-1] if found else default


def enumerate_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"input inventory root is absent: {root}")
    result = []
    for directory, dirs, files in os.walk(root, onerror=lambda error: (_ for _ in ()).throw(error)):
        dirs[:] = sorted(name for name in dirs if name not in {".git", "target", ".venv", ".devenv", "node_modules"})
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() and not path.resolve().is_file():
                raise ObservationError(f"unreadable input symlink {path}")
            result.append(path.relative_to(root).as_posix())
    return result


def inventory(root, prefix, required=(), classification="source"):
    return {"physical": str(Path(root).absolute()), "inventory": {"logical_prefix": prefix, "files": enumerate_files(root), "explicitly_required": sorted(set(required)), "class": classification}}


def capture(roots, captured_by_path=None):
    for root in roots:
        root["inventory"]["explicitly_required"] = sorted(set(root["inventory"]["explicitly_required"]))
    planned = helper("core-plan", [root["inventory"] for root in roots])
    physical = {}
    for root in roots:
        prefix = root["inventory"]["logical_prefix"].rstrip("/")
        for name in root["inventory"]["files"]:
            logical = prefix + "/" + name if prefix else name
            physical[logical] = str(Path(root["physical"]) / name)
    # One graph validation shares a single byte observation of shared inputs
    # (notably the sysroot); a later validation starts a fresh observation.
    cache = {} if captured_by_path is None else captured_by_path
    missing = {physical[item["logical_path"]]: item["logical_path"] for item in planned
               if physical[item["logical_path"]] not in cache}
    if missing:
        captured = helper("core-capture", [{"logical_path": logical, "physical": path}
                                          for path, logical in missing.items()])
        cache.update(zip(missing, captured, strict=True))
    return planned, [{**cache[physical[item["logical_path"]]], "logical_path": item["logical_path"]}
                     for item in planned]


def cargo_event(category, artifact):
    path = state() / category / os.environ["CHELIS_IDENTITY_SESSION"] / (key(artifact) + ".json")
    deadline = time.monotonic() + 60
    while not path.exists():
        if time.monotonic() >= deadline:
            raise ObservationError(f"missing exact Cargo {category} observation for {artifact}")
        time.sleep(0.01)
    return load(path)


def invocation_environment():
    if os.environ.get("CHELIS_IDENTITY_BACKEND") == "cargo":
        path = os.environ.get("CHELIS_IDENTITY_INVOCATION_ENVIRONMENT")
        if not path:
            raise ObservationError("Cargo invocation environment observation is required")
        return load(path)
    return {name: digest(os.fsencode(value)) for name, value in os.environ.items()}


def cargo_output_files(event):
    # Cargo also reports directory-valued debug bundles. They are not linkable
    # compiler outputs (rustc --print=file-names does not report them). Native
    # files must still all bind; missing files are retained here and fail reads.
    return [filename for filename in event["filenames"] if not Path(filename).is_dir()]


def check_receipt(receipt, captured_by_path=None):
    if receipt.get("protocol") != 1:
        raise ObservationError("unsupported observation receipt")
    if receipt.get("errors"):
        raise ObservationError("incomplete runtime dependency observation: " + "; ".join(receipt["errors"]))
    for output in receipt["outputs"]:
        if digest(Path(output["path"]).read_bytes()) != output["digest"]:
            raise ObservationError(f"changed compiler output: {output['path']}")
    required, captured = capture(refreshed_roots(receipt["roots"]), captured_by_path)
    if required != receipt["required_inputs"] or captured != receipt["captured"]:
        name = receipt["unit"]["package"]["name"]
        raise ObservationError(f"stale source/input inventory: {name}; Cargo does not track this input, so run `cargo clean -p {name}` and rebuild")
    ambient = receipt.get("ambient_environment", []) if not receipt.get("installed") else []
    if ambient:
        current_environment = invocation_environment()
        for variable in ambient:
            if current_environment.get(variable["name"]) != variable["digest"]:
                raise ObservationError(f"changed build input environment {variable['name']}")


def import_dependencies():
    for name in ("CHELIS_IDENTITY_DEPENDENCIES", "CHELIS_IDENTITY_BUILD_DEPENDENCIES"):
        if os.environ.get(name):
            for item in load(os.environ[name]):
                if set(item) != {"artifact", "observation"}:
                    raise ObservationError(f"malformed dependency binding in {name}")
                receipt = load(item["observation"])
                if item["artifact"] not in {entry["path"] for entry in receipt["outputs"]}:
                    raise ObservationError("dependency binding does not name an observed output")
                atomic(binding(item["artifact"]), item)


def source_workspace(manifest_dir):
    import tomllib
    for parent in [manifest_dir, *manifest_dir.parents]:
        manifest = parent / "Cargo.toml"
        if manifest.is_file() and "workspace" in tomllib.loads(manifest.read_text()):
            return parent
    return manifest_dir


def package_facts():
    import tomllib
    manifest_dir = Path(os.environ["CARGO_MANIFEST_DIR"]).absolute()
    manifest = tomllib.loads((manifest_dir / "Cargo.toml").read_text())
    if os.environ.get("CHELIS_IDENTITY_METADATA"):
        packages = load(os.environ["CHELIS_IDENTITY_METADATA"])["packages"]
        matches = [p for p in packages if Path(p["manifest_path"]).parent == manifest_dir]
        if len(matches) != 1:
            raise ObservationError(f"no unique Cargo package for {manifest_dir}")
        package = matches[0]
        source = package["source"]
        if source is None:
            source = "path:" + manifest_dir.relative_to(source_workspace(manifest_dir)).as_posix()
        checksum = None
        lock = Path(os.environ["CHELIS_IDENTITY_WORKSPACE"]) / "Cargo.lock"
        if lock.exists():
            entries = tomllib.loads(lock.read_text()).get("package", [])
            match = [p for p in entries if p["name"] == package["name"] and p["version"] == package["version"] and p.get("source") == package["source"]]
            if len(match) == 1:
                checksum = match[0].get("checksum")
        identity = {"name": package["name"], "version": package["version"], "source": source, "checksum": checksum}
    else:
        identity = {"name": os.environ["CARGO_PKG_NAME"], "version": os.environ["CARGO_PKG_VERSION"], "source": os.environ["CHELIS_IDENTITY_PACKAGE_SOURCE"], "checksum": os.environ.get("CHELIS_IDENTITY_PACKAGE_CHECKSUM") or None}
    prefix = "packages/" + identity["name"] + "-" + identity["version"] + "-" + digest(identity["source"].encode())[:16]
    return identity, manifest_dir, manifest, prefix


def execution_path(out):
    return state() / "executions" / (key(out) + ".json")


def unit_out_dir(crate_name):
    """Return OUT_DIR only when Cargo supplied it to this unit.

    Cargo sets OUT_DIR when compiling a package that has a build script, never
    for the build script itself, and never clears one. Any other value is
    inherited from an enclosing process, such as a test that runs a nested
    build, and names another unit's outputs and producer declaration.
    """
    out = os.environ.get("OUT_DIR")
    if not out or os.environ.get("CHELIS_IDENTITY_BACKEND") != "cargo":
        return out
    if crate_name == "build_script_build":
        return None
    import tomllib
    manifest_dir = Path(os.environ["CARGO_MANIFEST_DIR"]).absolute()
    package = tomllib.loads((manifest_dir / "Cargo.toml").read_text()).get("package", {})
    return out if package.get("build", (manifest_dir / "build.rs").exists()) else None


def build_execution(program, arguments):
    import_dependencies()
    compiled_path = receipt_for(program)
    compiled = load(compiled_path)
    expected = next((output["digest"] for output in compiled["outputs"] if output["path"] == str(Path(program).absolute())), None)
    if expected is None or digest(Path(program).read_bytes()) != expected:
        raise ObservationError("build-script executable differs from its exact compilation receipt")
    environment = dict(os.environ)
    # DEP namespaces belong only to immediate normal dependencies with `links`.
    if environment.get("CHELIS_IDENTITY_BACKEND") == "nix":
        environment = {k: v for k, v in environment.items() if not k.startswith("DEP_")}
        direct = environment.get("CHELIS_IDENTITY_DIRECT_DEPENDENCIES")
        if not direct and environment.get("CHELIS_IDENTITY_DEPENDENCIES") and load(environment["CHELIS_IDENTITY_DEPENDENCIES"]):
            raise ObservationError("immediate normal dependency bindings are required for Nix DEP metadata")
        dependencies = load(direct) if direct else []
        for dep in dependencies:
            receipt = load(dep["observation"])
            links = receipt.get("links")
            if links:
                for name, value in (receipt.get("build_execution") or {}).get("metadata", {}).items():
                    environment["DEP_" + links.upper().replace("-", "_") + "_" + name.upper().replace("-", "_")] = value
    argv0 = program
    if program.endswith(".identity-real"):
        original = program.removesuffix(".identity-real")
        if any(output["path"] == original for output in compiled["outputs"]):
            argv0 = original
    proc = subprocess.Popen([argv0, *arguments], executable=program, env=environment, stdout=subprocess.PIPE, close_fds=False)
    chunks = []
    assert proc.stdout is not None
    with proc.stdout:
        for line in iter(proc.stdout.readline, b""):
            chunks.append(line)
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
    code = proc.wait()
    if code:
        return code
    lines = b"".join(chunks).decode("utf-8").splitlines()
    declarations = {"inputs": [], "environment": [], "metadata": {}}
    known = {"rustc-link-lib", "rustc-link-search", "rustc-cfg", "rustc-check-cfg", "rustc-env", "rustc-flags", "warning", "error", "rustc-link-arg", "rustc-link-arg-bin", "rustc-link-arg-bins", "rustc-link-arg-tests", "rustc-link-arg-examples", "rustc-link-arg-benches", "rustc-cdylib-link-arg"}
    for line in lines:
        if not line.startswith("cargo:"):
            continue
        name, sep, value = line.removeprefix("cargo:").removeprefix(":").partition("=")
        if not sep:
            continue
        if name == "rerun-if-changed":
            declarations["inputs"].append(str(Path(value).absolute()))
        elif name == "rerun-if-env-changed":
            if not value.startswith("CHELIS_IDENTITY_") and value not in {"PROFILE", "CARGO_ENCODED_RUSTFLAGS"}:
                declarations["environment"].append({"name": value, "value": environment.get(value)})
        elif name == "metadata":
            k, _, v = value.partition("=")
            declarations["metadata"][k] = v
        elif name not in known:
            declarations["metadata"][name] = value
    declarations["environment"] = list({item["name"]: item for item in declarations["environment"]}.values())
    declarations.update({"receipt": str(compiled_path), "stdout": lines, "configuration_environment": [{"name": name, "value": environment.get(name)} for name in ("DEBUG", "OPT_LEVEL", "PROFILE", "TARGET", "HOST")]})
    out = os.environ["OUT_DIR"]
    atomic(execution_path(out), declarations)
    return 0


def clean_probe_args(args):
    result = []
    flags = {"--target", "--sysroot", "--cfg", "--check-cfg", "-C"}
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in flags:
            result.extend(args[index:index + 2]); index += 2
        else:
            if any(arg.startswith(flag + "=") for flag in flags) or arg.startswith("-C"):
                result.append(arg)
            index += 1
    return result


def normalize_values(values, mappings):
    roots = {}
    for physical, logical in mappings:
        roots.setdefault(physical, logical)
    return helper("core-normalize", {"values": values, "roots": [{"physical": physical, "logical": logical} for physical, logical in roots.items()]})


def normalized_flags(args, mappings):
    # Remove Cargo output routing/salts and named input/dependency paths, but keep
    # ordered compiler options. Core owns compatibility canonicalization.
    removed = {"--out-dir", "--emit", "--extern", "--crate-name", "--crate-type", "--error-format", "--json", "--diagnostic-width", "-o", "-L", "--cfg", "--check-cfg", "--target"}
    result = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--extern" and index + 1 < len(args) and "=" not in args[index + 1]:
            result.extend(args[index:index + 2]); index += 2; continue
        if arg.startswith("--extern=") and "=" not in arg.removeprefix("--extern="):
            result.append(arg); index += 1; continue
        if arg in removed:
            index += 2; continue
        if any(arg.startswith(flag + "=") for flag in removed) or arg.endswith(".rs"):
            index += 1; continue
        if arg == "-C" and index + 1 < len(args) and args[index + 1].split("=")[0] in {"metadata", "extra-filename", "incremental"}:
            index += 2; continue
        if arg.startswith("-C") and arg[2:].split("=")[0] in {"metadata", "extra-filename", "incremental"}:
            index += 1; continue
        result.append(arg)
        index += 1
    return normalize_values(result, mappings)


def compiler_inputs(real_rustc, args, roots, manifest_dir, workspace, out):
    """Ask rustc for the actual expanded input closure before record emission."""
    if out and (Path(out, "chelis-runtime-identity-producer.json").exists() or os.environ.get("CHELIS_IDENTITY_ROLE")):
        for name in ("chelis_runtime_identity.bin", "chelis_runtime_identity_provenance.bin"):
            generated = Path(out, name)
            if not generated.exists():
                generated.parent.mkdir(parents=True, exist_ok=True)
                # This dep-info-only probe emits no native output. Production
                # cannot start until emit_producer writes the derived bytes.
                generated.write_bytes(b"")
    fd, temporary = tempfile.mkstemp(suffix=".d", dir=state())
    os.close(fd)
    filtered = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"--emit", "-o"}:
            index += 2
        elif arg.startswith(("--emit=", "-o=", "-Cincremental=")):
            index += 1
        elif arg == "-C" and index + 1 < len(args) and args[index + 1].startswith("incremental="):
            # Incremental state belongs to the real compilation. A cache wrapper
            # may prune or relocate that directory while this probe runs.
            index += 2
        else:
            filtered.append(arg); index += 1
    try:
        probe([real_rustc, *filtered, "--emit=dep-info=" + temporary])
        text = Path(temporary).read_text(encoding="utf-8").replace("\\\n", "")
    finally:
        os.unlink(temporary)
    dependencies = []
    for line in text.splitlines():
        if ": " in line and not line.startswith("#"):
            dependencies = shlex.split(line.split(": ", 1)[1])
            break
    if not dependencies:
        raise ObservationError("rustc dep-info did not expose its required inputs")
    for name in dependencies:
        path = Path(os.path.abspath(name))
        if path.resolve() != Path(name).resolve():
            raise ObservationError(f"compiler input traverses a noncanonical directory symlink: {name}")
        if out and path.parent == Path(out) and path.name in {"chelis_runtime_identity.bin", "chelis_runtime_identity_provenance.bin"}:
            continue
        matched = False
        for root in roots:
            physical = Path(root["physical"])
            if path.is_relative_to(physical):
                relative = path.relative_to(physical).as_posix()
                if relative in root["inventory"]["files"]:
                    root["inventory"]["explicitly_required"].append(relative)
                    matched = True
                    break
        if not matched:
            if not path.is_relative_to(workspace):
                raise ObservationError(f"required compiler input has no declared logical root: {path}")
            relative = path.relative_to(workspace)
            prefix = "workspace" if relative.parent.as_posix() == "." else "workspace/" + relative.parent.as_posix()
            roots.append({"physical": str(path.parent), "inventory": {"logical_prefix": prefix, "files": [path.name], "explicitly_required": [path.name], "class": "build"}, "fixed": True})
    environment = []
    for line in text.splitlines():
        if line.startswith("# env-dep:"):
            name, separator, value = line.removeprefix("# env-dep:").partition("=")
            environment.append({"name": name, "value": value if separator else None})
    return environment


def target_codegen(real_rustc, args):
    # `native` and an omitted -Ctarget-cpu are requests, not effective CPUs.
    # Read the backend's observed target facts from a retained public function.
    flags = []
    for name in ("--target", "--sysroot"):
        value = one(args, name)
        if value:
            flags.extend([name, value])
    for value in values(args, "-C"):
        if value.partition("=")[0] not in {"incremental", "metadata", "extra-filename"}:
            flags.extend(["-C", value])
    with tempfile.TemporaryDirectory(prefix="target-probe-", dir=state()) as directory:
        source = Path(directory, "probe.rs")
        output = Path(directory, "probe.ll")
        source.write_text('#![no_std]\n#[unsafe(no_mangle)] pub extern "C" fn identity_target_probe() -> u8 { 0 }\n')
        probe([real_rustc, *flags, "--edition=2021", "--crate-name=chelis_identity_target_probe",
               "--crate-type=lib", "--emit=llvm-ir", "-o", str(output), str(source)])
        text = output.read_text()
    facts = {}
    for name, pattern in {
        "cpu": r'"target-cpu"="([^"]+)"',
        "llvm_triple": r'^target triple = "([^"]+)"',
        "data_layout": r'^target datalayout = "([^"]+)"',
    }.items():
        matches = set(re.findall(pattern, text, re.MULTILINE))
        if len(matches) != 1:
            raise ObservationError(f"compiler did not expose one effective target {name}")
        facts[name] = matches.pop()
    return facts


def collect_unit(
    real_rustc, args, workspace_wrapper=_UNSPECIFIED_WORKSPACE_WRAPPER
):
    if workspace_wrapper is _UNSPECIFIED_WORKSPACE_WRAPPER:
        workspace_wrapper = (
            workspace_compiler_wrapper(
                os.environ.get("CHELIS_IDENTITY_WORKSPACE_WRAPPER")
            )
            if workspace_compiler_applies()
            else None
        )
    identity, manifest_dir, manifest, prefix = package_facts()
    compiler = probe([real_rustc, "-vV"])
    cfg = probe([real_rustc, *clean_probe_args(args), "--print=cfg"]).splitlines()
    options = dict(item.split("=", 1) if "=" in item else (item, "yes") for item in values(args, "-C"))
    host = next((line.removeprefix("host: ") for line in compiler.splitlines() if line.startswith("host: ")), None)
    if not host:
        raise ObservationError("rustc verbose version lacks host")
    triple = one(args, "--target", host)
    target_spec = None
    specification_path = None
    if triple.endswith(".json"):
        specification_path = Path(triple).absolute()
        target_spec = helper("core-capture", [{"logical_path": "toolchain/target/" + specification_path.name, "physical": str(specification_path)}])[0]["digest"]
        triple = specification_path.stem
    features = sorted(value.split('"')[1] for value in cfg if value.startswith('feature="'))
    kind = "build_script" if one(args, "--crate-name") == "build_script_build" else ("proc_macro" if "proc-macro" in one(args, "--crate-type", "") else "library")
    required = ["Cargo.toml"]
    for arg in args:
        if arg.endswith(".rs"):
            required.append(Path(arg).absolute().relative_to(manifest_dir).as_posix())
    roots = [inventory(manifest_dir, prefix, required)]
    if specification_path:
        roots.append({"physical": str(specification_path.parent), "inventory": {"logical_prefix": "toolchain/target", "files": [specification_path.name], "explicitly_required": [specification_path.name], "class": "toolchain"}, "fixed": True})
    workspace = source_workspace(manifest_dir) if identity["source"].startswith("path:") else Path(os.environ["CHELIS_IDENTITY_WORKSPACE"])
    out = unit_out_dir(one(args, "--crate-name"))
    # Workspace declarations are recipe inputs that Cargo does not rebuild every
    # unit for. Producer units retain the owning workspace's declarations and
    # track them; every other unit records their effect in its observed
    # invocation, so a declaration edit cannot strand a cached dependency.
    producer = kind != "build_script" and bool(os.environ.get("CHELIS_IDENTITY_ROLE") or (out and Path(out, "chelis-runtime-identity-producer.json").exists()))
    for name in ("Cargo.toml", "rust-toolchain.toml", ".cargo/config.toml"):
        path = workspace / name
        if producer and path.is_file() and manifest_dir != workspace:
            suffix = Path(name).parent.as_posix()
            logical = "workspace" if suffix == "." else "workspace/" + suffix
            roots.append({"physical": str(path.parent), "inventory": {"logical_prefix": logical, "files": [path.name], "explicitly_required": [path.name], "class": "build"}, "fixed": True})
    # Implicit std/core/alloc and compiler-builtins are real code inputs too.
    # A mutable custom sysroot cannot borrow the compiler's version string as
    # evidence for the library bytes that its compilation actually consumes.
    toolchain_directories = probe([real_rustc, *clean_probe_args(args), "--print=sysroot", "--print=target-libdir"]).splitlines()
    if len(toolchain_directories) != 2 or not all(toolchain_directories):
        raise ObservationError("compiler did not expose its sysroot and target library directory")
    sysroot, library_dir = (Path(path).absolute() for path in toolchain_directories)
    libraries = enumerate_files(library_dir)
    if not libraries:
        raise ObservationError(f"empty compiler target library directory: {library_dir}")
    roots.append({"physical": str(library_dir), "inventory": {"logical_prefix": f"toolchain/{triple}/sysroot", "files": libraries, "explicitly_required": libraries.copy(), "class": "toolchain"}})
    dependencies = []
    for external in values(args, "--extern"):
        name, separator, artifact = external.partition("=")
        if not separator:
            # Cargo supplies compiler crates (notably proc_macro) by name.
            # Bind these to the actual sysroot bytes, not a Cargo receipt.
            provided = list(library_dir.glob(f"lib{name}-*.rlib"))
            if len(provided) != 1:
                raise ObservationError(f"no unique compiler-provided extern binding: {name}")
        else:
            dependencies.append({"name": name, "observation": str(receipt_for(artifact))})
    build = None
    execution = None
    ambient = []
    if out and kind != "build_script" and manifest["package"].get("build", (manifest_dir / "build.rs").exists()) and execution_path(out).exists():
        execution = load(execution_path(out))
        build = execution["receipt"]
        ambient = execution["environment"]
        native_directives = [line for line in execution["stdout"] if line.startswith(("cargo:rustc-link-lib=", "cargo::rustc-link-lib=", "cargo:rustc-link-search=", "cargo::rustc-link-search="))]
        if native_directives:
            raise ObservationError("CHELIS_IDENTITY_UNOBSERVABLE_NATIVE: runtime-reachable native build inputs require compiler-bound observations; a caller-provided file list is not evidence")
        if os.environ.get("CHELIS_IDENTITY_BACKEND") == "cargo":
            event = cargo_event("build-events", out)
            declared_cfg = []
            declared_env = []
            for line in execution["stdout"]:
                directive = line.removeprefix("cargo:").removeprefix(":")
                if directive.startswith("rustc-cfg="):
                    declared_cfg.append(directive.removeprefix("rustc-cfg="))
                elif directive.startswith("rustc-env="):
                    declared_env.append(directive.removeprefix("rustc-env=").split("=", 1))
            if sorted(declared_cfg) != sorted(event["cfgs"]) or sorted(declared_env) != sorted(event["env"]):
                raise ObservationError("Cargo build-script event differs from captured execution")
        for input_path in execution["inputs"]:
            path = Path(input_path)
            if path.is_dir():
                # A producer hook declares its own source directory for Cargo
                # invalidation; the core still selects production membership.
                if path == manifest_dir and Path(out, "chelis-runtime-identity-producer.json").exists():
                    continue
                relative = path.relative_to(manifest_dir).as_posix()
                roots[0]["inventory"]["explicitly_required"].extend(name for name in roots[0]["inventory"]["files"] if relative == "." or name.startswith(relative + "/"))
            elif path.is_file():
                try:
                    roots[0]["inventory"]["explicitly_required"].append(path.relative_to(manifest_dir).as_posix())
                except ValueError:
                    relative = path.relative_to(workspace)
                    logical = "workspace" if relative.parent.as_posix() == "." else "workspace/" + relative.parent.as_posix()
                    roots.append({"physical": str(path.parent), "inventory": {"logical_prefix": logical, "files": [path.name], "explicitly_required": [path.name], "class": "build"}, "fixed": True})
            else:
                raise ObservationError(f"missing declared build input {path}")
        generated = [name for name in enumerate_files(out) if not name.startswith("chelis_runtime_identity") and not name.startswith("chelis-runtime-identity")]
        if generated:
            roots.append({"physical": out, "inventory": {"logical_prefix": prefix + "/generated", "files": generated, "explicitly_required": generated.copy(), "class": "build"}, "generated": True})
    native_searches = [value for value in values(args, "-L") if not value.startswith("dependency=")]
    if os.environ.get("CHELIS_IDENTITY_BACKEND") == "nix" and out and any(Path(value).absolute() == Path(out).absolute() for value in native_searches):
        # Inspect the builder's OUT_DIR search only when it is actually passed.
        # Crates without a build script can have a nonexistent, unused OUT_DIR.
        generated_native = any(Path(name).suffix in {".a", ".o", ".obj", ".so", ".dylib", ".dll", ".lib"} or ".so." in Path(name).name for name in enumerate_files(out))
        if not generated_native:
            native_searches = [value for value in native_searches if Path(value).absolute() != Path(out).absolute()]
    if values(args, "-l") or native_searches:
        raise ObservationError("CHELIS_IDENTITY_UNOBSERVABLE_NATIVE: native linker inputs require compiler-bound observations")
    mappings = [(str(manifest_dir), prefix), (str(workspace), "workspace")]
    mappings.append((str(sysroot), "toolchain/sysroot"))
    if os.environ.get("CHELIS_IDENTITY_BACKEND") == "nix" and os.environ.get("NIX_BUILD_TOP"):
        mappings.append((str(Path(os.environ["NIX_BUILD_TOP"]).absolute()), "build"))
    if out:
        mappings.insert(0, (out, prefix + "/generated"))
    tools = []
    if workspace_wrapper:
        tools.append(
            {
                "name": "workspace-compiler",
                "identity": workspace_wrapper["identity"],
            }
        )
    crate_types = ",".join(values(args, "--crate-type")).split(",")
    if kind in {"build_script", "proc_macro"} or any(item in {"bin", "cdylib", "dylib"} for item in crate_types):
        linker = options.get("linker", "cc")
        tools.append({"name": "linker", "identity": probe([linker, "--version"])})
        if Path(linker).is_absolute():
            mappings.insert(0, (str(Path(linker).parent), "toolchain/linker"))
        tools[-1]["identity"] = normalize_values([tools[-1]["identity"]], mappings)[0]
    unit = {"package": identity, "target_name": one(args, "--crate-name"), "kind": kind,
            "target": {"triple": triple, **target_codegen(real_rustc, args), "features": sorted(value.split('"')[1] for value in cfg if value.startswith('target_feature="')), "specification": target_spec},
            "features": features, "configuration": {"opt_level": options.get("opt-level", "0"), "debuginfo": options.get("debuginfo", "0"), "debug_assertions": "debug_assertions" in cfg, "panic": next((v.split('"')[1] for v in cfg if v.startswith('panic="')), "unwind"), "rustflags": normalized_flags(args, mappings), "cfg": cfg},
            "compiler": {"verbose_version": compiler, "tools": tools}, "inputs": [], "dependencies": [], "build_script": None,
            "build_environment": list({item["name"]: dict(item) for item in execution["environment"] + execution["configuration_environment"]}.values()) if execution else []}
    observed_environment = compiler_inputs(real_rustc, args, roots, manifest_dir, workspace, out)
    if kind == "build_script":
        # The build-script program compiles only what rustc reads for it. Its
        # package's other sources are inputs of the units it helps build, and
        # Cargo never recompiles the script when they change.
        package_root = roots[0]
        package_root["inventory"]["files"] = sorted(set(package_root["inventory"]["explicitly_required"]))
        package_root["fixed"] = True
    unit["compiler_environment"] = list({item["name"]: dict(item) for item in observed_environment}.values())
    supplied = {"OUT_DIR", "CARGO", "CARGO_MANIFEST_DIR", "CARGO_MANIFEST_PATH", "CARGO_CRATE_NAME", "CARGO_BIN_NAME", "CARGO_PRIMARY_PACKAGE", "CARGO_TARGET_TMPDIR"}
    supplied.update("CARGO_PKG_" + suffix for suffix in ("VERSION", "VERSION_MAJOR", "VERSION_MINOR", "VERSION_PATCH", "VERSION_PRE", "AUTHORS", "NAME", "DESCRIPTION", "HOMEPAGE", "REPOSITORY", "LICENSE", "LICENSE_FILE", "RUST_VERSION", "README"))
    if execution:
        supplied.update(line.removeprefix("cargo:").removeprefix(":").removeprefix("rustc-env=").partition("=")[0]
                        for line in execution["stdout"] if line.startswith(("cargo:rustc-env=", "cargo::rustc-env=")))
    ambient_names = {item["name"] for item in ambient + observed_environment if item["name"] not in supplied}
    original_environment = invocation_environment() if ambient_names else {}
    ambient = [{"name": name, "digest": original_environment.get(name)} for name in sorted(ambient_names)]
    observations = [item for item in unit["build_environment"] + unit["compiler_environment"] if item["value"] is not None]
    normalized = normalize_values([item["value"] for item in observations], mappings)
    for item, value in zip(observations, normalized):
        item["value"] = value
    planned, captured = capture(roots)
    unit["inputs"] = [entry["logical_path"] for entry in planned]
    if kind != "build_script" and manifest["package"].get("build", (manifest_dir / "build.rs").exists()) and execution is None:
        raise ObservationError("CHELIS_IDENTITY_MISSING_OBSERVATION: build-script execution receipt is missing; rebuild in a clean managed target")
    return {"protocol": 1, "unit": unit, "roots": roots, "required_inputs": planned, "captured": captured, "dependencies": dependencies, "build_script": build, "outputs": [], "errors": [], "ambient_environment": ambient, "links": manifest["package"].get("links"), "build_execution": execution, "cwd": str(Path.cwd())}


def refresh_inventory(root):
    if root.get("fixed"):
        return root
    files = enumerate_files(root["physical"])
    if root.get("generated"):
        files = [name for name in files if not name.startswith("chelis_runtime_identity") and not name.startswith("chelis-runtime-identity")]
    return {**root, "inventory": {**root["inventory"], "files": files}}


# Keep fixed workspace declarations separate from recursively enumerated roots.
def refreshed_roots(roots):
    return [refresh_inventory(root) for root in roots]


def graph_recipe(runtime):
    units, required, captured, mappings = [], {}, {}, {}
    captured_by_path = {}
    visited, active = {}, set()
    def visit(receipt, token):
        if token in active:
            raise ObservationError("cycle in observed compilation graph")
        if token in visited:
            return visited[token]
        active.add(token)
        check_receipt(receipt, captured_by_path)
        index = len(units)
        visited[token] = index
        unit = dict(receipt["unit"])
        units.append(unit)
        unit["dependencies"] = [{"name": edge["name"], "unit": visit(load(edge["observation"]), edge["observation"])} for edge in receipt["dependencies"]]
        unit["build_script"] = visit(load(receipt["build_script"]), receipt["build_script"]) if receipt["build_script"] else None
        for entry in receipt["required_inputs"]:
            old = required.setdefault(entry["logical_path"], entry)
            if old != entry:
                raise ObservationError("conflicting logical input classes")
        for entry in receipt["captured"]:
            old = captured.setdefault(entry["logical_path"], entry)
            if old != entry:
                raise ObservationError("source changed during compilation")
        for root in receipt["roots"]:
            mappings[root["physical"]] = root["inventory"]["logical_prefix"]
        active.remove(token)
        return index
    runtime_index = visit(runtime, "runtime")
    if workspace := os.environ.get("CHELIS_IDENTITY_WORKSPACE"):
        mappings.setdefault(workspace, "workspace")
    recipe = {"schema_version": 1, "runtime_unit": runtime_index, "units": units, "required_inputs": list(required.values()), "cargo_profile": runtime["profile"], "public_abi": 2}
    return recipe, list(captured.values()), [{"physical": physical, "logical": logical} for physical, logical in sorted(mappings.items())]


def find_runtime(receipt):
    found = {}
    visited = set()
    def visit(item):
        if item.get("role") == "runtime":
            found[digest(json.dumps(item["unit"], sort_keys=True).encode())] = item
            return
        for edge in item.get("dependencies", []):
            path = edge["observation"]
            if path not in visited:
                visited.add(path)
                try:
                    dependency = load(path)
                except (OSError, ValueError, ObservationError):
                    # No unrelated consumer dependency can impose identity
                    # requirements. An absent runtime still fails uniqueness.
                    continue
                visit(dependency)
    visit(receipt)
    if len(found) != 1:
        raise ObservationError(f"expected one exact runtime dependency, observed {len(found)}")
    return next(iter(found.values()))


def emit_producer(receipt, role, out):
    runtime = receipt if role == "runtime" else find_runtime(receipt)
    recipe, captured, roots = graph_recipe(runtime)
    mode = os.environ.get("CHELIS_IDENTITY_PROVENANCE")
    if mode == "source-worktree":
        provenance = {"mode": "source_worktree", "source_root": os.environ["CHELIS_IDENTITY_WORKSPACE"], "recipe": recipe, "roots": roots}
    elif mode == "sealed-distribution":
        # The core computes the same descriptor first; sealed provenance then
        # binds that canonical closure without retaining physical source roots.
        provisional = {"mode": "source_worktree", "source_root": os.environ["CHELIS_IDENTITY_WORKSPACE"], "recipe": recipe, "roots": roots}
        initial = helper("core-derive", {"recipe": recipe, "captured": captured, "kind": role, "provenance": provisional})
        provenance = {"mode": "sealed_distribution", "source_closure": initial["descriptor"]["source"]}
    else:
        raise ObservationError("explicit source-worktree or sealed-distribution provenance is required")
    result = helper("core-derive", {"recipe": recipe, "captured": captured, "kind": role, "provenance": provenance})
    record = Path(out, "chelis_runtime_identity.bin")
    retained_provenance = Path(out, "chelis_runtime_identity_provenance.bin")
    record.write_bytes(bytes(result["record"]))
    retained_provenance.write_bytes(bytes(result["provenance"]))
    # rustc includes these bytes in the compilation this wrapper precedes, which
    # Cargo stamped before they were written. Date them to the declaration the
    # build script wrote before that stamp, or every later build is dirty.
    declaration = Path(out, "chelis-runtime-identity-producer.json")
    if declaration.exists():
        stamp = declaration.stat()
        for path in (record, retained_provenance):
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    receipt["descriptor"] = result["descriptor"]


def output_paths(real_rustc, args):
    explicit = one(args, "-o")
    if explicit:
        return [str(Path(explicit).absolute())]
    out = one(args, "--out-dir")
    if not out:
        raise ObservationError("compiler output directory is unobservable")
    emit = one(args, "--emit", "link")
    naming_args = args
    if "link" not in emit and "metadata" in emit:
        # Metadata uses the compiler's library basename even for binary and
        # cdylib checks. Ask rustc for that name; do not guess from crate names.
        naming_args = []
        index = 0
        while index < len(args):
            if args[index] == "--crate-type":
                index += 2
            elif args[index].startswith("--crate-type="):
                index += 1
            else:
                naming_args.append(args[index]); index += 1
        naming_args.append("--crate-type=rlib")
    names = probe([real_rustc, *naming_args, "--print=file-names"]).splitlines()
    outputs = [str((Path(out) / name).absolute()) for name in names]
    # Metadata-only cargo check produces .rmeta rather than the printed .rlib.
    if "link" not in emit:
        outputs = [str(Path(path).with_suffix(".rmeta")) for path in outputs if path.endswith(".rlib")]
    elif "metadata" in emit:
        outputs.extend(str(Path(path).with_suffix(".rmeta")) for path in list(outputs) if path.endswith(".rlib"))
    if not outputs:
        raise ObservationError("no observable rustc outputs")
    return outputs


def dep_info_path(args):
    for emit in values(args, "--emit"):
        for item in emit.split(","):
            kind, _, explicit = item.partition("=")
            if kind == "dep-info":
                if explicit:
                    return Path(explicit)
                directory = one(args, "--out-dir")
                extra = next((value.partition("=")[2] for value in values(args, "-C") if value.startswith("extra-filename=")), "")
                return Path(directory, one(args, "--crate-name") + extra + ".d") if directory else None
    return None


def track_inventory(receipt, dep_info):
    """Add a local unit's selected inputs to rustc's dependency metadata.

    Cargo reuses a local unit until a path in its dep-info changes, and rustc
    lists only the files it read. An added, removed or unread selected input
    would leave the receipt stale instead of rebuilding the unit. List every
    selected source and declaration input and, for an enumerated inventory,
    each directory that contains one or leads to one, so membership changes
    rebuild the unit too. Toolchain and generated inputs have their own owners.
    """
    planned = {entry["logical_path"] for entry in receipt["required_inputs"]}
    tracked = set()
    for root in receipt["roots"]:
        if root["inventory"]["class"] == "toolchain" or root.get("generated"):
            continue
        physical = Path(root["physical"])
        prefix = root["inventory"]["logical_prefix"].rstrip("/")
        for name in root["inventory"]["files"]:
            if (prefix + "/" + name if prefix else name) not in planned:
                continue
            path = physical / name
            tracked.add(path)
            while not root.get("fixed") and path != physical:
                path = path.parent
                tracked.add(path)
    lines = dep_info.read_text(encoding="utf-8").split("\n")
    rule = next((index for index, line in enumerate(lines) if ": " in line and not line.startswith("#")), None)
    if rule is None:
        raise ObservationError(f"rustc dep-info has no dependency rule: {dep_info}")
    lines[rule] += "".join(" " + str(path).replace(" ", "\\ ") for path in sorted(tracked))
    fd, temporary = tempfile.mkstemp(prefix=dep_info.name + ".", dir=dep_info.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    # Replace rather than rewrite: a cache wrapper may have restored the file as
    # a link to its shared store entry.
    os.replace(temporary, dep_info)


def build_script_launcher(helper, real):
    """Return the executable text that replaces an observed build script.

    A managed build observes the script's execution. An unmanaged Cargo that
    reuses the same target runs the original script unchanged.
    """
    helper, real = shlex.quote(str(helper)), shlex.quote(str(real))
    return ("#!/bin/sh\n"
            f'if [ "${{CHELIS_IDENTITY_PROTOCOL:-}}" != 1 ]; then exec {real} "$@"; fi\n'
            f'exec {helper} observe-build-script {real} "$@"\n')


def observe_rustc(real_rustc, args):
    command, workspace_wrapper = rustc_command(real_rustc, args)
    if not one(args, "--crate-name") or any(arg == "-" for arg in args) or any(arg.startswith("--print") for arg in args) or "--test" in args:
        return subprocess.call(command, close_fds=False)
    import_dependencies()
    outputs = output_paths(real_rustc, args)
    errors = []
    try:
        receipt = collect_unit(real_rustc, args, workspace_wrapper)
    except (OSError, KeyError, ValueError, ObservationError) as error:
        errors.append(str(error))
        receipt = {
            "protocol": 1,
            "dependencies": [],
            "outputs": [],
            "errors": errors,
            # A surface producer's own unit is outside the runtime identity
            # closure, but Cargo must still bind its uplifted native output to
            # this exact rustc target and feature selection.
            "unit": {
                "target_name": one(args, "--crate-name"),
                "features": sorted(
                    value.split('"', 2)[1]
                    for value in values(args, "--cfg")
                    if value.startswith('feature="') and value.endswith('"')
                ),
            },
        }
        for external in values(args, "--extern"):
            try:
                name, separator, artifact = external.partition("=")
                if not separator:
                    continue
                receipt["dependencies"].append({"name": name, "observation": str(receipt_for(artifact))})
            except (OSError, KeyError, IndexError, ObservationError):
                pass
    out = unit_out_dir(one(args, "--crate-name"))
    if receipt.get("errors"):
        # Observation failure stays attached to the unit, but standard links
        # metadata must still reach unrelated consumer build scripts.
        _, _, manifest, _ = package_facts()
        receipt["links"] = manifest["package"].get("links")
        if out and execution_path(out).exists():
            receipt["build_execution"] = load(execution_path(out))
    receipt["manifest_path"] = str(Path(os.environ["CARGO_MANIFEST_DIR"]).absolute() / "Cargo.toml")
    role = None
    declaration = Path(out, "chelis-runtime-identity-producer.json") if out else None
    if one(args, "--crate-name") != "build_script_build":
        if os.environ.get("CHELIS_IDENTITY_BACKEND") == "nix":
            role = os.environ.get("CHELIS_IDENTITY_ROLE") or None
        elif declaration and declaration.exists():
            declared = load(declaration)
            if declared.get("protocol") != 1:
                raise ObservationError("invalid producer declaration")
            role = declared["kind"]
            receipt["profile"] = declared["profile"]
    if role:
        receipt["role"] = role
        receipt.setdefault("profile", os.environ.get("PROFILE"))
        if receipt["profile"] not in {"debug", "release"}:
            raise ObservationError("producer PROFILE must explicitly be debug or release")
        if role == "runtime" and errors:
            raise ObservationError("; ".join(errors))
        if not out:
            raise ObservationError("producer OUT_DIR is absent")
        Path(out).mkdir(parents=True, exist_ok=True)
        emit_producer(receipt, role, out)
        if role != "runtime":
            triple = receipt.get("unit", {}).get("target", {}).get("triple") or one(args, "--target", probe([real_rustc, "-vV"]).split("host: ", 1)[1].splitlines()[0])
            symbols = ["EXPECTED_RUNTIME_RECORD", "CHELIS_BUILD_PROVENANCE_" + role.upper()]
            for symbol in symbols:
                command.extend(["-C", "link-arg=-Wl,-u,_" + symbol if "apple" in triple else "link-arg=-Wl,--undefined=" + symbol])
    code = subprocess.call(command, close_fds=False)
    if code:
        return code
    # Rustc lists only the files it read. Cargo must also rebuild a local unit
    # whose selected inputs or their membership changed, not reuse it stale.
    dep_info = dep_info_path(args)
    if (dep_info and os.environ.get("CHELIS_IDENTITY_BACKEND") == "cargo" and receipt.get("roots")
            and receipt["unit"]["package"]["source"].startswith("path:")):
        track_inventory(receipt, dep_info)
    if role:
        graph_recipe(receipt if role == "runtime" else find_runtime(receipt))
        for output in outputs:
            if is_native_producer(role, output):
                actual = json.loads(probe([os.environ["CHELIS_IDENTITY_HELPER"], "inspect", role, output]))
                if actual != receipt["descriptor"]:
                    raise ObservationError("compiler/cache output retained record differs from the independently derived record")
    path = state() / "receipts" / (key(outputs[0]) + ".json")
    if one(args, "--crate-name") == "build_script_build" and os.environ.get("CHELIS_IDENTITY_BACKEND") == "cargo":
        binary = Path(outputs[0])
        real = Path(str(binary) + ".identity-real")
        os.replace(binary, real)
        binary.write_text(build_script_launcher(os.environ["CHELIS_IDENTITY_HELPER"], real))
        binary.chmod(0o755)
        outputs.append(str(real))
    receipt["outputs"] = [{"path": output, "digest": digest(Path(output).read_bytes())} for output in outputs]
    atomic(path, receipt)
    for output in receipt["outputs"]:
        atomic(binding(output["path"]), {"artifact": output["path"], "observation": str(path)})
        atomic(state() / "output-digests" / output["digest"] / (key(path) + ".json"), {"observation": str(path)})
    return 0


def observe_output_move(previous, current):
    previous, current = Path(previous).absolute(), Path(current).absolute()
    if previous == current:
        return 0
    path = receipt_for(previous)
    receipt = load(path)
    outputs = [output for output in receipt["outputs"] if output["path"] == str(previous)]
    if receipt.get("protocol") != 1 or len(outputs) != 1 or previous.exists():
        raise ObservationError("output move does not name one observed, moved compiler output")
    if digest(current.read_bytes()) != outputs[0]["digest"]:
        raise ObservationError("moved compiler output bytes differ from their observation")
    outputs[0]["path"] = str(current)
    atomic(path, receipt)
    atomic(binding(current), {"artifact": str(current), "observation": str(path)})
    binding(previous).unlink()
    return 0


def is_native_producer(role, path):
    suffix = Path(path).suffix
    return (role == "runtime" and suffix == ".a") or (role == "python" and suffix in {".so", ".dylib", ".pyd"}) or (role == "cli" and suffix not in {".rlib", ".rmeta", ".d", ".a"})


def relocate_build_metadata(metadata, old, new):
    if not old or not new:
        if old or new:
            raise ObservationError("both build output relocation paths are required")
        return dict(metadata)
    old, new = str(Path(old).absolute()), str(Path(new).absolute())
    relocated = {}
    for name, value in metadata.items():
        if value == old or value.startswith(old + os.sep):
            destination = new + value[len(old):]
            if not Path(destination).exists():
                raise ObservationError(f"installed build metadata path is absent: {destination}")
            relocated[name] = destination
        elif re.search(r"(?<![\w./-])" + re.escape(old) + r"(?=$|/)", value):
            raise ObservationError(f"build metadata requires an unambiguous path value: {name}")
        else:
            relocated[name] = value
    return relocated


def install_observations(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--lib-dir", required=True)
    parser.add_argument("--bin-dir")
    parser.add_argument("--build-out-dir")
    parser.add_argument("--installed-build-out-dir")
    args = parser.parse_args(arguments)
    os.environ["CHELIS_IDENTITY_STATE"] = args.state
    destination = Path(args.lib_dir).absolute()
    receipt_dir = destination / "chelis-runtime-identity-receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    def install(file):
        file = str(file)
        if file in copied:
            return copied[file]
        receipt = load(file)
        if receipt.get("installed"):
            # An immutable dependency already owns its captured closure. Keep
            # that exact binding rather than duplicating every transitive DAG.
            return file
        if not receipt.get("errors"):
            check_receipt(receipt)
        token = key(file)
        target = receipt_dir / (token + ".json")
        copied[file] = str(target)
        installed = []
        for output in receipt["outputs"]:
            source = Path(output["path"])
            cwd = Path(receipt.get("cwd", Path.cwd()))
            mappings = [(cwd / "target/lib", destination / "lib"), (cwd / "target/build", destination / "lib")]
            if args.bin_dir:
                mappings.append((cwd / "target/bin", Path(args.bin_dir)))
            matches = []
            for old, new in mappings:
                if source.is_relative_to(old):
                    matches.append(new / source.relative_to(old))
            if len(matches) > 1:
                raise ObservationError(f"ambiguous output installation mapping: {source}")
            if matches:
                path = matches[0]
                if not path.is_file():
                    raise ObservationError(f"observed output was not installed: {path}")
                if is_native_producer(receipt.get("role"), path):
                    actual = json.loads(probe([os.environ["CHELIS_IDENTITY_HELPER"], "inspect", receipt["role"], str(path)]))
                    if actual != receipt["descriptor"]:
                        raise ObservationError("fixup changed retained producer record")
            else:
                # Build/compiler outputs needed only as closure evidence are
                # retained at an exact receipt-owned path, never rediscovered.
                path = receipt_dir / token / "outputs" / source.name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, path)
            installed.append({"path": str(path), "digest": digest(path.read_bytes())})
        receipt["outputs"] = installed
        receipt["dependencies"] = [{**edge, "observation": install(edge["observation"])} for edge in receipt["dependencies"]]
        if receipt.get("build_script"):
            receipt["build_script"] = install(receipt["build_script"])
        if execution := receipt.get("build_execution"):
            execution["metadata"] = relocate_build_metadata(execution["metadata"], args.build_out_dir, args.installed_build_out_dir)
        for number, root in enumerate(receipt.get("roots", [])):
            if Path(root["physical"]).is_relative_to("/nix/store"):
                # Immutable Nix inputs already have durable store ownership.
                # Retain their store reference, not a copy per unit.
                continue
            snapshot = receipt_dir / token / "inputs" / str(number)
            snapshot.mkdir(parents=True, exist_ok=True)
            required = {item["logical_path"] for item in receipt["required_inputs"]}
            prefix = root["inventory"]["logical_prefix"].rstrip("/")
            root["inventory"]["files"] = [name for name in root["inventory"]["files"] if (prefix + "/" + name if prefix else name) in required]
            for name in root["inventory"]["files"]:
                output = snapshot / name
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(root["physical"]) / name, output)
            root["physical"] = str(snapshot)
        receipt["installed"] = True
        if not receipt.get("errors"):
            check_receipt(receipt)
        atomic(target, receipt)
        return str(target)
    index = []
    # Only declared crate output roots are published. A build script may run
    # transient compiler probes below OUT_DIR; those are not installed crates.
    # Reachable auxiliary receipts are still retained recursively by install.
    output_roots = {(Path.cwd() / "target" / kind).resolve() for kind in ("lib", "bin", "build")}
    for file in sorted((state() / "receipts").glob("*.json")):
        if not any(Path(output["path"]).parent.resolve() in output_roots for output in load(file)["outputs"]):
            continue
        target = install(file)
        receipt = load(target)
        index.extend({"artifact": output["path"], "observation": target} for output in receipt["outputs"])
    if not index:
        raise ObservationError("no exact observed compiler outputs were installed")
    atomic(destination / "chelis-runtime-identity-observations.json", index)
    return 0


def validate_observation(path):
    receipt = load(path)
    runtime = receipt if receipt["role"] == "runtime" else find_runtime(receipt)
    recipe, captured, roots = graph_recipe(runtime)
    provenance = ({"mode": "sealed_distribution", "source_closure": receipt["descriptor"]["source"]} if receipt.get("installed")
                  else {"mode": "source_worktree", "source_root": os.environ["CHELIS_IDENTITY_WORKSPACE"], "recipe": recipe, "roots": roots})
    derived = helper("core-derive", {"recipe": recipe, "captured": captured, "kind": receipt["role"], "provenance": provenance})
    if derived["descriptor"] != receipt["descriptor"]:
        raise ObservationError("cached producer recipe no longer matches current runtime closure")
    for output in receipt["outputs"]:
        if digest(Path(output["path"]).read_bytes()) != output["digest"]:
            raise ObservationError("cached producer output changed")
        if is_native_producer(receipt["role"], output["path"]):
            actual = json.loads(probe([os.environ["CHELIS_IDENTITY_HELPER"], "inspect", receipt["role"], output["path"]]))
            if actual != derived["descriptor"]:
                raise ObservationError("cached producer retained record differs from independently derived expectation")
            if receipt.get("installed"):
                provenance = json.loads(probe([os.environ["CHELIS_IDENTITY_HELPER"], "inspect-provenance", receipt["role"], output["path"]]))
                if provenance != {"mode": "sealed_distribution", "source_closure": derived["descriptor"]["source"]}:
                    raise ObservationError("installed producer lacks sealed, byte-bound provenance")
    return 0


def main(arguments=None):
    args = list(sys.argv[1:] if arguments is None else arguments)
    try:
        if not args:
            raise ObservationError("missing observer command")
        if args[0] == "validate-observation" and len(args) == 2:
            return validate_observation(args[1])
        if args[0] == "observe-build-script":
            return build_execution(args[1], args[2:])
        if args[0] == "receipt-for" and len(args) == 2:
            print(receipt_for(args[1])); return 0
        if args[0] == "observe-output-move" and len(args) == 3:
            return observe_output_move(args[1], args[2])
        if args[0] == "install-observations":
            return install_observations(args[1:])
        if args[0] == "producer-artifact":
            parser = argparse.ArgumentParser()
            parser.add_argument("--lib-dir", required=True)
            parser.add_argument("--kind", required=True)
            options = parser.parse_args(args[1:])
            found = []
            for item in load(Path(options.lib_dir) / "chelis-runtime-identity-observations.json"):
                receipt = load(item["observation"])
                if receipt.get("role") != options.kind or not is_native_producer(options.kind, item["artifact"]):
                    continue
                if not receipt.get("installed") or item["artifact"] not in {output["path"] for output in receipt["outputs"]}:
                    raise ObservationError("installed index does not bind an observed native output")
                validate_observation(item["observation"])
                found.append(item["artifact"])
            if len(found) != 1:
                raise ObservationError(f"expected one installed {options.kind} output, got {found}")
            print(found[0]); return 0
        if args[0] == "observe-rustc":
            args.pop(0)
        return observe_rustc(args[0], args[1:])
    except (OSError, KeyError, ValueError, ObservationError) as error:
        print(f"runtime identity: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
