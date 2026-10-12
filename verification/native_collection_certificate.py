"""Authenticate native profiles and bind every measured row to an extraction job."""

import hashlib
import gzip
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

NATIVE_RELEASE = "1.98.0"
NATIVE_COMMIT = "88d9e12ae178fab0fb5cc050a94da85685d449ea"
NATIVE_TARGET = "aarch64-apple-darwin"
MANUALLY_DROP_SHA256 = "c21100b507893a11ee27b0711be3ac3fe6cc970b96bba55a7d3bae5978209818"
STANDARD_SOURCE_SHA256 = "5a9993673ce40804e6b50300df59ecb3d13942c1c89cd11e76acf705b331f982"
PROFILE_FORMAT = "chelis-native-collection-profile"
ROOT_PACKAGES = (
    "chelis-unord", "chelis-vocab", "chelis-deep", "chelis-pred",
    "chelis-types", "chelis-effects", "chelis-ir", "chelis-pipeline-core",
)
RUST_FLAGS = ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS")


def identity(path):
    path = Path(path).absolute()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked(entry):
    path = Path(entry["path"])
    if not path.is_absolute() or identity(path) != entry:
        raise ValueError("An artifact differs from its recorded identity: " + str(path))
    return path


def load(entry):
    return json.loads(checked(entry).read_text())


def save(path, value):
    path = Path(path).absolute()
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return identity(path)


def key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def source_manifest(source_root):
    """Use the existing core source scope for Git trees and source archives."""
    root = Path(source_root).resolve(strict=True)
    if (root / ".git").exists():
        names = subprocess.check_output(
            ["git", "ls-files", "-co", "--exclude-standard", "--", "crates",
             "Cargo.toml", "Cargo.lock", "rust-toolchain.toml"], cwd=root, text=True,
        ).splitlines()
    else:
        names = [str(path.relative_to(root)) for path in (root / "crates").rglob("*")
                 if path.is_file()]
        names += ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]
    files = {}
    digest = hashlib.sha256()
    for name in sorted(names):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("A source path leaves its root.")
        path = root / relative
        if path.is_file():
            value = identity(path)
            files[name] = value["sha256"]
            digest.update(name.encode() + b"\0" + value["sha256"].encode() + b"\n")
    return {"source_tree_sha256": digest.hexdigest(), "files": files}


def source_files(entry, root):
    sources = load(entry)
    for name, value in sources.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("A source path leaves its root.")
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError("A source supplier leaves its root.")
        checked({"path": str(path), **value})
    return {name: value["sha256"] for name, value in sources.items()}


def options(arguments, name):
    values = []
    for index, argument in enumerate(arguments):
        if argument == name:
            if index + 1 == len(arguments):
                raise ValueError("An option has no value: " + name)
            values.append(arguments[index + 1])
        elif argument.startswith(name + "="):
            values.append(argument[len(name) + 1:])
    return values


def one_option(arguments, name, *, required=False):
    values = options(arguments, name)
    if len(values) > 1 or (required and not values):
        raise ValueError("An option has no unique value: " + name)
    return values[0] if values else None


def feature_requests(arguments):
    return sorted({feature for value in options(arguments, "--features")
                   for feature in value.replace(",", " ").split()})


def successful_command(entry):
    record = load(entry)
    if record.get("exit", record.get("returncode")) != 0 or record.get("status", "ok") != "ok":
        raise ValueError("The command did not succeed.")
    if not record.get("command") or not Path(record["cwd"]).is_absolute():
        raise ValueError("The command lacks actual arguments or an absolute source directory.")
    log = record["log"]
    log_path = checked(log) if isinstance(log, dict) else Path(log)
    if not log_path.is_absolute() or not log_path.is_file():
        raise ValueError("The command log has no absolute file identity.")
    return record, identity(log_path)


def artifact_events(log):
    events = []
    for line in Path(log).read_text().splitlines():
        if line.startswith("{"):
            event = json.loads(line)
            if event.get("reason") == "compiler-artifact":
                events.append(event)
    return events


def artifact_unit(event):
    return {
        "package_id": event["package_id"], "crate_name": event["target"]["name"],
        "features": sorted(event["features"]), "target_kind": event["target"]["kind"],
        "profile": event["profile"], "filenames": event["filenames"],
        "source_path": event["target"]["src_path"],
    }


def library_unit(units, package_id):
    matches = [unit for unit in units if unit["package_id"] == package_id
               and unit["target_kind"] == ["lib"]]
    if len(matches) != 1:
        raise ValueError("The native supplier has no unique library artifact unit.")
    return matches[0]


def compatible_supplier(expected, actual):
    fields = ("package_id", "crate_name", "features", "target_kind", "profile", "source_path")
    if any(expected[field] != actual[field] for field in fields):
        raise ValueError("The private measurement uses a different native supplier configuration.")


def scalar(name):
    return {"kind": "scalar", "name": name}


def adt(name, arguments=()):
    return {"kind": "adt", "name": name, "types": list(arguments), "consts": []}


def measured_row(rust_type, native_type, source_package, path):
    expression = re.compile(r"^print-type-size type: `" + re.escape(native_type)
                            + r"`: ([0-9]+) bytes, alignment: ([0-9]+) bytes$")
    matches = [(number, match) for number, line in enumerate(Path(path).read_text().splitlines(), 1)
               if (match := expression.fullmatch(line))]
    if len(matches) != 1:
        raise ValueError("The native type has no unique measured row: " + native_type)
    number, match = matches[0]
    return {"rust_type": rust_type, "native_type": native_type, "source_package": source_package,
            "size": int(match[1]), "alignment": int(match[2]),
            "evidence": {"artifact": identity(path), "line": number}}


def configuration_manifest(source_root):
    root = Path(source_root).resolve()
    paths = []
    for directory in (root, *root.parents):
        paths += [directory / ".cargo" / name for name in ("config", "config.toml")]
    home = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))).absolute()
    paths += [home / name for name in ("config", "config.toml")]
    return {str(path): identity(path) if path.is_file() else None for path in paths}


def standard_library_declarations(supplier, descriptor, public_type, *, require_sources):
    """Authenticate canonical declaration ownership and exact native source bytes."""
    crate = supplier["crate_name"]
    if (crate not in ("alloc", "core", "std") or descriptor["kind"] != "adt"
            or descriptor["types"] or descriptor["consts"]
            or descriptor["name"].split("::", 1)[0] != crate):
        raise ValueError("The standard supplier does not own this complete declaration.")
    if not re.fullmatch(r"(?:alloc|core|std)(?:::[A-Za-z_][A-Za-z0-9_]*)+", public_type):
        raise ValueError("The explicit standard public Rust type is invalid.")
    archive = checked(supplier["source_archive"])
    if supplier["source_archive"]["sha256"] != STANDARD_SOURCE_SHA256:
        raise ValueError("The standard source archive differs from the native compiler pin.")
    download, log = successful_command(supplier["source_archive_command"])
    url = "https://codeload.github.com/rust-lang/rust/tar.gz/" + NATIVE_COMMIT
    if (download["command"][-1] != url
            or one_option(download["command"], "--output", required=True) != str(archive)
            or url + " 200" not in checked(log).read_text()):
        raise ValueError("The standard source archive lacks its actual pinned download.")
    sources = supplier["declaration_sources"]
    if len(sources) != 1:
        raise ValueError("The complete standard declaration needs one exact source.")
    entry = sources[0]
    declaration = load(entry["declaration_evidence"])
    metadata = declaration["declaration"]["metadata"]
    source_file = declaration["declaration"]["source_file"]
    parts = descriptor["name"].split("::")
    relative = "library/" + crate + "/src/" + "/".join(parts[1:-1]) + ".rs"
    llbc_file = "/rustc/" + relative
    member = "rust-" + NATIVE_COMMIT + "/" + relative
    if (entry["crate_name"] != crate or entry["llbc_file"] != llbc_file
            or entry["archive_member"] != member or declaration["declaring_crate"] != crate
            or source_file["crate_name"] != crate or source_file["name"] != {"Local": llbc_file}
            or metadata["is_local"] is not False or metadata["attr_info"]["public"] is not True
            or metadata["lang_item"] != parts[-1] or declaration["public_rust_path"] != public_type
            or declaration["canonical_name"] != descriptor["name"]
            or declaration["canonical_rust_type"] != descriptor):
        raise ValueError("The standard declaration source or public spelling differs.")
    checked(declaration["llbc"])
    with tarfile.open(archive, mode="r|gz") as stream:
        for selected in stream:
            if selected.name == member:
                if not selected.isfile():
                    raise ValueError("The standard declaration archive member is not a file.")
                data = stream.extractfile(selected).read()
                break
        else:
            raise ValueError("The standard declaration archive member is absent.")
    if not re.search(rb"(?m)^pub struct " + parts[-1].encode() + rb"\b", data):
        raise ValueError("The native archive does not declare the public standard type.")
    if require_sources and checked(entry["source"]).read_bytes() != data:
        raise ValueError("The standard declaration source differs from its archive member.")
    return [(entry, data)]


def standard_library_supplier(row, native, public_command, public_log):
    """Authenticate actual compiler queries, target artifacts, and public measurement."""
    supplier = row["standard_library_supplier"]
    if row["source_package"] != "rustc" or supplier["compiler"] != native["compiler"]:
        raise ValueError("The standard supplier uses another compiler.")
    if supplier["target"] != native["target"]:
        raise ValueError("The standard supplier uses another target.")
    compiler = checked(supplier["compiler"])
    target = supplier["target"]
    query, log = successful_command(supplier["target_libdir_command"])
    if query["command"] != [str(compiler), "--print", "target-libdir", "--target", target]:
        raise ValueError("The standard target directory query uses another compiler or target.")
    lines = checked(log).read_text().splitlines()
    if len(lines) != 1 or not Path(lines[0]).is_absolute():
        raise ValueError("The actual standard target directory output is invalid.")
    directory = Path(lines[0])
    for field, crate in (("artifact", supplier["crate_name"]), ("public_artifact", row["native_type"].split("::", 1)[0])):
        artifact = checked(supplier[field])
        candidates = sorted(directory.glob("lib" + crate + "-*.rlib"))
        if len(candidates) != 1 or artifact != candidates[0] or artifact.parent != directory:
            raise ValueError("The standard supplier has no exact actual target library artifact.")
    query, log = successful_command(supplier["sysroot_command"])
    if query["command"] != [str(compiler), "--print", "sysroot", "--target", target]:
        raise ValueError("The standard sysroot query uses another compiler or target.")
    sysroot = checked(log).read_text().splitlines()
    if (len(sysroot) != 1 or not Path(sysroot[0]).is_absolute()
            or directory != Path(sysroot[0]) / "lib/rustlib" / target / "lib"
            or one_option(public_command, "--sysroot", required=True) != sysroot[0]
            or one_option(public_command, "--target", required=True) != target
            or row["evidence"]["artifact"] != public_log):
        raise ValueError("The public standard measurement uses another compiler target supplier.")
    return standard_library_declarations(supplier, row["rust_type"], row["native_type"], require_sources=True)


def native_evidence(profile_path):
    """Authenticate native-only evidence without a compiler or native command."""
    evidence_id = identity(profile_path)
    measured = load(evidence_id)
    if measured.get("format") != PROFILE_FORMAT or measured.get("version") != 1:
        raise ValueError("The native measurement record uses a different format.")
    native = measured["native"]
    if (native["release"], native["commit"], native["target"]) != (
            NATIVE_RELEASE, NATIVE_COMMIT, NATIVE_TARGET):
        raise ValueError("The native compiler profile differs from the approved profile.")
    checked(native["compiler"])
    version = checked(native["version_output"]).read_text().splitlines()
    for expected in ("release: " + NATIVE_RELEASE, "commit-hash: " + NATIVE_COMMIT,
                     "host: " + NATIVE_TARGET):
        if expected not in version:
            raise ValueError("The native version output differs from the compiler identity.")
    profile = load(measured["feature_chain"]["manifest"])
    if profile != measured["feature_chain"]["configuration"]:
        raise ValueError("The native configuration differs from its manifest.")
    if profile["cargo_profile"] != "dev" or profile["target"] != NATIVE_TARGET:
        raise ValueError("The native Cargo profile or target differs.")
    receipt = load(measured["native_capture"])
    before, after = load(receipt["core_before"]), load(receipt["core_after"])
    if receipt["core_stable"] is not True or before != after:
        raise ValueError("The native capture lacks stable source evidence.")
    root = Path(measured["root_sources"]["root"])
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("The native source root is not canonical and absolute.")
    if source_files(measured["root_sources"]["manifest"], root) != before["files"]:
        raise ValueError("The native source supplier differs from its source manifest.")
    for field, expected in (("feature_chain", measured["feature_chain"]["manifest"]),
                            ("root_sources", measured["root_sources"]["manifest"] )):
        if receipt[field] != expected:
            raise ValueError("The native capture and profile identities differ: " + field)
    if load(receipt["configuration_before"]) != load(receipt["configuration_after"]):
        raise ValueError("The native Cargo configuration changed.")
    for name, entry in load(receipt["configuration_after"]).items():
        if entry is None:
            if Path(name).exists():
                raise ValueError("A native Cargo configuration file appeared.")
        else:
            checked(entry)
    commands = {}
    logs = {}
    for name, entry in receipt["commands"].items():
        command, log = successful_command(entry)
        if Path(command["cwd"]).resolve() != root:
            raise ValueError("A native command uses another source root.")
        commands[name], logs[name] = command, log
        for name_flag in RUST_FLAGS:
            if command["environment"][name_flag] != profile["rustflags"][name_flag]:
                raise ValueError("The native flags differ from the actual command.")
    root_command = commands["native-root"]
    if one_option(root_command["command"], "--target", required=True) != NATIVE_TARGET:
        raise ValueError("The native target differs from the actual command.")
    if "--release" in root_command["command"] or options(root_command["command"], "--profile"):
        raise ValueError("The native command does not use the dev profile.")
    if logs["native-version"] != native["version_output"]:
        raise ValueError("The native version output lacks its actual command identity.")
    if commands["native-version"]["command"] != [native["compiler"]["path"], "-vV"]:
        raise ValueError("The version command uses another native compiler.")
    for command in commands.values():
        executable = checked(command["executable_before"])
        if command["executable_before"] != command["executable_after"]:
            raise ValueError("A native command executable changed.")
        if Path(command["command"][0]).resolve() != executable:
            raise ValueError("A native command uses another executable.")
    for name, command in commands.items():
        if name == "native-root" or name.startswith("native-supplier"):
            compiler = command["environment"]["RUSTC"]
            if identity(compiler) != native["compiler"]:
                raise ValueError("A native Cargo command uses another compiler.")
    if identity(commands["native-public-compile"]["command"][0]) != native["compiler"]:
        raise ValueError("The public probe uses another native compiler.")
    events = artifact_events(checked(logs["native-root"]))
    if events != profile["native_artifacts"]:
        raise ValueError("The native artifact events differ from their command output.")
    units = [artifact_unit(event) for event in events]
    if profile["packages"] != units:
        raise ValueError("The feature units differ from the native artifact suppliers.")
    metadata = load(profile["metadata"])
    if profile["metadata"] != logs["native-metadata"]:
        raise ValueError("The metadata lacks its actual command identity.")
    package = one_option(root_command["command"], "--package", required=True)
    roots = [item for item in metadata["packages"]
             if item["id"] == profile["root_package_id"] and item["name"] == package]
    if len(roots) != 1:
        raise ValueError("The native root package differs from its actual command.")
    library_unit(units, profile["root_package_id"])
    for entry in receipt["artifact_files"]:
        retained = entry["retained"]
        checked(retained)
        original = entry["original"]
        if (original["sha256"], original["bytes"]) != (retained["sha256"], retained["bytes"]):
            raise ValueError("A retained Cargo artifact differs from its actual command artifact.")
    public_source = checked(receipt["public_probe_source"])
    public_command = commands["native-public-compile"]["command"]
    if str(public_source) not in public_command:
        raise ValueError("The public probe source differs from its actual command.")
    public_binary = one_option(public_command, "-o", required=True)
    if commands["native-public-layouts"]["command"] != [public_binary]:
        raise ValueError("The public layout output uses another probe binary.")
    private_logs = {}
    metadata_packages = {item["id"]: item for item in metadata["packages"]}
    for supplier_id, measurement in receipt["private_measurements"].items():
        name = measurement["command"]
        supplier_command = commands[name]["command"]
        expected_unit = library_unit(units, supplier_id)
        supplier_name = metadata_packages[supplier_id]["name"]
        if one_option(supplier_command, "--package", required=True) != supplier_name:
            raise ValueError("The private measurement uses another supplier package.")
        if one_option(supplier_command, "--target", required=True) != NATIVE_TARGET:
            raise ValueError("The private measurement uses another target.")
        if "--release" in supplier_command or options(supplier_command, "--profile"):
            raise ValueError("The private measurement does not use the dev profile.")
        supplier_units = [artifact_unit(event) for event in artifact_events(checked(logs[name]))]
        compatible_supplier(expected_unit, library_unit(supplier_units, supplier_id))
        if load(measurement["core_before"]) != before or load(measurement["core_after"]) != before:
            raise ValueError("The private measurement uses another source identity.")
        private_logs[supplier_id] = logs[name]
    seen = set()
    for row in measured["rows"]:
        descriptor = key(row["rust_type"])
        if descriptor in seen:
            raise ValueError("A complete Rust type has duplicate native suppliers.")
        seen.add(descriptor)
        proof = row["evidence"]
        raw = measured_row(row["rust_type"], row["native_type"], row["source_package"], checked(proof["artifact"]))
        if any(raw[field] != row[field] for field in ("size", "alignment")) or raw["evidence"]["line"] != proof["line"]:
            raise ValueError("A layout row differs from its raw native supplier.")
        if type(row["size"]) is not int or type(row["alignment"]) is not int or row["size"] < 0:
            raise ValueError("A native layout value is invalid.")
        if row["alignment"] <= 0 or row["alignment"] & (row["alignment"] - 1):
            raise ValueError("A native alignment is invalid.")
        if row["source_package"] == "rustc":
            if "standard_library_supplier" in row:
                supplier = row["standard_library_supplier"]
                if (supplier["target_libdir_command"] != receipt["commands"]["native-target-libdir"]
                        or supplier["sysroot_command"] != receipt["commands"]["native-sysroot"]):
                    raise ValueError("The standard supplier lacks its actual capture commands.")
                standard_library_supplier(row, native, public_command, logs["native-public-layouts"])
            elif row["rust_type"]["kind"] != "scalar" or proof["artifact"] != logs["native-public-layouts"]:
                raise ValueError("The compiler scalar row lacks its actual public measurement.")
        else:
            if "standard_library_supplier" in row:
                raise ValueError("A Cargo ADT cannot use the standard-library supplier route.")
            supplier = library_unit(units, row["source_package"])
            private_log = private_logs.get(row["source_package"])
            if proof["artifact"] == logs["native-public-layouts"]:
                libraries = [name for name in supplier["filenames"] if name.endswith(".rlib")]
                link = supplier["crate_name"] + "=" + libraries[0] if len(libraries) == 1 else None
                if link is None or link not in options(public_command, "--extern"):
                    raise ValueError("The public probe uses another native supplier library.")
            if proof["artifact"] not in (private_log, logs["native-public-layouts"]):
                raise ValueError("The row lacks its actual native measurement command.")
        if row["native_type"] == "std::mem::ManuallyDrop<(usize, lower::LoweredValue)>":
            checked(proof["manually_drop_source"])
            if (proof["manually_drop_source"]["sha256"] != MANUALLY_DROP_SHA256
                    or proof["wrapper_type"] != "std::mem::ManuallyDrop<(usize, chelis_ir::lower::LoweredValue)>"
                    or proof["inner_type"] != row["rust_type"]):
                raise ValueError("The transparent wrapper binds a different source or inner type.")
    if receipt["measurement_requests"] is not None:
        requests = load(receipt["measurement_requests"])
        if (requests.get("version") != 1 or Path(requests["source_root"]).resolve() != root
                or requests["package"] != package
                or sorted(requests["features"]) != feature_requests(root_command["command"])):
            raise ValueError("The measurement request differs from the authentic native profile.")
        for obligation in requests["obligations"]:
            checked(obligation)
        available = {key(row["rust_type"]): row for row in measured["rows"]}
        for request in requests["types"]:
            row = available.get(key(request["rust_type"]))
            if row is None or row["native_type"] != request["native_type"]:
                raise ValueError("An exact requested native measurement is absent.")
            supplier_name = request["supplier_package"]
            if supplier_name == "rustc":
                if row["source_package"] != "rustc":
                    raise ValueError("A requested scalar measurement uses another supplier.")
                if "standard_library_supplier" in request:
                    expected = request["standard_library_supplier"]
                    actual = row["standard_library_supplier"]
                    if any(actual[name] != expected[name] for name in
                           ("crate_name", "source_archive", "source_archive_command", "declaration_sources")):
                        raise ValueError("The standard measurement request differs from its actual supplier.")
                    standard_library_declarations(expected, request["rust_type"],
                                                  request["native_type"], require_sources=True)
            elif metadata_packages[row["source_package"]]["name"] != supplier_name:
                raise ValueError("A requested measurement uses another actual Cargo supplier.")
    return evidence_id, measured, receipt, before, profile


def charon_arguments(record, log_identity):
    arguments = record["command"]
    matches = [arguments[index:] for index in range(len(arguments) - 1)
               if Path(arguments[index]).name == "charon" and arguments[index + 1] == "cargo"]
    if not matches:
        for line in checked(log_identity).read_text().splitlines():
            if line.startswith("+ "):
                command = shlex.split(line[2:])
                if len(command) >= 2 and Path(command[0]).name == "charon" and command[1] == "cargo":
                    matches.append(command)
    if len(matches) != 1:
        raise ValueError("The extraction record has no unique actual Charon command.")
    return matches[0]


def command_file(value, cwd):
    path = Path(value)
    return path if path.is_absolute() else cwd / path


def bind_certificate(*, profile_certificate, llbc, extraction_command, core_before,
                     core_after, feature_chain, output):
    """Bind all authenticated native rows to the actual successful extraction."""
    evidence_id, measured, native_receipt, native_sources, profile = native_evidence(profile_certificate)
    root = Path(measured["root_sources"]["root"])
    before_id, after_id = identity(core_before), identity(core_after)
    before, after = load(before_id), load(after_id)
    if before != after or before != native_sources or source_manifest(root) != before:
        raise ValueError("The extraction and native capture do not share stable sources.")
    profile_id = identity(feature_chain)
    if load(profile_id) != profile:
        raise ValueError("The extraction requires a different native feature configuration.")
    command_id = identity(extraction_command)
    command, log_id = successful_command(command_id)
    arguments = charon_arguments(command, log_id)
    if "--" not in arguments:
        raise ValueError("The actual Charon command contains no Cargo arguments.")
    separator = arguments.index("--")
    charon, cargo = arguments[:separator], arguments[separator + 1:]
    cwd = Path(command["cwd"]).resolve()
    if cwd != root:
        raise ValueError("The extraction does not use the authenticated native source root.")
    llbc_id = identity(llbc)
    destination = identity(command_file(one_option(charon, "--dest-file", required=True), cwd))
    if (destination["sha256"], destination["bytes"]) != (llbc_id["sha256"], llbc_id["bytes"]) or not llbc_id["bytes"]:
        raise ValueError("The job LLBC differs from the actual nonempty Charon destination.")
    native_command, _ = successful_command(native_receipt["commands"]["native-root"])
    package = one_option(cargo, "--package", required=True)
    if package != one_option(native_command["command"], "--package", required=True):
        raise ValueError("The native root package differs from the extraction package.")
    features = feature_requests(cargo)
    if features != feature_requests(native_command["command"]):
        raise ValueError("The extraction and native capture request different features.")
    for flag in ("--no-default-features", "--all-features"):
        if (flag in cargo) != (flag in native_command["command"]):
            raise ValueError("The extraction and native capture use different feature flags.")
    if "--release" in cargo or options(cargo, "--profile"):
        raise ValueError("The extraction does not use the schema 1 dev profile.")
    target = one_option(cargo, "--target")
    if target is not None and target != profile["target"]:
        raise ValueError("The extraction uses another target.")
    for name in RUST_FLAGS:
        if name not in command["environment"] or command["environment"][name] != profile["rustflags"][name]:
            raise ValueError("The extraction lacks matching flag evidence: " + name)
    roots = options(charon, "--start-from")
    roots_file = one_option(charon, "--start-from-file")
    if roots and roots_file:
        raise ValueError("The extraction combines inline roots and a root file.")
    roots_id = identity(command_file(roots_file, cwd)) if roots_file else None
    shell_file = one_option(charon, "--opaque-hard-file")
    shell_id = identity(command_file(shell_file, cwd)) if shell_file else None
    opaque_file = one_option(charon, "--opaque-file")
    opaque_id = identity(command_file(opaque_file, cwd)) if opaque_file else None
    mir = one_option(charon, "--mir", required=True)
    output = Path(output).absolute()
    output.mkdir(exist_ok=False)
    if roots_id is None:
        path = output / "roots.txt"
        path.write_text("".join(value + "\n" for value in roots))
        roots_id = identity(path)
    if shell_id is None:
        path = output / "native-shell.opaque"
        path.write_text("")
        shell_id = identity(path)
    receipt = {
        "llbc_input": llbc_id, "core_before": before_id, "core_after": after_id,
        "core_stable": True, "feature_chain": profile_id,
        "root_sources": measured["root_sources"]["manifest"], "roots": roots_id,
        "roots_mode": "file" if roots_file else "inline" if roots else "whole-crate",
        "native_shell": shell_id, "native_shell_applied": shell_file is not None,
        "opaque_file": opaque_id, "includes": options(charon, "--include"), "mir": mir,
        "package": package, "features": features, "target": target,
        "extraction_command": command_id, "extraction_log": log_id, "extraction_flags": arguments,
        "native_command": native_receipt["commands"]["native-root"],
        "public_probe_source": native_receipt["public_probe_source"],
        "public_probe_command": native_receipt["commands"]["native-public-compile"],
        "native_profile_evidence": evidence_id, "native_capture": measured["native_capture"],
        "measurement_scope": "authenticated-profile-rows",
    }
    receipt_id = save(output / "source-bound-capture.json", receipt)
    certificate = {
        "version": 1, "native": measured["native"], "llbc_input": llbc_id,
        "root_sources": measured["root_sources"],
        "feature_chain": {"manifest": profile_id, "configuration": profile},
        "extraction_receipt": receipt_id, "rows": measured["rows"],
    }
    certificate_id = save(output / "native-collection-layouts.json", certificate)
    return {"certificate": certificate_id, "capture": receipt_id, "native_profile_evidence": evidence_id}


def add_profiles_argument(parser):
    parser.add_argument("--native-profiles", type=Path,
                        default=os.environ.get("NATIVE_COLLECTION_PROFILES"),
                        help="Manifest of authentic package-specific native profiles")


def select_profile(*, source_root, package, features, profiles=None):
    """Select exact source, package, and feature evidence without fallback."""
    selected = profiles if profiles is not None else os.environ.get("NATIVE_COLLECTION_PROFILES")
    if selected is None:
        raise ValueError("An explicit native profile manifest is required.")
    manifest_path = Path(selected).resolve(strict=True)
    manifest = load(identity(manifest_path))
    if manifest.get("version") != 1 or not isinstance(manifest.get("profiles"), list):
        raise ValueError("The native profile selection manifest is malformed.")
    root = Path(source_root).resolve(strict=True)
    requested = sorted(set(features))
    matches = [entry for entry in manifest["profiles"]
               if Path(entry["source_root"]).resolve() == root and entry["package"] == package
               and sorted(entry["features"]) == requested]
    if len(matches) != 1:
        raise ValueError("The job has no unique authentic native profile: " + package + " " + key(requested))
    profile_path = checked(matches[0]["native_profile"])
    evidence_id, measured, receipt, _, profile = native_evidence(profile_path)
    command, _ = successful_command(receipt["commands"]["native-root"])
    if (Path(measured["root_sources"]["root"]) != root
            or one_option(command["command"], "--package", required=True) != package
            or feature_requests(command["command"]) != requested):
        raise ValueError("The selected profile differs from the actual source, package, or features.")
    return {"profile_certificate": profile_path,
            "feature_chain": checked(measured["feature_chain"]["manifest"]),
            "native_profile_evidence": evidence_id}


def llbc_bytes_identity(path):
    """Authenticate original LLBC bytes, including retained gzip artifacts."""
    path = Path(path)
    if path.suffix != ".gz":
        value = identity(path)
        return value["bytes"], value["sha256"]
    digest = hashlib.sha256()
    count = 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            count += len(block)
            digest.update(block)
    return count, digest.hexdigest()


def verify_certificate(result, llbc):
    """Authenticate retained job evidence and its actual imported LLBC bytes."""
    certificate = load(result["certificate"])
    receipt = load(result["capture"])
    evidence_id, measured, native_receipt, native_sources, profile = native_evidence(
        checked(result["native_profile_evidence"]))
    if certificate.get("version") != 1 or certificate["extraction_receipt"] != result["capture"]:
        raise ValueError("The job certificate and capture identities differ.")
    imported = llbc_bytes_identity(llbc)
    expected = certificate["llbc_input"]
    if imported != (expected["bytes"], expected["sha256"]) or receipt["llbc_input"] != expected:
        raise ValueError("The retained certificate binds another LLBC.")
    if (receipt["native_profile_evidence"] != evidence_id
            or certificate["native"] != measured["native"]
            or certificate["root_sources"] != measured["root_sources"]
            or certificate["rows"] != measured["rows"]
            or load(certificate["feature_chain"]["manifest"]) != profile
            or certificate["feature_chain"]["configuration"] != profile):
        raise ValueError("The job does not retain every authenticated native row and profile identity.")
    before, after = load(receipt["core_before"]), load(receipt["core_after"])
    if before != after or before != native_sources or receipt["core_stable"] is not True:
        raise ValueError("The retained extraction source manifests differ.")
    for name in ("roots", "native_shell"):
        checked(receipt[name])
    if receipt["opaque_file"] is not None:
        checked(receipt["opaque_file"])
    command, log = successful_command(receipt["extraction_command"])
    arguments = charon_arguments(command, log)
    if "--" not in arguments:
        raise ValueError("The retained Charon command lacks Cargo arguments.")
    separator = arguments.index("--")
    charon, cargo = arguments[:separator], arguments[separator + 1:]
    native_command, _ = successful_command(native_receipt["commands"]["native-root"])
    root = Path(measured["root_sources"]["root"])
    if Path(command["cwd"]).resolve() != root:
        raise ValueError("The retained extraction uses another source root.")
    package = one_option(cargo, "--package", required=True)
    features = feature_requests(cargo)
    if (package != one_option(native_command["command"], "--package", required=True)
            or package != receipt["package"] or features != receipt["features"]
            or features != feature_requests(native_command["command"])):
        raise ValueError("The retained package or feature identity differs from its native profile.")
    if "--release" in cargo or options(cargo, "--profile"):
        raise ValueError("The retained extraction does not use the dev profile.")
    for flag in ("--no-default-features", "--all-features"):
        if (flag in cargo) != (flag in native_command["command"]):
            raise ValueError("The retained extraction uses different feature flags.")
    for name in RUST_FLAGS:
        if name not in command["environment"] or command["environment"][name] != profile["rustflags"][name]:
            raise ValueError("The retained extraction uses different Rust flags.")
    target = one_option(cargo, "--target")
    if target != receipt["target"] or (target is not None and target != profile["target"]):
        raise ValueError("The retained extraction uses another target.")
    if one_option(charon, "--mir", required=True) != receipt["mir"]:
        raise ValueError("The retained MIR identity differs.")
    destination = command_file(one_option(charon, "--dest-file", required=True), root)
    if destination.absolute() != Path(expected["path"]):
        if destination.is_file() and llbc_bytes_identity(destination) == imported:
            pass
        else:
            raise ValueError("The retained extraction destination binds another LLBC.")
    if charon_arguments(command, log) != receipt["extraction_flags"]:
        raise ValueError("The retained extraction arguments differ from the actual command.")
    if log != receipt["extraction_log"]:
        raise ValueError("The retained extraction log differs from its identity.")
    return certificate
