"""Capture native collection measurements without Charon or an LLBC input."""

import argparse
import json
import os
from pathlib import Path
import shlex
import re
import shutil
import subprocess
import sys

if __package__:
    from . import native_collection_certificate as certificate
else:
    import native_collection_certificate as certificate


def resolve_tool(name):
    found = shutil.which(name)
    if found is None:
        raise ValueError("The native tool is absent: " + name)
    return Path(found).resolve(strict=True)


def run(command, *, source_root, output, label, environment, disk_guard=None):
    """Retain actual arguments, environment, status, tools, and output."""
    command = [str(value) for value in command]
    log = output / (label + ".log")
    executable = Path(command[0]).resolve(strict=True)
    before = certificate.identity(executable)
    launch_command = command
    guard_identity = None
    if disk_guard is not None:
        guard_identity = certificate.identity(disk_guard)
        launch_command = [sys.executable, guard_identity["path"], *command]
    with log.open("xb") as handle:
        result = subprocess.run(launch_command, cwd=source_root, env=environment,
                                stdout=handle, stderr=subprocess.STDOUT)
    after = certificate.identity(executable)
    recorded_environment = dict(environment)
    for name in certificate.RUST_FLAGS:
        recorded_environment.setdefault(name, None)
    record = {
        "command": command, "cwd": str(source_root), "exit": result.returncode,
        "log": certificate.identity(log), "environment": recorded_environment,
        "executable_before": before, "executable_after": after,
        "launch_command": launch_command, "disk_guard": guard_identity,
    }
    receipt = certificate.save(output / (label + "-command.json"), record)
    if before != after:
        raise ValueError("The native command executable changed: " + label)
    if result.returncode != 0:
        raise RuntimeError("The native command failed. Its receipt remains at " + receipt["path"])
    return receipt, log


def root_package(metadata, package, source_root):
    matches = [item for item in metadata["packages"]
               if item["name"] == package and item["id"] in metadata["workspace_members"]]
    if len(matches) != 1:
        raise ValueError("The native root has no unique workspace package.")
    result = matches[0]
    if not Path(result["manifest_path"]).resolve().is_relative_to(source_root):
        raise ValueError("The native root package uses another source tree.")
    return result


def library_file(unit):
    libraries = [Path(name) for name in unit["filenames"] if name.endswith(".rlib")]
    if len(libraries) != 1:
        raise ValueError("The native supplier has no unique library file.")
    return libraries[0]


def rust_type_expression(value):
    """Render one supplied complete descriptor as a typed Rust expression."""
    kind = value["kind"]
    if kind == "scalar":
        if value["name"] not in ("bool", "char", "u8", "u16", "u32", "u64", "u128", "usize",
                                 "i8", "i16", "i32", "i64", "i128", "isize", "f32", "f64"):
            raise ValueError("The requested scalar type is invalid.")
        return value["name"]
    if kind == "tuple":
        parts = [rust_type_expression(item) for item in value["types"]]
        return "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"
    if kind != "adt" or value["consts"]:
        raise ValueError("The requested complete Rust descriptor is unsupported.")
    name = value["name"]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*", name):
        raise ValueError("The requested Rust type path is invalid.")
    arguments = [rust_type_expression(item) for item in value["types"]]
    return name + ("<" + ", ".join(arguments) + ">" if arguments else "")


def measurement_requests(path, *, source_root, package, features):
    if path is None:
        return [], None
    entry = certificate.identity(path)
    plan = certificate.load(entry)
    if (plan.get("version") != 1 or Path(plan["source_root"]).resolve() != source_root
            or plan["package"] != package or sorted(plan["features"]) != features):
        raise ValueError("The measurement request uses another source, package, or feature profile.")
    for obligation in plan["obligations"]:
        certificate.checked(obligation)
    seen = set()
    for request in plan["types"]:
        descriptor = certificate.key(request["rust_type"])
        if descriptor in seen:
            raise ValueError("The measurement request contains a duplicate complete type.")
        seen.add(descriptor)
        expression = rust_type_expression(request["rust_type"])
        if request["method"] not in ("public", "private"):
            raise ValueError("The requested native measurement method is invalid.")
        if "standard_library_supplier" in request:
            if request["supplier_package"] != "rustc" or request["method"] != "public":
                raise ValueError("The standard-library route needs an explicit public compiler supplier.")
            certificate.standard_library_declarations(request["standard_library_supplier"],
                request["rust_type"], request["native_type"], require_sources=True)
        else:
            if request["method"] == "public" and request["native_type"] != expression:
                raise ValueError("A public measurement label differs from its complete Rust type.")
            if request["supplier_package"] == "rustc" and request["rust_type"]["kind"] != "scalar":
                raise ValueError("The compiler scalar supplier cannot bind an ADT.")
    return plan["types"], entry


def retain_artifact(path):
    """Preserve actual Cargo identities through immutable hashed cache artifacts."""
    original = certificate.identity(path)
    source = Path(path)
    retained = original
    if (source.parent / "deps").is_dir():
        candidates = (source.parent / "deps").glob(source.stem + "-*" + source.suffix)
        for candidate in candidates:
            if candidate.stat().st_size == original["bytes"]:
                value = certificate.identity(candidate)
                if value["sha256"] == original["sha256"]:
                    retained = value
                    break
        if retained == original:
            raise ValueError("The mutable Cargo artifact has no authentic hashed cache artifact: " + str(source))
    return {"original": original, "retained": retained}


def native_names(path):
    names = {}
    pattern = re.compile(r"^print-type-size type: `(.+)`: [0-9]+ bytes, alignment: [0-9]+ bytes$")
    for line in Path(path).read_text().splitlines():
        match = pattern.fullmatch(line)
        if match:
            name = match.group(1)
            names[name] = names.get(name, 0) + 1
    return names


def save_cache(path, value):
    temporary = path.with_suffix(".new")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def capture_profile(*, source_root, package, features, target_dir, output,
                    manually_drop_source=None, measurements=None, disk_guard=None):
    """Capture exact measurements with one authenticated task-owned Cargo cache."""
    source_root = Path(source_root).resolve(strict=True)
    target_dir, output = Path(target_dir).resolve(), Path(output).resolve()
    features = sorted(set(features))
    if package not in certificate.ROOT_PACKAGES:
        raise ValueError("The native root is not a retained package.")
    if features and (features != ["lowering-trace"] or package not in ("chelis-ir", "chelis-pipeline-core")):
        raise ValueError("The native feature request is outside the production profile matrix.")
    extra_requests, measurements_id = measurement_requests(
        measurements, source_root=source_root, package=package, features=features)
    if disk_guard is None:
        raise ValueError("Native capture requires the approved disk guard path.")
    certificate.identity(disk_guard)
    wrapper_source = None
    if package in ("chelis-ir", "chelis-pipeline-core"):
        if manually_drop_source is None:
            raise ValueError("IR measurements require the authenticated native ManuallyDrop source path.")
        wrapper_source = certificate.identity(manually_drop_source)
        if wrapper_source["sha256"] != certificate.MANUALLY_DROP_SHA256:
            raise ValueError("The native ManuallyDrop source differs from the approved compiler pin.")
    if output.exists():
        raise ValueError("The native capture output directory must be new and exclusive.")
    if target_dir == output or target_dir.is_relative_to(source_root / "crates") or output.is_relative_to(source_root / "crates"):
        raise ValueError("Native artifacts must remain outside production crate sources.")
    before = certificate.source_manifest(source_root)
    configuration_before = certificate.configuration_manifest(source_root)
    rustc, cargo = resolve_tool("rustc"), resolve_tool("cargo")
    environment = dict(os.environ)
    rustflags = {name: environment.get(name) for name in certificate.RUST_FLAGS}
    environment.update({"RUSTC": str(rustc), "RUSTC_BOOTSTRAP": "1", "RUSTC_WRAPPER": "",
                        "RUSTC_WORKSPACE_WRAPPER": "", "CARGO_INCREMENTAL": "0"})
    cache_configuration = {
        "source_root": str(source_root), "source": before, "compiler": certificate.identity(rustc),
        "cargo": certificate.identity(cargo), "target": certificate.NATIVE_TARGET,
        "profile": "dev", "rustflags": rustflags, "cargo_configuration": configuration_before,
    }
    cache_path = target_dir / "native-collection-cache.json"
    if target_dir.exists():
        if not cache_path.is_file():
            raise ValueError("The target directory is not this task's authenticated native cache.")
        cache = json.loads(cache_path.read_text())
        if cache.get("version") != 1 or cache["configuration"] != cache_configuration:
            raise ValueError("The native cache source, compiler, target, profile, or flags differ.")
    else:
        target_dir.mkdir(parents=True, exist_ok=False)
        cache = {"version": 1, "configuration": cache_configuration, "captures": []}
        certificate.save(cache_path, cache)
    output.mkdir(parents=True, exist_ok=False)
    before_id = certificate.save(output / "core-before.json", before)
    sources = {name: {"bytes": (source_root / name).stat().st_size, "sha256": digest}
               for name, digest in before["files"].items()}
    sources_id = certificate.save(output / "root-sources.json", sources)
    configuration_before_id = certificate.save(output / "configuration-before.json", configuration_before)
    commands = {}

    def execute(label, command, *, heavy=False):
        entry, log = run(command, source_root=source_root, output=output, label=label,
                         environment=environment, disk_guard=disk_guard if heavy else None)
        commands[label] = entry
        return log

    version = execute("native-version", [rustc, "-vV"])
    for expected in ("release: " + certificate.NATIVE_RELEASE,
                     "commit-hash: " + certificate.NATIVE_COMMIT,
                     "host: " + certificate.NATIVE_TARGET):
        if expected not in version.read_text().splitlines():
            raise ValueError("The native compiler identity differs from the approved profile.")
    metadata_command = [cargo, "metadata", "--format-version", "1", "--locked",
                        "--filter-platform", certificate.NATIVE_TARGET]
    if features:
        metadata_command += ["--features", ",".join(package + "/" + value for value in features)]
    metadata_log = execute("native-metadata", metadata_command)
    metadata = json.loads(metadata_log.read_text())
    root = root_package(metadata, package, source_root)
    root_command = [cargo, "rustc", "--package", package, "--lib", "--locked", "--jobs", "2",
                    "--target", certificate.NATIVE_TARGET, "--target-dir", target_dir,
                    "--message-format=json"]
    if features:
        root_command += ["--features", ",".join(features)]
    root_command += ["--", "-Zprint-type-sizes"]
    native_log = execute("native-root", root_command, heavy=True)
    events = certificate.artifact_events(native_log)
    units = [certificate.artifact_unit(event) for event in events]
    certificate.library_unit(units, root["id"])
    metadata_packages = {item["id"]: item for item in metadata["packages"]}
    for unit in units:
        if unit["package_id"] not in metadata_packages:
            raise ValueError("A native artifact has no actual Cargo metadata supplier.")
    artifact_records = {name: retain_artifact(name) for event in events for name in event["filenames"]}
    profile = {
        "cargo_profile": "dev", "target": certificate.NATIVE_TARGET, "rustflags": rustflags,
        "packages": units, "root_package_id": root["id"],
        "metadata": certificate.identity(metadata_log), "native_artifacts": events,
        "unit_identity_fields": ["package_id", "crate_name", "features", "target_kind",
                                 "profile", "filenames", "source_path"],
    }
    profile_id = certificate.save(output / "feature-chain.json", profile)
    requests = [{
        "rust_type": certificate.scalar("usize"), "native_type": "usize",
        "supplier_package": "rustc", "method": "public",
    }]
    if package in ("chelis-ir", "chelis-pipeline-core"):
        lowered = certificate.adt("chelis_ir::lower::LoweredValue")
        requests += [
            {"rust_type": certificate.adt("chelis_ir::host::HostExpr",
                                         [certificate.adt("chelis_ir::host_type_state::HostTypeTerm")]),
             "native_type": "chelis_ir::host::HostExpr<chelis_ir::host_type_state::HostTypeTerm>",
             "supplier_package": "chelis-ir", "method": "public"},
            {"rust_type": lowered, "native_type": "lower::LoweredValue",
             "supplier_package": "chelis-ir", "method": "private"},
            {"rust_type": {"kind": "tuple", "types": [certificate.scalar("usize"), lowered]},
             "native_type": "std::mem::ManuallyDrop<(usize, lower::LoweredValue)>",
             "supplier_package": "chelis-ir", "method": "private"},
        ]
    descriptors = {certificate.key(item["rust_type"]): item for item in requests}
    for request in extra_requests:
        descriptor = certificate.key(request["rust_type"])
        if descriptor in descriptors:
            existing = descriptors[descriptor]
            if any(existing[name] != request[name] for name in ("rust_type", "native_type", "supplier_package", "method")):
                raise ValueError("An additional request would replace a known authentic measurement route.")
        else:
            descriptors[descriptor] = request
            requests.append(request)
    suppliers = {}
    for request in requests:
        name = request["supplier_package"]
        if name == "rustc":
            continue
        candidates = [item for item in metadata["packages"] if item["name"] == name
                      and any(unit["package_id"] == item["id"] and unit["target_kind"] == ["lib"] for unit in units)]
        if len(candidates) != 1:
            raise ValueError("A requested type has no unique actual native supplier: " + name)
        suppliers[name] = certificate.library_unit(units, candidates[0]["id"])
    private_measurements = {}
    private_paths = {}
    for index, (name, supplier) in enumerate(suppliers.items()):
        required = [item["native_type"] for item in requests
                    if item["supplier_package"] == name and item["method"] == "private"]
        if not required:
            continue
        selected = None
        counts = native_names(native_log)
        if supplier["package_id"] == root["id"]:
            if any(counts.get(value, 0) > 1 for value in required):
                raise ValueError("The native supplier output contains ambiguous measurement rows.")
            if all(counts.get(value) == 1 for value in required):
                selected = {"command": "native-root", "core_before": before_id, "core_after": None}
                private_paths[name] = native_log
        if selected is None:
            for entry in cache["captures"]:
                _, prior, prior_receipt, prior_source, prior_profile = certificate.native_evidence(certificate.checked(entry))
                if (prior_source != before or prior["native"]["compiler"] != certificate.identity(rustc)
                        or prior_profile["rustflags"] != rustflags or prior_profile["target"] != certificate.NATIVE_TARGET):
                    continue
                matching = [unit for unit in prior_profile["packages"] if unit["package_id"] == supplier["package_id"]
                            and unit["target_kind"] == ["lib"]]
                if len(matching) != 1:
                    continue
                try:
                    certificate.compatible_supplier(supplier, matching[0])
                except ValueError:
                    continue
                measurement = prior_receipt["private_measurements"].get(supplier["package_id"])
                if measurement is None:
                    continue
                command_id = prior_receipt["commands"][measurement["command"]]
                command, log_id = certificate.successful_command(command_id)
                counts = native_names(certificate.checked(log_id))
                if not all(counts.get(value) == 1 for value in required):
                    continue
                label = "native-supplier-" + str(index)
                commands[label] = command_id
                selected = {**measurement, "command": label}
                private_paths[name] = certificate.checked(log_id)
                for artifact in prior_receipt["artifact_files"]:
                    artifact_records.setdefault(artifact["original"]["path"], artifact)
                break
        if selected is None:
            label = "native-supplier-" + str(index)
            supplier_command = [cargo, "rustc", "--package", name, "--lib", "--locked", "--jobs", "2",
                                "--target", certificate.NATIVE_TARGET, "--target-dir", target_dir,
                                "--message-format=json", "--no-default-features"]
            if supplier["features"]:
                supplier_command += ["--features", ",".join(supplier["features"])]
            supplier_command += ["--", "-Zprint-type-sizes"]
            log = execute(label, supplier_command, heavy=True)
            supplier_events = certificate.artifact_events(log)
            actual_units = [certificate.artifact_unit(event) for event in supplier_events]
            certificate.compatible_supplier(supplier, certificate.library_unit(actual_units, supplier["package_id"]))
            for event in supplier_events:
                for filename in event["filenames"]:
                    artifact_records[filename] = retain_artifact(filename)
            counts = native_names(log)
            if not all(counts.get(value) == 1 for value in required):
                raise ValueError("The native supplier did not measure every exact requested private type.")
            selected = {"command": label, "core_before": before_id, "core_after": None}
            private_paths[name] = log
        private_measurements[supplier["package_id"]] = selected
    standard_suppliers = {}
    standard_requests = [item for item in requests if "standard_library_supplier" in item]
    sysroot = None
    if standard_requests:
        directory_log = execute("native-target-libdir",
            [rustc, "--print", "target-libdir", "--target", certificate.NATIVE_TARGET])
        sysroot_log = execute("native-sysroot",
            [rustc, "--print", "sysroot", "--target", certificate.NATIVE_TARGET])
        directories, sysroots = directory_log.read_text().splitlines(), sysroot_log.read_text().splitlines()
        if (len(directories) != 1 or not Path(directories[0]).is_absolute()
                or len(sysroots) != 1 or not Path(sysroots[0]).is_absolute()):
            raise ValueError("The actual native standard-library directory queries are invalid.")
        directory, sysroot = Path(directories[0]), Path(sysroots[0])
        for request in standard_requests:
            supplier = dict(request["standard_library_supplier"])
            artifacts = {}
            for field, name in (("artifact", supplier["crate_name"]),
                                ("public_artifact", request["native_type"].split("::", 1)[0])):
                candidates = sorted(directory.glob("lib" + name + "-*.rlib"))
                if len(candidates) != 1:
                    raise ValueError("The standard supplier has no unique actual target library artifact.")
                artifacts[field] = certificate.identity(candidates[0])
            supplier.update(artifacts)
            supplier.update({"compiler": certificate.identity(rustc), "target": certificate.NATIVE_TARGET,
                "target_libdir_command": commands["native-target-libdir"],
                "sysroot_command": commands["native-sysroot"]})
            standard_suppliers[certificate.key(request["rust_type"])] = supplier
    source = output / "public-layouts.rs"
    externs = {unit["crate_name"]: unit for unit in suppliers.values()}
    text = "".join("extern crate " + name + ";\n" for name in sorted(externs))
    if any(item["native_type"].startswith("alloc::") for item in standard_requests):
        text += "extern crate alloc;\n"
    text += "fn main() {\n"
    for request in requests:
        if request["method"] == "public":
            expression = (request["native_type"] if "standard_library_supplier" in request
                          else rust_type_expression(request["rust_type"]))
            label = json.dumps("print-type-size type: `" + request["native_type"] + "`: {} bytes, alignment: {} bytes")
            text += " println!(" + label + ", std::mem::size_of::<" + expression + ">(), std::mem::align_of::<" + expression + ">());\n"
    source.write_text(text + "}\n")
    flags = (rustflags["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
             if rustflags["CARGO_ENCODED_RUSTFLAGS"] is not None
             else shlex.split(rustflags["RUSTFLAGS"] or ""))
    binary = output / "public-layouts"
    public_command = [rustc, "--edition=2024", "--target", certificate.NATIVE_TARGET, *flags]
    if sysroot is not None:
        public_command += ["--sysroot", sysroot]
    for name, supplier in sorted(externs.items()):
        public_command += ["--extern", name + "=" + str(library_file(supplier))]
    directories = sorted({str(Path(filename).parent) for unit in units for filename in unit["filenames"]
                          if filename.endswith((".rlib", ".rmeta", ".dylib", ".so"))})
    for directory in directories:
        public_command += ["-L", "dependency=" + directory]
    public_command += [source, "-o", binary]
    execute("native-public-compile", public_command, heavy=True)
    public_log = execute("native-public-layouts", [binary])
    wrapper_id = None
    if wrapper_source is not None:
        wrapper_path = output / "native-manually-drop.rs"
        shutil.copyfile(certificate.checked(wrapper_source), wrapper_path)
        wrapper_id = certificate.identity(wrapper_path)
    rows = []
    for request in requests:
        name = request["supplier_package"]
        supplier_id = "rustc" if name == "rustc" else suppliers[name]["package_id"]
        log = public_log if request["method"] == "public" else private_paths[name]
        row = certificate.measured_row(request["rust_type"], request["native_type"], supplier_id, log)
        if "standard_library_supplier" in request:
            row["standard_library_supplier"] = standard_suppliers[certificate.key(request["rust_type"])]
        if request["native_type"] == "std::mem::ManuallyDrop<(usize, lower::LoweredValue)>":
            row["evidence"].update({
                "manually_drop_source": wrapper_id,
                "wrapper_type": "std::mem::ManuallyDrop<(usize, chelis_ir::lower::LoweredValue)>",
                "inner_type": request["rust_type"],
            })
        rows.append(row)
    after = certificate.source_manifest(source_root)
    after_id = certificate.save(output / "core-after.json", after)
    configuration_after = certificate.configuration_manifest(source_root)
    configuration_after_id = certificate.save(output / "configuration-after.json", configuration_after)
    if before != after or configuration_before != configuration_after:
        raise ValueError("The native source or Cargo configuration changed during capture.")
    for measurement in private_measurements.values():
        if measurement["core_after"] is None:
            measurement["core_after"] = after_id
    receipt = {
        "core_before": before_id, "core_after": after_id, "core_stable": True,
        "root_sources": sources_id, "feature_chain": profile_id,
        "configuration_before": configuration_before_id, "configuration_after": configuration_after_id,
        "commands": commands, "public_probe_source": certificate.identity(source),
        "artifact_files": list(artifact_records.values()), "private_measurements": private_measurements,
        "manually_drop_source": wrapper_id, "measurement_requests": measurements_id,
        "cache_configuration": cache_configuration,
    }
    receipt_id = certificate.save(output / "native-profile-capture.json", receipt)
    measured = {
        "format": certificate.PROFILE_FORMAT, "version": 1,
        "native": {"release": certificate.NATIVE_RELEASE, "commit": certificate.NATIVE_COMMIT,
                   "target": certificate.NATIVE_TARGET, "compiler": certificate.identity(rustc),
                   "version_output": certificate.identity(version)},
        "root_sources": {"root": str(source_root), "manifest": sources_id},
        "feature_chain": {"manifest": profile_id, "configuration": profile},
        "native_capture": receipt_id, "rows": rows,
    }
    measured_id = certificate.save(output / "native-profile.json", measured)
    certificate.native_evidence(measured_id["path"])
    cache["captures"].append(measured_id)
    save_cache(cache_path, cache)
    return {"native_profile": measured_id, "feature_chain": profile_id, "capture": receipt_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--package", choices=certificate.ROOT_PACKAGES, required=True)
    parser.add_argument("--features", default="")
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manually-drop-source", type=Path)
    parser.add_argument("--measurements", type=Path,
                        help="Exact consumed type requests with actual obligation identities")
    parser.add_argument("--disk-guard", type=Path, required=True,
                        help="Approved disk guard for each native compile command")
    args = parser.parse_args()
    result = capture_profile(source_root=args.source_root, package=args.package,
                             features=args.features.replace(",", " ").split(),
                             target_dir=args.target_dir, output=args.out,
                             manually_drop_source=args.manually_drop_source,
                             measurements=args.measurements, disk_guard=args.disk_guard)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
