"""Run one isolated Python wheel or editable-install consumer phase."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import chelis
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_rejected(path: Path, label: str, protected: Path | None = None) -> dict:
    before = snapshot_files(path.parent)
    protected_before = snapshot_files(protected) if protected is not None else None
    try:
        chelis.load(path)
    except chelis.ChelisError as error:
        if snapshot_files(path.parent) != before or (
            protected is not None and snapshot_files(protected) != protected_before
        ):
            raise AssertionError(f"{label} rejection changed artifact bytes or staging") from error
        return {
            "status": "rejected",
            "error_type": "ChelisError",
            "artifact_unchanged": True,
            "diagnostic": str(error),
        }
    raise AssertionError(f"{label} was admitted")


def copy_artifact(source: Path, destination: Path, manifest: dict) -> Path:
    destination.mkdir(parents=True)
    library = destination / source.name
    shutil.copy2(source, library)
    library.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if sha256(library) != sha256(source) or read_json(library.with_suffix(".json")) != manifest:
        raise AssertionError("negative-control artifact differs beyond its intended manifest")
    return library


def snapshot_files(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def different_digest(value: str) -> str:
    if not value:
        raise AssertionError("cannot mutate an empty digest")
    return ("0" if value[0] != "0" else "1") + value[1:]


def verify_artifact(library: Path, artifact_dir: Path, expected_mode: str) -> dict:
    library = library.resolve()
    artifact_dir = artifact_dir.resolve()
    manifest = read_json(library.with_suffix(".json"))
    staging = read_json(artifact_dir / "chelis_runtime.receipt.json")
    if (
        staging.get("schema") != "chelis-runtime-staging/1"
        or staging.get("mode") != expected_mode
    ):
        raise AssertionError(f"unexpected runtime staging receipt: {staging}")

    archive = artifact_dir / staging["archive"]
    archive_sha256 = sha256(archive)
    if archive_sha256 != staging.get("archive_sha256"):
        raise AssertionError("staged runtime archive differs from its receipt")
    if manifest.get("runtime_sha256") != archive_sha256:
        raise AssertionError("artifact runtime digest differs from the staged archive")

    headers = staging.get("headers")
    if not isinstance(headers, dict) or not headers:
        raise AssertionError("staging receipt has no public runtime headers")
    for name, digest in headers.items():
        header = artifact_dir / name
        if not header.is_file() or sha256(header) != digest:
            raise AssertionError(f"staged public runtime header failed its receipt: {name}")

    library_sha256 = sha256(library)
    if manifest.get("library_sha256") != library_sha256:
        raise AssertionError("compiled library differs from its artifact manifest")
    return {
        "library_path": str(library),
        "library_sha256": library_sha256,
        "runtime_archive": str(archive.resolve()),
        "runtime_archive_sha256": archive_sha256,
        "runtime_mode": staging["mode"],
        "header_count": len(headers),
        "manifest": manifest,
    }


def environment_evidence(args: argparse.Namespace) -> dict:
    work_root = args.work_root.resolve()
    checkout_root = args.checkout_root.resolve()
    checkout_probe = checkout_root / "Cargo.toml"
    checkout_read_denied = False
    if args.require_checkout_denied:
        try:
            with checkout_probe.open("rb") as source:
                source.read(1)
        except PermissionError:
            checkout_read_denied = True
        else:
            raise AssertionError(f"original checkout remains readable: {checkout_probe}")
    module_path = Path(chelis.__file__).resolve()
    if work_root.is_relative_to(checkout_root):
        raise AssertionError(f"consumer work root is inside the checkout: {work_root}")
    if Path.cwd().resolve().is_relative_to(checkout_root):
        raise AssertionError("consumer ran with the checkout as its working directory")
    if module_path.is_relative_to(checkout_root):
        raise AssertionError(f"chelis imported from the checkout: {module_path}")

    source_root = args.source_root.resolve() if args.source_root else None
    source_copy = args.source_copy.resolve() if args.source_copy else None
    if args.mode == "editable":
        if source_root is None or not source_root.is_dir():
            raise AssertionError("editable consumer has no disposable source root")
        if not module_path.is_relative_to(source_root):
            raise AssertionError(f"editable package did not import from its source: {module_path}")
    elif source_copy is not None and source_copy.exists():
        raise AssertionError(f"wheel source copy still exists during consumption: {source_copy}")
    developer_target = args.developer_target
    if args.mode != "editable" and developer_target is None:
        raise AssertionError("installed wheel consumer was not given a developer target")
    if developer_target is not None and (
        developer_target.exists() or developer_target.is_symlink()
    ):
        raise AssertionError(f"developer target is accessible during wheel execution: {developer_target}")

    neighboring_cli = shutil.which("chelis")
    if neighboring_cli is not None:
        raise AssertionError(f"a chelis executable is visible on PATH: {neighboring_cli}")
    return {
        "consumer_cwd": str(Path.cwd().resolve()),
        "checkout_read_denied": checkout_read_denied,
        "checkout_denial_probe": str(checkout_probe) if checkout_read_denied else None,
        "module_path": str(module_path),
        "source_copy_absent": source_copy is None or not source_copy.exists(),
        "developer_target_absent": developer_target is not None,
        "neighboring_cli_on_path": neighboring_cli is not None,
    }


def run_negative_controls(
    library: Path,
    artifact_dir: Path,
    manifest: dict,
    archive_sha256: str,
    source: Path,
    work_root: Path,
    runtime_dir_variable: str,
) -> dict[str, dict[str, object]]:
    negative_root = work_root / "negative"
    negative_root.mkdir()

    crossed = dict(manifest)
    crossed["runtime_sha256"] = different_digest(archive_sha256)
    crossed_library = copy_artifact(library, negative_root / "digest-mismatch", crossed)
    crossed_result = assert_rejected(crossed_library, "runtime digest mismatch", artifact_dir)

    altered = dict(manifest)
    altered["library_sha256"] = different_digest(manifest["library_sha256"])
    altered_library = copy_artifact(library, negative_root / "altered-library", altered)
    altered_result = assert_rejected(altered_library, "altered library digest", artifact_dir)

    results = {
        "runtime_digest_manifest_mismatch": crossed_result,
        "altered_library_digest": altered_result,
    }
    for field in ("runtime_sha256", "library_sha256"):
        missing = dict(manifest)
        del missing[field]
        missing_library = copy_artifact(
            library, negative_root / f"missing-{field}", missing
        )
        results[f"missing_{field}"] = assert_rejected(
            missing_library, f"missing {field}", artifact_dir
        )

    runtime_dir = work_root / "offered-runtime"
    runtime_dir.mkdir()
    (runtime_dir / "not-a-runtime").write_text("not used\n", encoding="utf-8")
    refused_artifacts = work_root / "runtime-dir-refused"
    staged_before = snapshot_files(artifact_dir)
    previous_runtime = os.environ.get(runtime_dir_variable)
    was_set = runtime_dir_variable in os.environ
    os.environ[runtime_dir_variable] = str(runtime_dir)
    try:
        try:
            chelis.compile_and_load(source, artifact_dir=refused_artifacts)
        except chelis.ChelisError as error:
            message = str(error)
        else:
            raise AssertionError(
                f"compile_and_load accepted {runtime_dir_variable}"
            )
    finally:
        if was_set:
            os.environ[runtime_dir_variable] = previous_runtime or ""
        else:
            os.environ.pop(runtime_dir_variable, None)
    if refused_artifacts.exists():
        raise AssertionError("runtime-directory refusal created staging artifacts")
    if snapshot_files(artifact_dir) != staged_before:
        raise AssertionError("runtime-directory refusal changed the existing artifact")
    results["set_runtime_directory_before_staging"] = {
        "status": "rejected",
        "error_type": "ChelisError",
        "staging_unchanged": True,
        "diagnostic": message,
    }
    return results


def compile_and_call(args: argparse.Namespace) -> dict:
    environment = environment_evidence(args)
    work_root = args.work_root.resolve()
    source_dir = work_root / "source"
    artifact_dir = work_root / "artifacts"
    source_dir.mkdir(parents=True, exist_ok=True)
    source = source_dir / "model.ch"
    source.write_text(
        "def main(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)\n",
        encoding="utf-8",
    )
    expected = np.asarray([0.0, 0.25, 1.5, 0.0], dtype=np.float32)
    inputs = np.asarray([-2.0, 0.25, 1.5, -4.0], dtype=np.float32)

    model = chelis.compile_and_load(source, artifact_dir=artifact_dir)
    observed = model(inputs).numpy()
    np.testing.assert_array_equal(observed, expected)
    expected_mode = "development" if args.mode == "editable" else "sealed"
    artifact = verify_artifact(Path(model.path), artifact_dir, expected_mode)
    manifest = artifact.pop("manifest")
    negative_controls = run_negative_controls(
        Path(artifact["library_path"]),
        artifact_dir,
        manifest,
        artifact["runtime_archive_sha256"],
        source,
        work_root,
        args.runtime_dir_variable,
    )

    result = {
        "schema": "chelis-python-wheel-consumer/2",
        "phase": "editable_compile_and_call" if args.mode == "editable" else "compile_and_call",
        "runtime_mode": artifact["runtime_mode"],
        "runtime_archive_sha256": artifact["runtime_archive_sha256"],
        "library_path": artifact["library_path"],
        "library_sha256": artifact["library_sha256"],
        "observed_value": observed.tolist(),
        "expected_value": expected.tolist(),
        "negative_controls": negative_controls,
        "environment": environment,
    }

    if args.mode == "editable":
        runtime_source = args.source_root.resolve() / "crates/chelis-runtime/src/lib.rs"
        original = runtime_source.read_bytes()
        changed_artifacts = work_root / "stale-source-artifacts"
        staged_before = snapshot_files(artifact_dir)
        try:
            runtime_source.write_bytes(original + b"\n// disposable freshness probe\n")
            if runtime_source.read_bytes() == original:
                raise AssertionError("declared runtime source was not changed")
            try:
                chelis.compile_and_load(source, artifact_dir=changed_artifacts)
            except chelis.ChelisError as error:
                message = str(error)
            else:
                raise AssertionError("editable extension accepted changed runtime sources")
        finally:
            runtime_source.write_bytes(original)
        if changed_artifacts.exists():
            raise AssertionError("stale editable build created staging artifacts")
        if runtime_source.read_bytes() != original:
            raise AssertionError("editable freshness probe did not restore its source copy")
        if snapshot_files(artifact_dir) != staged_before:
            raise AssertionError("stale-source rejection changed the existing artifact")
        result["freshness_mutation"] = {
            "declared_source": "crates/chelis-runtime/src/lib.rs",
            "changed_without_rebuild": True,
            "rejected": True,
            "error_type": "ChelisError",
            "diagnostic": message,
            "source_restored": True,
        }

    return result


def reload_persisted(args: argparse.Namespace) -> dict:
    environment = environment_evidence(args)
    compile_result = read_json(args.compile_result.resolve())
    if compile_result.get("schema") != "chelis-python-wheel-consumer/2":
        raise AssertionError(f"unexpected compile receipt: {compile_result}")
    library = Path(compile_result["library_path"]).resolve()
    artifact_dir = Path(compile_result["artifact_dir"]).resolve()
    source_file = Path(compile_result["source_file"]).resolve()
    if source_file.exists():
        raise AssertionError(f"original source still exists before persisted reload: {source_file}")

    artifact = verify_artifact(library, artifact_dir, "sealed")
    for field, key in (
        ("library_sha256", "library_sha256"),
        ("runtime_archive_sha256", "runtime_archive_sha256"),
    ):
        if artifact[field] != compile_result[key]:
            raise AssertionError(f"persisted artifact {field} changed after compilation")

    # Validate both digests from disk before chelis.load can open the library.
    model = chelis.load(library)
    inputs = np.asarray([-2.0, 0.25, 1.5, -4.0], dtype=np.float32)
    expected = np.asarray([0.0, 0.25, 1.5, 0.0], dtype=np.float32)
    observed = model(inputs).numpy()
    np.testing.assert_array_equal(observed, expected)
    return {
        "schema": "chelis-python-wheel-consumer/2",
        "phase": "persisted_reload",
        "runtime_mode": artifact["runtime_mode"],
        "runtime_archive_sha256": artifact["runtime_archive_sha256"],
        "library_sha256": artifact["library_sha256"],
        "observed_value": observed.tolist(),
        "expected_value": expected.tolist(),
        "original_source_absent": True,
        "digest_admission_verified_before_load": True,
        "environment": environment,
    }


def verify_crossed_bundle(args: argparse.Namespace) -> dict:
    environment = environment_evidence(args)
    library = args.library.resolve()
    artifact_dir = args.artifact_dir.resolve()
    artifact = verify_artifact(library, artifact_dir, "sealed")
    carried_runtime = args.expected_runtime_sha256
    linked_runtime = artifact["runtime_archive_sha256"]
    if not carried_runtime or linked_runtime == carried_runtime:
        raise AssertionError("crossed-wheel runtimes are not distinct")

    rejection = assert_rejected(library, "crossed sealed wheel")
    return {
        "schema": "chelis-python-wheel-consumer/2",
        "phase": "crossed_sealed_wheel_rejection",
        "consumer_runtime_sha256": carried_runtime,
        "linked_runtime_sha256": linked_runtime,
        "library_sha256": artifact["library_sha256"],
        "rejected": True,
        "rejection": rejection,
        "environment": environment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("editable", "compile", "reload", "crossed-load"), required=True
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--require-checkout-denied", action="store_true")
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--developer-target", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-copy", type=Path)
    parser.add_argument("--compile-result", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--expected-runtime-sha256")
    parser.add_argument("--runtime-dir-variable", required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    if args.mode in ("editable", "compile"):
        result = compile_and_call(args)
    elif args.mode == "reload":
        result = reload_persisted(args)
    else:
        result = verify_crossed_bundle(args)
    if args.mode == "compile":
        result["artifact_dir"] = str((args.work_root / "artifacts").resolve())
        result["source_file"] = str((args.work_root / "source/model.ch").resolve())
    write_json(args.result.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
