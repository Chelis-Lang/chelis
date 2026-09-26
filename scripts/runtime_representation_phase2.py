"""Execution receipt for runtime-representation Phase 2.

Phase 2 consumes the complete Phase 1 oracle, then proves the generated ABI,
opaque metadata/device ownership, Python/DLPack boundary, HIP software
contracts, and backend-header census on one unchanged source identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import uuid

from scripts import runtime_representation_phase1 as phase1
from scripts.dtype_builtin_atom_closure_oracle import source_identity

ROOT = phase1.ROOT
OracleFailure = phase1.OracleFailure
MANIFEST = ROOT / "spec/design/runtime_representation_phase2_tests.json"
MANIFEST_SHA256 = "85f36b8b5244b89856ee55dbf378504fa2e1424dae6d74c5e29a75165730c0c9"

PYTHON_BOUNDARY_BINARIES = (
    "binding_payloads",
    "capacity_census_bindings",
    "compiler_json_payloads",
    "execution_wire_facade",
    "native_device_validation",
    "native_execution_capture",
    "native_manifest_dimensions",
    "native_output_validation",
    "native_python_environment",
    "native_registration_probe",
    "native_tensor_boundary",
)
PYTHON_BOUNDARY_UNIT_TESTS = (
    "native_dlpack_owner_tests::mismatched_dlpack_request_rejects_and_releases_only_its_retained_owner",
    "native_tensor::tests::validated_owner_supports_foreign_consumer_threads",
    "tests::binding_constructor_metadata_tracks_actual_pyo3_slots",
    "tests::compiled_host_binding_executes_rank_zero_one_eight_and_nine",
    "tests::compiled_manifest_version_admission_precedes_metadata_and_library_use",
    "tests::dtype_mappings_are_exhaustive_and_reject_unknown_tags",
    "tests::element_count_reports_shapes_outside_the_int64_extent_domain",
    "tests::execution_dtype_gate_is_target_aware",
    "tests::host_dimension_backing_accepts_rank_zero_and_rank_above_eight",
    "tests::host_input_validation_accepts_rank_zero_empty_and_canonical_strides",
    "tests::host_tensor_is_opaque_and_read_view_has_the_exact_fixed_layout",
    "tests::native_shape_admission_checks_exact_int64_extents",
    "tests::unloading_an_artifact_whose_kernel_ran_does_not_abort_the_process",
)
PYTHON_BOUNDARY_FILTER = (
    "binary_id(/^chelis-python::("
    + "|".join(PYTHON_BOUNDARY_BINARIES)
    + ")$/) | test(/^("
    + "|".join(PYTHON_BOUNDARY_UNIT_TESTS)
    + ")$/)"
)


def check_options(argv):
    if "--skip-mutations" in argv or "--regenerate" in argv:
        raise OracleFailure(
            "Phase 2 requires the complete Phase 1 mutations and current execution; "
            "development shortcuts cannot pass"
        )


def phase2_legs():
    return (
        (
            "canonical ABI metadata and generated layouts",
            ("-p", "chelis-abi"),
        ),
        (
            "opaque C metadata-plan adapter",
            (
                "-p",
                "chelis-runtime",
                "--test",
                "metadata_plan_api",
                "--test",
                "metadata_plan_byte_offset",
            ),
        ),
        (
            "HIP descriptor contracts",
            (
                "-p",
                "chelis-backend-hip",
                "--test",
                "codegen_structure",
                "--test",
                "device_entry_contract",
            ),
        ),
        (
            "HIP published owner header contract",
            (
                "-p",
                "chelis-backend-hip",
                "--test",
                "device_owner_contract",
                "-E",
                "test(published_owner_is_opaque_and_packet_observation_cannot_be_mutated)",
            ),
        ),
        (
            "HIP pinned-runtime CPU execution contracts",
            (
                "-p",
                "chelis-backend-hip",
                "--test",
                "device_entry_execution",
                "--test",
                "device_owner_contract",
                "--run-ignored",
                "only",
            ),
        ),
        (
            "Python host device and DLPack boundaries",
            ("-p", "chelis-python", "-E", PYTHON_BOUNDARY_FILTER),
        ),
        (
            "backend runtime-header capacity census",
            (
                "-p",
                "chelis-cli",
                "--test",
                "capacity_census_tripwire",
                "-E",
                "test(backend_headers::)",
            ),
        ),
    )


def manual_exclusions():
    return (
        {
            "lane": "hip-hardware",
            "status": "not-executed",
            "command": (
                "scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness "
                "-- --ignored --test-threads=1"
            ),
            "reason": (
                "requires a HIP-capable device and is not replaced by the CPU SDK "
                "fixture or structural HIP execution"
            ),
        },
    )


def frozen_manifest(data, digest):
    if hashlib.sha256(data).hexdigest() != digest:
        raise OracleFailure("Phase 2 reviewed selection digest differs")
    return phase1.load_json(data.decode())


def validate_manifest(packet):
    legs = phase2_legs()
    if packet.get("schema") != 2:
        raise OracleFailure("unknown Phase 2 manifest schema")
    if packet.get("manual_exclusions") != list(manual_exclusions()):
        raise OracleFailure("Phase 2 manual hardware boundary drifted")
    python_required = packet.get("python_required")
    if (
        not isinstance(python_required, list)
        or not python_required
        or not all(isinstance(identity, str) and identity for identity in python_required)
        or len(set(python_required)) != len(python_required)
        or python_required != sorted(python_required)
    ):
        raise OracleFailure("Phase 2 Python self-test selection is empty or duplicate")
    rows = packet.get("legs")
    if not isinstance(rows, list) or len(rows) != len(legs):
        raise OracleFailure("Phase 2 leg inventory drifted")
    for row, (name, args) in zip(rows, legs, strict=True):
        if row.get("name") != name or row.get("args") != list(args):
            raise OracleFailure("Phase 2 frozen command drifted")
        required = row.get("required")
        if (
            not isinstance(required, list)
            or not required
            or not all(isinstance(identity, str) and identity for identity in required)
            or len(set(required)) != len(required)
            or required != sorted(required)
        ):
            raise OracleFailure("Phase 2 frozen identity selection is invalid")


def python_suite():
    return phase1.unittest.defaultTestLoader.loadTestsFromName(
        "scripts.test_runtime_representation_phase2"
    )


RUNTIME_DIR_REJECTING_PACKAGES = frozenset({"chelis-cli", "chelis-python"})


def leg_environment(args, archive: Path):
    """Name the pinned runtime directory for every leg whose packages still read it.

    The CLI and the Python extension carry their runtime and reject
    `CHELIS_RUNTIME_DIR`, so legs that run chelis-cli or chelis-python tests
    never receive it. The HIP harnesses still select their runtime through it
    and otherwise search for one (#1354), so every other leg receives the pin's
    directory. A leg cannot mix the two kinds of package. The export goes away
    when the HIP harnesses carry or name their runtime.
    """
    packages = {value for flag, value in zip(args, args[1:]) if flag == "-p"}
    rejecting = packages & RUNTIME_DIR_REJECTING_PACKAGES
    if not rejecting:
        return {"CHELIS_RUNTIME_DIR": str(archive.parent)}
    if packages - RUNTIME_DIR_REJECTING_PACKAGES:
        raise OracleFailure(
            f"a leg that runs {', '.join(sorted(rejecting))} tests, which reject "
            "CHELIS_RUNTIME_DIR, cannot also run "
            f"{', '.join(sorted(packages - RUNTIME_DIR_REJECTING_PACKAGES))}, "
            "which needs the pinned runtime directory"
        )
    return {}


def execute_leg(name, args, required, directory, *, environment=None):
    print(f"+ {name}: list and execute {len(required)} required tests", flush=True)
    listed = phase1.command(
        [*phase1.nextest_command("list", args), "--message-format", "json"],
        ROOT,
        directory,
        "list",
        scoped_environment=environment,
    )
    include_ignored = args[-2:] == ("--run-ignored", "only")
    selected, artifacts = phase1.selection(
        phase1.load_json(listed),
        ROOT,
        include_ignored=include_ignored,
    )
    additions = phase1.require_frozen_selection(selected, required)
    if additions:
        print(f"+ {name}: execute {len(additions)} added tests", flush=True)
        for identity in additions:
            print(f"  + {identity}", flush=True)
    junit = phase1.junit_path()
    junit.unlink(missing_ok=True)
    phase1.command(
        [
            *phase1.nextest_command("run", args),
            "--no-fail-fast",
            "--retries",
            "0",
        ],
        ROOT,
        directory,
        "run",
        scoped_environment=environment,
    )
    if not junit.is_file():
        raise OracleFailure("nextest produced no fresh Phase 2 execution receipt")
    xml = junit.read_text()
    (directory / "execution.xml").write_text(xml)
    executed = phase1.execution(xml, selected)
    phase1.verify_artifacts(artifacts)
    return {
        "name": name,
        "required": required,
        "selected": selected,
        "additions": additions,
        "executed": [{"id": identity, "outcome": "passed"} for identity in executed],
        "artifacts": artifacts,
    }


def run() -> Path:
    check_options(sys.argv[1:])
    identity = source_identity(ROOT)
    packet = frozen_manifest(MANIFEST.read_bytes(), MANIFEST_SHA256)
    validate_manifest(packet)
    run_id = str(uuid.uuid4())
    directory = ROOT / "target/runtime-representation-phase2" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    python_receipt = phase1.python_execution(python_suite(), packet["python_required"])
    phase1_receipt = phase1.run()
    executions = []
    with phase1.runtime_pin(directory / "runtime-build") as runtime_receipt:
        (archive,) = map(Path, runtime_receipt["pinned_artifact"])
        for index, (row, (name, args)) in enumerate(
            zip(packet["legs"], phase2_legs(), strict=True)
        ):
            executions.append(
                execute_leg(
                    name,
                    args,
                    row["required"],
                    directory / str(index),
                    environment=leg_environment(args, archive),
                )
            )
    if source_identity(ROOT) != identity:
        raise OracleFailure("source changed during Phase 2 execution")
    receipt = {
        "schema": 2,
        "head": identity[0],
        "source_digest": identity[1],
        "run_id": run_id,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "phase1_receipt": str(phase1_receipt),
        "runtime": runtime_receipt,
        "python": {
            **{key: value for key, value in python_receipt.items() if key != "executed"},
            "executed": [
                {"id": name, "outcome": "passed"}
                for name in python_receipt["executed"]
            ],
        },
        "manual_exclusions": list(manual_exclusions()),
        "legs": executions,
    }
    receipt_path = directory / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Current execution receipt: {receipt_path}")
    print("RUNTIME REPRESENTATION PHASE 2: PASS")
    return receipt_path
