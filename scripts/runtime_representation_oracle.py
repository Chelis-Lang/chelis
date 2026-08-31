#!/usr/bin/env python3
"""Authoritative executable oracle for chelis#893 runtime representation phases.

Phase 0 freezes a source-derived inventory, proves that its transition debt can
only shrink, runs two controlled detector mutations, and executes the release
reproducers and landed positive receipts. Later phases extend this same runner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "spec/design/runtime_representation_phase0_inventory.json"
DESIGN_PATH = REPO_ROOT / "spec/design/runtime_representation.md"
DIRECT_ACCESS_MUTATION_SOURCE = Path("crates/chelis-runtime/src/decimal_parse.rs")
DTYPE_MUTATION_SOURCE = Path("crates/chelis-vocab/src/lib.rs")

# This is the reviewed Phase 0 foundation digest. Updating it is a freeze move,
# not a regeneration step: spec/design/runtime_representation.md B1 requires a
# design amendment and a mutation whenever it changes.
FOUNDATION_SHA256 = "bc7b89f99e1c4190304e081284fc79743903df921c5df33bee7bcaea3289e230"

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".h", ".metal", ".rs"}
SOURCE_PREFIXES = (
    "crates/chelis-runtime/src/",
    "crates/chelis-runtime/include/",
    "crates/chelis-vocab/src/",
    "crates/chelis-ir/src/",
    "crates/chelis-python/src/",
    "crates/chelis-backend-c/src/",
    "crates/chelis-backend-hip/src/",
    "crates/chelis-backend-metal/src/",
)
BACKEND_PREFIXES = (
    "crates/chelis-backend-c/src/",
    "crates/chelis-backend-hip/src/",
    "crates/chelis-backend-metal/src/",
)

DECLARATION_PATTERNS = (
    re.compile(r"\b(?:struct|enum|union)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?"
        r"(?:extern\s+\"C\"\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"
    ),
    re.compile(r"^\s*macro_rules!\s+([A-Za-z_][A-Za-z0-9_]*)"),
)

DIRECT_DATA_RE = re.compile(r"(?:\.data\b|->data\b)")
RAW_POINTER_RE = re.compile(
    r"(?:as\s+\*(?:mut|const)\s+(?:f(?:16|32|64)|i(?:8|16|32|64)|u(?:8|16|32|64)|half::(?:f16|bf16)|Bool8)\b"
    r"|\*(?:mut|const)\s+(?:f(?:16|32|64)|i(?:8|16|32|64)|u(?:8|16|32|64)|half::(?:f16|bf16)|Bool8)\b"
    r"|\(\s*(?:const\s+)?(?:float|double|u?int(?:8|16|32|64)_t)\s*\*\s*\))"
)
FIXED_RANK_RE = re.compile(
    r"(?:CHELIS_MAX_DIM|\bMAX_DIM\b|\[(?:i32|i64|int|int32_t|int64_t)\s*;\s*(?:[2-9][0-9]*|[A-Z][A-Z0-9_]*)\])"
)
NARROW_METADATA_RE = re.compile(
    r"(?:\b(?:rank|ndim|size|storage_size)\s*:\s*i32\b"
    r"|\bint(?:32_t)?\s+[^;\"']*(?:rank|ndim|size|storage_size)\b"
    r"|(?:rank|ndim|size|storage_size)[^;\"']*\bi32::try_from)"
)
WIDTH_ARITHMETIC_RE = re.compile(
    r"(?:byte_width|dtype_size|elem(?:ent)?_size|byte_capacity|checked_mul|saturating_mul)"
)
BACKEND_SPELLING_RE = re.compile(
    r'\"[^\"\n]*(?:float|double|half|__half|hip_bfloat16|bool|u?int(?:8|16|32|64)_t)[^\"\n]*\"'
)
LOAD_STORE_RE = re.compile(
    r"(?i)(?:load|store|read|write).*(?:format!|\.line\(|pointer|ptr|\[[^]]+\]|->data|\.data)"
)
DESCRIPTOR_FIELDS = {
    "data",
    "dtype",
    "shape",
    "strides",
    "size",
    "byte_capacity",
    "rank",
    "ndim",
    "storage_size",
}
RUST_STRUCT_RE = re.compile(
    r"\bstruct\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\{(?P<body>.*?)\}",
    re.DOTALL,
)
C_STRUCT_RE = re.compile(
    r"\btypedef\s+struct(?:\s+[A-Za-z_][A-Za-z0-9_]*)?\s*"
    r"\{(?P<body>.*?)\}\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;",
    re.DOTALL,
)


class OracleFailure(RuntimeError):
    """A failed runtime-representation oracle obligation."""


@dataclass(frozen=True, order=True)
class InventoryRow:
    kind: str
    path: str
    owner: str
    signature: str
    occurrence: int
    deletion_phase: int

    @property
    def identity(self) -> str:
        digest = hashlib.sha256(self.signature.encode("utf-8")).hexdigest()[:20]
        return (
            f"kind={self.kind}|path={self.path}|owner={self.owner}|"
            f"signature={digest}|occurrence={self.occurrence}"
        )

    def to_baseline_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "kind": self.kind,
            "path": self.path,
            "owner": self.owner,
            "occurrence": self.occurrence,
            "deletion_phase": self.deletion_phase,
        }


@dataclass(frozen=True)
class OracleLeg:
    name: str
    argv: tuple[str, ...]


def _tracked_source_paths(root: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8")
        if not relative.startswith(SOURCE_PREFIXES):
            continue
        if PurePosixPath(relative).suffix not in SOURCE_SUFFIXES:
            continue
        if PurePosixPath(relative).name.endswith("_tests.rs"):
            continue
        paths.append(Path(relative))
    return tuple(sorted(paths))


def _owner_after_line(line: str, owner: str) -> str:
    for pattern in DECLARATION_PATTERNS:
        match = pattern.search(line)
        if match is not None:
            return match.group(1)
    return owner


def _normalized_signature(line: str) -> str:
    return " ".join(line.strip().split())


def descriptor_owner_names(source: str) -> set[str]:
    owners: set[str] = set()
    for match in RUST_STRUCT_RE.finditer(source):
        fields = set(
            re.findall(
                r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?"
                r"([A-Za-z_][A-Za-z0-9_]*)\s*:",
                match.group("body"),
            )
        )
        if {"data", "dtype"} <= fields and len(fields & DESCRIPTOR_FIELDS) >= 4:
            owners.add(match.group("name"))
    for match in C_STRUCT_RE.finditer(source):
        fields = set(
            re.findall(
                r"(?:^|;)\s*[^;{}]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*"
                r"(?:\[[^]]*\])?\s*(?=;|$)",
                match.group("body"),
            )
        )
        if {"data", "dtype"} <= fields and len(fields & DESCRIPTOR_FIELDS) >= 4:
            owners.add(match.group("name"))
    return owners


def _deletion_phase(kind: str, path: str, owner: str) -> int:
    if kind in {"dtype-contract", "normalized-key-arithmetic", "width-arithmetic"}:
        return 1
    if kind in {"fixed-rank-metadata", "narrow-metadata"}:
        return 2
    if path.startswith("crates/chelis-python/src/"):
        return 2
    if path.startswith(BACKEND_PREFIXES) or kind in {
        "backend-element-spelling",
        "load-store-template",
    }:
        return 4
    if kind == "descriptor-field" and owner == "ChelisGpuTensor":
        return 2
    return 3


def _line_kinds(
    path: str, owner: str, line: str, descriptor_owners: set[str]
) -> tuple[str, ...]:
    stripped = line.strip()
    if not stripped or stripped.startswith(("//", "///", "//!", "*", "/*")):
        return ()

    kinds: list[str] = []
    is_backend = path.startswith(BACKEND_PREFIXES)
    is_python = path.startswith("crates/chelis-python/src/")
    is_runtime = path.startswith("crates/chelis-runtime/")

    if (
        owner in descriptor_owners
        and re.match(r"^(?:pub\s+)?[A-Za-z_][A-Za-z0-9_]*\s*:", stripped)
    ) or (
        path.startswith("crates/chelis-runtime/include/")
        and "typedef struct" in stripped
        and any(name in stripped for name in descriptor_owners)
    ):
        kinds.append("descriptor-field")

    if DIRECT_DATA_RE.search(stripped) and (is_runtime or is_python or is_backend):
        kinds.append("direct-data-access")
    if RAW_POINTER_RE.search(stripped) and (is_runtime or is_python or is_backend):
        kinds.append("raw-element-pointer")
    if FIXED_RANK_RE.search(stripped) and (is_python or "backend-hip" in path):
        kinds.append("fixed-rank-metadata")
    if NARROW_METADATA_RE.search(stripped) and (is_python or "backend-hip" in path):
        kinds.append("narrow-metadata")
    if WIDTH_ARITHMETIC_RE.search(stripped) and (
        is_runtime
        or is_python
        or is_backend
        or path in {"crates/chelis-ir/src/analysis.rs", "crates/chelis-ir/src/dag.rs"}
        or path == "crates/chelis-vocab/src/lib.rs"
    ):
        kinds.append("width-arithmetic")
    if path == "crates/chelis-ir/src/dag.rs" and (
        owner.startswith(("normalize_", "flatten_", "push_product", "assemble_"))
        or owner == "normalized_key"
    ) and re.search(r"(?:DimExprKey|concrete|atoms?|gcd|product|quotient)", stripped):
        kinds.append("normalized-key-arithmetic")
    if is_backend and BACKEND_SPELLING_RE.search(stripped):
        kinds.append("backend-element-spelling")
    if is_backend and LOAD_STORE_RE.search(stripped):
        kinds.append("load-store-template")

    if path == "crates/chelis-vocab/src/lib.rs" and owner in {"Repr", "RuntimeDType"}:
        if re.search(r"(?:Self::|Repr::|^[A-Z][A-Za-z0-9_]*(?:\s*=\s*[0-9]+)?\s*,?$)", stripped):
            kinds.append("dtype-contract")
    if path == "crates/chelis-runtime/src/lib.rs" and (
        "TensorElement for" in stripped or "const DTYPE: RuntimeDType" in stripped
    ):
        kinds.append("dtype-contract")

    return tuple(dict.fromkeys(kinds))


def inventory_rows(root: Path) -> tuple[InventoryRow, ...]:
    candidates: list[tuple[str, str, str, str, int]] = []
    for relative in _tracked_source_paths(root):
        path = relative.as_posix()
        owner = "module"
        source = (root / relative).read_text(encoding="utf-8")
        descriptor_owners = descriptor_owner_names(source)
        cfg_test_module = False
        for line in source.splitlines():
            stripped = line.strip()
            if cfg_test_module and re.match(r"(?:pub\s+)?mod\s+\w+\s*\{", stripped):
                # Production source keeps its unit-test module at EOF. Test
                # fixtures are evidence, not raw-path transition debt.
                break
            cfg_test_module = stripped.startswith("#[cfg(test")
            owner = _owner_after_line(line, owner)
            signature = _normalized_signature(line)
            for kind in _line_kinds(path, owner, line, descriptor_owners):
                candidates.append(
                    (kind, path, owner, signature, _deletion_phase(kind, path, owner))
                )

    occurrences: defaultdict[tuple[str, str, str, str], int] = defaultdict(int)
    rows = []
    for kind, path, owner, signature, phase in candidates:
        key = (kind, path, owner, signature)
        occurrences[key] += 1
        rows.append(
            InventoryRow(kind, path, owner, signature, occurrences[key], phase)
        )
    return tuple(sorted(rows, key=lambda row: row.identity))


def _foundation_digest(rows: Sequence[dict[str, object]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def coverage_manifest() -> dict[str, object]:
    return {
        "source_inventory": {
            "artifact": "tracked runtime, ABI, IR-capacity, binding, and backend sources",
            "enumerator": "git ls-files plus closed source classifiers",
            "expected_success": "every hit is exact active debt from the frozen foundation",
            "mutations": [
                "new direct chelis_tensor.data access in a tracked runtime module",
                "new RuntimeDType variant without a complete contract",
            ],
        },
        "release_reproducers": [
            {"name": leg.name, "command": " ".join(leg.argv)} for leg in phase0_legs()
        ],
        "hardware_probes": list(hardware_probe_manifest()),
        "hardware_disposition": (
            "manual-required rows are harness registrations, not executed receipts"
        ),
        "acceptance": "RUNTIME REPRESENTATION PHASE 0: PASS",
    }


def build_foundation_baseline(rows: Sequence[InventoryRow]) -> dict[str, object]:
    foundation = [row.to_baseline_dict() for row in rows]
    return {
        "schema_version": 1,
        "foundation_sha256": _foundation_digest(foundation),
        "foundation_rows": foundation,
        "active_debt": [row.identity for row in rows],
        "coverage_manifest": coverage_manifest(),
    }


def load_baseline() -> dict[str, object]:
    try:
        loaded = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OracleFailure(f"cannot read Phase 0 inventory baseline: {error}") from error
    if not isinstance(loaded, dict):
        raise OracleFailure("Phase 0 inventory baseline must be a JSON object")
    return loaded


def validate_baseline(
    baseline: dict[str, object], rows: Sequence[InventoryRow]
) -> None:
    if baseline.get("schema_version") != 1:
        raise OracleFailure("unsupported Phase 0 inventory schema")
    if baseline.get("coverage_manifest") != coverage_manifest():
        raise OracleFailure("Phase 0 coverage manifest drifted")
    foundation = baseline.get("foundation_rows")
    active = baseline.get("active_debt")
    if not isinstance(foundation, list) or not all(isinstance(row, dict) for row in foundation):
        raise OracleFailure("foundation_rows must be a list of objects")
    if not isinstance(active, list) or not all(isinstance(value, str) for value in active):
        raise OracleFailure("active_debt must be a list of identities")

    computed_digest = _foundation_digest(foundation)
    if baseline.get("foundation_sha256") != computed_digest or computed_digest != FOUNDATION_SHA256:
        raise OracleFailure("Phase 0 foundation digest does not match the reviewed freeze")

    foundation_ids = [row.get("identity") for row in foundation]
    if not all(isinstance(value, str) for value in foundation_ids):
        raise OracleFailure("every foundation row must have an identity")
    if len(foundation_ids) != len(set(foundation_ids)):
        raise OracleFailure("duplicate identity in Phase 0 foundation")
    if len(active) != len(set(active)):
        raise OracleFailure("duplicate identity in active transition debt")

    foundation_set = set(foundation_ids)
    active_set = set(active)
    outside = active_set - foundation_set
    if outside:
        raise OracleFailure(
            "active transition debt contains an identity outside the frozen foundation: "
            + ", ".join(sorted(outside)[:5])
        )

    observed_by_id = {row.identity: row for row in rows}
    if len(observed_by_id) != len(rows):
        raise OracleFailure("derived inventory contains duplicate identities")
    observed_set = set(observed_by_id)
    unclassified = observed_set - active_set
    if unclassified:
        raise OracleFailure(
            "unclassified inventory hit: " + ", ".join(sorted(unclassified)[:5])
        )
    stale = active_set - observed_set
    if stale:
        raise OracleFailure(
            "stale active debt must be deleted from the shrink-only manifest: "
            + ", ".join(sorted(stale)[:5])
        )

    foundation_by_id = {str(row["identity"]): row for row in foundation}
    for identity, observed in observed_by_id.items():
        if foundation_by_id[identity] != observed.to_baseline_dict():
            raise OracleFailure(f"inventory metadata drifted for {identity}")


def validate_phase0_inventory() -> None:
    validate_baseline(load_baseline(), inventory_rows(REPO_ROOT))


def mutate_direct_data_access(source: str) -> str:
    marker = "runtime_representation_phase0_direct_access"
    if marker in source:
        raise OracleFailure("direct-access mutation is already present")
    mutation = f"""

#[allow(dead_code)]
unsafe fn {marker}(tensor: *mut crate::chelis_tensor) -> *mut u8 {{
    unsafe {{ (*tensor).data }}
}}
"""
    test_module = "\n#[cfg(test)]"
    if test_module in source:
        return source.replace(test_module, mutation + test_module, 1)
    return source + mutation


def mutate_incomplete_dtype(source: str) -> str:
    anchor = "    I16 = 8,\n}"
    if source.count(anchor) != 1:
        raise OracleFailure("incomplete-dtype mutation anchor drifted")
    return source.replace(anchor, "    I16 = 8,\n    Phase0Probe = 127,\n}", 1)


@contextmanager
def temporary_mutation(path: Path, mutate: Callable[[str], str]) -> Iterator[None]:
    original = path.read_bytes()
    path.write_text(mutate(original.decode("utf-8")), encoding="utf-8")
    try:
        yield
    finally:
        path.write_bytes(original)
        if path.read_bytes() != original:
            raise OracleFailure(f"failed to restore controlled mutation: {path}")


def _assert_mutation_sources_clean() -> None:
    completed = subprocess.run(
        (
            "git",
            "status",
            "--porcelain",
            "--",
            str(DIRECT_ACCESS_MUTATION_SOURCE),
            str(DTYPE_MUTATION_SOURCE),
        ),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise OracleFailure("could not inspect mutation source status")
    if completed.stdout.strip():
        raise OracleFailure("refusing to mutate a dirty detector source")


def _expect_unclassified_mutation(
    path: Path, mutate: Callable[[str], str], expected_kind: str
) -> None:
    baseline = load_baseline()
    with temporary_mutation(path, mutate):
        try:
            validate_baseline(baseline, inventory_rows(REPO_ROOT))
        except OracleFailure as error:
            message = str(error)
            if "unclassified inventory hit" not in message or f"kind={expected_kind}|" not in message:
                raise OracleFailure(
                    f"{expected_kind} mutation failed for the wrong reason: {message}"
                ) from error
        else:
            raise OracleFailure(f"{expected_kind} mutation was silently accepted")


def run_phase0_mutations() -> None:
    _assert_mutation_sources_clean()
    _expect_unclassified_mutation(
        REPO_ROOT / DIRECT_ACCESS_MUTATION_SOURCE,
        mutate_direct_data_access,
        "direct-data-access",
    )
    _expect_unclassified_mutation(
        REPO_ROOT / DTYPE_MUTATION_SOURCE,
        mutate_incomplete_dtype,
        "dtype-contract",
    )


def phase0_legs() -> tuple[OracleLeg, ...]:
    return (
        OracleLeg(
            "capacity collision release reproducer",
            (
                "cargo",
                "nextest",
                "run",
                "--release",
                "-p",
                "chelis-ir",
                "--test",
                "dim_canon_adversarial",
                "--test",
                "dim_canonicalization",
            ),
        ),
        OracleLeg(
            "count byte zero and foreign metadata release reproducers",
            (
                "cargo",
                "nextest",
                "run",
                "--release",
                "-p",
                "chelis-runtime",
                "--test",
                "exact_tagged_c_abi",
                "--test",
                "op33_empty_tensor_axis_decomposition",
                "--test",
                "op33_tensor_validation",
            ),
        ),
        OracleLeg(
            "landed representation receipts",
            (
                "cargo",
                "nextest",
                "run",
                "--release",
                "-p",
                "chelis-runtime",
                "--test",
                "bool8_storage_type",
                "--test",
                "dim_carrier_int64",
                "--test",
                "runtime_dtype_c_probe",
                "--test",
                "runtime_dtype_generated_header",
                "--test",
                "runtime_dtype_invalid_ffi",
            ),
        ),
    )


def hardware_probe_manifest() -> tuple[dict[str, str], ...]:
    return (
        {
            "lane": "hip",
            "status": "manual-required",
            "command": (
                "scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness "
                "-- --ignored --test-threads=1"
            ),
        },
        {
            "lane": "metal",
            "status": "manual-required",
            "command": (
                "cargo test -p chelis-backend-metal --test gpu_correctness "
                "-- --ignored --test-threads=1"
            ),
        },
    )


def validate_design_sequencing() -> None:
    source = " ".join(DESIGN_PATH.read_text(encoding="utf-8").split())
    required = (
        "Phase 3 creates the shared schema and installs the generated host and public-C artifacts.",
        "Phase 2 requires Phase 3 and consumes that landed schema",
        "Phase 2 does not gate this phase.",
    )
    for text in required:
        if text not in source:
            raise OracleFailure(f"Phase 2/3 descriptor sequencing drifted: missing {text!r}")


def _run_leg(leg: OracleLeg) -> None:
    print(f"+ {leg.name}: {' '.join(leg.argv)}", flush=True)
    completed = subprocess.run(leg.argv, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise OracleFailure(f"{leg.name} failed with exit {completed.returncode}")


def run_phase0(*, run_mutations: bool = True) -> None:
    validate_design_sequencing()
    validate_phase0_inventory()
    if run_mutations:
        run_phase0_mutations()
    for leg in phase0_legs():
        _run_leg(leg)
    print("RUNTIME REPRESENTATION PHASE 0: PASS")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.phase != 0:
        raise OracleFailure("only runtime-representation Phase 0 is implemented")
    run_phase0()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OracleFailure as error:
        raise SystemExit(f"RUNTIME REPRESENTATION PHASE 0: FAIL: {error}") from error
