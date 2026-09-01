#!/usr/bin/env python3
"""Authoritative executable oracle for chelis#893 runtime representation phases.

Phase 0 freezes a source-derived inventory, proves that its transition debt can
only shrink, runs controlled detector mutations, and executes the parser
contract, release reproducers, and landed positive receipts. Later phases
extend this same runner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "spec/design/runtime_representation_phase0_inventory.json"
DESIGN_PATH = REPO_ROOT / "spec/design/runtime_representation.md"
DIRECT_ACCESS_MUTATION_SOURCE = Path("crates/chelis-runtime/src/decimal_parse.rs")
DTYPE_MUTATION_SOURCE = Path("crates/chelis-vocab/src/lib.rs")

# This is the reviewed Phase 0 contract digest. Updating it is a freeze move,
# not a regeneration step: spec/design/runtime_representation.md B1 requires a
# design amendment and a mutation whenever it changes.
FREEZE_SHA256 = "6cd0147af6601599d43fe025cfda7742c4320d20aa7177f54f50fd463cb438f9"
PHASE0_COMMAND = (
    "uv run --managed-python --python 3.11 --no-project python "
    "scripts/runtime_representation_oracle.py --phase 0"
)

SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".cxx",
    ".h",
    ".h++",
    ".hh",
    ".hip",
    ".hpp",
    ".hxx",
    ".m",
    ".metal",
    ".mm",
    ".rs",
}
SOURCE_PREFIXES = (
    "crates/chelis-runtime/src/",
    "crates/chelis-runtime/include/",
    "crates/chelis-vocab/src/",
    "crates/chelis-ir/src/",
    "crates/chelis-python/src/",
)
BACKEND_SOURCE_RE = re.compile(
    r"^crates/chelis-backend-[^/]+/(?:src|runtime|include)/"
)

DECLARATION_PATTERNS = (
    re.compile(r"\b(?:struct|enum|union)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?"
        r"(?:extern\s+\"C\"\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"
    ),
    re.compile(r"^\s*macro_rules!\s+([A-Za-z_][A-Za-z0-9_]*)"),
)

RUST_ELEMENT_TYPE_RE = (
    r"(?:f(?:16|32|64)|i(?:8|16|32|64)|u(?:8|16|32|64)|"
    r"half::(?:f16|bf16)|Bool8)"
)
DIRECT_DATA_RE = re.compile(r"(?:\.data\b|->data\b)")
RAW_POINTER_RE = re.compile(
    rf"(?:as\s+\*(?:mut|const)\s+{RUST_ELEMENT_TYPE_RE}\b"
    rf"|\*(?:mut|const)\s+{RUST_ELEMENT_TYPE_RE}\b)"
)
FIXED_RANK_RE = re.compile(
    r"(?:\b[A-Z][A-Z0-9_]*MAX_DIM\b|\bMAX_DIM\b|\[(?:i32|i64|int|int32_t|int64_t)\s*;\s*(?:[2-9][0-9]*|[A-Z][A-Z0-9_]*)\])"
)
NARROW_METADATA_RE = re.compile(
    r"(?:\b(?:rank|ndim|size|storage_size)\s*:\s*i32\b"
    r"|\bint(?:32_t)?\s+[^;\"']*(?:rank|ndim|size|storage_size)\b"
    r"|(?:rank|ndim|size|storage_size)[^;\"']*\bi32::try_from)"
)
WIDTH_ARITHMETIC_RE = re.compile(
    rf"(?:byte_width|dtype_size|elem(?:ent)?_size|byte_capacity|checked_mul|saturating_mul"
    rf"|size_of\s*::\s*<\s*{RUST_ELEMENT_TYPE_RE}\s*>)"
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

    def __init__(self, message: str, *, code: str = "oracle.failure") -> None:
        super().__init__(message)
        self.code = code


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
            "deletion_phase": self.deletion_phase,
        }


@dataclass(frozen=True)
class OracleLeg:
    name: str
    argv: tuple[str, ...]


@dataclass(frozen=True)
class FailureExpectation:
    code: str
    reason_prefix: str


UNCLASSIFIED_FAILURE = FailureExpectation(
    "inventory.unclassified",
    "unclassified inventory hit",
)
SOURCE_REJECTED_FAILURE = FailureExpectation(
    "source.rejected",
    "fail-closed C-surface parser rejected",
)


@dataclass(frozen=True, init=False)
class MutationProbe:
    witness_id: str
    expected_kind: str
    path: Path
    mutate: Callable[[str], str]
    expected_failure: FailureExpectation = UNCLASSIFIED_FAILURE

    def __init__(
        self,
        expected_kind: str,
        path: Path,
        mutate: Callable[[str], str],
        expected_failure: FailureExpectation | str = UNCLASSIFIED_FAILURE,
        *,
        witness_id: str | None = None,
    ) -> None:
        if isinstance(expected_failure, str):
            expected_failure = FailureExpectation(
                "source.rejected"
                if expected_failure == SOURCE_REJECTED_FAILURE.reason_prefix
                else "oracle.failure",
                expected_failure,
            )
        object.__setattr__(
            self,
            "witness_id",
            witness_id or f"phase0.{mutate.__name__}",
        )
        object.__setattr__(self, "expected_kind", expected_kind)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "mutate", mutate)
        object.__setattr__(self, "expected_failure", expected_failure)

    @property
    def expected_error(self) -> str:
        return self.expected_failure.reason_prefix


def _mutation_probe(
    expected_kind: str,
    path: Path,
    mutate: Callable[[str], str],
    expected_failure: FailureExpectation = UNCLASSIFIED_FAILURE,
) -> MutationProbe:
    return MutationProbe(
        witness_id=f"phase0.{mutate.__name__}",
        expected_kind=expected_kind,
        path=path,
        mutate=mutate,
        expected_failure=expected_failure,
    )


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
        if not relative.startswith(SOURCE_PREFIXES) and not _is_backend_source(relative):
            continue
        if PurePosixPath(relative).suffix not in SOURCE_SUFFIXES:
            continue
        paths.append(Path(relative))
    return tuple(sorted(paths))


def _is_backend_source(path: str) -> bool:
    return BACKEND_SOURCE_RE.match(path) is not None


def _is_c_surface_source(path: str) -> bool:
    return (
        path.startswith("crates/chelis-runtime/")
        or path.startswith("crates/chelis-python/src/")
        or _is_backend_source(path)
    )


@cache
def _c_surface_binary() -> Path:
    configured = os.environ.get("CHELIS_C_SURFACE_BINARY")
    if configured:
        binary = Path(configured)
        if not binary.is_absolute():
            binary = REPO_ROOT / binary
        if not binary.is_file():
            raise OracleFailure(
                f"CHELIS_C_SURFACE_BINARY does not name a file: {binary}"
            )
        return binary

    target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    if not target.is_absolute():
        target = REPO_ROOT / target
    binary = target / "debug" / (
        "chelis-c-surface.exe" if os.name == "nt" else "chelis-c-surface"
    )
    completed = subprocess.run(
        (
            "cargo",
            "build",
            "-p",
            "chelis-c-surface",
            "--bin",
            "chelis-c-surface",
        ),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not binary.is_file():
        raise OracleFailure(
            "could not build the C-surface parser:\n"
            + completed.stdout
            + completed.stderr
        )
    return binary


def c_surface_inventory_rows(
    root: Path, paths: Sequence[Path]
) -> tuple[tuple[str, str, str, str], ...]:
    rows, _ = _c_surface_inventory(root, paths)
    return rows


def _c_surface_inventory(
    root: Path,
    paths: Sequence[Path],
) -> tuple[tuple[tuple[str, str, str, str], ...], dict[str, str]]:
    manifest = [path.as_posix() for path in paths if _is_c_surface_source(path.as_posix())]
    completed = subprocess.run(
        (str(_c_surface_binary()), "--repo", str(root)),
        cwd=root,
        input=json.dumps(manifest),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise OracleFailure(
            "fail-closed C-surface parser rejected the source tree: "
            + completed.stderr.strip(),
            code="source.rejected",
        )
    try:
        decoded = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise OracleFailure(f"C-surface parser emitted invalid JSON: {error}") from error
    if not isinstance(decoded, dict) or set(decoded) != {
        "rows",
        "production_rust_sources",
    }:
        raise OracleFailure("C-surface parser output must be a typed scan object")
    decoded_rows = decoded["rows"]
    production_sources = decoded["production_rust_sources"]
    if not isinstance(decoded_rows, list):
        raise OracleFailure("C-surface parser rows must be a JSON list")
    if not isinstance(production_sources, dict) or not all(
        isinstance(path, str) and isinstance(source, str)
        for path, source in production_sources.items()
    ):
        raise OracleFailure("C-surface production Rust sources must be strings")
    rows = []
    for row in decoded_rows:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "kind",
            "owner",
            "signature",
        }:
            raise OracleFailure("C-surface parser emitted an invalid row")
        values = tuple(row[key] for key in ("kind", "path", "owner", "signature"))
        if not all(isinstance(value, str) for value in values):
            raise OracleFailure("C-surface parser row fields must be strings")
        rows.append(values)
    return tuple(rows), production_sources


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


def descriptor_field_signatures(source: str) -> tuple[tuple[str, str], ...]:
    """Return descriptor field declarations with their structural owner."""

    fields: list[tuple[str, str]] = []
    owners = descriptor_owner_names(source)
    for match in RUST_STRUCT_RE.finditer(source):
        owner = match.group("name")
        if owner not in owners:
            continue
        for line in match.group("body").splitlines():
            if re.match(
                r"^\s*(?:pub(?:\([^)]*\))?\s+)?"
                r"[A-Za-z_][A-Za-z0-9_]*\s*:",
                line,
            ):
                fields.append((owner, _normalized_signature(line)))
    for match in C_STRUCT_RE.finditer(source):
        owner = match.group("name")
        if owner not in owners:
            continue
        for declaration in match.group("body").split(";"):
            signature = _normalized_signature(declaration)
            if signature and re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*(?:\[[^]]*\])?$", signature):
                fields.append((owner, signature + ";"))
    return tuple(fields)


def _deletion_phase(kind: str, path: str, owner: str) -> int:
    if kind in {"dtype-contract", "normalized-key-arithmetic", "width-arithmetic"}:
        return 1
    if kind in {"fixed-rank-metadata", "narrow-metadata"}:
        return 2
    if path.startswith("crates/chelis-python/src/"):
        return 2
    if kind == "descriptor-field" and owner in {"ChelisGpuTensor", "chelis_gpu_tensor"}:
        return 2
    if _is_backend_source(path) or kind in {
        "backend-element-spelling",
        "load-store-template",
    }:
        return 4
    return 3


def _line_kinds(path: str, owner: str, line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if not stripped or stripped.startswith(("//", "///", "//!", "*", "/*")):
        return ()

    kinds: list[str] = []
    is_backend = _is_backend_source(path)
    is_python = path.startswith("crates/chelis-python/src/")
    is_runtime = path.startswith("crates/chelis-runtime/")

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
    tracked_paths = _tracked_source_paths(root)
    c_surface_rows, production_rust_sources = _c_surface_inventory(root, tracked_paths)
    for kind, path, owner, signature in c_surface_rows:
        candidates.append(
            (kind, path, owner, signature, _deletion_phase(kind, path, owner))
        )
    for relative in tracked_paths:
        path = relative.as_posix()
        owner = "module"
        source = production_rust_sources.get(path)
        if source is None:
            source = (root / relative).read_text(encoding="utf-8")
        for descriptor_owner, signature in descriptor_field_signatures(source):
            candidates.append(
                (
                    "descriptor-field",
                    path,
                    descriptor_owner,
                    signature,
                    _deletion_phase("descriptor-field", path, descriptor_owner),
                )
            )
        for line in source.splitlines():
            stripped = line.strip()
            owner = _owner_after_line(line, owner)
            signature = _normalized_signature(line)
            for kind in _line_kinds(path, owner, line):
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


def _freeze_digest(
    rows: Sequence[dict[str, object]], manifest: dict[str, object]
) -> str:
    payload = json.dumps(
        {"coverage_manifest": manifest, "foundation_rows": rows},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mutation_implementation_sha256(mutate: Callable[[str], str]) -> str:
    pending = [mutate]
    sources: dict[str, str] = {}
    while pending:
        current = pending.pop()
        source_path = Path(inspect.getsourcefile(current) or "unknown")
        try:
            stable_path = source_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            stable_path = source_path.name
        identity = f"{stable_path}::{current.__qualname__}"
        if identity in sources:
            continue
        sources[identity] = inspect.getsource(current)
        for name in current.__code__.co_names:
            dependency = current.__globals__.get(name)
            if (
                inspect.isfunction(dependency)
                and dependency.__module__ == current.__module__
            ):
                pending.append(dependency)
    payload = "\n".join(
        f"{identity}\0{sources[identity]}" for identity in sorted(sources)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mutation_manifest(
    probes: Sequence[MutationProbe],
) -> list[dict[str, object]]:
    return [
        {
            "witness_id": probe.witness_id,
            "expected_kind": probe.expected_kind,
            "path": probe.path.as_posix(),
            "implementation_sha256": _mutation_implementation_sha256(probe.mutate),
            "expected_failure": {
                "code": probe.expected_failure.code,
                "reason_prefix": probe.expected_failure.reason_prefix,
            },
            "command": PHASE0_COMMAND,
        }
        for probe in probes
    ]


def coverage_manifest(
    probes: Sequence[MutationProbe] | None = None,
) -> dict[str, object]:
    probes = phase0_mutation_probes() if probes is None else probes
    return {
        "source_inventory": {
            "artifact": "tracked runtime, ABI, IR-capacity, binding, and backend sources",
            "enumerator": "git ls-files plus closed source classifiers",
            "expected_success": "every hit is exact active debt from the frozen foundation",
            "command": PHASE0_COMMAND,
            "mutations": mutation_manifest(probes),
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
    manifest = coverage_manifest()
    return {
        "schema_version": 3,
        "freeze_sha256": _freeze_digest(foundation, manifest),
        "foundation_rows": foundation,
        "active_debt": [row.identity for row in rows],
        "coverage_manifest": manifest,
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
    if baseline.get("schema_version") != 3:
        raise OracleFailure("unsupported Phase 0 inventory schema")
    manifest = baseline.get("coverage_manifest")
    if not isinstance(manifest, dict):
        raise OracleFailure("coverage_manifest must be an object")
    if manifest != coverage_manifest():
        raise OracleFailure("Phase 0 coverage manifest drifted")
    foundation = baseline.get("foundation_rows")
    active = baseline.get("active_debt")
    if not isinstance(foundation, list) or not all(isinstance(row, dict) for row in foundation):
        raise OracleFailure("foundation_rows must be a list of objects")
    if not isinstance(active, list) or not all(isinstance(value, str) for value in active):
        raise OracleFailure("active_debt must be a list of identities")

    computed_digest = _freeze_digest(foundation, manifest)
    if baseline.get("freeze_sha256") != computed_digest or computed_digest != FREEZE_SHA256:
        raise OracleFailure("Phase 0 freeze digest does not match the reviewed contract")

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
            "unclassified inventory hit: " + ", ".join(sorted(unclassified)[:5]),
            code="inventory.unclassified",
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


def _append_probe(source: str, marker: str, snippet: str) -> str:
    if marker in source:
        raise OracleFailure(f"{marker} mutation is already present")
    mutation = f"\n\n{snippet}\n"
    test_module = "\n#[cfg(test)]"
    if test_module in source:
        return source.replace(test_module, mutation + test_module, 1)
    return source + mutation


def mutate_backend_element_spelling(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_backend_spelling",
        'const runtime_representation_phase0_backend_spelling: &str = "float";',
    )


def mutate_descriptor_field(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_descriptor",
        """typedef struct {
    void *data;
    int64_t *shape;
    int64_t *strides;
    int64_t size;
    int32_t rank;
    uint8_t dtype;
} runtime_representation_phase0_descriptor;""",
    )


def mutate_fixed_rank_metadata(source: str) -> str:
    return _append_probe(
        source,
        "RUNTIME_REPRESENTATION_PHASE0_FIXED_RANK",
        "const RUNTIME_REPRESENTATION_PHASE0_FIXED_RANK: [i32; 8] = [0; 8];",
    )


def mutate_load_store_template(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_load_store",
        "fn runtime_representation_phase0_load_store() {\n"
        '    let runtime_representation_phase0_load_store = format!("phase0 pointer");\n'
        "}",
    )


def mutate_narrow_metadata(source: str) -> str:
    return _append_probe(
        source,
        "RuntimeRepresentationPhase0Narrow",
        "struct RuntimeRepresentationPhase0Narrow { ndim: i32 }",
    )


def mutate_normalized_key_arithmetic(source: str) -> str:
    return _append_probe(
        source,
        "normalize_runtime_representation_phase0",
        """fn normalize_runtime_representation_phase0() {
    let atoms: Vec<DimExprKey> = Vec::new();
}""",
    )


def mutate_raw_element_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_raw_pointer",
        "fn runtime_representation_phase0_raw_pointer(_: *mut f32) {}",
    )


def mutate_width_arithmetic(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_byte_width",
        "fn runtime_representation_phase0_byte_width() { let byte_width = 1usize; }",
    )


def mutate_free_standing_c_element_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_c_element_pointer",
        "extern void runtime_representation_phase0_c_element_pointer(float *payload);",
    )


def mutate_c_sizeof_width_authority(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_c_sizeof",
        "static size_t runtime_representation_phase0_c_sizeof(void) { return sizeof(float); }",
    )


def mutate_c_const_after_element_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_c_const_after_pointer",
        "extern void runtime_representation_phase0_c_const_after_pointer("
        "float const *payload);",
    )


def mutate_c_array_parameter(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_c_array_parameter",
        "extern void runtime_representation_phase0_c_array_parameter(float payload[]);",
    )


def mutate_c_const_after_pointer_cast(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_c_const_after_cast",
        "static float runtime_representation_phase0_c_const_after_cast(void *payload) "
        "{ return *((float const *)payload); }",
    )


def mutate_c_qualified_sizeof_width_authority(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_c_qualified_sizeof",
        "static size_t runtime_representation_phase0_c_qualified_sizeof(void) "
        "{ return sizeof(const float); }",
    )


def mutate_c_typedef_alias_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_c_typedef_pointer",
        "typedef float runtime_representation_phase0_element;\n"
        "extern void runtime_representation_phase0_c_typedef_pointer("
        "runtime_representation_phase0_element *payload);",
    )


def mutate_c_macro_alias_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_c_macro_pointer",
        "#define RUNTIME_REPRESENTATION_PHASE0_ELEMENT float\n"
        "extern void runtime_representation_phase0_c_macro_pointer("
        "RUNTIME_REPRESENTATION_PHASE0_ELEMENT *payload);",
    )


def mutate_c_unknown_arithmetic_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_c_unknown_pointer",
        "extern void runtime_representation_phase0_c_unknown_pointer("
        "_Float16 *payload);",
    )


def mutate_c_redefined_alias_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_c_redefined_alias",
        "#define RUNTIME_REPRESENTATION_PHASE0_REDEFINED float\n"
        "extern void runtime_representation_phase0_c_redefined_alias("
        "RUNTIME_REPRESENTATION_PHASE0_REDEFINED *payload);\n"
        "#undef RUNTIME_REPRESENTATION_PHASE0_REDEFINED\n"
        "#define RUNTIME_REPRESENTATION_PHASE0_REDEFINED void",
    )


def mutate_cxx_reference_and_template(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_cxx_reference",
        "extern void runtime_representation_phase0_cxx_reference(float &payload);\n"
        "extern void runtime_representation_phase0_cxx_template("
        "vector<float, 4> *payload);",
    )


def mutate_c_declaration_relocation(source: str) -> str:
    anchor = "static inline float chelis_sum_f32("
    if source.count(anchor) != 1:
        raise OracleFailure("C declaration relocation anchor drifted")
    return source.replace(
        anchor,
        "static inline float runtime_representation_phase0_relocated_sum_f32(",
        1,
    )


def mutate_c_atomic_element_pointer(source: str) -> str:
    anchor = "    float *C,"
    if source.count(anchor) < 1:
        raise OracleFailure("C atomic pointer anchor drifted")
    return source.replace(anchor, "    _Atomic(float) *C,", 1)


def mutate_cxx_rvalue_reference(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_cxx_rvalue",
        "#ifdef __cplusplus\n"
        "extern void runtime_representation_phase0_cxx_rvalue(float &&payload);\n"
        "#endif",
    )


def mutate_c_complete_declarator_shapes(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_complete_declarators",
        "typedef float *runtime_representation_phase0_pointer_alias;\n"
        "extern void runtime_representation_phase0_complete_declarators(\n"
        "    runtime_representation_phase0_pointer_alias alias_payload,\n"
        "    float first[4][8], float (*callback)(int),\n"
        "    float __attribute__((address_space(1))) *addressed);\n"
        "static float *runtime_representation_phase0_first, "
        "*runtime_representation_phase0_second;",
    )


def mutate_c_pointer_return(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_pointer_return",
        "extern float *runtime_representation_phase0_pointer_return(void);",
    )


def mutate_objc_pointer_return(source: str) -> str:
    return _append_probe(
        source,
        "RuntimeRepresentationPhase0Provider",
        "@interface RuntimeRepresentationPhase0Provider\n"
        "- (float *)runtimeRepresentationPhase0Values;\n"
        "@end",
    )


def mutate_rust_dynamic_c_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_rust_dynamic_pointer",
        "fn runtime_representation_phase0_rust_dynamic_pointer(ty: &str) -> String {\n"
        "    format!(r#\"extern void dynamic({ty} const *payload);\"#)\n"
        "}",
    )


def mutate_rust_positional_and_macro_rules_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_rust_positional_pointer",
        "fn runtime_representation_phase0_rust_positional_pointer(\n"
        "    ty: &str, name: &str,\n"
        ") -> String {\n"
        "    format!(\"{} *{}\", ty, name)\n"
        "}\n"
        "macro_rules! runtime_representation_phase0_macro_pointer {\n"
        "    () => { \"double *payload\" };\n"
        "}",
    )


def mutate_rust_split_c_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_rust_split_pointer",
        "fn runtime_representation_phase0_rust_split_pointer(ty: &str) -> String {\n"
        "    format!(\"{ty}\") + \" *payload\"\n"
        "}",
    )


def mutate_rust_unconstrained_c_source(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_unconstrained_source",
        "fn runtime_representation_phase0_unconstrained_source(\n"
        "    declaration: &str,\n"
        ") -> String {\n"
        "    format!(\"{declaration}\")\n"
        "}",
    )


def mutate_rust_stringify_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_stringify_pointer",
        "fn runtime_representation_phase0_stringify_pointer() -> &'static str {\n"
        "    stringify!(float *payload)\n"
        "}",
    )


def mutate_direct_data_access_after_test_module(source: str) -> str:
    marker = "runtime_representation_phase0_after_test_access"
    if marker in source or "#[cfg(test)]" not in source:
        raise OracleFailure("post-test direct-access mutation anchor drifted")
    return source + f"""

#[allow(dead_code)]
unsafe fn {marker}(tensor: *mut crate::chelis_tensor) -> *mut u8 {{
    unsafe {{ (*tensor).data }}
}}
"""


def phase0_mutation_probes() -> tuple[MutationProbe, ...]:
    return (
        MutationProbe(
            "backend-element-spelling",
            Path("crates/chelis-backend-metal/src/emit.rs"),
            mutate_backend_element_spelling,
        ),
        MutationProbe(
            "descriptor-field",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_descriptor_field,
            witness_id="phase0.mutate_descriptor_field.hip",
        ),
        MutationProbe(
            "direct-data-access",
            DIRECT_ACCESS_MUTATION_SOURCE,
            mutate_direct_data_access,
        ),
        MutationProbe("dtype-contract", DTYPE_MUTATION_SOURCE, mutate_incomplete_dtype),
        MutationProbe(
            "fixed-rank-metadata",
            Path("crates/chelis-python/src/lib.rs"),
            mutate_fixed_rank_metadata,
        ),
        MutationProbe(
            "load-store-template",
            Path("crates/chelis-backend-c/src/host_emit.rs"),
            mutate_load_store_template,
        ),
        MutationProbe(
            "narrow-metadata",
            Path("crates/chelis-backend-hip/src/emit.rs"),
            mutate_narrow_metadata,
        ),
        MutationProbe(
            "normalized-key-arithmetic",
            Path("crates/chelis-ir/src/dag.rs"),
            mutate_normalized_key_arithmetic,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-runtime/src/ieee_narrow.rs"),
            mutate_raw_element_pointer,
        ),
        MutationProbe(
            "width-arithmetic",
            Path("crates/chelis-runtime/src/format_shortest.rs"),
            mutate_width_arithmetic,
        ),
        MutationProbe(
            "descriptor-field",
            Path("crates/chelis-backend-metal/runtime/chelis_metal_runtime.h"),
            mutate_descriptor_field,
            witness_id="phase0.mutate_descriptor_field.metal",
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_free_standing_c_element_pointer,
        ),
        MutationProbe(
            "width-arithmetic",
            Path("crates/chelis-backend-metal/runtime/chelis_metal_runtime.h"),
            mutate_c_sizeof_width_authority,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_c_const_after_element_pointer,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_c_array_parameter,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_c_const_after_pointer_cast,
        ),
        MutationProbe(
            "width-arithmetic",
            Path("crates/chelis-backend-metal/runtime/chelis_metal_runtime.h"),
            mutate_c_qualified_sizeof_width_authority,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_c_typedef_alias_pointer,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_c_macro_alias_pointer,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_c_unknown_arithmetic_pointer,
            "fail-closed C-surface parser rejected",
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_c_redefined_alias_pointer,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_cxx_reference_and_template,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/src/kernels.rs"),
            mutate_rust_dynamic_c_pointer,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/src/kernels.rs"),
            mutate_rust_positional_and_macro_rules_pointer,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/src/kernels.rs"),
            mutate_rust_split_c_pointer,
            "fail-closed C-surface parser rejected",
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-runtime/include/chelis_simd.h"),
            mutate_c_declaration_relocation,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_c_atomic_element_pointer,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_cxx_rvalue_reference,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
            mutate_c_complete_declarator_shapes,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-runtime/include/chelis_simd.h"),
            mutate_c_pointer_return,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-metal/runtime/chelis_metal_runtime.h"),
            mutate_objc_pointer_return,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/src/kernels.rs"),
            mutate_rust_unconstrained_c_source,
        ),
        MutationProbe(
            "raw-element-pointer",
            Path("crates/chelis-backend-hip/src/kernels.rs"),
            mutate_rust_stringify_pointer,
        ),
        MutationProbe(
            "direct-data-access",
            DIRECT_ACCESS_MUTATION_SOURCE,
            mutate_direct_data_access_after_test_module,
        ),
    )


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
    paths = tuple(dict.fromkeys(str(probe.path) for probe in phase0_mutation_probes()))
    completed = subprocess.run(
        ("git", "status", "--porcelain", "--", *paths),
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
    path: Path,
    mutate: Callable[[str], str],
    expected_kind: str,
    expected_failure: FailureExpectation,
) -> None:
    baseline = load_baseline()
    with temporary_mutation(path, mutate):
        try:
            validate_baseline(baseline, inventory_rows(REPO_ROOT))
        except OracleFailure as error:
            message = str(error)
            if error.code != expected_failure.code or not message.startswith(
                expected_failure.reason_prefix
            ) or (
                expected_failure == UNCLASSIFIED_FAILURE
                and f"kind={expected_kind}|" not in message
            ):
                raise OracleFailure(
                    f"{expected_kind} mutation failed for the wrong reason: {message}"
                ) from error
        else:
            raise OracleFailure(f"{expected_kind} mutation was silently accepted")


def run_phase0_mutations() -> None:
    _assert_mutation_sources_clean()
    for probe in phase0_mutation_probes():
        _expect_unclassified_mutation(
            REPO_ROOT / probe.path,
            probe.mutate,
            probe.expected_kind,
            probe.expected_failure,
        )


def phase0_legs() -> tuple[OracleLeg, ...]:
    return (
        OracleLeg(
            "fail-closed C-surface parser contract",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-c-surface",
                "--test",
                "c_surface",
            ),
        ),
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
                "issue_888_capacity_collision",
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
    contradictions = (
        r"Phase 2 (?:installs|owns|creates|generates) [^.]{0,100}host descriptor",
        r"Phase 2 [^.]{0,120}delivers C3 completely",
        r"Phase 2 creates the shared schema",
        r"Phase 3 requires Phase 2",
    )
    for pattern in contradictions:
        if re.search(pattern, source, flags=re.IGNORECASE):
            raise OracleFailure(
                f"contradictory Phase 2/3 ownership matched {pattern!r}"
            )


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
