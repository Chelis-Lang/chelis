#!/usr/bin/env python3
"""The chelis#893 runtime-representation Phase 0 acceptance oracle.

Phase 0 changes no representation, allocation, descriptor, field privacy, or
kernel behavior. It proves three things and nothing more:

1. every representation seam named by `spec/design/runtime_representation.md`
   section C6 is enumerated and frozen, so later phases can prove they only
   deleted debt;
2. the live defects those phases repair have executable reproducers that must
   invert rather than be deleted; and
3. a new unlisted seam is detected.

# The universe is a file list, not a language

The inventory's completeness claim is over `INVENTORY_SOURCES`: an explicit,
reviewed list of the repository files that can carry a representation seam.
Fifty-one are Rust and four are plain C headers. A completeness claim stated
over a *language* instead cannot be discharged, because a reviewer can always
name one more construct; stated over a file list it is decidable, and
`_assert_source_list_current` proves the list still equals the tracked contents
of its roots, so a new file fails until someone registers it.

Rust is read with `syn`, a total parser for the language. The four C headers
are read with the numeric capacity census's own token vocabulary
(`tests/support/c_lexical.rs`), which keeps one C classifier in the repository
rather than two.

# Identity is the owning declaration

A row is `kind|path|owner`. Reformatting a literal inside a function does not
move the freeze; adding a function, field, or dtype variant that carries a seam
does. That is the granularity Phases 3 and 4 delete at, and it is what keeps a
755-row emitter file from needing a freeze adjudication on every edit.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "spec/design/runtime_representation_phase0_inventory.json"

# This is the reviewed Phase 0 contract digest. Updating it is a freeze move,
# not a regeneration step: spec/design/runtime_representation.md B1 requires a
# design amendment and a mutation whenever it changes.
FREEZE_SHA256 = "31b574b80a4301d5f6d1f44cb6fb4ecaa231cd697a08e84c94b17b175fd12299"
PHASE0_COMMAND = (
    "uv run --managed-python --python 3.11 --no-project python "
    "scripts/runtime_representation_oracle.py --phase 0"
)

# The roots whose tracked contents the frozen source list must equal. Keeping
# the roots beside the list is what makes the list checkable rather than
# aspirational.
INVENTORY_ROOTS = (
    "crates/chelis-runtime/src/*.rs",
    "crates/chelis-runtime/include/*.h",
    "crates/chelis-vocab/src/*.rs",
    "crates/chelis-ir/src/*.rs",
    "crates/chelis-python/src/*.rs",
    "crates/chelis-backend-*/src/*.rs",
    "crates/chelis-backend-*/runtime/*.h",
)

INVENTORY_SOURCES: tuple[str, ...] = (
    "crates/chelis-backend-c/src/blas.rs",
    "crates/chelis-backend-c/src/emit.rs",
    "crates/chelis-backend-c/src/emitted_expr.rs",
    "crates/chelis-backend-c/src/host_abi.rs",
    "crates/chelis-backend-c/src/host_abi_tests.rs",
    "crates/chelis-backend-c/src/host_emit.rs",
    "crates/chelis-backend-c/src/lib.rs",
    "crates/chelis-backend-c/src/memory.rs",
    "crates/chelis-backend-c/src/toolchain.rs",
    "crates/chelis-backend-hip/runtime/chelis_hip_runtime.h",
    "crates/chelis-backend-hip/src/blas.rs",
    "crates/chelis-backend-hip/src/emit.rs",
    "crates/chelis-backend-hip/src/fusion.rs",
    "crates/chelis-backend-hip/src/kernels.rs",
    "crates/chelis-backend-hip/src/launch.rs",
    "crates/chelis-backend-hip/src/lib.rs",
    "crates/chelis-backend-hip/src/memory.rs",
    "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h",
    "crates/chelis-backend-metal/src/blas.rs",
    "crates/chelis-backend-metal/src/dtype.rs",
    "crates/chelis-backend-metal/src/emit.rs",
    "crates/chelis-backend-metal/src/kernels.rs",
    "crates/chelis-backend-metal/src/lib.rs",
    "crates/chelis-ir/src/analysis.rs",
    "crates/chelis-ir/src/dag.rs",
    "crates/chelis-ir/src/eval.rs",
    "crates/chelis-ir/src/fuse.rs",
    "crates/chelis-ir/src/grad.rs",
    "crates/chelis-ir/src/host.rs",
    "crates/chelis-ir/src/host_type_state.rs",
    "crates/chelis-ir/src/lib.rs",
    "crates/chelis-ir/src/load_store_name.rs",
    "crates/chelis-ir/src/lower.rs",
    "crates/chelis-ir/src/optimize.rs",
    "crates/chelis-ir/src/pipeline.rs",
    "crates/chelis-ir/src/span_merge.rs",
    "crates/chelis-ir/src/span_sanitize.rs",
    "crates/chelis-ir/src/specialize.rs",
    "crates/chelis-ir/src/tier2.rs",
    "crates/chelis-ir/src/verify.rs",
    "crates/chelis-ir/src/vmap.rs",
    "crates/chelis-python/src/lib.rs",
    "crates/chelis-runtime/include/chelis_blas.h",
    "crates/chelis-runtime/include/chelis_math.h",
    "crates/chelis-runtime/include/chelis_runtime.h",
    "crates/chelis-runtime/include/chelis_runtime_dtype.h",
    "crates/chelis-runtime/include/chelis_simd.h",
    "crates/chelis-runtime/src/decimal_parse.rs",
    "crates/chelis-runtime/src/dtype_header.rs",
    "crates/chelis-runtime/src/format_shortest.rs",
    "crates/chelis-runtime/src/ieee_narrow.rs",
    "crates/chelis-runtime/src/lib.rs",
    "crates/chelis-runtime/src/ownership_ledger.rs",
    "crates/chelis-runtime/src/runtime_dtype_contract_tests.rs",
    "crates/chelis-vocab/src/lib.rs",)

# Which phase deletes each seam class, from the design's Part III phase map.
# Phase 1 closes the capacity and dtype-contract vocabulary, Phase 2 the device
# and binding mirrors, Phase 3 the host field seal, Phase 4 the typed lanes.
DELETION_PHASE_BY_KIND: dict[str, int] = {
    "dtype-contract": 1,
    "normalized-key-arithmetic": 1,
    "width-arithmetic": 1,
    "fixed-rank-metadata": 2,
    "narrow-metadata": 2,
    "backend-element-spelling": 4,
    "load-store-template": 4,
}


class OracleFailure(RuntimeError):
    """A failed runtime-representation oracle obligation."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "oracle.failure",
        details: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        #: The exact identities behind the message. A mutation check inspects
        #: these rather than string-matching a truncated message, so a witness
        #: cannot pass by accident when several rows appear at once.
        self.details = tuple(details)


@dataclass(frozen=True, order=True)
class InventoryRow:
    kind: str
    path: str
    owner: str
    deletion_phase: int
    #: A legible excerpt for a reviewer. Deliberately outside `identity` and
    #: outside the frozen digest, so rewording a line cannot move the freeze.
    sample: str = ""

    @property
    def identity(self) -> str:
        return f"kind={self.kind}|path={self.path}|owner={self.owner}"

    def to_baseline_dict(self) -> dict[str, object]:
        return {"identity": self.identity, "deletion_phase": self.deletion_phase}

    def to_active_dict(self) -> dict[str, object]:
        return {"identity": self.identity, "sample": self.sample}


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
    "fail-closed inventory scanner rejected",
)
SOURCE_LIST_FAILURE = FailureExpectation(
    "inventory.unregistered_source",
    "the frozen inventory source list is stale",
)


@dataclass(frozen=True)
class MutationProbe:
    witness_id: str
    expected_kind: str
    path: Path
    mutate: Callable[[str], str]
    expected_failure: FailureExpectation = UNCLASSIFIED_FAILURE


def _probe(
    expected_kind: str,
    path: str,
    mutate: Callable[[str], str],
    expected_failure: FailureExpectation = UNCLASSIFIED_FAILURE,
) -> MutationProbe:
    return MutationProbe(
        witness_id=f"phase0.{mutate.__name__}",
        expected_kind=expected_kind,
        path=Path(path),
        mutate=mutate,
        expected_failure=expected_failure,
    )


def _inventory_candidates(root: Path) -> tuple[str, ...]:
    """Every file on disk under an inventory root, minus ignored ones.

    This reads the filesystem rather than the git index on purpose: cargo
    compiles what is on disk, so an unstaged new file in a crate's `src/` is
    production source and can carry a seam. Enumerating the index instead would
    let a file be invisible to the inventory right up until someone staged it.
    """

    found = {
        path.relative_to(root).as_posix()
        for pattern in INVENTORY_ROOTS
        for path in root.glob(pattern)
        if path.is_file()
    }
    if not found:
        return ()
    completed = subprocess.run(
        ("git", "check-ignore", "--stdin"),
        cwd=root,
        input="\n".join(sorted(found)),
        check=False,
        capture_output=True,
        text=True,
    )
    # Exit 0 means some paths matched, 1 means none did; anything else is a
    # real failure rather than an empty ignore set.
    if completed.returncode not in (0, 1):
        raise OracleFailure("could not resolve ignored inventory candidates")
    ignored = {line for line in completed.stdout.split("\n") if line}
    return tuple(sorted(found - ignored))


def _assert_source_list_current(root: Path) -> None:
    """The frozen list must still equal its roots' tracked contents.

    Without this the list would silently rot: a new file under an inventory
    root would carry seams nobody scans. The failure names the exact sanctioned
    action rather than inviting a workaround.
    """

    tracked = set(_inventory_candidates(root))
    registered = set(INVENTORY_SOURCES)
    unregistered = sorted(tracked - registered)
    departed = sorted(registered - tracked)
    if unregistered:
        raise OracleFailure(
            "the frozen inventory source list is stale; these tracked files under an "
            "inventory root are not registered in INVENTORY_SOURCES: "
            + ", ".join(unregistered[:5]),
            code=SOURCE_LIST_FAILURE.code,
        )
    if departed:
        raise OracleFailure(
            "the frozen inventory source list is stale; these registered files no "
            "longer exist: " + ", ".join(departed[:5]),
            code=SOURCE_LIST_FAILURE.code,
        )


_SCAN_GENERATION = 0
_SCAN_CACHE: dict[int, tuple[InventoryRow, ...]] = {}


def _invalidate_inventory_cache() -> None:
    """Mutations rewrite tracked source, so the cached scan must be dropped."""

    global _SCAN_GENERATION
    _SCAN_GENERATION += 1


def _cargo_target_directory() -> Path:
    completed = subprocess.run(
        ("cargo", "metadata", "--format-version", "1", "--no-deps"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise OracleFailure(f"cargo metadata failed: {completed.stderr.strip()}")
    return Path(json.loads(completed.stdout)["target_directory"])


def _build_scanner() -> Path:
    completed = subprocess.run(
        ("cargo", "build", "--quiet", "-p", "chelis-repr-inventory"),
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise OracleFailure("could not build the structural inventory scanner")
    binary = _cargo_target_directory() / "debug" / "chelis-repr-inventory"
    if not binary.exists():
        raise OracleFailure(f"structural inventory scanner missing at {binary}")
    return binary


def scan_sources(root: Path) -> tuple[dict[str, str], ...]:
    """Run the structural scanner over the frozen source list."""

    binary = _build_scanner()
    completed = subprocess.run(
        (str(binary), "--repo", str(root)),
        input=json.dumps(list(INVENTORY_SOURCES)),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise OracleFailure(
            "fail-closed inventory scanner rejected the source tree: "
            + completed.stderr.strip(),
            code=SOURCE_REJECTED_FAILURE.code,
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise OracleFailure(f"inventory scanner emitted invalid JSON: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"rows"}:
        raise OracleFailure("inventory scanner must emit a typed scan object")
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise OracleFailure("inventory scanner must emit a row list")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "kind", "owner", "sample"}:
            raise OracleFailure(f"inventory scanner emitted an invalid row: {row!r}")
    return tuple(rows)


def inventory_rows(root: Path) -> tuple[InventoryRow, ...]:
    """Derive the deduplicated, owner-keyed seam inventory.

    The scan is cached per generation: a Phase 0 run derives the inventory once
    plus once per mutation, not once per mutation per file.
    """

    cached = _SCAN_CACHE.get(_SCAN_GENERATION)
    if cached is not None:
        return cached
    _assert_source_list_current(root)
    seen: dict[str, InventoryRow] = {}
    for row in scan_sources(root):
        kind = row["kind"]
        derived = InventoryRow(
            kind=kind,
            path=row["path"],
            owner=row["owner"],
            deletion_phase=_deletion_phase(kind, row["path"]),
            sample=row["sample"],
        )
        seen.setdefault(derived.identity, derived)
    rows = tuple(sorted(seen.values()))
    _SCAN_CACHE.clear()
    _SCAN_CACHE[_SCAN_GENERATION] = rows
    return rows


def _is_device_or_binding_mirror(path: str) -> bool:
    """Does this file hold a device or binding descriptor mirror?

    Phase 2 owns the binding and device portion of C4; Phase 3 owns the host
    runtime and public C. The split is by descriptor, not by crate: a backend
    crate's `runtime/` header IS the device descriptor, while its `src/` is
    lane code that Phase 4 retypes.
    """

    return path.startswith("crates/chelis-python/") or (
        path.startswith("crates/chelis-backend-") and "/runtime/" in path
    )


def _deletion_phase(kind: str, path: str) -> int:
    """The phase that deletes this seam, from the design's Part III map."""

    if kind in DELETION_PHASE_BY_KIND:
        return DELETION_PHASE_BY_KIND[kind]
    if _is_device_or_binding_mirror(path):
        return 2
    if path.startswith("crates/chelis-backend-"):
        # Backend `src/` is lane code: element spellings, load and store
        # templates, and the pointers they emit. Phase 4 retypes those.
        return 4
    return 3


def _freeze_digest(rows: Sequence[dict[str, object]], manifest: dict[str, object]) -> str:
    payload = json.dumps(
        {"coverage_manifest": manifest, "foundation_rows": rows},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mutation_implementation_sha256(mutate: Callable[[str], str]) -> str:
    """Hash a mutation's implementation, transitively through its helpers.

    Binding the implementation rather than a prose label is what stops a
    witness being weakened while its manifest entry still claims the old
    semantics.
    """

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
            if inspect.isfunction(dependency) and dependency.__module__ == current.__module__:
                pending.append(dependency)
    payload = "\n".join(f"{identity}\0{sources[identity]}" for identity in sorted(sources))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mutation_manifest(probes: Sequence[MutationProbe]) -> list[dict[str, object]]:
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


def coverage_manifest(probes: Sequence[MutationProbe] | None = None) -> dict[str, object]:
    probes = phase0_mutation_probes() if probes is None else probes
    return {
        "source_inventory": {
            "artifact": "the frozen INVENTORY_SOURCES file list",
            "enumerator": (
                "chelis-repr-inventory: syn for Rust, the capacity census token "
                "vocabulary for plain C headers"
            ),
            "universe": {
                "registered_sources": len(INVENTORY_SOURCES),
                "roots": list(INVENTORY_ROOTS),
                "closure_rule": (
                    "the registered list must equal its roots' tracked contents; a new "
                    "file fails until it is registered"
                ),
            },
            "identity": "kind|path|owner, where owner is the seam's enclosing declaration",
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
        "schema_version": 4,
        "freeze_sha256": _freeze_digest(foundation, manifest),
        "foundation_rows": foundation,
        "active_debt": [row.to_active_dict() for row in rows],
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


def validate_baseline(baseline: dict[str, object], rows: Sequence[InventoryRow]) -> None:
    """Check the derived inventory against the shrink-only frozen ledger.

    The digest binds the immutable foundation and the executable coverage
    manifest together. The active-debt list is deliberately outside it so it can
    shrink; what stops it growing is that every active identity must already be
    in the reviewed foundation.
    """

    if baseline.get("schema_version") != 4:
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
    if not isinstance(active, list) or not all(isinstance(row, dict) for row in active):
        raise OracleFailure("active_debt must be a list of objects")

    computed_digest = _freeze_digest(foundation, manifest)
    if baseline.get("freeze_sha256") != computed_digest or computed_digest != FREEZE_SHA256:
        raise OracleFailure("Phase 0 freeze digest does not match the reviewed contract")

    foundation_ids = [row.get("identity") for row in foundation]
    active_ids = [row.get("identity") for row in active]
    if not all(isinstance(value, str) for value in foundation_ids):
        raise OracleFailure("every foundation row must have an identity")
    if not all(isinstance(value, str) for value in active_ids):
        raise OracleFailure("every active-debt row must have an identity")
    if len(foundation_ids) != len(set(foundation_ids)):
        raise OracleFailure("duplicate identity in Phase 0 foundation")
    if len(active_ids) != len(set(active_ids)):
        raise OracleFailure("duplicate identity in active transition debt")

    foundation_set = set(foundation_ids)
    active_set = set(active_ids)
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
            code=UNCLASSIFIED_FAILURE.code,
            details=sorted(unclassified),
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


# ---------------------------------------------------------------------------
# Controlled mutations
#
# Each plants a REAL seam of its class, not a token pattern that happens to
# match a regular expression. A witness that plants something the structural
# scanner would have to be broken to miss is the only kind that proves the
# classifier works.
# ---------------------------------------------------------------------------


def _append_probe(source: str, marker: str, snippet: str) -> str:
    if marker in source:
        raise OracleFailure(f"{marker} mutation is already present")
    mutation = f"\n\n{snippet}\n"
    test_module = "\n#[cfg(test)]"
    if test_module in source:
        return source.replace(test_module, mutation + test_module, 1)
    return source + mutation


_PROBE_DESCRIPTOR = """#[repr(C)]
#[allow(dead_code)]
struct RuntimeRepresentationPhase0Descriptor {{
    data: *mut f32,
    dtype: i32,
    shape: {shape},
    strides: *const i64,
    ndim: {ndim},
    size: i64,
}}"""


def mutate_direct_data_access(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_direct_access",
        """#[allow(dead_code)]
unsafe fn runtime_representation_phase0_direct_access(
    tensor: *mut crate::chelis_tensor,
) -> *mut u8 {
    unsafe { (*tensor).data }
}""",
    )


def mutate_direct_data_access_after_test_module(source: str) -> str:
    """A production item AFTER a test module is still production source."""

    marker = "runtime_representation_phase0_after_test_access"
    if marker in source or "#[cfg(test)]" not in source:
        raise OracleFailure("post-test direct-access mutation anchor drifted")
    return (
        source
        + f"""

#[allow(dead_code)]
unsafe fn {marker}(tensor: *mut crate::chelis_tensor) -> *mut u8 {{
    unsafe {{ (*tensor).data }}
}}
"""
    )


def mutate_incomplete_dtype(source: str) -> str:
    anchor = "    I16 = 8,\n}"
    if source.count(anchor) != 1:
        raise OracleFailure("incomplete-dtype mutation anchor drifted")
    return source.replace(anchor, "    I16 = 8,\n    Phase0Probe = 127,\n}", 1)


def mutate_raw_element_pointer(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_raw_pointer",
        """#[allow(dead_code)]
fn runtime_representation_phase0_raw_pointer(bytes: *mut u8) -> *mut f32 {
    bytes as *mut f32
}""",
    )


def mutate_width_arithmetic(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_byte_width",
        """#[allow(dead_code)]
fn runtime_representation_phase0_byte_width() -> usize {
    size_of::<f32>()
}""",
    )


def mutate_normalized_key_arithmetic(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_saturating_fold",
        """#[allow(dead_code)]
fn runtime_representation_phase0_saturating_fold(concrete: usize, value: usize) -> usize {
    concrete.saturating_mul(value)
}""",
    )


def mutate_backend_element_spelling(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_backend_spelling",
        """#[allow(dead_code)]
fn runtime_representation_phase0_backend_spelling() -> &'static str {
    "float"
}""",
    )


def mutate_load_store_template(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_load_store",
        """#[allow(dead_code)]
fn runtime_representation_phase0_load_store(index: usize) -> String {
    format!("((float *)tensor->data)[{index}]")
}""",
    )


def mutate_narrow_metadata(source: str) -> str:
    return _append_probe(
        source,
        "RuntimeRepresentationPhase0Descriptor",
        _PROBE_DESCRIPTOR.format(shape="*const i64", ndim="i32"),
    )


def mutate_fixed_rank_metadata(source: str) -> str:
    return _append_probe(
        source,
        "RuntimeRepresentationPhase0Descriptor",
        _PROBE_DESCRIPTOR.format(shape="[i64; CHELIS_MAX_DIM]", ndim="i64"),
    )


def mutate_descriptor_field(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_descriptor",
        """typedef struct {
    float *data;
    int dtype;
    int shape[8];
    int strides[8];
    int ndim;
    int size;
} runtime_representation_phase0_descriptor;""",
    )


def mutate_unknown_c_arithmetic_spelling(source: str) -> str:
    """An arithmetic spelling nobody classified must stop the build.

    The rule is inverted for the reason `spec/design/dtype_semantics.md`
    section C6 gives: an allowlist of arithmetic spellings can never be
    complete, so a 16-bit float that no list mentions has to fail loudly rather
    than enter as an unflagged row.
    """

    return _append_probe(
        source,
        "runtime_representation_phase0_unknown_arithmetic",
        "extern _Float16 *runtime_representation_phase0_unknown_arithmetic(void);",
    )


def mutate_unregistered_inventory_source(_source: str) -> str:
    """A new file under an inventory root must fail until it is registered."""

    return """//! Temporary Phase 0 detector probe.

#[allow(dead_code)]
pub unsafe fn runtime_representation_phase0_unregistered(
    tensor: *mut u8,
) -> *mut f32 {
    tensor as *mut f32
}
"""


def phase0_mutation_probes() -> tuple[MutationProbe, ...]:
    runtime_probe = "crates/chelis-runtime/src/decimal_parse.rs"
    return (
        _probe("backend-element-spelling", "crates/chelis-backend-metal/src/emit.rs", mutate_backend_element_spelling),
        _probe("descriptor-field", "crates/chelis-backend-hip/runtime/chelis_hip_runtime.h", mutate_descriptor_field),
        _probe("direct-data-access", runtime_probe, mutate_direct_data_access),
        _probe("direct-data-access", runtime_probe, mutate_direct_data_access_after_test_module),
        _probe("dtype-contract", "crates/chelis-vocab/src/lib.rs", mutate_incomplete_dtype),
        _probe("fixed-rank-metadata", "crates/chelis-python/src/lib.rs", mutate_fixed_rank_metadata),
        _probe("load-store-template", "crates/chelis-backend-c/src/host_emit.rs", mutate_load_store_template),
        _probe("narrow-metadata", "crates/chelis-python/src/lib.rs", mutate_narrow_metadata),
        _probe("normalized-key-arithmetic", "crates/chelis-ir/src/dag.rs", mutate_normalized_key_arithmetic),
        _probe("raw-element-pointer", "crates/chelis-runtime/src/ieee_narrow.rs", mutate_raw_element_pointer),
        _probe("width-arithmetic", "crates/chelis-runtime/src/format_shortest.rs", mutate_width_arithmetic),
        _probe(
            "unknown-arithmetic-spelling",
            "crates/chelis-runtime/include/chelis_simd.h",
            mutate_unknown_c_arithmetic_spelling,
            SOURCE_REJECTED_FAILURE,
        ),
        _probe(
            "unregistered-inventory-source",
            "crates/chelis-runtime/src/runtime_representation_phase0_probe.rs",
            mutate_unregistered_inventory_source,
            SOURCE_LIST_FAILURE,
        ),
    )


@contextmanager
def temporary_mutation(path: Path, mutate: Callable[[str], str]) -> Iterator[None]:
    """Apply a mutation to tracked source and restore it byte for byte.

    The dirty-source check lives HERE rather than in the runner so no caller,
    unit tests included, can plant a mutation over uncommitted work.
    """

    _assert_source_clean(path)
    existed = path.exists()
    original = path.read_bytes() if existed else None
    _invalidate_inventory_cache()
    path.write_text(mutate(original.decode("utf-8") if original else ""), encoding="utf-8")
    try:
        yield
    finally:
        _invalidate_inventory_cache()
        if original is None:
            path.unlink(missing_ok=True)
            if path.exists():
                raise OracleFailure(f"failed to remove controlled mutation: {path}")
        else:
            path.write_bytes(original)
            if path.read_bytes() != original:
                raise OracleFailure(f"failed to restore controlled mutation: {path}")


def _assert_source_clean(path: Path) -> None:
    relative = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    completed = subprocess.run(
        ("git", "status", "--porcelain", "--", relative),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise OracleFailure("could not inspect mutation source status")
    if completed.stdout.strip():
        raise OracleFailure(f"refusing to mutate a dirty detector source: {relative}")


def _expect_mutation_rejected(probe: MutationProbe) -> None:
    baseline = load_baseline()
    with temporary_mutation(REPO_ROOT / probe.path, probe.mutate):
        try:
            validate_baseline(baseline, inventory_rows(REPO_ROOT))
        except OracleFailure as error:
            if error.code != probe.expected_failure.code or not str(error).startswith(
                probe.expected_failure.reason_prefix
            ):
                raise OracleFailure(
                    f"{probe.witness_id} failed for the wrong reason: {error}"
                ) from error
            if probe.expected_failure is UNCLASSIFIED_FAILURE and not any(
                identity.startswith(f"kind={probe.expected_kind}|")
                for identity in error.details
            ):
                raise OracleFailure(
                    f"{probe.witness_id} failed without naming a {probe.expected_kind} "
                    f"row: {sorted(error.details)[:5]}"
                ) from error
        else:
            raise OracleFailure(f"{probe.witness_id} mutation was silently accepted")


def run_phase0_mutations() -> None:
    for probe in phase0_mutation_probes():
        _expect_mutation_rejected(probe)


def phase0_legs() -> tuple[OracleLeg, ...]:
    return (
        OracleLeg(
            "structural inventory scanner contract",
            ("cargo", "nextest", "run", "-p", "chelis-repr-inventory", "--test", "inventory"),
        ),
        # [#888] has three witnesses at two levels: the key-level collision in
        # the IR, and the planner-level consequence in each backend that
        # consumes the key. The two planners carry the same defect in verbatim
        # copies, so one witness would understate the class. All three must
        # INVERT when Phase 1's exact CapacityKey lands, never be deleted.
        OracleLeg(
            "capacity collision key-level release reproducer",
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
            "capacity collision planner-level release reproducers",
            (
                "cargo",
                "nextest",
                "run",
                "--release",
                "-p",
                "chelis-backend-c",
                "-p",
                "chelis-backend-hip",
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
        OracleLeg(
            "shared capacity census classifier",
            (
                "cargo",
                "nextest",
                "run",
                "-p",
                "chelis-cli",
                "--test",
                "capacity_census_tripwire",
            ),
        ),
    )


def hardware_probe_manifest() -> tuple[dict[str, str], ...]:
    """Hardware lanes are registered with their exact command, never counted.

    A lane the default suite skipped has not run. Recording the command here is
    a registration, not a receipt.
    """

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


def _run_leg(leg: OracleLeg) -> None:
    print(f"+ {leg.name}: {' '.join(leg.argv)}", flush=True)
    completed = subprocess.run(leg.argv, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise OracleFailure(f"{leg.name} failed with exit {completed.returncode}")


def run_phase0(*, run_mutations: bool = True) -> None:
    validate_phase0_inventory()
    if run_mutations:
        run_phase0_mutations()
    for leg in phase0_legs():
        _run_leg(leg)
    print("RUNTIME REPRESENTATION PHASE 0: PASS")


def regenerate() -> None:
    """Rewrite the baseline from the current tree.

    Regeneration cannot bless growth: it rewrites the artifact, and the reviewed
    `FREEZE_SHA256` in this file still has to be moved by hand, which is the
    design's B1 freeze move rather than a regeneration step.
    """

    baseline = build_foundation_baseline(inventory_rows(REPO_ROOT))
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(baseline['foundation_rows'])} rows to {BASELINE_PATH}")
    print(f"freeze_sha256 = {baseline['freeze_sha256']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rewrite the frozen baseline from the current tree",
    )
    parser.add_argument(
        "--skip-mutations",
        action="store_true",
        help="skip the controlled mutations (development loop only)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.phase != 0:
        raise OracleFailure("only runtime-representation Phase 0 is implemented")
    if args.regenerate:
        regenerate()
        return 0
    run_phase0(run_mutations=not args.skip_mutations)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OracleFailure as error:
        raise SystemExit(f"RUNTIME REPRESENTATION PHASE 0: FAIL: {error}") from error
