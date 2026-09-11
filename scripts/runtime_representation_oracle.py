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
Seventy are Rust and seven are C or Objective-C headers. A completeness
claim stated over a *language* instead cannot be discharged, because a reviewer
can always name one more construct; stated over a file list it is decidable,
and `_assert_source_list_current` proves the list still equals the tracked
contents of its roots, so a new file fails until someone registers it.

Every source is read by a real parser for its language. Rust is read with
`syn`. The headers are read through clang's front end (`-fsyntax-only -Xclang
-ast-dump=json`) under a fixed target triple, `-ffreestanding -nostdlibinc`, a
committed stub SDK, and a scrubbed environment, so the parse is the same on
every host and every SDK state. Each header's lane names the closed set of
configurations it is parsed under and the row set is their union; an arm with
code (a declaration, a quoted include, or a define with a body) that no
configuration parses, an include resolving canonically outside the universe,
and a type word no vocabulary classifies (in a declaration, a block parameter,
or an expression) all fail the scan. The reader classifies the compiler's type spellings with
the capacity census's closed word lists (`tests/support/c_lexical.rs`), which
keeps one type-word authority in the repository rather than two. The owner of
a seam is the declaration the compiler says encloses it, so declaration FORM
is never this script's problem.

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
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "spec/design/runtime_representation_phase0_inventory.json"

# This is the reviewed Phase 0 contract digest. Updating it is a freeze move,
# not a regeneration step: spec/design/runtime_representation.md B1 requires a
# design amendment and a mutation whenever it changes.
FREEZE_SHA256 = "6dc5eb76161a9ae4f8f9061f1e7919bc30180e318c03b514f9a5a78deda94e4f"
PHASE0_COMMAND = (
    "uv run --managed-python --python 3.11 --no-project python "
    "scripts/runtime_representation_oracle.py --phase 0"
)

# The roots whose tracked contents the frozen source list must equal. Keeping
# the roots beside the list is what makes the list checkable rather than
# aspirational.
INVENTORY_ROOTS = (
    "crates/chelis-runtime/src/**/*.rs",
    "crates/chelis-runtime/include/**/*.h",
    "crates/chelis-runtime/build.rs",
    "crates/chelis-vocab/src/**/*.rs",
    "crates/chelis-vocab/build.rs",
    "crates/chelis-ir/src/**/*.rs",
    "crates/chelis-ir/build.rs",
    "crates/chelis-python/src/**/*.rs",
    "crates/chelis-python/build.rs",
    "crates/chelis-backend-*/src/**/*.rs",
    "crates/chelis-backend-*/runtime/**/*.h",
    # A build script is compiled by cargo like any other source and can carry
    # a seam; a root that cannot see it is a closure hole.
    "crates/chelis-backend-*/build.rs",
)

INVENTORY_SOURCES: tuple[str, ...] = (
    "crates/chelis-backend-c/build.rs",
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
    "crates/chelis-ir/src/axis_sources.rs",
    "crates/chelis-ir/src/capacity_key.rs",
    "crates/chelis-ir/src/dag.rs",
    "crates/chelis-ir/src/eval.rs",
    "crates/chelis-ir/src/evaluation.rs",
    "crates/chelis-ir/src/execution_spine.rs",
    "crates/chelis-ir/src/fuse.rs",
    "crates/chelis-ir/src/grad.rs",
    "crates/chelis-ir/src/host.rs",
    "crates/chelis-ir/src/host/staged.rs",
    "crates/chelis-ir/src/host_type_state.rs",
    "crates/chelis-ir/src/lib.rs",
    "crates/chelis-ir/src/load_store_name.rs",
    "crates/chelis-ir/src/lower.rs",
    "crates/chelis-ir/src/lower/static_controls.rs",
    "crates/chelis-ir/src/lowering_trace.rs",
    "crates/chelis-ir/src/optimize.rs",
    "crates/chelis-ir/src/ownership/classify.rs",
    "crates/chelis-ir/src/ownership/error.rs",
    "crates/chelis-ir/src/ownership/ir.rs",
    "crates/chelis-ir/src/ownership/last_use.rs",
    "crates/chelis-ir/src/ownership/lower.rs",
    "crates/chelis-ir/src/ownership/mod.rs",
    "crates/chelis-ir/src/ownership/render.rs",
    "crates/chelis-ir/src/ownership/storage.rs",
    "crates/chelis-ir/src/ownership/tests.rs",
    "crates/chelis-ir/src/ownership/verify.rs",
    "crates/chelis-ir/src/pipeline.rs",
    "crates/chelis-ir/src/span_merge.rs",
    "crates/chelis-ir/src/span_sanitize.rs",
    "crates/chelis-ir/src/specialize.rs",
    "crates/chelis-ir/src/tier2.rs",
    "crates/chelis-ir/src/verify.rs",
    "crates/chelis-ir/src/vmap.rs",
    "crates/chelis-python/src/lib.rs",
    "crates/chelis-python/src/compiler_json.rs",
    "crates/chelis-python/src/source_json.rs",
    "crates/chelis-runtime/include/chelis_blas.h",
    "crates/chelis-runtime/include/chelis_math.h",
    "crates/chelis-runtime/include/chelis_runtime.h",
    "crates/chelis-runtime/include/chelis_runtime_dtype.h",
    "crates/chelis-runtime/include/chelis_simd.h",
    "crates/chelis-runtime/src/decimal_parse.rs",
    "crates/chelis-runtime/src/dtype_header.rs",
    "crates/chelis-runtime/src/element.rs",
    "crates/chelis-runtime/src/format_shortest.rs",
    "crates/chelis-runtime/src/ieee_narrow.rs",
    "crates/chelis-runtime/src/lib.rs",
    "crates/chelis-runtime/src/metadata.rs",
    "crates/chelis-runtime/src/ownership_ledger.rs",
    "crates/chelis-runtime/src/runtime_dtype_contract_tests.rs",
    "crates/chelis-vocab/src/lib.rs",)

# Which phase deletes each seam class, from the design's Part III phase map.
# Phase 1 closes the capacity and dtype-contract vocabulary, Phase 2 the device
# and binding mirrors, Phase 3 the host field seal, Phase 4 the typed lanes.
DELETION_PHASE_BY_KIND: dict[str, int] = {
    "dtype-contract": 1,
    "legacy-capacity-key-use": 1,
    "normalized-key-arithmetic": 1,
    "saturating-capacity-fold": 1,
    "wrapping-capacity-fold": 1,
    "width-arithmetic": 1,
    "fixed-rank-metadata": 2,
    "narrow-metadata": 2,
    "backend-element-spelling": 4,
    "load-store-template": 4,
}

CAPACITY_KEY_OWNER = "crates/chelis-ir/src/capacity_key.rs"
CAPACITY_KEY_EXACT_PRODUCT_OWNER = "ExactLiteralProduct::include"
VOCAB_OWNER = "crates/chelis-vocab/src/lib.rs"
ELEMENT_OWNER = "crates/chelis-runtime/src/element.rs"
METADATA_OWNER = "crates/chelis-runtime/src/metadata.rs"
METADATA_FINAL_WIDTH_OWNERS = ("ElementCount::bytes", "ElementCount::scratch_len")
C_INDEX_PROJECTION_OWNERS = (
    ("crates/chelis-backend-c/src/emit.rs", "CEmitter::emit_elementwise_index_steps"),
    ("crates/chelis-backend-c/src/host_emit.rs", "HostEmitter < 'a >::emit_elementwise_index_step"),
)
ELEMENT_FINAL_CONTRACT_OWNERS = (
    "ElementStorage for f64", "ElementStorage for f32",
    "ElementStorage for F16Bits", "ElementStorage for Bf16Bits",
    "ElementStorage for i64", "ElementStorage for i32",
    "ElementStorage for i16", "ElementStorage for i8",
    "ElementStorage for Bool8", "TensorElement for T",
    "private :: Sealed for f64", "private :: Sealed for f32",
    "private :: Sealed for F16Bits", "private :: Sealed for Bf16Bits",
    "private :: Sealed for i64", "private :: Sealed for i32",
    "private :: Sealed for i16", "private :: Sealed for i8",
    "private :: Sealed for Bool8",
    "ArithmeticIdentity for f64", "ArithmeticIdentity for f32",
    "ArithmeticIdentity for i64", "ArithmeticIdentity for i32",
    "ArithmeticIdentity for i16", "ArithmeticIdentity for i8",
    "ArithmeticIdentity for ()",
)
VOCAB_FINAL_CONTRACT_OWNERS = (
    "ArithmeticRepr::Ieee754Binary32",
    "ArithmeticRepr::Ieee754Binary64",
    "ArithmeticRepr::ExactTwosComplement8",
    "ArithmeticRepr::ExactTwosComplement16",
    "ArithmeticRepr::ExactTwosComplement32",
    "ArithmeticRepr::ExactTwosComplement64",
)


def owner_module_final_form(kind: str, path: str, owner: str) -> bool:
    """Exact owners backed by the capacity, vocabulary, element, and metadata contracts."""

    return (
        kind == "exact-capacity-arithmetic"
        and path == CAPACITY_KEY_OWNER
        and owner == CAPACITY_KEY_EXACT_PRODUCT_OWNER
    ) or (
        path == VOCAB_OWNER
        and (
            (kind == "dtype-contract" and owner in VOCAB_FINAL_CONTRACT_OWNERS)
            or (kind == "width-arithmetic" and owner == "DTypeContract::byte_width")
        )
    ) or (
        path == ELEMENT_OWNER
        and (
            (kind == "dtype-contract" and owner in ELEMENT_FINAL_CONTRACT_OWNERS)
            or (kind == "width-arithmetic" and owner == "assert_registration")
        )
    ) or (
        path == METADATA_OWNER
        and kind == "width-arithmetic"
        and owner in METADATA_FINAL_WIDTH_OWNERS
    ) or (
        kind == "backend-element-spelling"
        and (path, owner) in C_INDEX_PROJECTION_OWNERS
    )


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
    # Owners the rejection must name, for a witness whose point is that one
    # declaration yields several identities.
    expected_owners: tuple[str, ...] = ()


def _probe(
    expected_kind: str,
    path: str,
    mutate: Callable[[str], str],
    expected_failure: FailureExpectation = UNCLASSIFIED_FAILURE,
    expected_owners: tuple[str, ...] = (),
) -> MutationProbe:
    return MutationProbe(
        witness_id=f"phase0.{mutate.__name__}",
        expected_kind=expected_kind,
        path=Path(path),
        mutate=mutate,
        expected_failure=expected_failure,
        expected_owners=expected_owners,
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
        if owner_module_final_form(kind, row["path"], row["owner"]):
            continue
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
            elif isinstance(dependency, (str, bytes, tuple, frozenset)):
                # A witness body can live in a shared module constant.
                # `_PROBE_DESCRIPTOR` is the body of two mutations, and without
                # this the digest did not move when it was edited.
                sources[f"const::{name}"] = repr(dependency)
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
            "expected_owners": list(probe.expected_owners),
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
                "chelis-repr-inventory: syn for Rust; clang's front end under fixed "
                "target lanes, a committed stub SDK, and a scrubbed environment for C "
                "and Objective-C headers, classified with the capacity census's closed "
                "type-word lists"
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
            "owner_module_final_forms": {
                **{
                    path: [{"kind": "backend-element-spelling", "owner": owner}]
                    for path, owner in C_INDEX_PROJECTION_OWNERS
                },
                METADATA_OWNER: [
                    {"kind": "width-arithmetic", "owner": owner}
                    for owner in METADATA_FINAL_WIDTH_OWNERS
                ],
                ELEMENT_OWNER: [
                    {"kind": "dtype-contract", "owner": owner}
                    for owner in ELEMENT_FINAL_CONTRACT_OWNERS
                ] + [{"kind": "width-arithmetic", "owner": "assert_registration"}],
                VOCAB_OWNER: [
                    {"kind": "dtype-contract", "owner": owner}
                    for owner in VOCAB_FINAL_CONTRACT_OWNERS
                ]
                + [{"kind": "width-arithmetic", "owner": "DTypeContract::byte_width"}],
                CAPACITY_KEY_OWNER: [
                    {
                        "kind": "exact-capacity-arithmetic",
                        "owner": CAPACITY_KEY_EXACT_PRODUCT_OWNER,
                    }
                ],
            },
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

    # The sample is evidence for a reviewer, not identity, so it stays outside
    # the digest and may be rewritten freely by a regeneration. It must still
    # be TRUE: an artifact whose evidence column has no currency check invites
    # a reader to trust a line that no longer exists.
    stored_samples = {str(row["identity"]): row.get("sample", "") for row in active}
    for identity, observed in observed_by_id.items():
        if stored_samples[identity] != observed.sample:
            raise OracleFailure(
                f"active-debt sample is stale for {identity}; regenerate the baseline"
            )


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


def _insert_inside_include_guard(source: str, marker: str, snippet: str) -> str:
    """Plant a C declaration before the include guard's closing `#endif`.

    A declaration appended after the guard is repeated every time the header
    is included, which for a header two others include is a redefinition
    error rather than the seam under test.
    """

    if marker in source:
        raise OracleFailure(f"{marker} mutation is already present")
    guard_end = source.rstrip().rfind("#endif")
    if guard_end < 0:
        raise OracleFailure("include-guard mutation anchor drifted")
    return source[:guard_end] + f"{snippet}\n\n" + source[guard_end:]


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


def mutate_incomplete_arithmetic_repr(source: str) -> str:
    anchor = "pub enum ArithmeticRepr {\n"
    if source.count(anchor) != 1:
        raise OracleFailure("incomplete-arithmetic-repr mutation anchor drifted")
    return source.replace(anchor, anchor + "    Phase0Probe,\n", 1)


def mutate_element_binding(source: str) -> str:
    """An aliased extra binding and its private seal need exact registration."""

    return _append_probe(source, "UnregisteredElement", """
#[derive(Clone, Copy)]
struct UnregisteredElement(f32);
use self::ElementStorage as Storage;
impl private::Sealed for UnregisteredElement {}
impl Storage for UnregisteredElement {
    const STORAGE_DTYPE: RuntimeDType = RuntimeDType::F32;
    const STORED_REPR: Repr = Repr::Ieee754Binary32;
    type ArithmeticStorage = f32;
}
""")


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


def mutate_metadata_owner_width(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_unchecked_metadata_width",
        """#[allow(dead_code)]
fn runtime_representation_phase0_unchecked_metadata_width() -> usize {
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


def mutate_retired_capacity_normalizer(source: str) -> str:
    """An old normalizer identity cannot re-enter the shrink-only debt set."""

    return _append_probe(
        source,
        "runtime_representation_retired_normalizer",
        """// runtime_representation_retired_normalizer
impl DimExpr {
    pub fn normalized_key(&self) -> usize {
        match self {
            Self::Mul(lhs, rhs) => lhs.as_concrete().unwrap_or(usize::MAX)
                .saturating_mul(rhs.as_concrete().unwrap_or(usize::MAX)),
            _ => self.as_concrete().unwrap_or(usize::MAX),
        }
    }
}""",
    )


def mutate_capacity_owner_saturation(source: str) -> str:
    """Saturation never becomes final inside the exact-capacity owner."""

    return _append_probe(
        source,
        "runtime_representation_capacity_owner_saturation",
        """#[allow(dead_code)]
fn runtime_representation_capacity_owner_saturation(value: usize) -> usize {
    value.saturating_mul(2)
}""",
    )


def mutate_capacity_owner_wrapping_product(source: str) -> str:
    """Primitive multiplication never becomes exact capacity arithmetic."""

    return _append_probe(
        source,
        "runtime_representation_capacity_owner_wrapping_product",
        """#[allow(dead_code)]
fn runtime_representation_capacity_owner_wrapping_product(a: u64, b: u64) -> u64 {
    a * b
}""",
    )


def mutate_capacity_owner_legacy_key(source: str) -> str:
    """The exact owner may not recover capacity through the legacy carrier."""

    return _append_probe(
        source,
        "runtime_representation_capacity_owner_legacy_key",
        """#[allow(dead_code)]
fn runtime_representation_capacity_owner_legacy_key(value: &crate::dag::DimExpr) {
    let _ = value.normalized_key();
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


def mutate_c_public_element_pointer_export(source: str) -> str:
    """A new public `float *` export inside the header's linkage block.

    A prototype at file scope, so owner attribution rather than descriptor
    scanning has to catch it.
    """

    anchor = "chelis_tensor *chelis_alloc("
    if source.count(anchor) != 1:
        raise OracleFailure("public element-pointer mutation anchor drifted")
    return source.replace(
        anchor,
        "float *runtime_representation_phase0_probe_row(chelis_tensor *t);\n" + anchor,
        1,
    )


def mutate_c_body_direct_data_access(source: str) -> str:
    """A new `->data` access inside a function body in a tracked header."""

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_touch",
        """static inline void runtime_representation_phase0_probe_touch(chelis_gpu_tensor *t) {
    t->data = 0;
}""",
    )


def mutate_c_extern_element_data(source: str) -> str:
    """An exported C DATA declaration carrying an element pointer.

    `AGENTS.md` names an exported C data declaration as numeric surface, and no
    witness covered it: the only C prototype witness plants the one declarator
    form that the paren-keyed naming rule could already see.
    """

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_table",
        "extern double *runtime_representation_phase0_probe_table;",
    )


def mutate_c_non_descriptor_struct_field(source: str) -> str:
    """An element pointer in a struct that is not a tensor descriptor.

    A helper aggregate's field is a carrier even though the descriptor rules
    do not apply to it.
    """

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_pair",
        """typedef struct {
    int64_t key;
    float *weights;
} runtime_representation_phase0_probe_pair;""",
    )


def mutate_c_tagged_struct_field(source: str) -> str:
    """A tagged `struct` with a separate typedef, the idiomatic public form.

    The round-3 finding: a reader that only recognised `typedef struct { ... }`
    let this form carry a `float *` past it without a row or an error.
    """

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_pool",
        """struct runtime_representation_phase0_probe_pool {
    float *slab;
    int64_t n;
};
typedef struct runtime_representation_phase0_probe_pool runtime_representation_phase0_probe_pool;""",
    )


def mutate_c_union_field(source: str) -> str:
    return _append_probe(
        source,
        "runtime_representation_phase0_probe_slot",
        """union runtime_representation_phase0_probe_slot {
    double *d;
    int64_t i;
};""",
    )


def mutate_c_macro_typed_carrier(source: str) -> str:
    """A carrier whose element type only a preprocessor can see."""

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_slab",
        """#define RUNTIME_REPRESENTATION_PHASE0_PROBE_ELEM float
RUNTIME_REPRESENTATION_PHASE0_PROBE_ELEM *runtime_representation_phase0_probe_slab(void);""",
    )


def mutate_c_multi_declarator_data(source: str) -> str:
    """Two declarators in one declaration are two owners, not one."""

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_pair_a",
        "extern float *runtime_representation_phase0_probe_pair_a, "
        "*runtime_representation_phase0_probe_pair_b;",
    )


def mutate_c_enum_width(source: str) -> str:
    """A file-scope enum constant that computes a width owns that seam."""

    return _insert_inside_include_guard(
        source,
        "RUNTIME_REPRESENTATION_PHASE0_PROBE_WIDTH",
        "enum { RUNTIME_REPRESENTATION_PHASE0_PROBE_WIDTH = sizeof(float) };",
    )


def mutate_objc_element_pointer_parameter(source: str) -> str:
    """An Objective-C header's C function carrying an element pointer."""

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_fill",
        """static inline void runtime_representation_phase0_probe_fill(
    id<MTLBuffer> buffer, const float *values
) {
    (void)buffer;
    (void)values;
}""",
    )


def mutate_c_carrier_in_simd_arm(source: str) -> str:
    """A carrier inside the AVX2 arm of the SIMD header.

    The scalar configuration never reads that arm; the lane's `avx2`
    configuration must, or a helper that only ships on x86 servers is invisible.
    """

    return _insert_inside_include_guard(
        source,
        "runtime_representation_phase0_probe_avx",
        """#ifdef __AVX2__
static inline void runtime_representation_phase0_probe_avx(float *p) {
    (void)p;
}
#endif""",
    )


def mutate_c_undeclared_conditional(source: str) -> str:
    """A carrier behind a conditional no lane configuration selects.

    The arm never parses, so the seam is invisible to every configuration; the
    scan has to fail naming the directive rather than pass without the row.
    """

    return _insert_inside_include_guard(
        source,
        "RUNTIME_REPRESENTATION_PHASE0_PROBE_CFG",
        """#ifdef RUNTIME_REPRESENTATION_PHASE0_PROBE_CFG
extern float *runtime_representation_phase0_probe_cfg;
#endif""",
    )


def mutate_objc_method_carrier(source: str) -> str:
    """An Objective-C method whose result is an element pointer."""

    return _append_probe(
        source,
        "RuntimeRepresentationPhase0Probe",
        """@interface RuntimeRepresentationPhase0Probe : NSObject
- (float *)elements;
@end""",
    )


def mutate_c_unclassified_cast_spelling(source: str) -> str:
    """An arithmetic spelling no vocabulary lists, in a cast rather than a
    declaration. The inverted type-word rule holds in expression position."""

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_cast",
        """static inline void runtime_representation_phase0_probe_cast(void *p) {
    ((_Float16 *)p)[0] = 0;
}""",
    )


def mutate_c_pointer_to_element_array(source: str) -> str:
    """A pointer to an array of elements, through a declarator group."""

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_rows",
        "void runtime_representation_phase0_probe_rows(float (*rows)[4]);",
    )


def mutate_c_include_outside_universe(source: str) -> str:
    """An include that resolves, through `..`, to a tracked C header no
    inventory root reaches. Round 5 showed a prefix test on the spelt path
    let this through; the universe is a canonical-path test now."""

    return _insert_inside_include_guard(
        source,
        "tree_sitter/parser.h",
        '#include "../../../grammars/tree-sitter-chelis-surf/src/tree_sitter/parser.h"',
    )


def mutate_c_include_in_dead_arm(source: str) -> str:
    """An include behind a conditional no configuration selects.

    An arm that only includes carries whatever it includes, so it needs a
    configuration exactly as an arm with a declaration does.
    """

    return _insert_inside_include_guard(
        source,
        "RUNTIME_REPRESENTATION_PHASE0_PROBE_DEAD",
        """#ifdef RUNTIME_REPRESENTATION_PHASE0_PROBE_DEAD
#include "chelis_runtime_dtype.h"
#endif""",
    )


def mutate_objc_block_parameter(source: str) -> str:
    """A block literal whose parameter is an element pointer, handed to an
    `id`, so no enclosing declaration's type reveals it."""

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_block",
        """static inline id runtime_representation_phase0_probe_block(void) {
    return ^(float *q) { q[0] = 0.0f; };
}""",
    )


def mutate_c_sizeof_in_array_bound(source: str) -> str:
    """A width computed in a declared type rather than in a statement."""

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_scratch",
        "static unsigned char runtime_representation_phase0_probe_scratch[sizeof(double) * 4];",
    )


def mutate_rust_cast_turbofish(source: str) -> str:
    """`p.cast::<f32>()` names the element type only in the turbofish."""

    return _append_probe(
        source,
        "runtime_representation_phase0_turbofish",
        """#[allow(dead_code)]
fn runtime_representation_phase0_turbofish(bytes: *mut u8) -> usize {
    bytes.cast::<f32>() as usize
}""",
    )


def mutate_c_int8_element_pointer(source: str) -> str:
    """An 8-bit element pointer, whose typedef resolves to a `char` spelling
    that names no element; the written spelling has to be read first."""

    return _append_probe(
        source,
        "runtime_representation_phase0_probe_i8",
        "extern int8_t *runtime_representation_phase0_probe_i8;",
    )


def mutate_c_elifdef_arm(source: str) -> str:
    """A carrier behind a C23 `#elifdef` no configuration selects."""

    return _insert_inside_include_guard(
        source,
        "RUNTIME_REPRESENTATION_PHASE0_PROBE_ELIF",
        """#ifdef __GNUC__
static const int runtime_representation_phase0_probe_live = 1;
#elifdef RUNTIME_REPRESENTATION_PHASE0_PROBE_ELIF
extern float *runtime_representation_phase0_probe_elifdef;
#endif""",
    )


def mutate_rust_path_module(source: str) -> str:
    """`#[path]` compiles a file the inventory roots do not reach."""

    return _append_probe(
        source,
        "runtime_representation_phase0_path_probe",
        '#[path = "../runtime_representation_phase0_path_probe.rs"]\n'
        "mod runtime_representation_phase0_path_probe;",
    )


def mutate_unregistered_subdirectory_source(_source: str) -> str:
    """A seam in a SUBDIRECTORY of an inventory root.

    Cargo compiles it, so it is production source. A single-level glob did not
    see it, which made the closure check evadable.
    """

    return """//! Temporary Phase 0 detector probe.

#[allow(dead_code)]
pub fn runtime_representation_phase0_subdirectory(concrete: usize, value: usize) -> usize {
    concrete.saturating_mul(value)
}
"""


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
        _probe("dtype-contract", VOCAB_OWNER, mutate_incomplete_arithmetic_repr),
        _probe("dtype-contract", ELEMENT_OWNER, mutate_element_binding,
               expected_owners=("Storage for UnregisteredElement",
                                "private :: Sealed for UnregisteredElement")),
        _probe("fixed-rank-metadata", "crates/chelis-python/src/lib.rs", mutate_fixed_rank_metadata),
        _probe("load-store-template", "crates/chelis-backend-c/src/host_emit.rs", mutate_load_store_template),
        _probe("narrow-metadata", "crates/chelis-python/src/lib.rs", mutate_narrow_metadata),
        _probe("normalized-key-arithmetic", "crates/chelis-ir/src/dag.rs", mutate_normalized_key_arithmetic),
        _probe("normalized-key-arithmetic", "crates/chelis-ir/src/dag.rs", mutate_retired_capacity_normalizer,
               expected_owners=("DimExpr::normalized_key",)),
        _probe(
            "saturating-capacity-fold",
            CAPACITY_KEY_OWNER,
            mutate_capacity_owner_saturation,
        ),
        _probe(
            "wrapping-capacity-fold",
            CAPACITY_KEY_OWNER,
            mutate_capacity_owner_wrapping_product,
        ),
        _probe(
            "legacy-capacity-key-use",
            CAPACITY_KEY_OWNER,
            mutate_capacity_owner_legacy_key,
        ),
        _probe("raw-element-pointer", "crates/chelis-runtime/src/ieee_narrow.rs", mutate_raw_element_pointer),
        _probe("width-arithmetic", "crates/chelis-runtime/src/format_shortest.rs", mutate_width_arithmetic),
        _probe("width-arithmetic", METADATA_OWNER, mutate_metadata_owner_width),
        _probe(
            "unknown-arithmetic-spelling",
            "crates/chelis-runtime/include/chelis_simd.h",
            mutate_unknown_c_arithmetic_spelling,
            SOURCE_REJECTED_FAILURE,
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_public_element_pointer_export,
        ),
        _probe(
            "direct-data-access",
            "crates/chelis-backend-hip/runtime/chelis_hip_runtime.h",
            mutate_c_body_direct_data_access,
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_extern_element_data,
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_non_descriptor_struct_field,
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_tagged_struct_field,
            expected_owners=("runtime_representation_phase0_probe_pool::slab",),
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_union_field,
            expected_owners=("runtime_representation_phase0_probe_slot::d",),
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-backend-hip/runtime/chelis_hip_runtime.h",
            mutate_c_macro_typed_carrier,
            expected_owners=("runtime_representation_phase0_probe_slab",),
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_multi_declarator_data,
            expected_owners=(
                "runtime_representation_phase0_probe_pair_a",
                "runtime_representation_phase0_probe_pair_b",
            ),
        ),
        _probe(
            "width-arithmetic",
            "crates/chelis-runtime/include/chelis_runtime_dtype.h",
            mutate_c_enum_width,
            expected_owners=("RUNTIME_REPRESENTATION_PHASE0_PROBE_WIDTH",),
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h",
            mutate_objc_element_pointer_parameter,
            expected_owners=("runtime_representation_phase0_probe_fill",),
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-runtime/include/chelis_simd.h",
            mutate_c_carrier_in_simd_arm,
            expected_owners=("runtime_representation_phase0_probe_avx",),
        ),
        _probe(
            "undeclared-conditional",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_undeclared_conditional,
            SOURCE_REJECTED_FAILURE,
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h",
            mutate_objc_method_carrier,
            expected_owners=("RuntimeRepresentationPhase0Probe::elements",),
        ),
        _probe(
            "unknown-arithmetic-spelling",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_unclassified_cast_spelling,
            SOURCE_REJECTED_FAILURE,
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_pointer_to_element_array,
            expected_owners=("runtime_representation_phase0_probe_rows",),
        ),
        _probe(
            "include-outside-universe",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_include_outside_universe,
            SOURCE_REJECTED_FAILURE,
        ),
        _probe(
            "undeclared-conditional",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_include_in_dead_arm,
            SOURCE_REJECTED_FAILURE,
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h",
            mutate_objc_block_parameter,
            expected_owners=("runtime_representation_phase0_probe_block",),
        ),
        _probe(
            "width-arithmetic",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_sizeof_in_array_bound,
            expected_owners=("runtime_representation_phase0_probe_scratch",),
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-runtime/src/format_shortest.rs",
            mutate_rust_cast_turbofish,
            expected_owners=("runtime_representation_phase0_turbofish",),
        ),
        _probe(
            "raw-element-pointer",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_int8_element_pointer,
            expected_owners=("runtime_representation_phase0_probe_i8",),
        ),
        _probe(
            "undeclared-conditional",
            "crates/chelis-runtime/include/chelis_runtime.h",
            mutate_c_elifdef_arm,
            SOURCE_REJECTED_FAILURE,
        ),
        _probe(
            "rust-path-module",
            "crates/chelis-ir/src/dag.rs",
            mutate_rust_path_module,
            SOURCE_REJECTED_FAILURE,
        ),
        _probe(
            "unregistered-inventory-source",
            "crates/chelis-runtime/src/runtime_representation_phase0_probe.rs",
            mutate_unregistered_inventory_source,
            SOURCE_LIST_FAILURE,
        ),
        _probe(
            "unregistered-inventory-source",
            "crates/chelis-ir/src/repr_probe/mod.rs",
            mutate_unregistered_subdirectory_source,
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
    created_directories = []
    if not existed:
        for parent in reversed(path.parents):
            if not parent.exists():
                created_directories.append(parent)
        path.parent.mkdir(parents=True, exist_ok=True)
    _invalidate_inventory_cache()
    path.write_text(mutate(original.decode("utf-8") if original else ""), encoding="utf-8")
    try:
        yield
    finally:
        _invalidate_inventory_cache()
        if original is None:
            path.unlink(missing_ok=True)
            for directory in reversed(created_directories):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
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
            expected = {
                f"kind={probe.expected_kind}|path={probe.path.as_posix()}|owner={owner}"
                for owner in probe.expected_owners
            }
            missing = expected - set(error.details)
            if missing:
                raise OracleFailure(
                    f"{probe.witness_id} failed without naming every expected owner: "
                    f"{sorted(missing)}"
                ) from error
        else:
            raise OracleFailure(f"{probe.witness_id} mutation was silently accepted")


def run_phase0_mutations() -> None:
    probes = phase0_mutation_probes()
    print(f"+ controlled mutations: {len(probes)}", flush=True)
    for probe in probes:
        _expect_mutation_rejected(probe)
        print(
            f"  rejected [{probe.expected_failure.code}] {probe.witness_id}",
            flush=True,
        )


def phase0_legs() -> tuple[OracleLeg, ...]:
    return (
        *(OracleLeg(
            f"checked host metadata {profile} contract",
            (
                "cargo", "nextest", "run", *flags, "-p", "chelis-runtime",
                "--features", "ownership-ledger",
                "--test", "checked_metadata", "--test", "metadata_compile",
                "--test", "checked_metadata_padding", "--test", "checked_c_metadata",
                "--test", "checked_c_indexing", "--test", "checked_c_movement", "--test", "checked_c_movement_plans", "--test", "checked_c_affine",
                "--test", "checked_c_reduction", "--test", "checked_c_sparse", "--test", "checked_c_matmul", "--test", "checked_c_window", "--test", "checked_c_literal", "--test", "checked_c_alloc_like",
                "--test", "exact_tagged_c_abi",
                "--test", "op33_empty_tensor_axis_decomposition",
                "--test", "op33_tensor_validation",
                "--test", "op33_legal_domain_matrix", "--test", "dim_carrier_int64",
                "--test", "tensor_repurpose", "--test", "tensor_write_guard",
            ),
        ) for profile, flags in (("debug", ()), ("release", ("--release",)))),
        OracleLeg(
            "checked C snapshot delegation and restoration mutations",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "checked_c_metadata", "--test", "checked_c_snapshot_metadata"),
        ),
        OracleLeg(
            "checked shape observation before storage submission execution",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "exec_compile", "-E", "test(checked_snapshot_)"),
        ),
        OracleLeg(
            "checked vmap shape observation on its shifted axis",
            ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "runtime_extent_slice_a", "-E", "test(vmap_shape_bound_with_concrete_batch_emits_c_without_to_end_ice)"),
        ),
        OracleLeg(
            "checked C reduction delegation and bypass mutations",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "checked_c_reduction"),
        ),
        OracleLeg(
            "checked C reductions and Count optimized sanitizer execution",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "exec_compile",
             "--test", "fused_compile", "-E", "binary(fused_compile) | test(checked_c_reduction_) | test(exec_count_) | test(exec_reduce_sum_) | test(ws_a1_exec_f64_reduce_sum) | test(ws_a1_exec_i32_reduce_sum) | test(exec_i8_reduce_sum) | test(exec_i16_reduce_sum) | test(direct_fused_sum_runtime_shape) | test(direct_fused_max_reduce_runtime_shape)"),
        ),
        OracleLeg(
            "checked C reduction example parity",
            ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "parity",
             "--test", "issue_1294_standard_lowerings", "-E",
             "test(parity_count_bool_axes) | test(canonical_sum_) | test(scalar_and_fused_sums_) | test(hosted_matmul_preserves_the_canonical_reduction_tree) | test(hosted_matmul_empty_reductions_)"),
        ),
        OracleLeg(
            "checked C JSON ordering scratch delegation and lifetime controls",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "checked_c_json_scratch"),
        ),
        OracleLeg(
            "checked C JSON ordering scratch recursive and empty execution",
            ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "issue_1314_json_bigint",
             "-E", "test(=json_object_serialization_is_recursive_canonical_unicode_order_in_eval_and_c)"),
        ),
        OracleLeg(
            "checked C JSON scratch ownership ledger and skipped-cleanup mutations",
            ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "issue_1314_json_bigint",
             "-E", "test(=json_scratch_execution_detects_skipped_cleanup)"),
        ),
        OracleLeg(
            "checked C literal ingress delegation controls",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "checked_c_host_metadata"),
        ),
        OracleLeg(
            "checked C literal storage optimized sanitizer execution",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "exec_compile",
             "-E", "test(checked_literals_)"),
        ),
        OracleLeg(
            "checked C window delegation and bypass mutations",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "checked_c_window"),
        ),
        OracleLeg(
            "checked C window geometry and gradient optimized sanitizer execution",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "exec_compile",
             "--test", "issue_254_reduce_window_emit", "-E", "test(checked_windows_) | test(exec_reduce_window_) | binary(issue_254_reduce_window_emit)"),
        ),
        OracleLeg(
            "checked C window executable example parity",
            ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "parity",
             "-E", "test(parity_checked_window_geometry) | test(parity_corpus_is_complete)"),
        ),
        OracleLeg(
            "checked C BLAS submission delegation and bypass mutations",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "checked_c_blas"),
        ),
        OracleLeg(
            "checked C BLAS optimized submission and vendor prototype execution",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "exec_compile",
             "-E", "test(checked_blas_) | test(blas_vendor_dimension_contract)"),
        ),
        OracleLeg(
            "checked C sparse delegation and bypass mutations",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "checked_c_sparse"),
        ),
        OracleLeg(
            "checked C sparse optimized sanitizer execution",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "exec_compile",
             "-E", "test(checked_c_sparse_)"),
        ),
        OracleLeg(
            "checked C sparse host summary execution and rejection",
            ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "cross_library_sparse_summaries",
             "--test", "parity", "-E", "binary(cross_library_sparse_summaries) | test(parity_checked_sparse_axes)"),
        ),
        OracleLeg(
            "checked C shared indexing cohort and restoration mutations",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "checked_c_indexing"),
        ),
        OracleLeg(
            "checked C shared indexing optimized sanitizer execution",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "exec_compile",
             "-E", "test(checked_c_indexing_)"),
        ),
        OracleLeg(
            "checked C movement delegation and restoration mutations",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "checked_c_movement", "--test", "checked_c_movement_plans"),
        ),
        OracleLeg(
            "movement local extent CLI parity and primitive diagnostics",
            ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "issue_616_runtime_movement_c_parity"),
        ),
        OracleLeg(
            "verified expansion primitive identity and local guards",
            ("cargo", "nextest", "run", "-p", "chelis-ir", "--test", "movement_expansion_kind"),
        ),
        OracleLeg(
            "verified expansion primitive identity and local guards in release",
            ("cargo", "nextest", "run", "--release", "-p", "chelis-ir", "--test", "movement_expansion_kind"),
        ),
        OracleLeg(
            "checked C movement optimized sanitizer execution",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "exec_compile",
             "-E", "test(checked_c_movement_) | test(a_local_class_guards) | test(a_literal_claim_on_a_symbolic_input) | test(numeric_local_extent_claims)"),
        ),
        OracleLeg(
            "checked C shared indexing dtype dispatch and storage reuse",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--lib",
             "--test", "host_emit_dtype_dispatch", "--test", "dtype_matrix_bf16_f16",
             "--test", "fused_in_place_exec", "--test", "fused_in_place_forall_alias"),
        ),
        OracleLeg(
            "checked C shared indexing cast behavior and first failure",
            ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "issue_759_checked_cast_default",
             "--test", "cast_trunc", "--test", "cbackend_cast_memcpy",
             "--test", "cbackend_cast_arithmetic_composition"),
        ),
        OracleLeg(
            "checked C reshape and shared indexing example parity",
            ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "parity",
             "-E", "test(parity_checked_reshape)"),
        ),
        OracleLeg(
            "checked C DAG reshape optimized UBSan execution",
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--lib",
             "-E", "test(checked_c_metadata_dag_reshape_executes_under_ubsan)"),
        ),
        OracleLeg(
            "checked C host reshape optimized UBSan execution",
            ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "cbackend_reshape_memcpy"),
        ),
        OracleLeg(
            "sealed runtime element contract",
            ("cargo", "nextest", "run", "--release", "-p", "chelis-runtime", "--test", "element_contract"),
        ),
        OracleLeg(
            "closed representation vocabulary contract",
            (
                "cargo", "nextest", "run", "-p", "chelis-vocab",
                "--test", "dtype_contract", "--test", "dtype_contract_compile",
            ),
        ),
        OracleLeg(
            "structural inventory scanner contract",
            ("cargo", "nextest", "run", "-p", "chelis-repr-inventory", "--test", "inventory"),
        ),
        OracleLeg(
            "exact capacity key contract",
            ("cargo", "test", "--release", "-p", "chelis-ir", "capacity_key::tests::"),
        ),
        OracleLeg(
            "opaque capacity key public surface",
            ("cargo", "test", "--release", "-p", "chelis-ir", "--doc", "capacity_key"),
        ),
        # The private CapacityKey leg above inverts #888's original key-level
        # witness. These shared-plan and backend-adapter legs preserve its
        # allocation consequences without retaining a lossy production API.
        # Finite DimExpr projections must reject overflow in release too.
        OracleLeg(
            "exact shared capacity and finite projection release controls",
            (
                "cargo",
                "nextest",
                "run",
                "--release",
                "-p",
                "chelis-ir",
                "--test",
                "issue_888_capacity_collision",
                "--test",
                "dim_expr_evaluation",
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
    if args.phase == 1:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.runtime_representation_phase1 import run
        try:
            run()
        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
            raise OracleFailure(str(error)) from error
        return 0
    if args.phase != 0:
        raise OracleFailure("only runtime-representation Phases 0 and 1 are implemented")
    if args.regenerate:
        regenerate()
        return 0
    run_phase0(run_mutations=not args.skip_mutations)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OracleFailure as error:
        phase = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else "?"
        raise SystemExit(f"RUNTIME REPRESENTATION PHASE {phase}: FAIL: {error}") from error
