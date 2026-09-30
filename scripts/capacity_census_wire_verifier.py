"""Execute the C6 wire proof and issue its sole final-authority witness.

The persisted baseline is a comparison target. No caller-supplied rustdoc,
descriptor, classification or prior receipt enters the verification factory.
Native cache/AST codecs retain their separate inventory; cache evidence here
binds the changed shared value codecs and actual context reconstruction.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from capacity_census_graph import GraphError
from capacity_census_typed import NUMERIC_PRIMITIVES
from capacity_census_wire_adapters import _target_lease, canonical, source_identity
from capacity_census_wire_runner import _unique_fields
from ci_timing import span

# The canonical codec graph projects the two verified half-crate adapters to
# their Chelis storage dtypes. They are not new Rust primitive spellings.
_FINAL_FLOATS = {"f16", "bf16", "f32", "f64"}
_FINAL_NUMERIC = NUMERIC_PRIMITIVES | _FINAL_FLOATS


def final_rows(classifications):
    """Format an already checked plan; this function cannot issue authority."""
    rows = []
    identities = set()
    for classification in classifications:
        leaf = classification.leaf
        identity = f"{leaf.path}: {leaf.primitive}"
        if (
            identity in identities
            or not leaf.path
            or leaf.primitive not in _FINAL_NUMERIC
            or classification.authority not in {"TaggedTransport", "NumericOperation"}
            or not classification.contract
        ):
            raise GraphError(
                f"invalid or duplicated final wire classification: {identity}"
            )
        if classification.authority == "NumericOperation" and not re.fullmatch(
            r"\[05-OP-[1-9][0-9]*\]", classification.contract
        ):
            raise GraphError("numeric wire operation lacks exact governing atom")
        identities.add(identity)
        rows.append(
            {
                "kind": "wire-schema-numeric-field",
                "id": identity,
                "flags": [
                    "float-carrier"
                    if leaf.primitive in _FINAL_FLOATS
                    else "numeric-field"
                ],
                "authority": classification.authority,
                "contract": classification.contract,
            }
        )
    if not rows:
        raise GraphError("wire verifier selected zero numeric leaves")
    return sorted(rows, key=lambda row: row["id"])


def compare_baseline(rows, text):
    try:
        baseline = json.loads(text, object_pairs_hook=_unique_fields)
    except (ValueError, TypeError) as error:
        raise GraphError("invalid final wire baseline") from error
    if (
        not isinstance(baseline, dict)
        or type(baseline.get("version")) is not int
        or baseline != {"version": 2, "rows": rows}
    ):
        raise GraphError(
            "wire baseline differs from executed final authority; a baseline edit supplies no admission"
        )


@dataclass(frozen=True, init=False, slots=True)
class VerifiedWireCensus:
    root: str
    source_sha256: str
    graph_identity: str
    schema: object
    caches: object
    publication: object
    report: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("final authority requires actual wire verification")

    def validate(self):
        if source_identity(Path(self.root)) != self.source_sha256:
            raise GraphError("wire authority is stale against current source bytes")
        documents = [
            json.loads(self.schema.canonical.document),
            json.loads(self.schema.document),
            *(json.loads(d) for d in self.schema.imported_documents),
        ]
        self.schema.validate(documents)
        self.caches.validate()
        self.publication.validate()

    def execution_report(self):
        self.validate()
        return json.loads(self.report)


def verify_wire_census(root: Path, target: Path) -> VerifiedWireCensus:
    from capacity_census_cache_publication import verify_cache_publication
    from capacity_census_wire_acceptance import execute_acceptance_controls
    from capacity_census_wire_invocation_owners import verify_invocation_ownership
    from capacity_census_wire_schema import verify_schema_codecs

    root, target = root.resolve(), target.resolve()
    if not target.is_relative_to(root / "target"):
        raise GraphError("wire verifier target must belong to this worktree")
    before = source_identity(root)
    # Keep the sequence lock outside Cargo's target. The codec builder must
    # distinguish a fresh target from an existing Cargo cache before cleaning.
    with _target_lease(root / "target/.wire-census-sequence"):
        with span("wire.schema", "census-stage"):
            schema = verify_schema_codecs(root, target)
        rows = final_rows(schema.classifications)
        with _target_lease(target):
            # Consumer builds can replace shared dependency artifacts. Finish
            # those builds before binding the cache proof's exact artifacts;
            # invocation collection has its own retained Cargo namespace.
            with span("wire.acceptance", "census-stage"):
                executions = execute_acceptance_controls(root, target, schema)
            with span("wire.cache", "census-stage"):
                caches = verify_cache_publication(root, target)
            with span("wire.publication", "census-stage"):
                publication = verify_invocation_ownership(root, target, schema, caches)
        if source_identity(root) != before or schema.canonical.source_sha256 != before:
            raise GraphError("wire proof source changed during execution")
        report = {
            "version": 2,
            "rows": rows,
            "source_sha256": before,
            "graph_identity": schema.graph_identity,
            "evidence": {
                "schema": schema.execution_report(),
                "cache": caches.execution_report(),
                "publication": publication.execution_report(),
                "acceptance": [asdict(e) for e in executions],
            },
        }
        witness = object.__new__(VerifiedWireCensus)
        for name, value in {
            "root": str(root),
            "source_sha256": before,
            "graph_identity": schema.graph_identity,
            "schema": schema,
            "caches": caches,
            "publication": publication,
            "report": canonical(report),
        }.items():
            object.__setattr__(witness, name, value)
        witness.validate()
        return witness
