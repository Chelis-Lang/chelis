"""Execution-backed closure of the shared durable cache publication owners.

This receipt proves the closed publication types and their unchanged key-format
selectors. It does not authorize any numeric descendant of a cache payload.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from capacity_census_graph import GraphError, RustdocGraph
from capacity_census_wire_adapters import canonical, source_identity
from capacity_census_wire_runner import TestExecution, run_libtest


class CachePublicationError(GraphError):
    pass


def require(condition, message):
    if not condition:
        raise CachePublicationError(message)


def seal_compiled_artifacts(
    directory: Path, artifacts: dict[str, str]
) -> dict[str, str]:
    """Copy one build's exact artifacts out of Cargo's mutable cache."""

    sources = {name: Path(filename) for name, filename in artifacts.items()}
    filenames = [source.name for source in sources.values()]
    require(
        len(filenames) == len(set(filenames)),
        "duplicate compiled artifact filename",
    )
    sealed_directory = directory / "sealed-dependencies"
    sealed_directory.mkdir(parents=True, exist_ok=True)
    sealed = {}
    for name, source in sorted(sources.items()):
        require(
            source.is_file() and not source.is_symlink(),
            f"missing or symlinked compiled cache publication artifact: {source}",
        )
        destination = sealed_directory / source.name
        require(
            not destination.is_symlink(),
            f"sealed cache publication artifact is a symlink: {destination}",
        )
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        # A Cargo output that Kache restored without reflinks is a hardlink
        # to a read-only store blob, and copy2 carries that mode to the copy.
        # Replace an earlier seal's copy instead of writing through it.
        destination.unlink(missing_ok=True)
        shutil.copy2(source, destination)
        require(
            hashlib.sha256(destination.read_bytes()).hexdigest() == before,
            f"compiled cache publication artifact changed while sealing: {source}",
        )
        sealed[name] = str(destination)
    return sealed


@dataclass(frozen=True)
class CompileCase:
    name: str
    success: bool
    error: str | None = None

    @property
    def diagnostic(self):
        return "E0603" if self.name == "unseal" else "E0277"

    @property
    def fixture(self):
        return f"crates/chelis-compiler-api/tests/fixtures/cache_publication/{self.name}.rs"


COMPILE_CASES = tuple(
    CompileCase(name, success, error)
    for name, success, error in (
        ("library", True, None),
        ("stdlib", True, None),
        ("scalar_save", False, "CachePayload"),
        ("scalar_load", False, "CachePayload"),
        ("nested", False, "CachePayload"),
        ("compiled", False, "CachePayload"),
        ("generic", False, "CachePayload"),
        ("implement", False, "Sealed"),
        ("unseal", False, "private"),
    )
)


def validate_fixture_inventory(root: Path) -> None:
    """The compile driver owns exact fixtures, never their containing directory.

    The historical producer is compiled by cache_wire_compatibility and remains
    subject to normal configuration-closure dep-info reconciliation. Its source
    must also be the source recorded with the historical binary artifacts.
    """
    base = root / "crates/chelis-compiler-api/tests/fixtures"
    expected = {case.fixture for case in COMPILE_CASES}
    require(len(expected) == len(COMPILE_CASES), "duplicate cache fixture inventory")
    for directory, wanted in (
        (base / "cache_publication", expected),
        (base / "cache_wire_v3", {str((base / "cache_wire_v3/producer.rs").relative_to(root))}),
    ):
        actual = {str(path.relative_to(root)) for path in directory.rglob("*.rs")}
        require(actual == wanted, f"cache fixture inventory differs: {directory}")
        require(
            all((root / path).is_file() and not (root / path).is_symlink() for path in wanted),
            f"cache fixture inventory contains a non-file or symlink: {directory}",
        )
    historical = base / "cache_wire_v3"
    manifest = json.loads((historical / "producer.json").read_text())
    require(
        hashlib.sha256((historical / "producer.rs").read_bytes()).hexdigest()
        == manifest.get("sha256", {}).get("producer.rs"),
        "historical cache producer source hash differs from its artifact manifest",
    )
OWNERS = (
    "chelis_compiler_api::library_cache::LibraryContext",
    "chelis_compiler_api::stdlib_cache::StdLibContext",
)
PREFIX = """#![allow(dead_code)]
use chelis_compiler_api::{LibraryContext, StdLibContext};
#[path = CACHE_SOURCE]
mod cache_envelope;
const fn same(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] { return false; }
        i += 1;
    }
    true
}
"""


@dataclass(frozen=True)
class CompileOutcome:
    name: str
    returncode: int
    stderr: str
    source_sha256: str
    command: tuple[str, ...]


def validate_compile_outcomes(outcomes):
    require(
        tuple(row.name for row in outcomes) == tuple(c.name for c in COMPILE_CASES),
        "missing, duplicate, reordered, or unexpected cache compile case",
    )
    for case, row in zip(COMPILE_CASES, outcomes, strict=True):
        # Rustc can emit many secondary notes. Keep the primary error visible
        # in the CI log without printing an unbounded compiler transcript.
        first_error = re.search(r"(?m)^error(?:\[[^\n]+\])?:", row.stderr)
        excerpt = row.stderr[first_error.start() if first_error else 0:][:3500]
        excerpt = excerpt or "<empty>"
        require(
            (row.returncode == 0) == case.success,
            f"wrong compile outcome: {case.name}; rustc stderr "
            f"(first error, at most 3500 chars): {excerpt}",
        )
        require(
            row.source_sha256 and row.command, f"missing execution inputs: {case.name}"
        )
        if case.error:
            require(
                case.error in row.stderr and case.diagnostic in row.stderr,
                f"wrong compiler rejection: {case.name}",
            )


RUNTIME_CASES = (
    "historical_caches_contain_actual_old_scalar_and_storage_positional_bytes",
    "current_stdlib_cache_preserves_scalar_storage_bits_and_checked_reconstruction",
    "current_compiled_disk_and_worker_preserve_scalar_storage_bits_and_reconstruct",
    "cache_reconstruction_rejects_changed_numeric_bits_after_checksum_recomputed",
    "historical_artifacts_match_the_recorded_actual_producer",
    "previous_compiled_context_is_rejected_before_live_source_lookup",
    "previous_worker_handoff_is_rejected_before_positional_payload_decode",
    "version_changes_alone_reject_old_subcontexts_before_payload_decode",
    "current_numeric_subcontexts_roundtrip_through_actual_cache_codecs",
    "cache_envelope::tests::absent_file_is_clean_miss",
    "cache_envelope::tests::empty_file_is_corrupt_not_miss",
    "cache_envelope::tests::key_mismatch_is_clean_miss_before_invalid_payload_decode",
    "cache_envelope::tests::missing_magic_is_corrupt_not_miss",
    "cache_envelope::tests::payload_hash_mismatch_is_corrupt_before_decode",
    "cache_envelope::tests::round_trips_on_matching_key_without_changing_envelope_bytes",
    "cache_envelope::tests::save_is_atomic_under_concurrent_writers",
    "cache_envelope::tests::truncated_payload_is_corrupt_not_miss",
    "cache_envelope::tests::version_mismatch_precedes_key_and_payload_admission",
)


def validate_runtime_receipt(receipt):
    require(
        type(receipt) is TestExecution, "framework-owned cache test receipt required"
    )
    require(
        receipt.selected == receipt.executed == tuple(sorted(RUNTIME_CASES)),
        "cache runtime selected and executed identities differ",
    )
    require(
        receipt.command and len(receipt.output_sha256) == 64,
        "cache runtime execution provenance is missing",
    )


def closed_payload_owners(document):
    """Read actual rustdoc trait implementations, including private seal."""
    require(document.get("format_version") == 60, "unsupported cache rustdoc format")
    index = document["index"]
    observed = []
    for trait_name in ("CachePayload", "Sealed"):
        traits = [
            item["inner"]["trait"]
            for item in index.values()
            if item.get("name") == trait_name and "trait" in item.get("inner", {})
        ]
        require(len(traits) == 1, f"missing or ambiguous {trait_name} trait")
        owners = []
        for item_id in traits[0]["implementations"]:
            implementation = index[str(item_id)]["inner"]["impl"]
            require(
                not implementation["generics"]["params"],
                "generic cache payload implementation",
            )
            target = implementation["for"].get("resolved_path")
            require(target is not None, "nonnominal cache payload implementation")
            path = document["paths"].get(str(target["id"]), {}).get("path")
            require(path is not None, "unresolved cache payload identity")
            owners.append("::".join(path))
        require(
            tuple(sorted(owners)) == OWNERS,
            f"unregistered {trait_name} payload owners: {owners}",
        )
        observed.append(tuple(sorted(owners)))
    require(observed[0] == observed[1], "cache payload and seal disagree")
    return observed[0]


@dataclass(frozen=True, init=False, slots=True)
class VerifiedCachePublication:
    root: str
    source_sha256: str
    spec_sha256: str
    document: str
    artifact_sha256: tuple[tuple[str, str], ...]
    outcomes: tuple[CompileOutcome, ...]
    runtime: TestExecution

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "use verify_cache_publication; owner metadata is not compiler evidence"
        )

    def validate(self):
        root = Path(self.root)
        validate_fixture_inventory(root)
        require(
            source_identity(root) == self.source_sha256,
            "cache publication source receipt is stale",
        )
        require(
            hashlib.sha256((root / "spec/10-serialization.md").read_bytes()).hexdigest()
            == self.spec_sha256,
            "cache publication governing specification changed",
        )
        validate_compile_outcomes(self.outcomes)
        validate_runtime_receipt(self.runtime)
        closed_payload_owners(json.loads(self.document))
        for filename, digest in self.artifact_sha256:
            require(
                hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest,
                f"compiled cache publication dependency changed: {filename}",
            )

    def execution_report(self):
        return {
            "source_sha256": self.source_sha256,
            "spec_sha256": self.spec_sha256,
            "document_sha256": hashlib.sha256(self.document.encode()).hexdigest(),
            "owners": OWNERS,
            "artifact_sha256": self.artifact_sha256,
            "cases": [asdict(row) for row in self.outcomes],
            "runtime": asdict(self.runtime),
        }


def verify_cache_publication(root: Path, target: Path) -> VerifiedCachePublication:
    """Build real API dependencies, compile all controls, and derive the seal.

    The caller owns the shared heavy-command slot, like the surrounding schema
    verifier. This function never borrows another worktree's build target.
    """
    root, target = root.resolve(), target.resolve()
    validate_fixture_inventory(root)
    require(
        target.is_relative_to(root / "target"),
        "cache target must belong to this worktree",
    )
    source_before = source_identity(root)
    spec_before = hashlib.sha256(
        (root / "spec/10-serialization.md").read_bytes()
    ).hexdigest()
    fixture_hashes = tuple(
        (str(path), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(
            (root / "crates/chelis-compiler-api/tests/fixtures/cache_wire_v3").iterdir()
        )
        if path.is_file()
    )
    directory = target / "cache-publication"
    directory.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CARGO_TARGET_DIR": str(target),
        "CARGO_BUILD_JOBS": "1",
        "CARGO_HUSKY_DONT_INSTALL_HOOKS": "1",
        "PYO3_PYTHON": sys.executable,
        "VIRTUAL_ENV": sys.prefix,
    }
    command = [
        "cargo",
        "test",
        "--locked",
        "-p",
        "chelis-compiler-api",
        "--test",
        "cache_wire_compatibility",
        "--no-run",
        "--message-format=json",
    ]
    build = subprocess.run(
        command, cwd=root, env=environment, capture_output=True, text=True
    )
    (directory / "build.jsonl").write_text(build.stdout)
    (directory / "build.stderr").write_text(build.stderr)
    require(
        build.returncode == 0,
        f"cache publication dependency build failed: {build.stderr[-4000:]}",
    )
    required = {
        "chelis_compiler_api",
        "chelis_ir",
        "chelis_unord",
        "serde",
        "sha2",
        "bincode",
    }
    artifacts = {}
    test_binary = None
    for line in build.stdout.splitlines():
        row = json.loads(line)
        name = row.get("target", {}).get("name")
        if (
            row.get("reason") == "compiler-artifact"
            and name == "cache_wire_compatibility"
        ):
            require(
                test_binary is None and row.get("executable"),
                "ambiguous cache runtime binary",
            )
            test_binary = row["executable"]
        if row.get("reason") == "compiler-artifact" and name in required:
            files = [f for f in row["filenames"] if f.endswith(".rlib")]
            require(
                len(files) == 1 and name not in artifacts,
                f"ambiguous cache dependency {name}",
            )
            artifacts[name] = files[0]
    require(set(artifacts) == required, "missing compiled cache dependencies")
    require(test_binary is not None, "missing cache runtime binary")
    artifacts["runtime_test"] = test_binary
    artifacts = seal_compiled_artifacts(directory, artifacts)
    test_binary = artifacts["runtime_test"]
    hashes = fixture_hashes + tuple(
        (filename, hashlib.sha256(Path(filename).read_bytes()).hexdigest())
        for _, filename in sorted(artifacts.items())
    )
    runtime = run_libtest(root, Path(test_binary), RUNTIME_CASES)
    validate_runtime_receipt(runtime)
    flags = [
        "--edition=2024",
        "--crate-type=lib",
        "-L",
        f"dependency={target / 'debug/deps'}",
    ]
    for name, filename in sorted(artifacts.items()):
        if name == "runtime_test":
            continue
        flags.extend(["--extern", f"{name}={filename}"])
    prefix = PREFIX.replace(
        "CACHE_SOURCE",
        json.dumps(str(root / "crates/chelis-compiler-api/src/cache_envelope.rs")),
    )
    outcomes = []
    for case in COMPILE_CASES:
        source = prefix + (root / case.fixture).read_text()
        path = directory / (case.name + ".rs")
        path.write_text(source)
        command = [
            "rustc",
            *flags,
            "--crate-name",
            f"cache_{case.name}",
            "--emit=metadata",
            "-o",
            str(directory / (case.name + ".rmeta")),
            str(path),
        ]
        result = subprocess.run(
            command, cwd=root, env=environment, capture_output=True, text=True
        )
        (directory / (case.name + ".stderr")).write_text(result.stderr)
        outcomes.append(
            CompileOutcome(
                case.name,
                result.returncode,
                result.stderr,
                hashlib.sha256(source.encode()).hexdigest(),
                tuple(command),
            )
        )
    validate_compile_outcomes(outcomes)
    command = [
        "rustdoc",
        *flags,
        "--crate-name",
        "cache_publication_fixture",
        "--output-format",
        "json",
        "-Z",
        "unstable-options",
        "--document-private-items",
        "-o",
        str(directory),
        str(directory / "library.rs"),
    ]
    documentation = subprocess.run(
        command,
        cwd=root,
        env={**environment, "RUSTC_BOOTSTRAP": "1"},
        capture_output=True,
        text=True,
    )
    (directory / "rustdoc.stderr").write_text(documentation.stderr)
    require(
        documentation.returncode == 0,
        f"cache publication rustdoc failed: {documentation.stderr[-4000:]}",
    )
    document = canonical(
        json.loads((directory / "cache_publication_fixture.json").read_text())
    )
    closed_payload_owners(json.loads(document))
    require(
        source_identity(root) == source_before,
        "source changed during cache publication verification",
    )
    receipt = object.__new__(VerifiedCachePublication)
    for field, value in dict(
        root=str(root),
        source_sha256=source_before,
        spec_sha256=spec_before,
        document=document,
        artifact_sha256=hashes,
        outcomes=tuple(outcomes),
        runtime=runtime,
    ).items():
        object.__setattr__(receipt, field, value)
    receipt.validate()
    (directory / "receipt.json").write_text(
        json.dumps(receipt.execution_report(), indent=2)
    )
    return receipt


def inventory_binary_graph(documents):
    """Report discovered leaves and barriers without granting codec authority."""
    graph = RustdocGraph(documents)
    roots = graph.serialization_definitions("chelis_compiler_api")
    result = []
    for identity in (
        "chelis_compiler_api::context::CompiledContext",
        "chelis_compiler_api::context::CompiledContextWire",
        "chelis_compiler_api::context::CacheEnvelope",
        "chelis_compiler_api::stdlib_cache::StdLibContext",
        "chelis_compiler_api::stdlib_cache::StdLibContextWire",
        "chelis_compiler_api::library_cache::LibraryContext",
        "chelis_compiler_api::library_cache::LibraryContextWire",
        "chelis_compiler_api::cache_envelope::Envelope",
    ):
        require(identity in roots, f"missing binary payload owner {identity}")
        row = {"identity": identity}
        try:
            discovered = graph.discover("chelis_compiler_api", roots[identity])
            row.update(
                graph_identity=discovered.identity,
                leaves=[asdict(leaf) for leaf in discovered.numeric_leaves],
            )
        except GraphError as error:
            row["barrier"] = str(error)
        if identity in (
            "chelis_compiler_api::context::CompiledContextWire",
            "chelis_compiler_api::stdlib_cache::StdLibContextWire",
            "chelis_compiler_api::library_cache::LibraryContextWire",
        ):
            # The actual custom codecs name these mirrors. Inspecting an
            # independently traversable mirror field reports partial facts;
            # it does not supply the missing execution-backed shape adapter.
            crate, item_id = graph.locations[identity]
            item = graph._item(crate, item_id)
            fields = item["inner"]["struct"]["kind"]["plain"]["fields"]
            row["declared_body_fields"] = []
            for field_id in fields:
                field = graph._item(crate, field_id)
                path = identity + "." + field["name"]
                observation = {"field": path}
                try:
                    discovered = graph.discover(crate, field["inner"]["struct_field"])
                    observation["leaves"] = [
                        {
                            "path": path if leaf.path == "$root" else leaf.path,
                            "primitive": leaf.primitive,
                        }
                        for leaf in discovered.numeric_leaves
                    ]
                except GraphError as error:
                    observation["barrier"] = str(error)
                row["declared_body_fields"].append(observation)
        result.append(row)
    return result
