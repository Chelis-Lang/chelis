"""Fail-closed acceptance for the Chelis semantic core.

Run from the repository root inside the pinned environment:

    nix develop --command python verification/accept.py --lacadile <LaCaDiLE checkout>

Every stage either succeeds or aborts the run with a nonzero status. There is no
skip, reuse or warning-only mode: a stage whose prerequisites are missing fails.

The static native-boundary checks (stage 3 and the shell checks of stage 7) also run alone,
without extraction, on a built or unbuilt Aeneas Lean library and existing plans:

    python verification/accept.py --static --aeneas-lean-dir <backends/lean> \
        [--plan default=<plan.json> --plan lowering-trace=<plan.json>]

All artifacts of one run live in a fresh directory under .work/acceptance/, and
each stage checks that its outputs were produced by this run.

Stages, in order:

1. environment       pinned native compiler, Lean toolchain and LaCaDiLE revision;
                     clean tracked tree.
2. toolchain         Nix builds the patched Charon, charon-ml, Aeneas and Aeneas
                     source; `charon`/`aeneas` on PATH must be exactly those outputs.
                     The Aeneas Lean library is prepared from that source (content
                     checked file by file) and built.
3. native boundary   trusted_base.json `native_models`: exactly the boundary items of
                     native_shell.json, each with exactly one Lean model of the recorded
                     name in the library; `runtime_assumptions` RA-a to RA-j and RA-tls,
                     cited by the Runtime proof headers; every `native_evidence` entry
                     resolves; static checks: no Cargo profile or rustflags select
                     panic = abort (RA-a), no non-test chelis-types code calls
                     catch_unwind (RA-j).
4. native            `cargo fmt --check`, strict Clippy and the workspace tests.
5. model matrix      native versus generated Lean behavior of every library model,
                     float observations, rejected roots and collection contracts.
6. production AD     fresh `grad_dag_checked` extraction and Lean compilation.
7. whole-core plan   fresh call graphs and chunk plans (default and lowering-trace); each
                     plan lists exactly native_shell.json, no core item depends on an
                     internal shell item, and the core dependents of each pinned primitive
                     match the exact chain list in verification/runtime_sites.json.
8. whole-core        separately compiled extraction of both chains and compilation
                     of the unmodified generated Lean.
9. coverage          independent re-validation of each chain's coverage report:
                     every owned and adopted plan item has exactly one accepted
                     status, each emitted one names an entry of its own chunk's
                     manifest (outside the External_Template axioms), and the
                     per-crate counts are recomputed from the items.
10. correspondence   the Lean bridge (ChelisBridge + Audit) built against each
                     chain's whole-core model; Audit fails on disallowed axioms.
                     Model-generated bridge modules (Runtime/CloneId.lean) must be
                     exactly what verification/lean/gen_clone_id.py generates from
                     that chain's fresh model; a stale module rejects the run.
                     verification/lean/ClosureAudit.lean then inventories the
                     transitive closure of every accepted theorem (axioms, opaque
                     constants, extern/implemented_by, witness paths).
11. assumptions      complete inventories for both chains must match the trusted base
                     in verification/trusted_base.json. Any unmatched name rejects.
                     Exact accepted names must match Audit.acceptedFor for the selected chain.
                     Each public theorem needs its exact caller and boundary proposition.
                     All 32 public-entry requirements remain in the contract.
                     Each collection obligation needs an accepted theorem with exactly
                     that proposition as its statement.
"""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
VERIFICATION = ROOT / "verification"
TOOLCHAIN = json.loads((VERIFICATION / "toolchain.json").read_text())
CRATES = (
    "chelis_unord", "chelis_vocab", "chelis_deep", "chelis_pred", "chelis_types",
    "chelis_effects", "chelis_ir", "chelis_pipeline_core",
)
CHAINS = ("default", "lowering-trace")
# Categories of local items that a chunk may legitimately not emit as a declaration of
# its own. Anything else missing from the manifests rejects the run.
JUSTIFIED_UNEMITTED = frozenset({
    "folded",            # constructors, global initializers and promoted constants
    "removed_by_passes",  # impls of hidden marker traits (e.g. Destruct)
    "synthetic",         # compiler-synthesized items without a source definition
    "generic_template",  # generic items whose instances are emitted instead
    "model_provided",    # items bound to a checked library model
    "native_shell",      # FCIS imperative shell listed in verification/native_shell.json
    "model_default",     # provided-method copies replaced by their trait model's default
    "vtable",            # vtable instances (Aeneas translates no vtables)
    "type_alias",        # type aliases, expanded at every use
    "trait_method_decl",  # required trait methods: fields of their trait declaration
})
COVERAGE_VERSION = 3
# Kinds of verification/native_shell.json entries. Primitives and shell types are the
# boundary the extracted core calls, so each must resolve to a Lean model; the other
# kinds must be unreachable from every extracted (non-shell) item.
SHELL_BOUNDARY = frozenset({"primitive", "shell_type"})
SHELL_INTERNAL = frozenset({"shell_state", "shell_entry", "shell_helper"})
# Kinds of proof obligations Aeneas emits with -emit-collection-obligations.
OBLIGATION_KINDS = frozenset({"iterator_size_hint_total"})
RUNTIME_SITE_PRIMITIVES = frozenset({
    *(f"chelis_types::native::{{impl chelis_types::native::NativeServices}}::{name}"
      for name in ("cancellation_observed", "profile_start", "profile_phase")),
    *(f"chelis_ir::lowering_runtime::{{impl chelis_ir::lowering_runtime::LoweringRuntime}}::{name}"
      for name in ("detail_profile_start", "detail_profile_phase",
                   "detail_profile_declaration_start", "detail_profile_declaration", "report_internal")),
})
PUBLIC_ENTRY_COUNTS = {"default": 27, "lowering-trace": 32}
PUBLIC_BOUNDARY_RELATIONS = {
    "resume": "ChelisBridge.Runtime.Entries.BoundaryRelation",
    "catch": "ChelisBridge.Runtime.Entries.CatchBoundaryRelation",
    "panic_on_rejection": "ChelisBridge.Runtime.Entries.PanicBoundaryRelation",
}


class AcceptanceFailure(RuntimeError):
    pass


def run(command, cwd=ROOT, *, env=None, capture=False, log=None):
    print("+ " + shlex.join(str(part) for part in command), flush=True)
    merged = os.environ.copy()
    merged.update(env or {})
    if log is not None:
        with open(log, "w") as handle:
            result = subprocess.run(
                [str(part) for part in command], cwd=cwd, env=merged, text=True,
                stdout=handle, stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise AcceptanceFailure(f"command failed ({result.returncode}); log: {log}")
        return log.read_text() if capture else ""
    result = subprocess.run(
        [str(part) for part in command], cwd=cwd, env=merged, text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    if result.returncode != 0:
        raise AcceptanceFailure(f"command failed ({result.returncode}): {shlex.join(map(str, command))}")
    return result.stdout if capture else ""


def fresh(path: Path, started: float) -> Path:
    """Refuse a missing, empty or pre-existing artifact."""
    if not path.exists():
        raise AcceptanceFailure(f"missing artifact: {path}")
    if path.is_file() and path.stat().st_size == 0:
        raise AcceptanceFailure(f"empty artifact: {path}")
    if path.stat().st_mtime < started:
        raise AcceptanceFailure(f"stale artifact (older than this run): {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digests(directory: Path) -> dict[str, str]:
    """Digest of every file below `directory`, ignoring Lake build state."""
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and ".lake" not in path.relative_to(directory).parts
    }


class Run:
    def __init__(self, args):
        self.args = args
        self.started = time.time()
        head = run(["git", "rev-parse", "HEAD"], capture=True).strip()
        dirty = run(["git", "status", "--porcelain", "--untracked-files=no"], capture=True).strip()
        if dirty and not args.allow_dirty:
            raise AcceptanceFailure("tracked files are modified; commit first or pass --allow-dirty")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.directory = ROOT / ".work" / "acceptance" / f"{stamp}-{head[:10]}"
        self.directory.mkdir(parents=True)
        self.report = {"revision": head, "dirty": bool(dirty), "stages": {}}
        self.env = {"RUSTC_WRAPPER": "", "CARGO_INCREMENTAL": "0"}
        # Filled by the toolchain stage and consumed by later stages.
        self.tools: dict[str, Path] = {}
        self.lean_library: Path | None = None
        self.whole_core: dict[str, Path] = {}
        self.inventories: dict[str, Path] = {}
        self.obligations: dict[str, list[dict]] = {}

    def stage(self, name, function):
        print(f"\n=== {name} ===", flush=True)
        begun = time.time()
        result = function()
        self.report["stages"][name] = {"seconds": round(time.time() - begun, 1), "result": result}
        (self.directory / "report.json").write_text(json.dumps(self.report, indent=2))

    def require_library(self) -> Path:
        if self.lean_library is None:
            raise AcceptanceFailure("the Aeneas Lean library was not prepared by this run")
        return self.lean_library


def stage_environment(context: Run):
    # The checkout root pins no Lean toolchain, so elan runs the pinned one explicitly
    # instead of relying on a user-configured default.
    lean = "+" + TOOLCHAIN["lean"]
    lean_version = TOOLCHAIN["lean"].rsplit(":v", 1)[1]
    versions = {
        "rustc": run(["rustc", "--version"], capture=True).strip(),
        "lean": run(["lean", lean, "--version"], capture=True).strip(),
        "lake": run(["lake", lean, "--version"], capture=True).strip(),
        "python": sys.version.split()[0],
    }
    if not versions["rustc"].startswith("rustc 1.98.0"):
        raise AcceptanceFailure(f"unexpected native compiler: {versions['rustc']}")
    if not versions["lean"].startswith(f"Lean (version {lean_version},"):
        raise AcceptanceFailure(f"unexpected Lean toolchain: {versions['lean']}")
    lacadile = context.args.lacadile.resolve()
    revision = run(["git", "-C", lacadile, "rev-parse", "HEAD"], capture=True).strip()
    if revision != TOOLCHAIN["lacadile"]["revision"]:
        raise AcceptanceFailure(f"LaCaDiLE checkout is at {revision}, not the pinned revision")
    versions["lacadile"] = revision
    return versions


def nix_output(attribute: str) -> Path:
    lines = run(
        ["nix", "build", "--no-link", "--print-out-paths", f".#{attribute}"], capture=True,
    ).split()
    if not lines:
        raise AcceptanceFailure(f"nix build .#{attribute} printed no output path")
    return Path(lines[0])


def stage_toolchain(context: Run):
    outputs = {name: nix_output(name) for name in ("charon", "charon-ml", "aeneas", "aeneas-source")}
    for tool in ("charon", "aeneas"):
        found = shutil.which(tool)
        if found is None or not Path(found).resolve().is_relative_to(outputs[tool]):
            raise AcceptanceFailure(
                f"`{tool}` on PATH ({found}) is not the pinned patched build {outputs[tool]}; "
                "run inside `nix develop`"
            )
        context.tools[tool] = Path(found).resolve()
    source = outputs["aeneas-source"] / "backends" / "lean"
    if (source / "lean-toolchain").read_text().strip() != TOOLCHAIN["lean"]:
        raise AcceptanceFailure("the patched Aeneas library does not pin the expected Lean toolchain")
    # A prepared copy is keyed by the content-addressed source path and checked file by
    # file, so neither a stale nor a locally edited library can be accepted.
    library = ROOT / ".work" / "acceptance" / "aeneas-lean" / outputs["aeneas-source"].name
    if library.exists():
        expected, actual = tree_digests(source), tree_digests(library)
        if expected != actual:
            changed = sorted(set(expected.items()) ^ set(actual.items()))[:5]
            raise AcceptanceFailure(f"prepared Aeneas library diverges from its source: {changed}")
        run(["lake", "build"], cwd=library, log=context.directory / "aeneas-lean-build.log")
    else:
        library.parent.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, VERIFICATION / "prepare_aeneas_lean.py",
            "--source", outputs["aeneas-source"], "--destination", library,
        ], log=context.directory / "aeneas-lean-build.log")
    context.lean_library = library
    return {
        **{name: str(path) for name, path in outputs.items()},
        "binaries": {name: sha256_file(path) for name, path in context.tools.items()},
        "aeneas_lean": str(library),
    }


# The ids of trusted_base.json `runtime_assumptions`: RA-a to RA-j (NATIVE_SHELL_SCOPE O2) and
# RA-tls (the thread-local abstraction alpha_t, O4).
RUNTIME_ASSUMPTIONS = frozenset({*(f"RA-{letter}" for letter in "abcdefghij"), "RA-tls"})
# The Runtime proof modules whose headers state the native semantics they assume.
RUNTIME_HEADERS = tuple(
    f"verification/lean/ChelisBridge/Runtime/{name}.lean"
    for name in ("Unwind", "Stack", "Cancellation", "Ids"))
EVIDENCE_KINDS = frozenset({"test", "text", "model", "accept", "lean"})


def lean_model_declarations(aeneas_lean: Path) -> tuple[dict[str, str], collections.Counter]:
    """The Lean declaration of each model pattern (extract_whole.lean_models) and the number
    of model attributes per normalized pattern in the library (attributes may span lines)."""
    from extract_whole import MODEL_ATTRIBUTE, lean_models, normalize_pattern

    counts = collections.Counter(
        normalize_pattern(match.group(1))
        for path in sorted((aeneas_lean / "Aeneas").rglob("*.lean"))
        for match in MODEL_ATTRIBUTE.finditer(path.read_text(errors="replace")))
    return lean_models(aeneas_lean), counts


def check_native_models(trusted: dict, aeneas_lean: Path) -> int:
    """`native_models` lists exactly the boundary items of native_shell.json, and each has
    exactly one model declaration in the library, with the recorded Lean name."""
    from extract_whole import normalize_pattern

    declared = {item["pattern"]: item["kind"]
                for item in json.loads((VERIFICATION / "native_shell.json").read_text())["items"]
                if item["kind"] in SHELL_BOUNDARY}
    entries = trusted.get("native_models", [])
    recorded = collections.Counter(entry["item"] for entry in entries)
    duplicated = sorted(item for item, count in recorded.items() if count > 1)
    missing = sorted(set(declared) - set(recorded))
    extra = sorted(set(recorded) - set(declared))
    if duplicated or missing or extra:
        raise AcceptanceFailure(
            f"trusted_base.json native_models differs from the boundary of native_shell.json: "
            f"missing {missing}, extra {extra}, duplicated {duplicated}")
    models, counts = lean_model_declarations(aeneas_lean)
    for entry in entries:
        key = normalize_pattern(entry["item"])
        if entry.get("kind") != declared[entry["item"]] or not entry.get("model"):
            raise AcceptanceFailure(f"native_models entry {entry['item']}: wrong kind or no model text")
        if counts[key] != 1:
            raise AcceptanceFailure(
                f"native boundary item {entry['item']} has {counts[key]} Lean model declarations "
                f"in {aeneas_lean}, not exactly one")
        if models.get(key) != entry.get("lean"):
            raise AcceptanceFailure(
                f"native boundary item {entry['item']}: the library models it as {models.get(key)}, "
                f"trusted_base.json records {entry.get('lean')}")
    return len(entries)


def lean_header(path: Path) -> str:
    """The module docstring (the first `/-! ... -/` block) of a Lean file."""
    text = path.read_text()
    start = text.find("/-!")
    return text[start:text.index("-/", start)] if start >= 0 else ""


def check_runtime_assumptions(trusted: dict) -> int:
    entries = trusted.get("runtime_assumptions", [])
    ids = collections.Counter(entry.get("id") for entry in entries)
    if set(ids) != RUNTIME_ASSUMPTIONS or any(count > 1 for count in ids.values()):
        raise AcceptanceFailure(
            f"trusted_base.json runtime_assumptions must list each of {sorted(RUNTIME_ASSUMPTIONS)} "
            f"once, found {sorted(ids.elements(), key=str)}")
    cited = collections.defaultdict(set)
    for entry in entries:
        if not entry.get("assumption") or not entry.get("justification") or not entry.get("evidence"):
            raise AcceptanceFailure(f"runtime assumption {entry['id']} lacks text or evidence")
        for header in entry.get("cited_by", []):
            if header not in RUNTIME_HEADERS:
                raise AcceptanceFailure(f"runtime assumption {entry['id']}: unknown header {header}")
            if entry["id"] not in re.findall(r"RA-[a-z]+", lean_header(ROOT / header)):
                raise AcceptanceFailure(f"{header}: its header does not cite {entry['id']}")
            cited[header].add(entry["id"])
    uncited = [header for header in RUNTIME_HEADERS if not cited[header]]
    if uncited:
        raise AcceptanceFailure(f"Runtime headers that cite no runtime assumption: {uncited}")
    return len(entries)


def check_evidence(trusted: dict) -> int:
    """Every evidence id cited by native_models and runtime_assumptions exists and resolves to
    a test function, a text, a model check, an accept.py check or an audited Lean theorem."""
    from check_models import CRATE_MODELS, MODELS

    evidence = {entry["id"]: entry for entry in trusted.get("native_evidence", [])}
    if len(evidence) != len(trusted.get("native_evidence", [])):
        raise AcceptanceFailure("trusted_base.json native_evidence has duplicate ids")
    audit = (VERIFICATION / "lean" / "ChelisBridge" / "Audit.lean").read_text()
    for entry in evidence.values():
        path, item, kind = ROOT / entry["path"], entry["item"], entry["kind"]
        if kind not in EVIDENCE_KINDS or not path.is_file() or not entry.get("description"):
            raise AcceptanceFailure(f"native evidence {entry['id']}: bad kind, path or description")
        text = path.read_text()
        found = {
            "test": re.search(rf"#\[test\]\s*(?:#\[[^\]]*\]\s*)*fn {re.escape(item)}\(", text),
            "text": item in text,
            "model": item in MODELS or item in CRATE_MODELS,
            "accept": re.search(rf"^def {re.escape(item)}\(", text, re.M),
            "lean": re.search(rf"\btheorem {re.escape(item.rsplit('.', 1)[-1])}\b", text)
                    and f"``{item}," in audit,
        }[kind]
        if not found:
            raise AcceptanceFailure(f"native evidence {entry['id']}: {kind} {item} not found in {entry['path']}")
    for section in ("native_models", "runtime_assumptions"):
        for entry in trusted.get(section, []):
            unknown = [ident for ident in entry.get("evidence", []) if ident not in evidence]
            if unknown:
                name = entry.get("item", entry.get("id"))
                raise AcceptanceFailure(f"{section} {name} cites unknown evidence {unknown}")
    return len(evidence)


def check_panic_strategy() -> list[str]:
    """RA-a: no Cargo profile or rustflags of the workspace select panic = abort."""
    import tomllib

    manifests = [ROOT / "Cargo.toml", *sorted((ROOT / "crates").glob("*/Cargo.toml"))]
    configs = [path for path in (ROOT / ".cargo" / "config.toml", ROOT / ".cargo" / "config")
               if path.is_file()]
    for path in manifests + configs:
        data = tomllib.loads(path.read_text())
        for profile, settings in data.get("profile", {}).items():
            for scope in [settings, *settings.get("package", {}).values()]:
                if scope.get("panic", "unwind") != "unwind":
                    raise AcceptanceFailure(
                        f"{path.relative_to(ROOT)}: profile {profile} sets panic = {scope['panic']!r}; "
                        f"the native shell requires unwinding (RA-a)")
        flags = [data.get("build", {}).get("rustflags", [])]
        flags += [target.get("rustflags", []) for target in data.get("target", {}).values()
                  if isinstance(target, dict)]
        for value in flags:
            joined = " ".join(value) if isinstance(value, list) else str(value)
            if re.search(r"panic\s*=\s*abort", joined):
                raise AcceptanceFailure(
                    f"{path.relative_to(ROOT)}: rustflags select panic=abort (RA-a)")
    return [str(path.relative_to(ROOT)) for path in manifests + configs]


LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)\'|.', re.S)


def non_test_rust(text: str) -> str:
    """Rust source without comments and without `#[cfg(test)]` items."""
    kept, index = [], 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
        elif text.startswith("/*", index):
            index = text.index("*/", index) + 2
        else:
            # A string or char literal is kept whole, so `//` or `"` inside it is not syntax.
            match = LITERAL.match(text, index)
            kept.append(match.group(0))
            index = match.end()
    source = "".join(kept)
    while (start := source.find("#[cfg(test)]")) >= 0:
        body = re.compile(r"[{;]").search(source, start + len("#[cfg(test)]"))
        end = body.end()
        if body.group(0) == "{":
            depth = 1
            while depth:
                depth += {"{": 1, "}": -1}.get(source[end], 0)
                end += 1
        source = source[:start] + source[end:]
    return source


def check_no_catch_unwind(crate: str = "chelis-types") -> int:
    """RA-j: no non-test code of the crate calls catch_unwind. Test modules declared by
    `#[cfg(test)] mod name;` (and their submodules) and `#[cfg(test)]` items are skipped."""
    source = ROOT / "crates" / crate / "src"
    files = sorted(source.rglob("*.rs"))
    test_roots = []
    for path in files:
        for name in re.findall(r"#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;",
                               path.read_text()):
            directory = path.parent if path.name in ("mod.rs", "lib.rs", "main.rs") else path.with_suffix("")
            test_roots += [directory / f"{name}.rs", directory / name]
    scanned = 0
    for path in files:
        if any(path == root or root in path.parents for root in test_roots):
            continue
        scanned += 1
        if re.search(r"\bcatch_unwind\b", non_test_rust(path.read_text())):
            raise AcceptanceFailure(
                f"{path.relative_to(ROOT)}: non-test code calls catch_unwind; a caught unwind "
                f"would expose a DiagnosticSink after skipped scope exits (RA-j)")
    return scanned


def check_runtime_sites() -> dict:
    """Check both chain lists and the complete primitive inventory."""
    contract = json.loads((VERIFICATION / "runtime_sites.json").read_text())
    if contract.get("schema_version") != 2:
        raise AcceptanceFailure("runtime_sites.json: expected schema_version 2")
    sites = contract.get("sites")
    reasons = contract.get("justifications")
    if not isinstance(sites, dict) or set(sites) != set(CHAINS):
        raise AcceptanceFailure("runtime_sites.json: expected exactly both chain lists")
    if not isinstance(reasons, dict) or set(reasons) != RUNTIME_SITE_PRIMITIVES:
        raise AcceptanceFailure("runtime_sites.json: incomplete primitive justifications")
    if any(not isinstance(reason, str) or not reason.strip() for reason in reasons.values()):
        raise AcceptanceFailure("runtime_sites.json: empty primitive justification")
    for chain, primitives in sites.items():
        if not isinstance(primitives, dict) or set(primitives) != RUNTIME_SITE_PRIMITIVES:
            raise AcceptanceFailure(f"runtime_sites.json: {chain} has an incomplete primitive inventory")
        for pattern, callers in primitives.items():
            if (not isinstance(callers, list) or not callers
                    or any(not isinstance(name, str) or not name.strip() for name in callers)
                    or len(callers) != len(set(callers))):
                raise AcceptanceFailure(f"runtime_sites.json: invalid sites for {chain}: {pattern}")
    return sites


def check_public_entries() -> dict[str, set[str]]:
    """Keep all 32 public-entry proof requirements and their chain assignments."""
    contract = json.loads((VERIFICATION / "public_entries.json").read_text())
    entries = contract.get("entries")
    if contract.get("schema_version") != 1 or not isinstance(entries, list) or len(entries) != 32:
        raise AcceptanceFailure("public_entries.json: expected all 32 entries in schema_version 1")
    required = {chain: set() for chain in CHAINS}
    seen = {field: set() for field in ("production", "inner", "theorem")}
    for entry in entries:
        if not isinstance(entry, dict):
            raise AcceptanceFailure("public_entries.json: invalid entry")
        for field, names in seen.items():
            name = entry.get(field)
            if not isinstance(name, str) or not name.strip() or name in names:
                raise AcceptanceFailure(f"public_entries.json: missing or duplicate {field}: {name}")
            names.add(name)
        if not entry["theorem"].startswith("ChelisBridge.Runtime.Entries."):
            raise AcceptanceFailure(f"public_entries.json: invalid theorem name: {entry['theorem']}")
        if entry.get("boundary_kind") not in {"resume", "catch", "panic_on_rejection"}:
            raise AcceptanceFailure(f"public_entries.json: invalid boundary kind: {entry['production']}")
        chains = entry.get("chains")
        if (not isinstance(chains, list) or not chains
                or any(not isinstance(chain, str) or chain not in CHAINS for chain in chains)
                or len(chains) != len(set(chains))):
            raise AcceptanceFailure(f"public_entries.json: invalid chains for {entry['production']}")
        for chain in chains:
            required[chain].add(entry["theorem"])
    for chain, names in required.items():
        if len(names) != PUBLIC_ENTRY_COUNTS[chain]:
            raise AcceptanceFailure(f"public_entries.json: {chain} needs {PUBLIC_ENTRY_COUNTS[chain]} entries")
    return required


def public_entry_statements(chain: str) -> dict[str, dict[str, str]]:
    """Read exact caller propositions for the selected chain."""
    check_public_entries()
    entries = json.loads((VERIFICATION / "public_entries.json").read_text())["entries"]
    statements = {}
    for entry in entries:
        if chain not in entry["chains"]:
            continue
        for field in ("statement", "lean_inner", "lean_boundary"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise AcceptanceFailure(
                    f"{chain}: missing exact public-entry {field}: {entry['production']}")
        if entry["lean_boundary"] != PUBLIC_BOUNDARY_RELATIONS[entry["boundary_kind"]]:
            raise AcceptanceFailure(
                f"{chain}: caller boundary differs from its kind: {entry['production']}")
        statements[entry["theorem"]] = {
            field: entry[field] for field in ("statement", "lean_inner", "lean_boundary", "boundary_kind")}
    return statements


def accepted_roots(chain: str, required: set[str]) -> set[str]:
    """Select common roots and the exact applicable caller roots."""
    source = (VERIFICATION / "lean" / "ChelisBridge" / "Audit.lean").read_text()
    lists = {}
    for name in ("acceptedCommon", "wholeCoreRoots", "defaultEntryRoots", "traceOnlyEntryRoots"):
        declaration = re.search(rf"^def {name} : List Name := \[(.*?)^\]", source, re.M | re.S)
        if declaration is None:
            raise AcceptanceFailure(f"Audit.lean: {name} not found")
        names = re.findall(r"^\s*`{1,2}([A-Za-z0-9_.']+)\s*,?\s*$", declaration.group(1), re.M)
        if not names or len(names) != len(set(names)):
            raise AcceptanceFailure(f"Audit.lean: invalid {name}")
        lists[name] = set(names)
    callers = lists["defaultEntryRoots"]
    if chain == "lowering-trace":
        callers = callers | lists["traceOnlyEntryRoots"]
    common = lists["acceptedCommon"] | lists["wholeCoreRoots"]
    if (lists["acceptedCommon"] & lists["wholeCoreRoots"]
            or callers != required or callers & common):
        raise AcceptanceFailure(f"{chain}: Audit caller roots differ from public_entries.json")
    return common | callers


def check_shell_plan(chain: str, plan: dict) -> dict:
    """Check the shell inventory and exact cancellation and profile callers."""
    if chain not in CHAINS:
        raise AcceptanceFailure(f"unknown plan chain: {chain}")
    if not isinstance(plan, dict) or plan.get("chain") != chain:
        raise AcceptanceFailure(f"{chain} plan: chain identity differs")
    sites = check_runtime_sites()[chain]
    declared = {item["pattern"]: item["kind"]
                for item in json.loads((VERIFICATION / "native_shell.json").read_text())["items"]}
    shell = plan.get("native_shell")
    if not isinstance(shell, dict) or set(shell) != set(declared):
        raise AcceptanceFailure(f"{chain} plan: native shell differs from native_shell.json")
    for pattern, item in shell.items():
        callers = item.get("core_dependents") if isinstance(item, dict) else None
        if (not isinstance(callers, list)
                or any(not isinstance(name, str) or not name.strip() for name in callers)
                or len(callers) != len(set(callers))):
            raise AcceptanceFailure(f"{chain} plan: invalid core_dependents for {pattern}")
    internal = [pattern for pattern, kind in declared.items() if kind in SHELL_INTERNAL]
    reached = {pattern: shell[pattern]["core_dependents"] for pattern in internal
               if shell[pattern]["core_dependents"]}
    if reached:
        raise AcceptanceFailure(f"{chain} plan: core items depend on internal shell items: {reached}")
    for pattern, pinned in sites.items():
        actual = set(shell[pattern]["core_dependents"])
        if actual != set(pinned):
            raise AcceptanceFailure(
                f"{chain} plan: the core sites of {pattern} differ from runtime_sites.json: "
                f"new {sorted(actual - set(pinned))}, gone {sorted(set(pinned) - actual)}")
    return {"internal": len(internal), "pinned": {pattern: len(pinned)
                                                for pattern, pinned in sites.items()}}


def static_checks(aeneas_lean: Path) -> dict:
    trusted = json.loads((VERIFICATION / "trusted_base.json").read_text())
    return {
        "native_models": check_native_models(trusted, aeneas_lean),
        "runtime_assumptions": check_runtime_assumptions(trusted),
        "evidence": check_evidence(trusted),
        "panic_strategy": check_panic_strategy(),
        "catch_unwind_free_files": check_no_catch_unwind(),
        "runtime_sites": {chain: {pattern: len(callers) for pattern, callers in primitives.items()}
                          for chain, primitives in check_runtime_sites().items()},
        "public_entries": {chain: len(names) for chain, names in check_public_entries().items()},
    }


def stage_native_boundary(context: Run):
    return static_checks(context.require_library())


def stage_native(context: Run):
    logs = context.directory / "native"
    logs.mkdir()
    target = context.directory / "cargo-target"
    env = {**context.env, "CARGO_TARGET_DIR": str(target)}
    run(["cargo", "fmt", "--all", "--check"], env=env, log=logs / "fmt.log")
    run([
        "cargo", "clippy", "--workspace", "--locked", "--all-targets", "--all-features",
        "--jobs", "2", "--", "-D", "warnings",
    ], env=env, log=logs / "clippy.log")
    run([
        "cargo", "test", "--workspace", "--locked", "--all-features", "--jobs", "2",
    ], env=env, log=logs / "test.log")
    passed = failed = ignored = 0
    for line in (logs / "test.log").read_text().splitlines():
        if line.startswith("test result:"):
            fields = line.replace(";", "").split()
            passed += int(fields[fields.index("passed") - 1])
            failed += int(fields[fields.index("failed") - 1])
            ignored += int(fields[fields.index("ignored") - 1])
    if failed or passed == 0:
        raise AcceptanceFailure(f"native tests: {passed} passed, {failed} failed")
    shutil.rmtree(target)
    return {"passed": passed, "failed": failed, "ignored": ignored}


def stage_models(context: Run):
    log = context.directory / "models.log"
    run([
        sys.executable, VERIFICATION / "check_models.py",
        "--aeneas-lean-dir", context.require_library(),
    ], env=context.env, log=log)
    return {"log": str(log)}


def stage_production_ad(context: Run):
    begun = time.time()
    log = context.directory / "production-ad.log"
    output = run([
        sys.executable, VERIFICATION / "check_production_ad.py",
        "--aeneas-lean-dir", context.require_library(),
    ], env=context.env, log=log, capture=True)
    compiled = [line.split(": ", 1)[1] for line in output.splitlines()
                if line.startswith("Compiled model: ")]
    if len(compiled) != 1:
        raise AcceptanceFailure(f"the production AD gate reported no compiled model; log: {log}")
    model = fresh(Path(compiled[0]), begun)
    return {"compiled": str(model), "sha256": sha256_file(model)}


def stage_plan(context: Run):
    plans = {}
    for chain in CHAINS:
        work = context.directory / "plan" / chain
        plan = context.directory / "plan" / f"{chain}.json"
        work.parent.mkdir(parents=True, exist_ok=True)
        begun = time.time()
        run([
            sys.executable, VERIFICATION / "plan_chunks.py", "--work", work, "--out", plan,
            "--chain", chain, "--charon", context.tools["charon"],
        ], env=context.env, log=context.directory / f"plan-{chain}.log")
        plans[chain] = fresh(plan, begun)
    context.whole_core = {f"plan:{chain}": path for chain, path in plans.items()}
    results = {}
    for chain, path in plans.items():
        plan = json.loads(path.read_text())
        results[chain] = {"plan": str(path), "chunks": len(plan["chunks"]),
                          "native_shell": check_shell_plan(chain, plan)}
    return results


def stage_whole_core(context: Run):
    results = {}
    for chain in CHAINS:
        out = context.directory / "whole-core" / chain
        out.parent.mkdir(parents=True, exist_ok=True)
        begun = time.time()
        run([
            sys.executable, VERIFICATION / "extract_whole.py",
            "--plan", context.whole_core[f"plan:{chain}"], "--out", out,
            "--charon", context.tools["charon"], "--aeneas", context.tools["aeneas"],
            "--aeneas-lean-dir", context.require_library(),
        ], env=context.env, log=context.directory / f"whole-core-{chain}.log")
        summary = json.loads(fresh(out / "summary.json", begun).read_text())
        if summary.get("mode") == "diagnostic" or summary.get("diagnostic"):
            raise AcceptanceFailure(f"{chain}: extraction ran in diagnostic mode")
        for tool in ("charon", "aeneas"):
            if summary.get("binaries", {}).get(tool) != sha256_file(context.tools[tool]):
                raise AcceptanceFailure(f"{chain}: extraction did not use the pinned {tool}")
        fresh(out / "coverage.json", begun)
        fresh(out / "lean", begun)
        context.obligations[chain] = read_obligations(chain, fresh(out / "obligations.json", begun))
        context.whole_core[chain] = out
        results[chain] = {key: summary.get(key) for key in ("chunks", "declarations", "modules")}
        results[chain]["obligations"] = len(context.obligations[chain])
    return results


def read_obligations(chain: str, path: Path) -> list[dict]:
    """The merged -emit-collection-obligations output of a chain (possibly empty)."""
    report = json.loads(path.read_text())
    if report.get("version") != 1 or not isinstance(report.get("obligations"), list):
        raise AcceptanceFailure(f"{chain}: malformed obligations report {path}")
    for obligation in report["obligations"]:
        if obligation.get("kind") not in OBLIGATION_KINDS:
            raise AcceptanceFailure(f"{chain}: unknown obligation kind in {obligation}")
        if not obligation.get("proposition") or not obligation.get("impl"):
            raise AcceptanceFailure(f"{chain}: incomplete obligation {obligation}")
    return report["obligations"]


def validate_coverage(chain: str, coverage: dict, plan: dict, run: Path) -> dict:
    if coverage.get("format") != "chelis-wholecore-coverage" or coverage.get("version") != COVERAGE_VERSION:
        raise AcceptanceFailure(f"{chain}: coverage report is not chelis-wholecore-coverage v{COVERAGE_VERSION}")
    if coverage.get("unknown_externals") != 0:
        raise AcceptanceFailure(f"{chain}: unknown externals: {coverage.get('unknown_externals')}")
    if coverage.get("opaque_local_items") != 0:
        raise AcceptanceFailure(f"{chain}: opaque local items: {coverage.get('opaque_local_items')}")
    if coverage.get("misplaced_items"):
        raise AcceptanceFailure(f"{chain}: chunks emit items other chunks own: {coverage['misplaced_items'][:5]}")
    statuses = validate_items(chain, coverage, plan, run)
    reject_sorry(chain, plan, run)
    crates = coverage.get("crates", {})
    expected = set(CRATES) if chain == "default" else {"chelis_ir", "chelis_pipeline_core"}
    if not expected <= set(crates):
        raise AcceptanceFailure(f"{chain}: coverage lacks crates {sorted(expected - set(crates))}")
    totals = {}
    for crate in sorted(expected):
        entry = crates[crate]
        unemitted = entry.get("unemitted", {})
        unknown = set(unemitted) - JUSTIFIED_UNEMITTED
        if unknown:
            raise AcceptanceFailure(f"{chain}/{crate}: unjustified unemitted categories {sorted(unknown)}")
        defects = {name: entry.get(name, 0) for name in ("missing", "axioms", "sorry", "duplicates")}
        if any(defects.values()):
            raise AcceptanceFailure(f"{chain}/{crate}: defective declarations {defects}")
        accounted = entry.get("emitted", 0) + sum(unemitted.values())
        if entry.get("local_items", -1) != accounted or accounted == 0:
            raise AcceptanceFailure(
                f"{chain}/{crate}: {entry.get('local_items')} local items but {accounted} accounted for"
            )
        recomputed = statuses.get(crate, collections.Counter())
        derived = {"emitted": entry.get("emitted", 0), **unemitted}
        derived["folded"] = derived.get("folded", 0) - sum(
            chunk.get("folded", 0) for chunk in plan["chunks"] if chunk["crate"] == crate)
        derived["native_shell"] = derived.get("native_shell", 0) - sum(
            len(item["members"]) for item in plan["native_shell"].values() if item["crate"] == crate)
        if {name: count for name, count in derived.items() if count} != dict(recomputed):
            raise AcceptanceFailure(
                f"{chain}/{crate}: reported counts {derived} differ from its items' statuses {dict(recomputed)}"
            )
        totals[crate] = {"local_items": entry["local_items"], "emitted": entry["emitted"], **unemitted}
    validate_shell(chain, coverage.get("native_shell", {}))
    return totals


def validate_items(chain: str, coverage: dict, plan: dict, run: Path) -> dict[str, collections.Counter]:
    """Fail closed on any owned or adopted plan item without an accepted status. The
    report must list exactly the plan's items; an emitted item must name an entry of its
    own chunk's manifest (read here) that no other item claims, outside the
    External_Template axioms; a justified item must carry the plan evidence its category
    needs. Returns the status counts per crate."""
    expected = [(chunk["name"], crate, item["kind"], item["name"])
                for chunk in plan["chunks"]
                for crate, item in ([(chunk["crate"], item) for item in chunk["owned"]]
                                    + [(item["crate"], item) for item in chunk.get("adopted", [])])]
    rows = coverage.get("items")
    if not isinstance(rows, list) or [tuple(row[1:5]) for row in rows] != expected:
        raise AcceptanceFailure(f"{chain}: coverage items differ from the plan's owned and adopted items")
    plan_items = {(chunk["name"], item["kind"], item["name"]): item
                  for chunk in plan["chunks"] for item in chunk["owned"] + chunk.get("adopted", [])}
    manifests: dict[str, dict[str, dict]] = {}
    for index, chunk in enumerate(plan["chunks"]):
        path = run / "chunks" / f"{index:02d}-{chunk['name']}" / "manifest.json"
        manifests[chunk["name"]] = ({entry["key"]: entry for entry in json.loads(path.read_text())["entries"]}
                                    if path.is_file() else {})
    trait_fields = {key.partition(" ")[2]: set(entry.get("methods") or ())
                    for entries in manifests.values() for key, entry in entries.items()
                    if key.startswith("trait_decl ")}
    status_of = {(row[3], row[4]): row[0] for row in rows}
    unaccounted, claimed = [], {}
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for status, chunk, crate, kind, name, key in rows:
        counts[crate][status] += 1
        item = plan_items[(chunk, kind, name)]
        source = str(item.get("source", ""))
        entry = manifests[chunk].get(key) if key else None
        if status == "emitted":
            if entry is None or entry["module"].endswith("External_Template") or key.partition(" ")[0] != kind:
                problem = f"its key {key!r} is no {kind} declaration of {chunk}'s manifest"
            elif key in claimed:
                problem = f"its key {key!r} is also claimed by {claimed[key]}"
            else:
                claimed[key] = name
                continue
        elif status not in JUSTIFIED_UNEMITTED - {"folded", "native_shell"}:
            problem = f"status {status}"
        elif status == "removed_by_passes" and not item.get("removed_by_passes"):
            problem = "not removed by Charon's passes"
        elif status == "synthetic" and source != "synthetic":
            problem = "not synthetic"
        elif status == "vtable" and not (kind == "global" and source.startswith("VTableInstance")):
            problem = "not a vtable instance"
        elif status == "model_default" and not (
                source.startswith("defaulted_method(")
                and status_of.get(("trait_impl", name.rpartition("::")[0])) == "emitted"):
            problem = "not a provided-method copy of an emitted implementation"
        elif status == "trait_method_decl" and name.rpartition("::")[2] not in trait_fields.get(
                name.rpartition("::")[0], ()):
            problem = "not a field of an emitted trait declaration"
        else:
            continue
        unaccounted.append(f"{chunk}: {kind} {name}: {problem}")
    if unaccounted:
        raise AcceptanceFailure(f"{chain}: {len(unaccounted)} owned items unaccounted for, e.g. {unaccounted[:5]}")
    return counts


def reject_sorry(chain: str, plan: dict, run: Path) -> None:
    """An emitted declaration is only translated if no generated module uses `sorry`
    (outside comments): Aeneas writes `sorry` for a body or part it could not translate."""
    import re
    found = []
    for chunk in plan["chunks"]:
        for path in sorted((run / "lean" / chunk["module"]).glob("*.lean")):
            text = re.sub(r"/-.*?-/", lambda match: "\n" * match.group(0).count("\n"),
                          path.read_text(errors="replace"), flags=re.S)
            for number, line in enumerate(text.splitlines(), 1):
                if re.search(r"\bsorry\b", line.split("--", 1)[0]):
                    found.append(f"{path.relative_to(run)}:{number}")
    if found:
        raise AcceptanceFailure(f"{chain}: generated Lean uses sorry ({len(found)} lines, e.g. {found[:5]})")


def validate_shell(chain: str, shell: dict) -> None:
    """The shell report must list exactly verification/native_shell.json, select
    exactly the members an entry pins, bind every boundary item to a Lean model,
    and show no extracted item reaching shell internals."""
    declared = json.loads((VERIFICATION / "native_shell.json").read_text())["items"]
    reported = shell.get("items", {})
    if set(reported) != {item["pattern"] for item in declared}:
        raise AcceptanceFailure(f"{chain}: native shell report differs from native_shell.json")
    for item in declared:
        entry = reported[item["pattern"]]
        if "members" in item and sorted(entry.get("members", [])) != sorted(item["members"]):
            raise AcceptanceFailure(
                f"{chain}: shell entry {item['pattern']} selects {sorted(entry.get('members', []))}, "
                f"not its pinned members")
        if item["kind"] in SHELL_BOUNDARY and not entry.get("lean_model"):
            raise AcceptanceFailure(f"{chain}: shell boundary item {item['pattern']} has no Lean model")
        if item["kind"] in SHELL_INTERNAL and entry.get("core_dependents"):
            raise AcceptanceFailure(
                f"{chain}: extracted items depend on shell internal {item['pattern']}: "
                f"{entry['core_dependents'][:5]}"
            )
        if item["kind"] not in SHELL_BOUNDARY | SHELL_INTERNAL:
            raise AcceptanceFailure(f"{chain}: unknown shell kind {item['kind']} for {item['pattern']}")


def stage_coverage(context: Run):
    return {
        chain: validate_coverage(
            chain, json.loads((context.whole_core[chain] / "coverage.json").read_text()),
            json.loads(context.whole_core[f"plan:{chain}"].read_text()), context.whole_core[chain])
        for chain in CHAINS
    }


def stage_correspondence(context: Run):
    bridge = VERIFICATION / "lean"
    run([
        sys.executable, bridge / "prepare_bridge.py",
        "--aeneas-lean-dir", context.require_library(), "--lacadile-clone", context.args.lacadile,
    ], cwd=bridge, log=context.directory / "bridge-prepare.log")
    results = {}
    for chain in CHAINS:
        model = context.whole_core[chain] / "lean"
        summary = json.loads((context.whole_core[chain] / "summary.json").read_text())
        roots = ",".join(summary["modules"])
        # The clone-identity theorems are generated from a model: this chain's fresh model
        # must generate exactly the committed module (checked, never regenerated here).
        clone_id = context.directory / f"clone-id-{chain}.log"
        run([sys.executable, bridge / "gen_clone_id.py", "--check", model, roots,
             bridge / "ChelisBridge" / "Runtime" / "CloneId.lean"], cwd=bridge, log=clone_id)
        log = context.directory / f"bridge-{chain}.log"
        begun = time.time()
        run([
            "lake", "build", "-R", "-K", f"model={model}", "-K", f"modelRoots={roots}",
            "-K", f"chain={chain}", "-K", "scope=whole-core",
            "ChelisBridge", "ChelisBridge.Audit",
        ], cwd=bridge, log=log)
        # The inventory must describe exactly this chain's build, so it is taken before
        # the next chain replaces the model.
        inventory = context.directory / f"assumptions-{chain}.json"
        run(["lake", "env", "lean", "--run", "ClosureAudit.lean", "whole-core", chain,
             VERIFICATION / "public_entries.json", inventory],
            cwd=bridge, log=context.directory / f"assumptions-{chain}.log")
        context.inventories[chain] = fresh(inventory, begun)
        results[chain] = {"log": str(log), "inventory": str(inventory), "clone_id": str(clone_id)}
    return results


def check_closure_inventory(chain: str, found: dict, required: set[str]) -> set[str]:
    """Reject incomplete closure reports and missing public-entry theorem evidence."""
    if chain not in CHAINS:
        raise AcceptanceFailure(f"unknown inventory chain: {chain}")
    fields = {"scope", "chain", "accepted", "accepted_names", "reachable", "axioms", "opaque", "extern",
              "modules", "discharged", "witnesses", "entry_statements"}
    if not isinstance(found, dict) or not fields.issubset(found):
        raise AcceptanceFailure(f"{chain}: incomplete closure inventory")
    if found["chain"] != chain:
        raise AcceptanceFailure(f"{chain}: closure inventory uses a different chain")
    if found["scope"] != "whole-core":
        raise AcceptanceFailure(f"{chain}: fragment inventory cannot satisfy whole-core acceptance")
    for field in ("accepted_names", "axioms", "opaque", "extern"):
        names = found[field]
        if (not isinstance(names, list)
                or any(not isinstance(name, str) or not name.strip() for name in names)):
            raise AcceptanceFailure(f"{chain}: invalid {field} inventory")
    accepted = found["accepted_names"]
    if (type(found["accepted"]) is not int or not accepted
            or found["accepted"] != len(accepted) or len(accepted) != len(set(accepted))):
        raise AcceptanceFailure(f"{chain}: accepted theorem names and count differ")
    accepted = set(accepted)
    expected = accepted_roots(chain, required)
    if accepted != expected:
        raise AcceptanceFailure(f"{chain}: theorem inventory differs from Audit.acceptedFor: "
                                f"missing {sorted(expected - accepted)}, "
                                f"extra {sorted(accepted - expected)}")
    missing = required - accepted
    if missing:
        raise AcceptanceFailure(f"{chain}: missing public-entry theorem evidence: {sorted(missing)}")
    if found["entry_statements"] != public_entry_statements(chain):
        raise AcceptanceFailure(f"{chain}: exact caller proposition evidence differs")
    modules = found["modules"]
    if (type(found["reachable"]) is not int or found["reachable"] < len(accepted)
            or not isinstance(modules, dict) or not modules
            or any(not isinstance(name, str) or not name or type(count) is not int or count <= 0
                   for name, count in modules.items())
            or sum(modules.values()) != found["reachable"]):
        raise AcceptanceFailure(f"{chain}: incomplete reachable declaration counts")
    discharged = found["discharged"]
    if (not isinstance(discharged, dict)
            or any(not isinstance(name, str) or not name
                   or not isinstance(theorem, str) or theorem not in accepted
                   for name, theorem in discharged.items())):
        raise AcceptanceFailure(f"{chain}: invalid discharged theorem evidence")
    witnesses = found["witnesses"]
    assumptions = set(found["axioms"] + found["opaque"] + found["extern"])
    if not isinstance(witnesses, dict) or set(witnesses) != assumptions:
        raise AcceptanceFailure(f"{chain}: incomplete assumption witnesses")
    for name, path in witnesses.items():
        if (not isinstance(path, list) or not path or path[-1] != name
                or path[0] not in accepted
                or any(not isinstance(step, str) or not step for step in path)):
            raise AcceptanceFailure(f"{chain}: invalid witness path for {name}")
    return accepted


def stage_assumptions(context: Run):
    """Check both complete inventories against the trust and public-entry contracts."""

    trusted = json.loads((VERIFICATION / "trusted_base.json").read_text())
    patterns = {
        kind: [re.compile(entry["match"]) for entry in trusted.get(kind, [])]
        for kind in ("axioms", "opaque", "extern")
    }
    if set(context.inventories) != set(CHAINS) or set(context.obligations) != set(CHAINS):
        raise AcceptanceFailure("assumptions: expected inventories and obligations for exactly both chains")
    required = check_public_entries()
    results = {}
    for chain in CHAINS:
        found = json.loads(context.inventories[chain].read_text())
        check_closure_inventory(chain, found, required[chain])
        for kind, allowed in patterns.items():
            unexpected = [name for name in found[kind]
                          if not any(pattern.fullmatch(name) for pattern in allowed)]
            if unexpected:
                paths = {name: found["witnesses"].get(name) for name in unexpected[:3]}
                raise AcceptanceFailure(
                    f"{chain}: unjustified {kind} in the accepted closure: {unexpected[:10]}; "
                    f"witness paths: {paths}"
                )
        undischarged = [obligation for obligation in context.obligations[chain]
                        if obligation["proposition"] not in found["discharged"]]
        if undischarged:
            raise AcceptanceFailure(
                f"{chain}: {len(undischarged)} collection obligations have no accepted theorem "
                f"stating exactly their proposition: "
                f"{[(o['impl'], o['proposition']) for o in undischarged[:5]]}"
            )
        results[chain] = {
            "accepted": found["accepted"], "reachable": found["reachable"],
            "accepted_names": found["accepted_names"],
            "public_entries": len(required[chain]),
            "obligations": len(context.obligations[chain]),
            **{kind: len(found[kind]) for kind in patterns},
        }
    return results


STAGES = [
    ("environment", stage_environment),
    ("toolchain", stage_toolchain),
    ("native boundary", stage_native_boundary),
    ("native", stage_native),
    ("model matrix", stage_models),
    ("production AD", stage_production_ad),
    ("whole-core plan", stage_plan),
    ("whole-core", stage_whole_core),
    ("coverage", stage_coverage),
    ("correspondence", stage_correspondence),
    ("assumptions", stage_assumptions),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lacadile", type=Path,
                        help="Local LaCaDiLE checkout at the pinned revision (not publicly clonable)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Development only: accept modified tracked files (recorded in the report)")
    parser.add_argument("--static", action="store_true",
                        help="Run only the static native-boundary checks (no extraction, no report)")
    parser.add_argument("--aeneas-lean-dir", type=Path,
                        help="--static: the Aeneas backends/lean directory whose models are checked")
    parser.add_argument("--plan", action="append", default=[], metavar="CHAIN=PATH",
                        help="--static: also check this existing chunk plan (repeatable)")
    args = parser.parse_args()
    if args.static:
        if args.aeneas_lean_dir is None:
            parser.error("--static requires --aeneas-lean-dir")
        print(json.dumps(static_checks(args.aeneas_lean_dir.resolve()), indent=2))
        for argument in args.plan:
            chain, _, path = argument.partition("=")
            if not _ or not path:
                parser.error("--plan needs CHAIN=PATH")
            print(f"{chain}: {json.dumps(check_shell_plan(chain, json.loads(Path(path).read_text())))}")
        print("\nSTATIC CHECKS PASSED")
        return
    if args.lacadile is None:
        parser.error("--lacadile is required")
    context = Run(args)
    print(f"Acceptance run: {context.directory}", flush=True)
    for name, function in STAGES:
        context.stage(name, lambda function=function: function(context))
    label = "ACCEPTED (development run on a modified tree)" if context.report["dirty"] else "ACCEPTED"
    print(f"\n{label}: {context.directory / 'report.json'}")


if __name__ == "__main__":
    try:
        main()
    except (AcceptanceFailure, subprocess.CalledProcessError, OSError, KeyError, ValueError) as error:
        print(f"\nREJECTED: {error}", file=sys.stderr)
        raise SystemExit(1) from None
