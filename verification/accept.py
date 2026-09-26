"""Fail-closed acceptance for the Chelis semantic core.

Run from the repository root inside the pinned environment:

    nix develop --command python verification/accept.py --lacadile <LaCaDiLE checkout>

Every stage either succeeds or aborts the run with a nonzero status. There is no
skip, reuse or warning-only mode: a stage whose prerequisites are missing fails.
All artifacts of one run live in a fresh directory under .work/acceptance/, and
each stage checks that its outputs were produced by this run.

Stages, in order:

1. environment       pinned native compiler, Lean toolchain and LaCaDiLE revision;
                     clean tracked tree.
2. toolchain         Nix builds the patched Charon, charon-ml, Aeneas and Aeneas
                     source; `charon`/`aeneas` on PATH must be exactly those outputs.
                     The Aeneas Lean library is prepared from that source (content
                     checked file by file) and built.
3. native            `cargo fmt --check`, strict Clippy and the workspace tests.
4. model matrix      native versus generated Lean behavior of every library model,
                     float observations, rejected roots and collection contracts.
5. production AD     fresh `grad_dag_checked` extraction and Lean compilation.
6. whole-core plan   fresh call graphs and chunk plans (default and lowering-trace).
7. whole-core        separately compiled extraction of both chains and compilation
                     of the unmodified generated Lean.
8. coverage          independent re-validation of each chain's coverage report.
9. correspondence    the Lean bridge (ChelisBridge + Audit) built against each
                     chain's whole-core model; Audit fails on disallowed axioms.
                     verification/lean/ClosureAudit.lean then inventories the
                     transitive closure of every accepted theorem (axioms, opaque
                     constants, extern/implemented_by, witness paths).
10. assumptions      each inventory against the justified trusted base in
                     verification/trusted_base.json; any unmatched name rejects.
                     Every collection obligation Aeneas emitted for the chain
                     (e.g. the totality of a crate iterator's own size_hint) must be
                     discharged by an accepted theorem whose statement is exactly
                     the obligation's proposition.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
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
})
# Kinds of verification/native_shell.json entries. Primitives and shell types are the
# boundary the extracted core calls, so each must resolve to a Lean model; the other
# kinds must be unreachable from every extracted (non-shell) item.
SHELL_BOUNDARY = frozenset({"primitive", "shell_type"})
SHELL_INTERNAL = frozenset({"shell_state", "shell_entry", "shell_helper"})
# Kinds of proof obligations Aeneas emits with -emit-collection-obligations.
OBLIGATION_KINDS = frozenset({"iterator_size_hint_total"})


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
    return {chain: {"plan": str(path), "chunks": len(json.loads(path.read_text())["chunks"])}
            for chain, path in plans.items()}


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


def validate_coverage(chain: str, coverage: dict) -> dict:
    if coverage.get("unknown_externals") != 0:
        raise AcceptanceFailure(f"{chain}: unknown externals: {coverage.get('unknown_externals')}")
    if coverage.get("opaque_local_items") != 0:
        raise AcceptanceFailure(f"{chain}: opaque local items: {coverage.get('opaque_local_items')}")
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
        if entry.get("missing", 0) or entry.get("duplicates", 0):
            raise AcceptanceFailure(
                f"{chain}/{crate}: {entry.get('missing', 0)} missing and "
                f"{entry.get('duplicates', 0)} duplicated declarations"
            )
        accounted = entry.get("emitted", 0) + sum(unemitted.values())
        if entry.get("local_items", -1) != accounted or accounted == 0:
            raise AcceptanceFailure(
                f"{chain}/{crate}: {entry.get('local_items')} local items but {accounted} accounted for"
            )
        totals[crate] = {"local_items": entry["local_items"], "emitted": entry["emitted"], **unemitted}
    validate_shell(chain, coverage.get("native_shell", {}))
    return totals


def validate_shell(chain: str, shell: dict) -> None:
    """The shell report must list exactly verification/native_shell.json, bind every
    boundary item to a Lean model, and show no extracted item reaching shell internals."""
    declared = json.loads((VERIFICATION / "native_shell.json").read_text())["items"]
    reported = shell.get("items", {})
    if set(reported) != {item["pattern"] for item in declared}:
        raise AcceptanceFailure(f"{chain}: native shell report differs from native_shell.json")
    for item in declared:
        entry = reported[item["pattern"]]
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
        chain: validate_coverage(chain, json.loads((context.whole_core[chain] / "coverage.json").read_text()))
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
        log = context.directory / f"bridge-{chain}.log"
        begun = time.time()
        run([
            "lake", "build", "-R", "-K", f"model={model}", "-K", f"modelRoots={roots}",
            "ChelisBridge", "ChelisBridge.Audit",
        ], cwd=bridge, log=log)
        # The inventory must describe exactly this chain's build, so it is taken before
        # the next chain replaces the model.
        inventory = context.directory / f"assumptions-{chain}.json"
        run(["lake", "env", "lean", "--run", "ClosureAudit.lean", inventory],
            cwd=bridge, log=context.directory / f"assumptions-{chain}.log")
        context.inventories[chain] = fresh(inventory, begun)
        results[chain] = {"log": str(log), "inventory": str(inventory)}
    return results


def stage_assumptions(context: Run):
    """Validate each chain's inventory (written by the correspondence stage right after
    that chain's bridge build) against the justified trusted base."""
    import re

    trusted = json.loads((VERIFICATION / "trusted_base.json").read_text())
    patterns = {
        kind: [re.compile(entry["match"]) for entry in trusted.get(kind, [])]
        for kind in ("axioms", "opaque", "extern")
    }
    results = {}
    for chain in CHAINS:
        found = json.loads(context.inventories[chain].read_text())
        if found.get("accepted", 0) == 0:
            raise AcceptanceFailure(f"{chain}: the audit saw no accepted theorems")
        for kind, allowed in patterns.items():
            unexpected = [name for name in found.get(kind, [])
                          if not any(pattern.fullmatch(name) for pattern in allowed)]
            if unexpected:
                paths = {name: found.get("witnesses", {}).get(name) for name in unexpected[:3]}
                raise AcceptanceFailure(
                    f"{chain}: unjustified {kind} in the accepted closure: {unexpected[:10]}; "
                    f"witness paths: {paths}"
                )
        undischarged = [obligation for obligation in context.obligations[chain]
                        if obligation["proposition"] not in found.get("discharged", {})]
        if undischarged:
            raise AcceptanceFailure(
                f"{chain}: {len(undischarged)} collection obligations have no accepted theorem "
                f"stating exactly their proposition: "
                f"{[(o['impl'], o['proposition']) for o in undischarged[:5]]}"
            )
        results[chain] = {
            "accepted": found["accepted"], "reachable": found["reachable"],
            "obligations": len(context.obligations[chain]),
            **{kind: len(found.get(kind, [])) for kind in patterns},
        }
    return results


STAGES = [
    ("environment", stage_environment),
    ("toolchain", stage_toolchain),
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
    parser.add_argument("--lacadile", type=Path, required=True,
                        help="Local LaCaDiLE checkout at the pinned revision (not publicly clonable)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Development only: accept modified tracked files (recorded in the report)")
    args = parser.parse_args()
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
