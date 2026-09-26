"""Separately compiled whole-core extraction (fail closed).

Input is a chunk plan written by `plan_chunks.py`. Every chunk runs Charon on one
workspace package with an explicit root set (`--start-from-file`), the
items owned by earlier chunks made opaque (`--opaque-file`) and the types
declared in their bodies kept transparent (`--include`), then Aeneas with
canonical names, every earlier manifest as `-external`, and `-emit-manifest`.
All chunks write into one Lean source root, compiled unmodified by one Lake
package that requires the Aeneas Lean library, one module at a time (imports
first) so that one Lean process runs at a time under the RSS cap.

    python verification/extract_whole.py --plan PLAN --out FRESH_DIR \
        --charon BIN --aeneas BIN --aeneas-lean-dir BUILT_LIB [--diagnostic]

Accepting mode (default) exits nonzero on the first failure: a nonzero tool
status, a missing/stale/empty artifact (checked by mtime, size and sha256), a
source tree or Charon binary that differs from the plan, or a coverage defect.

`--diagnostic` never accepts: Charon runs without --abort-on-error, Aeneas
without -abort-on-error but with -diagnostic-manifest (a chunk with errors
still writes a manifest of its emitted declarations, marked incomplete), failing
chunks do not stop the run, and every error is recorded. A chunk whose
predecessor wrote no manifest, or an incomplete one, is extracted with the
manifests that exist; the resulting unknown externals are reported as
consequential on the failed owner chunk. Lean modules whose imports failed or
were not generated are not built ("blocked"). The run writes blockers.json and
blockers.md (per-chunk status and first errors, per-module Lean results, errors
classified by blocker class with a suggested owner, counts per class), and the
summary says DIAGNOSTIC.

The FCIS imperative shell listed in verification/native_shell.json is opaque in
every chunk and never extracted (see plan_chunks.py).

Outputs under --out:
  summary.json   {"mode": "accepting"|"diagnostic", "accepted": bool, "chain",
                  "plan_sha256", "binaries": {"charon"|"aeneas": sha256 of the
                  binary given}, "tools": {"charon"|"aeneas": {"path", "sha256",
                  "files": {path: sha256}, "version"}}, "chunks": [{"name",
                  "module", "crate", "charon": STEP, "aeneas": STEP,
                  "llbc": {"bytes", "sha256"}, "manifest": {"entries", "sha256",
                  "complete"} | null}], "declarations", "modules": [Lean module
                  names], "aeneas_diagnostic_options": [options],
                  "lake": {"status", "built", "failed", "blocked", "wall_seconds",
                  "peak_rss_bytes", "modules": {module: STEP | {"status":
                  "blocked", "missing_imports", "failed_imports"}}} | null}
                 STEP = {"status": "ok"|"failed"|"skipped", "returncode",
                  "wall_seconds", "peak_rss_bytes" (largest single process, from
                  wait4), "peak_tree_rss_bytes" (sampled sum over the process
                  group), "log"}
  coverage.json  schema "chelis-wholecore-coverage" version 2 (re-validated by
                 verification/accept.py; keep stable):
                 {"crates": {crate: {"local_items", "emitted", "unemitted":
                  {category: count}, "missing": count, "duplicates": count,
                  "chunks": [names]}}         local_items = emitted + unemitted
                                              when missing = 0,
                  "owners": {chunk: owned local item count},
                  "categories": {category: {"justification", "count",
                  "examples": [names]}}, categories: removed_by_passes,
                  synthetic, generic_template, model_provided, folded,
                  native_shell,
                  "missing_items": [{"crate", "name", "kind", "owner"}],
                  "duplicate_items": [{"key", "modules"}],
                  "unknown_externals": count, "unknown_external_items":
                  [{"chunk", "name"}], "opaque_local_items": count,
                  "opaque_local_item_list": [{"chunk", "name"}],
                  "chunks_without_manifest": [names],
                  "chunks_with_partial_manifest": [names] (diagnostic runs),
                  "native_shell": {"sha256", "items": {pattern: {"kind",
                  "crate", "lean_model": Lean declaration | null,
                  "core_dependents": [non-shell local items with a deps edge
                  into the item], "members": [local items]}}, "defects":
                  [boundary items without model, internal items with core
                  dependents]},
                  "pipeline_reachable": {crate: count},
                  "emitted_declarations", "ok": bool}
                 A local item is an item of the 8 workspace crates in the
                 planner's callgraph dumps (plus the items folded into them).
                 It is emitted when an Aeneas manifest entry has its key (kind
                 + canonical name).
  obligations.json {"version": 1, "obligations": [{"kind": "iterator_size_hint_total",
                 "impl", "proposition" (Lean name `<Impl>.size_hint_total` in the
                 owner chunk), "chunk"}]}: always written (possibly with an empty
                 list); every chunk runs Aeneas with -emit-collection-obligations.
  lean/          Lake package `chelis_core`, library `ChelisCore`, whose roots
                 are summary.json "modules".
"""

import argparse
import collections
import gzip
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import CHARON_OPTIONS, CHARON_STRICT, charon_includes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_CRATES = (
    "chelis_unord", "chelis_vocab", "chelis_deep", "chelis_pred",
    "chelis_types", "chelis_effects", "chelis_ir", "chelis_pipeline_core",
)
AENEAS_ACCEPTANCE = (
    "-backend", "lean", "-checks", "-abort-on-error", "-warnings-as-errors",
    "-strict-joins", "-sequential", "-no-progress-bar", "-canonical-names",
)
# Diagnostic runs only (Aeneas rejects it with -abort-on-error): the manifest is always
# written, and a run with errors (or a partial -external) writes the emitted
# declarations only, marked incomplete ({"version": 2, "complete": false}); only runs
# with the option may load such a manifest.
AENEAS_DIAGNOSTIC = ("-diagnostic-manifest",)
PLAN_FORMAT = "chelis-wholecore-plan"
COVERAGE_FORMAT = "chelis-wholecore-coverage"
NATIVE_SHELL = ROOT / "verification" / "native_shell.json"
SHELL_BOUNDARY = ("primitive", "shell_type")
SHELL_INTERNAL = ("shell_state", "shell_entry", "shell_helper")


class ExtractionFailure(RuntimeError):
    pass


def load_native_shell() -> list[dict]:
    """verification/native_shell.json: the FCIS imperative shell, never extracted.
    Every chunk (and every planner dump) makes these patterns opaque."""
    items = json.loads(NATIVE_SHELL.read_text())["items"]
    patterns = [item["pattern"] for item in items]
    if len(set(patterns)) != len(patterns):
        raise ExtractionFailure(f"duplicate patterns in {NATIVE_SHELL}")
    for item in items:
        if item["kind"] not in SHELL_BOUNDARY + SHELL_INTERNAL or not item.get("justification"):
            raise ExtractionFailure(f"{NATIVE_SHELL}: bad entry {item}")
    return items


# ---------------------------------------------------------------------------
# Hashing, freshness and tool identity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fresh(path: Path, started: float, what: str) -> dict:
    """Refuse a missing, empty or stale artifact; return its identity."""
    if not path.is_file():
        raise ExtractionFailure(f"{what} is missing: {path}")
    stat = path.stat()
    if stat.st_size == 0:
        raise ExtractionFailure(f"{what} is empty: {path}")
    if stat.st_mtime < started:
        raise ExtractionFailure(f"{what} is stale (older than its step): {path}")
    return {"path": str(path), "bytes": stat.st_size, "sha256": sha256_file(path)}


def tool_identity(binary: str, version_args: list[str]) -> dict:
    """sha256 of the tool and of every existing absolute file a wrapper names."""
    path = Path(shutil.which(binary) or binary).resolve()
    if not path.is_file():
        raise ExtractionFailure(f"tool not found: {binary}")
    files = {str(path): sha256_file(path)}
    head = path.read_bytes()[:2]
    if head == b"#!":
        for candidate in re.findall(r"(/[^\s\"'$:]+)", path.read_text(errors="replace")):
            candidate_path = Path(candidate)
            if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
                files[str(candidate_path)] = sha256_file(candidate_path)
    sibling = path.parent / "charon-driver"
    if path.name == "charon" and sibling.is_file():
        files[str(sibling)] = sha256_file(sibling)
    for name in list(files):
        driver = Path(name).parent / "charon-driver"
        if Path(name).name == "charon" and driver.is_file():
            files[str(driver)] = sha256_file(driver)
    version = subprocess.run(
        [str(path), *version_args], capture_output=True, text=True
    ).stdout.strip()
    combined = hashlib.sha256(
        "".join(f"{name}\0{digest}\n" for name, digest in sorted(files.items())).encode()
    ).hexdigest()
    return {"path": str(path), "sha256": files[str(path)], "files": files,
            "identity": combined, "version": version}


def source_tree_sha256() -> str:
    """Content hash of every tracked or untracked file that defines the core."""
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "--", "crates", "Cargo.toml",
         "Cargo.lock", "rust-toolchain.toml"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    digest = hashlib.sha256()
    for name in sorted(filter(None, listed)):
        path = ROOT / name
        if path.is_file():
            digest.update(name.encode() + b"\0" + sha256_file(path).encode() + b"\n")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Measured, memory-gated execution (one heavy process at a time)


def free_memory_percent() -> int | None:
    try:
        out = subprocess.run([shutil.which("memory_pressure") or "/usr/bin/memory_pressure"], capture_output=True, text=True).stdout
    except OSError:
        return None
    match = re.search(r"free percentage:\s*(\d+)%", out)
    return int(match.group(1)) if match else None


def wait_for_memory(min_free: int, max_wait: int) -> None:
    waited = 0
    while True:
        free = free_memory_percent()
        if free is None or free >= min_free:
            return
        if waited >= max_wait:
            raise ExtractionFailure(
                f"free memory stayed at {free}% < {min_free}% for {waited}s; not starting a heavy step")
        print(f"  waiting for memory: {free}% free < {min_free}%", flush=True)
        time.sleep(30)
        waited += 30


def process_group_rss(pgid: int) -> int:
    out = subprocess.run([shutil.which("ps") or "/bin/ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True).stdout
    total = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == str(pgid):
            total += int(parts[1]) * 1024
    return total


class Runner:
    def __init__(self, *, min_free: int, max_wait: int, rss_cap: int, nice: int,
                 wrapper: list[str] | None = None):
        self.min_free = min_free
        self.max_wait = max_wait
        self.rss_cap = rss_cap
        self.nice = nice
        # A slot limiter shared with other users of the machine (e.g. .work/bin/heavy):
        # it blocks until a heavy slot is free, then runs the command as its child.
        self.wrapper = list(wrapper or ())

    def run(self, command, *, log: Path, env=None, cwd=ROOT, heavy=True, slot=None) -> dict:
        """`heavy`: memory-gated and niced; `slot` (default: `heavy`): also run
        through the shared slot limiter."""
        command = [str(part) for part in command]
        if heavy:
            wait_for_memory(self.min_free, self.max_wait)
            command = [*(self.wrapper if slot or slot is None else ()), "nice", "-n", str(self.nice),
                       *command]
        print("+ " + shlex.join(command), flush=True)
        merged = os.environ.copy()
        merged.update({"RUSTC_WRAPPER": "", "CARGO_INCREMENTAL": "0"})
        merged.update(env or {})
        started = time.time()
        peak_tree = 0
        capped = False
        with open(log, "w") as handle:
            process = subprocess.Popen(
                command, cwd=cwd, env=merged, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    break
                peak_tree = max(peak_tree, process_group_rss(process.pid))
                if self.rss_cap and peak_tree > self.rss_cap and not capped:
                    capped = True
                    print(f"  RSS cap exceeded ({peak_tree >> 20} MB); killing", flush=True)
                    os.killpg(process.pid, 9)
                time.sleep(1)
            process.returncode = os.waitstatus_to_exitcode(status)
        if capped:
            with open(log, "a") as handle:
                handle.write(f"\n[extract_whole] killed: process group RSS exceeded {self.rss_cap} bytes\n")
        # Darwin reports ru_maxrss in bytes; it covers reaped descendants.
        peak = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
        result = {
            "status": "ok" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "wall_seconds": round(time.time() - started, 2),
            "peak_rss_bytes": peak,
            "peak_tree_rss_bytes": max(peak_tree, peak),
            "rss_cap_killed": capped,
            "log": str(log),
            "started": started,
        }
        print(f"  -> {result['status']} rc={process.returncode} {result['wall_seconds']}s "
              f"peak {peak >> 20} MB (tree {result['peak_tree_rss_bytes'] >> 20} MB)", flush=True)
        return result


# ---------------------------------------------------------------------------
# Charon


def charon_cargo(charon: str) -> tuple[str, str]:
    """Charon's own cargo and host triple: the target dir layout and hashes are
    that cargo's, so only it can clean a package there."""
    toolchain = Path(subprocess.run([charon, "toolchain-path"], capture_output=True, text=True,
                                    check=True).stdout.strip())
    out = subprocess.run([toolchain / "bin" / "rustc", "-vV"], capture_output=True, text=True,
                         check=True).stdout
    host = next(line.removeprefix("host: ") for line in out.splitlines() if line.startswith("host: "))
    return str(toolchain / "bin" / "cargo"), host


def prepare_target(target: Path, charon: dict) -> None:
    """A cargo target dir is only reused by the Charon build that populated it:
    dependency bodies come from its exported promoted MIR."""
    stamp = target / "wholecore-charon.json"
    expected = {"charon": charon["identity"]}
    if target.exists():
        current = json.loads(stamp.read_text()) if stamp.is_file() else None
        if current != expected:
            print(f"  target dir {target} was built by another Charon; removing it", flush=True)
            shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(expected))
    # Cargo only cleans a directory marked as its cache.
    (target / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172789f06886806bc55\n")


OBLIGATION_KINDS = ("iterator_size_hint_total",)


def read_obligations(path: Path, started: float, chunk: str) -> list[dict]:
    """A chunk's -emit-collection-obligations file: {"version": 1, "obligations":
    [{"kind", "impl", "proposition"}]} (fresh; known kinds only)."""
    fresh(path, started, "collection obligations")
    data = json.loads(path.read_text())
    if data.get("version") != 1 or not isinstance(data.get("obligations"), list):
        raise ExtractionFailure(f"malformed collection obligations: {path}")
    obligations = []
    for entry in data["obligations"]:
        if entry.get("kind") not in OBLIGATION_KINDS or not entry.get("impl") or not entry.get("proposition"):
            raise ExtractionFailure(f"malformed collection obligation in {path}: {entry}")
        obligations.append({"kind": entry["kind"], "impl": entry["impl"],
                            "proposition": entry["proposition"], "chunk": chunk})
    return obligations


def charon_command(charon: str, *, package: str, features: list[str], llbc: Path,
                   target: Path, strict: bool, shell_file: Path, roots: list[str] | None = None,
                   roots_file: Path | None = None, opaque_file: Path | None = None,
                   transparent: list[str] = ()) -> list[str]:
    command = [charon, "cargo", *CHARON_OPTIONS, "--mir=promoted"]
    if strict:
        command += CHARON_STRICT
    else:
        command += ["--error-on-warnings"]
    command += charon_includes(WORKSPACE_CRATES)
    # Types declared in the bodies of opaque items (plan_chunks.py: "transparent").
    for pattern in transparent:
        command += ["--include", pattern]
    for root in roots or ():
        command += ["--start-from", root]
    if roots_file:
        command += ["--start-from-file", str(roots_file)]
    if opaque_file:
        command += ["--opaque-file", str(opaque_file)]
    # The native shell (verification/native_shell.json) is never extracted, not even
    # as a monomorphization template.
    command += ["--opaque-hard-file", str(shell_file)]
    command += ["--dest-file", str(llbc), "--", "--package", package, "--lib", "--jobs", "2",
                "--locked", "--target-dir", str(target)]
    if features:
        command += ["--features", ",".join(features)]
    return command


def clean_package(runner: Runner, charon: str, package: str, target: Path, log: Path) -> None:
    """Remove the package's artifacts so that Charon translates it again."""
    cargo, host = charon_cargo(charon)
    result = runner.run([cargo, "clean", "--package", package, "--target", host,
                         "--target-dir", str(target)], log=log, heavy=False)
    if result["status"] != "ok":
        raise ExtractionFailure(f"cargo clean failed; log: {log}")


# ---------------------------------------------------------------------------
# Log parsing (diagnostic classification)

ERROR_HEADER = re.compile(r"^\[(Error|Warning)\]\s?(.*)$")
SOURCE_LINE = re.compile(r"^Source: '([^']*)', lines? (\S+)")
COMPILER_LINE = re.compile(r"^Compiler source: (\S+), line (\d+)")
CHARON_ERROR = re.compile(r"^(error|warning)(\[[A-Z0-9]+\])?: (.*)$")
CHARON_LOCATION = re.compile(r"^\s*--> (\S+)")
# `lean`: "F.lean:L:C: error: M"; Lake 5 (Lean 4.29): "error: F.lean:L:C: M".
LEAN_ERROR = re.compile(r"^(?:error: (\S+\.lean):(\d+):(\d+): (?!error: )|(?:error: )?(\S+\.lean):(\d+):(\d+): error: )(.*)$")
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Blocker classes in priority order: (class, suggested owner, regex over
# "stage | message | compiler source"). The owners are the agents assigned to a
# known toolchain blocker; the fallbacks at the end catch unassigned causes.
CLASSES = (
    ("consequential", "owner of the failed earlier chunk / earlier error in this chunk",
     r"CONSEQUENTIAL|Ignoring the body of .* because of previous error"),
    ("native shell", "Main (verification/native_shell.json) / AeneasModels3 (primitive models)",
     r"NATIVE_SHELL|ThreadLocalRef|thread.local|thread_local"),
    ("check_ptr E13", "CharonFix4", r"check_ptr|CollectionContracts\.ml.*Not_found|Not_found.*CollectionContracts"),
    ("collection contract", "CharonFix4", r"E_UNSUPPORTED_COLLECTION|CollectionContracts|collection contract"),
    ("nested inductive E12", "AeneasNestedTypes",
     r"nested inductive|non valid occurrence of the datatypes|invalid occurrence of recursive|"
     r"PureNestedCollections"),
    ("loop joins", "AeneasSC",
     r"Markers remain after ownership reduction|InterpJoin|InterpLoops|InterpReduceCollapse|"
     r"[Ll]oop join|fixed.?point|strict.joins"),
    ("nested borrows E11", "AeneasNested2",
     r"[Nn]ested (mutable )?borrow|end_abs|InterpBorrowsCore\.ml|InterpBorrows\.ml|"
     r"SymbolicToPureValues\.ml|reborrow of a reborrow"),
    ("separate compilation", "AeneasSC",
     r"MANIFEST_MISMATCH"),
    ("missing std model", "AeneasModels3",
     r"UNKNOWN_EXTERNAL|not listed in the builtin|external, unknown|[Mm]issing (builtin|model)|no model|"
     r"unknown type declaration with region parameters|Missing borrow analysis for the type|"
     r"analyzing the following type declarations group"),
    ("Charon translation", "CharonFix4", r"^charon \|"),
    ("Lean compile", "NEW (Main)", r"^lean \|"),
    ("Aeneas translation", "NEW (Main)", r"^aeneas \|"),
)


def classify(error: dict) -> tuple[str, str]:
    text = f"{error['stage']} | {error['message']} | {error.get('compiler') or ''}"
    for name, owner, pattern in CLASSES:
        if re.search(pattern, text):
            return name, owner
    return "unclassified", "NEW (Main)"


def mask(message: str) -> str:
    """Error message with item names, ids and numbers masked, for grouping."""
    message = re.sub(r"'[^']*'", "'_'", message)
    message = re.sub(r"`[^`]*`", "`_`", message)
    message = re.sub(r"@\w+\d+", "@_", message)
    message = re.sub(r"\b\d+\b", "N", message)
    message = re.sub(r"\b(?:[a-z_][a-z0-9_]*::)+[A-Za-z0-9_{}<>#&' ,:()\[\]]+", "PATH", message)
    return message[:240]


def parse_aeneas_log(log: Path) -> list[dict]:
    """Errors of an Aeneas log with their Rust source span, OCaml compiler source
    and, when Aeneas names it, the item whose translation failed."""
    errors: list[dict] = []
    current = None
    last_error = None
    text = ANSI.sub("", log.read_text(errors="replace"))
    for line in text.splitlines():
        header = ERROR_HEADER.match(line)
        if header:
            current = {"level": header.group(1).lower(), "message": header.group(2),
                       "source": None, "compiler": None, "item": None}
            errors.append(current)
            if current["level"] == "error":
                last_error = current
            continue
        failed = re.match(r"\[Warn \] Could not translate the (?:body of )?\S+ '(.+?) because", line)
        if failed and last_error is not None and last_error["item"] is None:
            last_error["item"] = failed.group(1)
        if current is None:
            continue
        source = SOURCE_LINE.match(line)
        if source:
            current["source"] = f"{source.group(1)}:{source.group(2)}"
            continue
        compiler = COMPILER_LINE.match(line)
        if compiler:
            current["compiler"] = f"{compiler.group(1)}:{compiler.group(2)}"
            current = None
            continue
        if line.startswith("Unique errors") or not line.strip():
            current = None
        elif current["source"] is None and current["compiler"] is None:
            current["message"] += " " + line.strip()
    # Uncaught exceptions end the run without an [Error] header: keep the
    # exception text that follows.
    for match in re.finditer(r"^(Fatal error: exception .*|Uncaught exception:?)(.*(?:\n.+){0,4})", text, re.M):
        message = " ".join((match.group(1) + " " + match.group(2)).split())
        errors.append({"level": "error", "message": message, "source": None, "compiler": "uncaught",
                       "item": None})
    return [error for error in errors if error["level"] == "error"]


def parse_charon_log(log: Path) -> list[dict]:
    errors: list[dict] = []
    current = None
    for line in ANSI.sub("", log.read_text(errors="replace")).splitlines():
        header = CHARON_ERROR.match(line)
        if header and header.group(1) == "error":
            if re.match(r"could not compile|aborting due to", header.group(3)):
                continue
            current = {"level": "error", "message": header.group(3), "source": None, "compiler": "charon"}
            errors.append(current)
            continue
        if current is not None:
            location = CHARON_LOCATION.match(line)
            if location and current["source"] is None:
                current["source"] = location.group(1)
                current = None
    for match in re.finditer(r"^(thread '.*' panicked at .*|.*panicked at .*)$",
                             ANSI.sub("", log.read_text(errors="replace")), re.M):
        errors.append({"level": "error", "message": match.group(1), "source": None, "compiler": "charon"})
    return errors


def parse_lean_log(log: Path, lean_root: Path) -> list[dict]:
    errors = []
    for line in ANSI.sub("", log.read_text(errors="replace")).splitlines():
        match = LEAN_ERROR.match(line.strip())
        if match:
            file, row = (match.group(1), match.group(2)) if match.group(1) else (match.group(4), match.group(5))
            path = Path(file) if Path(file).is_absolute() else lean_root / file  # Lake runs in the package
            try:
                module = str(path.resolve().relative_to(lean_root.resolve()).with_suffix("")).replace("/", ".")
            except ValueError:
                module = str(path)
            errors.append({"level": "error", "message": match.group(7), "module": module,
                           "source": f"{module}:{row}", "compiler": "lean"})
    return errors


def unknown_externals(module_dir: Path) -> list[str]:
    """Rust names of the external definitions no manifest or model provides:
    with -split-files Aeneas writes them to *External_Template.lean files."""
    names = []
    for template in sorted(module_dir.glob("*External_Template.lean")):
        for match in re.finditer(r"^/--?\s*\[(.*)\]:?\s*$", template.read_text(errors="replace"), re.M):
            names.append(match.group(1))
    return sorted(set(names))


def opaque_local_items(callgraph: Path, owned: list[dict]) -> list[str]:
    """Items this chunk owns that Charon did not translate transparently."""
    graph = json.loads(callgraph.read_text())
    # Impls of hidden marker traits never reach the LLBC (coverage: removed_by_passes).
    wanted = {(item["kind"], item["name"]) for item in owned if not item.get("removed_by_passes")}
    found = set()
    opaque = []
    for item in graph["items"]:
        key = (item["kind"], item["name"])
        if key in wanted:
            found.add(key)
            if item["opacity"] != "Transparent" and item["kind"] != "trait_decl":
                opaque.append(f"{item['kind']} {item['name']} ({item['opacity']})")
    opaque += [f"{kind} {name} (not translated)" for kind, name in sorted(wanted - found)]
    return sorted(set(opaque))


def compress(path: Path) -> None:
    """Keep a large diagnostic artifact, gzipped (disk is limited)."""
    if path.is_file():
        with open(path, "rb") as source, gzip.open(f"{path}.gz", "wb", compresslevel=6) as target:
            shutil.copyfileobj(source, target)
        path.unlink()


# ---------------------------------------------------------------------------
# Extraction


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan.get("format") != PLAN_FORMAT or plan.get("version") != 1:
        raise ExtractionFailure(f"not a whole-core plan: {path}")
    if "native_shell" not in plan:
        raise ExtractionFailure(f"plan without a native shell section: {path}")
    names = [chunk["name"] for chunk in plan["chunks"]]
    modules = [chunk["module"] for chunk in plan["chunks"]]
    if len(set(names)) != len(names) or len(set(modules)) != len(modules):
        raise ExtractionFailure("chunk names and modules must be unique")
    for chunk in plan["chunks"]:
        if not chunk["roots"]:
            raise ExtractionFailure(f"chunk {chunk['name']} has no roots")
    return plan


CHUNK_IDENTITY = ("name", "module", "package", "features", "roots", "opaque", "transparent", "externals")


def reusable_chunks(earlier: Path, plan: dict, charon: dict, aeneas: dict) -> dict[str, dict]:
    """Diagnostic runs only: the chunk records of an earlier run (e.g. the default
    chain) whose chunks are identical in this plan (e.g. the feature-independent
    crates of the lowering-trace chain), made by the same tools and sources."""
    summary = json.loads((earlier / "summary.json").read_text())
    if (summary["tools"]["charon"]["identity"] != charon["identity"]
            or summary["tools"]["aeneas"]["identity"] != aeneas["identity"]):
        raise ExtractionFailure("--reuse run was made by other tools")
    earlier_plan = load_plan(Path(summary["plan"]))
    for key in ("native_shell_sha256", "charon_identity"):
        if earlier_plan["inputs"].get(key) != plan["inputs"].get(key):
            raise ExtractionFailure(f"--reuse run differs in {key}")
    if earlier_plan["inputs"]["sources_sha256"] != plan["inputs"]["sources_sha256"]:
        print("  WARNING: --reuse run was planned from another source tree", flush=True)
    chunks = {chunk["name"]: chunk for chunk in earlier_plan["chunks"]}
    records = {record["name"]: record for record in summary["chunks"]}
    return {chunk["name"]: records[chunk["name"]] for chunk in plan["chunks"]
            if chunk["name"] in records and chunk["name"] in chunks
            and all(chunks[chunk["name"]].get(key) == chunk.get(key) for key in CHUNK_IDENTITY)}


def extract(plan: dict, out: Path, args, runner: Runner, charon: dict, results: list[dict],
            reuse: dict[str, dict]) -> None:
    lean = out / "lean"
    lean.mkdir(parents=True)
    target = Path(args.target_dir).resolve()
    prepare_target(target, charon)
    manifests: dict[str, Path] = {}
    # Chunks whose manifest is missing or partial ("missing"/"partial"): the workspace
    # items they own are consequential unknown externals in later chunks.
    degraded: dict[str, str] = {}
    owners: dict[str, list[str]] = {}
    for chunk in plan["chunks"]:
        for item in chunk["owned"] + chunk.get("adopted", []):
            for key in {normalize_pattern(key) for key in (item["name"], item["pattern"]) if key}:
                owners.setdefault(key, []).append(chunk["name"])
    for index, chunk in enumerate(plan["chunks"]):
        work = out / "chunks" / f"{index:02d}-{chunk['name']}"
        work.mkdir(parents=True)
        if chunk["name"] in reuse:
            record = dict(reuse[chunk["name"]], reused_from=str(args.reuse))
            results.append(record)
            print(f"== chunk {index}: {chunk['name']} reused from {args.reuse}", flush=True)
            if (args.reuse / "lean" / chunk["module"]).is_dir():
                shutil.copytree(args.reuse / "lean" / chunk["module"], lean / chunk["module"])
            if record.get("manifest"):
                manifest = work / "manifest.json"
                shutil.copy(record["manifest"]["path"], manifest)
                if sha256_file(manifest) != record["manifest"]["sha256"]:
                    raise ExtractionFailure(f"reused manifest of {chunk['name']} changed")
                record["manifest"] = dict(record["manifest"], path=str(manifest))
                manifests[chunk["name"]] = manifest
                if not record["manifest"].get("complete", True):
                    degraded[chunk["name"]] = "partial"
            else:
                degraded[chunk["name"]] = "missing"
            continue
        record = {"name": chunk["name"], "module": chunk["module"], "crate": chunk["crate"],
                  "package": chunk["package"], "features": chunk["features"],
                  "roots": len(chunk["roots"]), "opaque": len(chunk["opaque"]),
                  "charon": None, "aeneas": None, "llbc": None, "manifest": None,
                  "errors": [], "missing_externals": [], "partial_externals": []}
        results.append(record)
        degraded[chunk["name"]] = "missing"  # until its manifest is read
        print(f"== chunk {index}: {chunk['name']} ({chunk['package']}, {len(chunk['roots'])} roots, "
              f"{len(chunk['opaque'])} opaque)", flush=True)
        roots = work / "roots.txt"
        roots.write_text("".join(f"{pattern}\n" for pattern in chunk["roots"]))
        # Items owned by earlier chunks; the native shell (never extracted).
        opaque = work / "opaque.txt"
        opaque.write_text("".join(f"{pattern}\n" for pattern in chunk["opaque"]))
        shell = work / "native-shell.opaque"
        shell.write_text("".join(f"{pattern}\n" for pattern in sorted(plan["native_shell"])))
        llbc = work / f"{chunk['module']}.llbc"
        clean_package(runner, args.charon, chunk["package"], target, work / "clean.log")
        charon_env = {"CHARON_COVERAGE_REPORT": str(work / "charon-coverage.json"),
                      "CHARON_CALLGRAPH": str(work / "callgraph.json")}
        step = runner.run(charon_command(
            args.charon, package=chunk["package"], features=chunk["features"], llbc=llbc,
            target=target, strict=not args.diagnostic, shell_file=shell, roots_file=roots,
            opaque_file=opaque if chunk["opaque"] else None, transparent=chunk.get("transparent", []),
        ), log=work / "charon.log", env=charon_env)
        record["charon"] = step
        record["errors"] += [dict(error, stage="charon") for error in parse_charon_log(work / "charon.log")]
        if step["status"] != "ok" and not args.diagnostic:
            raise ExtractionFailure(f"Charon failed on chunk {chunk['name']}; log: {step['log']}")
        try:
            record["llbc"] = fresh(llbc, step["started"], "LLBC")
        except ExtractionFailure as error:
            if not args.diagnostic:
                raise
            record["errors"].append({"stage": "charon", "level": "error", "message": str(error),
                                     "source": None, "compiler": "extract_whole"})
            record["aeneas"] = {"status": "skipped"}
            continue
        # Every root must select exactly its own item, transparently.
        try:
            report = fresh(work / "charon-coverage.json", step["started"], "Charon coverage report")
            selection = json.loads(Path(report["path"]).read_text()).get("start_from_selected", [])
            record["root_selection_defects"] = selection_defects(selection)
            fresh(work / "callgraph.json", step["started"], "Charon callgraph")
            record["opaque_local"] = opaque_local_items(work / "callgraph.json", chunk["owned"])
        except (ExtractionFailure, ValueError, KeyError) as error:
            if not args.diagnostic:
                raise
            record["errors"].append({"stage": "charon", "level": "error", "message": str(error),
                                     "source": None, "compiler": "extract_whole"})
        for defect in record.get("root_selection_defects", []):
            record["errors"].append({"stage": "charon", "level": "error", "source": None,
                                     "compiler": "start_from_selected",
                                     "message": f"root pattern selection defect: {defect}"})
        for name in record.get("opaque_local", []):
            record["errors"].append({"stage": "charon", "level": "error", "source": None,
                                     "compiler": "callgraph",
                                     "message": f"owned item not translated transparently: {name}"})
        compress(work / "callgraph.json")
        if (record.get("root_selection_defects") or record.get("opaque_local")) and not args.diagnostic:
            raise ExtractionFailure(f"chunk {chunk['name']}: root selection defects or opaque owned "
                                    f"items; see {out / 'summary.json'}")

        if args.clear_start_from:
            # Diagnostic workaround: Aeneas re-parses the recorded --start-from patterns with
            # charon-ml's name-pattern parser, which rejects Charon's `{impl ...}` elements
            # ("Cannot parse Charon's --start-from pattern"). With none recorded, Aeneas takes
            # every local item of the LLBC as a root: the chunk's own items plus opaque
            # signatures (earlier chunks' items come from their manifests).
            data = json.loads(llbc.read_text())
            record["cleared_start_from"] = len(data["translated"]["options"]["start_from"])
            data["translated"]["options"]["start_from"] = []
            llbc.write_text(json.dumps(data, separators=(",", ":")))
            del data
        manifest = work / "manifest.json"
        command = [args.aeneas, *AENEAS_ACCEPTANCE]
        if args.diagnostic:
            command.remove("-abort-on-error")
            command += AENEAS_DIAGNOSTIC
        command += ["-split-files", "-subdir", chunk["module"]]
        if args.funs_chunk_bytes:
            command += ["-funs-chunk-bytes", str(args.funs_chunk_bytes)]
        for earlier in chunk["externals"]:
            if earlier in manifests:
                command += ["-external", str(manifests[earlier])]
                if degraded.get(earlier) == "partial":
                    record["partial_externals"].append(earlier)
            elif args.diagnostic:
                record["missing_externals"].append(earlier)
            else:
                raise ExtractionFailure(f"chunk {chunk['name']}: no manifest from {earlier}")
        obligations = work / "obligations.json"
        command += ["-emit-collection-obligations", str(obligations)]
        command += ["-dest", str(lean), "-emit-manifest", str(manifest), str(llbc)]
        # Aeneas is memory-heavy only on large LLBC (> 20 MB).
        step = runner.run(command, log=work / "aeneas.log", slot=llbc.stat().st_size > 20 << 20)
        record["aeneas"] = step
        errors = parse_aeneas_log(work / "aeneas.log")
        record["errors"] += [dict(error, stage="aeneas") for error in errors]
        record["unknown_externals"] = unknown_externals(lean / chunk["module"])
        shell_names = {normalize_pattern(pattern) for pattern in plan["native_shell"]}
        for name in record["unknown_externals"]:
            # The earlier chunks owning an item of that name (names are not unique).
            candidates = [earlier for earlier in owners.get(normalize_pattern(name), ())
                          if earlier in chunk["externals"]]
            owner = next((earlier for earlier in candidates if earlier in degraded),
                         candidates[0] if candidates else None)
            if normalize_pattern(name) in shell_names:
                tag = "NATIVE_SHELL boundary item without a Lean model"
            elif owner in degraded:
                tag = f"CONSEQUENTIAL workspace item of {owner} ({degraded[owner]} manifest)"
            elif owner is not None:
                tag = f"MANIFEST_MISMATCH workspace item not found in the complete manifest of {owner}"
            elif name.split("::", 1)[0] in WORKSPACE_CRATES and (record["missing_externals"]
                                                                or record["partial_externals"]):
                tag = "CONSEQUENTIAL workspace item (owner unresolved) of a chunk without complete manifest"
            else:
                tag = "UNKNOWN_EXTERNAL definition without model or manifest"
            record["errors"].append({"stage": "aeneas", "level": "error", "message": f"{tag}: {name}",
                                     "source": None, "compiler": "External_Template"})
        compress(llbc)
        if step["status"] != "ok" and not args.diagnostic:
            raise ExtractionFailure(f"Aeneas failed on chunk {chunk['name']}; log: {step['log']}")
        if record["unknown_externals"] and not args.diagnostic:
            raise ExtractionFailure(f"chunk {chunk['name']}: unknown externals "
                                    f"{record['unknown_externals'][:5]}")
        try:
            identity = fresh(manifest, step["started"], "Aeneas manifest")
        except ExtractionFailure:
            if not args.diagnostic:
                raise
            record["errors"].append({"stage": "aeneas", "level": "error",
                                     "message": "no manifest (Aeneas reported errors); later chunks "
                                                "see this chunk's items as unknown externals",
                                     "source": None, "compiler": "extract_whole"})
            continue
        data = json.loads(manifest.read_text())
        # -diagnostic-manifest: a run with errors (or with a partial -external) writes a
        # manifest of the emitted declarations only, marked {"version": 2, "complete": false}.
        complete = data.get("version") == 1 and data.get("complete", True) is True
        if not complete and not args.diagnostic:
            raise ExtractionFailure(f"chunk {chunk['name']}: incomplete manifest {manifest}")
        record["manifest"] = dict(identity, entries=len(data["entries"]), complete=complete)
        manifests[chunk["name"]] = manifest
        if complete:
            degraded.pop(chunk["name"])
        else:
            degraded[chunk["name"]] = "partial"
        try:
            record["obligations"] = read_obligations(obligations, step["started"], chunk["name"])
        except (ExtractionFailure, ValueError) as error:
            if not args.diagnostic:
                raise
            record["errors"].append({"stage": "aeneas", "level": "error", "message": str(error),
                                     "source": None, "compiler": "extract_whole"})


def selection_defects(selection: list[dict]) -> list:
    """Roots whose pattern selected no item, several items, or an item it does not
    spell out. CHARON_COVERAGE_REPORT "start_from_selected" is
    [{"pattern", "items": [{"id", "name", "pattern", "pattern_spells_name"}]}]."""
    defects = []
    for entry in selection:
        items = entry["items"]
        if len(items) != 1:
            defects.append({"pattern": entry["pattern"], "selected": [item["name"] for item in items]})
        elif not items[0]["pattern_spells_name"] or items[0]["pattern"] != entry["pattern"]:
            defects.append({"pattern": entry["pattern"], "selected": items[0]["name"],
                            "canonical": items[0]["pattern"]})
    return defects


# ---------------------------------------------------------------------------
# Lean


def generated_modules(lean: Path) -> list[str]:
    return sorted(
        str(path.relative_to(lean).with_suffix("")).replace("/", ".")
        for path in lean.rglob("*.lean")
        if ".lake" not in path.parts and path.name != "lakefile.lean"
        and not path.stem.endswith("_Template")
    )


def write_lake_package(lean: Path, aeneas_lean: Path, modules: list[str]) -> None:
    (lean / "lakefile.lean").write_text(
        "import Lake\nopen Lake DSL\n\n"
        "package chelis_core where\n\n"
        f"require aeneas from \"{aeneas_lean}\"\n\n"
        "@[default_target]\nlean_lib ChelisCore where\n  srcDir := \".\"\n"
        f"  roots := #[{', '.join('`' + name for name in modules)}]\n"
    )
    shutil.copy(aeneas_lean / "lean-toolchain", lean / "lean-toolchain")
    manifest = aeneas_lean / "lake-manifest.json"
    if manifest.is_file():
        # Resolve the library's own dependencies from its package directory.
        (lean / ".lake").mkdir(exist_ok=True)
        packages = aeneas_lean / ".lake" / "packages"
        if packages.is_dir():
            (lean / ".lake" / "packages").mkdir(exist_ok=True)
            for package in sorted(packages.iterdir()):
                link = lean / ".lake" / "packages" / package.name
                if not link.exists():
                    link.symlink_to(package.resolve())


LEAN_IMPORT = re.compile(r"^import\s+(\S+)", re.M)


def lean_build(lean: Path, runner: Runner, diagnostic: bool, modules: list[str],
               chunk_modules: set[str]) -> tuple[dict, list[dict]]:
    """Build the generated modules one at a time, imports first (`lake build +M`),
    so that one Lean process runs at a time under the RSS cap (a package-wide
    `lake build` elaborates modules in parallel). A module importing a module that
    failed, or that was not generated (the FunsExternal of a chunk with unknown
    externals), is not built: "blocked" (diagnostic runs only; accepting runs stop
    at the first failure). Returns {"status", "built", "failed", "blocked",
    "modules": {module: STEP | {"status": "blocked", "missing_imports",
    "failed_imports"}}} and the Lean errors."""
    generated = set(modules)
    imports = {}
    for module in modules:
        text = (lean / Path(*module.split("."))).with_suffix(".lean").read_text(errors="replace")
        imports[module] = [name for name in LEAN_IMPORT.findall(text) if name.split(".")[0] in chunk_modules]
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(module: str) -> None:
        if state.get(module) == 2:
            return
        if state.get(module) == 1:
            raise ExtractionFailure(f"generated Lean modules import each other cyclically: {module}")
        state[module] = 1
        for name in imports[module]:
            if name in generated:
                visit(name)
        state[module] = 2
        order.append(module)

    for module in modules:
        visit(module)
    logs = lean.parent / "lake"
    logs.mkdir(exist_ok=True)
    steps: dict[str, dict] = {}
    errors: list[dict] = []
    for module in order:
        missing = [name for name in imports[module] if name not in generated]
        failed = [name for name in imports[module] if name in steps and steps[name]["status"] != "ok"]
        if missing or failed:
            if not diagnostic:
                raise ExtractionFailure(f"Lean module {module}: imports missing {missing} or failed {failed}")
            steps[module] = {"status": "blocked", "missing_imports": missing, "failed_imports": failed}
            errors.append({"level": "error", "module": module, "source": None, "compiler": "lake",
                           "message": f"CONSEQUENTIAL: not built; imports not generated {missing}, "
                                      f"imports failed {failed}"})
            continue
        log = logs / f"{module}.log"
        # Memory-gated and capped always; the shared slot only for large modules (> 256 KB).
        size = (lean / Path(*module.split("."))).with_suffix(".lean").stat().st_size
        step = runner.run(["lake", "build", f"+{module}"], cwd=lean, log=log, env={"LAKE_NO_CACHE": "1"},
                          slot=size > 256 << 10)
        steps[module] = step
        found = parse_lean_log(log, lean)
        if step["status"] != "ok":
            if not diagnostic:
                raise ExtractionFailure(f"lake build of {module} failed; log: {log}")
            if not any(error["module"] == module for error in found):
                reason = ("killed at the RSS cap" if step["rss_cap_killed"]
                          else f"exit status {step['returncode']}")
                found.append({"level": "error", "module": module, "source": None, "compiler": "lake",
                              "message": f"lake build failed without a Lean error ({reason}); log: {log}"})
        errors += found
    counts = collections.Counter(step["status"] for step in steps.values())
    return {"status": "ok" if counts["ok"] == len(order) else "failed", "built": counts["ok"],
            "failed": counts["failed"], "blocked": counts["blocked"],
            "wall_seconds": round(sum(step.get("wall_seconds", 0) for step in steps.values()), 2),
            "peak_rss_bytes": max((step.get("peak_tree_rss_bytes", 0) for step in steps.values()), default=0),
            "modules": {module: steps[module] for module in order}}, errors


# ---------------------------------------------------------------------------
# Coverage


def normalize_name(name: str) -> str:
    """Charon's pretty name and Aeneas' key print the same path with different
    region and tuple spellings; compare them modulo those."""
    name = re.sub(r"'\w+\b|'_\d*", "", name)
    name = re.sub(r"&\s*(mut\s*)?", lambda m: "&mut " if m.group(1) else "&", name)
    name = re.sub(r"<\s*>", "", name)
    name = re.sub(r"\s+", "", name)
    name = re.sub(r",(?=[>)\]])", "", name)
    return name


def normalize_pattern(pattern: str) -> str:
    """A name pattern modulo the spellings in which Charon's canonical patterns,
    Aeneas' library model patterns and external names differ: `{impl T}` vs
    `{T}`, generic arguments and blanks."""
    pattern = pattern.replace("{impl ", "{")
    previous = None
    while previous != pattern:
        previous = pattern
        pattern = re.sub(r"<[^<>]*>", "", pattern)
    return re.sub(r"\s+", "", pattern)


MODEL_ATTRIBUTE = re.compile(r'@\[rust_(?:fun|type|global|trait|trait_impl)\s+"([^"]+)"[^\]]*\]')
LEAN_DECLARATION = re.compile(
    r"^\s*(?:@\[.*\]\s*)?(?:(?:noncomputable|private|protected|partial|unsafe)\s+)*"
    r"(?:def|abbrev|structure|inductive|opaque|axiom|class|instance|theorem)\s+([^\s:({\[]+)")


def lean_models(aeneas_lean: Path) -> dict[str, str]:
    """Normalized Rust pattern -> Lean declaration of every rust_fun/rust_type/
    rust_global/rust_trait/rust_trait_impl model in the Aeneas Lean library."""
    models: dict[str, str] = {}
    for path in sorted((aeneas_lean / "Aeneas").rglob("*.lean")):
        namespaces: list[str] = []
        pending: list[str] = []
        for line in path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            opened = re.match(r"namespace\s+(\S+)", stripped)
            if opened:
                namespaces.append(opened.group(1))
                continue
            closed = re.match(r"end\s+(\S+)", stripped)
            if closed and namespaces and namespaces[-1] == closed.group(1):
                namespaces.pop()
                continue
            pending += [match.group(1) for match in MODEL_ATTRIBUTE.finditer(line)]
            declaration = LEAN_DECLARATION.match(line) if pending else None
            if declaration:
                name = ".".join(namespaces + [declaration.group(1)])
                for pattern in pending:
                    models.setdefault(normalize_pattern(pattern), name)
                pending = []
    return models


UNEMITTED_CATEGORIES = {
    "removed_by_passes": "impl of a hidden marker trait (e.g. Destruct); Charon's passes remove it",
    "synthetic": "synthetic item without a Rust definition (e.g. UNIT_METADATA)",
    "generic_template": "generic item specialized by --monomorphize-mut; its instances are emitted",
    "model_provided": "item bound to an Aeneas Lean library model instead of a definition",
    "folded": "constructor, global initializer or promoted constant folded into its owner item",
    "native_shell": "FCIS imperative shell (verification/native_shell.json): opaque in every "
                    "chunk, never extracted",
}


def coverage(plan: dict, results: list[dict], models: dict[str, str]) -> dict:
    emitted: dict[str, list[str]] = {}
    emitted_raw: dict[str, str] = {}
    for record in results:
        if not record.get("manifest"):
            continue
        for entry in json.loads(Path(record["manifest"]["path"]).read_text())["entries"]:
            kind, _, name = entry["key"].partition(" ")
            key = f"{kind} {normalize_name(name)}"
            emitted.setdefault(key, []).append(record["module"])
            emitted_raw[key] = entry["key"]

    crates: dict[str, dict] = {}
    for chunk in plan["chunks"]:
        crates.setdefault(chunk["crate"], {"local_items": 0, "emitted": 0, "unemitted": {},
                                           "missing": 0, "duplicates": 0, "chunks": []})
    categories = {name: {"justification": text, "count": 0, "examples": []}
                  for name, text in UNEMITTED_CATEGORIES.items()}

    def account(crate: str, category: str, example: str, count: int = 1) -> None:
        unemitted = crates[crate]["unemitted"]
        unemitted[category] = unemitted.get(category, 0) + count
        categories[category]["count"] += count
        if len(categories[category]["examples"]) < 5:
            categories[category]["examples"].append(example)

    missing = []
    owners: dict[str, int] = {}
    for chunk in plan["chunks"]:
        stats = crates[chunk["crate"]]
        stats["chunks"].append(chunk["name"])
        owners[chunk["name"]] = len(chunk["owned"])
        # Adopted items belong to an earlier crate but only this chunk translates them.
        for crate, item in ([(chunk["crate"], item) for item in chunk["owned"]]
                            + [(item["crate"], item) for item in chunk.get("adopted", [])]):
            stats = crates.setdefault(crate, {"local_items": 0, "emitted": 0, "unemitted": {},
                                              "missing": 0, "duplicates": 0, "chunks": []})
            stats["local_items"] += 1
            key = f"{item['kind']} {normalize_name(item['name'])}"
            if key in emitted:
                stats["emitted"] += 1
                continue
            category = None
            if item.get("removed_by_passes"):
                category = "removed_by_passes"
            elif item.get("source") == "synthetic":
                category = "synthetic"
            elif item.get("pattern") and normalize_pattern(item["pattern"]) in models:
                category = "model_provided"
            elif item.get("generic") and any(
                    other.startswith(key + "::<") or other.startswith(key + "<") for other in emitted):
                category = "generic_template"
            if category:
                account(crate, category, item["name"])
            else:
                stats["missing"] += 1
                missing.append({"crate": crate, "name": item["name"],
                                "kind": item["kind"], "owner": chunk["name"]})
        stats = crates[chunk["crate"]]
        if chunk.get("folded"):
            # Folded items are rustc definitions without a Charon item of their own.
            stats["local_items"] += chunk["folded"]
            account(chunk["crate"], "folded", f"{chunk['folded']} folded into items of {chunk['name']}",
                    chunk["folded"])

    shell: dict[str, dict] = {}
    for pattern, entry in sorted(plan["native_shell"].items()):
        for member in entry["members"]:
            crates[entry["crate"]]["local_items"] += 1
            account(entry["crate"], "native_shell", member["name"])
        shell[pattern] = {
            "kind": entry["kind"], "crate": entry["crate"],
            "lean_model": models.get(normalize_pattern(pattern)) if entry["kind"] in SHELL_BOUNDARY else None,
            "core_dependents": entry["core_dependents"],
            "members": [member["name"] for member in entry["members"]],
        }
    shell_defects = sorted(
        pattern for pattern, entry in shell.items()
        if (entry["kind"] in SHELL_BOUNDARY and not entry["lean_model"])
        or (entry["kind"] in SHELL_INTERNAL and entry["core_dependents"]))

    duplicates = []
    for key, modules in sorted(emitted.items()):
        if len(modules) > 1:
            crate = key.partition(" ")[2].split("::", 1)[0]
            if crate in crates:
                crates[crate]["duplicates"] += 1
            duplicates.append({"key": emitted_raw[key], "modules": modules})
    unknown = [{"chunk": record["name"], "name": name}
               for record in results for name in record.get("unknown_externals", [])]
    opaque_local = [{"chunk": record["name"], "name": name}
                    for record in results for name in record.get("opaque_local", [])]
    without_manifest = [record["name"] for record in results if not record.get("manifest")]
    without_manifest += [chunk["name"] for chunk in plan["chunks"][len(results):]]
    partial = [record["name"] for record in results
               if record.get("manifest") and not record["manifest"].get("complete", True)]
    report = {
        "format": COVERAGE_FORMAT, "version": 2, "chain": plan["chain"],
        "crates": crates, "owners": owners, "categories": categories,
        "missing_items": missing, "duplicate_items": duplicates,
        "unknown_externals": len(unknown), "unknown_external_items": unknown,
        "opaque_local_items": len(opaque_local), "opaque_local_item_list": opaque_local,
        "chunks_without_manifest": without_manifest, "chunks_with_partial_manifest": partial,
        "native_shell": {"sha256": plan["inputs"].get("native_shell_sha256"), "items": shell,
                         "defects": shell_defects},
        "pipeline_reachable": plan.get("pipeline_reachable", {}),
        "emitted_declarations": len(emitted),
    }
    report["ok"] = not (missing or duplicates or unknown or opaque_local or without_manifest or partial
                        or shell_defects)
    return report


def write_coverage_summary(report: dict, path: Path) -> None:
    lines = [f"# Whole-core coverage ({report['chain']} chain)", "",
             f"coverage ok: {report['ok']}", "",
             "| crate | local items | emitted | unemitted (justified) | missing | duplicates |",
             "|---|---:|---:|---|---:|---:|"]
    for crate, stats in report["crates"].items():
        unemitted = ", ".join(f"{name} {count}" for name, count in sorted(stats["unemitted"].items()))
        lines.append(f"| {crate} | {stats['local_items']} | {stats['emitted']} | {unemitted or '-'} | "
                     f"{stats['missing']} | {stats['duplicates']} |")
    lines += ["", "## Unemitted categories", ""]
    for name, category in report["categories"].items():
        lines.append(f"- {name} ({category['count']}): {category['justification']}; e.g. "
                     + ", ".join(f"`{example}`" for example in category["examples"][:3]))
    lines += ["", "## Native shell", ""]
    for pattern, entry in report["native_shell"]["items"].items():
        lines.append(f"- `{pattern}` ({entry['kind']}): Lean model {entry['lean_model'] or '-'}, "
                     f"{len(entry['members'])} local items, core dependents "
                     f"{len(entry['core_dependents'])}"
                     + (f" (e.g. `{entry['core_dependents'][0]}`)" if entry["core_dependents"] else ""))
    lines += ["", "## Chunk owners", ""]
    lines += [f"- {chunk}: {count} local items" for chunk, count in report["owners"].items()]
    lines += ["", f"duplicates: {len(report['duplicate_items'])}, unknown externals: "
              f"{report['unknown_externals']}, opaque local items: {report['opaque_local_items']}, "
              f"missing: {len(report['missing_items'])}, chunks without manifest: "
              f"{len(report['chunks_without_manifest'])}, partial manifests: "
              f"{len(report['chunks_with_partial_manifest'])}, shell defects: "
              f"{len(report['native_shell']['defects'])}", "",
              "## Pipeline-rooted reachability (items of each crate reachable from "
              "chelis_pipeline_core's local items)", ""]
    lines += [f"- {crate}: {count}" for crate, count in report["pipeline_reachable"].items()]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Diagnostic blocker report


def blockers(results: list[dict], lean_errors: list[dict], plan: dict) -> dict:
    """Every error of the run, classified: groups by (class, stage, masked
    message), the first errors of each chunk, and counts per class."""
    module_to_chunk = {chunk["module"]: chunk["name"] for chunk in plan["chunks"]}
    groups: dict[tuple, dict] = {}
    classes: dict[str, dict] = {}
    first: dict[str, list] = {record["name"]: [] for record in results}

    def add(chunk: str, error: dict) -> None:
        name, owner = classify(error)
        pattern = mask(error["message"])
        key = (name, error["stage"], pattern)
        group = groups.setdefault(key, {"class": name, "owner": owner, "stage": error["stage"],
                                        "compiler": error.get("compiler"), "pattern": pattern,
                                        "count": 0, "chunks": {}, "examples": []})
        group["count"] += 1
        group["chunks"][chunk] = group["chunks"].get(chunk, 0) + 1
        example = {"chunk": chunk, "message": error["message"][:600], "source": error.get("source"),
                   "item": error.get("item"), "compiler": error.get("compiler"), "class": name,
                   "owner": owner}
        if len(group["examples"]) < 6:
            group["examples"].append(example)
        totals = classes.setdefault(name, {"owner": owner, "count": 0, "chunks": set()})
        totals["count"] += 1
        totals["chunks"].add(chunk)
        if chunk in first and len(first[chunk]) < 3 and name != "consequential":
            first[chunk].append(example)

    for record in results:
        # A chunk's own errors first: its first errors are the ones to fix.
        for error in record["errors"]:
            add(record["name"], error)
        for missing in record.get("missing_externals", []):
            add(record["name"], {"stage": "aeneas", "compiler": "extract_whole", "source": None,
                                 "message": f"CONSEQUENTIAL: no manifest from {missing}; its items are "
                                            f"unknown externals here"})
    for error in lean_errors:
        chunk = module_to_chunk.get(error["module"].split(".")[0], error["module"])
        add(chunk, dict(error, stage="lean"))
    ordered = sorted(groups.values(), key=lambda group: (group["class"], -group["count"], group["pattern"]))
    counts = {name: {"owner": totals["owner"], "count": totals["count"], "chunks": len(totals["chunks"])}
              for name, totals in sorted(classes.items(), key=lambda entry: -entry[1]["count"])}
    return {"classes": counts, "first_errors": first, "groups": ordered}


def fmt_step(step) -> str:
    if not step:
        return "not run"
    if step.get("status") == "skipped":
        return "skipped"
    return (f"{step['status']} (rc {step['returncode']}, {step['wall_seconds']} s, "
            f"peak {step['peak_rss_bytes'] >> 20} MB)")


def write_blockers(report: dict, results: list[dict], lake: dict | None, lean_errors: list[dict],
                   coverage_report: dict, path: Path) -> None:
    lines = ["# Whole-core blockers (DIAGNOSTIC run, not an acceptance)", "",
             "## Chunks", "",
             "| chunk | Charon | Aeneas | manifest entries | errors | first error (class; source) |",
             "|---|---|---|---:|---:|---|"]
    for record in results:
        manifest = record["manifest"]["entries"] if record.get("manifest") else "none"
        if record.get("manifest") and not record["manifest"].get("complete", True):
            manifest = f"{manifest} (partial)"
        firsts = report["first_errors"].get(record["name"], [])
        first = (f"{firsts[0]['class']}; `{firsts[0]['source']}`: {firsts[0]['message'][:160]}"
                 if firsts else "-").replace("|", "\\|")
        lines.append(f"| {record['name']} | {fmt_step(record['charon'])} | {fmt_step(record['aeneas'])} "
                     f"| {manifest} | {len(record['errors']) + len(record.get('missing_externals', []))} "
                     f"| {first} |")
    if lake and "modules" in lake:
        lines += ["", f"Lake build (one module at a time): {lake['built']} built, {lake['failed']} failed, "
                      f"{lake['blocked']} blocked (import failed or not generated); "
                      f"{lake['wall_seconds']} s, peak {lake['peak_rss_bytes'] >> 20} MB", "",
                  "| module | status | wall s | peak MB | first error / blocked by |", "|---|---|---:|---:|---|"]
        first_lean: dict[str, dict] = {}
        for error in lean_errors:
            first_lean.setdefault(error["module"], error)
        for module, step in lake["modules"].items():
            if step["status"] == "blocked":
                detail = ", ".join(step["missing_imports"] + step["failed_imports"])
                lines.append(f"| {module} | blocked | - | - | {detail} |")
                continue
            example = first_lean.get(module)
            detail = (f"`{example['source']}`: {example['message'][:160]}" if example else "-").replace("|", "\\|")
            lines.append(f"| {module} | {step['status']}{' (RSS cap)' if step.get('rss_cap_killed') else ''} "
                         f"| {step['wall_seconds']} | {step['peak_tree_rss_bytes'] >> 20} | {detail} |")
    else:
        lines += ["", f"Lake build: {fmt_step(lake)}"]
    lines += ["", "## Counts per class", "",
              "| class | suggested owner | errors | chunks |", "|---|---|---:|---:|"]
    for name, totals in report["classes"].items():
        lines.append(f"| {name} | {totals['owner']} | {totals['count']} | {totals['chunks']} |")
    lines += ["", "## First errors per chunk", ""]
    for chunk, firsts in report["first_errors"].items():
        for example in firsts:
            lines.append(f"- {chunk} [{example['class']} -> {example['owner']}] `{example['source']}` "
                         f"({example['item'] or '-'}; {example['compiler']}): {example['message'][:400]}")
    lines += ["", "## Coverage per crate", "",
              "| crate | local items | emitted | unemitted (justified) | missing | duplicates |",
              "|---|---:|---:|---|---:|---:|"]
    for crate, stats in coverage_report["crates"].items():
        unemitted = ", ".join(f"{name} {count}" for name, count in sorted(stats["unemitted"].items()))
        lines.append(f"| {crate} | {stats['local_items']} | {stats['emitted']} | {unemitted or '-'} | "
                     f"{stats['missing']} | {stats['duplicates']} |")
    lines += ["", "## Causes", ""]
    for group in report["groups"]:
        lines.append(f"### [{group['class']} -> {group['owner']}] {group['stage']}: {group['pattern']}")
        lines.append("")
        lines.append(f"- count {group['count']}; compiler source {group['compiler']}; chunks "
                     + ", ".join(f"{chunk} ({count})" for chunk, count in group["chunks"].items()))
        for example in group["examples"]:
            lines.append(f"- `{example['source']}` ({example['item'] or '-'}): {example['message']}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory (must not exist)")
    parser.add_argument("--charon", default=os.environ.get("CHARON", "charon"))
    parser.add_argument("--aeneas", default=os.environ.get("AENEAS", "aeneas"))
    parser.add_argument("--aeneas-lean-dir", type=Path, required=True,
                        help="built Aeneas Lean library (backends/lean) matching --aeneas")
    parser.add_argument("--diagnostic", action="store_true",
                        help="never accept: continue past failures and enumerate every error")
    parser.add_argument("--target-dir", default=str(ROOT / "target" / "agents" / "wholecore"))
    parser.add_argument("--funs-chunk-bytes", type=int, default=2_000_000)
    parser.add_argument("--min-free-percent", type=int, default=20)
    parser.add_argument("--max-memory-wait", type=int, default=3600)
    parser.add_argument("--rss-cap-mb", type=int, default=7000,
                        help="kill a step whose process group exceeds this RSS (0 = no cap)")
    parser.add_argument("--heavy", default=os.environ.get("HEAVY_WRAPPER"),
                        help="slot limiter prefixed to every heavy step (default: $HEAVY_WRAPPER)")
    parser.add_argument("--no-lean", action="store_true", help="diagnostic only: skip lake build")
    parser.add_argument("--clear-start-from", action="store_true",
                        help="diagnostic only: clear the LLBC's recorded --start-from patterns before "
                             "Aeneas (workaround for Aeneas rejecting `{impl ...}` pattern elements)")
    parser.add_argument("--reuse", type=Path,
                        help="diagnostic only: take the chunks that are identical in this plan from an "
                             "earlier run's --out (e.g. the shared crates of the lowering-trace chain)")
    args = parser.parse_args()
    if (args.no_lean or args.reuse or args.clear_start_from) and not args.diagnostic:
        parser.error("--no-lean, --reuse and --clear-start-from are only allowed with --diagnostic")
    out = args.out.resolve()
    if out.exists():
        raise ExtractionFailure(f"output directory already exists: {out}")
    aeneas_lean = args.aeneas_lean_dir.expanduser().resolve()
    if not (aeneas_lean / "Aeneas.lean").is_file():
        raise ExtractionFailure(f"Aeneas Lean library is missing: {aeneas_lean}")
    plan = load_plan(args.plan)
    out.mkdir(parents=True)
    runner = Runner(min_free=args.min_free_percent, max_wait=args.max_memory_wait,
                    rss_cap=args.rss_cap_mb << 20, nice=5, wrapper=[args.heavy] if args.heavy else None)
    charon = tool_identity(args.charon, ["version"])
    aeneas = tool_identity(args.aeneas, ["-version"])
    inputs = plan["inputs"]
    mismatch = []
    if inputs["charon_identity"] != charon["identity"]:
        mismatch.append("the plan was made by another Charon")
    if inputs["sources_sha256"] != source_tree_sha256():
        mismatch.append("the core sources changed since the plan")
    if inputs.get("native_shell_sha256") != sha256_file(NATIVE_SHELL):
        mismatch.append("verification/native_shell.json changed since the plan")
    if plan.get("diagnostic_only"):
        mismatch.append("the plan was dumped past Charon errors (--allow-charon-errors)")
    if mismatch and not args.diagnostic:
        raise ExtractionFailure("; ".join(mismatch))
    summary = {"mode": "diagnostic" if args.diagnostic else "accepting", "accepted": False,
               "chain": plan["chain"], "plan": str(args.plan.resolve()),
               "plan_sha256": sha256_file(args.plan), "plan_mismatch": mismatch,
               "binaries": {"charon": charon["sha256"], "aeneas": aeneas["sha256"]},
               "tools": {"charon": charon, "aeneas": aeneas, "aeneas_lean": str(aeneas_lean)},
               "aeneas_diagnostic_options": list(AENEAS_DIAGNOSTIC) if args.diagnostic else []}

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    results: list[dict] = []
    summary["chunks"] = results
    reuse = reusable_chunks(args.reuse.resolve(), plan, charon, aeneas) if args.reuse else {}
    summary["reused_chunks"] = sorted(reuse)
    try:
        extract(plan, out, args, runner, charon, results, reuse)
    finally:
        save()
    lean = out / "lean"
    modules = generated_modules(lean)
    summary["modules"] = modules
    report = coverage(plan, results, lean_models(aeneas_lean))
    (out / "coverage.json").write_text(json.dumps(report, indent=2))
    write_coverage_summary(report, out / "coverage.md")
    summary["declarations"] = report["emitted_declarations"]
    # Propositions the admitted collections rely on; accept.py requires each to be
    # discharged by an accepted bridge theorem.
    (out / "obligations.json").write_text(json.dumps(
        {"version": 1,
         "obligations": [entry for record in results for entry in record.get("obligations", [])]},
        indent=2))
    summary["obligations"] = sum(len(record.get("obligations", [])) for record in results)
    save()
    if not args.diagnostic and not report["ok"]:
        raise ExtractionFailure(f"coverage defects; see {out / 'coverage.json'}")
    if not modules:
        raise ExtractionFailure("no generated Lean modules")
    write_lake_package(lean, aeneas_lean, modules)
    lake, lean_errors = None, []
    if not args.no_lean:
        lake, lean_errors = lean_build(lean, runner, args.diagnostic, modules,
                                       {chunk["module"] for chunk in plan["chunks"]})
    summary["lake"] = lake
    summary["lean_errors"] = len(lean_errors)
    if args.diagnostic:
        report_blockers = blockers(results, lean_errors, plan)
        (out / "blockers.json").write_text(json.dumps(report_blockers, indent=2))
        write_blockers(report_blockers, results, lake, lean_errors, report, out / "blockers.md")
        summary["accepted"] = False
        save()
        print(f"DIAGNOSTIC run (never an acceptance): {out / 'blockers.md'}")
        return
    summary["accepted"] = True
    save()
    print(f"EXTRACTION ACCEPTED ({plan['chain']} chain): {len(modules)} modules; {out / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (ExtractionFailure, subprocess.CalledProcessError, OSError, KeyError) as error:
        print(f"EXTRACTION REJECTED: {error}", file=sys.stderr)
        raise SystemExit(1) from None
