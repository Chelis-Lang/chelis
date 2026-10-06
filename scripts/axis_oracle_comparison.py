#!/usr/bin/env python3
"""Opt-in, resumable comparison of integration tests and axis-contract oracles.

Run only in a disposable checkout. No proof rejection earns detection credit
without an independently reproduced contract violation. Linux RSS is sampled
over the command's process group; cached rows bind all executable inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("crates/chelis-axis-core/src/verified.rs")
FUNCTIONS = ("is_permutation", "normalize_axis", "reduction_survivors", "checked_inverse")
HARNESS_ROOTS = dict(zip(("admission", "normalization", "survivors", "inverse"), FUNCTIONS))
SWAPS = {"==": ["!="], "!=": ["=="], "<": ["<=", ">"], ">": [">=", "<"],
         "<=": ["<", ">="], ">=": [">", "<="], "+": ["-"], "-": ["+"],
         "+=": ["-="], "true": ["false"], "false": ["true"], "0": ["1"], "1": ["0"]}
TOKEN = re.compile(r"\+=|-=|==|!=|<=|>=|\btrue\b|\bfalse\b|\b[01]\b|[<>+\-]")


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def lexical_mask(source: str) -> str:
    # Length-preserving: offsets and line numbers always address original bytes.
    return re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"',
                  lambda m: "".join("\n" if c == "\n" else " " for c in m[0]), source, flags=re.S)


def exec_ranges(source: str) -> dict[str, tuple[int, int]]:
    masked = lexical_mask(source)
    result = {}
    for name in FUNCTIONS:
        signatures = list(re.finditer(r"^    pub fn " + name + r"\(", masked, re.M))
        if len(signatures) != 1:
            raise ValueError(f"expected one exact production function: {name}")
        body = re.search(r"^    \{\s*$", masked[signatures[0].end():], re.M)
        if body is None:
            raise ValueError(f"cannot locate executable body: {name}")
        start = signatures[0].end() + body.start() + 4
        depth = 0
        for end in range(start, len(masked)):
            depth += (masked[end] == "{") - (masked[end] == "}")
            if depth == 0:
                result[name] = (start, end + 1)
                break
        else:
            raise ValueError(f"unclosed executable body: {name}")
    return result


def executable_mask(source: str) -> str:
    masked = lexical_mask(source)
    output = list("".join("\n" if c == "\n" else " " for c in masked))
    for start, end in exec_ranges(source).values():
        output[start:end] = masked[start:end]
    masked = "".join(output)
    # Loop annotations belong to the frozen oracle, not the mutation surface.
    return re.sub(r"\n[ \t]+(?:invariant|decreases)\b.*?(?=\n[ \t]*\{)",
                  lambda m: "".join("\n" if c == "\n" else " " for c in m[0]), masked, flags=re.S)


def mutants(source: str) -> list[dict]:
    result = []
    masked = executable_mask(source)
    for function, (start, end) in exec_ranges(source).items():
        for match in TOKEN.finditer(masked, start, end):
            # Ignore signs within type arrows and integer suffixes.
            if match[0] == "-" and masked[match.end():match.end()+1] == ">":
                continue
            for replacement in SWAPS.get(match[0], []):
                row = {"function": function, "start": match.start(), "end": match.end(),
                       "original": match[0], "replacement": replacement,
                       "line": source.count("\n", 0, match.start()) + 1,
                       "source_sha256": digest(source)}
                row["id"] = digest(json.dumps(row, sort_keys=True))[:12]
                result.append(row)
    return result


def apply_mutant(source: str, mutant: dict) -> str:
    if digest(source) != mutant["source_sha256"] or source[mutant["start"]:mutant["end"]] != mutant["original"]:
        raise ValueError("mutant baseline or source span changed")
    return source[:mutant["start"]] + mutant["replacement"] + source[mutant["end"]:]


def harness_mapping(source: str) -> dict[str, list[str]]:
    masked = executable_mask(source)
    graph = {name: set(re.findall(r"\b(" + "|".join(FUNCTIONS) + r")\s*\(", masked[a:b]))
             for name, (a, b) in exec_ranges(source).items()}
    result = {name: [] for name in FUNCTIONS}
    for harness, root in HARNESS_ROOTS.items():
        visited, pending = set(), [root]
        while pending:
            node = pending.pop()
            if node not in visited:
                visited.add(node)
                pending.extend(graph[node])
        for node in visited:
            result[node].append(harness)
    return result


def classify(oracle: str, code: int, output: str, timeout: bool) -> str:
    if timeout or (oracle == "kani" and re.search(r"(?:CBMC|verification|solver).*timed out", output, re.I)):
        return "timeout"
    failed_checks = [block for block in re.split(r"(?m)^Check \d+: ", output)[1:]
                     if re.search(r"(?m)^\s*- Status: FAILURE\s*$", block)] if oracle == "kani" else []
    if oracle == "kani" and (any("unsupported_construct" in block.splitlines()[0]
                                  or "not currently supported by Kani" in block for block in failed_checks)
                             or re.search(r"Failed Checks:[^\n]*unsupported", output, re.I)
                             or re.search(r"Description:[^\n]*unsupported[^\n]*\nStatus: FAILURE", output, re.I)):
        return "tool_error"
    if oracle == "kani" and (any("unwinding assertion" in block for block in failed_checks)
                             or re.search(r"unwind(?:ing)? assertion[^\n]*(?:\n[^\n]*){0,2}?FAILURE", output, re.I)):
        return "unwind_failure"
    if code == 0:
        return "pass"
    if oracle == "verus" and re.search(r"(?:precondition|postcondition|invariant|assertion).*?(?:not satisfied|not met|failed)|possible arithmetic (?:underflow|overflow)|decreases.*(?:not|failed)", output, re.I):
        return "proof_failure"
    if oracle == "kani" and "VERIFICATION:- FAILED" in output:
        return "assertion_failure"
    if oracle == "tests" and "test result: FAILED" in output:
        return "test_failure"
    return "tool_error"


def verus_credit(status: str, witness: dict | None) -> str:
    if status == "proof_failure":
        return "caught" if witness else "unwitnessed_proof_failure"
    return status


def selected_for_vermilion(row: dict) -> bool:
    proof = row.get("verus_credit")
    bounded_catch = any(k["status"] == "assertion_failure" for k in row.get("kani", {}).values())
    return (proof == "unwitnessed_proof_failure" or (proof == "caught" and not bounded_catch)
            or (proof == "pass" and bool(row.get("witness"))))


def execution_environment(original: dict[str, str]) -> dict[str, str]:
    env = original.copy()
    env.setdefault("PYO3_PYTHON", sys.executable)
    env["CARGO_BUILD_JOBS"] = "8"
    env.pop("RUSTFLAGS", None)
    return env


def vermilion_status(receipt: dict, structural: dict, source: str, refused: bool) -> str:
    if receipt["timeout"]:
        return "timeout"
    if refused or structural.get("source") != source or structural.get("phase") != "lean":
        return "tool_error"
    if receipt["exit_code"] == 0 and structural.get("exit") == 0:
        return "pass"
    return "proof_failure" if receipt["exit_code"] == 1 else "tool_error"


def group_rss(group: int) -> int:
    total = 0
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == group:
                    total += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                continue
    return total


def stop_process(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    # The parent can exit before a compiler/solver child; terminate the whole
    # group before the campaign restores its source or starts another command.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(command: list[str], cwd: Path, env: dict[str, str], log: Path, timeout: float = 900,
            rss_limit_bytes: int | None = None) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    peak, timed_out, memory_limited = 0, False, False
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                peak = max(peak, group_rss(process.pid))
                if time.monotonic() - started > timeout or (rss_limit_bytes is not None and peak > rss_limit_bytes):
                    memory_limited = rss_limit_bytes is not None and peak > rss_limit_bytes
                    timed_out = not memory_limited
                    stop_process(process)
                    break
                time.sleep(0.05)
        except BaseException:
            stop_process(process)
            raise
        code = process.wait()
    return {"command": command, "exit_code": code, "timeout": timed_out,
            "memory_limit": memory_limited, "rss_limit_bytes": rss_limit_bytes,
            "wall_seconds": time.monotonic() - started, "peak_group_rss_bytes": peak,
            "log": str(log), "log_sha256": digest(log.read_bytes())}


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_receipts(value: object) -> None:
    if isinstance(value, dict):
        if "log_sha256" in value:
            path = Path(value["log"])
            if not path.is_file() or digest(path.read_bytes()) != value["log_sha256"]:
                raise ValueError(f"cached execution log changed or missing: {path}")
        for child in value.values():
            validate_receipts(child)
    elif isinstance(value, list):
        for child in value:
            validate_receipts(child)


def kani_command(binary: str, harness: str, bound: int, unwind: int | None = None) -> list[str]:
    command = [binary, "kani", "-p", "chelis-axis-core", "--harness", harness,
               "--default-unwind", str(unwind if unwind is not None else bound + 3), "-Z", "unstable-options",
               "--harness-timeout", "300s", "-Z", "concrete-playback", "--concrete-playback", "print"]
    return command


def run_kani(args: argparse.Namespace, harness: str, bound: int, label: str, false_control: bool = False, unwind: int | None = None) -> dict:
    env = args.env.copy()
    env["CHELIS_AXIS_KANI_BOUND"] = str(bound)
    env["CARGO_TARGET_DIR"] = str(getattr(args, "kani_target", args.output / "targets/kani"))
    if false_control:
        env["RUSTFLAGS"] = "--cfg axis_false_postcondition"
    unwind = unwind if unwind is not None else bound + 3
    receipt = args.output / f"logs/{label}-{harness}-{bound}-{unwind}-{int(false_control)}.json"
    if receipt.exists():
        cached = json.loads(receipt.read_text())
        validate_receipts(cached)
        return cached
    row = execute(kani_command(args.kani, harness, bound, unwind), args.checkout, env,
                  args.output / f"logs/{label}-{harness}.log", 420)
    row["bound"] = bound
    row["unwind"] = unwind
    row["status"] = classify("kani", row["exit_code"], Path(row["log"]).read_text(), row["timeout"])
    if "CBMC timed out" in Path(row["log"]).read_text():
        row["status"] = "timeout"
    save(receipt, row)
    return row


def calibrate(args: argparse.Namespace) -> None:
    path = args.output / "calibration.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    validate_receipts(result)
    if not set(result) <= set(HARNESS_ROOTS):
        raise ValueError("calibration contains unknown harnesses")
    for harness in HARNESS_ROOTS:
        if harness in result:
            continue
        trials, passing, failing = [], -1, None
        bounds = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
        if harness == "normalization":
            bounds += (2147483647, 9223372036854775807, 18446744073709551615)
        for bound in bounds:
            trial = run_kani(args, harness, bound, f"calibration-{bound}", unwind=bound + 3 if harness != "normalization" else 32)
            trials.append(trial)
            print(f"{harness} bound {bound}: {trial['status']} ({trial['wall_seconds']:.1f}s)", flush=True)
            if trial["status"] == "pass":
                passing = bound
            elif trial["status"] == "timeout":
                failing = bound
                break
            else:
                raise RuntimeError(f"baseline {harness} failed: {trial['log']}")
        if failing is not None:
            while failing - passing > 1:
                bound = (passing + failing) // 2
                trial = run_kani(args, harness, bound, f"calibration-{bound}", unwind=32 if harness == "normalization" else bound + 3)
                trials.append(trial)
                print(f"{harness} bound {bound}: {trial['status']}", flush=True)
                if trial["status"] == "pass":
                    passing = bound
                elif trial["status"] == "timeout":
                    failing = bound
                else:
                    raise RuntimeError(f"baseline {harness} failed: {trial['log']}")
        if passing < 0:
            raise RuntimeError(f"no passing bound: {harness}")
        result[harness] = {"largest_completed_bound": passing, "search_ceiling": bounds[-1],
                           "ceiling_reached": passing == bounds[-1], "trials": trials}
        save(path, result)
    save(args.output / "bound.json", {"bound": min(v["largest_completed_bound"] for v in result.values()),
                                      "unwind_by_harness": {h: min(v["trials"], key=lambda t: t["bound"])["unwind"] if h == "normalization" else min(x["largest_completed_bound"] for x in result.values()) + 3 for h, v in result.items()},
                                      "runtime_abi_rank_ceiling": 2147483647,
                                      "metal_movement_rank_ceiling": 8})


def frozen_settings(args: argparse.Namespace) -> dict:
    settings = json.loads((args.output / "bound.json").read_text())
    identity = args.output / "campaign-settings.json"
    if identity.exists() and json.loads(identity.read_text()) != settings:
        raise ValueError("campaign bound or unwind settings changed")
    save(identity, settings)
    return settings


def campaign(args: argparse.Namespace, source: str, inventory: list[dict]) -> None:
    settings = frozen_settings(args)
    bound = settings["bound"]
    mapping = harness_mapping(source)
    result_path = args.output / "results.json"
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    validate_receipts(rows)
    completed = {row["id"] for row in rows}
    if len(completed) != len(rows) or not completed <= {m["id"] for m in inventory}:
        raise ValueError("cached mutant identities are duplicated or foreign")
    source_path = args.checkout / SOURCE
    if source_path.read_text() != source:
        raise ValueError("checkout source differs from frozen baseline")
    try:
        for number, mutant in enumerate(inventory, 1):
            if mutant["id"] in completed:
                continue
            source_path.write_text(apply_mutant(source, mutant))
            row = dict(mutant)
            env = args.env.copy()
            env["CARGO_TARGET_DIR"] = str(args.output / "targets/tests")
            build = execute(["cargo", "build", "--locked", "-p", "chelis-axis-core"], args.checkout,
                            env, args.output / f"logs/{mutant['id']}-build.log")
            row["build"] = build
            if build["exit_code"]:
                row["classification"] = "unbuildable" if not build["timeout"] else "build_timeout"
            else:
                row["tests"] = []
                for label, command in [
                    ("axis", ["cargo", "test", "--locked", "-p", "chelis-axis-core", "--test", "axis_contract"]),
                    ("types", ["cargo", "test", "--locked", "-p", "chelis-types", "--all-targets", "--no-fail-fast"]),
                    ("ir", ["cargo", "test", "--locked", "-p", "chelis-ir", "--all-targets", "--no-fail-fast"]),
                ]:
                    test = execute(command, args.checkout, env, args.output / f"logs/{mutant['id']}-{label}.log", 180, 8 * 1024**3)
                    test["status"] = "memory_limit" if test["memory_limit"] else classify("tests", test["exit_code"], Path(test["log"]).read_text(), test["timeout"])
                    test["suite"] = label
                    row["tests"].append(test)
                proof = execute([args.verus, "--crate-type=lib", str(SOURCE), "--time", "--output-json"],
                                args.checkout, args.env, args.output / f"logs/{mutant['id']}-verus.log", 300)
                proof["status"] = classify("verus", proof["exit_code"], Path(proof["log"]).read_text(), proof["timeout"])
                row["verus"] = proof
                row["kani"] = {h: run_kani(args, h, bound, mutant["id"], unwind=settings["unwind_by_harness"][h]) for h in mapping[mutant["function"]]}
                row["unaffected_harnesses"] = sorted(set(HARNESS_ROOTS) - set(row["kani"]))
                # A separate concrete Rust probe establishes evidence for proof credit.
                library = args.output / "targets/tests/debug/libchelis_axis_core.rlib"
                binary = args.output / "witness-probe"
                probe_build = execute(["rustc", "--edition=2024", str(ROOT / "crates/chelis-axis-core/proofs/axis_witness.rs"),
                                       "--extern", f"chelis_axis_core={library}", "-L",
                                       f"dependency={library.parent / 'deps'}", "-o", str(binary)],
                                      args.checkout, args.env, args.output / f"logs/{mutant['id']}-witness-build.log")
                row["witness_build"] = probe_build
                if probe_build["exit_code"] == 0:
                    probe = execute([str(binary), mutant["function"]], args.checkout, args.env,
                                    args.output / f"logs/{mutant['id']}-witness.log", 15, 512 * 1024**2)
                    row["witness_probe"] = probe
                    matches = [json.loads(line) for line in Path(probe["log"]).read_text().splitlines()
                               if line.startswith('{"function"')]
                    row["witness"] = matches[0] if matches else None
                else:
                    row["witness"] = None
                row["verus_credit"] = verus_credit(proof["status"], row["witness"])
                row["classification"] = "real_gap" if row["witness"] else "needs_manual_classification"
            rows.append(row)
            save(result_path, rows)
            source_path.write_text(source)
            print(f"mutant {number}/{len(inventory)} {mutant['id']}: {row['classification']} "
                  f"Verus={row.get('verus_credit', 'not_run')}", flush=True)
    finally:
        source_path.write_text(source)


def costs(args: argparse.Namespace, source: str) -> None:
    """Cold means an empty Cargo target; installed tools and OS caches are warm."""
    path = args.output / "costs.json"
    rows = json.loads(path.read_text()) if path.exists() else {}
    validate_receipts(rows)
    settings = frozen_settings(args)
    bound = settings["bound"]
    for phase in ("cold", "warm"):
        for oracle in ("tests", "kani", "verus"):
            key = f"{oracle}-{phase}"
            if key in rows:
                if any(r["status"] != "pass" for r in rows[key]):
                    raise RuntimeError(f"cached baseline oracle failed: {key}")
                continue
            receipts = []
            if oracle == "tests":
                env = args.env | {"CARGO_TARGET_DIR": str(args.output / "targets/tests")}
                for package, extra in (("chelis-axis-core", ["--test", "axis_contract"]), ("chelis-types", ["--all-targets", "--no-fail-fast"]), ("chelis-ir", ["--all-targets", "--no-fail-fast"])):
                    receipt = execute(["cargo", "test", "--locked", "-p", package, *extra], args.checkout,
                                      env, args.output / f"logs/{key}-{package}.log")
                    receipt["status"] = classify("tests", receipt["exit_code"], Path(receipt["log"]).read_text(), receipt["timeout"])
                    receipts.append(receipt)
            elif oracle == "kani":
                args.kani_target = args.output / "targets/cost-kani"
                receipts = [run_kani(args, harness, bound, key, unwind=settings["unwind_by_harness"][harness]) for harness in HARNESS_ROOTS]
                delattr(args, "kani_target")
            else:
                receipt = execute([args.verus, "--crate-type=lib", str(SOURCE), "--time", "--output-json"],
                                  args.checkout, args.env, args.output / f"logs/{key}.log", 300)
                receipt["status"] = classify("verus", receipt["exit_code"], Path(receipt["log"]).read_text(), receipt["timeout"])
                receipts = [receipt]
            rows[key] = receipts
            save(path, rows)
            if any(r["status"] != "pass" for r in receipts):
                raise RuntimeError(f"baseline cost oracle failed: {key}")
            print(f"{key}: {sum(r['wall_seconds'] for r in receipts):.2f}s", flush=True)
    if "negative-controls" not in rows:
        negative = args.output / "false-postcondition.rs"
        original = "valid == valid_permutation(axes@, rank)"
        if source.count(original) != 1:
            raise ValueError("cannot locate false postcondition control")
        negative.write_text(source.replace(original, "valid == false"))
        verus = execute([args.verus, "--crate-type=lib", "--crate-name=axis_false_postcondition", str(negative)], args.checkout, args.env,
                        args.output / "logs/negative-verus.log", 300)
        verus["status"] = classify("verus", verus["exit_code"], Path(verus["log"]).read_text(), verus["timeout"])
        kani = run_kani(args, "false_permutation_postcondition", bound, "negative", True, unwind=settings["unwind_by_harness"]["admission"])
        if verus["status"] != "proof_failure" or kani["status"] != "assertion_failure":
            raise RuntimeError("negative control did not fail for the expected reason")
        rows["negative-controls"] = {"verus": verus, "kani": kani,
                                      "witness": {"rank": 0, "axes": [], "expected_false": False, "actual": True}}
        save(path, rows)
    print("negative controls: expected contract failures", flush=True)
    if "runtime-artifacts" not in rows:
        target = args.output / "targets/runtime"
        env = args.env | {"CARGO_TARGET_DIR": str(target)}
        build = execute(["cargo", "build", "--locked", "--release", "-p", "chelis-axis-core"], args.checkout,
                        env, args.output / "logs/runtime-build.log")
        if build["exit_code"] != 0:
            raise RuntimeError("runtime measurement build failed")
        samples = {}
        for kind in ("plain", "verified"):
            binary = args.output / f"axis-runtime-{kind}"
            command = ["rustc", "--edition=2024", "-O", "-C", "strip=symbols", "--check-cfg", "cfg(axis_verified)",
                       str(ROOT / "crates/chelis-axis-core/proofs/axis_runtime.rs"), "--extern",
                       f"chelis_axis_core={target / 'release/libchelis_axis_core.rlib'}", "-L",
                       f"dependency={target / 'release/deps'}", "-o", str(binary)]
            if kind == "verified":
                command += ["--cfg", "axis_verified"]
            compile_receipt = execute(command, args.checkout, env, args.output / f"logs/runtime-{kind}-compile.log")
            if compile_receipt["exit_code"] != 0:
                raise RuntimeError("runtime benchmark compile failed")
            run = execute([str(binary)], args.checkout, env, args.output / f"logs/runtime-{kind}.csv", 300)
            if run["exit_code"] != 0:
                raise RuntimeError("runtime parity checks failed")
            samples[kind] = {"compile": compile_receipt, "run": run, "stripped_binary_bytes": binary.stat().st_size}
        rows["runtime-artifacts"] = {"build": build, "benchmarks": samples}
        save(path, rows)


def insert_tactic_helper(text: str, helper: str) -> str:
    marker = "namespace verified.is_permutation\n"
    if text.count(marker) != 1:
        raise ValueError("generated twin lacks exact expected namespace")
    if text.count(helper) > 1 or ("-- vrml:user:begin" in text and helper not in text):
        raise ValueError("preserved Lean helper is duplicated or differs from the checked helper")
    return text if helper in text else text.replace(marker, helper + "\n" + marker, 1)


def vermilion(args: argparse.Namespace, source: str, inventory: list[dict]) -> None:
    """Fresh statements, replayed tactics, then Lean checks every proof term."""
    if args.vermilion is None:
        raise ValueError("--vermilion must name the pinned, built checkout")
    frozen_settings(args)
    home = args.vermilion.resolve()
    pin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=home, text=True).strip()
    if pin != "696756d6b9bbaedd61d6cd743ec3165f2b639224":
        raise ValueError("Vermilion checkout differs from the reviewed pin")
    path = args.output / "vermilion.json"
    rows = json.loads(path.read_text()) if path.exists() else {}
    validate_receipts(rows)
    results = json.loads((args.output / "results.json").read_text())
    module_path = ROOT / "crates/chelis-axis-core/proofs/axis_vermilion_proofs.py"
    spec = importlib.util.spec_from_file_location("axis_lean_tactics", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load baseline Lean tactics")
    tactics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tactics)
    cases = [("baseline", source, None)]
    indexed = {m["id"]: m for m in inventory}
    cases += [(r["id"], apply_mutant(source, indexed[r["id"]]), r) for r in results if selected_for_vermilion(r)]
    for label, text, mutant_row in cases:
        if label in rows:
            continue
        case = home / f"examples/chelis-campaign-{label}"
        case.mkdir(exist_ok=True)
        (case / "verified.rs").write_text(text)
        env = args.env.copy()
        env.pop("CARGO_TARGET_DIR", None)
        env["PATH"] = f"{Path.home() / '.elan/bin'}:{env['PATH']}"
        command = ["./scripts/run_example.sh", str(case), "verified.rs", "--per-file", "--manual-proofs", "--lib", "Vermilion"]
        run_path = home / ".vermilion/verified-run.json"
        run_path.unlink(missing_ok=True)
        first = execute(command, home, env, args.output / f"logs/vermilion-{label}-fresh.log", 900)
        proof = case / "proofs/verified.lean"
        installed = []
        if proof.exists() and first["exit_code"] in (0, 1) and not first["timeout"]:
            twin = proof.read_text()
            if label == "baseline":
                tactics.main(proof)
                installed = [name for name, _ in tactics.PROOFS]
            else:
                twin = insert_tactic_helper(twin, tactics.HELPER)
                for (name, _), body in tactics.PROOFS.items():
                    # Mutants deliberately change statements. No trusted cache or
                    # old statement hashes: replay tactics against NEW statements.
                    block = re.search(r"-- vrml:begin " + re.escape(name) + r" [^\n]+\n.*?-- vrml:end " + re.escape(name), twin, re.S)
                    if block is None:
                        continue
                    value = block[0]
                    start = value.index(":= by\n") + len(":= by\n")
                    end = value.index("-- vrml:end ")
                    twin = twin[:block.start()] + value[:start] + body + value[end:] + twin[block.end():]
                    installed.append(name)
                proof.write_text(twin)
            run_path.unlink(missing_ok=True)
            final = execute(command, home, env, args.output / f"logs/vermilion-{label}-checked.log", 900)
        else:
            final = first
        structural = json.loads(run_path.read_text()) if run_path.exists() else {}
        generated = case / "generated/verified.json"
        refused = json.loads(generated.read_text()).get("refused", []) if generated.exists() else []
        status = vermilion_status(final, structural, str(case.relative_to(home) / "verified.rs"), bool(refused))
        rows[label] = {"source_sha256": digest(text), "fresh": first, "checked": final,
                       "structural_verdict": structural, "status": status, "replayed_tactics": installed,
                       "refused_functions": refused,
                       "credit": verus_credit(status, mutant_row.get("witness") if mutant_row else None),
                       "obligations": proof.read_text().count("-- vrml:begin ") if proof.exists() else 0}
        save(path, rows)
        print(f"Vermilion {label}: {status}", flush=True)
        if label == "baseline" and (status != "pass" or rows[label]["obligations"] != 75):
            raise RuntimeError("Vermilion baseline did not kernel-check")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("list", "calibrate", "run", "cost", "vermilion"))
    parser.add_argument("--checkout", type=Path, default=ROOT)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kani", default="cargo-kani")
    parser.add_argument("--verus", default="verus")
    parser.add_argument("--vermilion", type=Path)
    args = parser.parse_args()
    args.checkout, args.output = args.checkout.resolve(), args.output.resolve()
    primary = subprocess.check_output(["git", "worktree", "list", "--porcelain"], cwd=args.checkout, text=True).splitlines()[0].removeprefix("worktree ")
    if args.checkout == Path(primary).resolve():
        raise ValueError("experiment must not run in the primary developer checkout")
    args.output.mkdir(parents=True, exist_ok=True)
    args.env = execution_environment(os.environ.copy())
    source = subprocess.check_output(["git", "show", f"HEAD:{SOURCE}"], cwd=args.checkout).decode()
    current = (args.checkout / SOURCE).read_text()
    if current != source:
        # Recover the exact frozen mutation after a killed job, never a user's edit.
        if not any(current == apply_mutant(source, m) for m in mutants(source)):
            raise ValueError("checkout contains an unrecognized kernel edit")
        (args.checkout / SOURCE).write_text(source)
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.checkout, text=True).strip():
        raise ValueError("frozen checkout has tracked edits")
    def interrupted(signum: int, _frame: object) -> None:
        raise InterruptedError(f"experiment interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    identity = {"source_sha256": digest(source), "harness_sha256": digest((args.checkout / "crates/chelis-axis-core/src/kani_harnesses.rs").read_bytes()),
                "runner_sha256": digest(Path(__file__).read_bytes()),
                "cargo_lock_sha256": digest((args.checkout / "Cargo.lock").read_bytes()),
                "kani_version": subprocess.check_output([args.kani, "--version"], text=True).strip(),
                "verus_version": subprocess.check_output([args.verus, "--version"], text=True).strip(),
                "budget_seconds": 300, "harness_mapping": harness_mapping(source)}
    files = subprocess.check_output(["git", "ls-files", "-z", "crates", "Cargo.toml", "rust-toolchain.toml", ".cargo", "clippy.toml"], cwd=args.checkout).decode().split("\0")
    identity["frozen_inputs"] = {f: digest((args.checkout / f).read_bytes()) for f in files if f and (args.checkout / f).is_file()}
    identity["baseline_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip()
    identity["witness_sha256"] = digest((ROOT / "crates/chelis-axis-core/proofs/axis_witness.rs").read_bytes())
    identity["runtime_benchmark_sha256"] = digest((ROOT / "crates/chelis-axis-core/proofs/axis_runtime.rs").read_bytes())
    identity["machine"] = {"system": platform.platform(), "cpu": Path("/proc/cpuinfo").read_text().split("model name", 1)[-1].split("\n", 1)[0], "logical_cpus": os.cpu_count()}
    identity["rust_version"] = subprocess.check_output(["rustc", "--version"], cwd=args.checkout, text=True).strip()
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("campaign inputs changed; use a new output directory")
    save(manifest, identity)
    inventory = mutants(source)
    save(args.output / "mutants.json", inventory)
    if args.stage == "list":
        print(f"{len(inventory)} executable-only mutants", flush=True)
    elif args.stage == "calibrate":
        calibrate(args)
    elif args.stage == "run":
        campaign(args, source, inventory)
    elif args.stage == "cost":
        costs(args, source)
    else:
        vermilion(args, source, inventory)


if __name__ == "__main__":
    main()
