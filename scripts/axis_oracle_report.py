#!/usr/bin/env python3
"""Export audited per-mutant, per-bound and cold/warm tables from real receipts."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import re

from axis_oracle_comparison import classify, digest, selected_for_vermilion, validate_receipts, vermilion_status, verus_credit


def load_receipts(path: Path, log_store: Path | None = None):
    """Resolve archived logs by their digest in memory; preserve original provenance."""
    value = json.loads(path.read_text())

    def relocate(item):
        if isinstance(item, dict):
            if log_store is not None and "log_sha256" in item:
                sha = item["log_sha256"]
                if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
                    raise ValueError("archived log requires a SHA-256 digest")
                item["log"] = str(log_store / f"{sha}.log")
            for child in item.values():
                relocate(child)
        elif isinstance(item, list):
            for child in item:
                relocate(child)

    relocate(value)
    return value


def validate_campaign(results: list[dict], manifest: dict, settings: dict) -> None:
    for row in results:
        build = row["build"]
        if row["classification"] in ("unbuildable", "build_timeout"):
            if build["exit_code"] == 0:
                raise ValueError("excluded mutant actually compiled")
            continue
        if build["exit_code"] != 0 or "verus" not in row:
            raise ValueError("viable mutant lacks its build or Verus execution")
        suites = [receipt["suite"] for receipt in row.get("tests", [])]
        if len(suites) != 3 or set(suites) != {"axis", "types", "ir"}:
            raise ValueError("viable mutant must run all three test suites")
        expected = set(manifest["harness_mapping"][row["function"]])
        if set(row.get("kani", {})) != expected:
            raise ValueError("viable mutant lacks a reachable Kani harness or runs an unrelated harness")
        if set(row.get("unaffected_harnesses", [])) != set(settings["unwind_by_harness"]) - expected:
            raise ValueError("unaffected harness record differs from the frozen call graph")
        for harness, receipt in row["kani"].items():
            if receipt["bound"] != settings["bound"] or receipt["unwind"] != settings["unwind_by_harness"][harness]:
                raise ValueError("mutant Kani rank bound or unwind differs from the frozen campaign")


def adjudicate_verus(mutant: dict) -> dict:
    """Retain raw labels and correct the frozen parser's bounds-precondition omission."""
    row = copy.deepcopy(mutant)
    if "verus" not in row:
        return row
    proof = row["verus"]
    recorded = proof["status"]
    if row["verus_credit"] != verus_credit(recorded, row.get("witness")):
        raise ValueError("persisted Verus credit disagrees with proof status and witness")
    text = Path(proof["log"]).read_text()
    status = classify("verus", proof["exit_code"], text, proof["timeout"])
    row["verus_status_recorded"] = recorded
    row["verus_adjudication"] = ""
    if status != recorded:
        if not (recorded == "tool_error" and status == "proof_failure"
                and re.search(r"error: precondition not (?:met|satisfied)", text)):
            raise ValueError("unexplained Verus status differs from execution evidence")
        row["verus_adjudication"] = "Frozen parser omitted bounds-precondition rejections; independent witness still required."
        proof["status"] = status
        row["verus_credit"] = verus_credit(status, row.get("witness"))
    return row


def audit_vermilion_cases(rows: dict, cases: Path | None) -> None:
    """Older frozen runners need their actual lowering output audited explicitly."""
    for label, row in rows.items():
        if cases is None:
            if "refused_functions" not in row:
                raise ValueError("Vermilion lowering evidence missing; provide --vermilion-cases")
            continue
        if re.fullmatch(r"baseline|[0-9a-f]{12}", label) is None:
            raise ValueError("invalid Vermilion case identity")
        case = cases / f"chelis-campaign-{label}"
        if digest((case / "verified.rs").read_bytes()) != row["source_sha256"]:
            raise ValueError("Vermilion case source differs from measured mutant")
        generated = json.loads((case / "generated/verified.json").read_text())
        expected = {f"verified.{name}" for name in
                    ("is_permutation", "normalize_axis", "reduction_survivors", "checked_inverse")}
        if (generated["rust_file"] != f"examples/chelis-campaign-{label}/verified.rs"
                or generated["mode"] != "per-file"
                or {fn["function"] for fn in generated["functions"]} != expected):
            raise ValueError("Vermilion lowering does not cover the complete kernel")
        # The pinned manifest schema omits this field when no function is refused.
        refused = generated.get("refused", [])
        if not isinstance(refused, list):
            raise ValueError("invalid Vermilion lowering refusal evidence")
        if "refused_functions" in row and row["refused_functions"] != refused:
            raise ValueError("Vermilion recorded refusals differ from lowering evidence")
        row["refused_functions"] = refused


def adjudicate_kani(mutant: dict) -> dict:
    row = copy.deepcopy(mutant)
    row["kani_status_recorded"] = {}
    row["kani_adjudication"] = ""
    corrected = []
    for harness, receipt in row.get("kani", {}).items():
        recorded = receipt["status"]
        text = Path(receipt["log"]).read_text()
        status = classify("kani", receipt["exit_code"], text, receipt["timeout"])
        row["kani_status_recorded"][harness] = recorded
        if status != recorded:
            if not (recorded == "assertion_failure" and status in ("unwind_failure", "tool_error")):
                raise ValueError("unexplained Kani status differs from execution evidence")
            receipt["status"] = status
            corrected.append(harness)
    if corrected:
        row["kani_adjudication"] = "Frozen parser omitted status-before-description failure blocks: " + ";".join(corrected)
    return row


def validate_vermilion(rows: dict, results: list[dict]) -> None:
    indexed = {row["id"]: row for row in results}
    expected = {"baseline"} | {row["id"] for row in results if selected_for_vermilion(row)}
    if set(rows) != expected:
        raise ValueError("Vermilion baseline or selected mutant checks are incomplete")
    for label, row in rows.items():
        status = vermilion_status(row["checked"], row["structural_verdict"],
                                 f"examples/chelis-campaign-{label}/verified.rs",
                                 bool(row.get("refused_functions")))
        witness = indexed[label].get("witness") if label != "baseline" else None
        if row["status"] != status or row["credit"] != verus_credit(status, witness):
            raise ValueError("Vermilion verdict or credit does not match current case evidence")
    if rows["baseline"]["status"] != "pass" or rows["baseline"]["obligations"] != 75:
        raise ValueError("Vermilion baseline must check the complete kernel")


def verdict(receipts: list[dict], failure: str) -> str:
    statuses = {r["status"] for r in receipts}
    if failure in statuses:
        return "caught"
    return "pass" if statuses == {"pass"} else "inconclusive"


def summarize(mutant: dict, manual: dict) -> dict:
    row = {key: mutant.get(key) for key in ("id", "function", "line", "original", "replacement")}
    witness = mutant.get("witness")
    credit = verus_credit(mutant.get("verus", {}).get("status", "not_run"), witness)
    if mutant.get("verus_credit", "not_run") != credit:
        raise ValueError("persisted Verus credit disagrees with proof status and witness")
    classification = mutant["classification"]
    review = manual.get(mutant["id"])
    if review is not None:
        if not review.get("reason") or review["classification"] not in ("equivalent", "real_gap", "inconclusive"):
            raise ValueError("manual classification requires a supported class and reason")
        if witness and review["classification"] != "real_gap":
            raise ValueError("a reproduced violation cannot be classified equivalent or inconclusive")
        classification = review["classification"]
    if classification == "needs_manual_classification":
        raise ValueError(f"survivor needs manual classification: {mutant['id']}")
    row["classification"] = classification
    row["reason"] = review["reason"] if review else ("concrete contract violation" if witness else classification)
    row["witness"] = json.dumps(witness, sort_keys=True) if witness else ""
    row["within_abi_rank"] = witness.get("within_abi_rank") if witness else None
    row["within_axis_i32_domain"] = witness.get("within_axis_i32_domain") if witness else None
    row["tests"] = verdict(mutant.get("tests", []), "test_failure")
    row["test_suites"] = ";".join(f"{r['suite']}:{r['status']}" for r in mutant.get("tests", []))
    row["verus"] = credit
    row["verus_status"] = mutant.get("verus", {}).get("status", "not_run")
    row["verus_status_recorded"] = mutant.get("verus_status_recorded", row["verus_status"])
    row["verus_adjudication"] = mutant.get("verus_adjudication", "")
    kani = list(mutant.get("kani", {}).values())
    row["kani"] = verdict(kani, "assertion_failure")
    row["kani_harnesses"] = ";".join(f"{h}:{r['status']}" for h, r in mutant.get("kani", {}).items())
    row["kani_status_recorded"] = ";".join(f"{h}:{s}" for h, s in mutant.get("kani_status_recorded", {}).items())
    row["kani_adjudication"] = mutant.get("kani_adjudication", "")
    row["unaffected_harnesses"] = ";".join(mutant.get("unaffected_harnesses", []))
    row["verus_only_confirmed"] = bool(witness and row["verus"] == "caught" and row["tests"] == row["kani"] == "pass")
    for oracle, receipts in (("tests", mutant.get("tests", [])), ("kani", kani), ("verus", [mutant["verus"]] if "verus" in mutant else [])):
        row[f"{oracle}_seconds"] = sum(r["wall_seconds"] for r in receipts)
        row[f"{oracle}_peak_rss_bytes"] = max((r.get("peak_group_rss_bytes", 0) for r in receipts), default=0)
    return row


def write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extras", type=Path, help="Measured installation/setup and Vermilion baseline receipts")
    parser.add_argument("--log-store", type=Path, help="Content-addressed archive logs; original receipt paths remain unchanged")
    parser.add_argument("--vermilion-cases", type=Path, help="Audit generated lowering output from the frozen runner's cases")
    args = parser.parse_args()
    results = load_receipts(args.receipts / "results.json", args.log_store)
    inventory = json.loads((args.receipts / "mutants.json").read_text())
    if {m["id"] for m in results} != {m["id"] for m in inventory} or len(results) != len(inventory):
        raise ValueError("campaign is incomplete or contains duplicate identities")
    indexed = {m["id"]: m for m in inventory}
    for result in results:
        if any(result.get(key) != value for key, value in indexed[result["id"]].items()):
            raise ValueError("result location or operator differs from frozen inventory")
    validate_receipts(results)
    settings = json.loads((args.receipts / "bound.json").read_text())
    manifest = json.loads((args.receipts / "manifest.json").read_text())
    validate_campaign(results, manifest, settings)
    results = [adjudicate_kani(adjudicate_verus(result)) for result in results]
    manual = json.loads(args.manual.read_text())
    if not set(manual) <= {m["id"] for m in inventory}:
        raise ValueError("manual classification includes foreign mutants")
    rows = [summarize(m, manual) for m in results]
    vm_path = args.receipts / "vermilion.json"
    vm = load_receipts(vm_path, args.log_store)
    validate_receipts(vm)
    audit_vermilion_cases(vm, args.vermilion_cases)
    validate_vermilion(vm, results)
    for row in rows:
        measured = vm.get(row["id"], {})
        row["vermilion"] = measured.get("credit", "not_selected")
        row["vermilion_status"] = measured.get("status", "not_selected")
        receipts = {r["log"]: r for r in (measured["fresh"], measured["checked"])} if measured else {}
        row["vermilion_seconds"] = sum(r["wall_seconds"] for r in receipts.values()) if measured else None
        row["vermilion_peak_rss_bytes"] = max((r["peak_group_rss_bytes"] for r in receipts.values()), default=None)
        row["vermilion_obligations"] = measured.get("obligations")
    args.output.mkdir(parents=True, exist_ok=True)
    write_table(args.output / "mutants.csv", rows)
    calibration = load_receipts(args.receipts / "calibration.json", args.log_store)
    validate_receipts(calibration)
    bounds = []
    for harness, measured in calibration.items():
        for trial in measured["trials"]:
            text = Path(trial["log"]).read_text()
            solve = re.findall(r"Verification Time: ([\d.]+)s", text)
            bounds.append({"harness": harness, "bound": trial["bound"], "status": trial["status"],
                           "unwind": trial["unwind"], "wall_seconds": trial["wall_seconds"],
                           "verification_seconds": float(solve[-1]) if solve else None,
                           "peak_group_rss_bytes": trial["peak_group_rss_bytes"],
                           "largest_completed_bound": measured["largest_completed_bound"],
                           "search_ceiling_reached": measured["ceiling_reached"]})
    write_table(args.output / "bounds.csv", bounds)
    costs = load_receipts(args.receipts / "costs.json", args.log_store)
    validate_receipts(costs)
    cost_rows = []
    for oracle in ("tests", "kani", "verus"):
        row = {"oracle": oracle, "kani_campaign_rank_bound": settings["bound"],
               "production_abi_rank_ceiling": settings["runtime_abi_rank_ceiling"],
               "kani_bound_covers_abi_ceiling": settings["bound"] >= settings["runtime_abi_rank_ceiling"]}
        for phase in ("cold", "warm"):
            receipts = costs[f"{oracle}-{phase}"]
            row[f"{phase}_seconds"] = sum(r["wall_seconds"] for r in receipts)
            row[f"{phase}_peak_rss_bytes"] = max(r["peak_group_rss_bytes"] for r in receipts)
        cost_rows.append(row)
    if args.extras:
        extras = load_receipts(args.extras, args.log_store)
        for row in cost_rows:
            item = extras["setup"].get(row["oracle"], {})
            for key in ("install_bytes", "setup_seconds", "setup_memory_bytes", "setup_note"):
                row[key] = item.get(key)
        if "vermilion" in extras:
            vm_cost = extras["vermilion"]
            validate_receipts(vm_cost)
            item = extras["setup"]["vermilion"]
            cost_rows.append({"oracle": "vermilion", "kani_campaign_rank_bound": settings["bound"],
                              "production_abi_rank_ceiling": settings["runtime_abi_rank_ceiling"],
                              "kani_bound_covers_abi_ceiling": settings["bound"] >= settings["runtime_abi_rank_ceiling"],
                              **{f"{phase}_{key}": vm_cost[phase][name] for phase in ("cold", "warm")
                                 for key, name in (("seconds", "wall_seconds"), ("peak_rss_bytes", "peak_group_rss_bytes"))},
                              **{key: item.get(key) for key in ("install_bytes", "setup_seconds", "setup_memory_bytes", "setup_note")}})
    write_table(args.output / "costs.csv", cost_rows)
    real = [r for r in rows if r["classification"] == "real_gap"]
    outcome_kinds = ("caught", "pass", "unwitnessed_proof_failure", "inconclusive")
    summary = {"mutants": len(rows), "viable": sum(r["classification"] not in ("unbuildable", "build_timeout") for r in rows),
               "real_gaps": len(real), "equivalent": sum(r["classification"] == "equivalent" for r in rows),
               "witnessed_real_gaps": sum(bool(r["witness"]) for r in real),
               "unwitnessed_real_gaps": [r["id"] for r in real if not r["witness"]],
               "inconclusive_classifications": sum(r["classification"] == "inconclusive" for r in rows),
               "catches": {o: sum(r[o] == "caught" for r in real) for o in ("tests", "kani", "verus")},
               "outcomes": {o: {kind: sum((r[o] if r[o] in outcome_kinds else "inconclusive") == kind for r in real)
                                for kind in outcome_kinds} for o in ("tests", "kani", "verus")},
               "verus_only_confirmed": [r["id"] for r in real if r["verus_only_confirmed"]],
               "unwitnessed_proof_rejections": [r["id"] for r in rows if r["verus"] == "unwitnessed_proof_failure"],
               "verus_reclassified": [r["id"] for r in rows if r["verus_adjudication"]],
               "kani_reclassified": [r["id"] for r in rows if r["kani_adjudication"]],
               "vermilion": {"selected_mutants": len(vm) - 1,
                             "outcomes": {status: sum(r["vermilion"] == status for r in rows)
                                          for status in {r["credit"] for label, r in vm.items() if label != "baseline"}}},
               "marginal_catches": {o: [r["id"] for r in real if r[o] == "caught" and all(r[p] == "pass" for p in ("tests", "kani", "verus") if p != o)] for o in ("tests", "kani", "verus")},
               "bound": settings, "individual_maxima": {h: v["largest_completed_bound"] for h, v in calibration.items()}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
