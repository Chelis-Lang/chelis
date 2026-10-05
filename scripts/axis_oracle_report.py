#!/usr/bin/env python3
"""Export audited per-mutant, per-bound and cold/warm tables from real receipts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from axis_oracle_comparison import validate_receipts


def verdict(receipts: list[dict], failure: str) -> str:
    statuses = {r["status"] for r in receipts}
    if failure in statuses:
        return "caught"
    return "pass" if statuses == {"pass"} else "inconclusive"


def summarize(mutant: dict, manual: dict) -> dict:
    row = {key: mutant.get(key) for key in ("id", "function", "line", "original", "replacement")}
    witness = mutant.get("witness")
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
    row["verus"] = mutant.get("verus_credit", "not_run")
    row["verus_status"] = mutant.get("verus", {}).get("status", "not_run")
    kani = list(mutant.get("kani", {}).values())
    row["kani"] = verdict(kani, "assertion_failure")
    row["kani_harnesses"] = ";".join(f"{h}:{r['status']}" for h, r in mutant.get("kani", {}).items())
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
    args = parser.parse_args()
    results = json.loads((args.receipts / "results.json").read_text())
    inventory = json.loads((args.receipts / "mutants.json").read_text())
    if {m["id"] for m in results} != {m["id"] for m in inventory} or len(results) != len(inventory):
        raise ValueError("campaign is incomplete or contains duplicate identities")
    validate_receipts(results)
    manual = json.loads(args.manual.read_text())
    if not set(manual) <= {m["id"] for m in inventory}:
        raise ValueError("manual classification includes foreign mutants")
    rows = [summarize(m, manual) for m in results]
    vm_path = args.receipts / "vermilion.json"
    vm = json.loads(vm_path.read_text()) if vm_path.exists() else {}
    validate_receipts(vm)
    for row in rows:
        row["vermilion"] = vm.get(row["id"], {}).get("credit", "not_selected")
    args.output.mkdir(parents=True, exist_ok=True)
    write_table(args.output / "mutants.csv", rows)
    calibration = json.loads((args.receipts / "calibration.json").read_text())
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
    costs = json.loads((args.receipts / "costs.json").read_text())
    validate_receipts(costs)
    settings = json.loads((args.receipts / "bound.json").read_text())
    cost_rows = []
    for oracle in ("tests", "kani", "verus"):
        row = {"oracle": oracle, "kani_campaign_rank_bound": settings["bound"],
               "production_abi_rank_ceiling": settings["runtime_abi_rank_ceiling"],
               "bound_covers_abi_ceiling": settings["bound"] >= settings["runtime_abi_rank_ceiling"]}
        for phase in ("cold", "warm"):
            receipts = costs[f"{oracle}-{phase}"]
            row[f"{phase}_seconds"] = sum(r["wall_seconds"] for r in receipts)
            row[f"{phase}_peak_rss_bytes"] = max(r["peak_group_rss_bytes"] for r in receipts)
        cost_rows.append(row)
    if args.extras:
        extras = json.loads(args.extras.read_text())
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
                              "bound_covers_abi_ceiling": settings["bound"] >= settings["runtime_abi_rank_ceiling"],
                              **{f"{phase}_{key}": vm_cost[phase][name] for phase in ("cold", "warm")
                                 for key, name in (("seconds", "wall_seconds"), ("peak_rss_bytes", "peak_group_rss_bytes"))},
                              **{key: item.get(key) for key in ("install_bytes", "setup_seconds", "setup_memory_bytes", "setup_note")}})
    write_table(args.output / "costs.csv", cost_rows)
    real = [r for r in rows if r["classification"] == "real_gap"]
    summary = {"mutants": len(rows), "viable": sum(r["classification"] not in ("unbuildable", "build_timeout") for r in rows),
               "real_gaps": len(real), "equivalent": sum(r["classification"] == "equivalent" for r in rows),
               "inconclusive_classifications": sum(r["classification"] == "inconclusive" for r in rows),
               "catches": {o: sum(r[o] == "caught" for r in real) for o in ("tests", "kani", "verus")},
               "verus_only_confirmed": [r["id"] for r in real if r["verus_only_confirmed"]],
               "unwitnessed_proof_rejections": [r["id"] for r in rows if r["verus"] == "unwitnessed_proof_failure"],
               "marginal_catches": {o: [r["id"] for r in real if r[o] == "caught" and all(r[p] == "pass" for p in ("tests", "kani", "verus") if p != o)] for o in ("tests", "kani", "verus")},
               "bound": settings, "individual_maxima": {h: v["largest_completed_bound"] for h, v in calibration.items()}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
