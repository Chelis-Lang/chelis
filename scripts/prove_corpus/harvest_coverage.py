#!/usr/bin/env python3
"""Prove corpus harvest and discharge coverage measurement.

Harvests @property goals from downstream shell sources and the chelis test
suite, runs each through `chelis prove --json` preserving the real dispatch
path (no normalization), and produces a coverage map with differentiated
fuzz-tier buckets.

Usage:
    python3 scripts/prove_corpus/harvest_coverage.py [--smt-timeout 5000]

Output:
    scripts/prove_corpus/coverage_map.json   -- per-property discharge results
    scripts/prove_corpus/coverage_summary.md -- human-readable coverage report
"""
from __future__ import annotations
import json, os, subprocess, sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent

PROPERTY_SOURCES = [
    {"shell": "shoals", "label": "Shoals composites",
     "path": Path.home() / "Documents/scratch/shoals/properties/composites.ch"},
    {"shell": "shoals", "label": "Shoals research/contract",
     "path": Path.home() / "Documents/scratch/shoals/research/proof-infra/contract/src/contract.ch"},
    {"shell": "shoals", "label": "Shoals research/derivatives/capacity",
     "path": Path.home() / "Documents/scratch/shoals/research/proof-infra/derivatives/capacity.ch"},
    {"shell": "shoals", "label": "Shoals research/derivatives/reachability",
     "path": Path.home() / "Documents/scratch/shoals/research/proof-infra/derivatives/reachability.ch"},
    {"shell": "shoals", "label": "Shoals research/economic/reachability",
     "path": Path.home() / "Documents/scratch/shoals/research/proof-infra/economic/reachability.ch"},
    {"shell": "chelis", "label": "chelis opaque_invariants",
     "path": REPO_ROOT / "examples/opaque_invariants.ch"},
    {"shell": "chelis", "label": "chelis opaque_invariants_simplex",
     "path": REPO_ROOT / "examples/opaque_invariants_simplex.ch"},
]

CORPUS_DIR = REPO_ROOT / "tests/corpus/opaque_invariants/programs"
if CORPUS_DIR.exists():
    for f in sorted(CORPUS_DIR.glob("*.ch")):
        PROPERTY_SOURCES.append({"shell": "chelis", "label": f"chelis corpus/{f.stem}", "path": f})


@dataclass
class PropertyResult:
    name: str
    shell: str
    label: str
    source_file: str
    composite_verdict: str
    status: str
    coverage_bucket: str
    reason: Optional[str] = None
    goal_expr: Optional[str] = None
    qualifiers: list = field(default_factory=list)
    actual_tier: Optional[str] = None


def classify_coverage_bucket(result: dict) -> tuple[str, Optional[str]]:
    verdict = result.get("composite_verdict", "")
    status = result.get("status", "")
    reason = result.get("reason", "")
    failure = result.get("failure_summary", {})
    actual_tier = failure.get("actual_tier", "")

    if "proven" in verdict or status == "proved":
        return "proved", None
    if "disproved" in verdict or status == "disproved":
        return "disproved", reason or None

    reason_lower = (reason or "").lower()
    if "transcendental" in reason_lower or "erf" in reason_lower or "exp(" in reason_lower:
        return "fuzz_transcendental", reason
    if "type is not supported" in reason_lower or "not amenable" in reason_lower:
        return "fuzz_shape", reason
    if "timeout" in reason_lower or "timed out" in reason_lower:
        return "fuzz_engine_failed", reason
    if "unknown" in reason_lower:
        return "fuzz_engine_failed", reason
    if actual_tier == "fuzz":
        return "fuzz_engine_failed", reason or "fell to fuzz"
    return "fuzz_unsupported", reason or status or "unknown"


def harvest_source(source: dict, smt_timeout: int) -> list[PropertyResult]:
    path = source["path"]
    if not path.exists():
        return []
    chelis_bin = str(REPO_ROOT / "target" / "release" / "chelis")
    cmd = [chelis_bin, "prove", "--json", "--smt-timeout", str(smt_timeout), str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT))
    except subprocess.TimeoutExpired:
        return [PropertyResult(name="<timeout>", shell=source["shell"], label=source["label"],
                               source_file=str(path), composite_verdict="timeout",
                               status="timeout", coverage_bucket="fuzz_engine_failed",
                               reason="file-level timeout (120s)")]
    results = []
    for line in proc.stdout.splitlines():
        if not line.strip().startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("kind") != "property":
            continue
        bucket, bucket_reason = classify_coverage_bucket(data)
        results.append(PropertyResult(
            name=data.get("name", "<unknown>"), shell=source["shell"],
            label=source["label"], source_file=str(path),
            composite_verdict=data.get("composite_verdict", "unknown"),
            status=data.get("status", "unknown"), coverage_bucket=bucket,
            reason=bucket_reason, goal_expr=data.get("goal"),
            qualifiers=data.get("qualifiers", []),
            actual_tier=data.get("failure_summary", {}).get("actual_tier")))
    return results


def generate_summary(results: list[PropertyResult]) -> str:
    lines = ["# Prove Corpus Coverage Map\n", f"Total properties harvested: {len(results)}\n"]
    buckets: dict[str, list[PropertyResult]] = {}
    for r in results:
        buckets.setdefault(r.coverage_bucket, []).append(r)
    lines += ["## Coverage Summary\n", "| Bucket | Count | % |", "|--------|-------|---|"]
    for b in ["proved", "disproved", "fuzz_unsupported", "fuzz_engine_failed",
              "fuzz_transcendental", "fuzz_shape"]:
        items = buckets.get(b, [])
        pct = f"{100*len(items)/len(results):.1f}" if results else "0"
        lines.append(f"| {b} | {len(items)} | {pct}% |")
    lines += ["", "## Per-Shell Breakdown\n"]
    shells: dict[str, list[PropertyResult]] = {}
    for r in results:
        shells.setdefault(r.shell, []).append(r)
    for shell, items in sorted(shells.items()):
        proved = sum(1 for i in items if i.coverage_bucket == "proved")
        lines += [f"### {shell} ({proved}/{len(items)} prove)\n",
                  "| Property | Verdict | Bucket | Reason |", "|----------|---------|--------|--------|"]
        for i in items:
            lines.append(f"| {i.name} | {i.composite_verdict} | {i.coverage_bucket} | {(i.reason or '')[:60]} |")
        lines.append("")
    fuzz_total = sum(len(v) for k, v in buckets.items() if k.startswith("fuzz_"))
    lines += ["## Headline: Fuzz-Tier Analysis\n"]
    if results:
        lines.append(f"**{fuzz_total}/{len(results)} properties fall to fuzz** ({100*fuzz_total/len(results):.0f}%)\n")
    lines.append("Expected: interesting behavioral properties fall to fuzz; table-stakes "
                 "polynomial/shape properties prove. This is the reach frontier map.\n")
    for label, bucket_key in [("Transcendental-blocked", "fuzz_transcendental"),
                              ("Shape-blocked", "fuzz_shape"),
                              ("Engine-failed", "fuzz_engine_failed"),
                              ("Unsupported", "fuzz_unsupported")]:
        if buckets.get(bucket_key):
            lines += [f"### {label} ({len(buckets[bucket_key])})\n"]
            for r in buckets[bucket_key]:
                lines.append(f"- `{r.name}` ({r.label}): {r.reason}")
            lines.append("")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smt-timeout", type=int, default=2000)
    args = parser.parse_args()
    print(f"Harvesting from {len(PROPERTY_SOURCES)} sources...")
    all_results: list[PropertyResult] = []
    for source in PROPERTY_SOURCES:
        if not source["path"].exists():
            print(f"  SKIP {source['label']} ({source['path']})")
            continue
        print(f"  {source['label']}...", end=" ", flush=True)
        results = harvest_source(source, args.smt_timeout)
        print(f"{len(results)} properties")
        all_results.extend(results)
    print(f"\nTotal: {len(all_results)} properties")
    with open(OUTPUT_DIR / "coverage_map.json", "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    summary = generate_summary(all_results)
    with open(OUTPUT_DIR / "coverage_summary.md", "w") as f:
        f.write(summary)
    print(f"\nCoverage map: {OUTPUT_DIR / 'coverage_map.json'}")
    print(f"Summary: {OUTPUT_DIR / 'coverage_summary.md'}")
    buckets = {}
    for r in all_results:
        buckets[r.coverage_bucket] = buckets.get(r.coverage_bucket, 0) + 1
    print("\nQuick stats:")
    for b, c in sorted(buckets.items()):
        print(f"  {b}: {c}")

if __name__ == "__main__":
    main()
